"""Outer portfolio preparation and solver-backed plan projection.

This module owns the pure data boundary between triage/task preparation and the
occurrence-aware solver.  It deliberately does not dispatch workers or mutate a
repository.  The legacy portfolio module re-exports the public functions below
so existing callers keep their import path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import (
    FixPlanStatus,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TaskCluster,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
    VulnerabilityGroup,
)
from remediation_engine.contracts.solver_models import (
    DAGBuildResult,
    PortfolioReplanRequest,
    SolverBatch,
    SolverDependencyRequirement,
    SolverEvidenceDomain,
    SolverFindingRequirement,
    SolverPhase,
    SolverRemediationPlan,
    SolverRuntimeFingerprint,
    SolverStatus,
    SolverSubgraph,
    SolverTarget,
    SolverVersionCandidate,
)
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import solve_portfolio
from remediation_engine.solver.graph import (
    _dependency_order_kind,
    build_dependency_dag,
    cluster_packages,
    schedule_batches,
)
from remediation_engine.solver.subgraph import expand_candidate_relations, extract_solver_subgraph
from remediation_engine.tools.npm_graph import (
    NpmDependencyRecord,
    NpmGraphSnapshot,
    NpmLockfilePackage,
    check_npm_range,
    load_npm_graph_snapshot,
    lockfile_key_matches_package,
    make_occurrence_id,
    normalize_dependency_ancestry,
    resolve_lockfile_dependency_package,
)
from remediation_engine.tools.registry_cache import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    PackumentFetcher,
    RegistryPackumentCache,
    validate_packument,
)

_STABLE_VERSION = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?$")
_OVERRIDE_DEPENDENCY_TYPES = frozenset({"overrides", "resolutions", "pnpm_overrides"})
_EVIDENCE_METADATA_FIELDS = (
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
    "peerDependenciesMeta",
    "engines",
    "os",
    "cpu",
)
_TERMINAL_STATUSES = frozenset(
    {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE, TaskStatus.INCONCLUSIVE, TaskStatus.PIVOTED}
)
_SEVERITY_RANK = {
    Severity.CRITICAL.value: 0,
    Severity.HIGH.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.LOW.value: 3,
    Severity.INFO.value: 4,
    Severity.UNKNOWN.value: 5,
}


@dataclass(frozen=True)
class _PreparedPortfolioProblem:
    """One immutable portfolio input bundle reused across portfolio solves."""

    repo_root: Path
    host_repository_fingerprint: str
    workspace_repository_fingerprint: str
    groups: list[VulnerabilityGroup]
    task_queue: dict[str, RemediationTask]
    npm_snapshot: NpmGraphSnapshot
    targets: list[SolverTarget]
    findings: list[SolverFindingRequirement]
    subgraph: SolverSubgraph
    packuments: Mapping[str, Mapping[str, Any]]
    candidate_domains: Mapping[str, list[SolverVersionCandidate]]
    runtime_fingerprint: SolverRuntimeFingerprint | None
    candidate_catalog_complete: bool
    candidate_catalog_digest: str
    settings: AppSettings
    diagnostics: list[str]
    target_packages: tuple[str, ...] | None
    peer_conflict_pairs: tuple[tuple[str, str], ...]
    forced_singleton_task_ids: tuple[str, ...]


def _digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _catalog_semantic_diagnostics(diagnostics: Iterable[str]) -> list[str]:
    """Exclude cache persistence failures from registry-evidence identity."""
    return sorted(
        {
            message
            for message in diagnostics
            if not message.startswith(
                ("packument cache persistence failed for ", "write failed for ")
            )
        }
    )


def _manifest_path(group: VulnerabilityGroup, repo_root: Path) -> str:
    """Resolve one safe repository-relative package manifest path from a group."""
    root = repo_root.resolve()
    paths: set[str] = set()
    raw_paths = [
        localized.manifest_file for localized in group.localized_issues if localized.manifest_file
    ]
    raw_paths.extend(group.file_paths)
    if group.file_path:
        raw_paths.append(group.file_path)
    for raw_path in raw_paths:
        text = str(raw_path).replace("\\", "/").strip()
        if not text or "\x00" in text:
            return ""
        candidate = Path(text)
        if candidate.is_absolute():
            try:
                text = candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                return ""
        if text.startswith(("~/", "//")) or ".." in Path(text).parts:
            return ""
        paths.add(text)
    safe = [path for path in paths if Path(path).name == "package.json"]
    if len(safe) == 1 and len(paths) == 1:
        return safe[0]
    if not paths:
        return "package.json"
    return ""


def _group_manager(group: VulnerabilityGroup) -> str:
    managers = {
        (localized.package_manager or "").strip().lower()
        for localized in group.localized_issues
        if localized.package_manager
    }
    return next(iter(managers)) if len(managers) == 1 else "npm"


def _active_tasks(task_queue: Mapping[str, RemediationTask]) -> list[RemediationTask]:
    """Return one highest-revision task per group, retaining terminal leaves."""
    grouped: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        grouped.setdefault(task.parent_group_id, []).append(task)
    result: list[RemediationTask] = []
    for group_id in sorted(grouped):
        tasks = grouped[group_id]
        nonterminal = [task for task in tasks if task.status not in _TERMINAL_STATUSES]
        if nonterminal:
            result.append(max(nonterminal, key=lambda task: (task.task_revision, task.task_id)))
        else:
            result.append(max(tasks, key=lambda task: (task.task_revision, task.task_id)))
    return sorted(result, key=lambda task: task.task_id)


def prepare_portfolio_inputs(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    target_packages: Iterable[str] | None = None,
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Create missing finding tasks and synthetic direct-dependency tasks copy-on-write.

    Args:
        repo_root: Repository whose manifests and lockfiles define the portfolio scope.
        groups: Post-triage groups to prepare.
        task_queue: Existing immutable task projection.
        target_packages: Optional development package allowlist. ``None`` or an
            empty iterable preserves full-repository synthetic discovery.

    Returns:
        ``(groups, task_queue, diagnostics)`` containing detached Pydantic objects.
    """
    from remediation_engine.orchestration.task_utils import build_initial_remediation_task

    prepared_groups = [group.model_copy(deep=True) for group in groups]
    prepared_queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    represented = {task.parent_group_id for task in prepared_queue.values()}
    next_index = 1
    for group in sorted(prepared_groups, key=lambda item: item.group_id):
        if group.group_id in represented:
            continue
        while f"task-{next_index}" in prepared_queue:
            next_index += 1
        task_id = f"task-{next_index}"
        prepared_queue[task_id] = build_initial_remediation_task(group, task_id)
        represented.add(group.group_id)
        next_index += 1

    # A pure SAST portfolio must stay on the existing code-remediation path.
    # Synthetic npm coordination is meaningful only when triage supplied an SCA
    # seed; otherwise a package.json would accidentally turn a SAST-only run
    # into a dependency portfolio.
    if not any(group.issue_type == IssueType.SCA for group in prepared_groups):
        return prepared_groups, prepared_queue, []

    # The retained helper is copy-on-write and preserves synthetic identity,
    # refresh deferral, and idempotence.  It is invoked only at this outer
    # boundary; the inner Supervisor never calls it.
    from remediation_engine.orchestration.portfolio_orchestrator import (
        materialize_synthetic_dependency_tasks,
    )

    return materialize_synthetic_dependency_tasks(
        repo_root,
        prepared_groups,
        prepared_queue,
        target_packages=target_packages,
    )


def _issue_identity(issue: Any) -> str:
    """Return a namespaced stable identity for one finding record.

    CVE and GHSA identifiers identify advisories, not individual finding
    records.  A single advisory can affect multiple packages, so they are not
    safe primary identities for solver coverage.  Scanner-native IDs take
    precedence; the internal issue UUID is the required fallback.
    """
    scanner_identity = str(getattr(issue, "finding_id", "") or "").strip()
    if scanner_identity:
        return f"scanner:{scanner_identity}"
    internal_identity = str(getattr(issue, "id", "") or "").strip()
    if internal_identity:
        return f"record:{internal_identity}"
    raise ValueError("finding record has no stable identity")


def _record_index(snapshot: NpmGraphSnapshot) -> dict[tuple[str, str], NpmDependencyRecord]:
    return {
        (record.manifest_path, record.package_name): record
        for record in snapshot.dependency_records
    }


def _workspace_id(snapshot: NpmGraphSnapshot, manifest_path: str) -> str | None:
    roots = sorted(snapshot.workspace_membership_map.get(manifest_path, set()))
    return roots[0] if roots else None


def _issue_dependency_ancestry(group: VulnerabilityGroup, issue: Any | None) -> tuple[str, ...]:
    """Return the most specific scanner ancestry available for one issue."""
    issue_id = str(getattr(issue, "id", "") or "")
    if issue_id:
        for localized in group.localized_issues:
            if str(getattr(localized.issue, "id", "") or "") == issue_id:
                ancestry = normalize_dependency_ancestry(localized.dependency_ancestry)
                if ancestry:
                    return ancestry
    return normalize_dependency_ancestry(group.dependency_ancestry)


def _lockfile_packages_for_finding(
    snapshot: NpmGraphSnapshot,
    manifest_path: str,
    package_name: str,
    issue: Any | None,
    group: VulnerabilityGroup,
) -> list[NpmLockfilePackage]:
    """Resolve every matching physical lockfile occurrence for one finding."""
    issue_version = str(getattr(issue, "package_version", "") or "").strip().lstrip("=vV")
    raw_versions = [issue_version] if issue_version else list(group.versions or [])
    expected_versions = {
        str(value).strip().lstrip("=vV") for value in raw_versions if value and str(value).strip()
    }
    by_physical_key: dict[tuple[str, str | None], NpmLockfilePackage] = {}
    for package in sorted(
        snapshot.lockfile_packages,
        key=lambda item: (
            item.manifest_path,
            item.package_key,
            item.version or "",
            item.lockfile_path,
        ),
    ):
        version = str(package.version or "").strip().lstrip("=vV") or None
        if package.manifest_path != manifest_path or package.package_name != package_name:
            continue
        if expected_versions and version not in expected_versions:
            continue
        by_physical_key.setdefault((package.package_key, version), package)
    candidates = list(by_physical_key.values())
    ancestry = _issue_dependency_ancestry(group, issue)
    if ancestry:
        ancestry_matches = [
            package
            for package in candidates
            if normalize_dependency_ancestry(package.ancestry) == ancestry
        ]
        if len(ancestry_matches) == 1:
            return ancestry_matches
    return candidates


def _occurrence_id_for_lockfile_package(
    snapshot: NpmGraphSnapshot,
    package: NpmLockfilePackage,
) -> str:
    """Reuse the graph's direct identity or derive the exact physical key."""
    for occurrence in snapshot.occurrences:
        if (
            occurrence.manifest_path == package.manifest_path
            and occurrence.package_name == package.package_name
            and occurrence.lockfile_package_key == package.package_key
        ):
            return occurrence.occurrence_id
    return make_occurrence_id(package.manifest_path, package.package_name, package.package_key)


def _finding_occurrences(
    snapshot: NpmGraphSnapshot,
    manifest_path: str,
    package_name: str,
    issue: Any,
    group: VulnerabilityGroup,
) -> list[str]:
    """Return all evidence IDs matching one finding, without guessing a copy."""
    packages = _lockfile_packages_for_finding(snapshot, manifest_path, package_name, issue, group)
    if packages:
        return sorted(
            {_occurrence_id_for_lockfile_package(snapshot, package) for package in packages}
        )

    issue_version = str(getattr(issue, "package_version", "") or "").strip().lstrip("=vV")
    raw_versions = [issue_version] if issue_version else list(group.versions or [])
    expected_versions = {
        str(value).strip().lstrip("=vV") for value in raw_versions if value and str(value).strip()
    }
    candidates = [
        occurrence
        for occurrence in snapshot.occurrences
        if occurrence.manifest_path == manifest_path
        and occurrence.package_name == package_name
        and (occurrence.is_direct or occurrence.dependency_type == "workspace")
        and (
            not expected_versions
            or not occurrence.installed_version
            or str(occurrence.installed_version).strip().lstrip("=vV") in expected_versions
        )
    ]
    ancestry = _issue_dependency_ancestry(group, issue)
    if ancestry:
        ancestry_matches = [
            occurrence
            for occurrence in candidates
            if normalize_dependency_ancestry(occurrence.ancestry) == ancestry
        ]
        if len(ancestry_matches) == 1:
            candidates = ancestry_matches
    return sorted({occurrence.occurrence_id for occurrence in candidates})


def _representative_override_package(
    snapshot: NpmGraphSnapshot,
    manifest_path: str,
    package_name: str,
    group: VulnerabilityGroup,
) -> NpmLockfilePackage | None:
    """Choose one stable mutation identity without narrowing finding coverage."""
    by_identity: dict[tuple[str, str | None], NpmLockfilePackage] = {}
    issues = list(group.issues) or [None]
    for issue in issues:
        for package in _lockfile_packages_for_finding(
            snapshot, manifest_path, package_name, issue, group
        ):
            by_identity.setdefault((package.package_key, package.version), package)
    candidates = list(by_identity.values())
    if not candidates:
        return None
    ancestry = next(
        (_issue_dependency_ancestry(group, issue) for issue in issues if issue is not None),
        normalize_dependency_ancestry(group.dependency_ancestry),
    )
    ancestry_matches = [
        package
        for package in candidates
        if ancestry and normalize_dependency_ancestry(package.ancestry) == ancestry
    ]
    if len(ancestry_matches) == 1:
        return ancestry_matches[0]
    return min(candidates, key=lambda package: (package.package_key, package.version or ""))


def _group_fixed_version(group: VulnerabilityGroup, task: RemediationTask) -> str | None:
    if task.selected_version:
        return task.selected_version.strip().lstrip("vV")
    if group.fix_plan and group.fix_plan.fixed_version:
        return group.fix_plan.fixed_version.strip().lstrip("vV")
    for candidate in group.fix_plan_candidates:
        if candidate.plan.fixed_version:
            return candidate.plan.fixed_version.strip().lstrip("vV")
    for issue in group.issues:
        if issue.fixed_version:
            return issue.fixed_version.strip().lstrip("vV")
    return None


def _workaround_plan_ids(group: VulnerabilityGroup, issue: Any) -> list[str]:
    """Return retained workaround-plan identities for one grouped issue."""
    candidate_ids = sorted(
        {
            str(candidate.issue_id)
            for candidate in group.fix_plan_candidates
            if candidate.issue_id == issue.id
            and candidate.plan.status == FixPlanStatus.WORKAROUND_FOUND
            and candidate.plan.workaround_snippets
        }
    )
    if candidate_ids:
        return candidate_ids
    if (
        group.fix_plan
        and group.fix_plan.status == FixPlanStatus.WORKAROUND_FOUND
        and group.fix_plan.workaround_snippets
    ):
        return [str(issue.id)]
    return []


def _installed_version(
    group: VulnerabilityGroup, task: RemediationTask, record: NpmDependencyRecord | None
) -> str:
    values = [task.parent_package_version, *(group.versions or [])]
    if record and record.resolved_version:
        values.insert(0, record.resolved_version)
    for value in values:
        if value and str(value).strip():
            return str(value).strip().lstrip("vV")
    return "unknown"


def _build_targets_and_findings(
    snapshot: NpmGraphSnapshot,
    groups: Sequence[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
) -> tuple[list[SolverTarget], list[SolverFindingRequirement], list[str]]:
    records = _record_index(snapshot)
    groups_by_id = {group.group_id: group for group in groups}
    diagnostics: list[str] = []
    targets: list[SolverTarget] = []
    findings: list[SolverFindingRequirement] = []
    nonterminal_by_group: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        if task.status not in _TERMINAL_STATUSES:
            nonterminal_by_group.setdefault(task.parent_group_id, []).append(task)
    for group_id, active_tasks in sorted(nonterminal_by_group.items()):
        if len(active_tasks) > 1:
            diagnostics.append(
                f"group {group_id!r} has multiple nonterminal tasks; "
                "portfolio dispatch is blocked until one active leaf remains"
            )
    for task in _active_tasks(task_queue):
        group = groups_by_id.get(task.parent_group_id)
        if group is None or group.issue_type != IssueType.SCA:
            continue
        package_name = (group.vulnerable_component or "").strip()
        target_name = (task.target_package_name or package_name).strip()
        manifest_path = _manifest_path(group, Path("."))
        manager = _group_manager(group)
        record = records.get((manifest_path, target_name))
        override_action = task.target_dependency_type in _OVERRIDE_DEPENDENCY_TYPES
        nested_package = (
            _representative_override_package(snapshot, manifest_path, target_name, group)
            if record is None and target_name == package_name and override_action
            else None
        )
        target_mapped = record is not None or override_action
        if not package_name or not target_name or not manifest_path or manager not in {"", "npm"}:
            diagnostics.append(f"task {task.task_id!r} has an unsupported or ambiguous npm target")
            target_mapped = False
        if not target_mapped:
            diagnostics.append(
                f"task {task.task_id!r} does not map to a direct declaration or supported override"
            )
        if record is None and not override_action:
            diagnostics.append(
                f"task {task.task_id!r} has no direct declaration or supported override target"
            )
        if record is None and override_action and nested_package is None:
            diagnostics.append(
                f"task {task.task_id!r} has no physical package occurrence for its override target"
            )
        lock_key = (
            record.lockfile_package_key
            if record and record.lockfile_package_key
            else nested_package.package_key
            if nested_package is not None
            else f"node_modules/{target_name}"
        )
        occurrence_id = (
            record.occurrence_id
            if record
            else _occurrence_id_for_lockfile_package(snapshot, nested_package)
            if nested_package is not None
            else f"missing-target:{task.task_id}"
        )
        finding_ids = [
            _issue_identity(issue)
            for issue in group.issues
            if issue.cve_id or issue.ghsa_id or issue.finding_id
        ]
        has_version_evidence = bool(
            _group_fixed_version(group, task)
            or task.selected_version
            or task.allowed_target_versions
            or any(issue.fixed_version for issue in group.issues)
        )
        no_fix = task.no_fix_stage is not None or (
            group.fix_plan is not None and group.fix_plan.status == FixPlanStatus.NO_FIX
        )
        target_strategy = "no_fix" if no_fix else task.strategy.value
        group_ancestry = normalize_dependency_ancestry(group.dependency_ancestry)
        physical_ancestry = (
            normalize_dependency_ancestry(nested_package.ancestry)
            if nested_package is not None
            else ()
        )
        dependency_ancestry = (
            group_ancestry
            if len(group_ancestry) > len(physical_ancestry)
            else physical_ancestry
            if len(physical_ancestry) > 1
            else group_ancestry or physical_ancestry
        )
        target = SolverTarget(
            occurrence_id=occurrence_id,
            task_id=task.task_id,
            group_id=group.group_id,
            package_name=package_name,
            target_package_name=target_name,
            manifest_path=manifest_path or "package.json",
            lockfile_package_key=lock_key,
            installed_version=(
                nested_package.version
                if nested_package is not None and nested_package.version
                else _installed_version(group, task, record)
            ),
            dependency_type=task.target_dependency_type
            or (
                group.parent_declaration_type
                if group.parent_package_name
                else (record.declaration_type if record else "dependencies")
            ),
            strategy=target_strategy,
            is_synthetic=bool(task.is_synthetic or group.is_synthetic),
            is_finding_backed=bool(finding_ids),
            eligible_for_atomic_update=(
                target_mapped
                and (record is not None or nested_package is not None)
                and task.status not in _TERMINAL_STATUSES
                and task.current_attempt_id is None
                and (
                    target_strategy == "no_fix"
                    or task.strategy == RoutingStrategy.VERSION_BUMP
                    or has_version_evidence
                )
            ),
            workspace_id=_workspace_id(snapshot, manifest_path),
            dependency_ancestry=dependency_ancestry,
            finding_ids=sorted(set(finding_ids)),
            is_terminal=task.status in _TERMINAL_STATUSES,
            has_open_attempt=task.current_attempt_id is not None,
        )
        targets.append(target)
        for issue in sorted(group.issues, key=lambda item: _issue_identity(item)):
            if not (issue.cve_id or issue.ghsa_id or issue.finding_id):
                continue
            finding_id = _issue_identity(issue)
            transitive_parent_target = bool(
                target_name != package_name
                and (task.parent_package_name or group.parent_package_name)
            )
            fixed = (
                task.parent_minimum_version
                if transitive_parent_target
                else issue.fixed_version or _group_fixed_version(group, task)
            )
            workaround_plan_ids = _workaround_plan_ids(group, issue)
            vulnerable_occurrence_ids = _finding_occurrences(
                snapshot, manifest_path, package_name, issue, group
            )
            if not vulnerable_occurrence_ids:
                vulnerable_occurrence_ids = [
                    f"missing-occurrence:{finding_id}:{manifest_path}:{package_name}"
                ]
                diagnostics.append(
                    f"finding {finding_id!r} has no physical/direct occurrence for "
                    f"{manifest_path}:{package_name}"
                )
            for vulnerable_occurrence_id in vulnerable_occurrence_ids:
                findings.append(
                    SolverFindingRequirement(
                        finding_id=finding_id,
                        vulnerable_occurrence_id=vulnerable_occurrence_id,
                        cve_id=issue.cve_id,
                        ghsa_id=issue.ghsa_id,
                        severity=issue.severity.value
                        if isinstance(issue.severity, Severity)
                        else str(issue.severity),
                        vulnerable_package=package_name,
                        target_occurrence_id=occurrence_id,
                        fixed_version=fixed,
                        direct_parent_name=(group.parent_package_name or task.parent_package_name),
                        direct_parent_minimum_version=task.parent_minimum_version,
                        strategy_stage=task.strategy_stage.value,
                        workaround_available=bool(workaround_plan_ids),
                        workaround_plan_ids=workaround_plan_ids,
                        is_transitive=bool(
                            group.parent_package_name or task.parent_package_name or nested_package
                        ),
                    )
                )
    return targets, findings, sorted(set(diagnostics))


def _required_candidate_package_names(
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
) -> list[str]:
    """Return every mutable target and transitive ancestry package name."""
    mutable_targets = {
        target.occurrence_id: target for target in targets if target.eligible_for_atomic_update
    }
    names: set[str] = set()
    for target in mutable_targets.values():
        if target.target_package_name:
            names.add(target.target_package_name)
        names.update(name for name in target.dependency_ancestry if name)
    for finding in findings:
        if finding.target_occurrence_id not in mutable_targets:
            continue
        if finding.direct_parent_name:
            names.add(finding.direct_parent_name)
        if finding.is_transitive:
            names.add(finding.vulnerable_package)
    return sorted(names)


def _fetch_candidate_packuments(
    package_names: Iterable[str],
    settings: AppSettings,
    *,
    registry_fetcher: PackumentFetcher | None = None,
) -> tuple[dict[str, Mapping[str, Any]], bool, str, list[str]]:
    """Fetch fresh raw packuments and best-effort persist configured cache entries."""
    from remediation_engine.tools.registry_tools import _fetch_package_data

    cache = RegistryPackumentCache(settings.solver_cache_dir) if settings.solver_cache_dir else None
    required_names = sorted({str(name).strip() for name in package_names if str(name).strip()})
    packuments: dict[str, Mapping[str, Any]] = {}
    diagnostics: list[str] = []
    complete = True
    for package_name in required_names:
        try:
            packument = _fetch_package_data(package_name, fetcher=registry_fetcher, cache=None)
            packuments[package_name] = packument
            if cache is not None and not cache.put(package_name, packument):
                diagnostics.append(
                    f"packument cache persistence failed for {package_name!r}; "
                    "fresh metadata remains available in memory"
                )
        except Exception as exc:  # noqa: BLE001
            complete = False
            diagnostics.append(f"fresh packument unavailable for {package_name!r}: {exc}")
    if cache is not None:
        diagnostics.extend(cache.diagnostics)
    catalog_digest = _digest(
        {
            "packument_digests": {
                name: _digest(packument) for name, packument in sorted(packuments.items())
            },
            "missing_packages": sorted(set(required_names) - set(packuments)),
            "diagnostics": _catalog_semantic_diagnostics(diagnostics),
        }
    )
    return packuments, complete, catalog_digest, sorted(set(diagnostics))


def _transitive_child_floor(group: VulnerabilityGroup) -> str | None:
    """Return the highest stable child fix floor in a transitive group."""
    versions = [
        str(issue.fixed_version).strip().lstrip("vV")
        for issue in group.issues
        if issue.fixed_version and _semver_key(str(issue.fixed_version))
    ]
    if group.fix_plan and group.fix_plan.fixed_version:
        value = str(group.fix_plan.fixed_version).strip().lstrip("vV")
        if _semver_key(value):
            versions.append(value)
    return max(versions, key=lambda value: _semver_key(value) or (0, 0, 0)) if versions else None


def _resolve_transitive_parent_floors(
    targets: Sequence[SolverTarget],
    findings: list[SolverFindingRequirement],
    task_queue: Mapping[str, RemediationTask],
    groups: Sequence[VulnerabilityGroup],
    settings: AppSettings,
    diagnostics: list[str],
    *,
    packuments: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, set[str]]:
    """Resolve transitive parent floors from one prepared packument snapshot."""
    cache = RegistryPackumentCache(settings.solver_cache_dir) if settings.solver_cache_dir else None
    groups_by_id = {group.group_id: group for group in groups}
    tasks_by_id = dict(task_queue)
    compatible_by_target: dict[str, set[str]] = {}

    def packument_for(package_name: str) -> Mapping[str, Any] | None:
        if packuments is not None:
            return packuments.get(package_name)
        return cache.get(package_name) if cache is not None else None

    for target in sorted(targets, key=lambda item: item.occurrence_id):
        task = tasks_by_id.get(target.task_id)
        group = groups_by_id.get(task.parent_group_id) if task else None
        if (
            task is None
            or group is None
            or target.target_package_name == target.package_name
            or not (task.parent_package_name or group.parent_package_name)
            or task.parent_minimum_version
        ):
            continue
        child_floor = _transitive_child_floor(group)
        parent_name = (task.parent_package_name or group.parent_package_name or "").strip()
        if not parent_name or not child_floor:
            compatible_by_target[target.occurrence_id] = set()
            continue
        try:
            from remediation_engine.tools.registry_tools import select_npm_parent_version

            parent_data = packument_for(parent_name)
            if parent_data is None:
                compatible_by_target[target.occurrence_id] = set()
                diagnostics.append(f"transitive parent packument unavailable for {parent_name!r}")
                continue
            ancestry = tuple(group.dependency_ancestry)
            registry_data: dict[str, Mapping[str, Any]] = {parent_name: parent_data}
            missing_intermediate: str | None = None
            for intermediate in ancestry[1:-1]:
                if intermediate and intermediate not in registry_data:
                    intermediate_data = packument_for(intermediate)
                    if intermediate_data is None:
                        missing_intermediate = intermediate
                        break
                    registry_data[intermediate] = intermediate_data
            if missing_intermediate:
                compatible_by_target[target.occurrence_id] = set()
                diagnostics.append(
                    f"transitive ancestry packument unavailable for {missing_intermediate!r}"
                )
                continue
            result = select_npm_parent_version(
                dict(parent_data),
                parent_package_name=parent_name,
                child_package_name=group.vulnerable_component or target.package_name,
                child_fixed_version=child_floor,
                installed_parent_version=target.installed_version,
                selection="minimum",
                dependency_ancestry=ancestry or None,
                registry_data_by_package={
                    name: dict(value) for name, value in registry_data.items()
                },
            )
            compatible = {
                str(version).strip().lstrip("vV")
                for version in result.get("compatible", [])
                if _semver_key(str(version))
            }
            compatible_by_target[target.occurrence_id] = compatible
            if compatible:
                parent_floor = min(compatible, key=lambda value: _semver_key(value) or (0, 0, 0))
                for index, finding in enumerate(findings):
                    if finding.target_occurrence_id == target.occurrence_id:
                        findings[index] = finding.model_copy(update={"fixed_version": parent_floor})
            else:
                diagnostics.append(
                    f"no published {parent_name!r} release resolves "
                    f"{group.vulnerable_component!r} to {child_floor}"
                )
        except Exception as exc:  # noqa: BLE001
            compatible_by_target[target.occurrence_id] = set()
            diagnostics.append(f"transitive parent domain unavailable for {parent_name}: {exc}")
    if cache is not None:
        diagnostics.extend(cache.diagnostics)
    return compatible_by_target


def _semver_key(version: str) -> tuple[int, int, int] | None:
    match = _STABLE_VERSION.fullmatch(version.strip())
    return tuple(int(item) for item in match.groups()) if match else None


def _candidate(
    version: str,
    *,
    source: str,
    floor: str | None,
    attempted: bool = False,
    requirements: Sequence[SolverDependencyRequirement] = (),
    engines: Mapping[str, str] | None = None,
    os: Sequence[str] = (),
    cpu: Sequence[str] = (),
) -> SolverVersionCandidate | None:
    normalized = str(version or "").strip().lstrip("vV")
    key = _semver_key(normalized)
    if key is None:
        return None
    floor_key = _semver_key(floor or "")
    return SolverVersionCandidate(
        version=normalized,
        semver_key=key,
        source=source,
        meets_security_floor=floor_key is None or key >= floor_key,
        attempted=attempted,
        requirements=list(requirements),
        engines=dict(engines or {}),
        os=list(os),
        cpu=list(cpu),
    )


def _candidate_metadata(
    package_name: str,
    version: str,
    metadata: Mapping[str, Any],
) -> tuple[
    list[SolverDependencyRequirement],
    dict[str, str],
    list[str],
    list[str],
    list[str],
]:
    """Parse published requirements and runtime constraints without omission."""
    diagnostics: list[str] = []
    requirements: list[SolverDependencyRequirement] = []
    peer_meta_raw = metadata.get("peerDependenciesMeta", {})
    if not isinstance(peer_meta_raw, Mapping):
        diagnostics.append(f"malformed peerDependenciesMeta for {package_name}@{version}")
        peer_meta: Mapping[str, Any] = {}
    else:
        peer_meta = peer_meta_raw
    optional_dependencies = metadata.get("optionalDependencies", {})
    optional_names = (
        {str(name).strip() for name in optional_dependencies}
        if isinstance(optional_dependencies, Mapping)
        else set()
    )
    sections = (
        ("dependencies", "dependency", False),
        ("optionalDependencies", "optional_dependency", True),
        ("peerDependencies", "peer", False),
    )
    for section, kind, section_optional in sections:
        raw = metadata.get(section, {})
        if not isinstance(raw, Mapping):
            diagnostics.append(f"malformed {section} for {package_name}@{version}")
            continue
        for raw_name, raw_range in sorted(raw.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_name, str) or not raw_name.strip():
                diagnostics.append(
                    f"malformed package name in {section} for {package_name}@{version}"
                )
                continue
            if section == "dependencies" and raw_name.strip() in optional_names:
                continue
            if not isinstance(raw_range, str) or not raw_range.strip():
                diagnostics.append(
                    f"malformed range for {raw_name!r} in {section} of {package_name}@{version}"
                )
                continue
            optional = section_optional
            if kind == "peer":
                peer = peer_meta.get(raw_name, {})
                if not isinstance(peer, Mapping):
                    diagnostics.append(
                        f"malformed peer metadata for {raw_name!r} in {package_name}@{version}"
                    )
                    continue
                peer_optional = peer.get("optional", False)
                if not isinstance(peer_optional, bool):
                    diagnostics.append(
                        f"malformed optional peer flag for {raw_name!r} in {package_name}@{version}"
                    )
                    continue
                optional = peer_optional
            requirements.append(
                SolverDependencyRequirement(
                    package_name=raw_name.strip(),
                    version_range=raw_range.strip(),
                    kind=kind,
                    is_optional=optional,
                )
            )

    engines: dict[str, str] = {}
    raw_engines = metadata.get("engines", {})
    if not isinstance(raw_engines, Mapping):
        diagnostics.append(f"malformed engines for {package_name}@{version}")
    else:
        for name, requirement in raw_engines.items():
            if (
                not isinstance(name, str)
                or not isinstance(requirement, str)
                or not requirement.strip()
            ):
                diagnostics.append(f"malformed engine constraint for {package_name}@{version}")
                continue
            engines[name.strip()] = requirement.strip()

    def platform_values(field: str) -> list[str]:
        raw = metadata.get(field, [])
        if not isinstance(raw, (list, tuple)) or any(
            not isinstance(value, str) or not value.strip() for value in raw
        ):
            diagnostics.append(f"malformed {field} for {package_name}@{version}")
            return []
        return sorted({value.strip() for value in raw})

    requirements.sort(
        key=lambda item: (
            item.package_name,
            item.kind,
            item.version_range,
            item.is_optional,
        )
    )
    return (
        requirements,
        dict(sorted(engines.items())),
        platform_values("os"),
        platform_values("cpu"),
        diagnostics,
    )


def _solver_evidence_packument(
    package_name: str,
    packument: Mapping[str, Any],
) -> dict[str, Any]:
    """Project registry metadata to fields used by one-hop solver evidence.

    The complete version-key inventory is retained. Per-version metadata keeps
    every dependency, peer, runtime, and platform field consumed by
    :func:`_candidate_metadata`; unrelated fields such as tarball metadata,
    descriptions, and readmes do not affect the solver's evidence domain.

    Args:
        package_name: Expected npm package identity.
        packument: Fresh raw registry metadata.

    Returns:
        A compact, validated packument projection for evidence-domain analysis.

    Raises:
        ValueError: If the packument is malformed or lacks a versions mapping.
    """
    validated = validate_packument(package_name, packument)
    raw_versions = validated.get("versions")
    if not isinstance(raw_versions, Mapping):
        raise ValueError("registry packument must contain an object-valued versions field")
    versions: dict[str, Any] = {}
    for raw_version, metadata in raw_versions.items():
        if not isinstance(raw_version, str):
            raise ValueError("registry version keys must be strings")
        version = raw_version
        if not isinstance(metadata, Mapping):
            # Preserve malformed entries so the consumer fails closed instead
            # of accidentally treating an incomplete projection as evidence.
            versions[version] = metadata
            continue
        versions[version] = {
            field: metadata[field]
            for field in _EVIDENCE_METADATA_FIELDS
            if field in metadata
        }
    return {"name": validated.get("name", package_name), "versions": versions}


def _supported_dependency_range(version_range: str) -> bool:
    """Whether one npm specifier has a range model understood by this solver."""
    value = version_range.strip()
    if (
        not value
        or value.casefold() == "latest"
        or value.startswith(("npm:", "workspace:", "file:", "git:", "git+", "http:", "https:"))
    ):
        return False
    return check_npm_range(value, "1.0.0").matches is not None


def _override_value(value: Any) -> str | None:
    """Return a simple exact override, leaving nested selector maps unmodeled."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping) and set(value) == {"."}:
        replacement = value.get(".")
        if isinstance(replacement, str) and replacement.strip():
            return replacement.strip()
    return None


def _override_selector_may_match(selector: str, package_name: str) -> bool:
    """Detect unsupported scoped override selectors that could affect a package."""
    return bool(
        selector.endswith(f"/{package_name}")
        or selector.startswith(f"{package_name}@")
        or ("*" in selector and package_name in selector)
    )


def _nested_override_may_match(value: Any, package_name: str) -> bool:
    """Find package selectors in unsupported nested override objects."""
    if not isinstance(value, Mapping):
        return False
    for raw_selector, nested in value.items():
        if isinstance(raw_selector, str) and (
            raw_selector.strip() == package_name
            or _override_selector_may_match(raw_selector.strip(), package_name)
        ):
            return True
        if _nested_override_may_match(nested, package_name):
            return True
    return False


def _effective_dependency_override(
    snapshot: NpmGraphSnapshot,
    target: SolverTarget,
    package_name: str,
) -> tuple[str | None, bool]:
    """Resolve exact supported manifest overrides or report an unmodeled match."""
    manifests = snapshot.manifests_by_path
    workspace_roots = set(snapshot.workspace_membership_map.get(target.manifest_path, set()))
    if target.workspace_id:
        workspace_roots.add(target.workspace_id)
    if len(workspace_roots) > 1:
        return None, False
    scope_paths = [next(iter(workspace_roots)) if workspace_roots else target.manifest_path]
    section_priority = {"overrides": 0, "pnpm_overrides": 1, "resolutions": 2}
    matches: dict[int, set[str]] = {}
    for path in scope_paths:
        manifest = manifests.get(path)
        if manifest is None:
            continue
        data = manifest.data
        pnpm = data.get("pnpm")
        pnpm_overrides = pnpm.get("overrides") if isinstance(pnpm, Mapping) else None
        if "pnpm" in data and not isinstance(pnpm, Mapping):
            return None, False
        sections = (
            ("overrides", data.get("overrides")),
            ("pnpm_overrides", pnpm_overrides),
            ("resolutions", data.get("resolutions")),
        )
        for section, raw_values in sections:
            if raw_values is None:
                continue
            if not isinstance(raw_values, Mapping):
                return None, False
            for raw_selector, raw_value in raw_values.items():
                if not isinstance(raw_selector, str) or not raw_selector.strip():
                    return None, False
                selector = raw_selector.strip()
                if selector != package_name:
                    if _override_selector_may_match(selector, package_name):
                        return None, False
                    if _nested_override_may_match(raw_value, package_name):
                        return None, False
                    continue
                replacement = _override_value(raw_value)
                if replacement is None:
                    return None, False
                precedence = scope_paths.index(path) * 10 + section_priority[section]
                matches.setdefault(precedence, set()).add(replacement)
    if not matches:
        return None, True
    highest_priority = min(matches)
    replacements = matches[highest_priority]
    if len(replacements) != 1:
        return None, False
    return next(iter(replacements)), True


def _effective_candidate_requirements(
    snapshot: NpmGraphSnapshot,
    target: SolverTarget,
    requirements: Sequence[SolverDependencyRequirement],
) -> list[SolverDependencyRequirement]:
    """Apply supported install-scope overrides before solver range modeling."""
    result: list[SolverDependencyRequirement] = []
    for requirement in requirements:
        effective_range, supported_override = _effective_dependency_override(
            snapshot, target, requirement.package_name
        )
        version_range = effective_range or requirement.version_range
        is_range_supported = bool(
            requirement.is_range_supported
            and supported_override
            and _supported_dependency_range(version_range)
        )
        result.append(
            requirement.model_copy(
                update={
                    "version_range": version_range,
                    "is_range_supported": is_range_supported,
                }
            )
        )
    return result


def _candidate_domains(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
    task_queue: Mapping[str, RemediationTask],
    groups: Sequence[VulnerabilityGroup],
    settings: AppSettings,
    diagnostics: list[str],
    *,
    transitive_compatible_versions: Mapping[str, set[str]] | None = None,
    packuments: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, list[SolverVersionCandidate]], bool, str]:
    """Prepare complete, candidate-specific registry domains without pruning."""
    groups_by_task = {
        task.task_id: next(
            (group for group in groups if group.group_id == task.parent_group_id), None
        )
        for task in task_queue.values()
    }
    floors_by_target: dict[str, str | None] = {}
    for finding in findings:
        current = floors_by_target.get(finding.target_occurrence_id)
        if finding.fixed_version and (
            current is None
            or (_semver_key(finding.fixed_version) or (0, 0, 0))
            > (_semver_key(current) or (0, 0, 0))
        ):
            floors_by_target[finding.target_occurrence_id] = finding.fixed_version
    cache = RegistryPackumentCache(settings.solver_cache_dir) if settings.solver_cache_dir else None
    compatible_versions = transitive_compatible_versions or {}
    domains: dict[str, list[SolverVersionCandidate]] = {}
    observed_packuments: dict[str, Mapping[str, Any]] = dict(packuments or {})
    complete = packuments is not None
    completeness_diagnostics: list[str] = []
    limit = max(1, settings.solver_max_candidates_per_target)

    for target in sorted(targets, key=lambda item: item.occurrence_id):
        task = task_queue.get(target.task_id)
        group = groups_by_task.get(target.task_id)
        parent_target = bool(
            group
            and target.target_package_name != target.package_name
            and (task and task.parent_package_name or group.parent_package_name)
        )
        floor = floors_by_target.get(target.occurrence_id)
        if floor is None and parent_target and task is not None:
            floor = task.parent_minimum_version
        parent_proof = (
            compatible_versions.get(target.occurrence_id, set())
            if parent_target and task is not None and not task.parent_minimum_version
            else None
        )
        if (
            parent_target
            and task is not None
            and not task.parent_minimum_version
            and not parent_proof
        ):
            diagnostics.append(
                f"transitive parent target {target.occurrence_id!r} lacks a "
                "published compatibility proof"
            )
        values: list[SolverVersionCandidate] = []

        def add_candidate(
            candidate: SolverVersionCandidate | None,
            *,
            proof: set[str] | None = parent_proof,
            target_values: list[SolverVersionCandidate] = values,
        ) -> SolverVersionCandidate | None:
            if candidate is None:
                return None
            if proof is not None and candidate.version not in proof:
                candidate = candidate.model_copy(update={"meets_security_floor": False})
            target_values.append(candidate)
            return candidate

        for version, source in (
            (target.installed_version, "current"),
            (task.selected_version if task else None, "selected"),
            (floor, "security_floor"),
            (
                _group_fixed_version(group, task) if group and task and not parent_target else None,
                "group_plan",
            ),
            (
                task.parent_minimum_version if task and parent_target else None,
                "parent_minimum",
            ),
            *(
                (version, "approved_alternative")
                for version in (task.allowed_target_versions if task else [])
            ),
        ):
            if version:
                add_candidate(_candidate(str(version), source=source, floor=floor))

        eligible_registry_count = 0
        packument: Mapping[str, Any] | None = None
        if target.eligible_for_atomic_update and target.target_package_name:
            package_name = target.target_package_name
            raw_packument = (
                packuments.get(package_name)
                if packuments is not None
                else cache.get(package_name)
                if cache is not None
                else None
            )
            if raw_packument is None:
                complete = False
                message = f"required packument unavailable for {package_name!r}"
                completeness_diagnostics.append(message)
            else:
                from remediation_engine.tools.registry_cache import validate_packument

                try:
                    packument = validate_packument(package_name, raw_packument)
                    observed_packuments[package_name] = packument
                except (TypeError, ValueError) as exc:
                    complete = False
                    completeness_diagnostics.append(
                        f"malformed packument for {package_name!r}: {exc}"
                    )
            if packuments is None and packument is not None:
                complete = False
                completeness_diagnostics.append(
                    f"cached packument for {package_name!r} is not a fresh planning snapshot"
                )
            if packument is not None:
                raw_versions = packument.get("versions")
                if not isinstance(raw_versions, Mapping):
                    complete = False
                    completeness_diagnostics.append(
                        f"packument versions are malformed for {package_name!r}"
                    )
                else:
                    if packuments is not None:
                        published_versions = {
                            str(version)
                            for version in raw_versions
                            if _semver_key(str(version)) is not None
                        }
                        values[:] = [
                            candidate
                            for candidate in values
                            if candidate.source == "current"
                            or candidate.version in published_versions
                        ]
                    for raw_version, metadata in sorted(
                        raw_versions.items(), key=lambda item: str(item[0])
                    ):
                        normalized_version = str(raw_version)
                        if _semver_key(normalized_version) is None:
                            continue
                        if not isinstance(metadata, Mapping):
                            complete = False
                            completeness_diagnostics.append(
                                f"malformed metadata for {package_name}@{normalized_version}"
                            )
                            continue
                        requirements, engines, os_values, cpu_values, metadata_diagnostics = (
                            _candidate_metadata(package_name, normalized_version, metadata)
                        )
                        requirements = _effective_candidate_requirements(
                            snapshot, target, requirements
                        )
                        if metadata_diagnostics:
                            complete = False
                            completeness_diagnostics.extend(metadata_diagnostics)
                        candidate = _candidate(
                            normalized_version,
                            source="registry",
                            floor=floor,
                            requirements=requirements,
                            engines=engines,
                            os=os_values,
                            cpu=cpu_values,
                        )
                        retained = add_candidate(candidate)
                        if retained is not None and retained.meets_security_floor:
                            eligible_registry_count += 1
                    if eligible_registry_count > limit:
                        complete = False
                        completeness_diagnostics.append(
                            f"eligible release catalog for {package_name!r} has "
                            f"{eligible_registry_count} versions, exceeding resource guard {limit}"
                        )

        dedup: dict[str, SolverVersionCandidate] = {}
        for value in sorted(values, key=lambda item: (item.semver_key, item.version, item.source)):
            existing = dedup.get(value.version)
            if existing is None or (existing.source != "registry" and value.source == "registry"):
                dedup[value.version] = value
        ordered = list(dedup.values())
        domains[target.occurrence_id] = ordered
        if not ordered and target.eligible_for_atomic_update:
            diagnostics.append(f"empty candidate domain for {target.occurrence_id}")
    if cache is not None:
        diagnostics.extend(cache.diagnostics)
        completeness_diagnostics.extend(cache.diagnostics)
    diagnostics.extend(completeness_diagnostics)
    raw_packument_digests = {
        name: _digest(value) for name, value in sorted(observed_packuments.items())
    }
    catalog_digest = _digest(
        {
            "raw_packument_digests": raw_packument_digests,
            "candidate_domains": {
                occurrence_id: [item.model_dump(mode="json") for item in values]
                for occurrence_id, values in sorted(domains.items())
            },
            "complete": complete,
            "diagnostics": sorted(set(completeness_diagnostics)),
        }
    )
    return domains, complete, catalog_digest


def _evidence_requirement_keys(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    candidate_domains: Mapping[str, Sequence[SolverVersionCandidate]],
) -> dict[tuple[str, str, str], SolverTarget]:
    """Find required runtime dependencies without an eligible physical target."""
    eligible_by_id = {
        target.occurrence_id: target for target in targets if target.eligible_for_atomic_update
    }
    result: dict[tuple[str, str, str], SolverTarget] = {}
    for target in sorted(eligible_by_id.values(), key=lambda item: item.occurrence_id):
        for candidate in candidate_domains.get(target.occurrence_id, ()):
            for requirement in candidate.requirements:
                if (
                    requirement.kind != "dependency"
                    or requirement.is_optional
                    or not requirement.is_range_supported
                ):
                    continue
                package = resolve_lockfile_dependency_package(
                    snapshot, target.occurrence_id, requirement.package_name
                )
                if package is not None:
                    physical_id = make_occurrence_id(
                        package.manifest_path, package.package_name, package.package_key
                    )
                    if physical_id in eligible_by_id:
                        continue
                key = (
                    target.occurrence_id,
                    requirement.package_name,
                    requirement.kind,
                )
                result[key] = target
    return result


def _platform_values_match(allowed_values: Sequence[str], actual: str) -> bool:
    """Apply npm positive and negated platform allowlist semantics."""
    if not allowed_values or actual == "unknown":
        return True
    excluded = {value[1:] for value in allowed_values if value.startswith("!")}
    included = {value for value in allowed_values if not value.startswith("!")}
    return actual not in excluded and (not included or actual in included)


def _build_evidence_domains(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
    candidate_domains: Mapping[str, Sequence[SolverVersionCandidate]],
    existing_packuments: Mapping[str, Mapping[str, Any]],
    settings: AppSettings,
    *,
    registry_fetcher: PackumentFetcher | None,
    runtime_fingerprint: SolverRuntimeFingerprint | None,
) -> tuple[list[SolverEvidenceDomain], str, list[str]]:
    """Build one-hop dependency witnesses from complete, compact solver metadata."""
    requirements = _evidence_requirement_keys(snapshot, targets, candidate_domains)
    package_names = sorted({key[1] for key in requirements})
    packuments = dict(existing_packuments)
    diagnostics: list[str] = []
    missing_names = [name for name in package_names if name not in packuments]
    if registry_fetcher is not None and missing_names:
        fetched, _complete, _digest_value, fetch_diagnostics = _fetch_candidate_packuments(
            missing_names, settings, registry_fetcher=registry_fetcher
        )
        packuments.update(fetched)
        diagnostics.extend(fetch_diagnostics)

    floors_by_package: dict[str, str] = {}
    for finding in findings:
        if not finding.fixed_version or _semver_key(finding.fixed_version) is None:
            continue
        current = floors_by_package.get(finding.vulnerable_package)
        if current is None or (_semver_key(finding.fixed_version) or (0, 0, 0)) > (
            _semver_key(current) or (0, 0, 0)
        ):
            floors_by_package[finding.vulnerable_package] = finding.fixed_version

    domains: list[SolverEvidenceDomain] = []
    maximum_candidates = max(1, settings.solver_max_candidates_per_target)
    maximum_domains = max(0, settings.solver_max_model_variables)
    for (source_id, package_name, dependency_kind), source in sorted(requirements.items()):
        packument = packuments.get(package_name)
        if packument is None:
            diagnostics.append(
                f"evidence packument unavailable for {package_name!r}; dependency left unmodeled"
            )
            continue
        try:
            validated = _solver_evidence_packument(package_name, packument)
        except (TypeError, ValueError):
            diagnostics.append(
                f"evidence packument failed validation for {package_name!r}; dependency left unmodeled"
            )
            continue
        try:
            payload_size = len(
                json.dumps(
                    validated,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        except (TypeError, ValueError):
            diagnostics.append(
                f"evidence packument malformed for {package_name!r}; dependency left unmodeled"
            )
            continue
        if payload_size > DEFAULT_MAX_PAYLOAD_BYTES:
            diagnostics.append(
                f"projected solver metadata exceeds {DEFAULT_MAX_PAYLOAD_BYTES} bytes for "
                f"{package_name!r}; dependency left unmodeled"
            )
            continue
        raw_versions = validated.get("versions")
        if not isinstance(raw_versions, Mapping):
            diagnostics.append(
                f"evidence versions malformed for {package_name!r}; dependency left unmodeled"
            )
            continue
        stable_versions = [
            (str(raw_version).strip().lstrip("vV"), metadata)
            for raw_version, metadata in sorted(raw_versions.items(), key=lambda item: str(item[0]))
            if _semver_key(str(raw_version).strip().lstrip("vV")) is not None
        ]
        if len(stable_versions) > maximum_candidates:
            diagnostics.append(
                f"evidence candidate limit exceeded for {package_name!r}; dependency left unmodeled"
            )
            continue
        if len({version for version, _metadata in stable_versions}) != len(stable_versions):
            diagnostics.append(
                f"evidence versions ambiguous for {package_name!r}; dependency left unmodeled"
            )
            continue

        floor = floors_by_package.get(package_name)
        floor_key = _semver_key(floor or "")
        candidate_versions: list[str] = []
        malformed_metadata = False
        for version, metadata in stable_versions:
            if not isinstance(metadata, Mapping):
                malformed_metadata = True
                break
            _requirements, engines, os_values, cpu_values, metadata_diagnostics = (
                _candidate_metadata(package_name, version, metadata)
            )
            if metadata_diagnostics:
                malformed_metadata = True
                break
            version_key = _semver_key(version)
            if floor_key is not None and version_key is not None and version_key < floor_key:
                continue
            compatible = True
            if runtime_fingerprint is not None:
                for engine_name, engine_range in engines.items():
                    if engine_name == "node":
                        actual = runtime_fingerprint.node_version
                    elif engine_name == "npm":
                        actual = runtime_fingerprint.npm_version
                    else:
                        continue
                    if actual == "unknown":
                        continue
                    checked = check_npm_range(engine_range, actual)
                    if checked.matches is None:
                        malformed_metadata = True
                        break
                    if not checked.matches:
                        compatible = False
                        break
                if malformed_metadata:
                    break
                if not _platform_values_match(
                    os_values, runtime_fingerprint.platform
                ) or not _platform_values_match(cpu_values, runtime_fingerprint.architecture):
                    compatible = False
            if compatible:
                candidate_versions.append(version)
        if malformed_metadata:
            diagnostics.append(
                f"evidence release metadata malformed for {package_name!r}; dependency left unmodeled"
            )
            continue
        if len(domains) >= maximum_domains:
            diagnostics.append(
                f"evidence model-variable limit exceeded; dependency {package_name!r} "
                "left unmodeled"
            )
            continue
        identity = {
            "source_occurrence_id": source_id,
            "package_name": package_name,
            "manifest_path": source.manifest_path,
            "workspace_id": source.workspace_id,
            "dependency_kind": dependency_kind,
        }
        variable_id = f"evidence:{_digest(identity)[:32]}"
        domains.append(
            SolverEvidenceDomain(
                variable_id=variable_id,
                package_name=package_name,
                source_occurrence_id=source_id,
                manifest_path=source.manifest_path,
                workspace_id=source.workspace_id,
                dependency_kind=dependency_kind,
                candidate_versions=sorted(candidate_versions),
            )
        )
    evidence_digest = _digest(
        {
            "packument_digests": {
                name: _digest(packuments[name]) for name in package_names if name in packuments
            },
            "missing_packages": sorted(set(package_names) - set(packuments)),
            "evidence_domains": [
                domain.model_dump(mode="json")
                for domain in sorted(domains, key=lambda item: item.variable_id)
            ],
            "diagnostics": _catalog_semantic_diagnostics(diagnostics),
        }
    )
    return domains, evidence_digest, sorted(set(diagnostics))


def _severity_rank_for_findings(findings: Sequence[SolverFindingRequirement]) -> dict[str, int]:
    return {
        finding.finding_id: _SEVERITY_RANK.get(str(finding.severity).upper(), 5)
        for finding in findings
    }


def _edge_kind_to_contract(kind: str) -> TaskDependencyKind:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"peer", "strict_peer", "peer_conflict", "peer_coupling"}:
        return TaskDependencyKind.PEER
    if normalized in {
        "workspace",
        "workspace_coupling",
        "scope",
        "pinned",
        "pinned_dependency",
        "exact_pinned",
    }:
        return TaskDependencyKind.WORKSPACE
    return TaskDependencyKind.RUNTIME


def _project_clusters(
    batches: Sequence[SolverBatch],
    dag: DAGBuildResult,
    phases: Sequence[SolverPhase],
) -> tuple[list[TaskCluster], dict[str, str], list[str], list[str]]:
    batch_by_id = {batch.batch_id: batch for batch in batches}
    batch_to_cluster = {batch_id: batch_id for batch_id in batch_by_id}
    task_to_cluster = {
        task_id: batch_to_cluster[batch_id]
        for task_id, batch_id in dag.task_to_batch.items()
        if batch_id in batch_to_cluster
    }
    dependencies_by_cluster: dict[str, list[TaskDependency]] = {
        batch_id: [] for batch_id in batch_by_id
    }
    seen: set[tuple[str, str]] = set()
    for edge in dag.edges:
        source = edge.source_task_id
        target = edge.target_task_id
        if not source or not target:
            continue
        kind = edge.edge_kind.lower().replace("-", "_")
        upstream, downstream = (
            (target, source) if _dependency_order_kind(kind) else (source, target)
        )
        if downstream not in task_to_cluster:
            continue
        key = (upstream, downstream)
        if key in seen or upstream == downstream:
            continue
        seen.add(key)
        dependencies_by_cluster[task_to_cluster[downstream]].append(
            TaskDependency(
                upstream_task_id=upstream,
                downstream_task_id=downstream,
                edge_type=_edge_kind_to_contract(kind),
                version_constraint=edge.version_range,
            )
        )
    clusters: list[TaskCluster] = []
    for batch in sorted(batches, key=lambda item: item.batch_id):
        clusters.append(
            TaskCluster(
                cluster_id=batch.batch_id,
                task_ids=list(batch.task_ids),
                dependencies=sorted(
                    dependencies_by_cluster[batch.batch_id],
                    key=lambda item: (item.upstream_task_id, item.downstream_task_id),
                ),
                reason=(
                    batch.diagnostic
                    or ("atomic solver batch" if batch.atomic else "singleton solver task")
                ),
                atomic=batch.atomic,
                dispatchable=batch.dispatchable,
            )
        )
    cluster_order = [
        batch_id for phase in phases for batch_id in phase.batch_ids if batch_id in batch_by_id
    ]
    if not cluster_order:
        cluster_order = [
            batch.batch_id for batch in sorted(batches, key=lambda item: item.batch_id)
        ]
    task_order = [
        task_id for cluster_id in cluster_order for task_id in batch_by_id[cluster_id].task_ids
    ]
    return clusters, task_to_cluster, cluster_order, task_order


def _prepare_portfolio_problem(
    repo_root: str | Path,
    groups: Sequence[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    target_packages: Iterable[str] | None = None,
    peer_conflict_pairs: Sequence[tuple[str, str]] = (),
    forced_singleton_task_ids: Sequence[str] = (),
    portfolio_replan_request: PortfolioReplanRequest | None = None,
    npm_snapshot: NpmGraphSnapshot | None = None,
    repository_fingerprint: str | None = None,
    registry_fetcher: PackumentFetcher | None = None,
    settings: AppSettings,
    runtime_fingerprint: SolverRuntimeFingerprint | None = None,
) -> _PreparedPortfolioProblem:
    """Prepare one immutable set of npm, registry, and solver inputs."""
    root = Path(repo_root).resolve()
    host_snapshot: NpmGraphSnapshot | None = None
    if npm_snapshot is None or repository_fingerprint is None:
        host_snapshot = load_npm_graph_snapshot(root)
    snapshot = npm_snapshot or host_snapshot or NpmGraphSnapshot()
    host_fingerprint = (
        repository_fingerprint
        or (host_snapshot.repository_fingerprint if host_snapshot is not None else None)
        or snapshot.repository_fingerprint
    )
    group_list = [group.model_copy(deep=True) for group in groups]
    queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    target_scope = (
        tuple(sorted({str(value).strip() for value in target_packages if str(value).strip()}))
        if target_packages is not None
        else None
    )
    explicit_pairs = list(peer_conflict_pairs)
    if portfolio_replan_request is not None:
        explicit_pairs.extend(portfolio_replan_request.peer_conflict_pairs)
    forced = tuple(
        sorted(
            set(str(value).strip() for value in forced_singleton_task_ids if str(value).strip())
            | set(
                portfolio_replan_request.forced_singleton_task_ids
                if portfolio_replan_request
                else ()
            )
        )
    )
    targets, findings, diagnostics = _build_targets_and_findings(snapshot, group_list, queue)
    if not targets:
        raise ValueError("Cannot build a portfolio plan without active SCA package tasks.")

    required_names = _required_candidate_package_names(targets, findings)
    packuments: dict[str, Mapping[str, Any]] = {}
    packument_complete = False
    packument_digest = _digest({"required_packages": required_names, "fresh_snapshot": False})
    if registry_fetcher is not None:
        (
            packuments,
            packument_complete,
            packument_digest,
            packument_diagnostics,
        ) = _fetch_candidate_packuments(
            required_names,
            settings,
            registry_fetcher=registry_fetcher,
        )
        diagnostics.extend(packument_diagnostics)
    transitive_compatible_versions = _resolve_transitive_parent_floors(
        targets,
        findings,
        queue,
        group_list,
        settings,
        diagnostics,
        packuments=packuments if registry_fetcher is not None else None,
    )
    subgraph = extract_solver_subgraph(
        snapshot,
        targets,
        findings,
        peer_conflict_pairs=explicit_pairs,
        forced_singleton_task_ids=forced,
    )
    diagnostics.extend(subgraph.diagnostics)
    domains, domain_complete, domain_digest = _candidate_domains(
        snapshot,
        targets,
        findings,
        queue,
        group_list,
        settings,
        diagnostics,
        transitive_compatible_versions=transitive_compatible_versions,
        packuments=packuments if registry_fetcher is not None else None,
    )
    evidence_domains, evidence_domain_digest, evidence_diagnostics = _build_evidence_domains(
        snapshot,
        targets,
        findings,
        domains,
        packuments,
        settings,
        registry_fetcher=registry_fetcher,
        runtime_fingerprint=runtime_fingerprint,
    )
    diagnostics.extend(evidence_diagnostics)
    subgraph = expand_candidate_relations(
        subgraph,
        domains,
        npm_snapshot=snapshot,
        evidence_domains=evidence_domains,
    )
    diagnostics.extend(subgraph.diagnostics)
    candidate_catalog_complete = (
        registry_fetcher is not None and packument_complete and domain_complete
    )
    candidate_catalog_digest = _digest(
        {
            "fresh_packument_digest": packument_digest,
            "domain_digest": domain_digest,
            "complete": candidate_catalog_complete,
            "evidence_domain_digest": evidence_domain_digest,
            "evidence_domains": [domain.model_dump(mode="json") for domain in evidence_domains],
            "diagnostics": _catalog_semantic_diagnostics(diagnostics),
        }
    )
    return _PreparedPortfolioProblem(
        repo_root=root,
        host_repository_fingerprint=host_fingerprint,
        workspace_repository_fingerprint=snapshot.repository_fingerprint,
        groups=group_list,
        task_queue=queue,
        npm_snapshot=snapshot,
        targets=targets,
        findings=findings,
        subgraph=subgraph,
        packuments=packuments,
        candidate_domains=domains,
        runtime_fingerprint=runtime_fingerprint,
        candidate_catalog_complete=candidate_catalog_complete,
        candidate_catalog_digest=candidate_catalog_digest,
        settings=settings,
        diagnostics=sorted(set(diagnostics)),
        target_packages=target_scope,
        peer_conflict_pairs=tuple(sorted(tuple(sorted(pair)) for pair in explicit_pairs)),
        forced_singleton_task_ids=forced,
    )


def _empty_solver_plan(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
    status: SolverStatus,
    diagnostics: Sequence[str],
) -> SolverRemediationPlan:
    input_digest = _digest(
        {
            "targets": [target.model_dump(mode="json") for target in targets],
            "findings": [finding.model_dump(mode="json") for finding in findings],
        }
    )
    return SolverRemediationPlan(
        status=status,
        input_digest=input_digest,
        domain_digest=_digest({}),
        repository_digest=snapshot.repository_fingerprint,
        task_revisions={target.task_id: 0 for target in targets},
    )


def _project_prepared_portfolio_plan(
    prepared: _PreparedPortfolioProblem,
    solver_plan: SolverRemediationPlan,
    *,
    portfolio_iteration: int,
) -> Any:
    """Project prepared inputs and one solver result into a public portfolio plan."""
    from remediation_engine.contracts.schemas import PortfolioPlan

    findings = prepared.findings
    subgraph = prepared.subgraph
    domains = prepared.candidate_domains
    queue = prepared.task_queue
    explicit_pairs = list(prepared.peer_conflict_pairs)
    forced = list(prepared.forced_singleton_task_ids)
    diagnostics = list(prepared.diagnostics)
    diagnostics.extend(solver_plan.diagnostics)
    selected = solver_plan.selected_plan
    decisions = selected.task_decisions if selected is not None else []
    batches, edges, cluster_diagnostics = cluster_packages(
        subgraph,
        decisions,
        forced_singleton_task_ids=forced,
        peer_conflict_pairs=explicit_pairs,
        scope_coupling=not bool(prepared.target_packages),
    )
    diagnostics.extend(cluster_diagnostics)
    dag = build_dependency_dag(subgraph, batches, edges)
    diagnostics.extend(dag.diagnostics)
    phases, phase_diagnostics = schedule_batches(
        dag,
        severity_rank=_severity_rank_for_findings(findings),
        phase_budget=prepared.settings.solver_phase_budget,
    )
    diagnostics.extend(phase_diagnostics)
    clusters, task_to_cluster, cluster_order, task_order = _project_clusters(batches, dag, phases)
    current_task_revisions = {
        task_id: queue[task_id].task_revision for task_id in task_order if task_id in queue
    }
    decision_by_task = {decision.task_id: decision for decision in decisions}
    planned_task_revisions = {
        task_id: revision + (1 if task_id in decision_by_task else 0)
        for task_id, revision in current_task_revisions.items()
    }
    task_strategies: dict[str, RoutingStrategy] = {}
    for task_id in task_order:
        task = queue.get(task_id)
        decision = decision_by_task.get(task_id)
        if task is None:
            continue
        if decision is not None and str(decision.selected_strategy).lower().replace("-", "_") in {
            "code_workaround",
            "workaround",
            "no_fix",
        }:
            task_strategies[task_id] = RoutingStrategy.CODE_WORKAROUND
        elif decision is not None and str(decision.selected_strategy).lower().replace("-", "_") in {
            "version_bump",
            "versionbump",
        }:
            task_strategies[task_id] = RoutingStrategy.VERSION_BUMP
        else:
            task_strategies[task_id] = task.strategy
    if not task_order:
        raise ValueError("solver produced no task projection")

    selected_assignment = {
        decision.target_occurrence_id: decision.selected_version
        for decision in decisions
        if decision.target_occurrence_id and decision.selected_version
    }
    graph_payload = {
        "snapshot": prepared.workspace_repository_fingerprint,
        "host_repository_fingerprint": prepared.host_repository_fingerprint,
        "subgraph": subgraph.model_dump(mode="json"),
        "domains": {
            key: [candidate.model_dump(mode="json") for candidate in values]
            for key, values in sorted(domains.items())
        },
        "candidate_catalog_complete": prepared.candidate_catalog_complete,
        "candidate_catalog_digest": prepared.candidate_catalog_digest,
        "batches": [batch.model_dump(mode="json") for batch in batches],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "phases": [phase.model_dump(mode="json") for phase in phases],
        "forced_singletons": forced,
        "peer_conflict_pairs": sorted(tuple(sorted(pair)) for pair in explicit_pairs),
    }
    graph_digest = _digest(graph_payload)
    solver_input_digest = solver_plan.input_digest
    plan_digest = _digest(
        {
            "repository_fingerprint": prepared.host_repository_fingerprint,
            "workspace_graph_digest": prepared.workspace_repository_fingerprint,
            "graph_digest": graph_digest,
            "solver_input_digest": solver_input_digest,
            "candidate_catalog_complete": prepared.candidate_catalog_complete,
            "candidate_catalog_digest": prepared.candidate_catalog_digest,
            "task_revisions": current_task_revisions,
            "planned_task_revisions": planned_task_revisions,
            "task_order": task_order,
            "cluster_order": cluster_order,
            "task_to_cluster": task_to_cluster,
            "selected_assignment": dict(sorted(selected_assignment.items())),
            "solver_status": solver_plan.status.value,
        }
    )
    plan_id = f"portfolio-{plan_digest[:24]}"
    solver_plan = solver_plan.model_copy(update={"task_revisions": planned_task_revisions})
    if selected is not None:
        selected = selected.model_copy(update={"batches": list(batches), "phases": list(phases)})
        solver_plan = solver_plan.model_copy(update={"selected_plan": selected})
    return PortfolioPlan(
        plan_id=plan_id,
        portfolio_plan_id=plan_id,
        repository_fingerprint=prepared.host_repository_fingerprint,
        workspace_graph_digest=prepared.workspace_repository_fingerprint,
        graph_digest=graph_digest,
        plan_digest=plan_digest,
        solver_input_digest=solver_input_digest,
        solver_plan=solver_plan,
        portfolio_iteration=max(0, int(portfolio_iteration)),
        task_ids=task_order,
        clusters=clusters,
        cluster_order=cluster_order,
        task_order=task_order,
        task_to_cluster=task_to_cluster,
        task_revisions=current_task_revisions,
        planned_task_revisions=planned_task_revisions,
        task_strategies=task_strategies,
        diagnostics=sorted(set(diagnostics)),
    )


def build_portfolio_plan(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    target_packages: Iterable[str] | None = None,
    peer_conflict_pairs: Iterable[tuple[str, str]] = (),
    forced_singleton_task_ids: Iterable[str] = (),
    settings: AppSettings | None = None,
    registry_fetcher: PackumentFetcher | None = None,
    portfolio_iteration: int = 0,
    portfolio_replan_request: PortfolioReplanRequest | None = None,
) -> Any:
    """Build a solver plan, optionally using fresh registry metadata.

    This path does not run npm or require Docker. A registry fetcher can be
    supplied when a complete current candidate catalog is required.
    """
    resolved_settings = settings or AppSettings()
    prepared = _prepare_portfolio_problem(
        repo_root,
        list(groups),
        task_queue,
        target_packages=target_packages,
        peer_conflict_pairs=tuple(peer_conflict_pairs),
        forced_singleton_task_ids=tuple(forced_singleton_task_ids),
        portfolio_replan_request=portfolio_replan_request,
        registry_fetcher=registry_fetcher,
        settings=resolved_settings,
    )
    solver_plan = solve_portfolio(
        prepared.subgraph,
        prepared.candidate_domains,
        settings=resolved_settings,
        candidate_catalog_complete=prepared.candidate_catalog_complete,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
    )
    return _project_prepared_portfolio_plan(
        prepared,
        solver_plan,
        portfolio_iteration=portfolio_iteration,
    )


def _expected_target_identity(
    group: VulnerabilityGroup, task: RemediationTask
) -> tuple[str, str, str]:
    """Return ``(group_id, manifest_path, target_package_name)`` for a task."""
    manifest_path = _manifest_path(group, Path(".")) or "package.json"
    package_name = (task.target_package_name or group.vulnerable_component or "").strip()
    if not package_name:
        raise ValueError(f"task {task.task_id!r} has no target package identity")
    return task.parent_group_id, manifest_path, package_name


def _override_instruction(
    package_name: str,
    target_version: str,
    manifest_path: str,
    dependency_type: str,
) -> str:
    """Build a deterministic manifest instruction for one approved override."""
    if dependency_type == "pnpm_overrides":
        declaration = f'"pnpm": {{"overrides": {{"{package_name}": "{target_version}"}}}}'
        manager = "pnpm overrides"
    elif dependency_type == "resolutions":
        declaration = f'"resolutions": {{"{package_name}": "{target_version}"}}'
        manager = "Yarn resolutions"
    else:
        declaration = f'"overrides": {{"{package_name}": "{target_version}"}}'
        manager = "npm overrides"
    return (
        f"Add or update {declaration} in {manifest_path} to pin the "
        f"transitive package via {manager}."
    )


def _source_migration_instruction(
    package_name: str,
    installed_version: str | None,
    selected_version: str | None,
) -> str:
    """Build the deterministic source-migration instruction for one major upgrade."""
    return (
        f"Upgrade package {package_name} from installed version {installed_version or 'unknown'} "
        f"to selected version {selected_version or 'unknown'}.\n"
        "Migrate all affected production and test code to the selected package API "
        "while preserving behavior; do not change the solver-approved package or version."
    )


def apply_portfolio_plan(
    plan: Any,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Commit solver-approved task decisions to detached task objects.

    Solver status, catalog completeness, queue revisions, and occurrence
    identity are checked before any decision is committed. If a package-
    resolution certificate is present, it must match the selected assignment.
    A stale or malformed plan raises ``ValueError`` so the graph boundary can
    route to teardown rather than partially applying a solver result.
    """
    solver_plan = plan.solver_plan
    certificate = plan.resolution_certificate
    selected = solver_plan.selected_plan if solver_plan is not None else None
    certificate_status = getattr(getattr(certificate, "status", None), "value", None)
    if (
        solver_plan is None
        or solver_plan.status != SolverStatus.OPTIMAL
        or not solver_plan.candidate_catalog_complete
        or selected is None
    ):
        raise ValueError("portfolio task decisions require an OPTIMAL complete solver plan")
    if certificate is not None and (
        str(certificate_status or getattr(certificate, "status", "")).upper() != "CERTIFIED"
        or certificate.portfolio_plan_id != plan.portfolio_plan_id
        or certificate.solver_input_digest != plan.solver_input_digest
        or certificate.repository_fingerprint != plan.repository_fingerprint
        or certificate.workspace_graph_digest != plan.workspace_graph_digest
        or certificate.candidate_catalog_digest != solver_plan.candidate_catalog_digest
        or certificate.task_revisions != plan.task_revisions
        or certificate.candidate_plan_id != selected.candidate_plan_id
        or certificate.candidate_assignment_digest
        != _digest(dict(sorted(selected.selected_candidate_versions.items())))
        or certificate.unresolved_coverage_ids
    ):
        raise ValueError("portfolio plan contains a mismatched package-resolution certificate")
    prepared_groups = [group.model_copy(deep=True) for group in groups]
    committed = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    diagnostics: list[str] = []
    plan_id = getattr(plan, "portfolio_plan_id", None) or getattr(plan, "plan_id", None)
    if not plan_id:
        raise ValueError("portfolio plan has no immutable plan identity")
    plan_task_ids = list(getattr(plan, "task_ids", ()) or ())
    baseline_revisions = dict(getattr(plan, "task_revisions", {}) or {})
    groups_by_id = {group.group_id: group for group in prepared_groups}

    # This is the compare-and-swap boundary for the outer queue projection.
    for task_id in plan_task_ids:
        task = committed.get(task_id)
        if task is None:
            raise ValueError(f"portfolio plan references missing task {task_id!r}")
        baseline = baseline_revisions.get(task_id)
        if baseline is None:
            raise ValueError(f"portfolio plan has no baseline revision for task {task_id!r}")
        if task.task_revision != int(baseline):
            raise ValueError(
                f"portfolio plan is stale for task {task_id!r}: "
                f"expected revision {baseline}, current {task.task_revision}"
            )

    selected_decisions = list(getattr(selected, "task_decisions", ()) or ())
    decisions: dict[str, Any] = {}
    for decision in selected_decisions:
        if decision.task_id in decisions:
            raise ValueError(f"portfolio plan has duplicate solver decision {decision.task_id!r}")
        decisions[decision.task_id] = decision

    for task_id in plan_task_ids:
        task = committed[task_id]
        decision = decisions.get(task_id)
        if decision is None:
            if task.current_attempt_id is not None:
                diagnostics.append(
                    f"task {task_id!r} has an active attempt; plan decision not applied"
                )
            continue
        group = groups_by_id.get(task.parent_group_id)
        if group is None:
            raise ValueError(
                f"portfolio plan task {task_id!r} references missing group {task.parent_group_id!r}"
            )
        expected_group_id, expected_manifest, expected_package = _expected_target_identity(
            group,
            task,
        )
        decision_lockfile_key = str(decision.lockfile_package_key or "").replace("\\", "/")
        direct_occurrence = make_occurrence_id(expected_manifest, expected_package)
        physical_occurrence = make_occurrence_id(
            expected_manifest,
            expected_package,
            decision_lockfile_key,
        )
        decision_occurrence = str(decision.target_occurrence_id or "")
        if not lockfile_key_matches_package(decision_lockfile_key, expected_package):
            raise ValueError(
                f"portfolio decision for task {task_id!r} has invalid lockfile_package_key "
                f"{decision_lockfile_key!r}"
            )
        if decision_occurrence not in {direct_occurrence, physical_occurrence}:
            raise ValueError(
                f"portfolio decision for task {task_id!r} has invalid target occurrence "
                f"{decision_occurrence!r}"
            )
        if (
            decision_occurrence == physical_occurrence
            and physical_occurrence != direct_occurrence
            and decision.dependency_type not in _OVERRIDE_DEPENDENCY_TYPES
        ):
            raise ValueError(
                f"nested lockfile target for task {task_id!r} requires a package override"
            )
        identity_values = {
            "target_group_id": expected_group_id,
            "target_package_name": expected_package,
            "manifest_path": expected_manifest,
            "lockfile_package_key": decision_lockfile_key,
        }
        for field_name, expected in identity_values.items():
            actual = getattr(decision, field_name, None)
            if actual != expected:
                raise ValueError(
                    f"portfolio decision for task {task_id!r} has invalid {field_name}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        if task.current_attempt_id is not None:
            diagnostics.append(f"task {task_id!r} has an active attempt; plan decision not applied")
            continue
        updates: dict[str, Any] = {}
        decision_strategy = str(decision.selected_strategy).lower().replace("-", "_")
        if decision_strategy in {"version_bump", "versionbump"}:
            updates["strategy"] = RoutingStrategy.VERSION_BUMP
        elif decision_strategy in {"code_workaround", "workaround", "no_fix"}:
            updates["strategy"] = RoutingStrategy.CODE_WORKAROUND
        if decision.selected_version is not None:
            updates["selected_version"] = decision.selected_version
        allowed = list(
            dict.fromkeys(
                value
                for value in (decision.selected_version, *decision.allowed_alternative_versions)
                if value
            )
        )
        if decision.selected_version is None:
            updates["selected_version"] = None
        updates["allowed_target_versions"] = allowed
        updates["allowed_dependency_types"] = list(decision.allowed_dependency_types)
        updates["selected_plan_issue_ids"] = list(decision.selected_plan_issue_ids)
        if decision.dependency_type:
            updates["target_dependency_type"] = decision.dependency_type
        updates["portfolio_plan_id"] = plan_id
        try:
            updates["strategy_stage"] = SCARemediationStage(decision.strategy_stage)
        except ValueError:
            diagnostics.append(
                f"task {task_id!r} has unknown strategy stage {decision.strategy_stage!r}"
            )
        migration_required = bool(getattr(decision, "requires_source_migration", False))
        if decision.exact_instruction:
            instruction = decision.exact_instruction
            if (
                migration_required
                and decision.dependency_type in _OVERRIDE_DEPENDENCY_TYPES
                and decision.selected_version
            ):
                instruction = (
                    _override_instruction(
                        expected_package,
                        decision.selected_version,
                        expected_manifest,
                        decision.dependency_type,
                    )
                    + "\n"
                    + instruction
                )
            updates["instruction"] = instruction
        elif migration_required:
            instruction = _source_migration_instruction(
                expected_package,
                getattr(decision, "installed_version", None),
                decision.selected_version,
            )
            if decision.dependency_type in _OVERRIDE_DEPENDENCY_TYPES and decision.selected_version:
                instruction = (
                    _override_instruction(
                        expected_package,
                        decision.selected_version,
                        expected_manifest,
                        decision.dependency_type,
                    )
                    + "\n"
                    + instruction
                )
            updates["instruction"] = instruction
        elif decision.dependency_type in _OVERRIDE_DEPENDENCY_TYPES and decision.selected_version:
            updates["instruction"] = _override_instruction(
                expected_package,
                decision.selected_version,
                expected_manifest,
                decision.dependency_type,
            )
        elif not task.instruction:
            updates["instruction"] = (
                f"Apply the outer-solver-approved dependency decision for task {task_id}."
            )
        if any(getattr(task, key) != value for key, value in updates.items()):
            updates["task_revision"] = task.task_revision + 1
            committed[task_id] = task.model_copy(update=updates)
    return prepared_groups, committed, sorted(set(diagnostics))


__all__ = ["apply_portfolio_plan", "build_portfolio_plan", "prepare_portfolio_inputs"]
