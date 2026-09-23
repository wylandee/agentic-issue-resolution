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
    SolverFindingRequirement,
    SolverPhase,
    SolverRemediationPlan,
    SolverStatus,
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
from remediation_engine.solver.subgraph import extract_solver_subgraph
from remediation_engine.tools.npm_graph import (
    NpmDependencyRecord,
    NpmGraphSnapshot,
    load_npm_graph_snapshot,
    make_occurrence_id,
)
from remediation_engine.tools.registry_cache import RegistryPackumentCache, load_or_fetch_packument

_STABLE_VERSION = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)$")
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


def _digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


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


def _target_record(
    records: Mapping[tuple[str, str], NpmDependencyRecord],
    manifest_path: str,
    package_name: str,
    fallback: str,
) -> NpmDependencyRecord | None:
    record = records.get((manifest_path, package_name))
    if record is not None:
        return record
    if fallback != package_name:
        return records.get((manifest_path, fallback))
    return None


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
        if value and _STABLE_VERSION.fullmatch(str(value).strip()):
            return str(value).strip().lstrip("vV")
    return "0.0.0"


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
        record = _target_record(records, manifest_path, target_name, package_name)
        lock_key = (
            record.lockfile_package_key
            if record and record.lockfile_package_key
            else f"node_modules/{target_name}"
        )
        occurrence_id = (
            record.occurrence_id if record else make_occurrence_id(manifest_path, target_name)
        )
        if not package_name or not target_name or not manifest_path or manager not in {"", "npm"}:
            diagnostics.append(f"task {task.task_id!r} has an unsupported or ambiguous npm target")
            eligible = False
        else:
            eligible = True
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
        target = SolverTarget(
            occurrence_id=occurrence_id,
            task_id=task.task_id,
            group_id=group.group_id,
            package_name=package_name,
            target_package_name=target_name,
            manifest_path=manifest_path or "package.json",
            lockfile_package_key=lock_key,
            installed_version=_installed_version(group, task, record),
            dependency_type=task.target_dependency_type
            or (
                group.parent_declaration_type
                if group.parent_package_name
                else (record.declaration_type if record else "dependencies")
            ),
            strategy=target_strategy,
            eligible_for_atomic_update=(
                eligible
                and task.status not in _TERMINAL_STATUSES
                and task.current_attempt_id is None
                and (
                    target_strategy == "no_fix"
                    or task.strategy == RoutingStrategy.VERSION_BUMP
                    or has_version_evidence
                )
            ),
            workspace_id=_workspace_id(snapshot, manifest_path),
            dependency_ancestry=tuple(group.dependency_ancestry),
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
            workaround = bool(workaround_plan_ids)
            findings.append(
                SolverFindingRequirement(
                    finding_id=finding_id,
                    cve_id=issue.cve_id,
                    ghsa_id=issue.ghsa_id,
                    severity=issue.severity.value
                    if isinstance(issue.severity, Severity)
                    else str(issue.severity),
                    vulnerable_package=package_name,
                    target_occurrence_id=occurrence_id,
                    fixed_version=fixed,
                    direct_parent_name=group.parent_package_name,
                    direct_parent_minimum_version=task.parent_minimum_version,
                    strategy_stage=task.strategy_stage.value,
                    workaround_available=workaround,
                    workaround_plan_ids=workaround_plan_ids,
                    is_transitive=bool(group.parent_package_name),
                )
            )
    return targets, findings, sorted(set(diagnostics))


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
) -> dict[str, set[str]]:
    """Resolve initial transitive parent floors from cached published metadata.

    Initial transitive tasks intentionally do not commit ``parent_minimum_version``:
    the outer planner must prove which parent releases can carry the fixed child.
    A configured registry cache is the only metadata boundary used here.  When
    no cache is configured, the finding remains without a solver floor and the
    solver can fail closed rather than selecting an arbitrary parent release.

    Returns:
        Mapping from parent occurrence IDs to the stable parent versions proven
        compatible with the vulnerable child.  An empty set is an explicit
        proof failure for a configured cache.
    """
    cache = RegistryPackumentCache(settings.solver_cache_dir) if settings.solver_cache_dir else None
    if cache is None:
        return {}
    groups_by_id = {group.group_id: group for group in groups}
    tasks_by_id = dict(task_queue)
    compatible_by_target: dict[str, set[str]] = {}
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
            from remediation_engine.tools.registry_tools import (
                _fetch_package_data,
                select_npm_parent_version,
            )

            parent_data = load_or_fetch_packument(parent_name, _fetch_package_data, cache=cache)
            ancestry = tuple(group.dependency_ancestry)
            registry_data = {parent_name: parent_data}
            for intermediate in ancestry[1:-1]:
                if intermediate and intermediate not in registry_data:
                    registry_data[intermediate] = load_or_fetch_packument(
                        intermediate, _fetch_package_data, cache=cache
                    )
            result = select_npm_parent_version(
                parent_data,
                parent_package_name=parent_name,
                child_package_name=group.vulnerable_component or target.package_name,
                child_fixed_version=child_floor,
                installed_parent_version=target.installed_version,
                selection="minimum",
                dependency_ancestry=ancestry or None,
                registry_data_by_package=registry_data,
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
    dependency_ranges: Mapping[str, str] | None = None,
    peer_ranges: Mapping[str, str] | None = None,
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
        dependency_ranges=dict(dependency_ranges or {}),
        peer_ranges=dict(peer_ranges or {}),
        published_dependencies=dict(dependency_ranges or {}),
        published_peer_dependencies=dict(peer_ranges or {}),
    )


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
) -> dict[str, list[SolverVersionCandidate]]:
    """Prepare bounded local and optionally cached registry domains."""
    groups_by_task = {
        task.task_id: groups_by_id
        for task, groups_by_id in (
            (task, next((g for g in groups if g.group_id == task.parent_group_id), None))
            for task in task_queue.values()
        )
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
            _parent_proof: set[str] | None = parent_proof,
            _values: list[SolverVersionCandidate] = values,
        ) -> None:
            """Append a candidate, applying transitive compatibility proof."""
            if candidate is None:
                return
            if _parent_proof is not None and candidate.version not in _parent_proof:
                candidate = candidate.model_copy(update={"meets_security_floor": False})
            _values.append(candidate)

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
                add_candidate(_candidate(str(version), source=source, floor=floor, attempted=False))
        # Registry access is explicit and bounded: a configured solver cache is
        # the opt-in network/cache boundary for the outer planner. Unit tests
        # without a cache remain entirely offline and use local plan evidence.
        if cache is not None and target.target_package_name:
            try:
                from remediation_engine.tools.registry_tools import _fetch_package_data

                packument = load_or_fetch_packument(
                    target.target_package_name, _fetch_package_data, cache=cache
                )
                for raw_version, metadata in sorted((packument.get("versions") or {}).items()):
                    if not isinstance(metadata, Mapping):
                        continue
                    dependencies: dict[str, str] = {}
                    for field in ("dependencies", "optionalDependencies"):
                        values_for_field = metadata.get(field)
                        if isinstance(values_for_field, Mapping):
                            dependencies.update(
                                {
                                    str(name): str(requirement)
                                    for name, requirement in values_for_field.items()
                                    if isinstance(requirement, str)
                                }
                            )
                    peers = (
                        metadata.get("peerDependencies")
                        if isinstance(metadata.get("peerDependencies"), Mapping)
                        else {}
                    )
                    add_candidate(
                        _candidate(
                            str(raw_version),
                            source="registry",
                            floor=floor,
                            dependency_ranges=dependencies,
                            peer_ranges=peers,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(
                    f"registry domain unavailable for {target.target_package_name}: {exc}"
                )
        dedup: dict[str, SolverVersionCandidate] = {}
        for value in sorted(values, key=lambda item: (item.semver_key, item.version, item.source)):
            existing = dedup.get(value.version)
            if existing is None or (
                not existing.dependency_ranges
                and not existing.peer_ranges
                and (value.dependency_ranges or value.peer_ranges)
            ):
                dedup[value.version] = value
        ordered = list(dedup.values())
        limit = max(1, settings.solver_max_candidates_per_target)
        if len(ordered) > limit:
            eligible = [value for value in ordered if value.meets_security_floor]
            latest = (eligible or ordered)[-1]
            required: list[SolverVersionCandidate] = [latest]
            anchor_versions = [
                target.installed_version,
                floor,
                task.selected_version if task else None,
                *(task.allowed_target_versions if task else []),
            ]
            for raw_version in anchor_versions:
                normalized = str(raw_version or "").strip().lstrip("vV")
                candidate = dedup.get(normalized)
                if candidate is not None and candidate not in required:
                    required.append(candidate)
            for candidate in reversed(ordered):
                if len(required) >= limit:
                    break
                if candidate not in required:
                    required.append(candidate)
            ordered = sorted(required[:limit], key=lambda item: item.semver_key)
        domains[target.occurrence_id] = ordered
        if not domains[target.occurrence_id] and target.eligible_for_atomic_update:
            diagnostics.append(f"empty candidate domain for {target.occurrence_id}")
    if cache is not None:
        diagnostics.extend(cache.diagnostics)
    return domains


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


def build_portfolio_plan(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    target_packages: Iterable[str] | None = None,
    peer_conflict_pairs: Iterable[tuple[str, str]] = (),
    forced_singleton_task_ids: Iterable[str] = (),
    settings: AppSettings | None = None,
    portfolio_iteration: int = 0,
    portfolio_replan_request: PortfolioReplanRequest | None = None,
) -> Any:
    """Build a complete solver-backed immutable PortfolioPlan projection.

    Args:
        repo_root: Repository whose npm graph supplies occurrence metadata.
        groups: Prepared vulnerability and coordination groups.
        task_queue: Supervisor-owned task projection.
        target_packages: Optional development package scope. Scoped plans do
            not use namespace membership as atomic batch evidence.
        peer_conflict_pairs: Explicit QA-discovered peer conflict pairs.
        forced_singleton_task_ids: Tasks that must remain singleton batches.
        settings: Solver and registry settings.
        portfolio_iteration: Current outer portfolio iteration.
        portfolio_replan_request: Optional Supervisor replan constraints.

    Returns:
        An immutable solver-backed portfolio plan.
    """
    from remediation_engine.contracts.schemas import PortfolioPlan

    resolved_settings = settings or AppSettings()
    group_list = [group.model_copy(deep=True) for group in groups]
    queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    snapshot = load_npm_graph_snapshot(repo_root)
    targets, findings, diagnostics = _build_targets_and_findings(snapshot, group_list, queue)
    if not targets:
        raise ValueError("Cannot build a portfolio plan without active SCA package tasks.")
    transitive_compatible_versions = _resolve_transitive_parent_floors(
        targets,
        findings,
        queue,
        group_list,
        resolved_settings,
        diagnostics,
    )
    explicit_pairs = list(peer_conflict_pairs)
    if portfolio_replan_request is not None:
        explicit_pairs.extend(portfolio_replan_request.peer_conflict_pairs)
    forced = sorted(
        set(str(item).strip() for item in forced_singleton_task_ids if str(item).strip())
        | set(
            portfolio_replan_request.forced_singleton_task_ids if portfolio_replan_request else ()
        )
    )
    subgraph = extract_solver_subgraph(
        snapshot,
        targets,
        findings,
        peer_conflict_pairs=explicit_pairs,
        forced_singleton_task_ids=forced,
    )
    diagnostics.extend(subgraph.diagnostics)
    domains = _candidate_domains(
        snapshot,
        targets,
        findings,
        queue,
        group_list,
        resolved_settings,
        diagnostics,
        transitive_compatible_versions=transitive_compatible_versions,
    )
    solver_plan = solve_portfolio(subgraph, domains, settings=resolved_settings)
    diagnostics.extend(solver_plan.diagnostics)
    selected = solver_plan.selected_plan
    decisions = selected.task_decisions if selected is not None else []
    batches, edges, cluster_diagnostics = cluster_packages(
        subgraph,
        decisions,
        forced_singleton_task_ids=forced,
        peer_conflict_pairs=explicit_pairs,
        scope_coupling=not bool(target_packages),
    )
    diagnostics.extend(cluster_diagnostics)
    dag = build_dependency_dag(subgraph, batches, edges)
    diagnostics.extend(dag.diagnostics)
    phases, phase_diagnostics = schedule_batches(
        dag,
        severity_rank=_severity_rank_for_findings(findings),
        phase_budget=resolved_settings.solver_phase_budget,
    )
    diagnostics.extend(phase_diagnostics)
    clusters, task_to_cluster, cluster_order, task_order = _project_clusters(batches, dag, phases)
    current_task_revisions = {
        task_id: queue[task_id].task_revision for task_id in task_order if task_id in queue
    }
    decision_by_task = {decision.task_id: decision for decision in decisions}
    task_revisions = dict(current_task_revisions)
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
    # Non-SCA tasks are not solver targets, but exact PortfolioPlan membership
    # remains the active SCA leaf projection consumed by the Supervisor.
    if not task_order:
        raise ValueError("solver produced no task projection")
    graph_payload = {
        "snapshot": snapshot.repository_fingerprint,
        "subgraph": subgraph.model_dump(mode="json"),
        "domains": {
            key: [candidate.model_dump(mode="json") for candidate in values]
            for key, values in sorted(domains.items())
        },
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
            "repository_fingerprint": snapshot.repository_fingerprint,
            "graph_digest": graph_digest,
            "solver_input_digest": solver_input_digest,
            "task_revisions": task_revisions,
            "planned_task_revisions": planned_task_revisions,
            "task_order": task_order,
            "cluster_order": cluster_order,
            "task_to_cluster": task_to_cluster,
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
        repository_fingerprint=snapshot.repository_fingerprint,
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
        task_revisions=task_revisions,
        planned_task_revisions=planned_task_revisions,
        task_strategies=task_strategies,
        diagnostics=sorted(set(diagnostics)),
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


def apply_portfolio_plan(
    plan: Any,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Commit solver-approved task decisions to detached task objects.

    The plan's queue revisions and occurrence identity are checked before any
    decision is committed.  A stale or malformed plan raises ``ValueError`` so
    the graph boundary can route to teardown rather than partially applying a
    solver result.
    """
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

    selected = plan.solver_plan.selected_plan if plan.solver_plan else None
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
            group, task
        )
        expected_occurrence = make_occurrence_id(expected_manifest, expected_package)
        expected_lockfile_key = f"node_modules/{expected_package}"
        identity_values = {
            "target_occurrence_id": expected_occurrence,
            "target_group_id": expected_group_id,
            "target_package_name": expected_package,
            "manifest_path": expected_manifest,
            "lockfile_package_key": expected_lockfile_key,
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
        if decision.exact_instruction:
            updates["instruction"] = decision.exact_instruction
        elif not task.instruction:
            updates["instruction"] = (
                f"Apply the outer-solver-approved dependency decision for task {task_id}."
            )
        if any(getattr(task, key) != value for key, value in updates.items()):
            updates["task_revision"] = task.task_revision + 1
            committed[task_id] = task.model_copy(update=updates)
    return prepared_groups, committed, sorted(set(diagnostics))


__all__ = ["apply_portfolio_plan", "build_portfolio_plan", "prepare_portfolio_inputs"]
