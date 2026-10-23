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
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cmp_to_key
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
    SolverEdge,
    SolverEvidenceDomain,
    SolverFindingRequirement,
    SolverPackageOccurrence,
    SolverPhase,
    SolverPythonRuntimeFingerprint,
    SolverRemediationPlan,
    SolverRuntimeFingerprint,
    SolverStatus,
    SolverSubgraph,
    SolverTarget,
    SolverVersionCandidate,
)
from remediation_engine.contracts.version_policy import (
    compare_maven_versions,
    is_stable_maven_version,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import _as_domain as _normalize_solver_candidate_domain
from remediation_engine.solver.cpsat import solve_portfolio
from remediation_engine.solver.graph import (
    _dependency_order_kind,
    build_dependency_dag,
    cluster_packages,
    schedule_batches,
)
from remediation_engine.solver.subgraph import expand_candidate_relations, extract_solver_subgraph
from remediation_engine.tools.maven_manifest_locator import (
    find_dependency_in_pom,
    locate_maven_manifests,
    parse_pom_xml,
)
from remediation_engine.tools.maven_registry_tools import fetch_maven_registry_candidates
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
from remediation_engine.tools.odc_parser import extract_odc_fixed_version_floor
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


def _python_numeric_solver_version(value: str) -> tuple[str, tuple[int, ...]]:
    """Normalize stable final/post PEP 440 releases into an ordered key.

    The adapter maps epoch-zero final and post releases (without local
    versions) into a lexicographic tuple. A post discriminator preserves
    ``final < post < next release``. Short releases are zero-padded, and
    equivalent release spellings collapse to one canonical version.
    """
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(str(value).strip())
    except InvalidVersion as exc:
        raise ValueError(f"invalid PEP 440 version {value!r}") from exc
    if (
        parsed.epoch != 0
        or parsed.is_prerelease
        or parsed.is_devrelease
        or parsed.local is not None
        or len(parsed.release) > 3
    ):
        raise ValueError(f"PEP 440 version {value!r} is outside the supported stable subset")
    release_key = tuple(parsed.release) + (0,) * (3 - len(parsed.release))
    post_key = (1, parsed.post or 0) if parsed.is_postrelease else (0, 0)
    normalized = ".".join(str(part) for part in release_key)
    if parsed.is_postrelease:
        normalized += f".post{parsed.post or 0}"
    return normalized, (*release_key, *post_key)


def _python_scanner_fixed_version(issue: Any) -> str | None:
    """Return an explicit scanner floor or a package-matched ODC CPE bound."""
    explicit = str(getattr(issue, "fixed_version", None) or "").strip()
    if explicit:
        return explicit
    source = getattr(issue, "source", None)
    source_name = str(getattr(source, "value", source) or "").strip().casefold()
    if source_name != "odc":
        return None
    raw_payload = getattr(issue, "raw_payload", None)
    vulnerability = raw_payload.get("vulnerability") if isinstance(raw_payload, Mapping) else None
    package_name = str(getattr(issue, "package_name", None) or "").strip()
    if not isinstance(vulnerability, dict) or not package_name:
        return None
    return extract_odc_fixed_version_floor(vulnerability, package_name)


def _pipfile_declaration_is_unconditional(raw_name: str, value: Any) -> bool:
    """Return whether a Pipfile declaration is a static, unconditional pin."""
    from packaging.requirements import InvalidRequirement, Requirement

    try:
        parsed_name = Requirement(raw_name)
    except InvalidRequirement:
        return False
    if parsed_name.extras or parsed_name.marker:
        return False
    if isinstance(value, str):
        version_specifier = value.strip()
    elif isinstance(value, Mapping):
        if set(value) - {"version"}:
            return False
        version_specifier = value.get("version", "*")
        if not isinstance(version_specifier, str):
            return False
        version_specifier = version_specifier.strip()
    else:
        return False
    if version_specifier == "*":
        return True
    try:
        requirement = Requirement(f"{parsed_name.name}{version_specifier}")
    except InvalidRequirement:
        return False
    return not requirement.extras and requirement.marker is None


def _pipfile_declaration_matches_locked_version(
    raw_name: str, value: Any, locked_version: str
) -> bool:
    """Require one unconditional Pipfile constraint to admit its exact lock version."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version

    if not _pipfile_declaration_is_unconditional(raw_name, value):
        return False
    try:
        parsed_name = Requirement(raw_name)
        if isinstance(value, Mapping):
            specifier_text = str(value.get("version", "*")).strip()
        else:
            specifier_text = str(value).strip()
        if specifier_text == "*":
            return True
        declaration = Requirement(f"{parsed_name.name}{specifier_text}")
        return declaration.specifier.contains(Version(locked_version), prereleases=False)
    except (InvalidRequirement, InvalidVersion, TypeError, ValueError):
        return False


def _locked_python_version(category: Any, package_name: str) -> str | None:
    """Read one stable PEP 440 package version from a Pipfile.lock category."""
    from packaging.version import InvalidVersion, Version

    from remediation_engine.tools.package_identity import normalize_python_package_name

    if not isinstance(category, Mapping):
        return None
    records = [
        value
        for name, value in category.items()
        if normalize_python_package_name(str(name)) == package_name
    ]
    if len(records) != 1 or not isinstance(records[0], Mapping):
        return None
    raw = str(records[0].get("version") or "").strip()
    if raw.startswith("=="):
        raw = raw[2:]
    elif raw.startswith("="):
        raw = raw[1:]
    try:
        parsed = Version(raw)
    except InvalidVersion:
        return None
    if parsed.is_prerelease or parsed.is_devrelease or parsed.local is not None or parsed.epoch:
        return None
    return str(parsed)


def _resolve_python_pipfile_parent_context(
    repo_root: Path,
    manifest_path: str,
    group: VulnerabilityGroup,
    task: RemediationTask,
    requires_dist_fetcher: Any,
) -> dict[str, str] | None:
    """Prove one category-matched, unconditional Pipfile parent edge.

    The proof binds the declared parent and vulnerable child to the same
    Pipfile/Pipfile.lock category, verifies the current locked versions against
    the parent's PyPI ``Requires-Dist`` metadata, and rejects ambiguous,
    conditional, or deeper ancestry. Registry metadata is read by the
    Supervisor portfolio preparation boundary and a lookup failure is not
    evidence.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version

    from remediation_engine.tools.package_identity import normalize_python_package_name
    from remediation_engine.tools.pypi_registry_tools import get_pypi_release_requires_dist

    child_name = normalize_python_package_name(str(group.vulnerable_component or ""))
    if not child_name or Path(manifest_path).name.casefold() != "pipfile":
        return None
    ancestry = [
        normalize_python_package_name(str(name))
        for name in group.dependency_ancestry
        if str(name).strip()
    ]
    if len(ancestry) > 2 or (ancestry and ancestry[-1] != child_name):
        return None
    parent_hints = {
        normalize_python_package_name(str(value))
        for value in (
            task.parent_package_name,
            group.parent_package_name,
            task.target_package_name if task.target_package_name != child_name else None,
            ancestry[0] if len(ancestry) == 2 else None,
        )
        if value
    }
    if len(parent_hints) > 1:
        return None
    hinted_parent = next(iter(parent_hints), None)
    root = repo_root.resolve()
    pipfile = (root / manifest_path).resolve()
    try:
        pipfile.relative_to(root)
        pipfile_data = tomllib.loads(pipfile.read_text(encoding="utf-8"))
        lock_path = pipfile.with_name("Pipfile.lock")
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(pipfile_data, Mapping) or not isinstance(lock_data, Mapping):
        return None

    # The group's scanner evidence is the only source of the child's fix floor.
    child_floors = {
        floor
        for issue in group.issues
        if (issue.cve_id or issue.ghsa_id or issue.finding_id)
        and (floor := _python_scanner_fixed_version(issue))
    }
    try:
        child_floor = max(
            (_python_numeric_solver_version(value) for value in child_floors),
            key=lambda item: item[1],
        )[0]
    except (ValueError, TypeError):
        return None

    possible_edges: list[tuple[str, str, str, str, bool]] = []
    for pip_category, lock_category in (("packages", "default"), ("dev-packages", "develop")):
        declarations = pipfile_data.get(pip_category)
        if declarations is None:
            continue
        if not isinstance(declarations, Mapping):
            return None
        locked = lock_data.get(lock_category)
        child_version = _locked_python_version(locked, child_name)
        if child_version is None:
            continue
        for raw_name, declaration in declarations.items():
            try:
                parsed_name = Requirement(str(raw_name))
            except InvalidRequirement:
                return None
            parent_name = normalize_python_package_name(parsed_name.name)
            if hinted_parent and parent_name != hinted_parent:
                continue
            parent_version = _locked_python_version(locked, parent_name)
            if parent_version is None:
                continue
            possible_edges.append(
                (
                    parent_name,
                    parent_version,
                    child_version,
                    pip_category,
                    _pipfile_declaration_matches_locked_version(
                        str(raw_name), declaration, parent_version
                    ),
                )
            )

    proven: list[tuple[str, str, str, str, str]] = []
    unproven_child_edge = False
    fetch_requires = requires_dist_fetcher or get_pypi_release_requires_dist
    try:
        for parent_name, parent_version, child_version, category, unconditional in possible_edges:
            locked_child = Version(child_version)
            for raw_requirement in fetch_requires(parent_name, parent_version):
                requirement = Requirement(raw_requirement)
                if normalize_python_package_name(requirement.name) != child_name:
                    continue
                if not requirement.specifier.contains(locked_child, prereleases=False):
                    continue
                if requirement.marker is not None or requirement.extras or not unconditional:
                    unproven_child_edge = True
                    continue
                proven.append(
                    (
                        parent_name,
                        parent_version,
                        child_version,
                        category,
                        "default" if category == "packages" else "develop",
                    )
                )
    except (InvalidRequirement, InvalidVersion, TypeError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - missing PyPI metadata is not ancestry proof
        return None
    unique = list(dict.fromkeys(proven))
    if unproven_child_edge or len(unique) != 1:
        return None
    parent_name, parent_version, child_version, category, lock_category = unique[0]
    return {
        "parent_name": parent_name,
        "parent_version": parent_version,
        "child_name": child_name,
        "child_version": child_version,
        "child_floor": child_floor,
        "category": category,
        "lock_category": lock_category,
        "manifest_path": pipfile.relative_to(root).as_posix(),
        "lock_path": lock_path.relative_to(root).as_posix(),
    }


def _python_parent_release_accepts_child_floor(
    parent_name: str,
    parent_version: str,
    child_name: str,
    child_floor: str,
    requires_dist_fetcher: Any,
) -> bool:
    """Return whether one parent release unconditionally allows the child fix."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version

    from remediation_engine.tools.package_identity import normalize_python_package_name

    try:
        fixed = Version(child_floor)
        for raw_requirement in requires_dist_fetcher(parent_name, parent_version):
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                return False
            if normalize_python_package_name(requirement.name) != child_name:
                continue
            if requirement.marker is not None or requirement.extras:
                continue
            return requirement.specifier.contains(fixed, prereleases=False)
    except (InvalidVersion, TypeError, ValueError):
        return False
    except Exception:  # noqa: BLE001 - metadata failure cannot authorize a candidate
        return False
    return False


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


@dataclass(frozen=True)
class _PreparedPythonPortfolioProblem:
    """Common solver projection inputs for a statically declared PyPI target."""

    repo_root: Path
    host_repository_fingerprint: str
    workspace_repository_fingerprint: str
    groups: list[VulnerabilityGroup]
    task_queue: dict[str, RemediationTask]
    targets: list[SolverTarget]
    findings: list[SolverFindingRequirement]
    subgraph: SolverSubgraph
    packuments: Mapping[str, Mapping[str, Any]]
    candidate_domains: Mapping[str, list[SolverVersionCandidate]]
    runtime_fingerprint: SolverPythonRuntimeFingerprint | None
    candidate_catalog_complete: bool
    candidate_catalog_digest: str
    settings: AppSettings
    diagnostics: list[str]
    target_packages: tuple[str, ...] | None
    peer_conflict_pairs: tuple[tuple[str, str], ...]
    forced_singleton_task_ids: tuple[str, ...]
    manifest_fingerprints: Mapping[str, str]
    parent_contexts: Mapping[str, Mapping[str, str]]


def _digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def maven_repository_fingerprint(repo_root: str | Path) -> str:
    """Fingerprint every safely discovered POM and its exact source bytes."""
    root = Path(repo_root).resolve()
    documents: dict[str, str] = {}
    for path in locate_maven_manifests(root):
        relative = path.relative_to(root).as_posix()
        try:
            documents[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot fingerprint Maven POM {relative!r}") from exc
    if not documents:
        raise ValueError("repository has no safely discoverable Maven POMs")
    return _digest(documents)


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
    manifest_name = "pom.xml" if _group_manager(group) == "maven" else "package.json"
    safe = [path for path in paths if Path(path).name == manifest_name]
    if len(safe) == 1 and len(paths) == 1:
        return safe[0]
    if not paths:
        return manifest_name
    return ""


def _group_manager(group: VulnerabilityGroup) -> str:
    managers = {
        (localized.package_manager or "").strip().lower()
        for localized in group.localized_issues
        if localized.package_manager
    }
    if len(managers) > 1:
        return "ambiguous"
    if managers:
        return next(iter(managers))
    if any(
        str(issue.ecosystem or "").casefold() == "maven"
        or str(issue.purl or "").casefold().startswith("pkg:maven/")
        for issue in group.issues
    ):
        return "maven"
    return "npm"


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
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Create missing finding tasks and synthetic direct-dependency tasks copy-on-write.

    Args:
        repo_root: Repository whose manifests and lockfiles define the portfolio scope.
        groups: Post-triage groups to prepare.
        task_queue: Existing immutable task projection.
        target_packages: Optional development package allowlist. ``None`` or an
            empty iterable preserves full-repository synthetic discovery.
        project_language: Language selected by the workspace setup boundary.

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
        language = (
            ProjectLanguage.JAVA if _group_manager(group) == "maven" else ProjectLanguage.NODEJS
        )
        prepared_queue[task_id] = build_initial_remediation_task(
            group, task_id, project_language=language
        )
        represented.add(group.group_id)
        next_index += 1

    if project_language == ProjectLanguage.PYTHON:
        # Python portfolio discovery is finding-backed and declaration-scoped.
        # Do not pass a requirements-only repository through the npm synthetic
        # dependency materializer or invent package-lock occurrence identities.
        return prepared_groups, prepared_queue, []

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


def prepare_python_portfolio_problem(
    repo_root: str | Path,
    groups: Sequence[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    settings: AppSettings,
    registry_fetcher: Any | None = None,
    runtime_fingerprint: SolverPythonRuntimeFingerprint | None = None,
    parent_requires_dist_fetcher: Any | None = None,
    retry_diagnostics_by_task: Mapping[str, Any] | None = None,
) -> _PreparedPythonPortfolioProblem:
    """Build Python declaration occurrences and bounded PyPI candidate domains.

    Direct requirements files and one-hop, category-matched Pipfile parent
    updates enter the common portfolio. A Pipfile parent is dispatchable only
    when the lockfile, declaration, ancestry, and PyPI ``Requires-Dist`` proof
    agree. This adapter never invents npm lock records or trusts a worker hint
    as a security floor. Committed no-fix tasks are retained without a version
    floor so their unresolved coverage can reach the mitigation worker.
    """
    from remediation_engine.orchestration.tools_manifest_python import (
        _declaration_is_editable,
        _matching_declarations,
    )
    from remediation_engine.tools.package_identity import normalize_python_package_name

    if registry_fetcher is not None and parent_requires_dist_fetcher is None:
        from remediation_engine.tools.pypi_registry_tools import get_pypi_release_requires_dist

        parent_requires_dist_fetcher = get_pypi_release_requires_dist
    root = Path(repo_root).resolve()
    prepared_groups = [group.model_copy(deep=True) for group in groups]
    queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    diagnostics: list[str] = []
    targets: list[SolverTarget] = []
    findings: list[SolverFindingRequirement] = []
    occurrences: list[SolverPackageOccurrence] = []
    domains: dict[str, list[SolverVersionCandidate]] = {}
    parent_contexts: dict[str, dict[str, str]] = {}
    group_by_id = {group.group_id: group for group in prepared_groups}
    active = _active_tasks(queue)
    active_groups: dict[str, list[RemediationTask]] = {}
    for task in queue.values():
        if task.status not in _TERMINAL_STATUSES:
            active_groups.setdefault(task.parent_group_id, []).append(task)
    for group_id, group_tasks in active_groups.items():
        if len(group_tasks) > 1:
            diagnostics.append(
                f"group {group_id!r} has multiple nonterminal tasks; Python portfolio is blocked"
            )

    manifest_fingerprints: dict[str, str] = {}
    for task in active:
        group = group_by_id.get(task.parent_group_id)
        if group is None or group.issue_type != IssueType.SCA:
            continue
        package_name = normalize_python_package_name(str(group.vulnerable_component or ""))
        target_name = normalize_python_package_name(
            str(task.target_package_name or group.vulnerable_component or "")
        )
        transitive = bool(
            target_name != package_name
            or task.parent_package_name
            or group.parent_package_name
            or any(item.is_direct_dependency is False for item in group.localized_issues)
            or any(item.parent_package_name for item in group.localized_issues)
            or any(item.dependency_ancestry for item in group.localized_issues)
            or group.dependency_ancestry
        )
        non_python_evidence = any(
            issue.ecosystem and issue.ecosystem.casefold() not in {"pypi", "pip", "python"}
            for issue in group.issues
        )
        if not package_name or non_python_evidence:
            diagnostics.append(
                f"task {task.task_id!r} has missing package identity or non-PyPI evidence"
            )
            continue
        raw_paths = {
            str(value).replace("\\", "/").strip()
            for value in [
                *(localized.manifest_file for localized in group.localized_issues),
                *group.file_paths,
                group.file_path,
            ]
            if value
        }
        manifest_paths: set[str] = set()
        path_error = False
        for raw_path in raw_paths:
            candidate = Path(raw_path)
            if candidate.is_absolute():
                try:
                    candidate = candidate.resolve().relative_to(root)
                except ValueError:
                    path_error = True
                    continue
            if ".." in candidate.parts:
                path_error = True
                continue
            basename = candidate.name.casefold()
            if basename in {
                "requirements.txt",
                "pyproject.toml",
                "setup.cfg",
                "pipfile",
            } or (basename.startswith("requirements") and candidate.suffix.casefold() == ".txt"):
                manifest_paths.add(candidate.as_posix())
        if path_error or len(manifest_paths) != 1:
            diagnostics.append(
                f"task {task.task_id!r} lacks one safe, unambiguous Python manifest path"
            )
            continue
        manifest_path = next(iter(manifest_paths))
        manifest_name = Path(manifest_path).name.casefold()
        is_pipfile = manifest_name == "pipfile"
        is_requirements = (
            manifest_name.startswith("requirements")
            and Path(manifest_path).suffix.casefold() == ".txt"
        )
        if is_pipfile and not transitive:
            diagnostics.append(
                f"task {task.task_id!r} uses unsupported Python manifest {manifest_path!r}; "
                "direct Pipfile targets are not supported by the shared certifier"
            )
            continue
        if not is_pipfile and (not is_requirements or transitive):
            diagnostics.append(
                f"task {task.task_id!r} uses unsupported Python target/manifest {manifest_path!r}; "
                "transitive targets require a proven Pipfile parent"
            )
            continue
        resolved_manifest = (root / manifest_path).resolve()
        try:
            resolved_manifest.relative_to(root)
            content = resolved_manifest.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            diagnostics.append(f"task {task.task_id!r} Python manifest cannot be read safely")
            continue
        manifest_fingerprints[manifest_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        pipfile_lock_path: str | None = None
        parent_context: dict[str, str] | None = None
        if is_pipfile:
            pipfile_lock_path = (Path(manifest_path).parent / "Pipfile.lock").as_posix()
            lock_file = (root / pipfile_lock_path).resolve()
            try:
                lock_file.relative_to(root)
                lock_content = lock_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError, ValueError):
                diagnostics.append(f"task {task.task_id!r} Pipfile.lock evidence is unreadable")
                continue
            manifest_fingerprints[pipfile_lock_path] = hashlib.sha256(
                lock_content.encode("utf-8")
            ).hexdigest()
            if registry_fetcher is None:
                diagnostics.append(
                    f"task {task.task_id!r} Pipfile parent proof is pending workspace validation"
                )
                continue
            try:
                parent_context = _resolve_python_pipfile_parent_context(
                    root,
                    manifest_path,
                    group,
                    task,
                    parent_requires_dist_fetcher,
                )
            except Exception as exc:  # noqa: BLE001 - uncertain ancestry fails closed
                diagnostics.append(f"task {task.task_id!r} Pipfile parent proof failed: {exc}")
                continue
            if parent_context is None:
                diagnostics.append(
                    f"task {task.task_id!r} lacks a unique, unconditional Pipfile.lock parent proof"
                )
                continue
            target_name = parent_context["parent_name"]
        elif any(line.lstrip().startswith("-") for line in content.splitlines()):
            diagnostics.append(
                f"task {task.task_id!r} requirements file uses pip directives; "
                "nested requirements, constraints, and resolver options are unsupported"
            )
            continue
        declarations = [
            item
            for item in _matching_declarations(
                manifest_path, content, target_name if is_pipfile else package_name
            )
            if _declaration_is_editable(manifest_path, item[1], item[3])
        ]
        if len(declarations) != 1:
            diagnostics.append(
                f"task {task.task_id!r} does not map to exactly one static editable "
                "Python dependency declaration"
            )
            continue
        declaration_type, requirement, _, _ = declarations[0]
        installed_version = parent_context["parent_version"] if parent_context else None
        if installed_version is None and len(list(requirement.specifier)) == 1:
            specifier = next(iter(requirement.specifier))
            if specifier.operator in {"==", "==="} and "*" not in specifier.version:
                installed_version = specifier.version
        installed_version = (
            installed_version
            or next(
                (
                    str(issue.package_version).strip()
                    for issue in group.issues
                    if issue.package_version
                ),
                None,
            )
            or next((str(value).strip() for value in group.versions if value), None)
        )
        try:
            if not installed_version:
                raise ValueError("missing installed version evidence")
            installed_version, _ = _python_numeric_solver_version(installed_version)
        except ValueError:
            diagnostics.append(f"task {task.task_id!r} has no valid installed PEP 440 version")
            continue
        no_fix_mitigation = task.no_fix_stage is not None
        floor: str | None = None
        if not no_fix_mitigation:
            fixed_versions = sorted(
                {
                    floor
                    for issue in group.issues
                    if (issue.cve_id or issue.ghsa_id)
                    and (floor := _python_scanner_fixed_version(issue))
                }
            )
            if not fixed_versions:
                diagnostics.append(
                    f"task {task.task_id!r} has no scanner-backed fixed-version floor"
                )
                continue
            try:
                normalized_floors = [
                    _python_numeric_solver_version(value) for value in fixed_versions
                ]
                floor = max(normalized_floors, key=lambda value: value[1])[0]
            except ValueError:
                diagnostics.append(f"task {task.task_id!r} has an invalid PEP 440 security floor")
                continue
        parent_candidates: list[tuple[str, tuple[int, ...]]] = []
        parent_floor: str | None = None
        if parent_context is not None:
            try:
                retry_diagnostics = (retry_diagnostics_by_task or {}).get(task.task_id)
                attempted_by_target = getattr(retry_diagnostics, "attempted_versions_by_target", {})
                attempted_for_parent: list[str] = []
                if isinstance(attempted_by_target, Mapping):
                    attempted_for_parent = [
                        str(version)
                        for attempted_target, versions in attempted_by_target.items()
                        if normalize_python_package_name(str(attempted_target)) == target_name
                        for version in (versions or ())
                    ]
                attempted = {
                    str(value).strip()
                    for value in (
                        *getattr(retry_diagnostics, "attempted_versions", ()),
                        *getattr(retry_diagnostics, "executed_versions", ()),
                        *attempted_for_parent,
                    )
                    if str(value).strip()
                }
                attempted.add(installed_version)
                attempted_keys = {
                    key for value in attempted for _, key in [_python_numeric_solver_version(value)]
                }
                for candidate in registry_fetcher(target_name, installed_version, attempted):
                    version, key = _python_numeric_solver_version(str(candidate.version))
                    if key in attempted_keys:
                        continue
                    if _python_parent_release_accepts_child_floor(
                        target_name,
                        version,
                        package_name,
                        floor,
                        parent_requires_dist_fetcher,
                    ):
                        parent_candidates.append((version, key))
            except Exception as exc:  # noqa: BLE001 - registry evidence fails closed
                diagnostics.append(
                    f"PyPI parent candidate lookup failed for {target_name!r}: {exc}"
                )
                continue
            if not parent_candidates:
                diagnostics.append(
                    f"task {task.task_id!r} has no PyPI parent release that accepts "
                    f"the child security floor {floor}"
                )
                continue
            parent_floor = min(parent_candidates, key=lambda item: item[1])[0]
            parent_context["parent_minimum_version"] = parent_floor
            task = task.model_copy(
                update={
                    "target_package_name": target_name,
                    "parent_package_name": target_name,
                    "parent_package_version": installed_version,
                    "parent_minimum_version": parent_floor,
                    "target_dependency_type": declaration_type,
                }
            )
            queue[task.task_id] = task

        occurrence_id = f"python:{manifest_path}:{target_name}"
        finding_ids = sorted(
            {
                _issue_identity(issue)
                for issue in group.issues
                if issue.cve_id or issue.ghsa_id or issue.finding_id
            }
        )
        target = SolverTarget(
            occurrence_id=occurrence_id,
            task_id=task.task_id,
            group_id=group.group_id,
            package_name=package_name,
            target_package_name=target_name,
            manifest_path=manifest_path,
            lockfile_package_key=None,
            installed_version=installed_version,
            dependency_type=declaration_type,
            strategy=("no_fix" if no_fix_mitigation else task.strategy.value),
            is_synthetic=bool(task.is_synthetic or group.is_synthetic),
            is_finding_backed=bool(finding_ids),
            eligible_for_atomic_update=(
                task.status not in _TERMINAL_STATUSES
                and task.current_attempt_id is None
                and task.strategy == RoutingStrategy.VERSION_BUMP
            ),
            finding_ids=finding_ids,
            is_terminal=task.status in _TERMINAL_STATUSES,
            terminal_status=(task.status.value if task.status in _TERMINAL_STATUSES else None),
            has_open_attempt=task.current_attempt_id is not None,
        )
        targets.append(target)
        occurrences.append(
            SolverPackageOccurrence(
                occurrence_id=occurrence_id,
                manifest_path=manifest_path,
                package_name=target_name,
                lockfile_package_key=None,
                installed_version=installed_version,
                dependency_type=declaration_type,
                is_direct=True,
            )
        )
        vulnerable_occurrence_id = occurrence_id
        if parent_context is not None:
            vulnerable_occurrence_id = f"python-lock:{parent_context['lock_path']}:{parent_context['lock_category']}:{package_name}"
            occurrences.append(
                SolverPackageOccurrence(
                    occurrence_id=vulnerable_occurrence_id,
                    manifest_path=parent_context["lock_path"],
                    package_name=package_name,
                    lockfile_package_key=None,
                    installed_version=parent_context["child_version"],
                    dependency_type=declaration_type,
                    is_direct=False,
                    ancestry=[target_name, package_name],
                )
            )
            parent_contexts[occurrence_id] = dict(parent_context)
        supported_issues = [
            issue for issue in group.issues if issue.cve_id or issue.ghsa_id or issue.finding_id
        ]
        for issue in sorted(supported_issues, key=_issue_identity):
            if not (issue.cve_id or issue.ghsa_id or issue.finding_id):
                continue
            findings.append(
                SolverFindingRequirement(
                    finding_id=_issue_identity(issue),
                    vulnerable_occurrence_id=vulnerable_occurrence_id,
                    cve_id=issue.cve_id,
                    ghsa_id=issue.ghsa_id,
                    severity=issue.severity.value,
                    vulnerable_package=package_name,
                    target_occurrence_id=occurrence_id,
                    fixed_version=floor,
                    direct_parent_name=(target_name if parent_context is not None else None),
                    direct_parent_minimum_version=parent_floor,
                    is_transitive=parent_context is not None,
                )
            )
        if target.eligible_for_atomic_update and registry_fetcher is not None:
            try:
                converted_by_version: dict[str, SolverVersionCandidate] = {}
                if parent_context is not None:
                    parent_floor_key = _python_numeric_solver_version(parent_floor or "")[1]
                    for version, key in parent_candidates:
                        converted_by_version[version] = SolverVersionCandidate(
                            version=version,
                            semver_key=key,
                            source="pypi-parent-compatibility",
                            meets_security_floor=key >= parent_floor_key,
                        )
                else:
                    raw_candidates = registry_fetcher(package_name, floor, set())
                    floor_key = _python_numeric_solver_version(floor)[1]
                    for candidate in raw_candidates:
                        version, key = _python_numeric_solver_version(str(candidate.version))
                        converted_by_version[version] = SolverVersionCandidate(
                            version=version,
                            semver_key=key,
                            source="pypi",
                            meets_security_floor=key >= floor_key,
                        )
                domains[occurrence_id] = sorted(
                    converted_by_version.values(), key=lambda item: (item.semver_key, item.version)
                )
                if not converted_by_version:
                    diagnostics.append(f"task {task.task_id!r} has no stable PyPI candidate")
            except Exception as exc:  # noqa: BLE001 - registry evidence is fail-closed
                diagnostics.append(f"PyPI candidate lookup failed for {package_name!r}: {exc}")

    # Pip resolves the complete requirements set in one environment. Couple
    # all selected direct mutations into one atomic solver batch so QA/certifier
    # evidence always describes the combined declaration change.
    python_edges = [
        SolverEdge(
            source_occurrence_id=left.occurrence_id,
            target_occurrence_id=right.occurrence_id,
            source_task_id=left.task_id,
            target_task_id=right.task_id,
            edge_kind="workspace",
            is_peer_coupling=True,
        )
        for index, left in enumerate(sorted(targets, key=lambda item: item.occurrence_id))
        for right in sorted(targets, key=lambda item: item.occurrence_id)[index + 1 :]
    ]
    subgraph = SolverSubgraph(
        targets=targets,
        occurrences=occurrences,
        findings=findings,
        edges=python_edges,
        valid=bool(targets and findings) and not diagnostics,
        diagnostics=sorted(set(diagnostics)),
    )
    host_fingerprint = _digest(manifest_fingerprints)
    candidate_catalog_complete = (
        bool(targets and findings)
        and all(
            not target.eligible_for_atomic_update
            or (target.occurrence_id in domains and bool(domains[target.occurrence_id]))
            for target in targets
        )
        and not diagnostics
    )
    catalog_digest = _digest(
        {
            "ecosystem": "pypi",
            "manifest_fingerprints": manifest_fingerprints,
            "domains": {
                key: [item.model_dump(mode="json") for item in values]
                for key, values in sorted(domains.items())
            },
            "complete": candidate_catalog_complete,
            "diagnostics": sorted(set(diagnostics)),
        }
    )
    return _PreparedPythonPortfolioProblem(
        repo_root=root,
        host_repository_fingerprint=host_fingerprint,
        workspace_repository_fingerprint=host_fingerprint,
        groups=prepared_groups,
        task_queue=queue,
        targets=targets,
        findings=findings,
        subgraph=subgraph,
        packuments={},
        candidate_domains=domains,
        runtime_fingerprint=runtime_fingerprint,
        candidate_catalog_complete=candidate_catalog_complete,
        candidate_catalog_digest=catalog_digest,
        settings=settings,
        diagnostics=sorted(set(diagnostics)),
        target_packages=None,
        peer_conflict_pairs=(),
        forced_singleton_task_ids=(),
        manifest_fingerprints=dict(sorted(manifest_fingerprints.items())),
        parent_contexts={key: dict(value) for key, value in sorted(parent_contexts.items())},
    )


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
    *,
    include_current_versions: bool = False,
) -> list[NpmLockfilePackage]:
    """Resolve matching physical lockfile occurrences for one finding.

    QA-passed tasks may point at a graph already upgraded beyond the scanner's
    original package version, so their current physical occurrences remain
    evidence even when that old version is no longer present.
    """
    issue_version = str(getattr(issue, "package_version", "") or "").strip().lstrip("=vV")
    raw_versions = (
        []
        if include_current_versions
        else [issue_version]
        if issue_version
        else list(group.versions or [])
    )
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
    *,
    include_current_versions: bool = False,
) -> list[str]:
    """Return all physical finding occurrences without guessing a copy."""
    packages = _lockfile_packages_for_finding(
        snapshot,
        manifest_path,
        package_name,
        issue,
        group,
        include_current_versions=include_current_versions,
    )
    if packages:
        return sorted(
            {_occurrence_id_for_lockfile_package(snapshot, package) for package in packages}
        )

    issue_version = str(getattr(issue, "package_version", "") or "").strip().lstrip("=vV")
    raw_versions = (
        []
        if include_current_versions
        else [issue_version]
        if issue_version
        else list(group.versions or [])
    )
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


def _representative_lockfile_package(
    snapshot: NpmGraphSnapshot,
    manifest_path: str,
    package_name: str,
    group: VulnerabilityGroup,
    *,
    parent_occurrence_id: str | None = None,
    include_current_versions: bool = False,
) -> NpmLockfilePackage | None:
    """Choose a stable physical occurrence for one finding group.

    Prefer the package reachable from its directly declared parent when that
    occurrence is also supported by the finding's version/ancestry evidence.
    Otherwise use the group's most specific ancestry, then stable lockfile-key
    ordering. With ``include_current_versions`` enabled, allow a QA-passed task's
    already-upgraded physical occurrence even when the scanner's old version is
    no longer present. This identity is evidence-only for code-workaround tasks.
    """
    by_identity: dict[tuple[str, str | None], NpmLockfilePackage] = {}
    issues = list(group.issues) or [None]
    for issue in issues:
        for package in _lockfile_packages_for_finding(
            snapshot,
            manifest_path,
            package_name,
            issue,
            group,
            include_current_versions=include_current_versions,
        ):
            by_identity.setdefault((package.package_key, package.version), package)
    candidates = list(by_identity.values())
    if not candidates:
        return None
    if parent_occurrence_id:
        parent_package = resolve_lockfile_dependency_package(
            snapshot,
            parent_occurrence_id,
            package_name,
        )
        if parent_package is not None:
            parent_identity = (parent_package.package_key, parent_package.version)
            for package in candidates:
                if (package.package_key, package.version) == parent_identity:
                    return package
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
    *,
    repo_root: Path | None = None,
    additional_occurrences: list[SolverPackageOccurrence] | None = None,
) -> tuple[list[SolverTarget], list[SolverFindingRequirement], list[str]]:
    if any(
        _group_manager(group) == "maven" for group in groups if group.issue_type == IssueType.SCA
    ):
        return _build_maven_targets_and_findings(
            groups,
            task_queue,
            repo_root=repo_root or Path("."),
            additional_occurrences=additional_occurrences,
        )
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
        parent_name = (task.parent_package_name or group.parent_package_name or "").strip()
        parent_record = records.get((manifest_path, parent_name)) if parent_name else None
        transitive_workaround = task.strategy == RoutingStrategy.CODE_WORKAROUND and bool(
            parent_name
            or group.parent_contexts
            or group.dependency_ancestry
            or any(localized.is_direct_dependency is False for localized in group.localized_issues)
        )
        nested_package = (
            _representative_lockfile_package(
                snapshot,
                manifest_path,
                target_name,
                group,
                parent_occurrence_id=parent_record.occurrence_id if parent_record else None,
                include_current_versions=task.status == TaskStatus.QA_PASSED,
            )
            if record is None
            and target_name == package_name
            and (override_action or transitive_workaround)
            else None
        )
        target_mapped = (
            record is not None
            or override_action
            or (transitive_workaround and nested_package is not None)
        )
        if not package_name or not target_name or not manifest_path or manager not in {"", "npm"}:
            diagnostics.append(f"task {task.task_id!r} has an unsupported or ambiguous npm target")
            target_mapped = False
        if not target_mapped:
            diagnostics.append(
                f"task {task.task_id!r} does not map to a direct declaration or supported override"
            )
        if record is None and not override_action and nested_package is None:
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
                    or (
                        task.strategy != RoutingStrategy.CODE_WORKAROUND
                        and (task.strategy == RoutingStrategy.VERSION_BUMP or has_version_evidence)
                    )
                )
            ),
            workspace_id=_workspace_id(snapshot, manifest_path),
            dependency_ancestry=dependency_ancestry,
            finding_ids=sorted(set(finding_ids)),
            is_terminal=task.status in _TERMINAL_STATUSES,
            terminal_status=(task.status.value if task.status in _TERMINAL_STATUSES else None),
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
            fixed = issue.fixed_version or _group_fixed_version(group, task)
            workaround_plan_ids = _workaround_plan_ids(group, issue)
            vulnerable_occurrence_ids = _finding_occurrences(
                snapshot,
                manifest_path,
                package_name,
                issue,
                group,
                include_current_versions=task.status == TaskStatus.QA_PASSED,
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
                        direct_parent_name=(
                            (group.parent_package_name or task.parent_package_name)
                            if transitive_parent_target and task.parent_minimum_version
                            else None
                        ),
                        direct_parent_minimum_version=(
                            task.parent_minimum_version if transitive_parent_target else None
                        ),
                        strategy_stage=task.strategy_stage.value,
                        workaround_available=bool(workaround_plan_ids),
                        workaround_plan_ids=workaround_plan_ids,
                        is_transitive=bool(
                            group.parent_package_name or task.parent_package_name or nested_package
                        ),
                    )
                )
    return targets, findings, sorted(set(diagnostics))


def _build_maven_targets_and_findings(
    groups: Sequence[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    repo_root: Path,
    additional_occurrences: list[SolverPackageOccurrence] | None,
) -> tuple[list[SolverTarget], list[SolverFindingRequirement], list[str]]:
    """Build exact-GAV solver targets and finding evidence from localized POMs."""
    root = Path(repo_root).resolve()
    groups_by_id = {group.group_id: group for group in groups}
    targets: list[SolverTarget] = []
    findings: list[SolverFindingRequirement] = []
    diagnostics: list[str] = []
    for task in _active_tasks(task_queue):
        group = groups_by_id.get(task.parent_group_id)
        if group is None or group.issue_type != IssueType.SCA:
            continue
        if _group_manager(group) != "maven":
            diagnostics.append(
                f"task {task.task_id!r} uses a non-Maven target in a Maven portfolio"
            )
            continue
        coordinate = (task.target_package_name or group.vulnerable_component or "").strip()
        parts = coordinate.split(":")
        manifest_path = _manifest_path(group, root)
        if len(parts) != 2 or any(not part or part.strip() != part for part in parts):
            diagnostics.append(f"task {task.task_id!r} has no canonical Maven group:artifact")
            continue
        absolute_manifest = (root / manifest_path).resolve()
        try:
            absolute_manifest.relative_to(root)
            if absolute_manifest.name != "pom.xml" or absolute_manifest.is_symlink():
                raise ValueError("localized Maven target is not a contained regular POM")
            pom = parse_pom_xml(absolute_manifest)
            entry = find_dependency_in_pom(parts[0], parts[1], pom)
        except (OSError, ValueError) as exc:
            entry = None
            diagnostics.append(f"task {task.task_id!r} Maven target cannot be parsed: {exc}")
        localized_types = {
            str(item.declaration_type or "").strip()
            for item in group.localized_issues
            if item.package_manager == "maven" and item.declaration_type
        }
        dependency_type = (
            task.target_dependency_type
            if task.target_dependency_type in {"dependencies", "dependencyManagement"}
            else next(iter(localized_types))
            if len(localized_types) == 1
            else "dependencyManagement"
        )
        if dependency_type not in {"dependencies", "dependencyManagement"}:
            diagnostics.append(f"task {task.task_id!r} has an unsupported Maven declaration type")
            continue
        installed_version = (
            (entry.declared_version if entry is not None else None)
            or (group.versions[0] if group.versions else None)
            or (group.issues[0].package_version if group.issues else None)
            or task.parent_package_version
            or "unknown"
        )
        occurrence_id = f"{manifest_path}::maven::{coordinate}"
        finding_ids = sorted(
            {
                _issue_identity(issue)
                for issue in group.issues
                if issue.cve_id or issue.ghsa_id or issue.finding_id
            }
        )
        has_floor = any(issue.fixed_version for issue in group.issues) or bool(
            group.fix_plan and group.fix_plan.fixed_version
        )
        no_fix = task.no_fix_stage is not None or (
            group.fix_plan is not None and group.fix_plan.status == FixPlanStatus.NO_FIX
        )
        target = SolverTarget(
            occurrence_id=occurrence_id,
            task_id=task.task_id,
            group_id=group.group_id,
            package_name=(group.vulnerable_component or coordinate),
            target_package_name=coordinate,
            manifest_path=manifest_path,
            lockfile_package_key=f"{coordinate}@{installed_version}",
            installed_version=str(installed_version),
            version_scheme="maven",
            dependency_type=dependency_type,
            strategy="no_fix" if no_fix else task.strategy.value,
            is_synthetic=bool(task.is_synthetic or group.is_synthetic),
            is_finding_backed=bool(finding_ids),
            eligible_for_atomic_update=(
                absolute_manifest.is_file()
                and task.status not in _TERMINAL_STATUSES
                and task.current_attempt_id is None
                and (no_fix or (task.strategy == RoutingStrategy.VERSION_BUMP and has_floor))
            ),
            finding_ids=finding_ids,
            is_terminal=task.status in _TERMINAL_STATUSES,
            terminal_status=(task.status.value if task.status in _TERMINAL_STATUSES else None),
            has_open_attempt=task.current_attempt_id is not None,
        )
        targets.append(target)
        if additional_occurrences is not None:
            additional_occurrences.append(
                SolverPackageOccurrence(
                    occurrence_id=occurrence_id,
                    manifest_path=manifest_path,
                    package_name=coordinate,
                    lockfile_package_key=target.lockfile_package_key,
                    installed_version=str(installed_version),
                    dependency_type=dependency_type,
                    is_direct=entry is not None,
                )
            )
        for issue in sorted(group.issues, key=_issue_identity):
            if not (issue.cve_id or issue.ghsa_id or issue.finding_id):
                continue
            floor = issue.fixed_version or (
                group.fix_plan.fixed_version if group.fix_plan is not None else None
            )
            vulnerable_version = str(issue.package_version or installed_version)
            vulnerable_occurrence_id = f"{manifest_path}::maven::{coordinate}@{vulnerable_version}"
            findings.append(
                SolverFindingRequirement(
                    finding_id=_issue_identity(issue),
                    vulnerable_occurrence_id=vulnerable_occurrence_id,
                    cve_id=issue.cve_id,
                    ghsa_id=issue.ghsa_id,
                    severity=(
                        issue.severity.value
                        if isinstance(issue.severity, Severity)
                        else str(issue.severity)
                    ),
                    vulnerable_package=coordinate,
                    target_occurrence_id=occurrence_id,
                    fixed_version=floor,
                    strategy_stage=task.strategy_stage.value,
                    workaround_available=bool(_workaround_plan_ids(group, issue)),
                    workaround_plan_ids=_workaround_plan_ids(group, issue),
                    is_transitive=not bool(entry),
                )
            )
            if additional_occurrences is not None:
                additional_occurrences.append(
                    SolverPackageOccurrence(
                        occurrence_id=vulnerable_occurrence_id,
                        manifest_path=manifest_path,
                        package_name=coordinate,
                        lockfile_package_key=f"{coordinate}@{vulnerable_version}",
                        installed_version=vulnerable_version,
                        dependency_type=dependency_type,
                        is_direct=not bool(entry),
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
        if target.version_scheme == "maven":
            # Maven candidate floors come from the exact GAV finding and are
            # certified by native dependency resolution. This helper only
            # proves npm parent/child compatibility from packument metadata.
            continue
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
                        findings[index] = finding.model_copy(
                            update={
                                "direct_parent_name": parent_name,
                                "direct_parent_minimum_version": parent_floor,
                            }
                        )
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


def _maven_candidate(
    version: str,
    *,
    source: str,
    floor: str | None,
    attempted: bool = False,
) -> SolverVersionCandidate | None:
    """Build a Maven candidate using native ComparableVersion semantics."""
    normalized = str(version or "").strip()
    if not normalized:
        return None
    if source != "current" and not is_stable_maven_version(normalized):
        return None
    try:
        meets_floor = not floor or compare_maven_versions(normalized, floor) >= 0
    except ValueError:
        if source != "current":
            return None
        meets_floor = False
    return SolverVersionCandidate(
        version=normalized,
        version_scheme="maven",
        source=source,
        meets_security_floor=meets_floor,
        attempted=attempted,
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
            field: metadata[field] for field in _EVIDENCE_METADATA_FIELDS if field in metadata
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


def _requires_version_catalog(target: SolverTarget) -> bool:
    """Return whether an eligible target needs a release catalog."""
    if not target.eligible_for_atomic_update:
        return False
    strategy = target.strategy.strip().lower().replace("-", "_")
    return not (target.version_scheme == "maven" and strategy == "no_fix")


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
    maven_registry_fetcher: Any | None = None,
) -> tuple[dict[str, list[SolverVersionCandidate]], bool, str]:
    """Prepare candidate-specific registry domains with goal-directed pruning."""
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
            or (
                compare_maven_versions(finding.fixed_version, current) > 0
                if any(
                    target.occurrence_id == finding.target_occurrence_id
                    and target.version_scheme == "maven"
                    for target in targets
                )
                else (_semver_key(finding.fixed_version) or (0, 0, 0))
                > (_semver_key(current) or (0, 0, 0))
            )
        ):
            floors_by_target[finding.target_occurrence_id] = finding.fixed_version
    cache = RegistryPackumentCache(settings.solver_cache_dir) if settings.solver_cache_dir else None
    compatible_versions = transitive_compatible_versions or {}
    domains: dict[str, list[SolverVersionCandidate]] = {}
    observed_packuments: dict[str, Mapping[str, Any]] = dict(packuments or {})
    complete = True
    completeness_diagnostics: list[str] = []
    maven_catalogs: dict[str, list[str]] = {}
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
        if parent_target and task is not None:
            parent_floors = [
                finding.direct_parent_minimum_version
                for finding in findings
                if finding.target_occurrence_id == target.occurrence_id
                and finding.direct_parent_minimum_version
            ]
            floor = (
                max(parent_floors, key=lambda value: _semver_key(value) or (0, 0, 0))
                if parent_floors
                else task.parent_minimum_version
            )
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
        no_fix_target = target.strategy.strip().lower().replace("-", "_") == "no_fix"

        if target.version_scheme == "maven":
            installed_candidate = _maven_candidate(
                target.installed_version, source="current", floor=floor
            )
            if installed_candidate is not None:
                values.append(installed_candidate)
            if not no_fix_target:
                selected_candidate = _maven_candidate(
                    task.selected_version if task and task.selected_version else "",
                    source="selected",
                    floor=floor,
                )
                if selected_candidate is not None:
                    values.append(selected_candidate)
            if _requires_version_catalog(target):
                try:
                    if not floor:
                        raise ValueError("Maven target has no validated security floor")
                    fetcher = maven_registry_fetcher or fetch_maven_registry_candidates
                    catalog = fetcher(target.target_package_name, floor, set())
                    candidate_versions: list[str] = []
                    for item in catalog:
                        version = str(getattr(item, "version", "") or "").strip()
                        if not version:
                            continue
                        candidate_versions.append(version)
                        values.append(
                            SolverVersionCandidate(
                                version=version,
                                version_scheme="maven",
                                source="maven_registry",
                                meets_security_floor=(compare_maven_versions(version, floor) >= 0),
                                attempted=bool(getattr(item, "already_attempted", False)),
                            )
                        )
                    maven_catalogs[target.target_package_name] = sorted(
                        set(candidate_versions),
                        key=cmp_to_key(compare_maven_versions),
                    )
                    if not candidate_versions:
                        raise ValueError("Maven registry returned no stable candidates")
                except Exception as exc:  # noqa: BLE001 - candidate evidence fails closed
                    complete = False
                    completeness_diagnostics.append(
                        f"required Maven candidate catalog unavailable for "
                        f"{target.target_package_name!r}: {exc}"
                    )
            if floor and not no_fix_target:
                floor_candidate = _maven_candidate(floor, source="security_floor", floor=floor)
                if floor_candidate is not None:
                    values.append(floor_candidate)
            if not no_fix_target:
                for version in task.allowed_target_versions if task else ():
                    approved_candidate = _maven_candidate(
                        version, source="approved_alternative", floor=floor
                    )
                    if approved_candidate is not None:
                        values.append(approved_candidate)
            dedup_maven: dict[str, SolverVersionCandidate] = {}
            for candidate in values:
                previous = dedup_maven.get(candidate.version)
                if previous is None or candidate.source == "maven_registry":
                    dedup_maven[candidate.version] = candidate
            ordered_maven = _normalize_solver_candidate_domain(
                target,
                {target.occurrence_id: list(dedup_maven.values())},
                max_candidates=limit,
                security_floor=floor,
                diagnostics=diagnostics,
            )
            domains[target.occurrence_id] = ordered_maven
            if not ordered_maven and target.eligible_for_atomic_update:
                completeness_diagnostics.append(
                    f"empty Maven candidate domain for {target.occurrence_id!r}"
                )
            continue

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
                        add_candidate(candidate)

        dedup: dict[str, SolverVersionCandidate] = {}
        for value in sorted(values, key=lambda item: (item.semver_key, item.version, item.source)):
            existing = dedup.get(value.version)
            if existing is None or (existing.source != "registry" and value.source == "registry"):
                dedup[value.version] = value
        ordered = _normalize_solver_candidate_domain(
            target,
            {target.occurrence_id: list(dedup.values())},
            max_candidates=limit,
            security_floor=floor,
            diagnostics=diagnostics,
        )
        if len(ordered) > limit:
            complete = False
            completeness_diagnostics.append(
                f"candidate resource guard exceeded for {target.occurrence_id!r}; "
                "the full catalog was retained but cannot be certified within the configured limit"
            )
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
            "maven_candidate_catalogs": maven_catalogs,
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


def _attach_maven_dependency_paths(
    subgraph: SolverSubgraph,
    targets: Sequence[SolverTarget],
    paths_by_manifest: Mapping[str, Sequence[Sequence[str]]] | None,
) -> SolverSubgraph:
    """Add hard edges for vulnerable targets on the same resolved Maven path.

    Args:
        subgraph: Solver graph extracted from the package-manager snapshot.
        targets: Exact Maven mutation targets represented in the graph.
        paths_by_manifest: Resolved group:artifact paths keyed by source POM.

    Returns:
        The graph with dependency-ancestry edges between targets on a shared
        resolved path.
    """
    if not paths_by_manifest:
        return subgraph
    targets_by_manifest_and_coordinate: dict[tuple[str, str], SolverTarget] = {}
    for target in targets:
        targets_by_manifest_and_coordinate.setdefault(
            (target.manifest_path, target.target_package_name), target
        )
    existing = {
        (edge.source_occurrence_id, edge.target_occurrence_id, edge.edge_kind)
        for edge in subgraph.edges
    }
    added: dict[tuple[str, str, str], SolverEdge] = {}
    for manifest_path, paths in sorted(paths_by_manifest.items()):
        for path in paths:
            target_path: list[SolverTarget] = []
            seen_occurrences: set[str] = set()
            for coordinate in path:
                target = targets_by_manifest_and_coordinate.get((manifest_path, coordinate))
                if target is not None and target.occurrence_id not in seen_occurrences:
                    target_path.append(target)
                    seen_occurrences.add(target.occurrence_id)
            for left, right in zip(target_path, target_path[1:], strict=False):
                key = (left.occurrence_id, right.occurrence_id, "dependency_ancestry")
                if key in existing or key in added:
                    continue
                added[key] = SolverEdge(
                    source_occurrence_id=left.occurrence_id,
                    target_occurrence_id=right.occurrence_id,
                    source_task_id=left.task_id,
                    target_task_id=right.task_id,
                    edge_kind="dependency_ancestry",
                )
    if not added:
        return subgraph
    return subgraph.model_copy(
        update={
            "edges": [
                *subgraph.edges,
                *(added[key] for key in sorted(added)),
            ]
        }
    )


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
    workspace_repository_fingerprint: str | None = None,
    maven_dependency_paths_by_manifest: Mapping[str, Sequence[Sequence[str]]] | None = None,
    registry_fetcher: PackumentFetcher | None = None,
    maven_registry_fetcher: Any | None = None,
    settings: AppSettings,
    runtime_fingerprint: SolverRuntimeFingerprint | None = None,
) -> _PreparedPortfolioProblem:
    """Prepare one immutable set of npm, registry, and solver inputs."""
    root = Path(repo_root).resolve()
    host_snapshot: NpmGraphSnapshot | None = None
    if npm_snapshot is None or repository_fingerprint is None:
        host_snapshot = load_npm_graph_snapshot(root)
    snapshot = npm_snapshot or host_snapshot or NpmGraphSnapshot()
    is_maven_portfolio = any(
        _group_manager(group) == "maven" for group in groups if group.issue_type == IssueType.SCA
    )
    maven_fingerprint = maven_repository_fingerprint(root) if is_maven_portfolio else None
    host_fingerprint = (
        repository_fingerprint
        or maven_fingerprint
        or (host_snapshot.repository_fingerprint if host_snapshot is not None else None)
        or snapshot.repository_fingerprint
    )
    if is_maven_portfolio:
        workspace_fingerprint = workspace_repository_fingerprint or host_fingerprint
        if snapshot.repository_fingerprint != workspace_fingerprint:
            snapshot = replace(snapshot, repository_fingerprint=workspace_fingerprint)
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
    additional_occurrences: list[SolverPackageOccurrence] = []
    targets, findings, diagnostics = _build_targets_and_findings(
        snapshot,
        group_list,
        queue,
        repo_root=root,
        additional_occurrences=additional_occurrences,
    )
    if not targets:
        raise ValueError("Cannot build a portfolio plan without active SCA package tasks.")

    npm_targets = [target for target in targets if target.version_scheme == "semver"]
    required_names = _required_candidate_package_names(npm_targets, findings)
    packuments: dict[str, Mapping[str, Any]] = {}
    packument_digest = _digest({"required_packages": required_names, "fresh_snapshot": False})
    if npm_targets and registry_fetcher is not None:
        (
            packuments,
            _,
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
        packuments=packuments if npm_targets and registry_fetcher is not None else None,
    )
    subgraph = extract_solver_subgraph(
        snapshot,
        targets,
        findings,
        peer_conflict_pairs=explicit_pairs,
        forced_singleton_task_ids=forced,
        additional_occurrences=additional_occurrences,
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
        packuments=packuments if npm_targets and registry_fetcher is not None else None,
        maven_registry_fetcher=maven_registry_fetcher,
    )
    if npm_targets:
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
    else:
        evidence_domains = []
        evidence_domain_digest = _digest({"manager": "maven", "domains": domains})
        evidence_diagnostics = []
    diagnostics.extend(evidence_diagnostics)
    subgraph = expand_candidate_relations(
        subgraph,
        domains,
        npm_snapshot=snapshot,
        evidence_domains=evidence_domains,
    )
    subgraph = _attach_maven_dependency_paths(subgraph, targets, maven_dependency_paths_by_manifest)
    diagnostics.extend(subgraph.diagnostics)
    mutation_catalog_available = all(
        bool(target.target_package_name)
        and (
            any(
                candidate.source == "maven_registry"
                for candidate in domains.get(target.occurrence_id, ())
            )
            if target.version_scheme == "maven"
            else target.target_package_name in packuments
        )
        for target in targets
        if _requires_version_catalog(target)
    )
    # Completeness covers every target that may change package versions.
    # NO_FIX Maven targets remain package-resolution evidence but use a
    # separate removal/workaround action instead of a replacement release.
    # Missing third-party evidence may relax constraints, not release catalogs.
    candidate_catalog_complete = (
        (registry_fetcher is not None or any(t.version_scheme == "maven" for t in targets))
        and mutation_catalog_available
        and domain_complete
    )
    candidate_catalog_digest = _digest(
        {
            "fresh_packument_digest": packument_digest,
            "maven_candidate_domain_digest": (
                domain_digest if any(t.version_scheme == "maven" for t in targets) else None
            ),
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


def _scope_coupling_enabled(prepared: _PreparedPortfolioProblem) -> bool:
    """Enable Angular lockstep grouping only for a pure Angular namespace scope.

    Scoped plans normally avoid grouping packages by namespace alone. Angular
    peer-closure materialization is the exception: only requested or required
    Angular mutation targets are present, and those targets must be dispatched
    together to preserve the framework's version lockstep.
    """
    if not prepared.target_packages:
        return True
    if not any(package.startswith("@angular/") for package in prepared.target_packages):
        return False
    namespaces = {
        target.target_package_name.split("/", 1)[0]
        for target in prepared.subgraph.targets
        if target.target_package_name.startswith("@") and "/" in target.target_package_name
    }
    return namespaces == {"@angular"}


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
        scope_coupling=_scope_coupling_enabled(prepared),
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
    terminal_task_ids = {target.task_id for target in subgraph.targets if target.is_terminal}
    planned_task_revisions = {
        task_id: revision
        + (1 if task_id in decision_by_task or task_id in terminal_task_ids else 0)
        for task_id, revision in current_task_revisions.items()
    }
    task_strategies: dict[str, RoutingStrategy] = {}
    for task_id in task_order:
        task = queue.get(task_id)
        decision = decision_by_task.get(task_id)
        if task is None:
            continue
        if task_id in terminal_task_ids:
            task_strategies[task_id] = task.strategy
        elif decision is not None and str(decision.selected_strategy).lower().replace("-", "_") in {
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
    maven_registry_fetcher: Any | None = None,
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
        maven_registry_fetcher=maven_registry_fetcher,
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


def build_python_portfolio_plan(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    settings: AppSettings | None = None,
    registry_fetcher: Any | None = None,
    parent_requires_dist_fetcher: Any | None = None,
    runtime_fingerprint: SolverPythonRuntimeFingerprint | None = None,
    retry_diagnostics_by_task: Mapping[str, Any] | None = None,
    portfolio_iteration: int = 0,
) -> tuple[Any, SolverRemediationPlan, Any]:
    """Prepare PyPI evidence and solve through the common occurrence portfolio.

    Returns the prepared evidence, the shared CP-SAT result, and the same
    ``PortfolioPlan`` contract used by npm.  Resolver certification remains a
    separate outer Supervisor gate and is added by ``portfolio_certifier``.
    """
    resolved_settings = settings or AppSettings()
    prepared = prepare_python_portfolio_problem(
        repo_root,
        list(groups),
        task_queue,
        settings=resolved_settings,
        registry_fetcher=registry_fetcher,
        parent_requires_dist_fetcher=parent_requires_dist_fetcher,
        runtime_fingerprint=runtime_fingerprint,
        retry_diagnostics_by_task=retry_diagnostics_by_task,
    )
    solver_plan = solve_portfolio(
        prepared.subgraph,
        prepared.candidate_domains,
        settings=resolved_settings,
        candidate_catalog_complete=prepared.candidate_catalog_complete,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
    )
    plan = _project_prepared_portfolio_plan(
        prepared,
        solver_plan,
        portfolio_iteration=portfolio_iteration,
    )
    return prepared, plan.solver_plan, plan


def _expected_target_identity(
    group: VulnerabilityGroup, task: RemediationTask
) -> tuple[str, str, str]:
    """Return ``(group_id, manifest_path, target_package_name)`` for a task."""
    manifest_path = _manifest_path(group, Path(".")) or "package.json"
    package_name = (task.target_package_name or group.vulnerable_component or "").strip()
    if not package_name:
        raise ValueError(f"task {task.task_id!r} has no target package identity")
    if not _manifest_path(group, Path(".")):
        python_manifest = _python_requirements_manifest_path(
            group
        ) or _python_pipfile_manifest_path(group)
        python_evidence = any(
            str(issue.ecosystem or "").strip().casefold() in {"pypi", "pip", "python"}
            for issue in group.issues
        ) or any(
            str(localized.package_manager or "").strip().casefold() in {"pip", "pipenv"}
            for localized in group.localized_issues
        )
        if python_manifest and python_evidence:
            from remediation_engine.tools.package_identity import normalize_python_package_name

            package_name = normalize_python_package_name(package_name)
            manifest_path = python_manifest
    return task.parent_group_id, manifest_path, package_name


def _python_requirements_manifest_path(group: VulnerabilityGroup) -> str | None:
    """Return a unique safe requirements-file path for an explicitly Python group."""
    raw_paths = [
        *(localized.manifest_file for localized in group.localized_issues),
        *group.file_paths,
        group.file_path,
    ]
    paths: set[str] = set()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        text = str(raw_path).replace("\\", "/").strip()
        candidate = Path(text)
        if (
            not text
            or "\x00" in text
            or candidate.is_absolute()
            or ".." in candidate.parts
            or text.startswith(("~/", "//"))
        ):
            return None
        normalized = candidate.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        paths.add(normalized)
    if len(paths) != 1:
        return None
    manifest = next(iter(paths))
    name = Path(manifest).name.casefold()
    if name.startswith("requirements") and Path(manifest).suffix.casefold() == ".txt":
        return manifest
    return None


def _python_pipfile_manifest_path(group: VulnerabilityGroup) -> str | None:
    """Return one safe Pipfile path for a Python finding group."""
    raw_paths = [
        *(localized.manifest_file for localized in group.localized_issues),
        *group.file_paths,
        group.file_path,
    ]
    paths: set[str] = set()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        text = str(raw_path).replace("\\", "/").strip()
        candidate = Path(text)
        if (
            not text
            or "\x00" in text
            or candidate.is_absolute()
            or ".." in candidate.parts
            or text.startswith(("~/", "//"))
        ):
            return None
        normalized = candidate.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if candidate.name.casefold() == "pipfile":
            paths.add(normalized)
    return next(iter(paths)) if len(paths) == 1 else None


def python_manifest_repository_fingerprint(
    repo_root: str | Path, manifest_paths: Iterable[str]
) -> str | None:
    """Recompute the Python portfolio fingerprint from its committed manifests.

    The digest covers the same declaration files used to prepare the Python
    portfolio, including the adjacent Pipfile.lock for certified Pipfile
    parent targets. Unsafe, missing, or unreadable paths fail closed by
    returning ``None``.
    """
    root = Path(repo_root).resolve()
    normalized_paths: set[str] = set()
    for raw_path in manifest_paths:
        text = str(raw_path or "").replace("\\", "/").strip()
        candidate = Path(text)
        if (
            not text
            or "\x00" in text
            or candidate.is_absolute()
            or ".." in candidate.parts
            or text.startswith(("~/", "//"))
        ):
            return None
        normalized = candidate.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        name = candidate.name.casefold()
        is_requirements = name.startswith("requirements") and candidate.suffix.casefold() == ".txt"
        if not (is_requirements or name in {"pipfile", "pyproject.toml", "setup.cfg"}):
            return None
        normalized_paths.add(normalized)
        if name == "pipfile":
            normalized_paths.add((candidate.parent / "Pipfile.lock").as_posix())

    if not normalized_paths:
        return None
    content_digests: dict[str, str] = {}
    for relative_path in sorted(normalized_paths):
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            return None
        content_digests[relative_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return _digest(content_digests)


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


def _version_bump_instruction(
    task: RemediationTask,
    group: VulnerabilityGroup,
    package_name: str,
    target_version: str,
    manifest_path: str,
    dependency_type: str | None,
) -> str:
    """Build an instruction that names the exact solver-approved package version.

    Args:
        task: Finding-backed task whose package target was selected.
        group: Vulnerability group providing any transitive-child context.
        package_name: Exact package name selected for mutation.
        target_version: Version selected by the certified solver plan.
        manifest_path: Manifest containing the dependency declaration.
        dependency_type: Dependency section selected for the mutation.

    Returns:
        A deterministic version-bump instruction with no stale fix-plan version.
    """
    declaration = (
        dependency_type
        or task.target_dependency_type
        or group.parent_declaration_type
        or "dependencies"
    ).strip()
    parent_name = (task.parent_package_name or group.parent_package_name or "").strip()
    child_name = (group.vulnerable_component or "").strip()
    if parent_name == package_name and child_name and child_name != package_name:
        child_floor = (
            group.fix_plan.fixed_version.strip()
            if group.fix_plan and group.fix_plan.fixed_version
            else next(
                (
                    floor
                    for issue in group.issues
                    if (floor := _python_scanner_fixed_version(issue))
                ),
                None,
            )
        )
        child_floor_text = f' to at least "{child_floor}"' if child_floor else ""
        return (
            f'Update directly declared parent "{package_name}" in {declaration} to the exact '
            f'solver-approved version "{target_version}" so it resolves transitive package '
            f'"{child_name}"{child_floor_text}. '
            "Do not choose a different version or use a package override."
        )
    return (
        f'Update "{package_name}" in {manifest_path} ({declaration}) to the exact '
        f'solver-approved version "{target_version}". Do not select a different version.'
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
        or solver_plan.status not in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
        or not solver_plan.candidate_catalog_complete
        or selected is None
    ):
        raise ValueError("portfolio task decisions require an accepted complete solver plan")
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
            if task.status in _TERMINAL_STATUSES:
                raise ValueError(
                    f"portfolio plan has no audit decision for terminal task {task_id!r}"
                )
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
        is_maven_target = _group_manager(group) == "maven"
        requirements_manifest = _python_requirements_manifest_path(group)
        pipfile_manifest = _python_pipfile_manifest_path(group)
        python_target = expected_manifest in {requirements_manifest, pipfile_manifest}
        pipfile_target = expected_manifest == pipfile_manifest and pipfile_manifest is not None
        decision_target_name = str(decision.target_package_name or "").strip()
        certified_parent_target = False
        target_identity_package = expected_package

        if pipfile_target:
            from remediation_engine.tools.package_identity import normalize_python_package_name

            child_name = normalize_python_package_name(str(group.vulnerable_component or ""))
            parent_name = normalize_python_package_name(
                str(decision.parent_package_name or decision_target_name)
            )
            selected_version = str(decision.selected_version or "")
            parent_floor = str(decision.parent_minimum_version or "")
            scanner_floors = [
                floor
                for issue in group.issues
                if (issue.cve_id or issue.ghsa_id)
                and (floor := _python_scanner_fixed_version(issue))
            ]
            try:
                expected_child_floor = max(
                    (_python_numeric_solver_version(value) for value in scanner_floors),
                    key=lambda item: item[1],
                )[0]
                parent_selected_key = _python_numeric_solver_version(selected_version)[1]
                parent_floor_key = _python_numeric_solver_version(parent_floor)[1]
            except (ValueError, TypeError):
                expected_child_floor = ""
                parent_selected_key = None
                parent_floor_key = None
            certified_parent_target = bool(
                certificate is not None
                and str(certificate_status or "").upper() == "CERTIFIED"
                and group.issue_type == IssueType.SCA
                and child_name
                and parent_name
                and parent_name == normalize_python_package_name(decision_target_name)
                and parent_name != child_name
                and normalize_python_package_name(str(decision.vulnerable_package_name or ""))
                == child_name
                and str(decision.vulnerable_fixed_version or "") == expected_child_floor
                and decision.parent_package_version
                and parent_floor_key is not None
                and parent_selected_key is not None
                and parent_selected_key >= parent_floor_key
                and expected_manifest == pipfile_manifest
                and decision.lockfile_package_key is None
            )
            if not certified_parent_target:
                raise ValueError(
                    f"Pipfile decision for task {task_id!r} lacks a certified one-hop parent target"
                )
            target_identity_package = parent_name

        if is_maven_target:
            # Maven targets use exact POM/GAV identities, not npm lockfile keys.
            expected_occurrence = f"{expected_manifest}::maven::{expected_package}"
            expected_lockfile_key = (
                f"{expected_package}@{decision.installed_version}"
                if decision.installed_version
                else None
            )
            decision_lockfile_key = str(decision.lockfile_package_key or "").replace("\\", "/")
            direct_occurrence = expected_occurrence
            physical_occurrence = expected_occurrence
            lockfile_key_valid = (
                expected_lockfile_key is not None and decision_lockfile_key == expected_lockfile_key
            )
        elif python_target:
            decision_lockfile_key = None
            direct_occurrence = f"python:{expected_manifest}:{target_identity_package}"
            physical_occurrence = direct_occurrence
            lockfile_key_valid = decision.lockfile_package_key is None
        else:
            decision_lockfile_key = str(decision.lockfile_package_key or "").replace("\\", "/")
            direct_occurrence = make_occurrence_id(expected_manifest, target_identity_package)
            physical_occurrence = make_occurrence_id(
                expected_manifest,
                target_identity_package,
                decision_lockfile_key,
            )
            lockfile_key_valid = lockfile_key_matches_package(
                decision_lockfile_key,
                target_identity_package,
            )
        decision_occurrence = str(decision.target_occurrence_id or "")
        decision_strategy = str(decision.selected_strategy).lower().replace("-", "_")
        non_mutating_workaround = (
            task.strategy == RoutingStrategy.CODE_WORKAROUND
            and decision_strategy in {"code_workaround", "workaround"}
            and decision.selected_version is None
            and not decision.allowed_alternative_versions
            and bool(decision.selected_plan_issue_ids)
        )
        if python_target and decision.lockfile_package_key is not None:
            raise ValueError(
                f"Python portfolio decision for task {task_id!r} must use the no-lockfile identity"
            )
        if (
            python_target
            and not certified_parent_target
            and decision_target_name != expected_package
        ):
            raise ValueError(
                f"Python portfolio decision for task {task_id!r} changed its package target"
            )
        if not python_target and not lockfile_key_valid:
            raise ValueError(
                f"portfolio decision for task {task_id!r} has invalid lockfile_package_key "
                f"{decision_lockfile_key!r}"
            )
        if decision_occurrence not in {direct_occurrence, physical_occurrence}:
            raise ValueError(
                f"portfolio decision for task {task_id!r} has invalid target occurrence "
                f"{decision_occurrence!r}"
            )
        identity_values = {
            "target_group_id": expected_group_id,
            "target_package_name": target_identity_package,
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
        # Terminal decisions are audit-only: verify identity, then skip checks
        # that govern package mutations rather than reopening the task.
        if task.status in _TERMINAL_STATUSES:
            committed[task_id] = task.model_copy(
                update={
                    "portfolio_plan_id": plan_id,
                    "task_revision": task.task_revision + 1,
                }
            )
            continue
        if (
            not is_maven_target
            and not python_target
            and decision_occurrence == physical_occurrence
            and physical_occurrence != direct_occurrence
            and decision.dependency_type not in _OVERRIDE_DEPENDENCY_TYPES
            and not non_mutating_workaround
        ):
            raise ValueError(
                f"nested lockfile target for task {task_id!r} requires a package override"
            )
        if non_mutating_workaround:
            # Source-only workarounds must not retain package mutation
            # authorization inherited from the initial task or a prior stage.
            committed[task_id] = task.model_copy(
                update={
                    "selected_version": None,
                    "allowed_target_versions": [],
                    "allowed_dependency_types": [],
                    "target_dependency_type": None,
                    "portfolio_plan_id": plan_id,
                    "selected_plan_issue_ids": list(decision.selected_plan_issue_ids),
                    "task_revision": task.task_revision + 1,
                }
            )
            continue
        updates: dict[str, Any] = {}
        if certified_parent_target:
            updates.update(
                {
                    "target_package_name": target_identity_package,
                    "parent_package_name": target_identity_package,
                    "parent_package_version": decision.parent_package_version,
                    "parent_minimum_version": decision.parent_minimum_version,
                }
            )
        decision_strategy = str(decision.selected_strategy).lower().replace("-", "_")
        if decision_strategy in {"version_bump", "versionbump"}:
            updates["strategy"] = RoutingStrategy.VERSION_BUMP
        elif decision_strategy in {"code_workaround", "workaround", "no_fix"}:
            updates["strategy"] = RoutingStrategy.CODE_WORKAROUND
            if decision_strategy == "no_fix" and task.no_fix_stage is None:
                # No catalog-backed package fix or authorized workaround exists.
                # Keep this coverage unresolved and terminalize only this task so
                # viable packages in the certified partial plan can proceed.
                updates["status"] = TaskStatus.UNFIXABLE
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
        elif decision_strategy in {"version_bump", "versionbump"} and decision.selected_version:
            updates["instruction"] = _version_bump_instruction(
                task.model_copy(
                    update=(
                        {
                            "parent_package_name": target_identity_package,
                            "parent_package_version": decision.parent_package_version,
                            "parent_minimum_version": decision.parent_minimum_version,
                        }
                        if certified_parent_target
                        else {}
                    )
                ),
                group,
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
