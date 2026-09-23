"""Deterministic package-group portfolio planning for Phase 3."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from remediation_engine.contracts.schemas import (
    MAX_MULTI_PACKAGE_ACTION_SIZE,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    PackageFixPlanCandidate,
    PortfolioPlan,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskCluster,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.portfolio_solver import (
    apply_portfolio_plan as _apply_solver_portfolio_plan,
)
from remediation_engine.orchestration.portfolio_solver import (
    build_portfolio_plan as _build_solver_portfolio_plan,
)
from remediation_engine.orchestration.portfolio_solver import (
    prepare_portfolio_inputs as _prepare_solver_portfolio_inputs,
)
from remediation_engine.tools.npm_graph import npm_range_contains

_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.QA_PASSED,
        TaskStatus.UNFIXABLE,
        TaskStatus.INCONCLUSIVE,
        TaskStatus.PIVOTED,
    }
)
_SUPPORTED_MANAGERS = frozenset({"", "npm"})
_NAMESPACE_PREFIXES = ("@angular/", "@nestjs/")
# A development scope may cross a workspace boundary or add a peer whose
# currently selected target would otherwise violate its declared range. A
# namespace is not a dependency constraint, and a compatible peer is only
# validation evidence; neither should create another mutation task.
_SCOPED_CLOSURE_RELATIONSHIPS = frozenset({"workspace", "peer"})
_EDGE_KIND_PRIORITY = {
    TaskDependencyKind.PEER: 0,
    TaskDependencyKind.WORKSPACE: 1,
    TaskDependencyKind.RUNTIME: 2,
}
_MANIFEST_SECTIONS = (
    "dependencies",
    "devDependencies",
    "optionalDependencies",
    "peerDependencies",
)
_LOCKFILE_DEPENDENCY_SECTIONS = (
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
)
_SYNTHETIC_TARGET_NAMESPACE = "remediation-engine:synthetic-package"
_EXACT_VERSION_RE = re.compile(r"^[=vV]?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class DeltaIsolationResult:
    """Pure canary-attribution result for a failed package cluster."""

    status: Literal["IDENTIFIED", "INTERACTION_FAILURE", "AMBIGUOUS", "INCONCLUSIVE"]
    responsible_task_ids: tuple[str, ...] = ()
    tested_subsets: tuple[tuple[str, ...], ...] = ()
    executions: int = 0
    diagnostic: str = ""


@dataclass(frozen=True)
class _PackageNode:
    task_id: str
    group_id: str
    package_name: str
    target_package_name: str
    manifest_path: str
    strategy: RoutingStrategy
    eligible: bool


@dataclass(frozen=True)
class _DependencyRecord:
    """One direct dependency declaration discovered in a package manifest."""

    manifest_path: str
    package_name: str
    declaration_type: str
    requested_spec: str
    resolved_version: str | None

    @property
    def key(self) -> tuple[str, str]:
        """Return the manifest-occurrence identity for this dependency."""
        return self.manifest_path, self.package_name


def _digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relative_manifest(repo_root: Path, value: str | None) -> str:
    """Normalize a manifest path to a repository-relative POSIX path."""
    if not value:
        return "package.json"
    path = value.replace("\\", "/").lstrip("/")
    try:
        candidate = Path(path)
        if candidate.is_absolute():
            path = candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return ""
    if not path or ".." in Path(path).parts:
        return ""
    return path


def _group_manifest_path(group: VulnerabilityGroup, repo_root: Path) -> str:
    """Resolve the single manifest path used by a package group."""
    paths = {
        _relative_manifest(repo_root, localized.manifest_file)
        for localized in group.localized_issues
        if localized.manifest_file
    }
    paths.update(_relative_manifest(repo_root, path) for path in group.file_paths)
    if group.file_path:
        paths.add(_relative_manifest(repo_root, group.file_path))
    paths.discard("")
    manifest_paths = sorted(path for path in paths if Path(path).name == "package.json")
    return manifest_paths[0] if len(manifest_paths) == 1 and len(paths) == 1 else ""


def _group_manager(group: VulnerabilityGroup) -> str:
    """Return the normalized package manager for a group."""
    managers = {
        (localized.package_manager or "").strip().lower()
        for localized in group.localized_issues
        if localized.package_manager
    }
    return sorted(managers)[0] if len(managers) == 1 else ""


def _active_leaf_tasks(
    task_queue: dict[str, RemediationTask],
) -> tuple[list[RemediationTask], list[str]]:
    """Return one active leaf task per package group and consistency errors."""
    by_group: dict[str, list[RemediationTask]] = defaultdict(list)
    for task in task_queue.values():
        by_group[task.parent_group_id].append(task)

    active: list[RemediationTask] = []
    diagnostics: list[str] = []
    for group_id, tasks in sorted(by_group.items()):
        nonterminal = [task for task in tasks if task.status not in _TERMINAL_STATUSES]
        if len(nonterminal) > 1:
            diagnostics.append(
                f"package group {group_id!r} has multiple nonterminal tasks; clustering disabled"
            )
            continue
        if nonterminal:
            active.append(nonterminal[0])
            continue
        # Terminal groups still belong in the DAG so downstream tasks can see
        # that the prerequisite has completed, but they are not dispatchable.
        active.append(max(tasks, key=lambda task: (task.task_revision, task.task_id)))
    return active, diagnostics


def active_leaf_task_ids(
    task_queue: dict[str, RemediationTask],
    group_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return deterministic active-leaf task IDs, optionally scoped to groups."""
    allowed_groups = set(group_ids) if group_ids is not None else None
    tasks, _diagnostics = _active_leaf_tasks(task_queue)
    return tuple(
        task.task_id
        for task in tasks
        if allowed_groups is None or task.parent_group_id in allowed_groups
    )


def _load_manifests(repo_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Read npm manifests and return their contents plus a repository fingerprint.

    The fingerprint also covers npm lockfiles because an externally changed
    resolved graph can alter runtime and peer ordering without changing a
    manifest declaration.
    """
    manifests: dict[str, dict[str, Any]] = {}
    fingerprint_parts: list[tuple[str, str]] = []
    candidate_paths = {
        path
        for filename in ("package.json", "package-lock.json", "npm-shrinkwrap.json")
        for path in repo_root.rglob(filename)
    }
    for path in sorted(candidate_paths):
        relative = path.relative_to(repo_root).as_posix()
        if set(path.parts) & {".git", "node_modules", ".remedy-attempt-snapshots"}:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        fingerprint_parts.append((relative, raw))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if path.name == "package.json":
            manifests[relative] = payload
    return manifests, _digest(fingerprint_parts)


def _lockfile_package_name(package_key: str) -> str | None:
    """Extract the package name represented by an npm lockfile package key."""
    normalized = package_key.replace("\\", "/")
    marker = "node_modules/"
    if marker not in normalized:
        return None
    package_path = normalized.rsplit(marker, 1)[-1].strip("/")
    if not package_path:
        return None
    pieces = package_path.split("/")
    if pieces[0].startswith("@"):
        return "/".join(pieces[:2]) if len(pieces) >= 2 else None
    return pieces[0]


def _load_lockfile_packages(
    repo_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read npm lockfile package metadata keyed by manifest occurrence.

    Only direct package metadata is consumed by synthetic-task discovery. The
    shortest ``node_modules`` path wins so nested copies cannot change the
    deterministic peer relationship used for a root dependency.
    """
    candidates: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for filename in ("package-lock.json", "npm-shrinkwrap.json"):
        for lockfile in sorted(repo_root.rglob(filename)):
            if set(lockfile.parts) & {".git", "node_modules", ".remedy-attempt-snapshots"}:
                continue
            try:
                payload = json.loads(lockfile.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            packages = payload.get("packages") if isinstance(payload, dict) else None
            manifest_path = (lockfile.parent / "package.json").relative_to(repo_root).as_posix()
            if isinstance(packages, dict):
                for package_key, metadata in packages.items():
                    if not isinstance(package_key, str) or not isinstance(metadata, dict):
                        continue
                    package_name = _lockfile_package_name(package_key)
                    if not package_name:
                        continue
                    depth = package_key.replace("\\", "/").count("node_modules/")
                    candidates[(manifest_path, package_name)].append((depth, package_key, metadata))
                continue

            # npm lockfile v1 stores the hoisted package graph under a nested
            # ``dependencies`` object instead of ``packages``. Top-level
            # entries are the deterministic metadata available for direct
            # manifest declarations; nested copies are not needed for the
            # package-centric portfolio graph.
            dependencies = payload.get("dependencies") if isinstance(payload, dict) else None
            if not isinstance(dependencies, dict):
                continue
            for package_name, metadata in dependencies.items():
                if isinstance(package_name, str) and isinstance(metadata, dict):
                    candidates[(manifest_path, package_name)].append((0, package_name, metadata))

    return {
        key: metadata
        for key, values in candidates.items()
        for _depth, _package_key, metadata in [min(values, key=lambda item: (item[0], item[1]))]
    }


def _manifest_dependency_records(
    manifests: dict[str, dict[str, Any]],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], _DependencyRecord]:
    """Return one deterministic direct-dependency record per package occurrence."""
    records: dict[tuple[str, str], _DependencyRecord] = {}
    for manifest_path, manifest in sorted(manifests.items()):
        for declaration_type in _MANIFEST_SECTIONS:
            values = manifest.get(declaration_type, {})
            if not isinstance(values, dict):
                continue
            for package_name in sorted(str(name).strip() for name in values if str(name).strip()):
                key = (manifest_path, package_name)
                if key in records:
                    # npm permits a package to appear in multiple declaration
                    # sections in malformed or generated manifests. Retain the
                    # first section in the stable section order and let the
                    # planner diagnose any resulting action mismatch.
                    continue
                requested = values.get(package_name)
                requested_spec = str(requested).strip() if requested is not None else ""
                metadata = lockfile_packages.get(key, {})
                resolved = metadata.get("version")
                if not isinstance(resolved, str) or not resolved.strip():
                    resolved = (
                        requested_spec.lstrip("=vV")
                        if _EXACT_VERSION_RE.fullmatch(requested_spec)
                        else None
                    )
                records[key] = _DependencyRecord(
                    manifest_path=manifest_path,
                    package_name=package_name,
                    declaration_type=declaration_type,
                    requested_spec=requested_spec,
                    resolved_version=str(resolved).strip() if resolved else None,
                )
    return records


def _metadata_dependency_names(metadata: dict[str, Any], section: str) -> set[str]:
    """Return normalized dependency names from one lockfile package section."""
    values = metadata.get(section, {})
    if not isinstance(values, dict):
        return set()
    return {str(name).strip() for name in values if str(name).strip()}


def _is_optional_peer(metadata: dict[str, Any], package_name: str) -> bool:
    """Return whether a lockfile peer declaration is marked optional."""
    peer_metadata = metadata.get("peerDependenciesMeta", {})
    if not isinstance(peer_metadata, dict):
        return False
    details = peer_metadata.get(package_name)
    return isinstance(details, dict) and details.get("optional") is True


def _same_workspace_occurrence(
    left_manifest: str,
    right_manifest: str,
    workspace_memberships: dict[str, set[str]],
) -> bool:
    """Return whether two manifest occurrences share a workspace root."""
    if left_manifest == right_manifest:
        return True
    left_memberships = workspace_memberships.get(left_manifest, set())
    right_memberships = workspace_memberships.get(right_manifest, set())
    return bool(left_memberships and left_memberships & right_memberships)


def _dependency_relationships(
    records: dict[tuple[str, str], _DependencyRecord],
    manifests: dict[str, dict[str, Any]],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[tuple[str, str], set[str]]]:
    """Build the package relationship graph used to discover coordination packages.

    Direct manifest declarations are the complete set of package occurrences
    that can become portfolio nodes. Lockfile metadata adds relationships among
    those occurrences without requiring a registry lookup. Runtime edges are
    retained for discovery so a synthetic package can be materialized when it
    is part of the same resolved dependency component as a finding-backed
    package; the planner later uses runtime edges for ordering, not clustering.
    """
    relationships: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    workspace_memberships = _workspace_memberships(manifests)
    records_by_name: dict[str, list[_DependencyRecord]] = defaultdict(list)
    for record in records.values():
        records_by_name[record.package_name].append(record)

    def connect(
        left: tuple[str, str],
        right: tuple[str, str],
        relationship: str,
    ) -> None:
        if left == right:
            return
        relationships[left][right].add(relationship)
        relationships[right][left].add(relationship)

    # Keep the existing Angular/Nest namespace heuristic. The manifest path is
    # still part of each record key, so occurrences in independent manifests
    # remain distinct tasks even when the namespace couples their ordering.
    namespace_records = [
        record
        for record in records.values()
        if any(record.package_name.startswith(prefix) for prefix in _NAMESPACE_PREFIXES)
    ]
    for index, left in enumerate(namespace_records):
        for right in namespace_records[index + 1 :]:
            if any(
                left.package_name.startswith(prefix) and right.package_name.startswith(prefix)
                for prefix in _NAMESPACE_PREFIXES
            ):
                connect(left.key, right.key, "namespace")

    # Lockfile package metadata is the only reliable place to recover peer and
    # runtime relationships for direct dependencies such as Hono and Angular.
    # All discovery edges are undirected; the portfolio graph later records
    # runtime direction and peer coupling in both directions.
    for record in records.values():
        metadata = lockfile_packages.get(record.key, {})
        for section in _LOCKFILE_DEPENDENCY_SECTIONS:
            for dependency_name in sorted(_metadata_dependency_names(metadata, section)):
                if section != "peerDependencies":
                    relationship = "runtime"
                elif _is_optional_peer(metadata, dependency_name):
                    relationship = "optional_peer"
                else:
                    relationship = "peer"
                for dependency_record in records_by_name.get(dependency_name, []):
                    if not _same_workspace_occurrence(
                        record.manifest_path,
                        dependency_record.manifest_path,
                        workspace_memberships,
                    ):
                        continue
                    connect(record.key, dependency_record.key, relationship)

    # A workspace manifest can declare a local package in dependencies. Keep
    # that explicit relationship even when the package is not namespaced.
    for record in records.values():
        manifest = manifests.get(record.manifest_path, {})
        for declaration_type in _MANIFEST_SECTIONS:
            if record.package_name not in _dependency_names(manifest, declaration_type):
                continue
            for local_record in records_by_name.get(record.package_name, []):
                if local_record.manifest_path == record.manifest_path:
                    continue
                if _same_workspace_occurrence(
                    record.manifest_path,
                    local_record.manifest_path,
                    workspace_memberships,
                ):
                    connect(record.key, local_record.key, "workspace")

    return relationships


def _peer_requires_synchronized_target(
    left: tuple[str, str],
    right: tuple[str, str],
    records: dict[tuple[str, str], _DependencyRecord],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]],
    target_versions: dict[tuple[str, str], set[str]],
) -> bool:
    """Return whether a peer edge requires a second mutation target.

    A direct peer is coordination-relevant only when a known selected version
    falls outside the peer range.  This deliberately treats optional peers as
    installed compatibility constraints without recursively expanding their
    optional dependency universe.  A peer whose current version already
    satisfies the selected target remains validation-only and is not promoted
    into the development mutation scope.

    Args:
        left: One manifest/package occurrence in the relationship graph.
        right: The other occurrence in the relationship graph.
        records: Direct manifest dependency records keyed by occurrence.
        lockfile_packages: Lockfile metadata keyed by occurrence.
        target_versions: Selected versions already known for scoped occurrences.

    Returns:
        ``True`` when one endpoint's selected target is outside the other
        endpoint's declared peer range; otherwise ``False``.
    """
    for source_key, peer_key in ((left, right), (right, left)):
        versions = target_versions.get(peer_key, set())
        if not versions:
            continue
        metadata = lockfile_packages.get(source_key, {})
        peer_dependencies = metadata.get("peerDependencies", {})
        if not isinstance(peer_dependencies, dict):
            continue
        peer_name = records[peer_key].package_name
        if _is_optional_peer(metadata, peer_name):
            # Optional peers are compatibility evidence for npm/QA, not a
            # reason to expand the requested mutation scope.
            continue
        required_range = peer_dependencies.get(peer_name)
        if not isinstance(required_range, str) or not required_range.strip():
            continue
        results = [_npm_range_contains(required_range.strip(), version) for version in versions]
        if any(result is False for result in results):
            return True
    return False


def _scoped_dependency_keys(
    records: dict[tuple[str, str], _DependencyRecord],
    relationships: dict[tuple[str, str], dict[tuple[str, str], set[str]]],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]],
    target_packages: set[str],
    seed_versions: dict[tuple[str, str], set[str]],
) -> set[tuple[str, str]]:
    """Return the mutation closure for a development package scope.

    The closure starts at explicitly requested direct dependencies.  It keeps
    workspace coordination and adds peer occurrences only when a selected
    target version is outside the peer's declared range.  Namespace membership
    and already-compatible peers remain outside the mutation graph.

    Args:
        records: Direct manifest dependency records keyed by occurrence.
        relationships: Undirected occurrence relationship graph.
        lockfile_packages: Lockfile metadata keyed by occurrence.
        target_packages: Explicit development package allowlist.
        seed_versions: Supervisor-selected versions for finding-backed seeds.

    Returns:
        Direct occurrence keys allowed to become finding or coordination tasks.
    """
    seeds = {key for key, record in records.items() if record.package_name in target_packages}
    allowed = set(seeds)
    known_versions = {
        key: set(versions) for key, versions in seed_versions.items() if key in seeds and versions
    }
    pending = list(sorted(seeds))
    while pending:
        key = pending.pop(0)
        for neighbor, kinds in sorted(relationships.get(key, {}).items()):
            if neighbor in allowed:
                continue
            include = "workspace" in kinds
            if not include and "peer" in kinds:
                include = _peer_requires_synchronized_target(
                    key,
                    neighbor,
                    records,
                    lockfile_packages,
                    known_versions,
                )
            if not include:
                continue
            allowed.add(neighbor)
            # Propagate the alignment hint through a newly required peer so a
            # second exact-peer hop can be discovered deterministically. The
            # existing synthetic target validation still refuses an unsafe
            # inherited version when its other peer requirements disagree.
            if key in known_versions and neighbor not in known_versions:
                known_versions[neighbor] = set(known_versions[key])
            pending.append(neighbor)
    return allowed


def _synthetic_issue_id(manifest_path: str, package_name: str) -> UUID:
    """Return a stable UUID for a no-CVE coordination finding."""
    return uuid5(
        NAMESPACE_URL,
        f"{_SYNTHETIC_TARGET_NAMESPACE}:{manifest_path}:{package_name}",
    )


def _npm_range_contains(version_range: str, version: str) -> bool | None:
    """Check an npm peer range through the shared neutral graph helper."""
    return npm_range_contains(version_range, version)


def _synthetic_target_version(
    record: _DependencyRecord,
    inherited_version: str | None,
    relationships: dict[tuple[str, str], dict[tuple[str, str], set[str]]],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]],
    coordinated_keys: set[tuple[str, str]] | None = None,
) -> str | None:
    """Choose a deterministic target hint for one synthetic dependency.

    A package with no finding-backed seed is represented at its currently
    resolved lockfile version. That is a coordination-only no-op and avoids
    asking the Supervisor to invent a security version for a package that has
    no finding. An inherited target is used only for namespace/workspace/peer
    coupling, where synchronized versions are part of the relationship. Plain
    runtime relationships affect ordering but must not copy one package's
    fixed version onto another package.

    When a scoped peer component is synchronized, peer ranges from another
    member of that component describe the old installed generation. They are
    skipped here and enforced later against the complete solver candidate set.
    Ranges from packages outside the component remain validation constraints;
    a current version is retained only when it satisfies those requirements.
    """
    current_version = record.resolved_version
    relationship_kinds = {
        relationship
        for neighbor in relationships.get(record.key, {})
        for relationship in relationships[record.key][neighbor]
    }
    aligned = bool(relationship_kinds & {"namespace", "workspace", "peer"})
    if not inherited_version or not aligned:
        return current_version

    requirements: list[str] = []
    for neighbor in relationships.get(record.key, {}):
        metadata = lockfile_packages.get(neighbor, {})
        if coordinated_keys is not None and neighbor in coordinated_keys:
            continue
        peer_dependencies = metadata.get("peerDependencies", {})
        if not isinstance(peer_dependencies, dict):
            continue
        required_range = peer_dependencies.get(record.package_name)
        if (
            isinstance(required_range, str)
            and required_range.strip()
            and not _is_optional_peer(metadata, record.package_name)
        ):
            requirements.append(required_range.strip())
    if not requirements:
        return inherited_version

    inherited_results = [
        _npm_range_contains(required, inherited_version) for required in requirements
    ]
    if all(result is True for result in inherited_results):
        return inherited_version

    if current_version:
        current_results = [
            _npm_range_contains(required, current_version) for required in requirements
        ]
        if all(result is True for result in current_results):
            return current_version
    return None


def _refresh_synthetic_group(
    group: VulnerabilityGroup,
    target_version: str,
) -> VulnerabilityGroup:
    """Refresh a synthetic group's committed target without changing identity."""
    package_name = (group.vulnerable_component or "").strip()
    instruction = (
        f'Keep "{package_name}" at the Supervisor-committed coordination version '
        f"{target_version}; do not select another version."
    )
    old_plan = group.fix_plan
    plan = (
        old_plan.model_copy(
            update={
                "fixed_version": target_version,
                "workaround_snippets": None,
                "instruction": instruction,
                "strategy_used": "synthetic_portfolio_coordination",
            }
        )
        if old_plan is not None
        else FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version=target_version,
            instruction=instruction,
            strategy_used="synthetic_portfolio_coordination",
        )
    )
    if old_plan is not None and old_plan.fixed_version == target_version:
        return group

    candidates = [
        candidate.model_copy(
            update={
                "plan": candidate.plan.model_copy(
                    update={
                        "fixed_version": target_version,
                        "workaround_snippets": None,
                        "instruction": instruction,
                        "strategy_used": "synthetic_portfolio_coordination",
                    }
                )
            }
        )
        for candidate in group.fix_plan_candidates
    ]
    if not candidates:
        candidates = [PackageFixPlanCandidate(issue_id=group.representative_issue_id, plan=plan)]

    updated_issues = [
        issue.model_copy(
            update={
                "fixed_version": target_version,
                "raw_payload": {
                    **(issue.raw_payload or {}),
                    "target_version": target_version,
                },
            }
        )
        for issue in group.issues
    ]
    issue_by_id = {issue.id: issue for issue in updated_issues}
    updated_localized = [
        localized.model_copy(update={"issue": issue_by_id.get(localized.issue.id, localized.issue)})
        for localized in group.localized_issues
    ]
    return group.model_copy(
        deep=True,
        update={
            "issues": updated_issues,
            "localized_issues": updated_localized,
            "fix_plan_candidates": candidates,
            "fix_plan": plan,
        },
    )


def materialize_synthetic_dependency_tasks(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
    target_packages: Iterable[str] | None = None,
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Create package tasks for direct dependencies without CVEs.

    The helper is intentionally called at the outer Portfolio Orchestrator
    boundary after finding-backed tasks have been materialized. It never
    queries a registry or chooses a new security version. A synthetic task
    uses the resolved manifest/lockfile version as a coordination-only target.
    In development scope, only a workspace package or a peer whose selected
    version would become incompatible is materialized alongside the requested
    package.

    Args:
        repo_root: Repository containing npm manifests and optional lockfiles.
        groups: Current triage groups, including previously materialized synthetic groups.
        task_queue: Supervisor-owned task queue used to identify active seeds.
        target_packages: Optional development package allowlist. When present,
            only matching direct dependencies and their required mutation
            closure are materialized. ``None`` or an empty iterable retains
            full-repository behavior.

    Returns:
        A tuple containing the augmented groups, a copy-on-write task queue, and
        deterministic diagnostics for package occurrences that could not be
        safely materialized.
    """
    root = Path(repo_root).resolve()
    group_list = list(groups)
    manifests, _fingerprint = _load_manifests(root)
    if not manifests:
        return group_list, dict(task_queue), []
    lockfile_packages = _load_lockfile_packages(root)
    records = _manifest_dependency_records(manifests, lockfile_packages)
    if not records:
        return group_list, dict(task_queue), []

    relationships = _dependency_relationships(records, manifests, lockfile_packages)
    records_by_key = records
    initial_groups_by_id = {
        group.group_id: group for group in group_list if group.issue_type == IssueType.SCA
    }

    scoped_packages = {
        value.strip()
        for value in (target_packages or ())
        if isinstance(value, str) and value.strip()
    }

    seed_versions: dict[tuple[str, str], set[str]] = defaultdict(set)
    seed_keys = {
        key for key, record in records_by_key.items() if record.package_name in scoped_packages
    }
    for task in task_queue.values():
        group = initial_groups_by_id.get(task.parent_group_id)
        if group is None or group.is_synthetic or task.strategy != RoutingStrategy.VERSION_BUMP:
            continue
        target_version = (task.selected_version or "").strip()
        package_name = (group.vulnerable_component or "").strip()
        manifest_path = _group_manifest_path(group, root)
        key = (manifest_path, package_name)
        if (not scoped_packages or key in seed_keys) and target_version:
            seed_versions[key].add(target_version)

    if scoped_packages:
        all_keys = sorted(
            _scoped_dependency_keys(
                records_by_key,
                relationships,
                lockfile_packages,
                scoped_packages,
                seed_versions,
            )
        )
    else:
        all_keys = sorted(records_by_key)

    # A scoped replan must not retain synthetic tasks that were materialized by
    # an earlier, broader portfolio iteration. Finding-backed target groups are
    # retained only when their occurrence remains inside the scoped mutation
    # closure; non-SCA groups continue through the normal source-remediation
    # path.
    if scoped_packages:
        allowed_group_ids = {
            group.group_id
            for group in group_list
            if group.issue_type != IssueType.SCA
            or (
                _group_manifest_path(group, root),
                (group.vulnerable_component or "").strip(),
            )
            in set(all_keys)
        }
        group_list = [group for group in group_list if group.group_id in allowed_group_ids]
        groups_by_id = {
            group.group_id: group for group in group_list if group.issue_type == IssueType.SCA
        }
    else:
        groups_by_id = initial_groups_by_id

    # Find alignment components deterministically. Every direct dependency
    # record is eligible for a synthetic package occurrence in full-repository
    # mode. In a development scope, namespace membership is intentionally not
    # an alignment relationship: only an actual workspace or peer constraint
    # can synchronize a selected version.
    diagnostics: list[str] = []
    _alignment_parent, (alignment_find, alignment_union) = _union_find(all_keys)
    alignment_relationships = (
        _SCOPED_CLOSURE_RELATIONSHIPS if scoped_packages else {"namespace", "workspace", "peer"}
    )
    for left in all_keys:
        for right, kinds in relationships.get(left, {}).items():
            if right in all_keys and kinds & alignment_relationships:
                alignment_union(left, right)
    alignment_components: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in all_keys:
        alignment_components[alignment_find(key)].append(key)
    alignment_targets: dict[tuple[str, str], str] = {}
    coordinated_component_by_key: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for component in sorted(
        (tuple(sorted(values)) for values in alignment_components.values()),
        key=lambda value: value[0],
    ):
        component_keys = set(component)
        for key in component:
            coordinated_component_by_key[key] = component_keys
        seeded = [key for key in component if key in seed_versions]
        versions = sorted({version for key in seeded for version in seed_versions[key]})
        if len(versions) == 1:
            for key in component:
                alignment_targets[key] = versions[0]
        elif len(versions) > 1:
            diagnostics.append(
                "synthetic dependency targets will use resolved versions for aligned package "
                f"component {list(component)!r}: committed target versions disagree ({versions!r})"
            )

    augmented_groups = list(group_list)
    if scoped_packages:
        augmented_queue = {
            task_id: task.model_copy()
            for task_id, task in task_queue.items()
            if task.parent_group_id in {group.group_id for group in augmented_groups}
        }
    else:
        augmented_queue = {task_id: task.model_copy() for task_id, task in task_queue.items()}
    next_task_index = 1

    def allocate_task_id() -> str:
        """Return the next unused deterministic task identifier."""
        nonlocal next_task_index
        while f"task-{next_task_index}" in augmented_queue:
            next_task_index += 1
        task_id = f"task-{next_task_index}"
        next_task_index += 1
        return task_id

    existing_groups_by_key: dict[tuple[str, str], VulnerabilityGroup] = {}
    for group in groups_by_id.values():
        existing_groups_by_key[
            (_group_manifest_path(group, root), (group.vulnerable_component or "").strip())
        ] = group
    committed_target_keys = {
        (
            _group_manifest_path(group, root),
            (
                task.target_package_name
                or task.parent_package_name
                or (group.vulnerable_component or "")
            ).strip(),
        )
        for task in augmented_queue.values()
        if (group := groups_by_id.get(task.parent_group_id)) is not None
    }
    from remediation_engine.orchestration.task_utils import build_initial_remediation_task

    for key in all_keys:
        record = records_by_key[key]
        existing_group = existing_groups_by_key.get(key)
        if key in committed_target_keys and (
            existing_group is None or not existing_group.is_synthetic
        ):
            continue
        inherited_version = alignment_targets.get(key)
        if existing_group is not None and not existing_group.is_synthetic:
            # Finding-backed groups already have authoritative triage evidence
            # and must never be replaced by a synthetic group.
            continue
        target_version = _synthetic_target_version(
            record,
            inherited_version,
            relationships,
            lockfile_packages,
            coordinated_keys=(coordinated_component_by_key.get(key) if scoped_packages else None),
        )
        if target_version is None:
            reason = (
                "no resolved lockfile version or exact manifest version is available"
                if not record.resolved_version
                else f"inherited target {inherited_version!r} is incompatible with its peer requirements"
            )
            diagnostics.append(
                "synthetic dependency task skipped for "
                f"{record.manifest_path}:{record.package_name}: {reason}"
            )
            continue
        if existing_group is not None:
            if not existing_group.is_synthetic:
                continue
            if (
                existing_group.fix_plan is not None
                and existing_group.fix_plan.fixed_version == target_version
            ):
                group_tasks = [
                    task
                    for task in augmented_queue.values()
                    if task.parent_group_id == existing_group.group_id
                ]
                if group_tasks:
                    continue
                task_id = allocate_task_id()
                augmented_queue[task_id] = build_initial_remediation_task(
                    existing_group,
                    task_id,
                )
                continue
            group_tasks = [
                task
                for task in augmented_queue.values()
                if task.parent_group_id == existing_group.group_id
            ]
            if any(
                task.current_attempt_id is not None or task.status in _TERMINAL_STATUSES
                for task in group_tasks
            ):
                diagnostics.append(
                    "synthetic dependency target refresh deferred for "
                    f"{record.manifest_path}:{record.package_name}: task is already committed"
                )
                continue
            refreshed_group = _refresh_synthetic_group(existing_group, target_version)
            group_index = next(
                (
                    index
                    for index, candidate_group in enumerate(augmented_groups)
                    if candidate_group.group_id == existing_group.group_id
                ),
                None,
            )
            if group_index is not None:
                augmented_groups[group_index] = refreshed_group
            existing_groups_by_key[key] = refreshed_group
            groups_by_id[refreshed_group.group_id] = refreshed_group
            if not group_tasks:
                task_id = allocate_task_id()
                augmented_queue[task_id] = build_initial_remediation_task(
                    refreshed_group,
                    task_id,
                )
            for task in group_tasks:
                fresh_task = build_initial_remediation_task(
                    refreshed_group,
                    task.task_id,
                )
                updates = {
                    "strategy": fresh_task.strategy,
                    "qa_policy": fresh_task.qa_policy,
                    "strategy_stage": fresh_task.strategy_stage,
                    "selected_version": fresh_task.selected_version,
                    "selected_plan_issue_ids": list(fresh_task.selected_plan_issue_ids),
                    "instruction": fresh_task.instruction,
                }
                if any(getattr(task, name) != value for name, value in updates.items()):
                    updates["task_revision"] = task.task_revision + 1
                    augmented_queue[task.task_id] = task.model_copy(update=updates)
            continue
        issue_id = _synthetic_issue_id(record.manifest_path, record.package_name)
        issue = VulnerabilityIssue(
            id=issue_id,
            source=IssueSource.SYNTHETIC,
            issue_type=IssueType.SCA,
            severity=Severity.UNKNOWN,
            file_path=record.manifest_path,
            package_name=record.package_name,
            package_version=record.resolved_version,
            fixed_version=target_version,
            ecosystem="npm",
            message=(
                "Synthetic coordination finding: direct package occurrence has "
                "no active CVE and is represented in the portfolio graph."
            ),
            raw_payload={
                "synthetic": True,
                "reason": "unflagged direct dependency",
                "manifest_path": record.manifest_path,
                "target_version": target_version,
            },
        )
        localized = LocalizedIssue(
            issue=issue,
            manifest_file=record.manifest_path,
            is_direct_dependency=True,
            package_manager="npm",
            declaration_type=record.declaration_type,
            localization_confidence=1.0,
        )
        plan = FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version=target_version,
            instruction=(
                f'Keep "{record.package_name}" at the Supervisor-committed '
                f"coordination version {target_version}; do not select another version."
            ),
            strategy_used="synthetic_portfolio_coordination",
        )
        group = VulnerabilityGroup(
            group_id=f"sca:{record.manifest_path}:{record.package_name}",
            issue_type=IssueType.SCA,
            vulnerable_component=record.package_name,
            file_path=record.manifest_path,
            file_paths=[record.manifest_path],
            versions=[record.resolved_version] if record.resolved_version else [],
            sources=[IssueSource.SYNTHETIC],
            representative_issue_id=issue_id,
            issues=[issue],
            localized_issues=[localized],
            fix_plan_candidates=[PackageFixPlanCandidate(issue_id=issue_id, plan=plan)],
            fix_plan=plan,
            is_synthetic=True,
        )
        augmented_groups.append(group)
        existing_groups_by_key[key] = group
        task_id = allocate_task_id()
        augmented_queue[task_id] = build_initial_remediation_task(group, task_id)

    return augmented_groups, augmented_queue, sorted(set(diagnostics))


def _dependency_names(manifest: dict[str, Any], section: str) -> set[str]:
    """Return dependency names from one npm manifest section."""
    values = manifest.get(section, {})
    return set(values) if isinstance(values, dict) else set()


def _workspace_patterns(manifest: dict[str, Any]) -> list[str]:
    """Return npm workspace glob patterns from a manifest."""
    workspaces = manifest.get("workspaces")
    if isinstance(workspaces, list):
        return sorted(str(value).strip() for value in workspaces if str(value).strip())
    if isinstance(workspaces, dict) and isinstance(workspaces.get("packages"), list):
        return sorted(str(value).strip() for value in workspaces["packages"] if str(value).strip())
    return []


def _workspace_memberships(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Return workspace-root membership for every discovered manifest path."""
    memberships: dict[str, set[str]] = defaultdict(set)
    for manifest_path, manifest in manifests.items():
        base = Path(manifest_path).parent
        if not _workspace_patterns(manifest):
            continue
        # Treat the workspace root itself as a member so a root dependency on
        # a local workspace package forms a workspace edge too.
        memberships[manifest_path].add(manifest_path)
        for pattern in _workspace_patterns(manifest):
            # ``PurePath.match`` is intentionally avoided here: npm patterns
            # are relative to the declaring manifest and can contain ``**``.
            candidate_pattern = (base / pattern / "package.json").as_posix()
            for package_path, package_payload in manifests.items():
                if Path(package_path).match(candidate_pattern) and isinstance(
                    package_payload.get("name"), str
                ):
                    memberships[package_path].add(manifest_path)
    return memberships


def _build_nodes(
    repo_root: Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
) -> tuple[dict[str, _PackageNode], list[str]]:
    """Build one node per active package-group task."""
    groups_by_id = {group.group_id: group for group in groups if group.issue_type == IssueType.SCA}
    active_tasks, diagnostics = _active_leaf_tasks(task_queue)
    nodes: dict[str, _PackageNode] = {}
    for task in active_tasks:
        group = groups_by_id.get(task.parent_group_id)
        if group is None:
            diagnostics.append(
                f"task {task.task_id!r} references missing group {task.parent_group_id!r}"
            )
            continue
        package_name = (group.vulnerable_component or "").strip()
        target_package_name = (task.target_package_name or package_name).strip()
        manifest_path = _group_manifest_path(group, repo_root)
        manager = _group_manager(group)
        eligible = bool(
            package_name
            and target_package_name
            and manifest_path
            and manager in _SUPPORTED_MANAGERS
        )
        if not package_name:
            diagnostics.append(
                f"task {task.task_id!r} has no package identity; clustering disabled"
            )
        if not manifest_path:
            diagnostics.append(
                f"task {task.task_id!r} has ambiguous manifest paths; clustering disabled"
            )
        if manager not in _SUPPORTED_MANAGERS:
            diagnostics.append(
                f"task {task.task_id!r} uses unsupported package manager {manager!r}"
            )
        nodes[task.task_id] = _PackageNode(
            task_id=task.task_id,
            group_id=task.parent_group_id,
            package_name=package_name,
            target_package_name=target_package_name,
            manifest_path=manifest_path,
            strategy=task.strategy,
            eligible=eligible,
        )
    return nodes, diagnostics


def _build_edges(
    nodes: dict[str, _PackageNode],
    manifests: dict[str, dict[str, Any]],
    groups_by_id: dict[str, VulnerabilityGroup],
    lockfile_packages: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[tuple[str, str], set[TaskDependencyKind]]:
    """Build task-level dependency edges from manifests and npm lock metadata."""
    by_package: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        by_package[node.package_name].append(node.task_id)
        if node.target_package_name != node.package_name:
            by_package[node.target_package_name].append(node.task_id)

    edges: dict[tuple[str, str], set[TaskDependencyKind]] = defaultdict(set)
    workspace_memberships = _workspace_memberships(manifests)
    lockfile_packages = lockfile_packages or {}

    def related_package_tasks(package_name: str, manifest_path: str) -> list[str]:
        """Resolve package dependencies without crossing manifest occurrences."""
        manifest_memberships = workspace_memberships.get(manifest_path, set())
        return [
            task_id
            for task_id in by_package.get(package_name, [])
            if nodes[task_id].manifest_path == manifest_path
            or (
                manifest_memberships
                and manifest_memberships
                & workspace_memberships.get(nodes[task_id].manifest_path, set())
            )
        ]

    def add_package_edge(
        node: _PackageNode,
        dependency_name: str,
        kind: TaskDependencyKind,
    ) -> None:
        """Add edges from one package's dependency to its package task."""
        for upstream_id in related_package_tasks(dependency_name, node.manifest_path):
            if upstream_id == node.task_id:
                continue
            upstream_node = nodes[upstream_id]
            edges[(upstream_id, node.task_id)].add(kind)
            if kind == TaskDependencyKind.PEER:
                edges[(node.task_id, upstream_id)].add(kind)
            if workspace_memberships.get(node.manifest_path, set()) & workspace_memberships.get(
                upstream_node.manifest_path, set()
            ):
                edges[(upstream_id, node.task_id)].add(TaskDependencyKind.WORKSPACE)

    for node in nodes.values():
        manifest = manifests.get(node.manifest_path, {})
        manifest_name = str(manifest.get("name", "")).strip()
        # A package.json describes the dependencies of its own package. It is
        # incorrect to treat every root application dependency as an upstream
        # of every other direct dependency merely because they share a
        # manifest. Lockfile metadata below supplies the edges for application
        # dependencies such as express-jwt -> jsonwebtoken.
        if manifest_name in {node.package_name, node.target_package_name}:
            for section in _MANIFEST_SECTIONS:
                kind = (
                    TaskDependencyKind.PEER
                    if section == "peerDependencies"
                    else TaskDependencyKind.RUNTIME
                )
                for dependency_name in sorted(_dependency_names(manifest, section)):
                    add_package_edge(node, dependency_name, kind)

        # A package's resolved dependency and peer requirements are recorded
        # in package-lock metadata, not in the application's root
        # package.json. Include both kinds so every synthetic direct package
        # participates in the same deterministic DAG and peer cluster as its
        # finding-backed neighbors.
        package_metadata = lockfile_packages.get(
            (node.manifest_path, node.target_package_name or node.package_name),
            {},
        )
        if not package_metadata:
            package_metadata = lockfile_packages.get((node.manifest_path, node.package_name), {})
        for section in _LOCKFILE_DEPENDENCY_SECTIONS:
            kind = (
                TaskDependencyKind.PEER
                if section == "peerDependencies"
                else TaskDependencyKind.RUNTIME
            )
            for dependency_name in sorted(_metadata_dependency_names(package_metadata, section)):
                add_package_edge(node, dependency_name, kind)

        group = groups_by_id.get(node.group_id)
        if group is None:
            continue
        ancestry: list[str] = []
        for localized in group.localized_issues:
            for package_name in localized.dependency_ancestry:
                if package_name not in ancestry:
                    ancestry.append(package_name)
        for package_name in group.dependency_ancestry:
            if package_name not in ancestry:
                ancestry.append(package_name)
        if group.parent_package_name and group.parent_package_name not in ancestry:
            ancestry.insert(0, group.parent_package_name)
        ancestry_positions = {package_name: index for index, package_name in enumerate(ancestry)}
        for upstream_name, upstream_index in ancestry_positions.items():
            for downstream_name, downstream_index in ancestry_positions.items():
                if downstream_index != upstream_index + 1:
                    continue
                upstream_ids = related_package_tasks(upstream_name, node.manifest_path)
                downstream_ids = related_package_tasks(downstream_name, node.manifest_path)
                for upstream_id in upstream_ids:
                    for downstream_id in downstream_ids:
                        if upstream_id != downstream_id:
                            edges[(upstream_id, downstream_id)].add(TaskDependencyKind.RUNTIME)

    # Explicit namespace packages are a conservative coupling signal for
    # Angular/Nest package upgrades, including packages declared in different
    # workspace manifests.
    namespace_nodes = [
        node
        for node in nodes.values()
        if any(node.target_package_name.startswith(prefix) for prefix in _NAMESPACE_PREFIXES)
    ]
    for index, left in enumerate(namespace_nodes):
        for right in namespace_nodes[index + 1 :]:
            if any(
                left.target_package_name.startswith(prefix)
                and right.target_package_name.startswith(prefix)
                for prefix in _NAMESPACE_PREFIXES
            ):
                edges[(left.task_id, right.task_id)].add(TaskDependencyKind.WORKSPACE)
                edges[(right.task_id, left.task_id)].add(TaskDependencyKind.WORKSPACE)
    return edges


def _canonical_dependency_kind(kinds: Iterable[TaskDependencyKind]) -> TaskDependencyKind:
    """Select one deterministic dependency kind for a directed task pair.

    The internal portfolio graph retains every relationship kind for clustering,
    ordering, and graph hashing. ``TaskCluster.dependencies`` is a compact
    contract, however, and permits only one dependency per directed endpoint
    pair. Peer coupling is the strongest relationship, followed by workspace
    coupling and then ordinary runtime ordering.

    Args:
        kinds: Relationship kinds associated with one directed task pair.

    Returns:
        The highest-priority relationship kind, with a stable value tie-breaker.

    Raises:
        ValueError: If no relationship kinds are provided.
    """
    available = tuple(kinds)
    if not available:
        raise ValueError("at least one dependency kind is required")
    return min(available, key=lambda kind: (_EDGE_KIND_PRIORITY[kind], kind.value))


def _union_find(items: Iterable[str]) -> tuple[dict[str, str], Any]:
    """Create a small deterministic union-find structure."""
    parent = {item: item for item in items}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        parent[max(left_root, right_root)] = min(left_root, right_root)

    return parent, (find, union)


def _cluster_key(task_ids: Iterable[str]) -> str:
    """Return a stable cluster identifier."""
    return f"cluster-{_digest(sorted(task_ids))[:24]}"


def _eligible_component(
    task_ids: list[str],
    nodes: dict[str, _PackageNode],
    task_queue: dict[str, RemediationTask],
) -> tuple[bool, str | None]:
    """Check whether a component can be represented atomically."""
    if len(task_ids) > MAX_MULTI_PACKAGE_ACTION_SIZE:
        return False, "cluster exceeds the multi-package action size limit"
    if any(not nodes[task_id].eligible for task_id in task_ids):
        return False, "cluster contains an unsupported or ambiguous package item"
    if len({nodes[task_id].target_package_name for task_id in task_ids}) != len(task_ids):
        return False, "cluster contains duplicate target package names"
    strategies = {task_queue[task_id].strategy for task_id in task_ids}
    if not strategies <= {RoutingStrategy.VERSION_BUMP}:
        return False, "cluster contains a non-version package strategy"
    if any(task_queue[task_id].status in _TERMINAL_STATUSES for task_id in task_ids):
        return False, "cluster contains a terminal task"
    return True, None


def _bounded_component_sets(
    task_ids: list[str],
    nodes: dict[str, _PackageNode],
    task_queue: dict[str, RemediationTask],
) -> list[list[str]]:
    """Partition an otherwise eligible oversized component deterministically.

    The action-size cap is an execution safety limit, not a reason to discard
    every coupling edge. Finding-backed package tasks are placed before
    synthetic coordination tasks so a bounded cluster keeps the actionable
    package groups together whenever the cap permits it. The remaining
    coordination tasks are still represented as bounded clusters or
    singletons, and the graph retains their dependency evidence.
    """
    if any(not nodes[task_id].eligible for task_id in task_ids):
        return [[task_id] for task_id in task_ids]
    if len({nodes[task_id].target_package_name for task_id in task_ids}) != len(task_ids):
        return [[task_id] for task_id in task_ids]
    if {task_queue[task_id].strategy for task_id in task_ids} - {RoutingStrategy.VERSION_BUMP}:
        return [[task_id] for task_id in task_ids]
    if any(task_queue[task_id].status in _TERMINAL_STATUSES for task_id in task_ids):
        return [[task_id] for task_id in task_ids]

    prioritized = sorted(
        task_ids,
        key=lambda task_id: (
            task_queue[task_id].is_synthetic,
            _stable_task_key(task_id, nodes),
        ),
    )
    return [
        prioritized[index : index + MAX_MULTI_PACKAGE_ACTION_SIZE]
        for index in range(0, len(prioritized), MAX_MULTI_PACKAGE_ACTION_SIZE)
    ]


def _stable_task_key(task_id: str, nodes: dict[str, _PackageNode]) -> tuple[str, str, str]:
    """Return the stable ordering key for a package task."""
    node = nodes[task_id]
    return node.manifest_path, node.package_name, task_id


def isolate_delta_failure(
    task_ids: Iterable[str],
    probe: Callable[[tuple[str, ...]], Literal["PASS", "FAIL", "INCONCLUSIVE"]],
) -> DeltaIsolationResult:
    """Bisect a failing cluster using deterministic subset probes.

    The supplied probe must start each subset from the same baseline. The
    algorithm performs at most ``2N - 1`` probes for ``N <= 10`` and never
    turns attribution into a QA pass.
    """
    ordered = tuple(dict.fromkeys(sorted(str(task_id) for task_id in task_ids)))
    if not ordered:
        return DeltaIsolationResult("INCONCLUSIVE", diagnostic="no package tasks supplied")
    if len(ordered) > 10:
        return DeltaIsolationResult(
            "INCONCLUSIVE", diagnostic="delta isolation is bounded at ten tasks"
        )
    tested: list[tuple[str, ...]] = []
    outcomes: dict[tuple[str, ...], Literal["PASS", "FAIL", "INCONCLUSIVE"]] = {}
    executions = 0

    def run(subset: tuple[str, ...]) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]:
        nonlocal executions
        if subset in outcomes:
            return outcomes[subset]
        executions += 1
        tested.append(subset)
        outcomes[subset] = probe(subset)
        return outcomes[subset]

    combined = run(ordered)
    if combined != "FAIL":
        return DeltaIsolationResult(
            "INCONCLUSIVE",
            tested_subsets=tuple(tested),
            executions=executions,
            diagnostic="combined cluster did not produce a reproducible failure",
        )

    def bisect(subset: tuple[str, ...]) -> DeltaIsolationResult:
        if len(subset) == 1:
            outcome = run(subset)
            return DeltaIsolationResult(
                "IDENTIFIED" if outcome == "FAIL" else "INCONCLUSIVE",
                responsible_task_ids=subset if outcome == "FAIL" else (),
                diagnostic="singleton canary failed"
                if outcome == "FAIL"
                else "singleton canary was inconclusive",
            )
        midpoint = len(subset) // 2
        left, right = subset[:midpoint], subset[midpoint:]
        left_result = run(left)
        right_result = run(right)
        if "INCONCLUSIVE" in {left_result, right_result}:
            return DeltaIsolationResult("INCONCLUSIVE", diagnostic="a canary was inconclusive")
        if left_result == "PASS" and right_result == "PASS":
            return DeltaIsolationResult(
                "INTERACTION_FAILURE", diagnostic="both halves passed; combined action failed"
            )
        if left_result == "FAIL" and right_result == "FAIL":
            return DeltaIsolationResult("AMBIGUOUS", diagnostic="both halves failed")
        failing_subset = left if left_result == "FAIL" else right
        return bisect(failing_subset)

    result = bisect(ordered)
    return result.__class__(
        status=result.status,
        responsible_task_ids=result.responsible_task_ids,
        tested_subsets=tuple(tested),
        executions=executions,
        diagnostic=result.diagnostic,
    )


def _topological_cluster_order(
    cluster_ids: list[str],
    clusters: dict[str, TaskCluster],
    task_to_cluster: dict[str, str],
    edges: dict[tuple[str, str], set[TaskDependencyKind]],
    nodes: dict[str, _PackageNode],
    diagnostics: list[str],
) -> list[str]:
    """Topologically order clusters after collapsing strongly connected components.

    Peer/workspace components have already been collapsed into portfolio
    clusters.  Remaining strongly connected components are retained as
    separate dispatch clusters, but collapsed for ordering so a non-peer
    dependency cycle cannot deadlock the Supervisor.  Members of a cyclic
    component are emitted in their stable package order and the diagnostic
    makes the conservative fallback visible to operators.
    """
    predecessors: dict[str, set[str]] = {cluster_id: set() for cluster_id in cluster_ids}
    successors: dict[str, set[str]] = {cluster_id: set() for cluster_id in cluster_ids}
    edge_kinds: dict[tuple[str, str], set[TaskDependencyKind]] = defaultdict(set)
    for (upstream, downstream), _ in edges.items():
        left, right = task_to_cluster[upstream], task_to_cluster[downstream]
        if left == right:
            continue
        predecessors[right].add(left)
        successors[left].add(right)
        edge_kinds[(left, right)].update(edges[(upstream, downstream)])

    def key(cluster_id: str) -> tuple[str, str, str]:
        return min(_stable_task_key(task_id, nodes) for task_id in clusters[cluster_id].task_ids)

    # Tarjan's algorithm gives SCCs without depending on set iteration order.
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(cluster_id: str) -> None:
        nonlocal index
        indices[cluster_id] = index
        lowlinks[cluster_id] = index
        index += 1
        stack.append(cluster_id)
        on_stack.add(cluster_id)
        for successor in sorted(successors[cluster_id], key=key):
            if successor not in indices:
                visit(successor)
                lowlinks[cluster_id] = min(lowlinks[cluster_id], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[cluster_id] = min(lowlinks[cluster_id], indices[successor])
        if lowlinks[cluster_id] != indices[cluster_id]:
            return
        members: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            members.append(member)
            if member == cluster_id:
                break
        components.append(tuple(sorted(members, key=key)))

    for cluster_id in sorted(cluster_ids, key=key):
        if cluster_id not in indices:
            visit(cluster_id)

    component_by_cluster = {
        cluster_id: component for component in components for cluster_id in component
    }
    component_key = {
        component: min(key(cluster_id) for cluster_id in component) for component in components
    }
    component_predecessors: dict[tuple[str, ...], set[tuple[str, ...]]] = {
        component: set() for component in components
    }
    component_successors: dict[tuple[str, ...], set[tuple[str, ...]]] = {
        component: set() for component in components
    }
    for (left, right), _kinds in edge_kinds.items():
        left_component = component_by_cluster[left]
        right_component = component_by_cluster[right]
        if left_component == right_component:
            continue
        component_successors[left_component].add(right_component)
        component_predecessors[right_component].add(left_component)

    cyclic_components = [component for component in components if len(component) > 1]
    for component in cyclic_components:
        internal_kinds = [
            kinds
            for (left, right), kinds in edge_kinds.items()
            if component_by_cluster[left] == component and component_by_cluster[right] == component
        ]
        if any(TaskDependencyKind.PEER not in kinds for kinds in internal_kinds):
            diagnostics.append(
                "non-peer portfolio dependency cycle detected; using stable SCC fallback "
                f"for {list(component)!r}"
            )
        else:
            diagnostics.append(f"peer portfolio cycle collapsed for {list(component)!r}")

    ready = [component for component in components if not component_predecessors[component]]
    ready.sort(key=component_key.get)
    ordered_components: list[tuple[str, ...]] = []
    while ready:
        current = ready.pop(0)
        ordered_components.append(current)
        for successor in sorted(component_successors[current], key=component_key.get):
            component_predecessors[successor].discard(current)
            if not component_predecessors[successor] and successor not in ready:
                ready.append(successor)
        ready.sort(key=component_key.get)

    # The SCC condensation graph is a DAG by construction. Retain a safe
    # fallback in case malformed input introduced an untracked endpoint.
    if len(ordered_components) != len(components):
        remaining = sorted(
            set(components) - set(ordered_components),
            key=component_key.get,
        )
        diagnostics.append("portfolio SCC condensation was incomplete; using stable fallback")
        ordered_components.extend(remaining)
    return [cluster_id for component in ordered_components for cluster_id in component]


def _legacy_build_portfolio_plan(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
    *,
    peer_conflict_pairs: Iterable[tuple[str, str]] = (),
    forced_singleton_task_ids: Iterable[str] = (),
) -> PortfolioPlan:
    """Build a deterministic package-group portfolio plan.

    Args:
        repo_root: Repository whose npm manifests should be inspected.
        groups: Current triage package groups.
        task_queue: Supervisor-owned task queue.
        peer_conflict_pairs: Optional task-ID pairs that must be joined after
            deterministic QA identified a peer conflict.
        forced_singleton_task_ids: Task IDs that must not be clustered after
            delta-isolation attribution.

    Returns:
        A validated immutable ``PortfolioPlan``. Unsupported or ambiguous
        package items remain represented as singleton clusters with diagnostics.
    """
    root = Path(repo_root).resolve()
    group_list = list(groups)
    manifests, repository_fingerprint = _load_manifests(root)
    lockfile_packages = _load_lockfile_packages(root)
    nodes, diagnostics = _build_nodes(root, group_list, task_queue)
    if not nodes:
        raise ValueError("Cannot build a portfolio plan without active package tasks.")
    groups_by_id = {
        group.group_id: group for group in group_list if group.issue_type == IssueType.SCA
    }
    edges = _build_edges(nodes, manifests, groups_by_id, lockfile_packages)

    _parent, (find, union) = _union_find(nodes)
    forced_singletons = {task_id for task_id in forced_singleton_task_ids if task_id in nodes}
    for (upstream, downstream), kinds in edges.items():
        if (
            (TaskDependencyKind.PEER in kinds or TaskDependencyKind.WORKSPACE in kinds)
            and upstream not in forced_singletons
            and downstream not in forced_singletons
        ):
            union(upstream, downstream)
    for left, right in peer_conflict_pairs:
        if (
            left in nodes
            and right in nodes
            and left not in forced_singletons
            and right not in forced_singletons
        ):
            union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for task_id in nodes:
        components[find(task_id)].append(task_id)

    clusters: dict[str, TaskCluster] = {}
    task_to_cluster: dict[str, str] = {}
    for component_tasks in components.values():
        task_ids = sorted(component_tasks, key=lambda task_id: _stable_task_key(task_id, nodes))
        eligible, reason = _eligible_component(task_ids, nodes, task_queue)
        if (
            not eligible
            and len(task_ids) > 1
            and reason == "cluster exceeds the multi-package action size limit"
        ):
            component_sets = _bounded_component_sets(task_ids, nodes, task_queue)
            if len(component_sets) > 1 and any(len(item) > 1 for item in component_sets):
                diagnostics.append(
                    f"cluster {task_ids!r} was partitioned into bounded dispatch groups "
                    f"of at most {MAX_MULTI_PACKAGE_ACTION_SIZE} tasks"
                )
            else:
                diagnostics.append(
                    f"cluster {task_ids!r} was reduced to singleton dispatch: {reason}"
                )
        elif not eligible and len(task_ids) > 1:
            diagnostics.append(f"cluster {task_ids!r} was reduced to singleton dispatch: {reason}")
            component_sets = [[task_id] for task_id in task_ids]
        else:
            component_sets = [task_ids]

        for singleton_or_component in component_sets:
            cluster_task_ids = list(singleton_or_component)
            cluster_id = _cluster_key(cluster_task_ids)
            local_dependencies: list[TaskDependency] = []
            for (upstream, downstream), kinds in sorted(edges.items()):
                if downstream not in cluster_task_ids:
                    continue
                local_dependencies.append(
                    TaskDependency(
                        upstream_task_id=upstream,
                        downstream_task_id=downstream,
                        edge_type=_canonical_dependency_kind(kinds),
                    )
                )
            clusters[cluster_id] = TaskCluster(
                cluster_id=cluster_id,
                task_ids=cluster_task_ids,
                dependencies=local_dependencies,
                reason=("coupled package group" if len(cluster_task_ids) > 1 else "package group"),
            )
            for task_id in cluster_task_ids:
                task_to_cluster[task_id] = cluster_id

    cluster_ids = sorted(clusters)
    cluster_order = _topological_cluster_order(
        cluster_ids,
        clusters,
        task_to_cluster,
        edges,
        nodes,
        diagnostics,
    )
    task_order = [
        task_id for cluster_id in cluster_order for task_id in clusters[cluster_id].task_ids
    ]
    task_revisions = {task_id: task_queue[task_id].task_revision for task_id in task_order}
    task_strategies = {task_id: task_queue[task_id].strategy for task_id in task_order}
    graph_payload = {
        "nodes": [
            {
                "task_id": task_id,
                "group_id": nodes[task_id].group_id,
                "package_name": nodes[task_id].package_name,
                "target_package_name": nodes[task_id].target_package_name,
                "manifest_path": nodes[task_id].manifest_path,
                "strategy": nodes[task_id].strategy.value,
            }
            for task_id in sorted(nodes)
        ],
        "edges": [
            [upstream, downstream, sorted(kind.value for kind in kinds)]
            for (upstream, downstream), kinds in sorted(edges.items())
        ],
        "clusters": {cluster_id: clusters[cluster_id].task_ids for cluster_id in cluster_order},
    }
    graph_digest = _digest(graph_payload)
    plan_digest = _digest(
        {
            "graph_digest": graph_digest,
            "task_order": task_order,
            "task_revisions": task_revisions,
            "task_strategies": {
                task_id: strategy.value for task_id, strategy in sorted(task_strategies.items())
            },
        }
    )
    return PortfolioPlan(
        plan_id=f"portfolio-{plan_digest[:24]}",
        repository_fingerprint=repository_fingerprint,
        graph_digest=graph_digest,
        plan_digest=plan_digest,
        task_ids=task_order,
        clusters=[clusters[cluster_id] for cluster_id in cluster_order],
        cluster_order=cluster_order,
        task_order=task_order,
        task_to_cluster=task_to_cluster,
        task_revisions=task_revisions,
        task_strategies=task_strategies,
        diagnostics=sorted(set(diagnostics)),
    )


def repository_fingerprint(repo_root: str | Path) -> str:
    """Return the deterministic fingerprint used by portfolio plans."""
    _manifests, fingerprint = _load_manifests(Path(repo_root).resolve())
    return fingerprint


__all__ = [
    "DeltaIsolationResult",
    "active_leaf_task_ids",
    "build_portfolio_plan",
    "isolate_delta_failure",
    "materialize_synthetic_dependency_tasks",
    "repository_fingerprint",
]


def prepare_portfolio_inputs(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
    target_packages: Iterable[str] | None = None,
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Prepare the outer portfolio inputs using detached task/group objects.

    Args:
        repo_root: Repository whose manifests and lockfiles define the portfolio
            scope.
        groups: Post-triage vulnerability groups to prepare.
        task_queue: Existing task projection to copy before preparation.
        target_packages: Optional development package allowlist. When supplied,
            synthetic dependency discovery is restricted to the selected
            packages and their coordination closure.

    Returns:
        Detached prepared groups, task queue, and preparation diagnostics.
    """
    return _prepare_solver_portfolio_inputs(
        repo_root,
        groups,
        task_queue,
        target_packages=target_packages,
    )


def build_portfolio_plan(
    repo_root: str | Path,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
    *,
    target_packages: Iterable[str] | None = None,
    peer_conflict_pairs: Iterable[tuple[str, str]] = (),
    forced_singleton_task_ids: Iterable[str] = (),
    settings: Any | None = None,
    portfolio_iteration: int = 0,
    portfolio_replan_request: Any | None = None,
) -> PortfolioPlan:
    """Build the solver-backed portfolio plan at the stable public boundary.

    Args:
        repo_root: Repository whose manifests and lockfiles define the graph.
        groups: Prepared vulnerability and coordination groups.
        task_queue: Supervisor-owned task projection.
        target_packages: Optional development package scope.
        peer_conflict_pairs: Explicit QA-discovered peer conflict pairs.
        forced_singleton_task_ids: Tasks that must remain singleton batches.
        settings: Solver and registry settings.
        portfolio_iteration: Current outer portfolio iteration.
        portfolio_replan_request: Optional Supervisor replan constraints.

    Returns:
        An immutable solver-backed portfolio plan.
    """
    return _build_solver_portfolio_plan(
        repo_root,
        groups,
        task_queue,
        target_packages=target_packages,
        peer_conflict_pairs=peer_conflict_pairs,
        forced_singleton_task_ids=forced_singleton_task_ids,
        settings=settings,
        portfolio_iteration=portfolio_iteration,
        portfolio_replan_request=portfolio_replan_request,
    )


def apply_portfolio_plan(
    plan: PortfolioPlan,
    groups: Iterable[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask], list[str]]:
    """Commit solver-approved decisions to detached task objects."""
    return _apply_solver_portfolio_plan(plan, groups, task_queue)


__all__ = [
    "DeltaIsolationResult",
    "active_leaf_task_ids",
    "apply_portfolio_plan",
    "build_portfolio_plan",
    "isolate_delta_failure",
    "materialize_synthetic_dependency_tasks",
    "prepare_portfolio_inputs",
    "repository_fingerprint",
]
