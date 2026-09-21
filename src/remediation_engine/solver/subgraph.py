"""Deterministic, bounded extraction of the solver occurrence subgraph.

This module is intentionally independent of orchestration state.  The graph
snapshot is the only source of npm metadata and the caller-provided targets are
the only occurrences that may become mutation targets; nested lockfile entries
are used as evidence for edges, never silently promoted into tasks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from remediation_engine.contracts.solver_models import (
    SolverEdge,
    SolverFindingRequirement,
    SolverPeerConstraint,
    SolverSubgraph,
    SolverTarget,
)
from remediation_engine.tools.npm_graph import (
    NpmGraphSnapshot,
    NpmLockfilePackage,
    check_npm_range,
    normalize_dependency_ancestry,
)

_EXACT_VERSION_RE = re.compile(r"^[=vV]?\d+\.\d+\.\d+(?:[-+]\w[\w.-]*)?$")
_DEPENDENCY_SECTIONS = ("dependencies", "optionalDependencies", "peerDependencies")


def _validated_target(value: SolverTarget | Mapping[str, Any]) -> SolverTarget:
    """Validate one target while accepting JSON-compatible callers."""
    if isinstance(value, SolverTarget):
        return value
    return SolverTarget.model_validate(value)


def _validated_finding(
    value: SolverFindingRequirement | Mapping[str, Any],
) -> SolverFindingRequirement:
    """Validate one finding requirement while accepting JSON-compatible callers."""
    if isinstance(value, SolverFindingRequirement):
        return value
    return SolverFindingRequirement.model_validate(value)


_ModelT = TypeVar("_ModelT")


def _deduplicate_models(values: Sequence[_ModelT], identity: Any, label: str) -> list[_ModelT]:
    """Deduplicate identical models and reject conflicting identities.

    A repeated JSON record is harmless and is removed.  Reusing an occurrence,
    task, or finding identity for a different record is an input error rather
    than an arbitrary last-write-wins choice.
    """
    by_identity: dict[str, _ModelT] = {}
    result: list[_ModelT] = []
    for value in values:
        key = str(identity(value))
        previous = by_identity.get(key)
        if previous is not None:
            if previous != value:
                raise ValueError(f"{label} identity {key!r} has conflicting records.")
            continue
        by_identity[key] = value
        result.append(value)
    return result


def _target_identity(target: SolverTarget) -> str:
    return target.occurrence_id


def _finding_identity(finding: SolverFindingRequirement) -> str:
    return finding.finding_id


def _normalise_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Normalize undirected peer-conflict pairs and report malformed entries."""
    result: set[tuple[str, str]] = set()
    diagnostics: list[str] = []
    for raw_pair in pairs:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            diagnostics.append(f"invalid peer conflict pair {raw_pair!r}")
            continue
        left, right = (str(raw_pair[0]).strip(), str(raw_pair[1]).strip())
        if not left or not right or left == right:
            diagnostics.append(f"invalid peer conflict pair {raw_pair!r}")
            continue
        result.add(tuple(sorted((left, right))))
    return sorted(result), diagnostics


def _package_key_for_target(target: SolverTarget, package: NpmLockfilePackage | None) -> str:
    """Return a normalized physical key for matching nested lock entries."""
    if package is not None:
        return package.package_key.replace("\\", "/")
    return target.lockfile_package_key.replace("\\", "/")


def _metadata_for_target(
    target: SolverTarget,
    lock_packages: Mapping[tuple[str, str], NpmLockfilePackage],
) -> NpmLockfilePackage | None:
    """Find exact physical lock metadata for a target, if present."""
    key = (target.manifest_path, target.lockfile_package_key.replace("\\", "/"))
    return lock_packages.get(key)


def _child_target(
    source: SolverTarget,
    child_name: str,
    targets_by_id: Mapping[str, SolverTarget],
    lock_packages: Mapping[tuple[str, str], NpmLockfilePackage],
) -> SolverTarget | None:
    """Resolve a represented child without creating a nested mutation target."""
    candidates = [
        target
        for target in targets_by_id.values()
        if target.manifest_path == source.manifest_path and target.target_package_name == child_name
    ]
    if not candidates:
        return None
    source_package = _metadata_for_target(source, lock_packages)
    source_key = _package_key_for_target(source, source_package)
    nested_key = f"{source_key.rstrip('/')}/node_modules/{child_name}"
    exact = [
        target
        for target in candidates
        if target.lockfile_package_key.replace("\\", "/") == nested_key
    ]
    if exact:
        return min(exact, key=lambda target: (target.occurrence_id, target.task_id))
    # A nested lockfile package is a distinct physical dependency.  It is not
    # safe to bind that edge to an unrelated direct occurrence merely because
    # the nested package is not itself a mutation target.
    if (source.manifest_path, nested_key) in lock_packages:
        return None
    # Hoisted children are represented by the direct lockfile key.  Prefer a
    # physical entry at the shallowest depth and then deterministic identity.
    return min(
        candidates,
        key=lambda target: (
            target.lockfile_package_key.replace("\\", "/").count("node_modules/"),
            target.occurrence_id,
            target.task_id,
        ),
    )


def _workspace_roots(snapshot: NpmGraphSnapshot, target: SolverTarget) -> set[str]:
    """Return workspace roots for one target, including its explicit override."""
    roots = set(snapshot.workspace_membership_map.get(target.manifest_path, set()))
    if target.workspace_id:
        roots.add(target.workspace_id)
    return roots


def _same_boundary(snapshot: NpmGraphSnapshot, left: SolverTarget, right: SolverTarget) -> bool:
    """Whether two targets share a manifest or a workspace boundary."""
    if left.manifest_path == right.manifest_path:
        return True
    return bool(_workspace_roots(snapshot, left) & _workspace_roots(snapshot, right))


def _scope(package_name: str) -> str | None:
    """Return an npm scope, never pairing unscoped packages."""
    if package_name.startswith("@") and "/" in package_name:
        return package_name.split("/", 1)[0]
    return None


def _edge(
    source: SolverTarget,
    target: SolverTarget,
    edge_kind: str,
    *,
    version_range: str | None = None,
    is_optional: bool = False,
    is_peer_coupling: bool = False,
) -> SolverEdge | None:
    """Create an edge while preventing invalid self relationships."""
    if source.occurrence_id == target.occurrence_id:
        return None
    return SolverEdge(
        source_occurrence_id=source.occurrence_id,
        target_occurrence_id=target.occurrence_id,
        source_task_id=source.task_id,
        target_task_id=target.task_id,
        edge_kind=edge_kind,
        version_range=version_range,
        is_optional=is_optional,
        is_peer_coupling=is_peer_coupling,
    )


def _dependency_range(metadata: Mapping[str, Any], section: str, package_name: str) -> str | None:
    values = metadata.get(section)
    if not isinstance(values, Mapping) or package_name not in values:
        return None
    value = values.get(package_name)
    return str(value).strip() if value is not None else ""


def _build_metadata_edges(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    *,
    diagnostics: list[str],
) -> tuple[list[SolverEdge], list[SolverPeerConstraint]]:
    """Extract represented runtime and peer relationships from lock metadata."""
    targets_by_id = {target.occurrence_id: target for target in targets}
    lock_packages = {
        (package.manifest_path, package.package_key.replace("\\", "/")): package
        for package in snapshot.lockfile_packages
    }
    edges: dict[tuple[Any, ...], SolverEdge] = {}
    peer_constraints: dict[tuple[Any, ...], SolverPeerConstraint] = {}
    for source in sorted(targets, key=lambda item: item.occurrence_id):
        package = _metadata_for_target(source, lock_packages)
        if package is None:
            continue
        metadata = package.metadata
        for section in _DEPENDENCY_SECTIONS:
            values = metadata.get(section)
            if not isinstance(values, Mapping):
                continue
            for raw_name in sorted(values, key=str):
                child_name = str(raw_name).strip()
                if not child_name:
                    continue
                child = _child_target(source, child_name, targets_by_id, lock_packages)
                requirement = _dependency_range(metadata, section, child_name)
                optional = section == "optionalDependencies"
                if section == "peerDependencies":
                    peer_meta = metadata.get("peerDependenciesMeta")
                    peer_meta_value = (
                        peer_meta.get(child_name) if isinstance(peer_meta, Mapping) else None
                    )
                    optional = optional or (
                        isinstance(peer_meta_value, Mapping)
                        and bool(peer_meta_value.get("optional"))
                    )
                    if not requirement:
                        requirement = "<invalid-empty-peer-range>"
                        diagnostics.append(
                            f"invalid peer range for {source.occurrence_id!r} -> {child_name!r}: empty range"
                        )
                    elif (
                        child is not None
                        and check_npm_range(requirement, child.installed_version).matches is None
                    ):
                        diagnostics.append(
                            f"invalid peer range {requirement!r} for {source.occurrence_id!r} -> {child_name!r}"
                        )
                if child is None:
                    if section == "peerDependencies":
                        diagnostics.append(
                            f"unrepresented peer dependency {child_name!r} for {source.occurrence_id!r}"
                        )
                    continue
                if section == "peerDependencies":
                    constraint = SolverPeerConstraint(
                        source_occurrence_id=source.occurrence_id,
                        target_occurrence_id=child.occurrence_id,
                        version_range=requirement,
                        is_strict=not optional,
                        is_optional=optional,
                    )
                    peer_key = (
                        constraint.source_occurrence_id,
                        constraint.target_occurrence_id,
                        constraint.version_range,
                        constraint.is_strict,
                        constraint.is_optional,
                    )
                    peer_constraints[peer_key] = constraint
                    relation = _edge(
                        source,
                        child,
                        "peer",
                        version_range=requirement,
                        is_optional=optional,
                        is_peer_coupling=True,
                    )
                else:
                    relation = _edge(
                        source,
                        child,
                        "runtime",
                        version_range=requirement,
                        is_optional=optional,
                    )
                if relation is not None:
                    edges[
                        (
                            relation.source_occurrence_id,
                            relation.target_occurrence_id,
                            relation.edge_kind,
                            relation.version_range,
                            relation.is_optional,
                        )
                    ] = relation
                if (
                    section != "peerDependencies"
                    and requirement
                    and _EXACT_VERSION_RE.fullmatch(requirement)
                ):
                    pinned = _edge(
                        source,
                        child,
                        "pinned",
                        version_range=requirement,
                    )
                    if pinned is not None:
                        edges[
                            (
                                pinned.source_occurrence_id,
                                pinned.target_occurrence_id,
                                pinned.edge_kind,
                                pinned.version_range,
                                pinned.is_optional,
                            )
                        ] = pinned
    return list(edges.values()), list(peer_constraints.values())


def _build_ancestry_edges(
    targets: Sequence[SolverTarget],
) -> list[SolverEdge]:
    """Connect adjacent represented package names in declared ancestry paths."""
    by_manifest_name: dict[tuple[str, str], list[SolverTarget]] = {}
    for target in targets:
        by_manifest_name.setdefault((target.manifest_path, target.target_package_name), []).append(
            target
        )
    edges: dict[tuple[str, str, str], SolverEdge] = {}
    for leaf in targets:
        ancestry = normalize_dependency_ancestry(leaf.dependency_ancestry)
        if len(ancestry) < 2:
            continue
        for parent_name, child_name in zip(ancestry, ancestry[1:], strict=False):
            parents = by_manifest_name.get((leaf.manifest_path, parent_name), ())
            children = by_manifest_name.get((leaf.manifest_path, child_name), ())
            for parent in parents:
                for child in children:
                    relation = _edge(parent, child, "dependency_ancestry")
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
    return list(edges.values())


def _build_boundary_edges(
    snapshot: NpmGraphSnapshot, targets: Sequence[SolverTarget]
) -> list[SolverEdge]:
    """Add deterministic workspace and same-scope coupling within safe boundaries."""
    edges: dict[tuple[str, str, str], SolverEdge] = {}
    ordered = sorted(targets, key=lambda item: item.occurrence_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if not _same_boundary(snapshot, left, right):
                continue
            if left.manifest_path != right.manifest_path and _workspace_roots(
                snapshot, left
            ) & _workspace_roots(snapshot, right):
                for source, target in ((left, right), (right, left)):
                    relation = _edge(source, target, "workspace", is_peer_coupling=True)
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
            left_scope, right_scope = (
                _scope(left.target_package_name),
                _scope(right.target_package_name),
            )
            if left_scope is not None and left_scope == right_scope:
                for source, target in ((left, right), (right, left)):
                    relation = _edge(source, target, "scope", is_peer_coupling=True)
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
    return list(edges.values())


def extract_solver_subgraph(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
    *,
    peer_conflict_pairs: Sequence[tuple[str, str]] = (),
    forced_singleton_task_ids: Sequence[str] = (),
) -> SolverSubgraph:
    """Extract a bounded immutable occurrence graph for portfolio solving.

    Args:
        snapshot: Neutral, immutable npm graph metadata.
        targets: Finding-backed, synthetic, or direct task occurrences already
            deemed safe by the outer preparation boundary.
        findings: Finding coverage requirements, including GHSA-only findings.
        peer_conflict_pairs: Explicit occurrence (or task) pairs that must not
            be silently lost between portfolio iterations.
        forced_singleton_task_ids: Task IDs that cannot be atomically coupled.

    Returns:
        A deterministic :class:`SolverSubgraph`.  Duplicate exact records are
        collapsed; conflicting reuse of an occurrence, task, or finding ID
        raises ``ValueError``.

    Raises:
        ValueError: If a duplicate identity has conflicting fields or a model
            fails typed validation.
    """
    if not isinstance(snapshot, NpmGraphSnapshot):
        raise TypeError("snapshot must be an NpmGraphSnapshot")
    normalized_targets = _deduplicate_models(
        [_validated_target(value) for value in targets], _target_identity, "target"
    )
    normalized_findings = _deduplicate_models(
        [_validated_finding(value) for value in findings], _finding_identity, "finding"
    )
    by_task: dict[str, SolverTarget] = {}
    for target in normalized_targets:
        previous = by_task.get(target.task_id)
        if previous is not None and previous != target:
            raise ValueError(f"target task identity {target.task_id!r} has conflicting records.")
        by_task[target.task_id] = target
    normalized_targets.sort(key=lambda item: (item.occurrence_id, item.task_id))
    normalized_findings.sort(key=lambda item: (item.target_occurrence_id, item.finding_id))

    diagnostics = list(snapshot.diagnostics)
    valid = True
    target_ids = {target.occurrence_id for target in normalized_targets}
    occurrence_index = {occurrence.occurrence_id: occurrence for occurrence in snapshot.occurrences}
    for target in normalized_targets:
        occurrence = occurrence_index.get(target.occurrence_id)
        if occurrence is None:
            valid = False
            diagnostics.append(
                f"target occurrence {target.occurrence_id!r} is not present in graph snapshot"
            )
        elif (
            occurrence.manifest_path != target.manifest_path
            or occurrence.package_name != target.target_package_name
        ):
            valid = False
            diagnostics.append(
                f"target occurrence {target.occurrence_id!r} does not match graph snapshot identity"
            )
    for finding in normalized_findings:
        if finding.target_occurrence_id not in target_ids:
            valid = False
            diagnostics.append(
                f"finding {finding.finding_id!r} references unknown target occurrence "
                f"{finding.target_occurrence_id!r}"
            )
        elif finding.vulnerable_package != next(
            target.package_name
            for target in normalized_targets
            if target.occurrence_id == finding.target_occurrence_id
        ):
            valid = False
            diagnostics.append(
                f"finding {finding.finding_id!r} package does not match target occurrence "
                f"{finding.target_occurrence_id!r}"
            )

    pairs, pair_diagnostics = _normalise_pairs(peer_conflict_pairs)
    diagnostics.extend(pair_diagnostics)
    valid = valid and not pair_diagnostics
    task_to_occurrence = {target.task_id: target.occurrence_id for target in normalized_targets}
    external_prerequisites: set[str] = set()
    edges, peer_constraints = _build_metadata_edges(
        snapshot, normalized_targets, diagnostics=diagnostics
    )
    edges.extend(_build_ancestry_edges(normalized_targets))
    edges.extend(_build_boundary_edges(snapshot, normalized_targets))
    for raw_left, raw_right in pairs:
        left = task_to_occurrence.get(raw_left, raw_left)
        right = task_to_occurrence.get(raw_right, raw_right)
        if left not in target_ids:
            valid = False
            diagnostics.append(f"peer conflict endpoint {raw_left!r} is external or unknown")
            external_prerequisites.add(raw_left)
        if right not in target_ids:
            valid = False
            diagnostics.append(f"peer conflict endpoint {raw_right!r} is external or unknown")
            external_prerequisites.add(raw_right)
        if left not in target_ids or right not in target_ids or left == right:
            continue
        source = next(target for target in normalized_targets if target.occurrence_id == left)
        target = next(target for target in normalized_targets if target.occurrence_id == right)
        relation = _edge(source, target, "peer_conflict", is_peer_coupling=True)
        if relation is not None:
            edges.append(relation)
        reverse = _edge(target, source, "peer_conflict", is_peer_coupling=True)
        if reverse is not None:
            edges.append(reverse)
    forced = sorted(
        {str(task_id).strip() for task_id in forced_singleton_task_ids if str(task_id).strip()}
    )
    for task_id in forced:
        if task_id not in by_task:
            valid = False
            diagnostics.append(f"forced singleton task ID {task_id!r} is external or unknown")
    # Explicit task-level conflicts are represented as occurrence-level graph
    # edges above.  Deduplicate all edge and peer records before model creation
    # so repeated extraction and repeated metadata cannot alter the digest.
    unique_edges: dict[tuple[Any, ...], SolverEdge] = {}
    for edge in edges:
        key = (
            edge.source_occurrence_id,
            edge.target_occurrence_id,
            edge.edge_kind,
            edge.version_range,
            edge.is_optional,
            edge.is_peer_coupling,
        )
        unique_edges[key] = edge
    unique_peers: dict[tuple[Any, ...], SolverPeerConstraint] = {}
    for peer in peer_constraints:
        key = (
            peer.source_occurrence_id,
            peer.target_occurrence_id,
            peer.version_range,
            peer.is_strict,
            peer.is_optional,
            peer.candidate_specific_source_version,
        )
        unique_peers[key] = peer
    return SolverSubgraph(
        targets=normalized_targets,
        findings=normalized_findings,
        peer_constraints=sorted(
            unique_peers.values(),
            key=lambda item: (
                item.source_occurrence_id,
                item.target_occurrence_id,
                item.version_range,
                item.is_optional,
            ),
        ),
        edges=sorted(
            unique_edges.values(),
            key=lambda item: (
                item.source_occurrence_id,
                item.target_occurrence_id,
                item.edge_kind,
                item.version_range or "",
                item.is_optional,
            ),
        ),
        forced_singleton_task_ids=forced,
        external_prerequisite_task_ids=sorted(external_prerequisites),
        valid=valid,
        diagnostics=sorted(set(diagnostics)),
    )


__all__ = ["extract_solver_subgraph"]
