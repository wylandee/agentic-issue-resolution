"""Deterministic npm package-lock target selection and scan artifacts.

The resolver intentionally contains no Docker or subprocess code.  QA reads a
live lockfile from the workspace volume, passes its decoded ``packages`` map
here, and owns materialising the returned artifact files. Targeted QA scans
select each matching exact package node; the dependency-closure helpers remain
available for callers that need the package's transitive graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from remediation_engine.tools.manifest_locator import (
    _lockfile_dependency_edges,
    _lockfile_package_candidate_keys,
    _lockfile_package_candidates,
)


class ClosureResolutionError(ValueError):
    """Raised when a package-lock closure cannot be represented safely."""


@dataclass(frozen=True)
class LockfilePackageNode:
    """One exact npm ``packages`` entry, retaining its physical lockfile key."""

    lockfile_key: str
    package_name: str
    version: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DependencyClosure:
    """Resolved transitive closure for one or more installed target nodes."""

    source_lockfile: str
    root_keys: tuple[str, ...]
    nodes: tuple[LockfilePackageNode, ...]
    includes_optional: bool
    includes_peer: bool
    complete: bool
    fallback_reason: str | None = None
    lockfile_version: int = 3
    failure_detail: str | None = None
    npm_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TargetPackageResolution:
    """Selection result for one exact target entry in an npm lockfile."""

    source_lockfile: str
    target_node: LockfilePackageNode | None
    lockfile_version: int
    candidate_keys: tuple[str, ...] = ()
    fallback_reason: str | None = None

    @property
    def complete(self) -> bool:
        """Return whether one unambiguous target package was selected."""
        return self.target_node is not None and self.fallback_reason is None


def _package_name_from_key(lockfile_key: str, metadata: Mapping[str, Any]) -> str:
    """Return the npm package name represented by a physical lockfile key."""
    if not lockfile_key:
        return str(metadata.get("name") or "")
    return lockfile_key.rsplit("node_modules/", 1)[-1]


def _dependency_requirements(metadata: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return dependency edges with their npm declaration category."""
    return _lockfile_dependency_edges(dict(metadata))


def _selected_candidate_key(
    packages: Mapping[str, Any],
    parent_key: str,
    package_name: str,
    requirement: str,
    selected_keys: set[str],
    overrides: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve an edge while restricting the result to selected closure nodes."""
    candidates = _lockfile_package_candidates(
        dict(packages),
        parent_key,
        package_name,
        requirement,
        overrides=dict(overrides or {}),
    )
    for candidate_key, _metadata in candidates:
        if candidate_key in selected_keys:
            return candidate_key
    return None


def _unresolved_dependency_edge_detail(
    packages: Mapping[str, Any],
    *,
    parent_key: str,
    parent: LockfilePackageNode,
    package_name: str,
    requirement: str,
    category: str,
) -> str:
    """Describe a required dependency edge that has no matching lockfile node."""
    lookup_keys = _lockfile_package_candidate_keys(parent_key, package_name)
    present_candidates = []
    for key in lookup_keys:
        metadata = packages.get(key)
        if isinstance(metadata, Mapping):
            version = str(metadata.get("version") or "<missing version>")
            present_candidates.append(f"{key}@{version}")
    return (
        f"Unresolved {category} edge from "
        f"{parent.package_name}@{parent.version} at {parent_key or '<root>'}: "
        f"{package_name}@{requirement}; npm lookup order={lookup_keys!r}; "
        f"present candidates={present_candidates or '<none>'}."
    )


def _closure_failure(
    *,
    source_lockfile: str,
    root_keys: tuple[str, ...],
    nodes: Mapping[str, LockfilePackageNode],
    includes_optional: bool,
    includes_peer: bool,
    reason: str,
    lockfile_version: int,
    failure_detail: str | None = None,
) -> DependencyClosure:
    """Build a failed closure result while retaining useful diagnostic nodes."""
    return DependencyClosure(
        source_lockfile=source_lockfile,
        root_keys=root_keys,
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        includes_optional=includes_optional,
        includes_peer=includes_peer,
        complete=False,
        fallback_reason=reason,
        lockfile_version=lockfile_version,
        failure_detail=failure_detail,
    )


def resolve_target_package(
    packages: Mapping[str, Any],
    *,
    source_lockfile: str = "package-lock.json",
    target_package: str,
    target_version: str | None = None,
    dependency_ancestry: Sequence[str] = (),
    lockfile_version: int = 3,
) -> TargetPackageResolution:
    """Select one exact package-lock entry without traversing its dependencies.

    Args:
        packages: The npm lockfile ``packages`` object.
        source_lockfile: Workspace-relative source path for evidence.
        target_package: Package controlled by the current remediation task.
        target_version: Expected installed version after the worker edit.
        dependency_ancestry: Scanner/group ancestry used to disambiguate nested
            copies of the same package.
        lockfile_version: Original package-lock format version.

    Returns:
        A resolution containing the selected package node, or a stable fallback
        reason when the lockfile is invalid, the target is absent, or multiple
        entries remain ambiguous.
    """
    resolutions, reason = resolve_target_packages(
        packages,
        source_lockfile=source_lockfile,
        target_package=target_package,
        target_version=target_version,
        dependency_ancestry=dependency_ancestry,
        lockfile_version=lockfile_version,
    )
    if reason:
        return TargetPackageResolution(
            source_lockfile=source_lockfile,
            target_node=None,
            lockfile_version=lockfile_version,
            fallback_reason=reason,
        )
    if len(resolutions) != 1:
        return TargetPackageResolution(
            source_lockfile=source_lockfile,
            target_node=None,
            lockfile_version=lockfile_version,
            candidate_keys=tuple(
                sorted(
                    resolution.target_node.lockfile_key
                    for resolution in resolutions
                    if resolution.target_node is not None
                )
            ),
            fallback_reason="multiple_targets",
        )
    return resolutions[0]


def resolve_target_packages(
    packages: Mapping[str, Any],
    *,
    source_lockfile: str = "package-lock.json",
    target_package: str,
    target_version: str | None = None,
    dependency_ancestry: Sequence[str] = (),
    lockfile_version: int = 3,
) -> tuple[list[TargetPackageResolution], str | None]:
    """Resolve every exact lockfile entry matching a task-owned package.

    A matching physical occurrence is independently scannable, so duplicate
    nested copies are returned as separate resolutions. Dependency ancestry
    still narrows candidates when it distinguishes one path.

    Args:
        packages: The npm lockfile ``packages`` object.
        source_lockfile: Workspace-relative source path for evidence.
        target_package: Package controlled by the current remediation task.
        target_version: Optional installed version to restrict matching entries.
        dependency_ancestry: Scanner/group ancestry used to narrow nested copies.
        lockfile_version: Original package-lock format version.

    Returns:
        Matching exact package entries and an optional stable failure reason.
        The result is empty when the lockfile is invalid or no target matches.
    """
    if (
        not isinstance(packages, Mapping)
        or not packages
        or lockfile_version not in {2, 3}
        or any(
            not isinstance(key, str) or not isinstance(value, Mapping)
            for key, value in packages.items()
        )
    ):
        return [], "invalid_lockfile"

    target_package = target_package.strip()
    if not target_package:
        return [], "no_matching_target"

    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, metadata in packages.items():
        version = str(metadata.get("version") or "").strip()
        package_name = _package_name_from_key(key, metadata)
        if package_name != target_package or not version:
            continue
        if target_version and version != target_version:
            continue
        candidates.append((key, dict(metadata)))

    if not candidates:
        return [], "no_matching_target"

    ancestry_names = [name.strip() for name in dependency_ancestry if name and name.strip()]

    def ancestry_score(key: str) -> int:
        """Score physical nesting against the logical ancestry hint."""
        key_names = [part for part in key.split("/node_modules/") if part]
        if key_names and key_names[0].startswith("node_modules/"):
            key_names[0] = key_names[0].removeprefix("node_modules/")
        score = 0
        start = 0
        for name in ancestry_names:
            try:
                index = key_names.index(name, start)
            except ValueError:
                continue
            score += 1
            start = index + 1
        return score

    candidate_scores = {key: ancestry_score(key) for key, _ in candidates}
    best_score = max(candidate_scores.values(), default=0)
    if best_score:
        candidates = [item for item in candidates if candidate_scores[item[0]] == best_score]

    candidate_keys = tuple(sorted(key for key, _ in candidates))
    return [
        TargetPackageResolution(
            source_lockfile=source_lockfile,
            target_node=LockfilePackageNode(
                lockfile_key=key,
                package_name=_package_name_from_key(key, metadata),
                version=str(metadata.get("version") or ""),
                metadata=deepcopy(metadata),
            ),
            lockfile_version=lockfile_version,
            candidate_keys=candidate_keys,
        )
        for key, metadata in candidates
    ], None


def resolve_dependency_closure_for_target(
    packages: Mapping[str, Any],
    resolution: TargetPackageResolution,
    *,
    include_optional: bool = True,
    include_peer: bool = True,
    npm_overrides: Mapping[str, Any] | None = None,
) -> DependencyClosure:
    """Resolve dependencies starting from one already-selected physical node.

    Args:
        packages: The npm lockfile ``packages`` object used to resolve edges.
        resolution: Exact target selection produced from that same lockfile.
        include_optional: Traverse optional dependency edges.
        include_peer: Traverse peer dependency edges.

    Returns:
        A complete closure rooted at ``resolution.target_node``, or an
        incomplete result with a stable fallback reason. The target is never
        resolved again by package name.
    """
    source_lockfile = resolution.source_lockfile
    lockfile_version = resolution.lockfile_version
    target_node = resolution.target_node
    override_rules = dict(npm_overrides or {})
    if (
        not resolution.complete
        or target_node is None
        or lockfile_version not in {2, 3}
        or not isinstance(packages, Mapping)
        or not packages
        or any(
            not isinstance(key, str) or not isinstance(value, Mapping)
            for key, value in packages.items()
        )
    ):
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=resolution.candidate_keys,
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason=resolution.fallback_reason or "invalid_lockfile",
            lockfile_version=lockfile_version,
            failure_detail=(
                "Invalid closure input: target resolution or lockfile metadata is invalid."
            ),
        )

    target_key = target_node.lockfile_key
    if target_key not in packages:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(target_key,),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="incomplete_closure",
            lockfile_version=lockfile_version,
            failure_detail=f"Selected target lockfile key {target_key!r} is absent from the packages map.",
        )

    node_map: dict[str, LockfilePackageNode] = {target_key: target_node}
    queue = [target_key]
    visited: set[str] = set()
    root_keys = (target_key,)

    while queue:
        current_key = queue.pop(0)
        if current_key in visited:
            continue
        visited.add(current_key)
        current = node_map[current_key]
        for package_name, requirement, category in _dependency_requirements(current.metadata):
            if category == "optionalDependencies" and not include_optional:
                continue
            if category == "peerDependencies" and not include_peer:
                continue

            child_key = _selected_candidate_key(
                packages,
                current_key,
                package_name,
                requirement,
                set(node_map),
                overrides=override_rules,
            )
            if child_key is None:
                all_candidates = _lockfile_package_candidates(
                    dict(packages),
                    current_key,
                    package_name,
                    requirement,
                    overrides=override_rules,
                )
                if not all_candidates:
                    peer_meta = current.metadata.get("peerDependenciesMeta", {})
                    peer_optional = (
                        category == "peerDependencies"
                        and isinstance(peer_meta, Mapping)
                        and isinstance(peer_meta.get(package_name), Mapping)
                        and bool(peer_meta[package_name].get("optional"))
                    )
                    if category == "optionalDependencies" or peer_optional:
                        continue
                    return _closure_failure(
                        source_lockfile=source_lockfile,
                        root_keys=root_keys,
                        nodes=node_map,
                        includes_optional=include_optional,
                        includes_peer=include_peer,
                        reason="incomplete_closure",
                        lockfile_version=lockfile_version,
                        failure_detail=_unresolved_dependency_edge_detail(
                            packages,
                            parent_key=current_key,
                            parent=current,
                            package_name=package_name,
                            requirement=requirement,
                            category=category,
                        ),
                    )
                child_key, child_metadata = all_candidates[0]
                node_map[child_key] = LockfilePackageNode(
                    lockfile_key=child_key,
                    package_name=_package_name_from_key(child_key, child_metadata),
                    version=str(child_metadata.get("version") or ""),
                    metadata=deepcopy(dict(child_metadata)),
                )
            if child_key not in visited:
                queue.append(child_key)

    return DependencyClosure(
        source_lockfile=source_lockfile,
        root_keys=root_keys,
        nodes=tuple(node_map[key] for key in sorted(node_map)),
        includes_optional=include_optional,
        includes_peer=include_peer,
        complete=True,
        lockfile_version=lockfile_version,
        npm_overrides=override_rules or None,
    )


def resolve_dependency_closure(
    packages: Mapping[str, Any],
    *,
    source_lockfile: str = "package-lock.json",
    target_package: str,
    target_version: str | None = None,
    dependency_ancestry: Sequence[str] = (),
    include_optional: bool = True,
    include_peer: bool = True,
    lockfile_version: int = 3,
    npm_overrides: Mapping[str, Any] | None = None,
) -> DependencyClosure:
    """Resolve a transitive npm dependency closure from a decoded lockfile.

    Args:
        packages: The npm lockfile ``packages`` object.
        source_lockfile: Workspace-relative source path for diagnostics.
        target_package: Package controlled by the current remediation task.
        target_version: Expected installed version after the worker edit.
        dependency_ancestry: Scanner/group ancestry used to disambiguate nodes.
        include_optional: Traverse optional dependency edges.
        include_peer: Traverse peer dependency edges.
        lockfile_version: Original package-lock format version.
        npm_overrides: Root package npm overrides for this lockfile.

    Returns:
        A complete closure or a result with ``complete=False`` and a stable
        fallback reason. Circular references are handled by the visited set
        and are not failures.
    """
    resolution = resolve_target_package(
        packages,
        source_lockfile=source_lockfile,
        target_package=target_package,
        target_version=target_version,
        dependency_ancestry=dependency_ancestry,
        lockfile_version=lockfile_version,
    )
    return resolve_dependency_closure_for_target(
        packages,
        resolution,
        include_optional=include_optional,
        include_peer=include_peer,
        npm_overrides=npm_overrides,
    )




def _synthetic_package_key(package_name: str) -> str:
    """Return a safe npm physical key for one synthetic package name."""
    parts = package_name.split("/")
    is_scoped = package_name.startswith("@")
    if (
        not package_name
        or (is_scoped and len(parts) != 2)
        or (not is_scoped and len(parts) != 1)
        or any(
            not part or part in {".", ".."} or "\\" in part
            for part in parts
        )
        or (is_scoped and not parts[0].startswith("@"))
    ):
        raise ClosureResolutionError("incomplete_closure")
    return f"node_modules/{package_name}"


def build_sliced_lockfile_artifacts(
    closure: DependencyClosure,
) -> dict[str, str]:
    """Build a synthetic npm project containing one complete package closure.

    The selected root is rebased to ``node_modules/<name>``. Dependencies are
    hoisted when the nearest-package lookup still selects their exact original
    occurrence; conflicting versions remain nested beneath the declaring
    package. Every emitted physical package has a matching minimal manifest.

    Args:
        closure: Complete dependency closure rooted at exact package-lock nodes.

    Returns:
        The root manifest, sliced package lock, and package manifests suitable
        for writing into a targeted ODC project.

    Raises:
        ClosureResolutionError: If a required edge or physical occurrence
            cannot be represented without changing npm's nearest resolution.
    """
    if not closure.complete:
        reason = closure.fallback_reason or "incomplete_closure"
        if closure.failure_detail:
            raise ClosureResolutionError(f"{reason}: {closure.failure_detail}")
        raise ClosureResolutionError(reason)
    if not closure.nodes or not closure.root_keys:
        raise ClosureResolutionError("empty_closure")

    nodes_by_key = {node.lockfile_key: node for node in closure.nodes}
    if len(nodes_by_key) != len(closure.nodes):
        raise ClosureResolutionError("incomplete_closure")
    if any(not isinstance(node.metadata, Mapping) for node in closure.nodes):
        raise ClosureResolutionError("incomplete_closure")
    original_packages = {
        node.lockfile_key: dict(node.metadata)
        for node in closure.nodes
    }
    npm_overrides = closure.npm_overrides or {}
    edge_categories = ("dependencies", "optionalDependencies", "peerDependencies")
    retained_edges: dict[tuple[str, str], dict[str, str]] = {}
    edges_by_parent: dict[str, list[tuple[str, str, str, str]]] = {
        key: [] for key in nodes_by_key
    }

    for node in closure.nodes:
        peer_meta = node.metadata.get("peerDependenciesMeta", {})
        if "peerDependenciesMeta" in node.metadata and not isinstance(peer_meta, Mapping):
            raise ClosureResolutionError("incomplete_closure")
        for category in edge_categories:
            if category not in node.metadata:
                continue
            declared_edges = node.metadata[category]
            if not isinstance(declared_edges, Mapping):
                raise ClosureResolutionError("incomplete_closure")
            retained: dict[str, str] = {}
            for package_name, requirement in sorted(declared_edges.items()):
                if not isinstance(package_name, str) or not isinstance(requirement, str):
                    raise ClosureResolutionError("incomplete_closure")
                _synthetic_package_key(package_name)
                if category == "optionalDependencies" and not closure.includes_optional:
                    continue
                if category == "peerDependencies" and not closure.includes_peer:
                    continue

                candidates = _lockfile_package_candidates(
                    original_packages,
                    node.lockfile_key,
                    package_name,
                    requirement,
                    overrides=npm_overrides,
                )
                if not candidates:
                    peer_optional = (
                        category == "peerDependencies"
                        and isinstance(peer_meta, Mapping)
                        and isinstance(peer_meta.get(package_name), Mapping)
                        and bool(peer_meta[package_name].get("optional"))
                    )
                    if category == "optionalDependencies" or peer_optional:
                        continue
                    raise ClosureResolutionError("incomplete_closure")

                child_key = candidates[0][0]
                if child_key not in nodes_by_key:
                    raise ClosureResolutionError("incomplete_closure")
                retained[package_name] = requirement
                edges_by_parent[node.lockfile_key].append(
                    (category, package_name, requirement, child_key)
                )
            retained_edges[(node.lockfile_key, category)] = retained

    root_dependencies: dict[str, str] = {}
    original_to_synthetic: dict[str, str] = {}
    synthetic_to_original: dict[str, str] = {}
    synthetic_packages: dict[str, dict[str, Any]] = {}
    for root_key in closure.root_keys:
        node = nodes_by_key.get(root_key)
        if node is None or not node.version:
            raise ClosureResolutionError("incomplete_closure")
        if node.package_name in root_dependencies:
            raise ClosureResolutionError("incomplete_closure")
        root_key_synthetic = _synthetic_package_key(node.package_name)
        if root_key_synthetic in synthetic_to_original:
            raise ClosureResolutionError("incomplete_closure")
        root_dependencies[node.package_name] = node.version
        original_to_synthetic[root_key] = root_key_synthetic
        synthetic_to_original[root_key_synthetic] = root_key
        synthetic_packages[root_key_synthetic] = {"version": node.version}

    queue = list(closure.root_keys)
    visited: set[str] = set()
    queue_index = 0
    while queue_index < len(queue):
        parent_original_key = queue[queue_index]
        queue_index += 1
        if parent_original_key in visited:
            continue
        visited.add(parent_original_key)
        parent_synthetic_key = original_to_synthetic.get(parent_original_key)
        if parent_synthetic_key is None:
            raise ClosureResolutionError("incomplete_closure")

        for _category, package_name, requirement, child_original_key in sorted(
            edges_by_parent[parent_original_key],
            key=lambda edge: (edge[0], edge[1], edge[2], edge[3]),
        ):
            child_node = nodes_by_key[child_original_key]
            child_synthetic_key = original_to_synthetic.get(child_original_key)
            if child_synthetic_key is not None:
                resolved = _lockfile_package_candidates(
                    synthetic_packages,
                    parent_synthetic_key,
                    package_name,
                    requirement,
                    overrides=npm_overrides,
                )
                if not resolved or resolved[0][0] != child_synthetic_key:
                    raise ClosureResolutionError("incomplete_closure")
            else:
                child_synthetic_key = None
                candidate_keys = _lockfile_package_candidate_keys(
                    parent_synthetic_key,
                    package_name,
                )
                for candidate_key in reversed(candidate_keys):
                    if candidate_key in synthetic_to_original:
                        continue
                    synthetic_packages[candidate_key] = {"version": child_node.version}
                    synthetic_to_original[candidate_key] = child_original_key
                    resolved = _lockfile_package_candidates(
                        synthetic_packages,
                        parent_synthetic_key,
                        package_name,
                        requirement,
                        overrides=npm_overrides,
                    )
                    if resolved and resolved[0][0] == candidate_key:
                        child_synthetic_key = candidate_key
                        original_to_synthetic[child_original_key] = candidate_key
                        break
                    del synthetic_packages[candidate_key]
                    del synthetic_to_original[candidate_key]
                if child_synthetic_key is None:
                    raise ClosureResolutionError("incomplete_closure")
            queue.append(child_original_key)

    if visited != set(nodes_by_key):
        raise ClosureResolutionError("incomplete_closure")

    root_name = "remediation-engine-targeted-scan"
    root_version = "0.0.0"
    root_metadata = {
        "name": root_name,
        "version": root_version,
        "dependencies": dict(root_dependencies),
    }
    package_lock_packages: dict[str, dict[str, Any]] = {"": root_metadata}
    artifacts = {
        "package.json": json.dumps(
            {
                "name": root_name,
                "version": root_version,
                "private": True,
                "dependencies": dict(root_dependencies),
                "overrides": dict(npm_overrides),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    }

    for original_key in sorted(nodes_by_key):
        node = nodes_by_key[original_key]
        synthetic_key = original_to_synthetic.get(original_key)
        if synthetic_key is None or not node.version:
            raise ClosureResolutionError("incomplete_closure")
        metadata = deepcopy(dict(node.metadata))
        raw_name = metadata.get("name")
        if raw_name is not None and raw_name != node.package_name:
            raise ClosureResolutionError("incomplete_closure")
        raw_version = metadata.get("version")
        if raw_version is not None and str(raw_version) != node.version:
            raise ClosureResolutionError("incomplete_closure")
        metadata["name"] = node.package_name
        metadata["version"] = node.version
        for category in edge_categories:
            if category in metadata:
                metadata[category] = retained_edges.get((original_key, category), {})
        if "peerDependenciesMeta" in metadata:
            peer_meta = metadata["peerDependenciesMeta"]
            if not isinstance(peer_meta, Mapping):
                raise ClosureResolutionError("incomplete_closure")
            retained_peer_names = set(
                retained_edges.get((original_key, "peerDependencies"), {})
            )
            retained_peer_meta = {
                name: deepcopy(value)
                for name, value in peer_meta.items()
                if name in retained_peer_names
            }
            if retained_peer_meta:
                metadata["peerDependenciesMeta"] = retained_peer_meta
            else:
                metadata.pop("peerDependenciesMeta")
        package_lock_packages[synthetic_key] = metadata
        artifacts[f"{synthetic_key}/package.json"] = json.dumps(
            {
                "name": node.package_name,
                "version": node.version,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"

    for parent_original_key, parent_edges in edges_by_parent.items():
        parent_synthetic_key = original_to_synthetic[parent_original_key]
        for _category, package_name, requirement, child_original_key in parent_edges:
            resolved = _lockfile_package_candidates(
                package_lock_packages,
                parent_synthetic_key,
                package_name,
                requirement,
                overrides=npm_overrides,
            )
            if (
                not resolved
                or resolved[0][0] != original_to_synthetic[child_original_key]
            ):
                raise ClosureResolutionError("incomplete_closure")

    lockfile = {
        "name": root_name,
        "version": root_version,
        "lockfileVersion": closure.lockfile_version,
        "requires": True,
        "packages": package_lock_packages,
    }
    artifacts["package-lock.json"] = json.dumps(lockfile, indent=2, sort_keys=True) + "\n"
    return artifacts




__all__ = [
    "ClosureResolutionError",
    "DependencyClosure",
    "LockfilePackageNode",
    "TargetPackageResolution",
    "build_sliced_lockfile_artifacts",
    "resolve_dependency_closure",
    "resolve_dependency_closure_for_target",
    "resolve_target_package",
    "resolve_target_packages",
]
