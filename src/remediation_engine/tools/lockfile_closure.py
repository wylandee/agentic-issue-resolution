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
) -> str | None:
    """Resolve an edge while restricting the result to selected closure nodes."""
    candidates = _lockfile_package_candidates(dict(packages), parent_key, package_name, requirement)
    for candidate_key, _metadata in candidates:
        if candidate_key in selected_keys:
            return candidate_key
    return None


def _closure_failure(
    *,
    source_lockfile: str,
    root_keys: tuple[str, ...],
    nodes: Mapping[str, LockfilePackageNode],
    includes_optional: bool,
    includes_peer: bool,
    reason: str,
    lockfile_version: int,
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

    Returns:
        A complete closure or a result with ``complete=False`` and a stable
        fallback reason.  Circular references are handled by the visited set
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
    if not resolution.complete or resolution.target_node is None:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=resolution.candidate_keys,
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason=resolution.fallback_reason or "invalid_lockfile",
            lockfile_version=lockfile_version,
        )

    node_map: dict[str, LockfilePackageNode] = {
        resolution.target_node.lockfile_key: resolution.target_node
    }
    queue = list(sorted(node_map))
    visited: set[str] = set()
    root_keys = tuple(sorted(node_map))

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
            )
            if child_key is None:
                all_candidates = _lockfile_package_candidates(
                    dict(packages), current_key, package_name, requirement
                )
                if not all_candidates:
                    if category == "optionalDependencies":
                        continue
                    peer_meta = current.metadata.get("peerDependenciesMeta", {})
                    peer_optional = (
                        category == "peerDependencies"
                        and isinstance(peer_meta, Mapping)
                        and isinstance(peer_meta.get(package_name), Mapping)
                        and bool(peer_meta[package_name].get("optional"))
                    )
                    if peer_optional:
                        continue
                    return _closure_failure(
                        source_lockfile=source_lockfile,
                        root_keys=root_keys,
                        nodes=node_map,
                        includes_optional=include_optional,
                        includes_peer=include_peer,
                        reason="incomplete_closure",
                        lockfile_version=lockfile_version,
                    )
                # The edge exists, but the selected closure does not yet
                # contain it.  Choose npm's nearest candidate and add it.
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
    )


def _selected_edge_key(
    packages: Mapping[str, Any],
    parent_key: str,
    package_name: str,
    requirement: str,
    selected_keys: set[str],
) -> str | None:
    """Return the nearest selected child for one dependency declaration."""
    candidates = _lockfile_package_candidates(dict(packages), parent_key, package_name, requirement)
    return next((key for key, _ in candidates if key in selected_keys), None)


def build_sliced_lockfile_artifacts(
    closure: DependencyClosure,
) -> dict[str, str]:
    """Build synthetic package files for a complete closure.

    The returned mapping is intentionally suitable for ``DockerSandbox.write_file``.
    It does not write files or contact Docker.
    """
    if not closure.complete:
        raise ClosureResolutionError(closure.fallback_reason or "incomplete_closure")
    if not closure.nodes:
        raise ClosureResolutionError("empty_closure")

    selected_keys = {node.lockfile_key for node in closure.nodes}
    nodes_by_key = {node.lockfile_key: node for node in closure.nodes}
    packages: dict[str, dict[str, Any]] = {}
    for node in closure.nodes:
        metadata = deepcopy(node.metadata)
        for category in ("dependencies", "optionalDependencies", "peerDependencies"):
            values = metadata.get(category)
            if not isinstance(values, Mapping):
                continue
            retained: dict[str, str] = {}
            for package_name, requirement in values.items():
                if not isinstance(package_name, str) or not isinstance(requirement, str):
                    continue
                child_key = _selected_edge_key(
                    {candidate.lockfile_key: candidate.metadata for candidate in closure.nodes},
                    node.lockfile_key,
                    package_name,
                    requirement,
                    selected_keys,
                )
                if child_key is None:
                    if category == "optionalDependencies":
                        continue
                    raise ClosureResolutionError(
                        f"required edge {node.lockfile_key} -> {package_name} is outside closure"
                    )
                retained[package_name] = requirement
            metadata[category] = retained
        packages[node.lockfile_key] = metadata

    root_dependencies: dict[str, str] = {}
    for root_key in closure.root_keys:
        node = nodes_by_key.get(root_key)
        if node is None:
            raise ClosureResolutionError(f"missing closure root {root_key}")
        root_dependencies[node.package_name] = node.version

    root_metadata = {
        "name": "remediation-engine-targeted-scan",
        "version": "0.0.0",
        "dependencies": dict(root_dependencies),
    }
    packages[""] = root_metadata
    package_json = {
        "name": "remediation-engine-targeted-scan",
        "version": "0.0.0",
        "private": True,
        "dependencies": dict(root_dependencies),
    }
    lockfile = {
        "name": package_json["name"],
        "version": package_json["version"],
        "lockfileVersion": closure.lockfile_version,
        "requires": True,
        "packages": packages,
    }
    return {
        "package.json": json.dumps(package_json, indent=2, sort_keys=True) + "\n",
        "package-lock.json": json.dumps(lockfile, indent=2, sort_keys=True) + "\n",
    }


def build_target_only_lockfile_artifacts(
    resolution: TargetPackageResolution,
) -> dict[str, str]:
    """Build a synthetic npm project containing only the selected target.

    The exact package entry is selected from the live lockfile first, including
    nested-copy disambiguation. The synthetic lockfile then places that one
    package at the project's root and removes its dependency edges so ODC does
    not scan its transitive dependencies during task-scoped QA.

    Args:
        resolution: Complete exact target selection from a live npm lockfile.

    Returns:
        The project manifest, lockfile, and installed package manifest for the
        one-package scan project. The minimal ``node_modules`` entry lets ODC
        enumerate the selected package without installing its dependency tree.

    Raises:
        ValueError: If the target selection is incomplete or invalid.
    """
    node = resolution.target_node
    if not resolution.complete or node is None:
        raise ValueError(resolution.fallback_reason or "incomplete_target")

    target_name = node.package_name
    target_version = node.version
    root_dependencies = {target_name: target_version}
    root_name = "remediation-engine-targeted-scan"
    root_version = "0.0.0"
    target_metadata = deepcopy(node.metadata)
    for category in (
        "dependencies",
        "optionalDependencies",
        "peerDependencies",
        "peerDependenciesMeta",
    ):
        target_metadata.pop(category, None)

    package_key = f"node_modules/{target_name}"
    packages = {
        "": {
            "name": root_name,
            "version": root_version,
            "dependencies": root_dependencies,
        },
        package_key: target_metadata,
    }
    package_json = {
        "name": root_name,
        "version": root_version,
        "private": True,
        "dependencies": root_dependencies,
    }
    installed_package_json = {
        "name": target_name,
        "version": target_version,
    }
    lockfile = {
        "name": root_name,
        "version": root_version,
        "lockfileVersion": resolution.lockfile_version,
        "requires": True,
        "packages": packages,
    }
    return {
        "package.json": json.dumps(package_json, indent=2, sort_keys=True) + "\n",
        "package-lock.json": json.dumps(lockfile, indent=2, sort_keys=True) + "\n",
        f"node_modules/{target_name}/package.json": json.dumps(
            installed_package_json,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }


__all__ = [
    "ClosureResolutionError",
    "DependencyClosure",
    "LockfilePackageNode",
    "TargetPackageResolution",
    "build_sliced_lockfile_artifacts",
    "build_target_only_lockfile_artifacts",
    "resolve_dependency_closure",
    "resolve_target_package",
    "resolve_target_packages",
]
