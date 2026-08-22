"""Deterministic npm package-lock dependency closure resolution.

The resolver intentionally contains no Docker or subprocess code.  QA reads a
live lockfile from the workspace volume, passes its decoded ``packages`` map
here, and owns materialising the returned artifact files.
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
    if not isinstance(packages, Mapping) or not packages:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="invalid_lockfile",
            lockfile_version=lockfile_version,
        )
    if lockfile_version not in {2, 3}:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="invalid_lockfile",
            lockfile_version=lockfile_version,
        )

    if any(
        not isinstance(key, str) or not isinstance(value, Mapping)
        for key, value in packages.items()
    ):
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="invalid_lockfile",
            lockfile_version=lockfile_version,
        )

    target_package = target_package.strip()
    if not target_package:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="no_matching_target",
            lockfile_version=lockfile_version,
        )

    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, metadata in packages.items():
        if not isinstance(key, str) or not isinstance(metadata, Mapping):
            continue
        version = str(metadata.get("version") or "").strip()
        package_name = _package_name_from_key(key, metadata)
        if package_name != target_package or not version:
            continue
        if target_version and version != target_version:
            continue
        candidates.append((key, dict(metadata)))

    if not candidates:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=(),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="no_matching_target",
            lockfile_version=lockfile_version,
        )

    ancestry_names = [name.strip() for name in dependency_ancestry if name and name.strip()]

    def ancestry_score(key: str) -> int:
        """Score physical nesting against the logical ancestry hint."""
        key_names = [part for part in key.split("/node_modules/") if part]
        if key_names and key_names[0].startswith("node_modules/"):
            key_names[0] = key_names[0].removeprefix("node_modules/")
        if not ancestry_names:
            return 0
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

    if len(candidates) > 1:
        return _closure_failure(
            source_lockfile=source_lockfile,
            root_keys=tuple(sorted(key for key, _ in candidates)),
            nodes={},
            includes_optional=include_optional,
            includes_peer=include_peer,
            reason="multiple_targets",
            lockfile_version=lockfile_version,
        )

    node_map: dict[str, LockfilePackageNode] = {
        key: LockfilePackageNode(
            lockfile_key=key,
            package_name=_package_name_from_key(key, metadata),
            version=str(metadata.get("version") or ""),
            metadata=deepcopy(dict(metadata)),
        )
        for key, metadata in candidates
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


__all__ = [
    "ClosureResolutionError",
    "DependencyClosure",
    "LockfilePackageNode",
    "build_sliced_lockfile_artifacts",
    "resolve_dependency_closure",
]
