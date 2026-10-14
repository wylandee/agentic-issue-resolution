"""Offline tests for deterministic npm lockfile closure slicing."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from remediation_engine.tools.lockfile_closure import (
    ClosureResolutionError,
    build_sliced_lockfile_artifacts,
    resolve_dependency_closure,
    resolve_dependency_closure_for_target,
    resolve_target_packages,
)
from remediation_engine.tools.manifest_locator import _lockfile_package_candidates


def _lockfile(packages: dict[str, dict], version: int = 3) -> dict:
    """Build the smallest package-lock payload accepted by the resolver."""
    return {"lockfileVersion": version, "packages": packages}


def _assert_artifact_graph_consistent(artifacts: dict[str, str]) -> None:
    """Assert lockfile occurrences, npm edges, and manifests describe one graph."""
    lockfile = json.loads(artifacts["package-lock.json"])
    packages = lockfile["packages"]
    root_dependencies = packages[""]["dependencies"]
    assert json.loads(artifacts["package.json"])["dependencies"] == root_dependencies

    occurrence_keys = set(packages) - {""}
    manifest_paths = {
        path.removesuffix("/package.json")
        for path in artifacts
        if path.startswith("node_modules/") and path.endswith("/package.json")
    }
    assert manifest_paths == occurrence_keys
    for package_key, metadata in packages.items():
        if not package_key:
            continue
        manifest = json.loads(artifacts[f"{package_key}/package.json"])
        assert manifest == {"name": metadata["name"], "version": metadata["version"]}
        for category in ("dependencies", "optionalDependencies", "peerDependencies"):
            for package_name, requirement in metadata.get(category, {}).items():
                candidates = _lockfile_package_candidates(
                    packages,
                    package_key,
                    package_name,
                    requirement,
                )
                assert candidates
    for package_name, version in root_dependencies.items():
        root_package = packages[f"node_modules/{package_name}"]
        assert root_package["version"] == version


def test_resolves_nested_duplicate_with_ancestry_and_preserves_keys() -> None:
    packages = {
        "": {},
        "node_modules/parent": {
            "version": "1.0.0",
            "dependencies": {"child": "1.0.0"},
        },
        "node_modules/parent/node_modules/child": {
            "version": "1.0.0",
            "dependencies": {"leaf": "1.0.0"},
        },
        "node_modules/child": {"version": "1.0.0"},
        "node_modules/parent/node_modules/child/node_modules/leaf": {
            "version": "1.0.0",
        },
    }

    closure = resolve_dependency_closure(
        packages,
        target_package="child",
        target_version="1.0.0",
        dependency_ancestry=("parent", "child"),
    )

    assert closure.complete
    assert closure.root_keys == ("node_modules/parent/node_modules/child",)
    assert [node.lockfile_key for node in closure.nodes] == [
        "node_modules/parent/node_modules/child",
        "node_modules/parent/node_modules/child/node_modules/leaf",
    ]


def test_exact_target_resolution_builds_a_distinct_closure_for_each_occurrence() -> None:
    packages = {
        "": {},
        "node_modules/a": {
            "version": "1.0.0",
            "dependencies": {"dep": "1.0.0"},
        },
        "node_modules/a/node_modules/dep": {"version": "1.0.0"},
        "node_modules/parent/node_modules/a": {
            "version": "1.0.0",
            "dependencies": {"dep": "2.0.0"},
        },
        "node_modules/parent/node_modules/a/node_modules/dep": {"version": "2.0.0"},
        "node_modules/dep": {"version": "1.0.0"},
    }
    resolutions, reason = resolve_target_packages(
        packages,
        source_lockfile="package-lock.json",
        target_package="a",
        target_version="1.0.0",
    )

    assert reason is None
    assert len(resolutions) == 2
    closures = {
        closure.root_keys[0]: closure
        for closure in (
            resolve_dependency_closure_for_target(packages, resolution)
            for resolution in resolutions
        )
    }
    assert closures["node_modules/a"].complete
    assert {node.lockfile_key: node.version for node in closures["node_modules/a"].nodes}[
        "node_modules/a/node_modules/dep"
    ] == "1.0.0"
    assert closures["node_modules/parent/node_modules/a"].complete
    assert {
        node.lockfile_key: node.version
        for node in closures["node_modules/parent/node_modules/a"].nodes
    }["node_modules/parent/node_modules/a/node_modules/dep"] == "2.0.0"


def test_cycles_are_complete_and_artifact_retains_required_edges() -> None:
    packages = {
        "": {},
        "node_modules/a": {"version": "1.0.0", "dependencies": {"b": "1.0.0"}},
        "node_modules/a/node_modules/b": {
            "version": "1.0.0",
            "dependencies": {"a": "1.0.0"},
        },
    }

    closure = resolve_dependency_closure(packages, target_package="a", target_version="1.0.0")
    artifacts = build_sliced_lockfile_artifacts(closure)
    lockfile = json.loads(artifacts["package-lock.json"])

    assert closure.complete
    assert set(lockfile["packages"]) == {"", "node_modules/a", "node_modules/b"}
    assert lockfile["packages"]["node_modules/a"]["dependencies"] == {"b": "1.0.0"}
    assert lockfile["packages"]["node_modules/b"]["dependencies"] == {"a": "1.0.0"}
    _assert_artifact_graph_consistent(artifacts)


def test_missing_required_edge_is_incomplete() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"missing": "^1.0.0"},
            },
        },
        target_package="a",
        target_version="1.0.0",
    )

    assert not closure.complete
    assert closure.fallback_reason == "incomplete_closure"
    with pytest.raises(ClosureResolutionError, match="incomplete_closure"):
        build_sliced_lockfile_artifacts(closure)


def test_complete_closure_with_a_missing_required_node_is_rejected() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"dep": "1.0.0"},
            },
            "node_modules/a/node_modules/dep": {"version": "1.0.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )
    truncated_closure = replace(closure, nodes=(closure.nodes[0],))

    with pytest.raises(ClosureResolutionError, match="incomplete_closure"):
        build_sliced_lockfile_artifacts(truncated_closure)


def test_multiple_targets_without_ancestry_fails_closed() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
            "node_modules/parent/node_modules/a": {"version": "1.0.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )

    assert not closure.complete
    assert closure.fallback_reason == "multiple_targets"


def test_no_matching_target_fails_closed() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
        },
        target_package="a",
        target_version="2.0.0",
    )

    assert not closure.complete
    assert closure.fallback_reason == "no_matching_target"


def test_optional_missing_dependency_and_peer_edges_are_omitted() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "optionalDependencies": {"platform-only": "1.0.0"},
                "peerDependencies": {"peer-only": "1.0.0"},
                "peerDependenciesMeta": {"peer-only": {"optional": True}},
            },
        },
        target_package="a",
        target_version="1.0.0",
    )

    assert closure.complete
    artifacts = build_sliced_lockfile_artifacts(closure)
    lockfile = json.loads(artifacts["package-lock.json"])
    package = lockfile["packages"]["node_modules/a"]

    assert package["optionalDependencies"] == {}
    assert package["peerDependencies"] == {}
    assert "peerDependenciesMeta" not in package
    _assert_artifact_graph_consistent(artifacts)


def test_peer_edges_and_peer_metadata_are_retained() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "peerDependencies": {"peer": "^2.0.0"},
                "peerDependenciesMeta": {"peer": {"optional": True}},
            },
            "node_modules/peer": {"version": "2.1.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )
    artifacts = build_sliced_lockfile_artifacts(closure)
    lockfile = json.loads(artifacts["package-lock.json"])
    package = lockfile["packages"]["node_modules/a"]

    assert package["peerDependencies"] == {"peer": "^2.0.0"}
    assert package["peerDependenciesMeta"] == {"peer": {"optional": True}}
    _assert_artifact_graph_consistent(artifacts)


def test_nested_scoped_target_is_rebased_with_matching_manifests() -> None:
    packages = {
        "": {},
        "node_modules/@scope/a": {"version": "1.0.0"},
        "node_modules/parent/node_modules/@scope/a": {
            "version": "2.0.0",
            "dependencies": {"@scope/dep": "2.0.0"},
        },
        "node_modules/parent/node_modules/@scope/a/node_modules/@scope/dep": {
            "version": "2.0.0",
        },
    }
    resolutions, reason = resolve_target_packages(
        packages,
        source_lockfile="frontend/package-lock.json",
        target_package="@scope/a",
        target_version="2.0.0",
    )
    assert reason is None
    assert len(resolutions) == 1
    closure = resolve_dependency_closure_for_target(packages, resolutions[0])
    artifacts = build_sliced_lockfile_artifacts(closure)
    lockfile = json.loads(artifacts["package-lock.json"])

    assert closure.root_keys == ("node_modules/parent/node_modules/@scope/a",)
    assert lockfile["packages"][""]["dependencies"] == {"@scope/a": "2.0.0"}
    assert set(lockfile["packages"]) == {
        "",
        "node_modules/@scope/a",
        "node_modules/@scope/dep",
    }
    _assert_artifact_graph_consistent(artifacts)


def test_conflicting_dependency_versions_remain_nested() -> None:
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"left": "1.0.0", "right": "1.0.0"},
            },
            "node_modules/left": {
                "version": "1.0.0",
                "dependencies": {"shared": "1.0.0"},
            },
            "node_modules/right": {
                "version": "1.0.0",
                "dependencies": {"shared": "2.0.0"},
            },
            "node_modules/shared": {"version": "1.0.0"},
            "node_modules/right/node_modules/shared": {"version": "2.0.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )
    artifacts = build_sliced_lockfile_artifacts(closure)
    lockfile = json.loads(artifacts["package-lock.json"])
    packages = lockfile["packages"]

    assert "node_modules/shared" in packages
    assert "node_modules/right/node_modules/shared" in packages
    assert (
        _lockfile_package_candidates(
            packages,
            "node_modules/left",
            "shared",
            "1.0.0",
        )[0][0]
        == "node_modules/shared"
    )
    assert (
        _lockfile_package_candidates(
            packages,
            "node_modules/right",
            "shared",
            "2.0.0",
        )[0][0]
        == "node_modules/right/node_modules/shared"
    )
    _assert_artifact_graph_consistent(artifacts)


def test_nested_optional_and_peer_edges_preserve_physical_package_occurrences():
    packages = {
        "": {},
        "node_modules/root": {
            "version": "1.0.0",
            "dependencies": {"left": "1.0.0", "right": "1.0.0"},
        },
        "node_modules/root/node_modules/left": {
            "version": "1.0.0",
            "dependencies": {"shared": "1.0.0"},
            "optionalDependencies": {"optional-missing": "1.0.0"},
            "peerDependencies": {"peer": "1.0.0"},
        },
        "node_modules/root/node_modules/left/node_modules/shared": {
            "version": "1.0.0",
        },
        "node_modules/root/node_modules/left/node_modules/peer": {
            "version": "1.0.0",
        },
        "node_modules/root/node_modules/right": {
            "version": "1.0.0",
            "dependencies": {"shared": "2.0.0"},
        },
        "node_modules/root/node_modules/right/node_modules/shared": {
            "version": "2.0.0",
        },
    }

    resolutions, reason = resolve_target_packages(
        packages,
        source_lockfile="packages/root/package-lock.json",
        target_package="root",
        target_version="1.0.0",
    )

    assert reason is None
    assert len(resolutions) == 1
    closure = resolve_dependency_closure_for_target(packages, resolutions[0])
    assert closure.complete
    assert {(node.package_name, node.version, node.lockfile_key) for node in closure.nodes} == {
        ("root", "1.0.0", "node_modules/root"),
        ("left", "1.0.0", "node_modules/root/node_modules/left"),
        (
            "shared",
            "1.0.0",
            "node_modules/root/node_modules/left/node_modules/shared",
        ),
        ("peer", "1.0.0", "node_modules/root/node_modules/left/node_modules/peer"),
        ("right", "1.0.0", "node_modules/root/node_modules/right"),
        (
            "shared",
            "2.0.0",
            "node_modules/root/node_modules/right/node_modules/shared",
        ),
    }
    artifacts = build_sliced_lockfile_artifacts(closure)
    _assert_artifact_graph_consistent(artifacts)
    synthetic_packages = json.loads(artifacts["package-lock.json"])["packages"]
    left_metadata = next(
        metadata
        for key, metadata in synthetic_packages.items()
        if key and metadata["name"] == "left"
    )
    assert left_metadata["peerDependencies"] == {"peer": "1.0.0"}
    assert "optional-missing" not in left_metadata.get("optionalDependencies", {})
