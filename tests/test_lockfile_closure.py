"""Offline tests for deterministic npm lockfile closure slicing."""

from __future__ import annotations

import json

from remediation_engine.tools.lockfile_closure import (
    build_sliced_lockfile_artifacts,
    resolve_dependency_closure,
)


def _lockfile(packages: dict[str, dict], version: int = 3) -> dict:
    """Build the smallest package-lock payload accepted by the resolver."""
    return {"lockfileVersion": version, "packages": packages}


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
    assert set(lockfile["packages"]) == {
        "",
        "node_modules/a",
        "node_modules/a/node_modules/b",
    }
    assert lockfile["packages"]["node_modules/a"]["dependencies"] == {"b": "1.0.0"}
    assert lockfile["packages"]["node_modules/a/node_modules/b"]["dependencies"] == {"a": "1.0.0"}


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


def test_ambiguous_target_without_ancestry_fails_closed() -> None:
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
    assert closure.fallback_reason == "ambiguous_target"


def test_optional_missing_edge_is_omitted_but_required_edge_rejected_by_artifact() -> None:
    optional = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "optionalDependencies": {"platform-only": "1.0.0"},
            },
        },
        target_package="a",
        target_version="1.0.0",
    )
    artifacts = build_sliced_lockfile_artifacts(optional)
    assert (
        json.loads(artifacts["package-lock.json"])["packages"]["node_modules/a"][
            "optionalDependencies"
        ]
        == {}
    )
