"""Offline tests for occurrence-aware npm dependency closure resolution."""

from __future__ import annotations

import pytest

from remediation_engine.tools.npm import (
    resolve_dependency_closure_for_target,
    resolve_target_packages,
)


def _closure(
    packages: dict[str, dict],
    *,
    target_package: str,
    target_version: str | None = None,
    dependency_ancestry: tuple[str, ...] = (),
    include_optional: bool = True,
    include_peer: bool = True,
    npm_overrides: dict[str, str] | None = None,
):
    resolutions, reason = resolve_target_packages(
        packages,
        target_package=target_package,
        target_version=target_version,
        dependency_ancestry=dependency_ancestry,
    )
    assert reason is None
    assert len(resolutions) == 1
    return resolve_dependency_closure_for_target(
        packages,
        resolutions[0],
        include_optional=include_optional,
        include_peer=include_peer,
        npm_overrides=npm_overrides,
    )


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

    closure = _closure(
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


def test_each_physical_target_gets_its_own_closure() -> None:
    packages = {
        "": {},
        "node_modules/a": {"version": "1.0.0", "dependencies": {"dep": "1.0.0"}},
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


def test_cycles_resolve_and_missing_required_edges_are_diagnostic() -> None:
    cycle = _closure(
        {
            "": {},
            "node_modules/a": {"version": "1.0.0", "dependencies": {"b": "1.0.0"}},
            "node_modules/a/node_modules/b": {
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
        },
        target_package="a",
        target_version="1.0.0",
    )
    assert cycle.complete
    assert {node.lockfile_key for node in cycle.nodes} == {
        "node_modules/a",
        "node_modules/a/node_modules/b",
    }

    missing = _closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"missing": "^1.0.0"},
            },
            "node_modules/a/node_modules/missing": {"version": "0.9.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )
    assert not missing.complete
    assert missing.fallback_reason == "incomplete_closure"
    assert "dependencies edge from a@1.0.0 at node_modules/a" in (missing.failure_detail or "")
    assert "missing@^1.0.0" in (missing.failure_detail or "")
    assert "node_modules/a/node_modules/missing@0.9.0" in (missing.failure_detail or "")


def test_npm_alias_closure_preserves_physical_alias_identity() -> None:
    closure = _closure(
        {
            "": {},
            "node_modules/@isaacs/cliui": {
                "name": "@isaacs/cliui",
                "version": "8.0.2",
                "dependencies": {
                    "string-width": "^5.1.2",
                    "string-width-cjs": "npm:string-width@^4.2.0",
                    "@test/alias": "npm:@test/real@^2.0.0",
                },
            },
            "node_modules/string-width": {"name": "string-width", "version": "5.1.2"},
            "node_modules/string-width-cjs": {"name": "string-width", "version": "4.2.3"},
            "node_modules/@test/alias": {"name": "@test/real", "version": "2.1.0"},
        },
        target_package="@isaacs/cliui",
        target_version="8.0.2",
    )

    assert closure.complete
    assert {
        (node.lockfile_key, node.package_name, node.metadata.get("name"), node.version)
        for node in closure.nodes
    } == {
        ("node_modules/@isaacs/cliui", "@isaacs/cliui", "@isaacs/cliui", "8.0.2"),
        ("node_modules/string-width", "string-width", "string-width", "5.1.2"),
        ("node_modules/string-width-cjs", "string-width-cjs", "string-width", "4.2.3"),
        ("node_modules/@test/alias", "@test/alias", "@test/real", "2.1.0"),
    }


@pytest.mark.parametrize(
    ("requirement", "metadata_name"),
    [
        ("npm:string-width@^5.0.0", "string-width"),
        ("npm:string-width@^4.2.0", "not-string-width"),
    ],
)
def test_npm_alias_closure_rejects_mismatched_name_or_range(
    requirement: str,
    metadata_name: str,
) -> None:
    closure = _closure(
        {
            "": {},
            "node_modules/parent": {
                "version": "1.0.0",
                "dependencies": {"string-width-cjs": requirement},
            },
            "node_modules/parent/node_modules/string-width-cjs": {
                "name": metadata_name,
                "version": "4.2.3",
            },
        },
        target_package="parent",
        target_version="1.0.0",
    )
    assert not closure.complete
    assert closure.fallback_reason == "incomplete_closure"
    assert f"string-width-cjs@{requirement}" in (closure.failure_detail or "")


@pytest.mark.parametrize(
    ("parent_package", "parent_version", "peer_package", "peer_range", "peer_version"),
    [
        ("@listr2/prompt-adapter-inquirer", "3.0.5", "@inquirer/prompts", ">= 3 < 8", "7.10.1"),
        ("update-browserslist-db", "1.2.3", "browserslist", ">= 4.21.0", "4.28.2"),
    ],
)
def test_peer_closure_accepts_npm_comparator_whitespace(
    parent_package: str,
    parent_version: str,
    peer_package: str,
    peer_range: str,
    peer_version: str,
) -> None:
    closure = _closure(
        {
            "": {},
            f"node_modules/{parent_package}": {
                "version": parent_version,
                "peerDependencies": {peer_package: peer_range},
            },
            f"node_modules/{peer_package}": {"version": peer_version},
        },
        target_package=parent_package,
        target_version=parent_version,
    )
    assert closure.complete
    assert {node.lockfile_key: node.version for node in closure.nodes} == {
        f"node_modules/{parent_package}": parent_version,
        f"node_modules/{peer_package}": peer_version,
    }


def test_npm_overrides_resolve_the_locked_version() -> None:
    packages = {
        "": {
            "name": "frontend",
            "version": "20.0.0",
            "dependencies": {"jsdom": "29.1.1"},
        },
        "node_modules/jsdom": {
            "version": "29.1.1",
            "dependencies": {"lru-cache": "^11.3.5"},
        },
        "node_modules/jsdom/node_modules/lru-cache": {"version": "11.2.7"},
        "node_modules/lru-cache": {"version": "5.1.1"},
    }
    without_override = _closure(
        packages,
        target_package="jsdom",
        target_version="29.1.1",
    )
    with_override = _closure(
        packages,
        target_package="jsdom",
        target_version="29.1.1",
        npm_overrides={"lru-cache@^11": "~11.2.7"},
    )

    assert not without_override.complete
    assert "lru-cache@^11.3.5" in (without_override.failure_detail or "")
    assert with_override.complete
    assert {node.lockfile_key: node.version for node in with_override.nodes} == {
        "node_modules/jsdom": "29.1.1",
        "node_modules/jsdom/node_modules/lru-cache": "11.2.7",
    }


def test_root_closure_includes_project_dev_dependencies() -> None:
    closure = _closure(
        {
            "": {
                "name": "app",
                "version": "1.0.0",
                "dependencies": {"dep": "1.0.0"},
                "devDependencies": {"dev-tool": "^4.0.0"},
            },
            "node_modules/dep": {"version": "1.0.0"},
            "node_modules/dev-tool": {"version": "4.2.1"},
        },
        target_package="app",
        target_version="1.0.0",
    )

    assert closure.complete
    assert set(closure.root_keys) == {""}
    assert {node.package_name for node in closure.nodes} == {"app", "dep", "dev-tool"}


def test_optional_missing_edges_are_omitted_and_present_peer_metadata_is_retained() -> None:
    closure = _closure(
        {
            "": {},
            "node_modules/a": {
                "version": "1.0.0",
                "optionalDependencies": {"missing-optional": "1.0.0"},
                "peerDependencies": {"peer": "^2.0.0", "optional-peer": "1.0.0"},
                "peerDependenciesMeta": {"optional-peer": {"optional": True}},
            },
            "node_modules/peer": {"version": "2.1.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )

    assert closure.complete
    assert {node.package_name for node in closure.nodes} == {"a", "peer"}
    root = next(node for node in closure.nodes if node.package_name == "a")
    assert root.metadata["peerDependenciesMeta"] == {"optional-peer": {"optional": True}}


def test_target_resolution_returns_all_occurrences_or_no_match_reason() -> None:
    packages = {
        "": {},
        "node_modules/a": {"version": "1.0.0"},
        "node_modules/parent/node_modules/a": {"version": "1.0.0"},
    }
    resolutions, reason = resolve_target_packages(
        packages,
        target_package="a",
        target_version="1.0.0",
    )
    assert reason is None
    assert {item.target_node.lockfile_key for item in resolutions if item.target_node} == {
        "node_modules/a",
        "node_modules/parent/node_modules/a",
    }

    missing, reason = resolve_target_packages(
        packages,
        target_package="a",
        target_version="2.0.0",
    )
    assert missing == []
    assert reason == "no_matching_target"
