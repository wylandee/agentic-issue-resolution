"""Deterministic package-portfolio planning and delta-isolation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from remediation_engine.contracts import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    Severity,
    TaskDependencyKind,
    TaskStatus,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.portfolio_orchestrator import (
    build_portfolio_plan,
    isolate_delta_failure,
    materialize_synthetic_dependency_tasks,
)
from remediation_engine.orchestration.qa_test_parsing import parse_peer_conflict_evidence
from remediation_engine.orchestration.state import initial_orchestrator_state
from remediation_engine.orchestration.supervisor_node import run_supervisor_node
from remediation_engine.orchestration.supervisor_routing import _portfolio_cluster_targets
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.triage.grouper import group_issues


def _group(package_name: str, manifest_path: str):
    issue = VulnerabilityIssue(
        source=IssueSource.SYNTHETIC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=package_name,
        package_version="1.0.0",
        file_path=manifest_path,
        cve_id=(
            "CVE-2026-"
            + str(
                int(hashlib.sha256(f"{package_name}:{manifest_path}".encode()).hexdigest()[:8], 16)
            )[:6]
        ),
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest_path,
        package_manager="npm",
        declaration_type="dependencies",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )
    plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="2.0.0",
        instruction=f"Update {package_name}.",
        strategy_used="osv_api",
    )
    return group_issues([issue], sca_issue_plans=[(localized, plan)])[0]


def _write_manifest(root: Path, relative_path: str, payload: dict) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tasks(*groups):
    return {
        f"task-{index}": build_initial_remediation_task(group, f"task-{index}")
        for index, group in enumerate(groups, start=1)
    }


def test_runtime_dependency_order_is_stable(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "root", "dependencies": {"lodash": "1.0.0"}},
    )
    _write_manifest(
        tmp_path,
        "packages/app/package.json",
        {"name": "app", "dependencies": {"lodash": "1.0.0"}},
    )
    lodash = _group("lodash", "package.json")
    app = _group("app", "packages/app/package.json")
    queue = _tasks(lodash, app)

    first = build_portfolio_plan(tmp_path, [lodash, app], queue)
    second = build_portfolio_plan(tmp_path, [lodash, app], queue)

    assert first.model_dump() == second.model_dump()
    assert first.task_order.index("task-1") < first.task_order.index("task-2")
    assert all(len(cluster.task_ids) == 1 for cluster in first.clusters)


def test_workspace_namespace_packages_form_one_atomic_cluster(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "root", "workspaces": ["packages/*"]},
    )
    _write_manifest(tmp_path, "packages/core/package.json", {"name": "@angular/core"})
    _write_manifest(tmp_path, "packages/common/package.json", {"name": "@angular/common"})
    core = _group("@angular/core", "packages/core/package.json")
    common = _group("@angular/common", "packages/common/package.json")
    queue = _tasks(core, common)

    plan = build_portfolio_plan(tmp_path, [core, common], queue)

    assert len(plan.clusters) == 1
    assert set(plan.clusters[0].task_ids) == {"task-1", "task-2"}
    cluster_id, targets = _portfolio_cluster_targets(plan, queue, qa=False)
    assert cluster_id == plan.clusters[0].cluster_id
    assert set(targets) == {"task-1", "task-2"}


def test_oversized_namespace_keeps_finding_tasks_in_first_bounded_cluster(tmp_path: Path):
    package_names = [f"@angular/package-{index}" for index in range(11)]
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {package_name: "1.0.0" for package_name in package_names}},
    )
    finding = _group(package_names[0], "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [finding],
        _tasks(finding),
    )

    plan = build_portfolio_plan(tmp_path, groups, queue)

    assert diagnostics == []
    assert max(len(cluster.task_ids) for cluster in plan.clusters) <= 10
    first_cluster = next(cluster for cluster in plan.clusters if "task-1" in cluster.task_ids)
    assert len(first_cluster.task_ids) == 10
    assert len(first_cluster.task_ids) > 1
    assert any("partitioned into bounded dispatch groups" in item for item in plan.diagnostics)


def test_missing_namespace_dependency_gets_a_coordination_task(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "juice-shop",
            "dependencies": {
                "@angular/common": "^21.2.14",
                "@angular/core": "^21.2.14",
                "@angular/platform-browser": "^21.2.14",
            },
        },
    )
    core = _group("@angular/core", "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        _tasks(core),
    )

    synthetic = [group for group in groups if group.is_synthetic]
    assert diagnostics == []
    assert {group.vulnerable_component for group in synthetic} == {
        "@angular/common",
        "@angular/platform-browser",
    }
    assert all(not group.cve_ids and not group.ghsa_ids for group in synthetic)
    assert all(group.fix_plan and group.fix_plan.fixed_version == "2.0.0" for group in synthetic)
    synthetic_tasks = [task for task in queue.values() if task.is_synthetic]
    assert len(synthetic_tasks) == 2
    assert all(task.selected_version == "2.0.0" for task in synthetic_tasks)

    plan = build_portfolio_plan(tmp_path, groups, queue)

    assert len(plan.clusters) == 1
    assert set(plan.clusters[0].task_ids) == set(queue)


def test_missing_peer_dependency_gets_a_coordination_task_from_lockfile(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "peer-app",
            "dependencies": {
                "hono": "^4.0.0",
                "@hono/node-server": "^1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "peer-app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "peer-app"},
                "node_modules/hono": {
                    "version": "4.0.0",
                    "peerDependencies": {"@hono/node-server": "^1.0.0"},
                },
                "node_modules/@hono/node-server": {"version": "1.0.0"},
            },
        },
    )
    hono = _group("hono", "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [hono],
        _tasks(hono),
    )

    assert diagnostics == []
    assert [group.vulnerable_component for group in groups if group.is_synthetic] == [
        "@hono/node-server"
    ]
    plan = build_portfolio_plan(tmp_path, groups, queue)
    assert len(plan.clusters) == 1
    assert len(plan.clusters[0].task_ids) == 2


def test_unflagged_direct_dependency_gets_a_synthetic_task_without_a_seed(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "flagged-package": "^1.0.0",
                "unflagged-package": "1.2.3",
            },
        },
    )
    flagged = _group("flagged-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [flagged],
        _tasks(flagged),
    )

    synthetic = next(group for group in groups if group.vulnerable_component == "unflagged-package")
    synthetic_task = next(task for task in queue.values() if task.is_synthetic)
    assert diagnostics == []
    assert synthetic.group_id == "sca:package.json:unflagged-package"
    assert synthetic.fix_plan is not None
    assert synthetic.fix_plan.fixed_version == "1.2.3"
    assert synthetic_task.selected_version == "1.2.3"
    assert synthetic_task.parent_group_id == synthetic.group_id


def test_all_exact_direct_dependencies_are_materialized_without_finding_groups(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "first-package": "1.0.0",
                "second-package": "2.0.0",
            },
        },
    )

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [],
        {},
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {
        "first-package",
        "second-package",
    }
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:first-package",
        "sca:package.json:second-package",
    }
    assert all(task.is_synthetic for task in queue.values())


def test_runtime_edges_use_lockfile_package_metadata(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "app-a": "1.0.0",
                "app-b": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "node_modules/app-a": {
                    "version": "1.0.0",
                    "dependencies": {"app-b": "1.0.0"},
                },
                "node_modules/app-b": {"version": "1.0.0"},
            },
        },
    )
    app_a = _group("app-a", "package.json")
    app_b = _group("app-b", "package.json")
    queue = _tasks(app_a, app_b)

    plan = build_portfolio_plan(tmp_path, [app_a, app_b], queue)

    assert plan.task_order.index("task-2") < plan.task_order.index("task-1")
    assert any(
        dependency.upstream_task_id == "task-2"
        and dependency.downstream_task_id == "task-1"
        and dependency.edge_type == TaskDependencyKind.RUNTIME
        for cluster in plan.clusters
        for dependency in cluster.dependencies
    )


def test_runtime_related_synthetic_package_keeps_its_resolved_version(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "app-a": "1.0.0",
                "app-b": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "node_modules/app-a": {
                    "version": "1.0.0",
                    "dependencies": {"app-b": "1.0.0"},
                },
                "node_modules/app-b": {"version": "1.0.0"},
            },
        },
    )
    app_a = _group("app-a", "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [app_a],
        _tasks(app_a),
    )

    app_b = next(group for group in groups if group.vulnerable_component == "app-b")
    assert diagnostics == []
    assert app_b.is_synthetic
    assert app_b.fix_plan is not None
    assert app_b.fix_plan.fixed_version == "1.0.0"
    assert queue["task-2"].selected_version == "1.0.0"


def test_synthetic_dependency_materialization_is_idempotent(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    original_queue = _tasks(core)
    first_groups, first_queue, first_diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        original_queue,
    )
    second_groups, second_queue, second_diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        first_groups,
        first_queue,
    )

    assert first_diagnostics == second_diagnostics == []
    assert [group.group_id for group in second_groups] == [group.group_id for group in first_groups]
    assert list(second_queue) == list(first_queue)
    assert [task.model_dump() for task in second_queue.values()] == [
        task.model_dump() for task in first_queue.values()
    ]


def test_synthetic_target_refresh_keeps_group_identity(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    groups, queue, _diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        _tasks(core),
    )
    core_task = queue["task-1"].model_copy(update={"selected_version": "3.0.0"})
    refreshed_groups, refreshed_queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        groups,
        {**queue, "task-1": core_task},
    )

    synthetic_group = next(group for group in refreshed_groups if group.is_synthetic)
    synthetic_task = next(task for task in refreshed_queue.values() if task.is_synthetic)
    assert diagnostics == []
    assert synthetic_group.group_id == "sca:package.json:@angular/common"
    assert synthetic_group.fix_plan is not None
    assert synthetic_group.fix_plan.fixed_version == "3.0.0"
    assert synthetic_task.selected_version == "3.0.0"
    assert synthetic_task.task_revision == 1


def test_supervisor_adds_synthetic_tasks_before_portfolio_dispatch(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    state = initial_orchestrator_state(str(tmp_path), [core])

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "portfolio"
    synthetic_groups = [group for group in result["valid_groups"] if group.is_synthetic]
    synthetic_tasks = [task for task in result["task_queue"].values() if task.is_synthetic]
    assert [group.vulnerable_component for group in synthetic_groups] == ["@angular/common"]
    assert len(synthetic_tasks) == 1
    assert synthetic_tasks[0].parent_group_id == synthetic_groups[0].group_id


def test_multi_kind_angular_edges_materialize_once_per_task_pair(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "juice-shop",
            "dependencies": {
                "@angular/common": "^21.2.14",
                "@angular/compiler": "^21.2.14",
                "@angular/core": "^21.2.14",
            },
        },
    )
    groups = [
        _group("@angular/common", "package.json"),
        _group("@angular/compiler", "package.json"),
        _group("@angular/core", "package.json"),
    ]

    plan = build_portfolio_plan(tmp_path, groups, _tasks(*groups))

    assert len(plan.clusters) == 1
    dependencies = plan.clusters[0].dependencies
    edge_keys = [
        (dependency.upstream_task_id, dependency.downstream_task_id) for dependency in dependencies
    ]
    assert len(edge_keys) == len(set(edge_keys))
    dependency_by_edge = {
        (dependency.upstream_task_id, dependency.downstream_task_id): dependency.edge_type
        for dependency in dependencies
    }
    assert dependency_by_edge[("task-1", "task-2")] == TaskDependencyKind.WORKSPACE


def test_unrelated_package_groups_remain_singletons(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    left = _group("left-package", "package.json")
    right = _group("right-package", "package.json")
    queue = _tasks(left, right)

    plan = build_portfolio_plan(tmp_path, [left, right], queue)

    assert len(plan.clusters) == 2
    assert {tuple(cluster.task_ids) for cluster in plan.clusters} == {
        ("task-1",),
        ("task-2",),
    }


def test_delta_isolation_identifies_a_unique_failing_package():
    result = isolate_delta_failure(
        ["task-b", "task-a"],
        lambda subset: "FAIL" if "task-a" in subset else "PASS",
    )

    assert result.status == "IDENTIFIED"
    assert result.responsible_task_ids == ("task-a",)
    assert result.executions == 3


def test_delta_isolation_preserves_interaction_and_ambiguity():
    interaction = isolate_delta_failure(
        ["a", "b"], lambda subset: "FAIL" if len(subset) == 2 else "PASS"
    )
    ambiguous = isolate_delta_failure(["a", "b"], lambda _subset: "FAIL")

    assert interaction.status == "INTERACTION_FAILURE"
    assert ambiguous.status == "AMBIGUOUS"
    assert interaction.executions <= 3
    assert ambiguous.executions <= 3


def test_active_cluster_with_internal_dependencies_is_dispatchable(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    first_group = _group("@nestjs/core", "package.json")
    second_group = _group("@nestjs/common", "package.json")
    queue = _tasks(first_group, second_group)
    for task in queue.values():
        task.status = TaskStatus.PENDING

    plan = build_portfolio_plan(tmp_path, [first_group, second_group], queue)
    cluster_id, target_ids = _portfolio_cluster_targets(plan, queue, qa=False)

    assert cluster_id is not None
    assert set(target_ids) == set(queue)


def test_supervisor_preserves_all_cluster_targets_through_commit(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    first_group = _group("@nestjs/core", "package.json")
    second_group = _group("@nestjs/common", "package.json")
    queue = _tasks(first_group, second_group)
    plan = build_portfolio_plan(tmp_path, [first_group, second_group], queue)
    state = initial_orchestrator_state(str(tmp_path), [first_group, second_group])
    state.update(
        {
            "task_queue": queue,
            "portfolio_plan": plan,
            "portfolio_dirty": False,
            "status": "workspace_ready",
        }
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "update_subagent"
    assert result["active_cluster_id"] == plan.clusters[0].cluster_id
    assert set(result["active_target_task_ids"]) == set(queue)
    assert result["active_multi_package_action"] is not None
    assert {
        mutation.task_id for mutation in result["active_multi_package_action"].package_mutations
    } == set(queue)


def test_peer_parser_preserves_quoted_ranges_and_scoped_packages():
    evidence = parse_peer_conflict_evidence(
        "npm error Found: react@18.2.0\n"
        'npm error peer react@"^18.0.0 || ^19.0.0" from @angular/core@17.0.0',
        "",
    )

    assert len(evidence) == 1
    assert evidence[0].peer_package == "react"
    assert evidence[0].requester_package == "@angular/core"
    assert evidence[0].required_range == "^18.0.0 || ^19.0.0"
    assert evidence[0].observed_version == "18.2.0"
