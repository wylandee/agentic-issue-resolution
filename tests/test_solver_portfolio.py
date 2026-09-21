"""Focused deterministic tests for the occurrence-aware portfolio solver."""

from __future__ import annotations

from pathlib import Path

import pytest

from remediation_engine.contracts.solver_models import (
    SolverEdge,
    SolverFindingRequirement,
    SolverPeerConstraint,
    SolverStatus,
    SolverSubgraph,
    SolverTarget,
    SolverTaskDecision,
    SolverVersionCandidate,
)
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import solve_portfolio
from remediation_engine.solver.graph import build_dependency_dag, cluster_packages, schedule_batches
from remediation_engine.solver.subgraph import _child_target, extract_solver_subgraph
from remediation_engine.tools.npm_graph import (
    NpmLockfilePackage,
    check_npm_range,
    load_npm_graph_snapshot,
)


def _target(task_id: str = "task-1", occurrence_id: str = "package.json::foo") -> SolverTarget:
    return SolverTarget(
        occurrence_id=occurrence_id,
        task_id=task_id,
        group_id=f"group-{task_id}",
        package_name="foo",
        target_package_name="foo",
        manifest_path="package.json",
        lockfile_package_key="node_modules/foo",
        installed_version="1.0.0",
        finding_ids=["GHSA-AAAA-BBBB-CCCC"],
    )


def _finding() -> SolverFindingRequirement:
    return SolverFindingRequirement(
        finding_id="GHSA-AAAA-BBBB-CCCC",
        ghsa_id="GHSA-AAAA-BBBB-CCCC",
        severity="HIGH",
        vulnerable_package="foo",
        target_occurrence_id="package.json::foo",
        fixed_version="1.2.0",
    )


def test_npm_range_accepts_operator_whitespace():
    result = check_npm_range(">= 21.0.0 < 22.0.0", "21.2.12")

    assert result.matches is True
    assert result.diagnostic is None


def test_nested_lock_dependency_is_not_bound_to_direct_target():
    source = _target("task-1", "package.json::download").model_copy(
        update={
            "group_id": "group-download",
            "package_name": "download",
            "target_package_name": "download",
            "lockfile_package_key": "node_modules/download",
            "installed_version": "8.0.0",
        }
    )
    direct_child = _target("task-2", "package.json::file-type").model_copy(
        update={
            "group_id": "group-file-type",
            "package_name": "file-type",
            "target_package_name": "file-type",
            "lockfile_package_key": "node_modules/file-type",
            "installed_version": "16.5.4",
        }
    )
    nested_key = "node_modules/download/node_modules/file-type"
    lock_packages = {
        ("package.json", "node_modules/download"): NpmLockfilePackage(
            lockfile_path="package-lock.json",
            manifest_path="package.json",
            package_key="node_modules/download",
            package_name="download",
            metadata={"version": "8.0.0", "dependencies": {"file-type": "^11.1.0"}},
        ),
        ("package.json", nested_key): NpmLockfilePackage(
            lockfile_path="package-lock.json",
            manifest_path="package.json",
            package_key=nested_key,
            package_name="file-type",
            metadata={"version": "11.1.0"},
        ),
    }

    assert (
        _child_target(
            source,
            "file-type",
            {source.occurrence_id: source, direct_child.occurrence_id: direct_child},
            lock_packages,
        )
        is None
    )


def test_subgraph_retains_ghsa_only_finding_and_rejects_duplicate_targets():
    snapshot = load_npm_graph_snapshot(Path("/tmp/does-not-exist-remedy-solver"))
    subgraph = extract_solver_subgraph(snapshot, [_target()], [_finding()])
    assert [finding.finding_id for finding in subgraph.findings] == ["GHSA-AAAA-BBBB-CCCC"]
    with pytest.raises(ValueError, match="conflicting records"):
        extract_solver_subgraph(
            snapshot,
            [
                _target(),
                _target(occurrence_id="package.json::foo").model_copy(update={"task_id": "task-2"}),
            ],
            [_finding()],
        )


def test_cp_sat_selects_smallest_security_floor_and_deterministic_alternative():
    subgraph = SolverSubgraph(targets=[_target()], findings=[_finding()])
    candidates = {
        "package.json::foo": [
            SolverVersionCandidate(version="1.0.0", source="current"),
            SolverVersionCandidate(version="1.2.0", source="fixed"),
            SolverVersionCandidate(version="1.3.0", source="registry"),
        ]
    }
    result = solve_portfolio(
        subgraph,
        candidates,
        settings=AppSettings(solver_top_k=2, solver_num_search_workers=1),
    )
    assert result.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
    assert result.selected_plan is not None
    assert result.selected_plan.task_decisions[0].selected_version == "1.2.0"
    assert len(result.candidate_plans) == 2


def test_forced_singleton_and_phase_budget_are_preserved():
    left = _target("task-1", "package.json::left").model_copy(
        update={"package_name": "left", "target_package_name": "left", "finding_ids": []}
    )
    right = _target("task-2", "package.json::right").model_copy(
        update={"package_name": "right", "target_package_name": "right", "finding_ids": []}
    )
    decisions = [
        SolverTaskDecision(task_id="task-1", selected_version="2.0.0"),
        SolverTaskDecision(task_id="task-2", selected_version="2.0.0"),
    ]
    subgraph = SolverSubgraph(targets=[left, right])
    batches, edges, diagnostics = cluster_packages(
        subgraph,
        decisions,
        forced_singleton_task_ids=["task-1"],
        peer_conflict_pairs=(),
    )
    assert diagnostics == []
    assert all(len(batch.task_ids) == 1 for batch in batches)
    dag = build_dependency_dag(subgraph, batches, edges)
    phases, phase_diagnostics = schedule_batches(dag, phase_budget=1)
    assert phase_diagnostics == []
    assert len(phases) == 2


def test_dependency_dag_rejects_unknown_occurrence_edges():
    target = _target()
    subgraph = SolverSubgraph(
        targets=[target],
        edges=[
            SolverEdge(
                source_occurrence_id="package.json::missing",
                target_occurrence_id=target.occurrence_id,
                edge_kind="runtime",
            )
        ],
    )
    batches, _, _ = cluster_packages(
        subgraph,
        [SolverTaskDecision(task_id=target.task_id, selected_version="2.0.0")],
        forced_singleton_task_ids=(),
        peer_conflict_pairs=(),
    )
    dag = build_dependency_dag(subgraph, batches, [])
    assert dag.valid is False
    assert any("unknown" in diagnostic for diagnostic in dag.diagnostics)


def test_allowed_alternatives_respect_the_target_security_floor():
    result = solve_portfolio(
        SolverSubgraph(targets=[_target()], findings=[_finding()]),
        {
            "package.json::foo": [
                SolverVersionCandidate(version="1.0.0", source="current"),
                SolverVersionCandidate(version="1.2.0", source="fixed"),
                SolverVersionCandidate(version="1.3.0", source="registry"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert result.selected_plan is not None
    decision = result.selected_plan.task_decisions[0]
    assert decision.selected_version == "1.2.0"
    assert decision.allowed_alternative_versions == ["1.3.0"]


def test_optional_peer_constraints_do_not_make_the_model_infeasible():
    left = _target("task-1", "package.json::left").model_copy(
        update={"package_name": "left", "target_package_name": "left", "finding_ids": []}
    )
    right = _target("task-2", "package.json::right").model_copy(
        update={"package_name": "right", "target_package_name": "right", "finding_ids": []}
    )
    subgraph = SolverSubgraph(
        targets=[left, right],
        peer_constraints=[
            SolverPeerConstraint(
                source_occurrence_id=left.occurrence_id,
                target_occurrence_id=right.occurrence_id,
                version_range="<1.0.0",
                is_strict=False,
                is_optional=True,
            )
        ],
    )
    result = solve_portfolio(
        subgraph,
        {
            left.occurrence_id: [SolverVersionCandidate(version="1.0.0")],
            right.occurrence_id: [SolverVersionCandidate(version="1.0.0")],
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert result.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
    assert result.selected_plan is not None


def test_unknown_explicit_peer_conflict_invalidates_the_subgraph():
    snapshot = load_npm_graph_snapshot(Path("/tmp/does-not-exist-remedy-solver"))
    subgraph = extract_solver_subgraph(
        snapshot,
        [_target()],
        [],
        peer_conflict_pairs=[("task-1", "task-missing")],
    )
    assert subgraph.valid is False
    assert any("external or unknown" in diagnostic for diagnostic in subgraph.diagnostics)


def test_terminal_target_is_retained_but_not_dispatchable():
    target = _target().model_copy(update={"is_terminal": True})
    batches, _edges, diagnostics = cluster_packages(
        SolverSubgraph(targets=[target]),
        [SolverTaskDecision(task_id=target.task_id, selected_version="2.0.0")],
    )
    assert diagnostics == ["task 'task-1': terminal task retained as non-dispatchable singleton"]
    assert len(batches) == 1
    assert batches[0].dispatchable is False


def test_missing_target_occurrence_invalidates_the_subgraph():
    snapshot = load_npm_graph_snapshot(Path("/tmp/does-not-exist-remedy-solver"))
    subgraph = extract_solver_subgraph(snapshot, [_target()], [_finding()])
    assert subgraph.valid is False
    assert any("not present in graph snapshot" in diagnostic for diagnostic in subgraph.diagnostics)


def test_optional_peer_edges_do_not_form_atomic_batches():
    left = _target("task-1", "package.json::left").model_copy(
        update={"package_name": "left", "target_package_name": "left", "finding_ids": []}
    )
    right = _target("task-2", "package.json::right").model_copy(
        update={"package_name": "right", "target_package_name": "right", "finding_ids": []}
    )
    subgraph = SolverSubgraph(
        targets=[left, right],
        edges=[
            SolverEdge(
                source_occurrence_id=left.occurrence_id,
                target_occurrence_id=right.occurrence_id,
                edge_kind="peer",
                is_optional=True,
            )
        ],
    )
    batches, _edges, diagnostics = cluster_packages(
        subgraph,
        [
            SolverTaskDecision(task_id=left.task_id, selected_version="2.0.0"),
            SolverTaskDecision(task_id=right.task_id, selected_version="2.0.0"),
        ],
    )
    assert diagnostics == []
    assert {tuple(batch.task_ids) for batch in batches} == {("task-1",), ("task-2",)}


def test_workaround_target_projects_authorized_plan_ids():
    target = _target().model_copy(update={"strategy": "code_workaround"})
    finding = _finding().model_copy(
        update={"workaround_available": True, "workaround_plan_ids": ["fix-123"]}
    )
    result = solve_portfolio(
        SolverSubgraph(targets=[target], findings=[finding]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="1.2.0"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert result.selected_plan is not None
    decision = result.selected_plan.task_decisions[0]
    assert decision.selected_strategy == "code_workaround"
    assert decision.selected_version is None
    assert decision.selected_plan_issue_ids == ["fix-123"]


def test_workaround_falls_back_when_candidates_miss_security_floor():
    finding = _finding().model_copy(
        update={
            "fixed_version": "2.0.0",
            "workaround_available": True,
            "workaround_plan_ids": ["fix-456"],
        }
    )
    target = _target()
    result = solve_portfolio(
        SolverSubgraph(targets=[target], findings=[finding]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="1.5.0"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert result.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
    assert result.selected_plan is not None
    decision = result.selected_plan.task_decisions[0]
    assert decision.selected_strategy == "code_workaround"
    assert decision.selected_plan_issue_ids == ["fix-456"]


def test_unproven_workaround_target_is_no_fix():
    target = _target().model_copy(update={"strategy": "code_workaround"})
    result = solve_portfolio(
        SolverSubgraph(targets=[target], findings=[_finding()]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="1.2.0"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert result.selected_plan is not None
    decision = result.selected_plan.task_decisions[0]
    assert decision.selected_strategy == "no_fix"
    assert decision.selected_plan_issue_ids == []
