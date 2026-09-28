"""Focused deterministic tests for the occurrence-aware portfolio solver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from remediation_engine.contracts.solver_models import (
    SolverCandidateConflict,
    SolverCandidateCutKind,
    SolverCandidateLiteral,
    SolverCandidateRejectionReason,
    SolverCandidateRelation,
    SolverDependencyRequirement,
    SolverEdge,
    SolverEvidenceDomain,
    SolverFindingRequirement,
    SolverPackageOccurrence,
    SolverPeerConstraint,
    SolverStatus,
    SolverSubgraph,
    SolverTarget,
    SolverTaskDecision,
    SolverVersionCandidate,
)
from remediation_engine.orchestration.portfolio_solver import (
    _candidate_metadata,
    _effective_candidate_requirements,
)
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import _trace_solver_inputs, solve_portfolio
from remediation_engine.solver.graph import build_dependency_dag, cluster_packages, schedule_batches
from remediation_engine.solver.subgraph import (
    _child_target,
    expand_candidate_relations,
    extract_solver_subgraph,
)
from remediation_engine.tools.npm_graph import (
    NpmLockfilePackage,
    check_npm_range,
    load_npm_graph_snapshot,
    load_npm_graph_snapshot_from_documents,
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
        vulnerable_occurrence_id="package.json::foo",
        fixed_version="1.2.0",
    )


def _solver_subgraph(
    targets: list[SolverTarget],
    findings: list[SolverFindingRequirement] | tuple[SolverFindingRequirement, ...] = (),
    **kwargs: object,
) -> SolverSubgraph:
    """Build direct solver fixtures with explicit mutation/evidence occurrences."""
    occurrences: dict[str, SolverPackageOccurrence] = {
        target.occurrence_id: SolverPackageOccurrence(
            occurrence_id=target.occurrence_id,
            manifest_path=target.manifest_path,
            package_name=target.target_package_name,
            lockfile_package_key=target.lockfile_package_key,
            installed_version=target.installed_version,
            dependency_type=target.dependency_type,
            is_direct=True,
            ancestry=target.dependency_ancestry,
        )
        for target in targets
    }
    targets_by_id = {target.occurrence_id: target for target in targets}
    for finding in findings:
        if finding.vulnerable_occurrence_id in occurrences:
            continue
        target = targets_by_id.get(finding.target_occurrence_id)
        occurrences[finding.vulnerable_occurrence_id] = SolverPackageOccurrence(
            occurrence_id=finding.vulnerable_occurrence_id,
            manifest_path=target.manifest_path if target else "package.json",
            package_name=finding.vulnerable_package,
            installed_version=None,
            is_direct=False,
        )
    return SolverSubgraph(
        targets=targets,
        occurrences=list(occurrences.values()),
        findings=list(findings),
        **kwargs,
    )


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _forbidden_conflict(
    reason_code: SolverCandidateRejectionReason,
    cut_kind: SolverCandidateCutKind,
    literals: list[tuple[str, str]],
    assignment: dict[str, str],
) -> SolverCandidateConflict:
    return SolverCandidateConflict(
        assignment_digest=_json_digest(dict(sorted(assignment.items()))),
        reason_code=reason_code,
        cut_kind=cut_kind,
        literals=[
            SolverCandidateLiteral(variable_id=variable_id, version=version)
            for variable_id, version in literals
        ],
        evidence_digest=_json_digest({"evidence": "deterministic test fixture"}),
        summary="deterministic resolver rejection",
    )


def _two_target_solver_problem():
    left = _target("task-left", "package.json::left").model_copy(
        update={
            "package_name": "left",
            "target_package_name": "left",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    right = _target("task-right", "package.json::right").model_copy(
        update={
            "package_name": "right",
            "target_package_name": "right",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    return (
        left,
        right,
        _solver_subgraph(targets=[left, right]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="2.0.0"),
            ]
            for target in (left, right)
        },
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


def test_finding_coverage_and_batch_projection_are_occurrence_scoped():
    target = _target()
    first = _finding()
    nested_occurrence_id = "package.json::foo::node_modules/bar/node_modules/foo"
    second = SolverFindingRequirement(
        finding_id=first.finding_id,
        ghsa_id=first.ghsa_id,
        severity=first.severity,
        vulnerable_package=first.vulnerable_package,
        target_occurrence_id=first.target_occurrence_id,
        vulnerable_occurrence_id=nested_occurrence_id,
        fixed_version=first.fixed_version,
    )
    subgraph = _solver_subgraph(targets=[target], findings=[first, second])
    result = solve_portfolio(
        subgraph,
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="1.2.0"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )

    assert result.selected_plan is not None
    selected = result.selected_plan
    expected_coverage_ids = {first.coverage_id, second.coverage_id}
    assert len(expected_coverage_ids) == 2
    assert set(selected.coverage_ids) == expected_coverage_ids
    batches, _edges, diagnostics = cluster_packages(subgraph, selected.task_decisions)
    assert diagnostics == []
    assert batches[0].resolved_finding_ids == [first.finding_id]
    assert batches[0].resolved_coverage_ids == sorted(expected_coverage_ids)
    assert batches[0].unresolved_coverage_ids == []


def test_solver_subgraph_rejects_missing_mutation_evidence():
    with pytest.raises(ValueError, match="mutation target"):
        SolverSubgraph(targets=[_target()], findings=[_finding()])


def test_candidate_metadata_preserves_optional_peer_and_runtime_constraints():
    requirements, engines, os_values, cpu_values, diagnostics = _candidate_metadata(
        "example",
        "2.0.0",
        {
            "dependencies": {"runtime-child": "^1.0.0"},
            "optionalDependencies": {"optional-child": "~2.0.0"},
            "peerDependencies": {"peer-host": ">=3.0.0"},
            "peerDependenciesMeta": {"peer-host": {"optional": True}},
            "engines": {"node": ">=20"},
            "os": ["darwin"],
            "cpu": ["arm64"],
        },
    )

    assert diagnostics == []
    candidate = SolverVersionCandidate(
        version="2.0.0",
        requirements=requirements,
        engines=engines,
        os=os_values,
        cpu=cpu_values,
    )
    by_name = {item.package_name: item for item in candidate.requirements}
    assert (by_name["runtime-child"].kind, by_name["runtime-child"].is_optional) == (
        "dependency",
        False,
    )
    assert (by_name["optional-child"].kind, by_name["optional-child"].is_optional) == (
        "optional_dependency",
        True,
    )
    assert (by_name["peer-host"].kind, by_name["peer-host"].is_optional) == ("peer", True)
    assert candidate.engines == {"node": ">=20"}
    assert candidate.os == ["darwin"]
    assert candidate.cpu == ["arm64"]


def test_candidate_relation_uses_nearest_physical_dependency():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    direct_child = _target("task-child", "package.json::child").model_copy(
        update={
            "package_name": "child",
            "target_package_name": "child",
            "lockfile_package_key": "node_modules/child",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    base = _solver_subgraph(targets=[parent, direct_child])
    nested_key = "node_modules/parent/node_modules/child"
    nested_id = f"package.json::child::{nested_key}"
    nested_child = SolverPackageOccurrence(
        occurrence_id=nested_id,
        manifest_path="package.json",
        package_name="child",
        lockfile_package_key=nested_key,
        installed_version="1.2.0",
        is_direct=False,
        ancestry=["parent", "child"],
    )
    subgraph = SolverSubgraph(
        targets=base.targets,
        occurrences=[*base.occurrences, nested_child],
    )
    candidate = SolverVersionCandidate(
        version="3.0.0",
        requirements=[
            SolverDependencyRequirement(
                package_name="child",
                version_range="^2.0.0",
                kind="dependency",
            ),
            SolverDependencyRequirement(
                package_name="@types/node",
                version_range="^20.0.0",
                kind="peer",
                is_optional=True,
            ),
        ],
    )

    expanded = expand_candidate_relations(
        subgraph,
        {
            parent.occurrence_id: [candidate],
            direct_child.occurrence_id: [SolverVersionCandidate(version="1.0.0")],
        },
    )

    assert expanded.valid
    by_name = {item.package_name: item for item in expanded.candidate_relations}
    assert by_name["child"].target_occurrence_id == nested_id
    assert by_name["child"].target_occurrence_id != direct_child.occurrence_id
    assert by_name["@types/node"].target_occurrence_id is None


def test_candidate_relation_expansion_rejects_invalid_and_missing_requirements():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    subgraph = _solver_subgraph(targets=[parent])
    invalid = expand_candidate_relations(
        subgraph,
        {
            parent.occurrence_id: [
                SolverVersionCandidate(
                    version="1.0.0",
                    requirements=[
                        SolverDependencyRequirement(
                            package_name="child",
                            version_range="not a range",
                            kind="dependency",
                        )
                    ],
                )
            ]
        },
    )
    missing = expand_candidate_relations(subgraph, {})

    assert invalid.valid is False
    assert any("invalid candidate range" in item for item in invalid.diagnostics)
    assert missing.valid is False
    assert any("missing domain" in item for item in missing.diagnostics)


def test_candidate_relations_constrain_joint_version_assignment():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    child = _target("task-child", "package.json::child").model_copy(
        update={
            "package_name": "child",
            "target_package_name": "child",
            "lockfile_package_key": "node_modules/child",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    parent_candidates = [
        SolverVersionCandidate(
            version="1.0.0",
            requirements=[
                SolverDependencyRequirement(
                    package_name="child",
                    version_range="^1.0.0",
                    kind="dependency",
                )
            ],
        ),
        SolverVersionCandidate(
            version="2.0.0",
            requirements=[
                SolverDependencyRequirement(
                    package_name="child",
                    version_range="^2.0.0",
                    kind="dependency",
                )
            ],
        ),
    ]
    child_candidates = [
        SolverVersionCandidate(version="1.5.0"),
        SolverVersionCandidate(version="2.5.0"),
    ]
    domains = {
        parent.occurrence_id: parent_candidates,
        child.occurrence_id: child_candidates,
    }
    subgraph = expand_candidate_relations(
        _solver_subgraph(targets=[parent, child]),
        domains,
    )

    result = solve_portfolio(
        subgraph,
        domains,
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )

    assert result.selected_plan is not None
    selected = {
        decision.target_occurrence_id: decision.selected_version
        for decision in result.selected_plan.task_decisions
    }
    assert (selected[parent.occurrence_id], selected[child.occurrence_id]) in {
        ("1.0.0", "1.5.0"),
        ("2.0.0", "2.5.0"),
    }


def test_forbidden_assignment_rejects_exact_map_and_partial_map():
    left = _target("task-left", "package.json::left").model_copy(
        update={
            "package_name": "left",
            "target_package_name": "left",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    right = _target("task-right", "package.json::right").model_copy(
        update={
            "package_name": "right",
            "target_package_name": "right",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    subgraph = _solver_subgraph(targets=[left, right])
    domains = {
        target.occurrence_id: [
            SolverVersionCandidate(version="1.0.0"),
            SolverVersionCandidate(version="2.0.0"),
        ]
        for target in (left, right)
    }
    settings = AppSettings(solver_top_k=1, solver_num_search_workers=1)
    first = solve_portfolio(subgraph, domains, settings=settings)
    assert first.selected_plan is not None
    rejected = {
        decision.target_occurrence_id: decision.selected_version
        for decision in first.selected_plan.task_decisions
    }

    alternative = solve_portfolio(
        subgraph,
        domains,
        settings=settings,
        forbidden_assignments=[rejected],
    )
    assert alternative.selected_plan is not None
    assert {
        decision.target_occurrence_id: decision.selected_version
        for decision in alternative.selected_plan.task_decisions
    } != rejected

    malformed = solve_portfolio(
        subgraph,
        domains,
        settings=settings,
        forbidden_assignments=[{left.occurrence_id: "1.0.0"}],
    )
    assert malformed.status == SolverStatus.UNKNOWN
    assert malformed.selected_plan is None
    assert any("exactly the eligible occurrence IDs" in item for item in malformed.diagnostics)


def test_top_k_enumeration_preserves_primary_optimal_status():
    target = _target().model_copy(update={"finding_ids": [], "is_finding_backed": False})
    result = solve_portfolio(
        _solver_subgraph(targets=[target]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="2.0.0"),
            ]
        },
        settings=AppSettings(solver_top_k=4, solver_num_search_workers=1),
    )

    assert result.status == SolverStatus.OPTIMAL
    assert len(result.candidate_plans) == 2
    assert result.solver_statistics is not None
    assert result.solver_statistics.status_sequence[-1] == "INFEASIBLE"


def test_candidate_resource_guard_returns_unknown_without_pruning():
    target = _target().model_copy(update={"finding_ids": [], "is_finding_backed": False})
    result = solve_portfolio(
        _solver_subgraph(targets=[target]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="2.0.0"),
            ]
        },
        settings=AppSettings(
            solver_max_candidates_per_target=1,
            solver_num_search_workers=1,
        ),
    )

    assert result.status == SolverStatus.UNKNOWN
    assert result.candidate_catalog_complete is False
    assert result.selected_plan is None
    assert any("no candidates were pruned" in item for item in result.diagnostics)


def test_major_version_candidate_commits_source_migration_instruction():
    target = _target().model_copy(update={"installed_version": "8.5.1"})
    finding = _finding().model_copy(update={"fixed_version": "9.0.0"})
    result = solve_portfolio(
        _solver_subgraph(targets=[target], findings=[finding]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="8.5.1"),
                SolverVersionCandidate(version="9.0.2"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )

    assert result.selected_plan is not None
    decision = result.selected_plan.task_decisions[0]
    assert decision.selected_version == "9.0.2"
    assert decision.installed_version == "8.5.1"
    assert decision.requires_source_migration is True
    assert "foo" in decision.exact_instruction
    assert "8.5.1" in decision.exact_instruction
    assert "9.0.2" in decision.exact_instruction
    assert "Migrate all affected production and test code" in decision.exact_instruction
    invalid_version = "git+https://github.com/example/foo"
    invalid_target = target.model_copy(update={"installed_version": invalid_version})
    invalid_result = solve_portfolio(
        _solver_subgraph(targets=[invalid_target], findings=[finding]),
        {invalid_target.occurrence_id: [SolverVersionCandidate(version="9.0.2")]},
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )
    assert invalid_result.selected_plan is not None
    invalid_decision = invalid_result.selected_plan.task_decisions[0]
    assert invalid_decision.installed_version == invalid_version
    assert invalid_decision.requires_source_migration is True


def test_cp_sat_selects_smallest_security_floor_and_deterministic_alternative():
    subgraph = _solver_subgraph(targets=[_target()], findings=[_finding()])
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
    decision = result.selected_plan.task_decisions[0]
    assert decision.installed_version == "1.0.0"
    assert decision.requires_source_migration is False
    assert decision.exact_instruction is None
    assert len(result.candidate_plans) == 2
    assert result.solver_statistics is not None
    assert result.solver_statistics.solve_calls >= 1
    assert result.solver_statistics.raw_status_name in {"OPTIMAL", "FEASIBLE"}
    assert len(result.solver_statistics.status_sequence) == (result.solver_statistics.solve_calls)
    assert any("CP-SAT raw status" in diagnostic for diagnostic in result.diagnostics)


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
    subgraph = _solver_subgraph(targets=[left, right])
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
    subgraph = _solver_subgraph(
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
        _solver_subgraph(targets=[_target()], findings=[_finding()]),
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
    subgraph = _solver_subgraph(
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
        _solver_subgraph(targets=[target]),
        [SolverTaskDecision(task_id=target.task_id, selected_version="2.0.0")],
    )
    assert diagnostics == ["task 'task-1': terminal task retained as non-dispatchable singleton"]
    assert len(batches) == 1
    assert batches[0].dispatchable is False


def test_missing_target_occurrence_invalidates_the_subgraph():
    snapshot = load_npm_graph_snapshot(Path("/tmp/does-not-exist-remedy-solver"))
    subgraph = extract_solver_subgraph(snapshot, [_target()], [_finding()])
    assert subgraph.valid is False
    assert any(
        "not present in bounded graph evidence" in diagnostic for diagnostic in subgraph.diagnostics
    )


def test_optional_peer_edges_do_not_form_atomic_batches():
    left = _target("task-1", "package.json::left").model_copy(
        update={"package_name": "left", "target_package_name": "left", "finding_ids": []}
    )
    right = _target("task-2", "package.json::right").model_copy(
        update={"package_name": "right", "target_package_name": "right", "finding_ids": []}
    )
    subgraph = _solver_subgraph(
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


def test_scoped_clustering_does_not_use_namespace_as_atomic_evidence():
    left = _target("task-1", "package.json::@angular/common").model_copy(
        update={
            "package_name": "@angular/common",
            "target_package_name": "@angular/common",
            "finding_ids": [],
        }
    )
    right = _target("task-2", "package.json::@angular/core").model_copy(
        update={
            "package_name": "@angular/core",
            "target_package_name": "@angular/core",
            "finding_ids": [],
        }
    )
    decisions = [
        SolverTaskDecision(task_id=left.task_id, selected_version="21.2.17"),
        SolverTaskDecision(task_id=right.task_id, selected_version="21.2.17"),
    ]

    batches, _edges, diagnostics = cluster_packages(
        _solver_subgraph(
            targets=[left, right],
            edges=[
                SolverEdge(
                    source_occurrence_id=left.occurrence_id,
                    target_occurrence_id=right.occurrence_id,
                    edge_kind="scope",
                    is_peer_coupling=True,
                )
            ],
        ),
        decisions,
        scope_coupling=False,
    )

    assert diagnostics == []
    assert {tuple(batch.task_ids) for batch in batches} == {("task-1",), ("task-2",)}


def test_scoped_clustering_keeps_strict_peer_atomic():
    left = _target("task-1", "package.json::@angular/common").model_copy(
        update={
            "package_name": "@angular/common",
            "target_package_name": "@angular/common",
            "finding_ids": [],
        }
    )
    right = _target("task-2", "package.json::@angular/core").model_copy(
        update={
            "package_name": "@angular/core",
            "target_package_name": "@angular/core",
            "finding_ids": [],
        }
    )

    batches, _edges, diagnostics = cluster_packages(
        _solver_subgraph(
            targets=[left, right],
            edges=[
                SolverEdge(
                    source_occurrence_id=left.occurrence_id,
                    target_occurrence_id=right.occurrence_id,
                    edge_kind="peer",
                    version_range="21.2.17",
                )
            ],
        ),
        [
            SolverTaskDecision(task_id=left.task_id, selected_version="21.2.17"),
            SolverTaskDecision(task_id=right.task_id, selected_version="21.2.17"),
        ],
        scope_coupling=False,
    )

    assert diagnostics == []
    assert len(batches) == 1
    assert set(batches[0].task_ids) == {"task-1", "task-2"}


def test_workaround_target_projects_authorized_plan_ids():
    target = _target().model_copy(update={"strategy": "code_workaround"})
    finding = _finding().model_copy(
        update={"workaround_available": True, "workaround_plan_ids": ["fix-123"]}
    )
    result = solve_portfolio(
        _solver_subgraph(targets=[target], findings=[finding]),
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
        _solver_subgraph(targets=[target], findings=[finding]),
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


def test_unproven_workaround_cannot_satisfy_direct_coverage():
    target = _target().model_copy(update={"strategy": "code_workaround"})
    result = solve_portfolio(
        _solver_subgraph(targets=[target], findings=[_finding()]),
        {
            target.occurrence_id: [
                SolverVersionCandidate(version="1.0.0"),
                SolverVersionCandidate(version="1.2.0"),
            ]
        },
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )

    assert result.status == SolverStatus.INFEASIBLE
    assert result.selected_plan is None
    assert result.unresolved_finding_ids == ["GHSA-AAAA-BBBB-CCCC"]


@pytest.mark.parametrize(
    "reason_code",
    [
        SolverCandidateRejectionReason.RUNTIME_ENGINE,
        SolverCandidateRejectionReason.RUNTIME_PLATFORM,
    ],
)
def test_unary_runtime_conflicts_forbid_only_the_named_candidate(reason_code):
    left, right, subgraph, domains = _two_target_solver_problem()
    conflict = _forbidden_conflict(
        reason_code,
        SolverCandidateCutKind.UNARY,
        [(left.occurrence_id, "1.0.0")],
        {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"},
    )

    result = solve_portfolio(
        subgraph,
        domains,
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
        forbidden_conflicts=[conflict],
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.selected_plan is not None
    assert result.selected_plan.selected_candidate_versions[left.occurrence_id] != "1.0.0"


@pytest.mark.parametrize(
    "reason_code",
    [
        SolverCandidateRejectionReason.DEPENDENCY_RANGE,
        SolverCandidateRejectionReason.PEER_CONFLICT,
    ],
)
def test_pair_conflicts_preserve_alternate_compatible_partners(reason_code):
    left, right, subgraph, domains = _two_target_solver_problem()
    conflict = _forbidden_conflict(
        reason_code,
        SolverCandidateCutKind.PAIR,
        [(left.occurrence_id, "1.0.0"), (right.occurrence_id, "1.0.0")],
        {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"},
    )

    result = solve_portfolio(
        subgraph,
        domains,
        settings=AppSettings(solver_top_k=4, solver_num_search_workers=1),
        forbidden_conflicts=[conflict],
    )

    assignments = [candidate.selected_candidate_versions for candidate in result.candidate_plans]
    assert result.status == SolverStatus.OPTIMAL
    assert all(
        not (
            assignment[left.occurrence_id] == "1.0.0" and assignment[right.occurrence_id] == "1.0.0"
        )
        for assignment in assignments
    )
    assert any(
        assignment[left.occurrence_id] == "1.0.0" and assignment[right.occurrence_id] == "2.0.0"
        for assignment in assignments
    )


def test_exact_assignment_conflict_uses_existing_full_map_no_good():
    left, right, subgraph, domains = _two_target_solver_problem()
    rejected = {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"}
    conflict = _forbidden_conflict(
        SolverCandidateRejectionReason.UNCLASSIFIED,
        SolverCandidateCutKind.EXACT_ASSIGNMENT,
        list(rejected.items()),
        rejected,
    )

    result = solve_portfolio(
        subgraph,
        domains,
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
        forbidden_conflicts=[conflict],
    )

    assert result.selected_plan is not None
    assert result.selected_plan.selected_candidate_versions != rejected


@pytest.mark.parametrize(
    "invalid_kind",
    ["stale", "out_of_domain", "duplicate", "partial_exact", "evidence_variable"],
)
def test_invalid_forbidden_conflicts_fail_closed(invalid_kind):
    left, right, subgraph, domains = _two_target_solver_problem()
    assignment = {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"}
    if invalid_kind == "stale":
        conflict = _forbidden_conflict(
            SolverCandidateRejectionReason.RUNTIME_ENGINE,
            SolverCandidateCutKind.UNARY,
            [("stale-occurrence", "1.0.0")],
            assignment,
        )
    elif invalid_kind == "out_of_domain":
        conflict = _forbidden_conflict(
            SolverCandidateRejectionReason.RUNTIME_ENGINE,
            SolverCandidateCutKind.UNARY,
            [(left.occurrence_id, "9.0.0")],
            assignment,
        )
    elif invalid_kind == "duplicate":
        conflict = {
            "assignment_digest": _json_digest(assignment),
            "reason_code": "runtime_engine",
            "cut_kind": "unary",
            "literals": [
                {"variable_id": left.occurrence_id, "version": "1.0.0"},
                {"variable_id": left.occurrence_id, "version": "2.0.0"},
            ],
            "evidence_digest": _json_digest({"evidence": "duplicate"}),
            "summary": "duplicate variable",
        }
    elif invalid_kind == "evidence_variable":
        conflict = _forbidden_conflict(
            SolverCandidateRejectionReason.RUNTIME_ENGINE,
            SolverCandidateCutKind.UNARY,
            [("evidence:child", "1.0.0")],
            assignment,
        )
    else:
        conflict = _forbidden_conflict(
            SolverCandidateRejectionReason.UNCLASSIFIED,
            SolverCandidateCutKind.EXACT_ASSIGNMENT,
            [(left.occurrence_id, "1.0.0")],
            {left.occurrence_id: "1.0.0"},
        )

    result = solve_portfolio(
        subgraph,
        domains,
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
        forbidden_conflicts=[conflict],
    )

    assert result.status == SolverStatus.UNKNOWN
    assert result.selected_plan is None


def test_conflicts_change_input_digest_without_changing_candidate_domain():
    left, right, subgraph, domains = _two_target_solver_problem()
    settings = AppSettings(solver_top_k=1, solver_num_search_workers=1)
    baseline = solve_portfolio(subgraph, domains, settings=settings)
    conflict = _forbidden_conflict(
        SolverCandidateRejectionReason.RUNTIME_PLATFORM,
        SolverCandidateCutKind.UNARY,
        [(left.occurrence_id, "1.0.0")],
        {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"},
    )

    constrained = solve_portfolio(
        subgraph,
        domains,
        settings=settings,
        forbidden_conflicts=[conflict],
    )

    assert constrained.input_digest != baseline.input_digest
    assert constrained.domain_digest == baseline.domain_digest


def test_duplicate_conflicts_are_canonicalized_before_input_hashing():
    left, right, subgraph, domains = _two_target_solver_problem()
    conflict = _forbidden_conflict(
        SolverCandidateRejectionReason.RUNTIME_PLATFORM,
        SolverCandidateCutKind.UNARY,
        [(left.occurrence_id, "1.0.0")],
        {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"},
    )
    settings = AppSettings(solver_top_k=1, solver_num_search_workers=1)

    single = solve_portfolio(subgraph, domains, settings=settings, forbidden_conflicts=[conflict])
    repeated = solve_portfolio(
        subgraph, domains, settings=settings, forbidden_conflicts=[conflict, conflict]
    )

    assert repeated.input_digest == single.input_digest
    assert repeated.domain_digest == single.domain_digest


def test_solver_trace_summary_includes_bounded_rejection_inputs():
    left, right, subgraph, domains = _two_target_solver_problem()
    assignment = {left.occurrence_id: "1.0.0", right.occurrence_id: "1.0.0"}
    conflict = _forbidden_conflict(
        SolverCandidateRejectionReason.RUNTIME_ENGINE,
        SolverCandidateCutKind.UNARY,
        [(left.occurrence_id, "1.0.0")],
        assignment,
    )

    summary = _trace_solver_inputs(
        {
            "subgraph": subgraph,
            "candidate_domains": domains,
            "forbidden_assignments": [assignment],
            "forbidden_conflicts": [conflict],
        }
    )

    assert summary["forbidden_assignment_count"] == 1
    assert summary["forbidden_assignments"][0]["values"] == assignment
    assert summary["forbidden_conflict_count"] == 1
    assert summary["forbidden_conflicts"][0]["evidence_digest"] == conflict.evidence_digest
    assert len(summary["forbidden_conflicts"][0]["literals"]) == 1


def test_optional_dependencies_override_same_name_required_declarations():
    requirements, _engines, _os_values, _cpu_values, diagnostics = _candidate_metadata(
        "parent",
        "2.0.0",
        {
            "dependencies": {"child": "^1.0.0"},
            "optionalDependencies": {"child": "^2.0.0"},
        },
    )

    assert diagnostics == []
    assert len(requirements) == 1
    assert requirements[0].kind == "optional_dependency"
    assert requirements[0].version_range == "^2.0.0"


def test_supported_manifest_override_precedence_is_workspace_scoped():
    snapshot = load_npm_graph_snapshot_from_documents(
        {
            "package.json": json.dumps(
                {
                    "name": "root",
                    "workspaces": ["packages/*"],
                    "overrides": {"child": "^2.0.0"},
                    "resolutions": {"child": "^4.0.0"},
                    "pnpm": {"overrides": {"child": "^3.0.0"}},
                }
            ),
            "packages/app/package.json": json.dumps({"name": "app"}),
        }
    )
    workspace_target = _target().model_copy(
        update={"manifest_path": "packages/app/package.json", "workspace_id": "package.json"}
    )
    requirement = SolverDependencyRequirement(
        package_name="child", version_range="^1.0.0", kind="dependency"
    )

    effective = _effective_candidate_requirements(snapshot, workspace_target, [requirement])

    assert effective[0].version_range == "^2.0.0"
    assert effective[0].is_range_supported is True


@pytest.mark.parametrize(
    "specifier",
    ["npm:other@^1.0.0", "workspace:*", "file:../child", "git+https://example.test/child.git"],
)
def test_unsupported_dependency_specs_remain_unmodeled(specifier: str):
    snapshot = load_npm_graph_snapshot_from_documents({"package.json": json.dumps({"name": "app"})})
    requirement = SolverDependencyRequirement(
        package_name="child", version_range=specifier, kind="dependency"
    )

    effective = _effective_candidate_requirements(snapshot, _target(), [requirement])

    assert effective[0].version_range == specifier
    assert effective[0].is_range_supported is False


def test_nested_override_matches_are_left_unmodeled():
    snapshot = load_npm_graph_snapshot_from_documents(
        {"package.json": json.dumps({"name": "app", "overrides": {"parent": {"child": "^2.0.0"}}})}
    )
    requirement = SolverDependencyRequirement(
        package_name="child", version_range="^1.0.0", kind="dependency"
    )

    effective = _effective_candidate_requirements(snapshot, _target(), [requirement])

    assert effective[0].version_range == "^1.0.0"
    assert effective[0].is_range_supported is False


def test_evidence_domain_contract_is_separate_from_physical_occurrences():
    source = _target("task-source", "package.json::source").model_copy(
        update={
            "package_name": "source",
            "target_package_name": "source",
            "lockfile_package_key": "node_modules/source",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    base = _solver_subgraph(targets=[source])
    evidence_domain = SolverEvidenceDomain(
        variable_id="evidence:source-child",
        package_name="child",
        source_occurrence_id=source.occurrence_id,
        manifest_path=source.manifest_path,
        dependency_kind="dependency",
        candidate_versions=["v2.0.0", "1.0.0"],
    )
    relation = SolverCandidateRelation(
        source_occurrence_id=source.occurrence_id,
        source_candidate_version="2.0.0",
        package_name="child",
        version_range="^1.0.0",
        kind="dependency",
        evidence_variable_id=evidence_domain.variable_id,
    )
    subgraph = SolverSubgraph(
        targets=base.targets,
        occurrences=base.occurrences,
        candidate_relations=[relation],
        evidence_domains=[evidence_domain],
    )

    assert evidence_domain.candidate_versions == ["1.0.0", "2.0.0"]
    assert [item.package_name for item in subgraph.occurrences] == ["source"]
    assert subgraph.candidate_relations[0].target_occurrence_id is None
    with pytest.raises(ValueError, match="exactly one referenced domain"):
        SolverSubgraph(
            targets=base.targets,
            occurrences=base.occurrences,
            candidate_relations=[
                relation.model_copy(update={"evidence_variable_id": "evidence:missing"})
            ],
        )
    child = _target("task-child", "package.json::child").model_copy(
        update={
            "package_name": "child",
            "target_package_name": "child",
            "lockfile_package_key": "node_modules/child",
        }
    )
    base_with_child = _solver_subgraph(targets=[source, child])
    with pytest.raises(ValueError, match="cannot bind a mutation target and evidence variable"):
        SolverSubgraph(
            targets=base_with_child.targets,
            occurrences=base_with_child.occurrences,
            candidate_relations=[
                relation.model_copy(update={"target_occurrence_id": child.occurrence_id})
            ],
            evidence_domains=[evidence_domain],
        )


def test_evidence_witnesses_model_child_ranges_without_creating_child_tasks():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    child_domain = SolverEvidenceDomain(
        variable_id="evidence:parent-child",
        package_name="child",
        source_occurrence_id=parent.occurrence_id,
        manifest_path=parent.manifest_path,
        dependency_kind="dependency",
        candidate_versions=["1.5.0", "2.5.0"],
    )
    candidates = [
        SolverVersionCandidate(
            version="1.0.0",
            requirements=[
                SolverDependencyRequirement(
                    package_name="child", version_range="^1.0.0", kind="dependency"
                )
            ],
        ),
        SolverVersionCandidate(
            version="2.0.0",
            requirements=[
                SolverDependencyRequirement(
                    package_name="child", version_range="^2.0.0", kind="dependency"
                )
            ],
        ),
    ]
    relations = [
        SolverCandidateRelation(
            source_occurrence_id=parent.occurrence_id,
            source_candidate_version=candidate.version,
            package_name="child",
            version_range=candidate.requirements[0].version_range,
            kind="dependency",
            evidence_variable_id=child_domain.variable_id,
        )
        for candidate in candidates
    ]
    subgraph = _solver_subgraph(
        targets=[parent],
        candidate_relations=relations,
        evidence_domains=[child_domain],
    )

    result = solve_portfolio(
        subgraph,
        {parent.occurrence_id: candidates},
        settings=AppSettings(solver_top_k=2, solver_num_search_workers=1),
    )

    assert result.status == SolverStatus.OPTIMAL
    assert len(result.candidate_plans) == 2
    assert {
        plan.selected_candidate_versions[parent.occurrence_id] for plan in result.candidate_plans
    } == {"1.0.0", "2.0.0"}
    assert all(
        set(plan.selected_candidate_versions) == {parent.occurrence_id}
        for plan in result.candidate_plans
    )


def test_complete_empty_evidence_domain_excludes_required_source_candidate():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    child_domain = SolverEvidenceDomain(
        variable_id="evidence:parent-child-empty",
        package_name="child",
        source_occurrence_id=parent.occurrence_id,
        manifest_path=parent.manifest_path,
        dependency_kind="dependency",
        candidate_versions=[],
    )
    candidate = SolverVersionCandidate(
        version="1.0.0",
        requirements=[
            SolverDependencyRequirement(
                package_name="child", version_range="^1.0.0", kind="dependency"
            )
        ],
    )
    relation = SolverCandidateRelation(
        source_occurrence_id=parent.occurrence_id,
        source_candidate_version="1.0.0",
        package_name="child",
        version_range="^1.0.0",
        kind="dependency",
        evidence_variable_id=child_domain.variable_id,
    )
    subgraph = _solver_subgraph(
        targets=[parent],
        candidate_relations=[relation],
        evidence_domains=[child_domain],
    )

    result = solve_portfolio(
        subgraph,
        {parent.occurrence_id: [candidate]},
        settings=AppSettings(solver_top_k=1, solver_num_search_workers=1),
    )

    assert result.status == SolverStatus.INFEASIBLE
    assert result.selected_plan is None


def test_evidence_variable_guard_disables_filter_without_pruning_mutations():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    child_domain = SolverEvidenceDomain(
        variable_id="evidence:parent-child-large",
        package_name="child",
        source_occurrence_id=parent.occurrence_id,
        manifest_path=parent.manifest_path,
        dependency_kind="dependency",
        candidate_versions=["2.0.0"],
    )
    candidates = [
        SolverVersionCandidate(
            version="1.0.0",
            requirements=[
                SolverDependencyRequirement(
                    package_name="child", version_range="^1.0.0", kind="dependency"
                )
            ],
        ),
        SolverVersionCandidate(version="2.0.0"),
    ]
    relation = SolverCandidateRelation(
        source_occurrence_id=parent.occurrence_id,
        source_candidate_version="1.0.0",
        package_name="child",
        version_range="^1.0.0",
        kind="dependency",
        evidence_variable_id=child_domain.variable_id,
    )
    subgraph = _solver_subgraph(
        targets=[parent],
        candidate_relations=[relation],
        evidence_domains=[child_domain],
    )

    result = solve_portfolio(
        subgraph,
        {parent.occurrence_id: candidates},
        settings=AppSettings(
            solver_top_k=1,
            solver_num_search_workers=1,
            solver_max_model_variables=3,
        ),
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.selected_plan is not None
    assert result.selected_plan.selected_candidate_versions[parent.occurrence_id] == "1.0.0"
    assert any("evidence-only model variable guard" in item for item in result.diagnostics)


def test_physical_nonmutable_child_keeps_provenance_with_evidence_witness():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    child_key = "node_modules/parent/node_modules/child"
    child_id = f"package.json::child::{child_key}"
    child = SolverPackageOccurrence(
        occurrence_id=child_id,
        manifest_path="package.json",
        package_name="child",
        lockfile_package_key=child_key,
        installed_version="1.5.0",
        is_direct=False,
    )
    base = _solver_subgraph(targets=[parent])
    subgraph = SolverSubgraph(
        targets=base.targets,
        occurrences=[*base.occurrences, child],
    )
    candidate = SolverVersionCandidate(
        version="2.0.0",
        requirements=[
            SolverDependencyRequirement(
                package_name="child", version_range="^1.0.0", kind="dependency"
            )
        ],
    )
    evidence_domain = SolverEvidenceDomain(
        variable_id="evidence:parent-child-scope",
        package_name="child",
        source_occurrence_id=parent.occurrence_id,
        manifest_path="package.json",
        dependency_kind="dependency",
        candidate_versions=["1.5.0"],
    )

    expanded = expand_candidate_relations(
        subgraph,
        {parent.occurrence_id: [candidate]},
        evidence_domains=[evidence_domain],
    )
    validated = SolverSubgraph.model_validate(expanded.model_dump(mode="json"))

    relation = validated.candidate_relations[0]
    assert relation.target_occurrence_id == child_id
    assert relation.evidence_variable_id == evidence_domain.variable_id
    assert child_id not in {target.occurrence_id for target in validated.targets}


def test_evidence_versions_participate_in_candidate_domain_and_input_digests():
    parent = _target("task-parent", "package.json::parent").model_copy(
        update={
            "package_name": "parent",
            "target_package_name": "parent",
            "lockfile_package_key": "node_modules/parent",
            "finding_ids": [],
            "is_finding_backed": False,
        }
    )
    candidate = SolverVersionCandidate(
        version="1.0.0",
        requirements=[
            SolverDependencyRequirement(
                package_name="child", version_range="^1.0.0", kind="dependency"
            )
        ],
    )
    evidence_domain = SolverEvidenceDomain(
        variable_id="evidence:parent-child-digest",
        package_name="child",
        source_occurrence_id=parent.occurrence_id,
        manifest_path=parent.manifest_path,
        dependency_kind="dependency",
        candidate_versions=["1.5.0"],
    )
    relation = SolverCandidateRelation(
        source_occurrence_id=parent.occurrence_id,
        source_candidate_version="1.0.0",
        package_name="child",
        version_range="^1.0.0",
        kind="dependency",
        evidence_variable_id=evidence_domain.variable_id,
    )
    base = _solver_subgraph(
        targets=[parent],
        candidate_relations=[relation],
        evidence_domains=[evidence_domain],
    )
    settings = AppSettings(solver_top_k=1, solver_num_search_workers=1)
    feasible = solve_portfolio(base, {parent.occurrence_id: [candidate]}, settings=settings)
    changed_domain = evidence_domain.model_copy(update={"candidate_versions": ["2.5.0"]})
    changed_subgraph = base.model_copy(update={"evidence_domains": [changed_domain]})
    infeasible = solve_portfolio(
        changed_subgraph, {parent.occurrence_id: [candidate]}, settings=settings
    )

    assert feasible.selected_plan is not None
    assert infeasible.status == SolverStatus.INFEASIBLE
    assert infeasible.selected_plan is None
    assert feasible.domain_digest != infeasible.domain_digest
    assert feasible.input_digest != infeasible.input_digest
