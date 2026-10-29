"""Tests for Supervisor-owned advisory plan preparation."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import Mock
from uuid import uuid4

from remediation_engine import cli
from remediation_engine.contracts.schemas import (
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    PackageFixPlanCandidate,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskSpawnRequest,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import SolverStatus
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration import supervisor_node, supervisor_portfolio, supervisor_spawn
from remediation_engine.orchestration.portfolio_orchestrator import (
    build_portfolio_plan,
    prepare_portfolio_inputs,
)
from remediation_engine.orchestration.supervisor_fix_plans import (
    ensure_group_fix_plans,
    ensure_workaround_fix_plan,
    plan_fixes_for_groups,
    prepare_supervisor_fix_plan_state,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from remediation_engine.triage.grouper import group_issues


def _issue(cve_id: str = "CVE-2026-1000") -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        package_name="demo-package",
        ecosystem="npm",
        package_version="1.0.0",
        cve_id=cve_id,
        severity=Severity.HIGH,
        file_path="package-lock.json",
    )


def _localized(issue: VulnerabilityIssue) -> LocalizedIssue:
    return LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=True,
        declaration_type="dependencies",
    )


def _plan(
    status: FixPlanStatus,
    *,
    version: str | None = None,
    snippets: list[str] | None = None,
) -> FixPlan:
    return FixPlan(
        status=status,
        fixed_version=version,
        workaround_snippets=snippets,
        instruction="Apply the confirmed remediation evidence.",
        strategy_used="test",
    )


def _group(*issues: VulnerabilityIssue) -> VulnerabilityGroup:
    return group_issues(
        list(issues),
        localized_sca_issues=[_localized(issue) for issue in issues],
    )[0]


def test_grouping_emits_unplanned_sca_observation_groups() -> None:
    group = _group(_issue())
    assert group.fix_plan is None
    assert group.fix_plan_candidates == []
    assert group.localized_issues[0].manifest_file == "package.json"


def test_offline_portfolio_reports_unplanned_sca_without_creating_tasks(tmp_path) -> None:
    group = _group(_issue())

    prepared_groups, queue, diagnostics = prepare_portfolio_inputs(
        tmp_path,
        [group],
        {},
    )
    plan = build_portfolio_plan(tmp_path, prepared_groups, queue)

    assert queue == {}
    assert prepared_groups[0].fix_plan is None
    assert len(diagnostics) == 1
    assert "missing Supervisor fix plans" in diagnostics[0]
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.clusters
    assert all(not cluster.dispatchable for cluster in plan.clusters)
    assert diagnostics[0] in plan.diagnostics


def test_offline_solve_emits_diagnostic_and_exits_nonzero_for_unplanned_sca(
    monkeypatch, tmp_path
) -> None:
    group = _group(_issue())
    output: dict[str, object] = {}
    monkeypatch.setattr(cli, "_load_solve_inputs", lambda *_args: ([group], {}))
    monkeypatch.setattr(
        cli,
        "_write_json",
        lambda _path, payload, **_kwargs: output.update(payload),
    )

    status = cli._run_solve(
        Namespace(
            input=None,
            groups_input=tmp_path / "groups.json",
            tasks_input=None,
            repo=tmp_path,
            output=tmp_path / "result.json",
        ),
        AppSettings(),
    )

    assert status == 1
    assert output["solver_status"] == SolverStatus.UNKNOWN.value
    assert any("missing Supervisor fix plans" in item for item in output["diagnostics"])
    assert output["dispatchable"] is False
    assert output["task_decisions"] == []


def test_planning_keeps_one_candidate_per_finding() -> None:
    issues = [_issue("CVE-2026-1001"), _issue("CVE-2026-1002")]
    group = _group(*issues)
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    planned = ensure_group_fix_plans(group, lookup=lookup)

    assert lookup.call_count == 2
    assert {candidate.issue_id for candidate in planned.fix_plan_candidates} == {
        issue.id for issue in issues
    }
    assert planned.fix_plan is not None
    assert planned.fix_plan.fixed_version == "1.2.0"
    assert group.fix_plan is None


def test_batch_cache_reuses_equivalent_lookup_and_skips_existing_or_out_of_scope_groups() -> None:
    issue = _issue()
    group = _group(issue)
    duplicate_observation = group.model_copy(update={"group_id": "same-input-other-group"})
    planned_group = group.model_copy(
        update={"fix_plan": _plan(FixPlanStatus.NO_FIX), "fix_plan_candidates": []}
    )
    sast_group = VulnerabilityGroup(
        group_id="sast:test",
        issue_type=IssueType.SAST,
        vulnerable_component="rule",
        representative_issue_id=uuid4(),
        issues=[],
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    result = plan_fixes_for_groups(
        [group, duplicate_observation, planned_group, sast_group],
        lookup=lookup,
    )

    assert lookup.call_count == 1
    assert result[0].fix_plan is not None
    assert result[1].fix_plan is not None
    assert result[2] is planned_group
    assert result[3] is sast_group
    again = plan_fixes_for_groups(result, lookup=lookup)
    assert again[0] is result[0]
    assert lookup.call_count == 1


def test_targeted_workaround_lookup_preserves_version_floor_and_uses_skip_osv() -> None:
    issue = _issue()
    group = _group(issue)
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    group = group.model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [
                PackageFixPlanCandidate(issue_id=issue.id, plan=version_plan),
            ],
        }
    )
    lookup = Mock(
        return_value=_plan(
            FixPlanStatus.WORKAROUND_FOUND,
            snippets=["Disable the affected option."],
        )
    )

    enriched = ensure_workaround_fix_plan(group, lookup=lookup)

    assert lookup.call_count == 1
    assert lookup.call_args.kwargs == {"skip_osv": True}
    assert enriched.fix_plan is not None
    assert enriched.fix_plan.fixed_version == "1.2.0"
    assert any(
        candidate.plan.status == FixPlanStatus.VERSION_FOUND
        for candidate in enriched.fix_plan_candidates
    )
    assert any(
        candidate.plan.status == FixPlanStatus.WORKAROUND_FOUND
        for candidate in enriched.fix_plan_candidates
    )


def test_workaround_no_fix_result_does_not_replace_existing_candidates() -> None:
    issue = _issue()
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    group = _group(issue).model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [PackageFixPlanCandidate(issue_id=issue.id, plan=version_plan)],
        }
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.NO_FIX))

    result = ensure_workaround_fix_plan(group, lookup=lookup)

    assert result is group
    assert len(result.fix_plan_candidates) == 1
    assert result.fix_plan.fixed_version == "1.2.0"


def test_summary_only_workaround_evidence_is_reused_without_another_lookup() -> None:
    issue = _issue()
    group = _group(issue).model_copy(
        update={
            "fix_plan": _plan(
                FixPlanStatus.WORKAROUND_FOUND,
                snippets=["Disable the affected option."],
            ),
            "fix_plan_candidates": [],
        }
    )
    lookup = Mock(side_effect=AssertionError("existing workaround should be reused"))

    result = ensure_workaround_fix_plan(group, lookup=lookup)

    assert result is group
    lookup.assert_not_called()


def test_lookup_and_response_validation_errors_use_typed_no_fix_fallback(caplog) -> None:
    group = _group(_issue())

    failed_call = ensure_group_fix_plans(
        group,
        lookup=Mock(side_effect=RuntimeError("service unavailable")),
    )
    invalid_response = ensure_group_fix_plans(
        group,
        lookup=Mock(return_value={"status": "not-a-plan"}),
    )

    assert failed_call.fix_plan is not None
    assert failed_call.fix_plan.status == FixPlanStatus.NO_FIX
    assert failed_call.fix_plan.strategy_used == "NO_FIX"
    assert failed_call.fix_plan_candidates[0].plan.strategy_used == "planner_error"
    assert invalid_response.fix_plan is not None
    assert invalid_response.fix_plan.status == FixPlanStatus.NO_FIX
    assert "Supervisor fix planning failed" in caplog.text


def test_portfolio_entry_plans_groups_before_preparing_solver_inputs(monkeypatch, tmp_path) -> None:
    group = _group(_issue())
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_fix_plans.plan_fix",
        lookup,
    )

    def stop_before_solver(repo_root, groups, task_queue, **kwargs):
        observed["group"] = groups[0]
        observed["queue"] = task_queue
        raise RuntimeError("test stopped after portfolio preparation boundary")

    monkeypatch.setattr(supervisor_portfolio, "prepare_portfolio_inputs", stop_before_solver)
    result = supervisor_portfolio.run_portfolio_plan_tool(
        {
            "repo_root": str(tmp_path),
            "valid_groups": [group],
            "task_queue": {},
        }
    )

    assert lookup.call_count == 1
    assert observed["group"].fix_plan.status == FixPlanStatus.VERSION_FOUND
    assert result["valid_groups"][0].fix_plan_candidates
    assert result["task_queue"] == {}


def test_direct_supervisor_entry_plans_before_validating_portfolio(monkeypatch, tmp_path) -> None:
    group = _group(_issue())
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_fix_plans.plan_fix",
        lookup,
    )

    result = supervisor_node._run_supervisor_decision_node(
        {
            "repo_root": str(tmp_path),
            "valid_groups": [group],
            "task_queue": {},
            "status": "supervisor_entered",
        }
    )

    assert result["next_routing_step"] == "portfolio"
    assert result["valid_groups"][0].fix_plan_candidates
    lookup.assert_called_once()


def test_accepted_workaround_child_gets_targeted_evidence_before_selection(monkeypatch) -> None:
    first_issue = _issue("CVE-2026-1001")
    second_issue = _issue("CVE-2026-1002")
    group = _group(first_issue, second_issue)
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    workaround_plan = _plan(
        FixPlanStatus.WORKAROUND_FOUND,
        snippets=["Disable the affected option."],
    )
    enriched = group.model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [
                PackageFixPlanCandidate(issue_id=first_issue.id, plan=version_plan),
                PackageFixPlanCandidate(issue_id=second_issue.id, plan=workaround_plan),
            ],
        }
    )
    targeted_lookup = Mock(return_value=enriched)
    monkeypatch.setattr(supervisor_spawn, "ensure_workaround_fix_plan", targeted_lookup)
    parent = RemediationTask(
        task_id="task-parent",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
    )
    request = TaskSpawnRequest(
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the workaround.",
        reason="The update path is exhausted.",
    )
    groups = [group]
    ledger: dict[str, str] = {}

    new_tasks, child_ids = supervisor_spawn._materialize_spawn_requests(
        [request],
        {parent.task_id: parent},
        {group.group_id: group},
        [],
        valid_groups=groups,
        qa_evaluations={
            parent.task_id: QAEvaluation(
                task_id=parent.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="The target remains vulnerable.",
            )
        },
        workaround_lookup_fingerprints=ledger,
    )

    child = new_tasks[child_ids[parent.task_id][0]]
    targeted_lookup.assert_called_once_with(group)
    assert groups[0] is enriched
    assert child.selected_plan_issue_ids == [str(second_issue.id)]
    assert len(ledger) == 1


def test_rejected_or_duplicate_workaround_spawns_do_not_lookup(monkeypatch) -> None:
    issue = _issue()
    group = _group(issue)
    parent = RemediationTask(
        task_id="task-parent",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
    )
    existing_child = RemediationTask(
        task_id="task-child",
        parent_group_id=group.group_id,
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
    )
    request = TaskSpawnRequest(
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the workaround.",
        reason="Retry the same pivot.",
    )
    targeted_lookup = Mock(side_effect=AssertionError("lookup should not run"))
    monkeypatch.setattr(supervisor_spawn, "ensure_workaround_fix_plan", targeted_lookup)

    new_tasks, child_ids = supervisor_spawn._materialize_spawn_requests(
        [request],
        {parent.task_id: parent, existing_child.task_id: existing_child},
        {group.group_id: group},
        [],
        qa_evaluations={
            parent.task_id: QAEvaluation(
                task_id=parent.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="The target remains vulnerable.",
            )
        },
        workaround_lookup_fingerprints={},
    )

    assert new_tasks == {}
    assert child_ids[parent.task_id] == [existing_child.task_id]
    targeted_lookup.assert_not_called()


def test_reopened_task_refresh_preserves_identity_attempt_history_and_revision() -> None:
    issue = _issue()
    group = _group(issue)
    task = build_initial_remediation_task(group, "task-7")
    task = task.model_copy(
        update={
            "task_revision": 4,
            "status": TaskStatus.PENDING,
            "parent_task_id": "task-parent",
            "ancestry_depth": 1,
        }
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    planned_groups, queue, cleared = prepare_supervisor_fix_plan_state(
        [group],
        {task.task_id: task},
        [task.task_id],
        ProjectLanguage.NODEJS,
        lookup=lookup,
    )

    refreshed = queue[task.task_id]
    assert planned_groups[0].fix_plan is not None
    assert refreshed.task_id == task.task_id
    assert refreshed.task_revision == 4
    assert refreshed.parent_task_id == "task-parent"
    assert refreshed.ancestry_depth == 1
    assert refreshed.strategy == RoutingStrategy.VERSION_BUMP
    assert refreshed.selected_version is None
    assert refreshed.selected_plan_issue_ids == [str(issue.id)]
    assert cleared == []
    lookup.assert_called_once()


def test_unmatched_refresh_marker_is_retained_until_a_group_is_available() -> None:
    task = RemediationTask(
        task_id="task-orphaned",
        parent_group_id="sca:manifest:missing",
        strategy=RoutingStrategy.VERSION_BUMP,
    )

    planned_groups, queue, pending = prepare_supervisor_fix_plan_state(
        [],
        {task.task_id: task},
        [task.task_id],
        ProjectLanguage.NODEJS,
    )

    assert planned_groups == []
    assert queue[task.task_id] is not task
    assert pending == [task.task_id]
