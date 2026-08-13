"""Focused contracts for deterministic Supervisor routing."""

from __future__ import annotations

from uuid import uuid4

from remediation_engine.contracts import (
    DecisionCode,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LLMAdvisory,
    NoFixMitigationStage,
    PlannerAdvice,
    PlannerBatchAdvice,
    RegistryCandidate,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    is_version_space_exhausted,
    select_version,
    validate_transition,
)
from remediation_engine.orchestration.supervisor_node import (
    _calculate_eligible_actions,
    _commit_task_transition,
    _convert_planner_advice_to_retry_plan,
    _deterministic_routing,
    _emit_audit,
    _merge_advisory,
    _select_deterministic_action,
    _task_sort_key,
    supervisor_router,
)


def _group(group_id: str, severity: Severity = Severity.HIGH, *, no_fix: bool = False):
    issue = VulnerabilityIssue(
        id=str(uuid4()),
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=severity,
        message="test finding",
    )
    fix_plan = FixPlan(
        status=FixPlanStatus.NO_FIX if no_fix else FixPlanStatus.VERSION_FOUND,
        fixed_version=None if no_fix else "1.2.3",
        instruction="remove package" if no_fix else "update package",
        strategy_used="test",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="test-package",
        file_path="package.json",
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=fix_plan,
    )


def _task(
    task_id: str,
    group_id: str,
    *,
    strategy: RoutingStrategy = RoutingStrategy.VERSION_BUMP,
    status: TaskStatus = TaskStatus.PENDING,
    retry_count: int = 0,
    no_fix_stage: NoFixMitigationStage | None = None,
) -> RemediationTask:
    return RemediationTask(
        task_id=task_id,
        parent_group_id=group_id,
        strategy=strategy,
        status=status,
        retry_count=retry_count,
        no_fix_stage=no_fix_stage,
        instruction="update package" if strategy == RoutingStrategy.VERSION_BUMP else "work around",
    )


def _route(tasks, groups, **kwargs):
    qa_evaluations = kwargs.pop("qa_evaluations", {})
    retry_diagnostics = kwargs.pop("retry_diagnostics_by_task", {})
    return _deterministic_routing(
        tasks,
        {group.group_id: group for group in groups},
        qa_evaluations,
        retry_diagnostics,
        **kwargs,
    )


def test_decision_codes_cover_fixed_priority_routes():
    assert _route({}, [], triage_required=False).decision_code == DecisionCode.NO_VALID_GROUPS
    assert (
        _route({}, [], current_status="qa_completed", triage_required=True).decision_code
        == DecisionCode.TRIAGE_REQUIRED
    )
    assert (
        _route({"t": _task("t", "g", status=TaskStatus.QA_PASSED)}, [_group("g")]).decision_code
        == DecisionCode.NO_ACTIONABLE_TASKS
    )
    assert (
        _route(
            {"t": _task("t", "g", status=TaskStatus.OPTIMISTICALLY_FIXED)}, [_group("g")]
        ).decision_code
        == DecisionCode.QA_READY_BATCH
    )
    assert (
        _route(
            {
                "t": _task(
                    "t",
                    "g",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
                )
            },
            [_group("g", no_fix=True)],
        ).decision_code
        == DecisionCode.NO_FIX_LIFECYCLE
    )
    assert (
        _route(
            {"t": _task("t", "g", status=TaskStatus.NEEDS_RETRY)},
            [_group("g")],
            retry_diagnostics_by_task={
                "t": UpdateRetryDiagnostics(task_id="t", exhausted_update_path=True)
            },
        ).decision_code
        == DecisionCode.EXHAUSTED_UPDATE_PIVOT
    )
    assert (
        _route({"t": _task("t", "g", status=TaskStatus.NEEDS_RETRY)}, [_group("g")]).decision_code
        == DecisionCode.RETRY_VERSION_BUMP
    )
    assert (
        _route({"t": _task("t", "g")}, [_group("g")]).decision_code == DecisionCode.NEW_VERSION_BUMP
    )
    assert (
        _route(
            {"t": _task("t", "g", strategy=RoutingStrategy.CODE_WORKAROUND)}, [_group("g")]
        ).decision_code
        == DecisionCode.WORKAROUND_DISPATCH
    )


def test_stable_sort_uses_severity_then_task_id():
    groups = [_group("low", Severity.LOW), _group("critical", Severity.CRITICAL)]
    tasks = {
        "task-b": _task("task-b", "low"),
        "task-a": _task("task-a", "critical"),
    }
    decision = _route(tasks, groups)
    assert decision.target_task_ids == ["task-a", "task-b"]
    assert _task_sort_key(tasks["task-a"], {g.group_id: g for g in groups}) < _task_sort_key(
        tasks["task-b"], {g.group_id: g for g in groups}
    )


def test_same_state_same_decision_is_replayable():
    groups = [_group("g1"), _group("g2")]
    tasks = {"task-2": _task("task-2", "g2"), "task-1": _task("task-1", "g1")}
    decisions = [_route(tasks, groups).model_dump(mode="json") for _ in range(10)]
    assert all(decision == decisions[0] for decision in decisions)


def test_transition_table_accepts_worker_and_qa_edges_and_rejects_terminal_edges():
    assert validate_transition(TaskStatus.PENDING, TaskStatus.OPTIMISTICALLY_FIXED)
    assert validate_transition(TaskStatus.OPTIMISTICALLY_FIXED, TaskStatus.INCONCLUSIVE)
    assert not validate_transition(TaskStatus.QA_PASSED, TaskStatus.PENDING)
    assert not validate_transition(TaskStatus.PENDING, TaskStatus.QA_PASSED)


def test_commit_rejects_invalid_transition_without_mutation():
    task = _task("task-1", "g", status=TaskStatus.QA_PASSED)
    queue = {task.task_id: task}
    events = []
    _commit_task_transition(
        queue,
        task.task_id,
        updates={"status": TaskStatus.PENDING},
        consistency_events=events,
    )
    assert queue[task.task_id] == task
    assert events and events[0].error_code == "INVALID_TRANSITION"
    assert events[0].action == "rejected"


def test_supervisor_router_recomputes_invalid_route():
    group = _group("g")
    state = {
        "next_routing_step": "garbage",
        "valid_groups": [group],
        "task_queue": {"task-1": _task("task-1", "g")},
        "qa_evaluations": {},
        "retry_diagnostics_by_task": {},
        "status": "supervisor_entered",
    }
    assert supervisor_router(state) == "update_subagent"


def test_advisory_cannot_change_authoritative_routing():
    group = _group("g")
    deterministic = _route({"task-1": _task("task-1", "g")}, [group])
    merged = _merge_advisory(
        deterministic,
        LLMAdvisory(
            reasoning="advisory context",
            feedback_by_task={"task-1": "retry with evidence", "other": "discard"},
            new_constraints=["keep package pinned"],
        ),
    )
    assert merged.next_node == deterministic.next_node
    assert merged.target_task_ids == deterministic.target_task_ids
    assert merged.decision_code == deterministic.decision_code
    assert merged.feedback_by_task == {"task-1": "retry with evidence"}
    assert merged.new_constraints == ["keep package pinned"]


def test_version_policy_is_deterministic_and_skips_attempts():
    candidates = [
        RegistryCandidate(
            version="1.2.3",
            semver_key=(1, 2, 3),
            security_floor_met=True,
            is_stable=True,
            same_major=True,
            already_attempted=False,
        ),
        RegistryCandidate(
            version="1.4.0",
            semver_key=(1, 4, 0),
            security_floor_met=True,
            is_stable=True,
            same_major=True,
            already_attempted=False,
        ),
        RegistryCandidate(
            version="2.0.0",
            semver_key=(2, 0, 0),
            security_floor_met=True,
            is_stable=True,
            same_major=False,
            already_attempted=False,
        ),
    ]
    assert select_version(candidates, SCARemediationStage.OSV_MINIMUM, set()) == "1.2.3"
    assert select_version(candidates, SCARemediationStage.NPM_SAME_MAJOR, set()) == "1.4.0"
    assert select_version(candidates, SCARemediationStage.NPM_LATEST, {"2.0.0"}) == "1.4.0"
    assert is_version_space_exhausted(candidates, SCARemediationStage.CODE_WORKAROUND, set())


def test_planner_advice_is_typed_and_stage_regression_is_blocked(monkeypatch):
    group = _group("g")
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY).model_copy(
        update={"strategy_stage": SCARemediationStage.NPM_LATEST}
    )
    diagnostics = UpdateRetryDiagnostics(task_id="task-1")
    candidates = [
        RegistryCandidate(
            version="1.3.0",
            semver_key=(1, 3, 0),
            security_floor_met=True,
            is_stable=True,
            same_major=True,
            already_attempted=False,
        )
    ]
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.fetch_registry_candidates",
        lambda *args, **kwargs: candidates,
    )
    advice = PlannerAdvice(
        task_id="task-1",
        requested_stage=SCARemediationStage.OSV_MINIMUM,
        reasoning="try the minimum safe stage",
    )
    batch = PlannerBatchAdvice(advice=[advice])
    assert batch.advice[0].task_id == "task-1"
    plan = _convert_planner_advice_to_retry_plan(advice, task, diagnostics, group)
    assert plan.strategy_stage == SCARemediationStage.NPM_LATEST
    assert plan.selected_version == "1.3.0"


def test_planner_advice_exhaustion_pivots_to_workaround(monkeypatch):
    group = _group("g")
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY).model_copy(
        update={"strategy_stage": SCARemediationStage.NPM_LATEST}
    )
    diagnostics = UpdateRetryDiagnostics(task_id="task-1", attempted_versions=["1.3.0"])
    candidate = RegistryCandidate(
        version="1.3.0",
        semver_key=(1, 3, 0),
        security_floor_met=True,
        is_stable=True,
        same_major=True,
        already_attempted=True,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.fetch_registry_candidates",
        lambda *args, **kwargs: [candidate],
    )
    plan = _convert_planner_advice_to_retry_plan(
        PlannerAdvice(task_id="task-1", requested_stage=SCARemediationStage.NPM_LATEST),
        task,
        diagnostics,
        group,
    )
    assert plan.action == "pivot_workaround"
    assert plan.exhausted_update_path is True
    assert plan.selected_version is None


def test_phase_projection_and_audit_are_typed():
    group = _group("g")
    tasks = {"task-1": _task("task-1", "g")}
    eligible = _calculate_eligible_actions(tasks, {"g": group}, {}, {})
    decision = _select_deterministic_action(eligible, tasks, {"g": group}, {}, {})
    audit = _emit_audit(decision, [], 4)
    assert eligible.new_version_bumps == ["task-1"]
    assert audit.decision_code == DecisionCode.NEW_VERSION_BUMP
    assert audit.state_revision == 4
