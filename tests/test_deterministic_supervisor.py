"""Focused contracts for deterministic Supervisor routing."""

from __future__ import annotations

from uuid import uuid4

from remediation_engine.contracts import (
    DecisionCode,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    NoFixMitigationStage,
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
    _build_deterministic_retry_plan,
    _calculate_eligible_actions,
    _commit_task_transition,
    _deterministic_routing,
    _emit_audit,
    _no_fix_failure_transition,
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
            {"t": _task("t", "g", status=TaskStatus.QA_PASSED)},
            [_group("g")],
            workspace_volume="workspace-volume",
        ).decision_code
        == DecisionCode.FINAL_FULL_SCAN_REQUIRED
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
    assert decision.target_task_ids == ["task-a"]
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
    assert validate_transition(TaskStatus.OPTIMISTICALLY_FIXED, TaskStatus.UNFIXABLE)
    assert not validate_transition(TaskStatus.QA_PASSED, TaskStatus.PENDING)
    assert not validate_transition(TaskStatus.PENDING, TaskStatus.QA_PASSED)


def test_final_no_fix_qa_failure_terminalizes_and_closes_attempt():
    task = _task(
        "task-1",
        "g",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        retry_count=1,
        no_fix_stage=NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
    ).model_copy(
        update={
            "current_attempt_id": "attempt-1",
            "selected_version": "1.2.3",
        }
    )
    queue = {task.task_id: task}
    updates, reset_workspace = _no_fix_failure_transition(task, _group("g", no_fix=True))

    assert reset_workspace is False
    _commit_task_transition(queue, task.task_id, updates=updates)

    committed = queue[task.task_id]
    assert committed.status == TaskStatus.UNFIXABLE
    assert committed.no_fix_stage == NoFixMitigationStage.UNFIXABLE
    assert committed.retry_count == 2
    assert committed.current_attempt_id is None
    assert committed.selected_version is None


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


def test_commit_rejects_unscoped_retry_to_pass_transition():
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY)
    queue = {task.task_id: task}
    events = []

    _commit_task_transition(
        queue,
        task.task_id,
        updates={"status": TaskStatus.QA_PASSED},
        consistency_events=events,
    )

    assert queue[task.task_id] == task
    assert events and events[0].error_code == "INVALID_TRANSITION"


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


def test_deterministic_retry_planner_preserves_committed_stage(monkeypatch):
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
    plan = _build_deterministic_retry_plan(task, diagnostics, group)
    assert plan.strategy_stage == SCARemediationStage.NPM_LATEST
    assert plan.selected_version == "1.3.0"


def test_deterministic_retry_planner_exhaustion_pivots_to_workaround(monkeypatch):
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
    plan = _build_deterministic_retry_plan(task, diagnostics, group)
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
