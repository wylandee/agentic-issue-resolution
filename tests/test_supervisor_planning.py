"""Focused tests for supervisor planning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from remediation_engine.agents.supervisor import (
    _build_deterministic_retry_plan,
    _commit_registry_resolution_fallback,
    _deterministic_routing,
    _emit_audit,
    supervisor_router,
)
from remediation_engine.agents.supervisor_reasoning import (
    _SUPERVISOR_ACTION_CONTRACTS,
    _SUPERVISOR_STATIC_INSTRUCTIONS,
    TacticalCandidateSet,
    _model_action,
    allowed_tactical_strategies,
    build_supervisor_messages,
    build_tactical_context,
    propose_and_verify_tactical_action,
    registry_candidate_sets_for_context,
    registry_candidates_for_context,
    verify_tactical_action,
)
from remediation_engine.contracts import (
    DecisionCode,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    NoFixMitigationStage,
    RegistryCandidate,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TacticalStrategy,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    select_version,
    validate_transition,
)
from remediation_engine.contracts.schemas import (
    CodeWorkaroundSupervisorAction,
    DependencyParentContext,
    FailureCategory,
    PackageOverrideSupervisorAction,
    PortfolioEscalationSupervisorAction,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    ScannerExecutionStatus,
    TacticalSupervisorAction,
    TaskAttemptSnapshot,
)
from remediation_engine.orchestration.state import (
    _commit_task_transition,
    _no_fix_failure_transition,
    _task_sort_key,
    build_initial_remediation_task,
)
from remediation_engine.settings import AppSettings


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


def test_registry_fallback_pivots_to_child_override_when_parent_has_no_candidate():
    group = _group("transitive").model_copy(
        update={
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
            "parent_declaration_type": "dependencies",
        }
    )
    task = _task("task", group.group_id).model_copy(
        update={
            "target_package_name": "direct-parent",
            "target_dependency_type": "dependencies",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
        }
    )
    task_queue = {task.task_id: task}
    decision = _commit_registry_resolution_fallback(
        task_queue,
        task,
        group,
        None,
        {},
        {},
        (
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="direct-parent",
                dependency_type="dependencies",
                security_floor="1.2.3",
            ),
            TacticalCandidateSet(
                strategy=TacticalStrategy.PACKAGE_OVERRIDE,
                target_package_name="test-package",
                dependency_type="overrides",
                security_floor="1.2.3",
                versions=("2.0.1",),
                canonical_version="2.0.1",
            ),
        ),
        consistency_events=[],
        errors=[],
    )

    assert decision is not None
    assert decision.next_node == "update_subagent"
    committed = task_queue[task.task_id]
    assert committed.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
    assert committed.target_package_name == "test-package"
    assert committed.target_dependency_type == "overrides"
    assert committed.selected_version == "2.0.1"


def test_phase1_decision_codes_are_reserved_and_routing_remains_additive():
    assert DecisionCode.TACTICAL_ACTION_REJECTED.value == "TACTICAL_ACTION_REJECTED"
    assert DecisionCode.PEER_CONFLICT_ESCALATION.value == "PEER_CONFLICT_ESCALATION"
    assert DecisionCode.ATOMIC_CLUSTER_DISPATCH.value == "ATOMIC_CLUSTER_DISPATCH"
    assert _route({}, [], triage_required=False).decision_code == DecisionCode.NO_VALID_GROUPS


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
    assert validate_transition(TaskStatus.PENDING, TaskStatus.INCONCLUSIVE)
    assert validate_transition(TaskStatus.OPTIMISTICALLY_FIXED, TaskStatus.INCONCLUSIVE)
    assert validate_transition(TaskStatus.OPTIMISTICALLY_FIXED, TaskStatus.UNFIXABLE)
    assert validate_transition(TaskStatus.OPTIMISTICALLY_FIXED, TaskStatus.PENDING)
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


def test_verified_absent_pass_transition_is_explicitly_scoped() -> None:
    for status in (TaskStatus.PENDING, TaskStatus.NEEDS_RETRY):
        task = _task("task-1", "g", status=status)
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

        _commit_task_transition(
            queue,
            task.task_id,
            updates={"status": TaskStatus.QA_PASSED},
            allow_verified_absent_target_pass=True,
        )

        assert queue[task.task_id].status == TaskStatus.QA_PASSED
        assert queue[task.task_id].current_attempt_id is None
        assert queue[task.task_id].selected_version is None


def test_verified_absent_pass_opt_in_rejects_live_attempts() -> None:
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY).model_copy(
        update={"current_attempt_id": "attempt-live", "selected_version": "1.2.3"}
    )
    queue = {task.task_id: task}
    events = []

    _commit_task_transition(
        queue,
        task.task_id,
        updates={"status": TaskStatus.QA_PASSED},
        allow_verified_absent_target_pass=True,
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


def test_deterministic_retry_planner_preserves_committed_stage(monkeypatch):
    group = _group("g")
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY).model_copy(
        update={
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "allowed_target_versions": ["1.3.0"],
        }
    )
    diagnostics = UpdateRetryDiagnostics(task_id="task-1")
    registry_calls = []
    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_policy.fetch_registry_candidates",
        lambda *args, **kwargs: registry_calls.append((args, kwargs)),
    )
    plan = _build_deterministic_retry_plan(task, diagnostics, group)
    assert plan.strategy_stage == SCARemediationStage.NPM_LATEST
    assert plan.selected_version == "1.3.0"
    assert registry_calls == []


def test_deterministic_retry_planner_does_not_widen_approved_candidate_pool(monkeypatch):
    group = _group("g")
    task = _task("task-1", "g", status=TaskStatus.NEEDS_RETRY)
    task = task.model_copy(update={"allowed_target_versions": ["1.0.0", "1.2.0", "2.0.0"]})
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        target_package_name="test-package",
        target_dependency_type="dependencies",
        attempted_versions=["1.0.0"],
        candidate_versions_considered=["1.0.0", "1.2.0", "2.0.0"],
    )
    calls: list[set[str]] = []

    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_policy.fetch_registry_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    plan = _build_deterministic_retry_plan(task, diagnostics, group)

    assert calls == []
    assert plan.selected_version == "1.2.0"
    assert plan.candidate_versions_considered == ["1.0.0", "1.2.0", "2.0.0"]


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
        "remediation_engine.agents.supervisor_policy.fetch_registry_candidates",
        lambda *args, **kwargs: [candidate],
    )
    plan = _build_deterministic_retry_plan(task, diagnostics, group)
    assert plan.action == "pivot_workaround"
    assert plan.exhausted_update_path is True
    assert plan.selected_version is None


def test_authoritative_route_and_audit_are_typed():
    group = _group("g")
    tasks = {"task-1": _task("task-1", "g")}
    decision = _deterministic_routing(tasks, {"g": group}, {}, {})
    audit = _emit_audit(decision, [], 4)
    assert decision.next_node == "update_subagent"
    assert decision.target_task_ids == ["task-1"]
    assert audit.decision_code == DecisionCode.NEW_VERSION_BUMP
    assert audit.state_revision == 4


def _pypi_group(group_id: str, *, transitive: bool = False) -> VulnerabilityGroup:
    group = _group(group_id)
    issue = group.issues[0].model_copy(
        update={
            "ecosystem": "pypi",
            "purl": "pkg:pypi/Requests@1.0.0",
            "package_name": "Requests",
            "package_version": "1.0.0",
        }
    )
    manifest = "Pipfile.lock" if transitive else "requirements.txt"
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest,
        package_manager="pipenv" if transitive else "pip",
        is_direct_dependency=not transitive,
        declaration_type=None if transitive else "requirements",
    )
    return group.model_copy(
        update={
            "vulnerable_component": "Requests",
            "issues": [issue],
            "localized_issues": [localized],
            "file_path": manifest,
            "file_paths": [manifest],
        }
    )


def test_initial_python_direct_task_preserves_declaration():
    direct_group = _pypi_group("pypi-direct")
    direct_task = build_initial_remediation_task(direct_group, "task-direct")
    assert direct_task.target_package_name == "Requests"
    assert direct_task.target_dependency_type == "requirements"
    assert direct_task.strategy_stage == SCARemediationStage.OSV_MINIMUM


def test_pypi_retry_uses_pep440_order_and_canonical_attempted_target(monkeypatch):
    group = _pypi_group("pypi-direct")
    task = build_initial_remediation_task(group, "task-pypi").model_copy(
        update={
            "strategy_stage": SCARemediationStage.PYPI_SAME_MAJOR,
            "allowed_target_versions": ["1.9.0", "1.10.0", "2.0.0"],
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        attempted_versions_by_target={"Requests": ["1.10.0"]},
        target_package_name="REQUESTS",
        target_dependency_type="requirements",
    )
    queried = {}

    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_policy.fetch_registry_candidates",
        lambda *args, **kwargs: queried.update(args=args, kwargs=kwargs),
    )
    plan = _build_deterministic_retry_plan(task, diagnostics, group)
    assert queried == {}
    assert plan.selected_version == "1.9.0"
    assert plan.target_package_name == "Requests"
    assert plan.target_dependency_type == "requirements"


def _tactical_group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        message="dependency vulnerability",
        id=str(uuid4()),
    )
    return VulnerabilityGroup(
        group_id="group-1",
        issue_type=IssueType.SCA,
        vulnerable_component="test-pkg",
        file_path="package.json",
        cve_ids=["CVE-2026-0001"],
        versions=["1.0.0"],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=str(uuid4()),
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="1.2.3",
            instruction="Update the dependency.",
            strategy_used="test",
        ),
    )


def _tactical_task() -> RemediationTask:
    return RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.VERSION_BUMP,
        instruction="Initial dependency update.",
    )


def test_supervisor_prompt_keeps_static_and_dynamic_messages_separate() -> None:
    context = build_tactical_context(
        _tactical_task(),
        _tactical_group(),
        candidate_versions=["1.2.3", "1.3.0"],
        candidate_dependency_types=["dependencies"],
    )

    messages = build_supervisor_messages(context)

    assert len(messages) == 2
    assert messages[0].content == _SUPERVISOR_STATIC_INSTRUCTIONS
    assert "task-1" not in messages[0].content
    assert "Target package: test-pkg" in messages[1].content
    normalized_static_prompt = " ".join(messages[0].content.split())
    assert (
        "The task strategy and committed stage are historical context, not a request to repeat that strategy"
        in normalized_static_prompt
    )
    assert (
        "The registry candidate inventory additionally constrains versions"
        in normalized_static_prompt
    )
    assert (
        "This list is the authoritative strategy boundary; choose exactly one listed strategy."
        in messages[1].content
    )
    assert "Task strategy metadata (non-authoritative): version_bump" in messages[1].content
    assert "Remediation family:" not in messages[1].content
    assert "leading alternative" in messages[1].content
    assert messages[1].content.index("## Task") < messages[1].content.index(
        "## Available Remediation Actions"
    )


def test_prompt_hides_dead_parent_when_override_is_the_only_registry_action() -> None:
    group = _tactical_group().model_copy(
        update={
            "parent_contexts": [
                DependencyParentContext(
                    package_name="direct-parent",
                    package_version="1.0.0",
                    declaration_type="dependencies",
                )
            ]
        }
    )
    task = _tactical_task().model_copy(
        update={
            "target_package_name": "direct-parent",
            "target_dependency_type": "dependencies",
        }
    )
    context = build_tactical_context(
        task,
        group,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="direct-parent",
                dependency_type="dependencies",
                security_floor="1.2.3",
            ),
            TacticalCandidateSet(
                strategy=TacticalStrategy.PACKAGE_OVERRIDE,
                target_package_name="test-pkg",
                dependency_type="overrides",
                security_floor="1.2.3",
                versions=("2.0.1",),
                canonical_version="2.0.1",
            ),
        ),
    )

    prompt = build_supervisor_messages(context)[1].content

    assert "Registry/action inventory hint (not an LLM decision): package_override" in prompt
    assert "Target package: test-pkg" in prompt
    assert "action=package_override: target=test-pkg" in prompt
    assert "action=version_bump" not in prompt
    assert "direct-parent" not in prompt
    assert "Parent package:" not in prompt


def test_verified_parent_candidate_prevents_child_override_pivot() -> None:
    group = _tactical_group().model_copy(
        update={
            "parent_contexts": [
                DependencyParentContext(
                    package_name="direct-parent",
                    package_version="1.0.0",
                    declaration_type="dependencies",
                )
            ]
        }
    )
    task = _tactical_task().model_copy(
        update={
            "target_package_name": "direct-parent",
            "target_dependency_type": "dependencies",
        }
    )
    context = build_tactical_context(
        task,
        group,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="direct-parent",
                dependency_type="dependencies",
                security_floor="1.2.3",
                versions=("1.2.3",),
                canonical_version="1.2.3",
            ),
            TacticalCandidateSet(
                strategy=TacticalStrategy.PACKAGE_OVERRIDE,
                target_package_name="test-pkg",
                dependency_type="overrides",
                security_floor="1.2.3",
                versions=("2.0.1",),
                canonical_version="2.0.1",
            ),
        ),
    )
    prompt = build_supervisor_messages(context)[1].content
    action = PackageOverrideSupervisorAction(
        diagnostic_basis="The child also has a verified override candidate.",
        selected_strategy=TacticalStrategy.PACKAGE_OVERRIDE,
        target_version="2.0.1",
        rationale="The override would avoid changing the parent.",
    )

    verification = verify_tactical_action(context, action)

    assert "action=version_bump: target=direct-parent" in prompt
    assert "action=package_override" not in prompt
    assert "- Strategies: version_bump, code_workaround" in prompt
    assert verification.accepted is False
    assert verification.reason == (
        "A verified parent update is available; complete the parent update before switching to a package override."
    )


def test_verified_noncanonical_candidate_can_be_selected_with_rationale() -> None:
    task = _tactical_task().model_copy(
        update={
            "target_package_name": "test-pkg",
            "target_dependency_type": "dependencies",
        }
    )
    context = build_tactical_context(
        task,
        _tactical_group(),
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="test-pkg",
                dependency_type="dependencies",
                security_floor="1.2.3",
                versions=("1.2.3", "2.0.0"),
                canonical_version="1.2.3",
            ),
        ),
    )
    action = TacticalSupervisorAction(
        diagnostic_basis="The newer verified version avoids the observed compatibility failure.",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        target_version="2.0.0",
        rationale="The later verified candidate is justified by the failed lower candidate.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is True
    assert verification.selected_version == "2.0.0"


def test_verified_version_produces_authoritative_targeted_instruction() -> None:
    context = build_tactical_context(
        _tactical_task(),
        _tactical_group(),
        candidate_versions=["1.2.3"],
        candidate_dependency_types=["dependencies"],
    )
    action = TacticalSupervisorAction(
        diagnostic_basis="Verified candidate meets the committed security floor.",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        target_version="1.2.3",
        rationale="The verified fixed version addresses the finding.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is True
    assert verification.instruction is not None
    assert "AUTHORIZED TARGET: test-pkg" in verification.instruction
    assert "EXACT VERSION OR HYPOTHESIS: 1.2.3" in verification.instruction
    assert "PROHIBITED OPERATIONS" in verification.instruction


def test_unverified_version_is_rejected_before_dispatch() -> None:
    context = build_tactical_context(
        _tactical_task(),
        _tactical_group(),
        candidate_versions=["1.2.3"],
    )
    action = TacticalSupervisorAction(
        diagnostic_basis="The proposed version is absent from the verified candidate set.",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        target_version="9.9.9",
        rationale="Try the newest version.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is False
    assert "candidate" in verification.reason


def test_tactical_model_is_disabled_without_an_api_key() -> None:
    context = build_tactical_context(
        _tactical_task(), _tactical_group(), candidate_versions=["1.2.3"]
    )

    action, verification = propose_and_verify_tactical_action(
        context,
        settings=AppSettings(openai_api_key=""),
    )

    assert action is None
    assert verification is None


def test_tactical_model_uses_one_of_four_tools_not_a_root_union() -> None:
    """Provider requests use concrete action tools instead of root ``oneOf``."""
    context = build_tactical_context(
        _tactical_task(), _tactical_group(), candidate_versions=["1.2.3"]
    )
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "VersionBumpSupervisorAction",
                "args": {
                    "diagnostic_basis": "The verified candidate meets the security floor.",
                    "selected_strategy": "version_bump",
                    "target_version": "1.2.3",
                    "rationale": "The lowest verified candidate is sufficient.",
                },
                "id": "tactical-1",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.agents.supervisor_reasoning.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action = _model_action(
            context,
            AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is not None
    assert action.selected_strategy == TacticalStrategy.VERSION_BUMP
    llm.with_structured_output.assert_not_called()
    contracts = llm.bind_tools.call_args.args[0]
    assert [contract.__name__ for contract in contracts] == [
        "VersionBumpSupervisorAction",
        "PackageOverrideSupervisorAction",
        "CodeWorkaroundSupervisorAction",
        "PortfolioEscalationSupervisorAction",
    ]
    assert llm.bind_tools.call_args.kwargs == {
        "tool_choice": "required",
        "strict": True,
        "parallel_tool_calls": False,
    }


def test_tactical_model_uses_supervisor_specific_luna_configuration(monkeypatch) -> None:
    context = build_tactical_context(
        _tactical_task(), _tactical_group(), candidate_versions=["1.2.3"]
    )
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "VersionBumpSupervisorAction",
                "args": {
                    "diagnostic_basis": "The verified candidate meets the security floor.",
                    "selected_strategy": "version_bump",
                    "target_version": "1.2.3",
                    "rationale": "Use the verified version.",
                },
                "id": "tactical-luna",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm
    captured: dict[str, object] = {}

    def create_model(model_name: str, **kwargs: object) -> MagicMock:
        captured.update(model_name=model_name, **kwargs)
        return llm

    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_reasoning.create_chat_model",
        create_model,
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", MagicMock())
    with patch(
        "remediation_engine.agents.supervisor_reasoning.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action = _model_action(
            context,
            AppSettings(
                openai_api_key="test-key",
                remedy_llm_model="gpt-6-luna",
                remedy_llm_reasoning_effort="medium",
                supervisor_llm_model="gpt-6-luna",
                supervisor_llm_reasoning_effort="xhigh",
            ),
        )

    assert action is not None
    assert captured["model_name"] == "gpt-6-luna"
    assert captured["reasoning_effort"] == "xhigh"


def test_tactical_model_repairs_an_incomplete_override_tool_call_once() -> None:
    """A missing version is repaired, never inferred from the candidate set."""
    group = _tactical_group().model_copy(
        update={
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
            "parent_declaration_type": "dependencies",
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "target_package_name": "test-pkg",
            "target_dependency_type": "overrides",
        }
    )
    context = build_tactical_context(
        task,
        group,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.PACKAGE_OVERRIDE,
                target_package_name="test-pkg",
                dependency_type="overrides",
                security_floor="1.2.3",
                versions=("1.2.3",),
                canonical_version="1.2.3",
            ),
        ),
    )
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "PackageOverrideSupervisorAction",
                "args": {
                    "diagnostic_basis": "A child override candidate is available.",
                    "rationale": "Use the authorized child override.",
                },
                "id": "override-incomplete",
            }
        ],
    )
    repaired = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "PackageOverrideSupervisorAction",
                "args": {
                    "diagnostic_basis": "The transitive child has a verified candidate.",
                    "selected_strategy": "package_override",
                    "target_version": "1.2.3",
                    "rationale": "Pin the vulnerable child at the authorized version.",
                },
                "id": "override-repaired",
            }
        ],
    )
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [first, repaired]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.agents.supervisor_reasoning.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action, verification = propose_and_verify_tactical_action(
            context,
            settings=AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is not None
    assert action.selected_strategy == TacticalStrategy.PACKAGE_OVERRIDE
    assert action.target_version == "1.2.3"
    assert verification is not None and verification.accepted is True
    assert bound_llm.invoke.call_count == 2
    repair_prompt = "\n".join(
        str(message.content) for message in bound_llm.invoke.call_args_list[1].args[0]
    )
    assert "selected_strategy, target_version" in repair_prompt


def test_peer_conflict_without_source_evidence_requires_portfolio_referral() -> None:
    """Do not advertise or accept a workaround without authoritative source paths."""
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.PEER_CONFLICT,
        retry_feedback="The dependency solver reports an unresolved peer conflict.",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["No compatible peer candidate is available."]
        ),
    )
    context = build_tactical_context(
        _tactical_task(),
        _tactical_group(),
        evaluation=evaluation,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="test-pkg",
                dependency_type="dependencies",
                security_floor="1.2.3",
                versions=(),
                canonical_version=None,
                peer_compatible=False,
            ),
        ),
    )
    dynamic = build_supervisor_messages(context)[1].content

    assert "- Strategies: escalate_to_portfolio" in dynamic
    assert "action=escalate_to_portfolio" in dynamic
    assert "code_workaround" not in dynamic

    unsupported_workaround = CodeWorkaroundSupervisorAction(
        diagnostic_basis="QA identifies a peer conflict without source evidence.",
        selected_strategy=TacticalStrategy.CODE_WORKAROUND,
        workaround_hypothesis="Change source behavior to avoid the peer conflict.",
        target_files_hint=["src/auth.ts"],
        rationale="Try a source workaround despite the missing source evidence.",
    )
    rejected = verify_tactical_action(context, unsupported_workaround)

    assert rejected.accepted is False
    assert (
        rejected.reason
        == "Peer conflict has no compatible candidate or evidence-backed source workaround; "
        "portfolio escalation is required."
    )
    assert rejected.instruction is None

    escalation = PortfolioEscalationSupervisorAction(
        diagnostic_basis="Peer conflict has no compatible verified candidate or source fix.",
        selected_strategy=TacticalStrategy.ESCALATE_TO_PORTFOLIO,
        rationale="Refer the incompatible dependency constraints for coordinated resolution.",
    )
    accepted = verify_tactical_action(context, escalation)
    assert accepted.accepted is True

    source_evaluation = evaluation.model_copy(
        update={
            "failure_evidence": QAFailureEvidence(
                exact_diagnostics=["No compatible peer candidate is available."],
                source_locations=["src/auth.ts"],
            )
        }
    )
    source_context = build_tactical_context(
        _tactical_task(),
        _tactical_group(),
        evaluation=source_evaluation,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.VERSION_BUMP,
                target_package_name="test-pkg",
                dependency_type="dependencies",
                security_floor="1.2.3",
                versions=(),
                canonical_version=None,
                peer_compatible=False,
            ),
        ),
    )
    evidence_backed_workaround = CodeWorkaroundSupervisorAction(
        diagnostic_basis="QA identifies the source location for a compatibility fix.",
        selected_strategy=TacticalStrategy.CODE_WORKAROUND,
        workaround_hypothesis="Guard the incompatible source behavior.",
        target_files_hint=["src/auth.ts"],
        rationale="Use the authoritative source evidence to resolve the issue safely.",
    )
    source_verification = verify_tactical_action(source_context, evidence_backed_workaround)

    assert source_verification.accepted is True
    assert "AUTHORIZED TARGET: test-pkg" in (source_verification.instruction or "")


def test_incomplete_override_repair_stays_fail_closed() -> None:
    """Two malformed model responses never become an inferred package version."""
    group = _tactical_group().model_copy(
        update={
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
            "parent_declaration_type": "dependencies",
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "target_package_name": "test-pkg",
            "target_dependency_type": "overrides",
        }
    )
    context = build_tactical_context(
        task,
        group,
        candidate_sets=(
            TacticalCandidateSet(
                strategy=TacticalStrategy.PACKAGE_OVERRIDE,
                target_package_name="test-pkg",
                dependency_type="overrides",
                security_floor="1.2.3",
                versions=("1.2.3",),
                canonical_version="1.2.3",
            ),
        ),
    )
    malformed = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "PackageOverrideSupervisorAction",
                "args": {
                    "diagnostic_basis": "A child override candidate is available.",
                    "rationale": "Use the authorized child override.",
                },
                "id": "override-incomplete",
            }
        ],
    )
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [malformed, malformed]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.agents.supervisor_reasoning.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action, verification = propose_and_verify_tactical_action(
            context,
            settings=AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is None
    assert verification is None
    assert bound_llm.invoke.call_count == 2


def test_tactical_model_rejects_free_text_without_a_tool_call() -> None:
    """A free-form response remains unavailable and triggers fallback."""
    context = build_tactical_context(
        _tactical_task(), _tactical_group(), candidate_versions=["1.2.3"]
    )
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(content="Use the latest version.")
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.agents.supervisor_reasoning.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action = _model_action(
            context,
            AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is None


def test_provider_action_tools_have_flat_required_schemas() -> None:
    """Each provider tool is independently strict without a root ``oneOf``."""
    for contract in _SUPERVISOR_ACTION_CONTRACTS:
        converted = convert_to_openai_tool(contract, strict=True)
        schema = converted["function"]["parameters"]
        assert converted["function"]["strict"] is True
        assert "oneOf" not in schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(contract.model_fields)


def test_supervisor_prompt_uses_gate_statuses_and_bounded_failure_summary() -> None:
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The targeted API regression needs a compatibility workaround.",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["TypeError: jwt.verify is not a function"],
            failed_tests=["auth middleware rejects a valid token"],
            source_locations=["/workspace/src/auth.ts:12:4"],
            raw_excerpt="raw-secret-log-that-must-not-be-replayed",
        ),
        deterministic_gates=QADeterministicGates(
            status="fail",
            install_passed=True,
            scanner_execution_status=ScannerExecutionStatus.SUCCESS,
            target_scanner_cleared=True,
            tests_passed=False,
            package_manifest_state="match",
            package_graph_state="match",
        ),
    )
    context = build_tactical_context(_tactical_task(), _tactical_group(), evaluation=evaluation)

    dynamic = build_supervisor_messages(context)[1].content

    assert (
        "Deterministic policy gate (policy-specific; inspect individual gates too): FAIL" in dynamic
    )
    assert "Install gate: PASS" in dynamic
    assert "Target scanner gate: PASS" in dynamic
    assert "Unit-test gate: FAIL" in dynamic
    assert "Manifest gate: PASS" in dynamic
    assert "Dependency-graph gate: PASS" in dynamic
    assert "Manifest/lockfile paths (context only; never workaround targets):" in dynamic
    assert "Source locations (eligible workaround hints):" in dynamic
    assert "Affected files (evidence only; not automatically workaround targets):" in dynamic
    assert "TypeError: jwt.verify is not a function" in dynamic
    assert "raw-secret-log-that-must-not-be-replayed" not in dynamic
    assert dynamic.index("## QA Deterministic Gates") < dynamic.index("## QA Failure Evidence")
    assert dynamic.index("## QA Failure Evidence") < dynamic.index("## Worker Diagnostics")
    assert "raw_excerpt" not in dynamic


def test_evidence_and_model_file_paths_share_workspace_normalization() -> None:
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The targeted API regression needs a compatibility workaround.",
        failure_evidence=QAFailureEvidence(
            source_locations=["file:///workspace/src\\auth.ts:12:4"],
        ),
    )
    context = build_tactical_context(_tactical_task(), _tactical_group(), evaluation=evaluation)
    action = CodeWorkaroundSupervisorAction(
        diagnostic_basis="The source location identifies the failed API call.",
        selected_strategy=TacticalStrategy.CODE_WORKAROUND,
        workaround_hypothesis="Guard the incompatible JWT API call.",
        target_files_hint=["src/auth.ts:12:4"],
        rationale="A source guard directly addresses the breaking API evidence.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is True
    assert action.target_files_hint == ["src/auth.ts"]


def test_workaround_manifest_and_lockfile_hints_are_never_authorized() -> None:
    """Manifest hints stay forbidden even when QA evidence names them."""
    group = _tactical_group().model_copy(
        update={"file_paths": ["package.json", "package-lock.json"]}
    )
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The source change needs a compatibility workaround.",
        failure_evidence=QAFailureEvidence(
            source_locations=["src/auth.ts"],
            affected_files=["package.json", "package-lock.json"],
        ),
    )
    context = build_tactical_context(_tactical_task(), group, evaluation=evaluation)
    action = CodeWorkaroundSupervisorAction(
        diagnostic_basis="QA identifies the source file and affected manifests.",
        selected_strategy=TacticalStrategy.CODE_WORKAROUND,
        workaround_hypothesis="Guard the incompatible source behavior.",
        target_files_hint=["src/auth.ts", "package.json", "package-lock.json"],
        rationale="The change should remain limited to evidence-backed source code.",
    )

    rejected = verify_tactical_action(context, action)

    assert rejected.accepted is False
    assert (
        rejected.reason
        == "Manifest and lockfile paths are never valid workaround target-file hints."
    )
    assert rejected.instruction is None

    source_only_action = action.model_copy(update={"target_files_hint": ["src/auth.ts"]})
    accepted = verify_tactical_action(context, source_only_action)

    assert accepted.accepted is True
    assert accepted.instruction is not None
    assert "AUTHORIZED TARGET: test-pkg" in accepted.instruction
    assert "EXACT VERSION OR HYPOTHESIS: Guard the incompatible source behavior." in (
        accepted.instruction
    )
    assert "PROHIBITED OPERATIONS" in accepted.instruction


def test_parent_minimum_is_not_used_as_a_security_floor() -> None:
    task = _tactical_task().model_copy(update={"parent_minimum_version": "9.9.9"})
    calls: list[str] = []

    def provider(
        package_name: str, security_floor: str, attempted_versions: set[str], ecosystem="npm"
    ):
        calls.append(security_floor)
        from remediation_engine.contracts.version_policy import RegistryCandidate

        return [
            RegistryCandidate(
                version="1.2.3",
                semver_key=(1, 2, 3),
                security_floor_met=True,
                is_stable=True,
                same_major=True,
                already_attempted=False,
            )
        ]

    context = build_tactical_context(task, _tactical_group())
    candidate_sets, error = registry_candidate_sets_for_context(
        context,
        registry_provider=provider,
    )

    assert error is None
    assert calls == ["1.2.3"]
    assert candidate_sets[0].versions == ("1.2.3",)


def test_default_registry_fetcher_receives_ecosystem_as_keyword() -> None:
    calls: list[tuple[str, str, str]] = []

    def fetch_candidates(
        package_name: str,
        security_floor: str,
        attempted_versions: set[str],
        *,
        project_language,
        maven_mode: bool = False,
        ecosystem: str | None = None,
    ) -> list[RegistryCandidate]:
        calls.append((package_name, security_floor, ecosystem or ""))
        return [
            RegistryCandidate(
                version="1.2.3",
                semver_key=(1, 2, 3),
                security_floor_met=True,
                is_stable=True,
                same_major=True,
                already_attempted=False,
            )
        ]

    with patch(
        "remediation_engine.agents.supervisor_reasoning._supervisor_fetch_registry_candidates",
        side_effect=fetch_candidates,
    ):
        candidate_sets, error = registry_candidate_sets_for_context(
            build_tactical_context(_tactical_task(), _tactical_group())
        )

    assert error is None
    assert calls == [("test-pkg", "1.2.3", "npm")]
    assert candidate_sets[0].versions == ("1.2.3",)


def test_transitive_inventory_keeps_child_override_when_parent_has_no_compatible_version():
    group = _tactical_group().model_copy(
        update={
            "parent_contexts": [
                DependencyParentContext(
                    package_name="direct-parent",
                    package_version="1.0.0",
                    declaration_type="dependencies",
                )
            ]
        }
    )
    task = _tactical_task().model_copy(
        update={
            "target_package_name": "direct-parent",
            "target_dependency_type": "dependencies",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
        }
    )

    def provider(
        _package_name: str, _security_floor: str, _attempted_versions: set[str], ecosystem="npm"
    ):
        return [
            RegistryCandidate(
                version="2.0.1",
                semver_key=(2, 0, 1),
                security_floor_met=True,
                is_stable=True,
                same_major=False,
                already_attempted=False,
            )
        ]

    with patch(
        "remediation_engine.agents.supervisor_reasoning._supervisor_plan_npm_parent_version",
        return_value="- Eligible Candidates: NONE\n- Selected Version: NONE",
    ):
        candidate_sets, error = registry_candidate_sets_for_context(
            build_tactical_context(task, group),
            registry_provider=provider,
        )

    assert error is None
    assert [candidate.strategy for candidate in candidate_sets] == [
        TacticalStrategy.VERSION_BUMP,
        TacticalStrategy.PACKAGE_OVERRIDE,
    ]
    assert candidate_sets[0].versions == ()
    assert candidate_sets[1].target_package_name == "test-pkg"
    assert candidate_sets[1].versions == ("2.0.1",)


def test_transitive_inventory_suppresses_child_override_when_parent_is_compatible() -> None:
    group = _tactical_group().model_copy(
        update={
            "parent_contexts": [
                DependencyParentContext(
                    package_name="direct-parent",
                    package_version="1.0.0",
                    declaration_type="dependencies",
                )
            ]
        }
    )
    task = _tactical_task().model_copy(
        update={
            "target_package_name": "direct-parent",
            "target_dependency_type": "dependencies",
        }
    )
    child_registry_calls: list[str] = []

    def provider(package_name: str, _security_floor: str, _attempted_versions: set[str]):
        child_registry_calls.append(package_name)
        return [
            RegistryCandidate(
                version="2.0.1",
                semver_key=(2, 0, 1),
                security_floor_met=True,
                is_stable=True,
                same_major=False,
                already_attempted=False,
            )
        ]

    with patch(
        "remediation_engine.agents.supervisor_reasoning._supervisor_plan_npm_parent_version",
        return_value="- Eligible Candidates: 1.2.3\n- Selected Version: 1.2.3",
    ):
        candidate_sets, error = registry_candidate_sets_for_context(
            build_tactical_context(task, group),
            registry_provider=provider,
        )

    assert error is None
    assert child_registry_calls == []
    assert len(candidate_sets) == 1
    assert candidate_sets[0].strategy == TacticalStrategy.VERSION_BUMP
    assert candidate_sets[0].target_package_name == "direct-parent"
    assert candidate_sets[0].versions == ("1.2.3",)


def test_package_override_stage_does_not_require_parent_version() -> None:
    group = _tactical_group().model_copy(
        update={
            "parent_contexts": [
                DependencyParentContext(package_name="direct-parent", package_version=None)
            ]
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "target_package_name": "test-pkg",
            "target_dependency_type": "overrides",
        }
    )
    calls: list[tuple[str, str, set[str]]] = []

    def provider(
        package_name: str, security_floor: str, attempted_versions: set[str], ecosystem="npm"
    ):
        calls.append((package_name, security_floor, attempted_versions))
        return [
            RegistryCandidate(
                version="1.2.3",
                semver_key=(1, 2, 3),
                security_floor_met=True,
                is_stable=True,
                same_major=True,
                already_attempted=False,
            )
        ]

    context = build_tactical_context(task, group)
    candidate_sets, error = registry_candidate_sets_for_context(
        context,
        registry_provider=provider,
    )
    legacy_versions, legacy_error = registry_candidates_for_context(
        context,
        registry_provider=provider,
    )

    assert error is None
    assert legacy_error is None
    assert calls == [("test-pkg", "1.2.3", set()), ("test-pkg", "1.2.3", set())]
    assert legacy_versions == ("1.2.3",)
    assert len(candidate_sets) == 1
    assert candidate_sets[0].strategy == TacticalStrategy.PACKAGE_OVERRIDE
    assert candidate_sets[0].target_package_name == "test-pkg"


def test_package_override_stage_accepts_only_authorized_override_action() -> None:
    group = _tactical_group().model_copy(
        update={
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
            "parent_declaration_type": "dependencies",
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "target_package_name": "test-pkg",
            "target_dependency_type": "overrides",
        }
    )

    def provider(
        _package_name: str, _security_floor: str, _attempted_versions: set[str], ecosystem="npm"
    ):
        return [
            RegistryCandidate(
                version="1.2.3",
                semver_key=(1, 2, 3),
                security_floor_met=True,
                is_stable=True,
                same_major=True,
                already_attempted=False,
            )
        ]

    base_context = build_tactical_context(task, group)
    candidate_sets, error = registry_candidate_sets_for_context(
        base_context,
        registry_provider=provider,
    )
    context = build_tactical_context(task, group, candidate_sets=candidate_sets)

    assert error is None
    assert allowed_tactical_strategies(task, group) == (
        TacticalStrategy.PACKAGE_OVERRIDE,
        TacticalStrategy.CODE_WORKAROUND,
    )
    assert (
        verify_tactical_action(
            context,
            TacticalSupervisorAction(
                diagnostic_basis="A direct update is not authorized at the override stage.",
                selected_strategy=TacticalStrategy.VERSION_BUMP,
                target_version="1.2.3",
                rationale="Try a direct update anyway.",
            ),
        ).accepted
        is False
    )

    verification = verify_tactical_action(
        context,
        PackageOverrideSupervisorAction(
            diagnostic_basis="The vulnerable transitive child has a verified override candidate.",
            selected_strategy=TacticalStrategy.PACKAGE_OVERRIDE,
            target_version="1.2.3",
            rationale="Override the vulnerable child without changing its parent.",
        ),
    )

    assert verification.accepted is True
    assert verification.target_package_name == "test-pkg"
    assert verification.target_dependency_type == "overrides"
    assert verification.allowed_target_versions == ("1.2.3",)
    assert "AUTHORIZED TARGET: test-pkg" in (verification.instruction or "")


def test_retry_registry_candidates_remain_inside_initial_approved_pool() -> None:
    group = _tactical_group()
    task = _tactical_task().model_copy(
        update={
            "selected_version": "1.0.0",
            "target_package_name": "test-pkg",
            "target_dependency_type": "dependencies",
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-initial",
        task_id=task.task_id,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        selected_version="1.0.0",
        allowed_target_versions=["1.0.0", "1.2.0", "2.0.0"],
        target_package_name="test-pkg",
        target_dependency_type="dependencies",
        instruction="Update test-pkg.",
        instruction_digest="digest",
        dispatch_node="update_subagent",
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        target_package_name="test-pkg",
        target_dependency_type="dependencies",
        attempted_versions=["1.0.0"],
        candidate_versions_considered=["1.0.0", "1.2.0", "2.0.0"],
    )
    calls: list[set[str]] = []

    def provider(
        package_name: str, security_floor: str, attempted_versions: set[str], ecosystem="npm"
    ):
        calls.append(attempted_versions)
        return [
            RegistryCandidate(
                version=version,
                semver_key=tuple(int(part) for part in version.split(".")),
                security_floor_met=True,
                is_stable=True,
                same_major=version.startswith("1."),
                already_attempted=False,
                selection_roles=roles,
            )
            for version, roles in (
                ("1.0.0", ("osv_minimum",)),
                ("1.1.0", ("osv_minimum",)),
                ("1.2.0", ("same_major",)),
                ("2.0.0", ("npm_latest",)),
            )
        ]

    candidate_sets, error = registry_candidate_sets_for_context(
        build_tactical_context(
            task,
            group,
            retry_diagnostics=diagnostics,
            prior_attempts=[snapshot],
        ),
        registry_provider=provider,
    )

    assert error is None
    assert calls == [set()]
    assert candidate_sets[0].versions == ("1.2.0", "2.0.0")
    assert "1.1.0" not in candidate_sets[0].versions


def test_pypi_tactical_candidates_use_pep440_roles_and_normalized_identity():
    base_group = _tactical_group()
    pypi_issue = base_group.issues[0].model_copy(
        update={
            "ecosystem": "pypi",
            "purl": "pkg:pypi/Requests@1.0.0",
            "package_name": "Requests",
        }
    )
    group = base_group.model_copy(
        update={
            "vulnerable_component": "Requests",
            "issues": [pypi_issue],
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PYPI_SAME_MAJOR,
            "target_package_name": "requests",
            "target_dependency_type": "requirements",
        }
    )
    calls = []

    def provider(package_name, floor, attempted, ecosystem):
        calls.append((package_name, floor, set(attempted), ecosystem))
        return [
            RegistryCandidate(
                version=version,
                ecosystem="pypi",
                semver_key=None,
                security_floor_met=True,
                is_stable=True,
                same_major=version.startswith("1."),
                already_attempted=False,
                selection_roles=roles,
            )
            for version, roles in (
                ("1.2.3", ("osv_minimum",)),
                ("1.9.0", ("same_major",)),
                ("1.10.0", ("same_major",)),
                ("2.0.0", ("pypi_latest",)),
            )
        ]

    candidate_sets, error = registry_candidate_sets_for_context(
        build_tactical_context(task, group),
        registry_provider=provider,
    )

    assert error is None
    assert calls == [("requests", "1.2.3", set(), "pypi")]
    assert len(candidate_sets) == 1
    assert candidate_sets[0].versions == ("1.10.0",)
    assert candidate_sets[0].target_package_name == "requests"


def test_pypi_transitive_tactical_path_never_offers_package_override():
    base_group = _tactical_group()
    pypi_issue = base_group.issues[0].model_copy(
        update={
            "ecosystem": "pypi",
            "purl": "pkg:pypi/Requests@1.0.0",
            "package_name": "Requests",
        }
    )
    group = base_group.model_copy(
        update={
            "vulnerable_component": "Requests",
            "issues": [pypi_issue],
            "parent_contexts": [
                DependencyParentContext(
                    package_name="Direct_Parent",
                    package_version="1.0.0",
                    declaration_type="packages",
                )
            ],
        }
    )
    task = _tactical_task().model_copy(
        update={
            "strategy_stage": SCARemediationStage.PYPI_LATEST,
            "target_package_name": "direct-parent",
            "target_dependency_type": "packages",
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
        }
    )
    with patch(
        "remediation_engine.agents.supervisor_reasoning._supervisor_plan_parent_version",
        return_value=("- Selected: 1.4.0\n- Eligible Candidates: 1.4.0\n- PyPI Latest: 1.4.0"),
    ):
        candidate_sets, error = registry_candidate_sets_for_context(
            build_tactical_context(task, group)
        )

    assert error is None
    assert allowed_tactical_strategies(task, group) == (
        TacticalStrategy.VERSION_BUMP,
        TacticalStrategy.CODE_WORKAROUND,
    )
    assert candidate_sets[0].target_package_name == "direct-parent"
    assert candidate_sets[0].dependency_type == "packages"
