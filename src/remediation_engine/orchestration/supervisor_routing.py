"""Deterministic Supervisor routing, reconciliation, and transition helpers."""

from __future__ import annotations

import logging
from typing import Any

from remediation_engine.contracts.decision_codes import DecisionCode, validate_transition
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    NoFixMitigationStage,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
    SupervisorDecision,
    SupervisorRetryPlan,
    TaskAttemptSnapshot,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.contracts.supervisor_phases import (
    AuditRecord,
    EligibleActions,
    ReconciliationResult,
)
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.supervisor_planner import (
    QA_DISPATCH_LIMIT,
    UPDATE_DISPATCH_LIMIT,
    _build_high_level_retry_instruction,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _WORKABLE_STATUSES,
    MAX_RETRIES,
    _has_existing_workaround_child,
    _is_exhausted_update_pivot_candidate,
    _qa_ready_task_ids,
    _task_sort_key,
)
from remediation_engine.orchestration.task_utils import build_no_fix_retry_instruction

logger = logging.getLogger(__name__)


def _build_consistency_event(
    *,
    error_code: str,
    task_id: str | None,
    expected_attempt_id: str | None,
    received_attempt_id: str | None,
    action: str,
    details: str,
) -> StateConsistencyEvent:
    return StateConsistencyEvent(
        error_code=error_code,
        task_id=task_id,
        expected_attempt_id=expected_attempt_id,
        received_attempt_id=received_attempt_id,
        action=action,
        details=details,
    )


def _dedupe_consistency_events(events: list[StateConsistencyEvent]) -> list[StateConsistencyEvent]:
    seen: set[tuple[str | None, str | None, str]] = set()
    result: list[StateConsistencyEvent] = []
    for event in events:
        key = (event.task_id, event.received_attempt_id, event.error_code)
        if key not in seen:
            seen.add(key)
            result.append(event)
    return result


def _build_workaround_retry_instruction(
    task: RemediationTask,
    evaluation: QAEvaluation | None,
    group: VulnerabilityGroup | None = None,
) -> str:
    """Synthesize a high-level retry instruction for a generic workaround task."""
    category_str = (
        evaluation.failure_category.value
        if evaluation and evaluation.failure_category
        else "unknown"
    )
    feedback_str = (
        evaluation.retry_feedback
        if evaluation and evaluation.retry_feedback
        else "No feedback provided."
    )
    component = (group.vulnerable_component if group else None) or task.parent_group_id
    return (
        f"RETRY: Your previous code workaround attempt for {component} failed QA.\n"
        f"Failure category: {category_str}\n"
        f"QA feedback: {feedback_str}\n\n"
        f"Original instruction: {task.instruction}\n\n"
        f"Fix the issues identified by QA and re-apply a valid code workaround."
    )


def _deterministic_routing(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    action_summaries: list[AgentActionSummary] | None = None,
    active_target_task_ids: list[str] | None = None,
    current_status: str = "",
    triage_required: bool = False,
    workspace_volume: str | None = None,
    final_full_scan_completed: bool = False,
) -> SupervisorDecision:
    """
    Pure-Python routing used as the Supervisor's authoritative state machine.

    Implements the fixed Supervisor priority rules.
    """
    tasks = sorted(task_queue.values(), key=lambda task: _task_sort_key(task, group_by_id))
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
    # Post-QA triage is a Supervisor-owned handoff.  It must run before
    # teardown even when the current task queue is already terminal, because
    # the global scan may have discovered a new package vulnerability.
    if triage_required and current_status in {"qa_completed", "qa_failed", "final_scan_completed"}:
        return SupervisorDecision(
            decision_code=DecisionCode.TRIAGE_REQUIRED,
            next_node="triage",
            target_task_ids=[],
            instructions="Re-triage the complete parseable post-remediation scan before the next remediation decision.",
            decision_reason="QA produced a parseable scan and marked post-QA triage as required.",
        )

    if not group_by_id and not task_queue:
        return SupervisorDecision(
            decision_code=DecisionCode.NO_VALID_GROUPS,
            next_node="teardown",
            target_task_ids=[],
            instructions="No valid vulnerability groups are available for remediation.",
            decision_reason="Deterministic routing found no valid groups.",
        )

    # All tasks are terminal â†’ authoritative full scan, then teardown.
    if not non_terminal:
        if workspace_volume and task_queue and not final_full_scan_completed:
            return SupervisorDecision(
                decision_code=DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                next_node="final_full_scan",
                target_task_ids=[],
                instructions="Run the authoritative full Dependency-Check scan before teardown.",
                decision_reason="All remediation tasks are terminal and the final full scan is not complete.",
            )
        return SupervisorDecision(
            decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
            next_node="teardown",
            target_task_ids=[],
            instructions="All tasks are terminal. Proceeding to teardown.",
            decision_reason="No actionable tasks remain.",
        )

    # If an active task is optimistically_fixed â†’ route it to qa_critic.
    current_task_qa_ready = _qa_ready_task_ids(
        task_queue,
        preferred_ids=list(active_target_task_ids or []),
        group_by_id=group_by_id,
        limit=QA_DISPATCH_LIMIT,
    )
    if current_status != "qa_completed" and current_task_qa_ready:
        return SupervisorDecision(
            decision_code=DecisionCode.QA_READY,
            next_node="qa_critic",
            target_task_ids=current_task_qa_ready,
            instructions="Run QA on the current remediated task before starting more remediation.",
            decision_reason=(
                f"Routing task '{current_task_qa_ready[0]}' to QA after a successful worker attempt."
            ),
        )

    all_qa_ready = _qa_ready_task_ids(
        task_queue,
        group_by_id=group_by_id,
        limit=QA_DISPATCH_LIMIT,
    )
    if all_qa_ready:
        return SupervisorDecision(
            decision_code=DecisionCode.QA_READY_BATCH,
            next_node="qa_critic",
            target_task_ids=all_qa_ready,
            instructions="Run QA on the next remaining optimistically fixed task.",
            decision_reason=f"Routing task '{all_qa_ready[0]}' to QA.",
        )

    # Collect tasks that still need work
    workable = [t for t in non_terminal if t.status in _WORKABLE_STATUSES]

    # NO_FIX is a deterministic same-task state machine.  Keep it ahead of
    # generic workaround routing so an untrusted decision cannot skip a
    # mitigation stage or
    # let MAX_RETRIES terminate the lifecycle early.
    no_fix_workable = sorted(
        [
            task
            for task in workable
            if task.strategy == RoutingStrategy.CODE_WORKAROUND
            and (
                (
                    task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL
                    and task.status == TaskStatus.PENDING
                )
                or (
                    task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
                    and task.status == TaskStatus.NEEDS_RETRY
                )
            )
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if no_fix_workable:
        target = no_fix_workable[0]
        evaluation = qa_evaluations.get(target.task_id)
        revised_instructions: dict[str, str] = {}
        feedback_by_task: dict[str, str] = {}
        if target.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
            revised_instructions[target.task_id] = build_no_fix_retry_instruction(
                target,
                group_by_id.get(target.parent_group_id),
                evaluation=evaluation,
            )
        if evaluation and evaluation.retry_feedback:
            feedback_by_task[target.task_id] = evaluation.retry_feedback
        return SupervisorDecision(
            decision_code=DecisionCode.NO_FIX_LIFECYCLE,
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions=("Advance the NO_FIX task through its supervisor-owned mitigation stage."),
            decision_reason=(
                f"Deterministic NO_FIX routing selected {target.no_fix_stage.value} "
                f"for task '{target.task_id}'."
            ),
        )

    # All VERSION_BUMP QA failures follow the ordered version stages. A
    # BREAKING_CHANGE is evidence for the next stage, not an immediate pivot.

    exhausted_retries = sorted(
        [
            task
            for task in workable
            if _is_exhausted_update_pivot_candidate(
                task,
                retry_diagnostics_by_task.get(task.task_id),
            )
            and not _has_existing_workaround_child(task, task_queue)
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if exhausted_retries:
        spawn_requests: list[TaskSpawnRequest] = []
        feedback_by_task: dict[str, str] = {}
        for task in exhausted_retries:
            component = (
                group_by_id.get(task.parent_group_id).vulnerable_component
                if task.parent_group_id in group_by_id
                else task.parent_group_id
            )
            eval_ = qa_evaluations.get(task.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[task.task_id] = eval_.retry_feedback
            spawn_requests.append(
                TaskSpawnRequest(
                    parent_task_id=task.task_id,
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction=(
                        f"Original Context: {task.instruction}\n\n"
                        f"Pivot Directive: Implement a code workaround or isolation strategy for {component} "
                        "because manifest-based update remediation appears exhausted after bounded registry-guided retries."
                    ),
                    reason=(
                        "Deterministic fallback: exhausted manifest remediation must pivot "
                        "to a workaround child task."
                    ),
                )
            )
        return SupervisorDecision(
            decision_code=DecisionCode.EXHAUSTED_UPDATE_PIVOT,
            next_node="workaround_subagent",
            target_task_ids=[exhausted_retries[0].task_id],
            spawn_requests=spawn_requests,
            feedback_by_task=feedback_by_task,
            instructions="Pivot exhausted update remediation to workaround child tasks.",
            decision_reason=(
                f"Retry diagnostics show {len(exhausted_retries)} update task(s) no longer have a remaining manifest-based update path."
            ),
        )

    retry_version_bump = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.VERSION_BUMP
            and t.status == TaskStatus.NEEDS_RETRY
            and not _has_existing_workaround_child(t, task_queue)
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if retry_version_bump:
        batch = retry_version_bump[:UPDATE_DISPATCH_LIMIT]
        feedback_by_task: dict[str, str] = {}
        revised_instructions: dict[str, str] = {}
        for task in batch:
            evaluation = qa_evaluations.get(task.task_id)
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[task.task_id] = evaluation.retry_feedback
            revised_instructions[task.task_id] = _build_high_level_retry_instruction(
                task,
                group_by_id.get(task.parent_group_id),
                evaluation,
                retry_diagnostics_by_task.get(task.task_id),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.RETRY_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions="Route the retry-bound dependency task back to the update worker with its high-level retry goal.",
            decision_reason=(
                f"Routing retry VERSION_BUMP task '{batch[0].task_id}' to update_subagent for registry-guided evidence gathering."
            ),
        )

    # VERSION_BUMP tasks route to update_subagent one at a time for non-retry work.
    version_bump = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.VERSION_BUMP
            and t.status != TaskStatus.NEEDS_RETRY
            and not _has_existing_workaround_child(t, task_queue)
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if version_bump:
        batch = version_bump[:UPDATE_DISPATCH_LIMIT]
        feedback_by_task: dict[str, str] = {}
        for t in batch:
            eval_ = qa_evaluations.get(t.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[t.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            instructions="Apply the required version bump in the package manifest for this task only.",
            decision_reason=(f"Routing VERSION_BUMP task '{batch[0].task_id}' to update_subagent."),
        )

    # CODE_WORKAROUND tasks: send exactly one at a time to workaround_subagent
    workaround = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.CODE_WORKAROUND
            and t.no_fix_stage is None
            and t.retry_count < MAX_RETRIES
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if workaround:
        target = workaround[0]
        eval_ = qa_evaluations.get(target.task_id)
        feedback: dict[str, str] = {}
        revised_instructions: dict[str, str] = {}
        if eval_ and eval_.retry_feedback:
            feedback[target.task_id] = eval_.retry_feedback
        if target.status == TaskStatus.NEEDS_RETRY and eval_:
            revised_instructions[target.task_id] = _build_workaround_retry_instruction(
                target,
                eval_,
                group_by_id.get(target.parent_group_id),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.WORKAROUND_DISPATCH,
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback,
            revised_instructions=revised_instructions,
            instructions="Apply the minimal safe code workaround for this vulnerability.",
            decision_reason=(f"Routing task '{target.task_id}' to workaround_subagent."),
        )

    # Unexpected: no workable tasks found â†’ teardown as safe default
    return SupervisorDecision(
        decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
        next_node="teardown",
        target_task_ids=[],
        instructions="No actionable tasks remain.",
        decision_reason=("Deterministic fallback: no workable tasks found, routing to teardown."),
    )


def _reconcile_results(
    state: OrchestratorState,
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    *,
    qa_evaluations: dict[str, QAEvaluation] | None = None,
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    consistency_events: list[StateConsistencyEvent] | None = None,
    auto_constraints: list[str] | None = None,
    errors: list[str] | None = None,
) -> ReconciliationResult:
    """Project worker/QA outcomes into a detached reconciliation result.

    This phase deliberately does not select a route.  It is small enough to
    run in replay tests without constructing an LLM or touching the graph.
    The production node retains its richer attempt-correlation reducer; this
    helper exposes the same state-machine boundary for focused callers.
    """

    projected_tasks = {task_id: task.model_copy() for task_id, task in task_queue.items()}
    projected_evaluations = dict(qa_evaluations or state.get("qa_evaluations", {}) or {})
    projected_diagnostics = dict(
        retry_diagnostics_by_task or state.get("retry_diagnostics_by_task", {}) or {}
    )
    projected_events = list(consistency_events or [])
    projected_constraints = list(auto_constraints or [])
    projected_errors = list(errors or [])

    if state.get("status") in {"qa_completed", "qa_failed"}:
        for task_id in sorted(projected_evaluations):
            evaluation = projected_evaluations[task_id]
            task = projected_tasks.get(task_id)
            if task is None or task.status != TaskStatus.OPTIMISTICALLY_FIXED:
                continue
            if evaluation.passed:
                if validate_transition(task.status, TaskStatus.QA_PASSED):
                    projected_tasks[task_id] = task.model_copy(
                        update={"status": TaskStatus.QA_PASSED}
                    )
            elif evaluation.contract_error or evaluation.evidence_inconclusive:
                # QA contract/evidence failures rerun the same candidate
                # without advancing remediation or consuming a worker retry.
                continue
            elif validate_transition(task.status, TaskStatus.NEEDS_RETRY):
                projected_tasks[task_id] = task.model_copy(
                    update={
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": task.retry_count + 1,
                    }
                )

    return ReconciliationResult(
        task_queue=projected_tasks,
        qa_evaluations=projected_evaluations,
        retry_diagnostics_by_task=projected_diagnostics,
        consistency_events=projected_events,
        auto_constraints=projected_constraints,
        errors=projected_errors,
    )


def _validate_invariants(
    task_queue: dict[str, RemediationTask],
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    target_task_ids: list[str],
    next_node: str,
) -> tuple[list[StateConsistencyEvent], list[str]]:
    """Validate committed state without mutating the caller's task queue."""

    projected_tasks = {task_id: task.model_copy() for task_id, task in task_queue.items()}
    return _validate_committed_state(
        projected_tasks,
        dict(attempt_snapshots_by_id),
        dict(retry_plans_by_task),
        dict(retry_diagnostics_by_task),
        list(target_task_ids),
        next_node,
    )


def _calculate_eligible_actions(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    *,
    active_target_task_ids: list[str] | None = None,
    current_status: str = "",
    triage_required: bool = False,
) -> EligibleActions:
    """Return the pure eligibility projection consumed by routing."""

    ordered = sorted(task_queue.values(), key=lambda task: _task_sort_key(task, group_by_id))
    non_terminal = [task for task in ordered if task.status not in _TERMINAL_STATUSES]
    workable = [task for task in non_terminal if task.status in _WORKABLE_STATUSES]
    current_qa = _qa_ready_task_ids(
        task_queue,
        preferred_ids=active_target_task_ids or [],
        group_by_id=group_by_id,
    )
    all_qa = _qa_ready_task_ids(task_queue, group_by_id=group_by_id)
    no_fix = [
        task
        for task in workable
        if task.strategy == RoutingStrategy.CODE_WORKAROUND
        and task.no_fix_stage
        in {
            NoFixMitigationStage.PACKAGE_REMOVAL,
            NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        }
    ]
    exhausted = [
        task
        for task in workable
        if _is_exhausted_update_pivot_candidate(task, retry_diagnostics_by_task.get(task.task_id))
        and not _has_existing_workaround_child(task, task_queue)
    ]
    retries = [
        task
        for task in workable
        if task.strategy == RoutingStrategy.VERSION_BUMP
        and task.status == TaskStatus.NEEDS_RETRY
        and task not in exhausted
        and not _has_existing_workaround_child(task, task_queue)
    ]
    pending_updates = [
        task
        for task in workable
        if task.strategy == RoutingStrategy.VERSION_BUMP
        and task.status == TaskStatus.PENDING
        and not _has_existing_workaround_child(task, task_queue)
    ]
    workarounds = [
        task
        for task in workable
        if task.strategy == RoutingStrategy.CODE_WORKAROUND and task.no_fix_stage is None
    ]
    return EligibleActions(
        non_terminal_tasks=[task.task_id for task in non_terminal],
        qa_ready_task_ids=(
            current_qa if current_status != "qa_completed" and current_qa else all_qa
        ),
        workable_tasks=[task.task_id for task in workable],
        no_fix_workable=[task.task_id for task in no_fix],
        exhausted_pivots=[task.task_id for task in exhausted],
        retry_version_bumps=[task.task_id for task in retries],
        new_version_bumps=[task.task_id for task in pending_updates],
        workaround_tasks=[task.task_id for task in workarounds],
        triage_required=triage_required
        and current_status in {"qa_completed", "qa_failed", "final_scan_completed"},
    )


def _select_deterministic_action(
    eligible: EligibleActions,
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> SupervisorDecision:
    """Select the same fixed-priority action as the authoritative router."""

    return _deterministic_routing(
        task_queue,
        group_by_id,
        qa_evaluations,
        retry_diagnostics_by_task,
        active_target_task_ids=eligible.qa_ready_task_ids,
        current_status="",
        triage_required=eligible.triage_required,
    )


def _apply_transition(
    decision: SupervisorDecision,
    task_queue: dict[str, RemediationTask],
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    group_by_id: dict[str, VulnerabilityGroup],
    state_revision: int,
) -> dict[str, Any]:
    """Apply a dispatch projection without performing another route decision."""

    projected_tasks = {task_id: task.model_copy() for task_id, task in task_queue.items()}
    projected_snapshots = dict(attempt_snapshots_by_id)
    target_ids = _normalize_target_task_ids_for_node(
        decision.next_node,
        list(decision.target_task_ids),
        projected_tasks,
        retry_diagnostics_by_task,
        group_by_id,
    )
    if decision.next_node in {"update_subagent", "workaround_subagent"}:
        for task_id in target_ids:
            task = projected_tasks[task_id]
            if task.current_attempt_id is None and task.instruction:
                plan = retry_plans_by_task.get(task_id)
                diagnostics = retry_diagnostics_by_task.get(task_id)
                allowed_versions, allowed_dependency_types = (
                    _ordered_update_candidates(task, plan=plan, diagnostics=diagnostics)
                    if decision.next_node == "update_subagent"
                    else ([], [])
                )
                committed, _snapshot = _create_attempt_snapshot(
                    task,
                    dispatch_node=decision.next_node,
                    snapshots_by_id=projected_snapshots,
                    state_revision=state_revision,
                    plan_id=(plan.plan_id if plan is not None else None),
                    allowed_target_versions=allowed_versions,
                    allowed_dependency_types=allowed_dependency_types,
                )
                projected_tasks[task_id] = committed
    return {
        "task_queue": projected_tasks,
        "attempt_snapshots_by_id": projected_snapshots,
        "next_routing_step": decision.next_node,
        "active_target_task_ids": target_ids,
        "decision_code": decision.decision_code,
    }


def _emit_audit(
    decision: SupervisorDecision,
    consistency_events: list[StateConsistencyEvent],
    state_revision: int,
) -> AuditRecord:
    """Build the typed audit record for a deterministic decision."""

    return AuditRecord(
        decision_code=decision.decision_code or DecisionCode.INVALID_LLM_DECISION,
        next_node=decision.next_node,
        target_task_ids=list(decision.target_task_ids),
        reasoning=decision.decision_reason,
        state_revision=state_revision,
        consistency_events=list(consistency_events),
    )


def _no_fix_decision_requires_fallback(
    decision: SupervisorDecision,
    task_queue: dict[str, RemediationTask],
) -> bool:
    """Return whether an untrusted router decision violates NO_FIX routing."""
    actionable = [
        task
        for task in task_queue.values()
        if task.status in _WORKABLE_STATUSES
        and task.strategy == RoutingStrategy.CODE_WORKAROUND
        and task.no_fix_stage
        in {
            NoFixMitigationStage.PACKAGE_REMOVAL,
            NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        }
    ]
    if not actionable:
        return False

    expected_task_id = actionable[0].task_id
    if actionable[0].status == TaskStatus.OPTIMISTICALLY_FIXED:
        return (
            decision.next_node != "qa_critic"
            or expected_task_id not in decision.target_task_ids
            or expected_task_id in decision.unfixable_task_ids
            or decision.task_status_updates.get(expected_task_id) == TaskStatus.UNFIXABLE
            or decision.updated_task_strategies.get(expected_task_id)
            not in (None, RoutingStrategy.CODE_WORKAROUND)
        )
    if decision.next_node != "workaround_subagent":
        return True
    if decision.target_task_ids != [expected_task_id]:
        return True
    if expected_task_id in decision.unfixable_task_ids:
        return True
    if decision.task_status_updates.get(expected_task_id) == TaskStatus.UNFIXABLE:
        return True
    if decision.updated_task_strategies.get(expected_task_id) not in (
        None,
        RoutingStrategy.CODE_WORKAROUND,
    ):
        return True
    return any(request.parent_task_id == expected_task_id for request in decision.spawn_requests)


def _validate_committed_state(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._validate_committed_state(*args, **kwargs)


def _normalize_target_task_ids_for_node(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._normalize_target_task_ids_for_node(*args, **kwargs)


def _ordered_update_candidates(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._ordered_update_candidates(*args, **kwargs)


def _create_attempt_snapshot(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._create_attempt_snapshot(*args, **kwargs)
