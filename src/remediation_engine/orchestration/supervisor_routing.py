"""Deterministic Supervisor routing decisions and audit helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    NoFixMitigationStage,
    PortfolioPlan,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
    SupervisorDecision,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.contracts.supervisor_phases import AuditRecord
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


def _portfolio_cluster_targets(
    portfolio_plan: PortfolioPlan | None,
    task_queue: dict[str, RemediationTask],
    *,
    qa: bool,
    preferred_ids: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Return the first active cluster only when its next action is ready.

    A blocked head cluster is returned with no targets so callers can prevent
    fallback routing from dispatching a later batch out of portfolio order.
    """
    if portfolio_plan is None:
        return None, []
    preferred = set(preferred_ids or [])
    clusters_by_id = {cluster.cluster_id: cluster for cluster in portfolio_plan.clusters}
    cluster_positions = {
        cluster_id: position for position, cluster_id in enumerate(portfolio_plan.cluster_order)
    }
    has_cycle_fallback = any(
        "cycle" in diagnostic.casefold() for diagnostic in portfolio_plan.diagnostics
    )

    def upstreams_terminal(cluster: Any) -> bool:
        members = set(cluster.task_ids)
        return all(
            dependency.upstream_task_id in members
            or task_queue.get(dependency.upstream_task_id) is None
            or task_queue[dependency.upstream_task_id].status in _TERMINAL_STATUSES
            or (
                has_cycle_fallback
                and cluster_positions.get(
                    portfolio_plan.task_to_cluster.get(dependency.upstream_task_id, ""),
                    -1,
                )
                > cluster_positions.get(cluster.cluster_id, -1)
            )
            for dependency in cluster.dependencies
        )

    ordered_cluster_ids = list(portfolio_plan.cluster_order)
    if preferred:
        preferred_cluster_ids = [
            portfolio_plan.task_to_cluster[task_id]
            for task_id in portfolio_plan.task_order
            if task_id in preferred and task_id in portfolio_plan.task_to_cluster
        ]
        ordered_cluster_ids = list(dict.fromkeys(preferred_cluster_ids + ordered_cluster_ids))

    for cluster_id in ordered_cluster_ids:
        cluster = clusters_by_id.get(cluster_id)
        if cluster is None:
            return cluster_id, []
        tasks = [task_queue.get(task_id) for task_id in cluster.task_ids]
        if any(task is None for task in tasks):
            return cluster_id, []
        concrete_tasks = [task for task in tasks if task is not None]
        if all(task.status in _TERMINAL_STATUSES for task in concrete_tasks):
            continue
        if not getattr(cluster, "dispatchable", True) or not upstreams_terminal(cluster):
            logger.warning(
                "portfolio head cluster %s is blocked; preserving later batches",
                cluster_id,
            )
            return cluster_id, []
        if qa:
            if all(task.status == TaskStatus.OPTIMISTICALLY_FIXED for task in concrete_tasks):
                return cluster_id, list(cluster.task_ids)
            return cluster_id, []
        if any(task.status in _TERMINAL_STATUSES for task in concrete_tasks):
            return cluster_id, []
        if any(task.current_attempt_id is not None for task in concrete_tasks):
            return cluster_id, []
        if not all(task.status in _WORKABLE_STATUSES for task in concrete_tasks):
            return cluster_id, []
        if any(_has_existing_workaround_child(task, task_queue) for task in concrete_tasks):
            return cluster_id, []
        strategies = {task.strategy for task in concrete_tasks}
        if len(strategies) != 1 or not strategies.issubset(
            {RoutingStrategy.VERSION_BUMP, RoutingStrategy.CODE_WORKAROUND}
        ):
            return cluster_id, []
        if strategies == {RoutingStrategy.CODE_WORKAROUND} and len(concrete_tasks) != 1:
            logger.warning(
                "portfolio workaround cluster %s has multiple tasks; preserving later batches",
                cluster_id,
            )
            return cluster_id, []
        return cluster_id, list(cluster.task_ids)
    return None, []


def _build_consistency_event(
    *,
    error_code: str,
    task_id: str | None,
    expected_attempt_id: str | None,
    received_attempt_id: str | None,
    action: str,
    details: str,
) -> StateConsistencyEvent:
    # ``deferred`` was used by the registry-pending branch before it was
    # added to the typed event contract.  Keep this compatibility boundary
    # fail-closed: a deferred proposal is a rejected dispatch, and must never
    # crash the Supervisor while constructing its diagnostic event.
    if action == "deferred":
        action = "rejected"
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
    portfolio_plan: PortfolioPlan | None = None,
    delta_replay_task_ids: Iterable[str] = (),
) -> SupervisorDecision:
    """
    Pure-Python routing used as the Supervisor's authoritative state machine.

    Implements the fixed Supervisor priority rules.
    """
    tasks = sorted(task_queue.values(), key=lambda task: _task_sort_key(task, group_by_id))
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
    # Defer incremental scan triage until ordinary remediation work is exhausted.
    if not non_terminal and triage_required:
        return SupervisorDecision(
            decision_code=DecisionCode.TRIAGE_REQUIRED,
            next_node="triage",
            target_task_ids=[],
            instructions="Route cumulative accepted scan findings to post-remediation triage.",
            decision_reason=(
                "No actionable remediation tasks remain and accepted scans contain "
                "baseline-absent or remaining target findings."
            ),
        )

    if not group_by_id and not task_queue:
        return SupervisorDecision(
            decision_code=DecisionCode.NO_VALID_GROUPS,
            next_node="teardown",
            target_task_ids=[],
            instructions="No valid vulnerability groups are available for remediation.",
            decision_reason="Deterministic routing found no valid groups.",
        )

    # No final full-workspace scan node remains; terminal tasks go to triage
    # when accepted incremental findings exist, otherwise directly to teardown.
    if not non_terminal:
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
    portfolio_cluster_id, portfolio_qa_targets = _portfolio_cluster_targets(
        portfolio_plan,
        task_queue,
        qa=True,
        preferred_ids=list(active_target_task_ids or []),
    )
    if portfolio_cluster_id and portfolio_qa_targets:
        is_batch = len(portfolio_qa_targets) > 1
        return SupervisorDecision(
            decision_code=(DecisionCode.QA_READY_BATCH if is_batch else DecisionCode.QA_READY),
            next_node="qa_critic",
            target_task_ids=portfolio_qa_targets,
            cluster_id=portfolio_cluster_id if is_batch else None,
            instructions=(
                "Run QA atomically across the completed package cluster."
                if is_batch
                else "Run QA on the next completed portfolio task."
            ),
            decision_reason=(
                f"All package tasks in cluster '{portfolio_cluster_id}' are ready for QA."
            ),
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
    if all_qa_ready and portfolio_cluster_id is None:
        return SupervisorDecision(
            decision_code=DecisionCode.QA_READY_BATCH,
            next_node="qa_critic",
            target_task_ids=all_qa_ready,
            instructions="Run QA on the next remaining optimistically fixed task.",
            decision_reason=f"Routing task '{all_qa_ready[0]}' to QA.",
        )

    # Collect tasks that still need work
    workable = [t for t in non_terminal if t.status in _WORKABLE_STATUSES]

    portfolio_work_cluster_id, portfolio_work_targets = _portfolio_cluster_targets(
        portfolio_plan,
        task_queue,
        qa=False,
        preferred_ids=list(active_target_task_ids or []),
    )
    portfolio_work_target_ids = set(portfolio_work_targets)
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
            and (
                portfolio_plan is None
                or task.task_id not in portfolio_plan.task_ids
                or task.task_id in portfolio_work_target_ids
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

    delta_replay_task_ids = set(delta_replay_task_ids)
    delta_replay_targets = [
        task_id
        for task_id in portfolio_work_targets
        if task_id in delta_replay_task_ids
        and task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
    ]
    if delta_replay_targets:
        target = task_queue[delta_replay_targets[0]]
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[target.task_id],
            instructions=(
                "Run the restored delta peer's committed candidate as a singleton "
                "so authoritative QA can validate it independently."
            ),
            decision_reason=(
                f"Dispatching restored delta peer '{target.task_id}' from the "
                "committed portfolio head before unrelated retries."
            ),
        )

    # Tactical reasoning gets the first opportunity to pivot after a QA
    # failure. This deterministic ladder is fallback-only, so a
    # BREAKING_CHANGE can become a workaround without burning later versions.

    exhausted_retries = sorted(
        [
            task
            for task in workable
            if _is_exhausted_update_pivot_candidate(
                task,
                retry_diagnostics_by_task.get(task.task_id),
            )
            and not _has_existing_workaround_child(task, task_queue)
            and (
                portfolio_plan is None
                or task.task_id not in portfolio_plan.task_ids
                or task.task_id in portfolio_work_target_ids
            )
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

    # A conclusive QA failure is still the active task's responsibility, even
    # when the portfolio plan has already advanced to another ready cluster.
    # Give a singleton retry to tactical planning before dispatching unrelated
    # work. Never split a committed multi-task cluster.
    active_retry_task = None
    active_retry_evaluation = None
    if len(active_target_task_ids or ()) == 1:
        active_task_id = active_target_task_ids[0]
        candidate_task = task_queue.get(active_task_id)
        candidate_evaluation = qa_evaluations.get(active_task_id)
        if (
            candidate_task is not None
            and candidate_task.strategy == RoutingStrategy.VERSION_BUMP
            and candidate_task.status == TaskStatus.NEEDS_RETRY
            and candidate_task.current_attempt_id is None
            and candidate_evaluation is not None
            and not candidate_evaluation.passed
            and not candidate_evaluation.contract_error
            and not candidate_evaluation.evidence_inconclusive
            and not _has_existing_workaround_child(candidate_task, task_queue)
        ):
            cluster_id = (
                portfolio_plan.task_to_cluster.get(active_task_id)
                if portfolio_plan is not None
                else None
            )
            active_cluster = (
                next(
                    (
                        cluster
                        for cluster in portfolio_plan.clusters
                        if cluster.cluster_id == cluster_id
                    ),
                    None,
                )
                if portfolio_plan is not None and cluster_id is not None
                else None
            )
            if active_cluster is None or len(active_cluster.task_ids) == 1:
                active_retry_task = candidate_task
                active_retry_evaluation = candidate_evaluation

    if active_retry_task is not None and active_retry_evaluation is not None:
        retry_instruction = _build_high_level_retry_instruction(
            active_retry_task,
            group_by_id.get(active_retry_task.parent_group_id),
            active_retry_evaluation,
            retry_diagnostics_by_task.get(active_retry_task.task_id),
        )
        retry_feedback = (
            {active_retry_task.task_id: active_retry_evaluation.retry_feedback}
            if active_retry_evaluation.retry_feedback
            else {}
        )
        return SupervisorDecision(
            decision_code=DecisionCode.RETRY_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[active_retry_task.task_id],
            feedback_by_task=retry_feedback,
            revised_instructions={active_retry_task.task_id: retry_instruction},
            instructions=(
                "Route the active QA-failed dependency task through tactical/retry planning "
                "before dispatching unrelated portfolio work."
            ),
            decision_reason=(
                f"Prioritizing active QA retry task '{active_retry_task.task_id}' "
                "before unrelated portfolio dispatch."
            ),
        )

    portfolio_cluster_id = portfolio_work_cluster_id
    portfolio_update_targets = portfolio_work_targets
    if portfolio_cluster_id and portfolio_update_targets:
        target = task_queue[portfolio_update_targets[0]]
        if target.strategy == RoutingStrategy.CODE_WORKAROUND:
            if target.no_fix_stage is not None or target.retry_count >= MAX_RETRIES:
                return SupervisorDecision(
                    decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                    next_node="teardown",
                    target_task_ids=[],
                    instructions="Stop safely; the committed workaround batch cannot be dispatched.",
                    decision_reason=(
                        f"Portfolio head cluster '{portfolio_cluster_id}' contains a "
                        f"non-dispatchable workaround task '{target.task_id}'."
                    ),
                )
            evaluation = qa_evaluations.get(target.task_id)
            feedback_by_task: dict[str, str] = {}
            revised_instructions: dict[str, str] = {}
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[target.task_id] = evaluation.retry_feedback
            if target.status == TaskStatus.NEEDS_RETRY and evaluation:
                revised_instructions[target.task_id] = _build_workaround_retry_instruction(
                    target,
                    evaluation,
                    group_by_id.get(target.parent_group_id),
                )
            return SupervisorDecision(
                decision_code=DecisionCode.WORKAROUND_DISPATCH,
                next_node="workaround_subagent",
                target_task_ids=[target.task_id],
                feedback_by_task=feedback_by_task,
                revised_instructions=revised_instructions,
                instructions="Apply the next committed portfolio code workaround.",
                decision_reason=(
                    f"Dispatching the ordered workaround task '{target.task_id}' "
                    f"from cluster '{portfolio_cluster_id}'."
                ),
            )
    if portfolio_cluster_id and len(portfolio_update_targets) > 1:
        return SupervisorDecision(
            decision_code=DecisionCode.ATOMIC_CLUSTER_DISPATCH,
            next_node="update_subagent",
            target_task_ids=portfolio_update_targets,
            cluster_id=portfolio_cluster_id,
            instructions="Apply the committed package-cluster update atomically.",
            decision_reason=(
                f"Dispatching the next dependency cluster '{portfolio_cluster_id}' "
                "after all upstream package groups completed."
            ),
        )
    if portfolio_cluster_id and portfolio_update_targets:
        target = task_queue[portfolio_update_targets[0]]
        if target.status == TaskStatus.NEEDS_RETRY:
            evaluation = qa_evaluations.get(target.task_id)
            feedback_by_task: dict[str, str] = {}
            revised_instructions = {
                target.task_id: _build_high_level_retry_instruction(
                    target,
                    group_by_id.get(target.parent_group_id),
                    evaluation,
                    retry_diagnostics_by_task.get(target.task_id),
                )
            }
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[target.task_id] = evaluation.retry_feedback
            return SupervisorDecision(
                decision_code=DecisionCode.RETRY_VERSION_BUMP,
                next_node="update_subagent",
                target_task_ids=[target.task_id],
                feedback_by_task=feedback_by_task,
                revised_instructions=revised_instructions,
                instructions=(
                    "Route the next committed portfolio task back to the update worker "
                    "with its high-level retry goal."
                ),
                decision_reason=(
                    f"Retrying the ready singleton portfolio task '{target.task_id}' "
                    f"from cluster '{portfolio_cluster_id}' before dependent clusters."
                ),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[target.task_id],
            instructions=(
                "Apply the next committed portfolio task update before dispatching "
                "dependent package clusters."
            ),
            decision_reason=(
                f"Dispatching the ready singleton portfolio task '{target.task_id}' "
                f"from cluster '{portfolio_cluster_id}' before dependent clusters."
            ),
        )
    if portfolio_plan is not None:
        active_planned_tasks = [
            task_queue[task_id]
            for task_id in portfolio_plan.task_ids
            if task_id in task_queue and task_queue[task_id].status not in _TERMINAL_STATUSES
        ]
        if portfolio_cluster_id is not None or active_planned_tasks:
            return SupervisorDecision(
                decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                next_node="teardown",
                target_task_ids=[],
                instructions="Stop safely; do not dispatch past the committed portfolio head.",
                decision_reason=(
                    "The committed portfolio head is not dispatchable; later batches remain blocked."
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
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> bool:
    """Return whether an untrusted router decision violates NO_FIX routing."""
    actionable = sorted(
        [
            task
            for task in task_queue.values()
            if task.status in _WORKABLE_STATUSES
            and task.strategy == RoutingStrategy.CODE_WORKAROUND
            and task.no_fix_stage
            in {
                NoFixMitigationStage.PACKAGE_REMOVAL,
                NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
            }
        ],
        key=lambda task: _task_sort_key(task, group_by_id or {}),
    )
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
