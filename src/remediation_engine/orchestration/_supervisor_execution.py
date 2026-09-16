"""Private supervisor state-transition and projection helpers.

The public facade binds dynamic dependencies around these implementations so
existing monkeypatches of ``supervisor_node`` symbols remain effective.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
from collections.abc import Iterator
from typing import Any

from remediation_engine.contracts.decision_codes import (
    validate_transition as _default_validate_transition,
)
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    NoFixMitigationStage,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
    SupervisorRetryPlan,
    TaskAttemptSnapshot,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    WorkaroundReplayPlan,
)
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.supervisor_planner import (
    QA_DISPATCH_LIMIT as _DEFAULT_QA_DISPATCH_LIMIT,
)
from remediation_engine.orchestration.supervisor_planner import (
    UPDATE_DISPATCH_LIMIT as _DEFAULT_UPDATE_DISPATCH_LIMIT,
)
from remediation_engine.orchestration.supervisor_planner import (
    instruction_digest as _default_instruction_digest,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES as _DEFAULT_TERMINAL_STATUSES,
)
from remediation_engine.orchestration.supervisor_policy import (
    _WORKABLE_STATUSES as _DEFAULT_WORKABLE_STATUSES,
)
from remediation_engine.orchestration.supervisor_policy import (
    _dispatchable_task_ids_for_status as _default_dispatchable_task_ids_for_status,
)
from remediation_engine.orchestration.supervisor_policy import (
    _is_exhausted_update_pivot_candidate as _default_is_exhausted_update_pivot_candidate,
)
from remediation_engine.orchestration.supervisor_policy import (
    _qa_ready_task_ids as _default_qa_ready_task_ids,
)
from remediation_engine.orchestration.supervisor_routing import (
    _dedupe_consistency_events as _default_dedupe_consistency_events,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage as _default_advance_no_fix_stage,
)
from remediation_engine.orchestration.task_utils import (
    build_no_fix_retry_instruction as _default_build_no_fix_retry_instruction,
)
from remediation_engine.orchestration.task_utils import (
    select_package_fix_plan as _default_select_package_fix_plan,
)

_logger = logging.getLogger(__name__)
_dependencies: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "supervisor_helper_dependencies", default=None
)


def _dependency(name: str, default: Any = None) -> Any:
    current = _dependencies.get()
    if current is not None and name in current:
        return current[name]
    return default


@contextlib.contextmanager
def _bind_dependencies(**dependencies: Any) -> Iterator[None]:
    """Bind facade-owned collaborators for one helper invocation."""
    token = _dependencies.set(dependencies)
    try:
        yield
    finally:
        _dependencies.reset(token)


def _build_consistency_event(*args: Any, **kwargs: Any) -> StateConsistencyEvent:
    builder = _dependency("_build_consistency_event")
    if builder is None:
        raise RuntimeError("supervisor consistency-event builder is unavailable")
    return builder(*args, **kwargs)


def _validate_transition(*args: Any, **kwargs: Any) -> bool:
    return _dependency("validate_transition", _default_validate_transition)(*args, **kwargs)


def _instruction_digest(*args: Any, **kwargs: Any) -> str:
    return _dependency("instruction_digest", _default_instruction_digest)(*args, **kwargs)


def _dispatchable_task_ids_for_status(*args: Any, **kwargs: Any) -> list[str]:
    return _dependency(
        "_dispatchable_task_ids_for_status", _default_dispatchable_task_ids_for_status
    )(*args, **kwargs)


def _is_exhausted_update_pivot_candidate(*args: Any, **kwargs: Any) -> bool:
    return _dependency(
        "_is_exhausted_update_pivot_candidate", _default_is_exhausted_update_pivot_candidate
    )(*args, **kwargs)


def _qa_ready_task_ids(*args: Any, **kwargs: Any) -> list[str]:
    return _dependency("_qa_ready_task_ids", _default_qa_ready_task_ids)(*args, **kwargs)


def _dedupe_consistency_events(*args: Any, **kwargs: Any) -> list[StateConsistencyEvent]:
    return _dependency("_dedupe_consistency_events", _default_dedupe_consistency_events)(
        *args, **kwargs
    )


def _advance_no_fix_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _dependency("advance_no_fix_stage", _default_advance_no_fix_stage)(*args, **kwargs)


def _build_no_fix_retry_instruction(*args: Any, **kwargs: Any) -> str:
    return _dependency("build_no_fix_retry_instruction", _default_build_no_fix_retry_instruction)(
        *args, **kwargs
    )


def _proxy_validate_committed_state(*args: Any, **kwargs: Any) -> Any:
    implementation = _dependency("_validate_committed_state")
    if implementation is None:
        implementation = _impl__validate_committed_state
    return implementation(*args, **kwargs)


def _proxy_update_worker_task_ids(*args: Any, **kwargs: Any) -> Any:
    implementation = _dependency("_update_worker_task_ids")
    if implementation is None:
        implementation = _impl__update_worker_task_ids
    return implementation(*args, **kwargs)


def _proxy_resolve_task_id_from_identifier(*args: Any, **kwargs: Any) -> Any:
    implementation = _dependency("_resolve_task_id_from_identifier")
    if implementation is None:
        implementation = _impl__resolve_task_id_from_identifier
    return implementation(*args, **kwargs)


_ATTEMPT_INPUT_FIELDS = frozenset(
    {
        "task_revision",
        "strategy_stage",
        "selected_version",
        "exhausted_update_path",
        "instruction",
        "strategy",
        "no_fix_stage",
        "qa_policy",
        "target_package_name",
        "target_dependency_type",
        "parent_minimum_version",
    }
)


def _impl__commit_task_transition(
    task_queue: dict[str, RemediationTask],
    task_id: str,
    *,
    updates: dict[str, Any],
    close_attempt: bool = False,
    clear_selected_version: bool = False,
    allow_breaking_change_pivot: bool = False,
    consistency_events: list[StateConsistencyEvent] | None = None,
) -> RemediationTask | None:
    """Commit one coherent supervisor transition for a task.

    ``task_queue`` is the authoritative projection.  This helper makes the
    transition explicit and ensures that any change to worker-input fields is
    either paired with a new task revision or closes the old attempt first.
    Worker successes that are waiting for QA intentionally do not use this
    helper for a status-only update: their current snapshot remains valid QA
    input.  Every replan, surrender, terminalization, and pivot does use it.
    ``allow_breaking_change_pivot`` is retained for the Supervisor-owned
    breaking-change pivot path; the normal ``NEEDS_RETRY -> PIVOTED`` transition
    is now explicit in the shared transition table.
    """
    task = task_queue.get(task_id)
    if task is None:
        return None

    if "status" in updates:
        current_status = task.status
        try:
            new_status = (
                updates["status"]
                if isinstance(updates["status"], TaskStatus)
                else TaskStatus(updates["status"])
            )
        except (TypeError, ValueError):
            new_status = None
        pivot_transition_allowed = (
            allow_breaking_change_pivot
            and current_status == TaskStatus.NEEDS_RETRY
            and new_status == TaskStatus.QA_PASSED
        )
        if new_status is None or (
            new_status != current_status
            and not _validate_transition(current_status, new_status)
            and not pivot_transition_allowed
        ):
            event = _build_consistency_event(
                error_code="INVALID_TRANSITION",
                task_id=task_id,
                expected_attempt_id=task.current_attempt_id,
                received_attempt_id=None,
                action="rejected",
                details=(
                    f"Rejected transition {current_status.value} -> "
                    f"{updates['status']!s}; it is not in VALID_TRANSITIONS."
                ),
            )
            if consistency_events is not None:
                consistency_events.append(event)
            _dependency("logger", _logger).error(
                "supervisor: rejected invalid transition %s -> %s for task '%s'.",
                current_status.value,
                updates["status"],
                task_id,
            )
            return task
        updates = {**updates, "status": new_status}

    committed_updates = dict(updates)
    # Terminal status is a complete transition, not just a status projection.
    # Clear all future worker input here so callers cannot leave a selected
    # version or live attempt for the validator to repair later.
    committed_status = committed_updates.get("status")
    if isinstance(committed_status, TaskStatus) and committed_status in _dependency(
        "_TERMINAL_STATUSES", _DEFAULT_TERMINAL_STATUSES
    ):
        close_attempt = True
        clear_selected_version = True
    input_changed = any(
        field in committed_updates and committed_updates[field] != getattr(task, field)
        for field in _ATTEMPT_INPUT_FIELDS
    )
    if close_attempt:
        committed_updates["current_attempt_id"] = None
        if task.current_attempt_id is not None:
            input_changed = True
    if clear_selected_version:
        committed_updates["selected_version"] = None
        if task.selected_version is not None:
            input_changed = True

    if not committed_updates:
        return task

    if input_changed:
        committed_updates["task_revision"] = task.task_revision + 1

    committed_task = task.model_copy(update=committed_updates)
    task_queue[task_id] = committed_task
    return committed_task


def _impl__validate_committed_state(
    task_queue: dict[str, RemediationTask],
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    active_target_task_ids: list[str],
    next_node: str,
) -> tuple[list[StateConsistencyEvent], list[str]]:
    """Validate the state projection that will be handed to the next node."""
    events: list[StateConsistencyEvent] = []
    errors: list[str] = []

    # Reconcile every task before validating routing.  Worker and QA bridges
    # are not allowed to repair planner-owned fields, so if a stale reducer
    # or compatibility projection paired a task with a different current
    # snapshot, the immutable snapshot wins for an active task.  A terminal
    # task has no authorized future worker input, so its dangling attempt is
    # detached instead of being allowed to leak into the next prompt.
    for task_id, task in list(task_queue.items()):
        if task.status in _dependency("_TERMINAL_STATUSES", _DEFAULT_TERMINAL_STATUSES) and (
            task.current_attempt_id is not None or task.selected_version is not None
        ):
            expected_attempt_id = task.current_attempt_id
            task_queue[task_id] = task.model_copy(
                update={
                    "current_attempt_id": None,
                    "selected_version": None,
                    "task_revision": task.task_revision + 1,
                }
            )
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_FIELDS_NORMALIZED",
                    task_id=task_id,
                    expected_attempt_id=expected_attempt_id,
                    received_attempt_id=expected_attempt_id,
                    action="repaired",
                    details=(
                        "Terminal task cannot retain a current worker attempt or "
                        "a dispatchable selected version."
                    ),
                )
            )
            continue
        if not task.current_attempt_id:
            continue
        snapshot = snapshots_by_id.get(task.current_attempt_id)
        if snapshot is None:
            errors.append(f"supervisor: task {task_id} references a missing attempt snapshot.")
            continue

        snapshot_matches = (
            snapshot.task_id == task.task_id
            and snapshot.task_revision == task.task_revision
            and snapshot.strategy_stage == task.strategy_stage
            and snapshot.no_fix_stage == task.no_fix_stage
            and snapshot.qa_policy == task.qa_policy
            and snapshot.selected_version == task.selected_version
            and snapshot.target_package_name == task.target_package_name
            and snapshot.target_dependency_type == task.target_dependency_type
            and snapshot.parent_minimum_version == task.parent_minimum_version
            and snapshot.selected_plan_issue_ids == task.selected_plan_issue_ids
            and snapshot.instruction == task.instruction
            and snapshot.instruction_digest == _instruction_digest(task.instruction)
            and (
                (
                    snapshot.dispatch_node == "update_subagent"
                    and task.strategy == RoutingStrategy.VERSION_BUMP
                )
                or (
                    snapshot.dispatch_node == "workaround_subagent"
                    and task.strategy == RoutingStrategy.CODE_WORKAROUND
                )
                or snapshot.dispatch_node == "qa_critic"
            )
        )
        if snapshot_matches:
            continue

        task_queue[task_id] = task.model_copy(
            update={
                "task_revision": snapshot.task_revision,
                "current_attempt_id": snapshot.attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "no_fix_stage": snapshot.no_fix_stage,
                "qa_policy": snapshot.qa_policy,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "target_dependency_type": snapshot.target_dependency_type,
                "parent_minimum_version": snapshot.parent_minimum_version,
                "selected_plan_issue_ids": list(snapshot.selected_plan_issue_ids),
                "instruction": snapshot.instruction,
            }
        )
        events.append(
            _build_consistency_event(
                error_code="TASK_SNAPSHOT_REPAIRED",
                task_id=task_id,
                expected_attempt_id=task.current_attempt_id,
                received_attempt_id=snapshot.attempt_id,
                action="repaired",
                details="Active task fields were restored from its committed attempt snapshot.",
            )
        )

    for task_id in list(retry_plans_by_task):
        task = task_queue.get(task_id)
        plan = retry_plans_by_task[task_id]
        if task is None:
            retry_plans_by_task.pop(task_id, None)
            continue
        if task.status in _dependency("_TERMINAL_STATUSES", _DEFAULT_TERMINAL_STATUSES):
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_PLAN_CLEARED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="repaired",
                    details="Removed a retry plan from a terminal task.",
                )
            )
            continue
        if plan.action == "retry_update" and (
            plan.selected_version is None or task.exhausted_update_path
        ):
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="INVALID_RETRY_PLAN_CLEARED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="replanned",
                    details="Cleared a retry plan that cannot be dispatched safely.",
                )
            )
            continue
        if plan.action == "retry_update" and (
            plan.source_task_revision > task.task_revision
            or plan.source_task_revision < max(0, task.task_revision - 1)
            or plan.strategy_stage != task.strategy_stage
            or plan.selected_version != task.selected_version
            or plan.exact_instruction != task.instruction
            or plan.exhausted_update_path != task.exhausted_update_path
        ):
            retry_plans_by_task.pop(task_id, None)
            events.append(
                _build_consistency_event(
                    error_code="RETRY_PLAN_TASK_CONTRADICTION",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="replanned",
                    details="Cleared a retry plan that disagreed with the committed task queue.",
                )
            )
            continue

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        if task.status in _dependency("_TERMINAL_STATUSES", _DEFAULT_TERMINAL_STATUSES):
            errors.append(f"supervisor: terminal task {task_id} remained active.")
            events.append(
                _build_consistency_event(
                    error_code="TERMINAL_TASK_ACTIVE",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=None,
                    action="ignored",
                    details="Terminal task removed from dispatch projection.",
                )
            )
            continue
        if task.current_attempt_id is None:
            errors.append(f"supervisor: active task {task_id} has no attempt snapshot.")
            events.append(
                _build_consistency_event(
                    error_code="ACTIVE_TASK_WITHOUT_ATTEMPT",
                    task_id=task_id,
                    expected_attempt_id=None,
                    received_attempt_id=None,
                    action="replanned",
                    details="Active target cannot be dispatched without a committed snapshot.",
                )
            )
            continue
        snapshot = snapshots_by_id.get(task.current_attempt_id)
        if snapshot is None:
            errors.append(f"supervisor: active task {task_id} references missing attempt.")
            continue
        if task.qa_policy is None or snapshot.qa_policy is None:
            errors.append(f"supervisor: active task {task_id} has missing QA policy provenance.")
            events.append(
                _build_consistency_event(
                    error_code="MISSING_QA_POLICY_PROVENANCE",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=task.current_attempt_id,
                    action="replanned",
                    details=(
                        "Active task and committed attempt must both carry a "
                        "supervisor-owned QA policy."
                    ),
                )
            )
            continue
        if (
            snapshot.task_revision != task.task_revision
            or snapshot.strategy_stage != task.strategy_stage
            or snapshot.no_fix_stage != task.no_fix_stage
            or snapshot.qa_policy != task.qa_policy
            or snapshot.selected_version != task.selected_version
            or snapshot.selected_plan_issue_ids != task.selected_plan_issue_ids
            or snapshot.instruction != task.instruction
            or snapshot.instruction_digest != _instruction_digest(task.instruction)
        ):
            errors.append(f"supervisor: task {task_id} disagrees with its attempt snapshot.")
            events.append(
                _build_consistency_event(
                    error_code="TASK_SNAPSHOT_CONTRADICTION",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=task.current_attempt_id,
                    action="replanned",
                    details="Task projection and committed attempt snapshot differ.",
                )
            )
        if next_node == "update_subagent" and task.exhausted_update_path:
            errors.append(f"supervisor: exhausted task {task_id} cannot route to update.")

    for task_id, diagnostics in retry_diagnostics_by_task.items():
        task = task_queue.get(task_id)
        if task is None:
            continue
        if diagnostics.selected_version != task.selected_version:
            diagnostics = diagnostics.model_copy(update={"selected_version": task.selected_version})
            retry_diagnostics_by_task[task_id] = diagnostics
            events.append(
                _build_consistency_event(
                    error_code="DIAGNOSTICS_PROJECTION_REPAIRED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=diagnostics.committed_attempt_id,
                    action="repaired",
                    details="Planner-owned selected version restored from task state.",
                )
            )
        if (
            next_node == "update_subagent"
            and task.selected_version
            and task.selected_version.strip().lstrip("vV").lower()
            in {
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            }
        ):
            errors.append(f"supervisor: selected version for {task_id} was already attempted.")
    return events, errors


def _impl_reconcile_phase5_state_before_teardown(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Apply the final supervisor state barrier before teardown.

    Teardown is a cleanup operation, not a routing decision.  It must receive
    a terminal task projection with no retry plans, active targets, or current
    worker inputs.  This function deliberately performs no LLM calls and uses
    the same validator as the supervisor return path, so direct teardown
    callers and graph executions share the same invariant.
    """
    task_queue: dict[str, RemediationTask] = {
        task_id: task.model_copy() for task_id, task in dict(state.get("task_queue", {})).items()
    }
    snapshots_by_id: dict[str, TaskAttemptSnapshot] = dict(state.get("attempt_snapshots_by_id", {}))
    retry_plans_by_task: dict[str, SupervisorRetryPlan] = dict(state.get("retry_plans_by_task", {}))
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    prior_events = list(state.get("consistency_events", []) or [])
    prior_event_keys = {
        (event.task_id, event.received_attempt_id, event.error_code) for event in prior_events
    }

    # A pivot child owns the remediation outcome. Repair stale parent status
    # projections before teardown so a historical QA_PASSED parent cannot
    # conceal an unresolved child in the final task queue.
    pivot_repair_events: list[StateConsistencyEvent] = []
    children_by_parent: dict[str, list[RemediationTask]] = {}
    for task in task_queue.values():
        if task.parent_task_id:
            children_by_parent.setdefault(task.parent_task_id, []).append(task)
    for parent_id, children in sorted(children_by_parent.items()):
        parent = task_queue.get(parent_id)
        if parent is None or parent.status == TaskStatus.PIVOTED:
            continue
        task_queue[parent_id] = parent.model_copy(
            update={
                "status": TaskStatus.PIVOTED,
                "current_attempt_id": None,
                "selected_version": None,
                "task_revision": parent.task_revision + 1,
            }
        )
        pivot_repair_events.append(
            _build_consistency_event(
                error_code="PIVOT_PARENT_STATUS_REPAIRED",
                task_id=parent_id,
                expected_attempt_id=parent.current_attempt_id,
                received_attempt_id=children[0].current_attempt_id,
                action="repaired",
                details=(
                    "Parent task status was normalized to PIVOTED because a child task "
                    "owns the current remediation outcome."
                ),
            )
        )

    events, errors = _proxy_validate_committed_state(
        task_queue,
        snapshots_by_id,
        retry_plans_by_task,
        retry_diagnostics_by_task,
        [],
        "teardown",
    )
    new_events = [
        event
        for event in _dedupe_consistency_events([*pivot_repair_events, *events])
        if (event.task_id, event.received_attempt_id, event.error_code) not in prior_event_keys
    ]
    prior_errors = set(state.get("errors", []) or [])
    new_errors = list(dict.fromkeys(error for error in errors if error not in prior_errors))

    return {
        "task_queue": task_queue,
        "retry_plans_by_task": retry_plans_by_task,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "workspace_rollback_anchors_by_task": {},
        "active_target_task_ids": [],
        "next_routing_step": "teardown",
        "state_revision": int(state.get("state_revision", 0)) + 1,
        "consistency_events": new_events,
        "errors": new_errors,
    }


def _impl__update_worker_task_ids(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    preferred_ids: list[str] | None = None,
    limit: int | None = _dependency("UPDATE_DISPATCH_LIMIT", _DEFAULT_UPDATE_DISPATCH_LIMIT),
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    task_ids = _dispatchable_task_ids_for_status(
        task_queue,
        set(_dependency("_WORKABLE_STATUSES", _DEFAULT_WORKABLE_STATUSES)),
        preferred_ids=preferred_ids,
        strategy=RoutingStrategy.VERSION_BUMP,
        group_by_id=group_by_id,
    )
    dispatchable = [
        task_id
        for task_id in task_ids
        if not _is_exhausted_update_pivot_candidate(
            task_queue[task_id],
            retry_diagnostics_by_task.get(task_id),
        )
    ]
    if limit is not None:
        return dispatchable[:limit]
    return dispatchable


def _impl__normalize_target_task_ids_for_node(
    next_node: str,
    target_task_ids: list[str],
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
    allow_cluster: bool = False,
) -> list[str]:
    """Clamp returned active targets to the lifecycle state accepted by next_node."""
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    if next_node == "qa_critic":
        return _qa_ready_task_ids(
            task_queue,
            preferred_ids=target_task_ids,
            group_by_id=group_by_id,
            limit=(
                None
                if allow_cluster
                else _dependency("QA_DISPATCH_LIMIT", _DEFAULT_QA_DISPATCH_LIMIT)
            ),
        )
    if next_node == "update_subagent":
        return _proxy_update_worker_task_ids(
            task_queue,
            retry_diagnostics_by_task,
            preferred_ids=target_task_ids,
            limit=(
                None
                if allow_cluster
                else _dependency("UPDATE_DISPATCH_LIMIT", _DEFAULT_UPDATE_DISPATCH_LIMIT)
            ),
            group_by_id=group_by_id,
        )
    if next_node == "workaround_subagent":
        return _dispatchable_task_ids_for_status(
            task_queue,
            set(_dependency("_WORKABLE_STATUSES", _DEFAULT_WORKABLE_STATUSES)),
            preferred_ids=target_task_ids,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            limit=1,
            group_by_id=group_by_id,
        )
    return []


def _impl__resolve_task_id_from_identifier(
    identifier: str,
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> str | None:
    """Resolve an identifier only when it is an exact task ID.

    Supervisor projections are task-keyed. Parent group IDs are intentionally
    not accepted here because a group can own multiple independent tasks.
    """
    del active_target_task_ids
    return identifier if identifier in task_queue else None


def _impl__normalize_qa_evaluations_for_tasks(
    qa_evaluations: dict[str, QAEvaluation],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, QAEvaluation]:
    """Keep only evaluations explicitly keyed by an active task ID."""
    active_ids = set(active_target_task_ids)
    normalized: dict[str, QAEvaluation] = {}
    for task_id, evaluation in qa_evaluations.items():
        if (
            task_id not in active_ids
            or task_id not in task_queue
            or evaluation.task_id != task_id
            or task_id in normalized
        ):
            continue
        normalized[task_id] = evaluation
    return normalized


def _impl__constraint_entry_for_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    """Build a deterministic constraints-ledger entry for a QA-passed task."""
    component = (group.vulnerable_component or task.parent_group_id).strip()
    fix_plan = _default_select_package_fix_plan(group, task.strategy).plan

    if task.strategy == RoutingStrategy.VERSION_BUMP:
        fixed_version = (fix_plan.fixed_version if fix_plan else None) or "unknown"
        return f"{component}: keep resolved version at {fixed_version}"

    return f"{component}: preserve validated security workaround"


def _impl__missing_retry_revised_instructions(
    next_node: str,
    target_task_ids: list[str],
    revised_instructions: dict[str, str],
    task_queue: dict[str, RemediationTask],
) -> list[str]:
    """Return retry-bound update targets that are missing exact revised instructions."""
    if next_node != "update_subagent":
        return []

    missing: list[str] = []
    for task_id in target_task_ids:
        task = task_queue.get(task_id)
        if task is None or task.status != TaskStatus.NEEDS_RETRY:
            continue
        instruction = revised_instructions.get(task_id, "").strip()
        if not instruction:
            missing.append(task_id)
            continue
        # Retry workers are execution-only: a retry instruction is invalid
        # unless it carries both the strategy stage and a concrete semver.
        if "strategy stage" not in instruction.lower() or not re.search(
            r"(?<!\d)v?\d+\.\d+\.\d+(?!\d)", instruction
        ):
            missing.append(task_id)
    return missing


def _impl__latest_action_summary_by_task(
    action_summaries: list[AgentActionSummary],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, AgentActionSummary]:
    """Return the most recent action summary keyed by resolved task_id."""
    latest: dict[str, AgentActionSummary] = {}
    for summary in action_summaries:
        resolved_task_id = _proxy_resolve_task_id_from_identifier(
            summary.task_id,
            task_queue,
            active_target_task_ids,
        )
        if resolved_task_id is None:
            continue
        latest[resolved_task_id] = summary
    return latest


def _impl__no_fix_failure_transition(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    *,
    evaluation: QAEvaluation | None = None,
    failure_feedback: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build the supervisor-owned transition after a failed NO_FIX attempt.

    Returns a task update mapping and whether the next attempt must reset the
    task-local workspace to the package-removal stage baseline.
    """
    updates = _advance_no_fix_stage(task)
    next_stage = updates.get("no_fix_stage")
    if next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        updates["qa_policy"] = QAPolicy.NO_FIX_CODE_REMOVAL
    reset_workspace = next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    if reset_workspace:
        retry_task = task.model_copy(update=updates)
        updates["instruction"] = _build_no_fix_retry_instruction(
            retry_task,
            group,
            evaluation=evaluation,
            failure_feedback=failure_feedback,
        )
    return updates, reset_workspace


def _impl__reset_no_fix_replay_plan(
    replay_plan: WorkaroundReplayPlan | None,
) -> WorkaroundReplayPlan | None:
    """Return a replay plan that restores the stage baseline without replaying edits."""
    if replay_plan is None:
        return None
    return replay_plan.model_copy(
        update={
            "successful_edit_sets": [],
            "validated_files": [],
            "validation_calls": 0,
            "per_gate_results": {},
            "final_selected_targeted_test": None,
            "original_to_alternative_test_mapping": {},
            "alternative_test_mapping_evidence": {},
            "alternative_test_mapping_details": {},
        }
    )
