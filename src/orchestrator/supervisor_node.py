"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 2 architecture: High-Level Commander
------------------------------------------
The Supervisor now acts as a high-level commander:

  Router:
    A zero-shot ``ChatOpenAI.with_structured_output(SupervisorDecision)`` call
    that decides which worker should run next, whether retry instructions must
    be refreshed, and whether a child task must be spawned for a strategy pivot.

  Guardrails (Python):
    Validate and apply the decision: reject unknown task IDs, clamp cardinality,
    apply copy-on-write task updates, materialize spawn requests, enforce depth
    and queue-size caps.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
build_supervisor_prompt(state) -> str
    Builds the Router prompt text.
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import (
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    QAEvaluation,
    QAAttemptResult,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    SupervisorDecision,
    SupervisorRetryPlan,
    StateConsistencyEvent,
    TaskAttemptSnapshot,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    WorkerAttemptResult,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.orchestrator.subagent_runtime import ToolEvent, run_bounded_subagent_loop
from src.orchestrator.task_utils import build_initial_remediation_task
from src.orchestrator.trajectory_exporter import invoke_with_trajectory
from src.tools.registry_tools import plan_npm_version

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 5
UPDATE_BATCH_SIZE: int = 10

_VALID_NEXT_NODES: Set[str] = {
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "teardown",
}
_DEFAULT_MODEL = "gpt-4o-mini"

# Keep planner stage parsing and validation centralized.  The router prompt
# uses the enum values, but planner scratchpads from older prompts commonly
# use the shorter ``same_major`` spelling.  Accept that spelling explicitly;
# do not let an unrecognised value silently become the task's current stage.
_PLANNER_STAGE_ALIASES: Dict[str, SCARemediationStage] = {
    "osv": SCARemediationStage.OSV_MINIMUM,
    "minimum": SCARemediationStage.OSV_MINIMUM,
    "osv_minimum": SCARemediationStage.OSV_MINIMUM,
    "same_major": SCARemediationStage.NPM_SAME_MAJOR,
    "npm_same_major": SCARemediationStage.NPM_SAME_MAJOR,
    "latest": SCARemediationStage.NPM_LATEST,
    "npm_latest": SCARemediationStage.NPM_LATEST,
    "code_workaround": SCARemediationStage.CODE_WORKAROUND,
}
_SCA_STAGE_ORDER: Dict[SCARemediationStage, int] = {
    SCARemediationStage.OSV_MINIMUM: 0,
    SCARemediationStage.NPM_SAME_MAJOR: 1,
    SCARemediationStage.NPM_LATEST: 2,
    SCARemediationStage.CODE_WORKAROUND: 3,
}

# ---------------------------------------------------------------------------
# Task status helpers
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({
    TaskStatus.QA_PASSED,
    TaskStatus.UNFIXABLE,
})
_WORKABLE_STATUSES = frozenset({
    TaskStatus.PENDING,
    TaskStatus.NEEDS_RETRY,
})


def _dispatchable_task_ids_for_status(
    task_queue: Dict[str, RemediationTask],
    statuses: Set[TaskStatus],
    preferred_ids: Optional[List[str]] = None,
    strategy: Optional[RoutingStrategy] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Return non-terminal task IDs matching status and optional strategy filters."""
    ordered_ids = list(preferred_ids) if preferred_ids is not None else list(task_queue.keys())
    dispatchable: List[str] = []
    seen: Set[str] = set()

    for task_id in ordered_ids:
        if task_id in seen:
            continue
        seen.add(task_id)
        task = task_queue.get(task_id)
        if task is None:
            continue
        if task.status in _TERMINAL_STATUSES:
            continue
        if task.status not in statuses:
            continue
        if strategy is not None and task.strategy != strategy:
            continue
        dispatchable.append(task_id)
        if limit is not None and len(dispatchable) >= limit:
            break

    return dispatchable


def _qa_ready_task_ids(
    task_queue: Dict[str, RemediationTask],
    preferred_ids: Optional[List[str]] = None,
) -> List[str]:
    return _dispatchable_task_ids_for_status(
        task_queue,
        {TaskStatus.OPTIMISTICALLY_FIXED},
        preferred_ids=preferred_ids,
    )


def _is_exhausted_update_pivot_candidate(
    task: RemediationTask,
    diagnostics: Optional[UpdateRetryDiagnostics],
) -> bool:
    """Return True when a retry update task must pivot instead of retrying update."""
    return (
        task.strategy == RoutingStrategy.VERSION_BUMP
        and task.status == TaskStatus.NEEDS_RETRY
        and (
            task.strategy_stage == SCARemediationStage.CODE_WORKAROUND
            or (
                diagnostics is not None
                and (diagnostics.package_abandoned or diagnostics.exhausted_update_path)
            )
        )
    )


def _next_sca_stage(stage: SCARemediationStage) -> SCARemediationStage:
    """Advance one ordered SCA version strategy stage."""
    if stage == SCARemediationStage.OSV_MINIMUM:
        return SCARemediationStage.NPM_SAME_MAJOR
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        return SCARemediationStage.NPM_LATEST
    return SCARemediationStage.CODE_WORKAROUND


def _selection_for_stage(stage: SCARemediationStage) -> Optional[str]:
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        return "same_major"
    if stage == SCARemediationStage.NPM_LATEST:
        return "latest"
    return None


def _instruction_digest(instruction: str) -> str:
    """Return the stable digest used to correlate worker input and output."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _attempts_for_task(
    snapshots_by_id: Dict[str, TaskAttemptSnapshot],
    task_id: str,
) -> List[TaskAttemptSnapshot]:
    return sorted(
        (snapshot for snapshot in snapshots_by_id.values() if snapshot.task_id == task_id),
        key=lambda snapshot: (snapshot.attempt_number, snapshot.created_at),
    )


def _build_consistency_event(
    *,
    error_code: str,
    task_id: Optional[str],
    expected_attempt_id: Optional[str],
    received_attempt_id: Optional[str],
    action: str,
    details: str,
) -> StateConsistencyEvent:
    return StateConsistencyEvent(
        error_code=error_code,
        task_id=task_id,
        expected_attempt_id=expected_attempt_id,
        received_attempt_id=received_attempt_id,
        action=action,  # type: ignore[arg-type]
        details=details,
    )


def _dedupe_consistency_events(
    events: List[StateConsistencyEvent],
) -> List[StateConsistencyEvent]:
    """Keep one consistency event per task/attempt/error tuple."""
    result: List[StateConsistencyEvent] = []
    seen: Set[Tuple[Optional[str], Optional[str], str]] = set()
    for event in events:
        key = (event.task_id, event.received_attempt_id, event.error_code)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def _current_action_summaries(
    action_summaries: List[AgentActionSummary],
    task_queue: Dict[str, RemediationTask],
    limit: int,
) -> List[AgentActionSummary]:
    """Return only summaries belonging to each task's committed attempt."""
    relevant: List[AgentActionSummary] = []
    for summary in action_summaries:
        task = task_queue.get(summary.task_id)
        if task is None:
            continue
        if task.current_attempt_id:
            if summary.attempt_id != task.current_attempt_id:
                continue
        elif summary.attempt_id is not None:
            continue
        relevant.append(summary)
    return relevant[-limit:]


def _create_attempt_snapshot(
    task: RemediationTask,
    *,
    dispatch_node: str,
    snapshots_by_id: Dict[str, TaskAttemptSnapshot],
    state_revision: int,
    plan_id: Optional[str] = None,
) -> Tuple[RemediationTask, TaskAttemptSnapshot]:
    """Commit the exact worker input and return the revised task projection."""
    attempt_id = str(uuid.uuid4())
    task_revision = task.task_revision + 1
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        state_revision=state_revision,
        task_revision=task_revision,
        attempt_number=len(_attempts_for_task(snapshots_by_id, task.task_id)) + 1,
        strategy_stage=task.strategy_stage,
        selected_version=task.selected_version,
        instruction=task.instruction,
        instruction_digest=_instruction_digest(task.instruction),
        dispatch_node=dispatch_node,  # type: ignore[arg-type]
        plan_id=plan_id,
    )
    snapshots_by_id[attempt_id] = snapshot
    updated_task = task.model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
        }
    )
    return updated_task, snapshot


_ATTEMPT_INPUT_FIELDS = frozenset(
    {
        "task_revision",
        "strategy_stage",
        "selected_version",
        "exhausted_update_path",
        "instruction",
        "strategy",
    }
)


def _commit_task_transition(
    task_queue: Dict[str, RemediationTask],
    task_id: str,
    *,
    updates: Dict[str, Any],
    close_attempt: bool = False,
    clear_selected_version: bool = False,
) -> Optional[RemediationTask]:
    """Commit one coherent supervisor transition for a task.

    ``task_queue`` is the authoritative projection.  This helper makes the
    transition explicit and ensures that any change to worker-input fields is
    either paired with a new task revision or closes the old attempt first.
    Worker successes that are waiting for QA intentionally do not use this
    helper for a status-only update: their current snapshot remains valid QA
    input.  Every replan, surrender, terminalization, and pivot does use it.
    """
    task = task_queue.get(task_id)
    if task is None:
        return None

    committed_updates = dict(updates)
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


def _validate_committed_state(
    task_queue: Dict[str, RemediationTask],
    snapshots_by_id: Dict[str, TaskAttemptSnapshot],
    retry_plans_by_task: Dict[str, SupervisorRetryPlan],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    active_target_task_ids: List[str],
    next_node: str,
) -> Tuple[List[StateConsistencyEvent], List[str]]:
    """Validate the state projection that will be handed to the next node."""
    events: List[StateConsistencyEvent] = []
    errors: List[str] = []

    # Reconcile every task before validating routing.  Worker and QA bridges
    # are not allowed to repair planner-owned fields, so if a stale reducer
    # or compatibility projection paired a task with a different current
    # snapshot, the immutable snapshot wins for an active task.  A terminal
    # task has no authorized future worker input, so its dangling attempt is
    # detached instead of being allowed to leak into the next prompt.
    for task_id, task in list(task_queue.items()):
        if task.status in _TERMINAL_STATUSES and (
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
            errors.append(
                f"supervisor: task {task_id} references a missing attempt snapshot."
            )
            continue

        snapshot_matches = (
            snapshot.task_id == task.task_id
            and snapshot.task_revision == task.task_revision
            and snapshot.strategy_stage == task.strategy_stage
            and snapshot.selected_version == task.selected_version
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
                "selected_version": snapshot.selected_version,
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
        if task.status in _TERMINAL_STATUSES:
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
            or
            plan.strategy_stage != task.strategy_stage
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
        if task.status in _TERMINAL_STATUSES:
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
        if (
            snapshot.task_revision != task.task_revision
            or snapshot.strategy_stage != task.strategy_stage
            or snapshot.selected_version != task.selected_version
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
            diagnostics = diagnostics.model_copy(
                update={"selected_version": task.selected_version}
            )
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
            errors.append(
                f"supervisor: selected version for {task_id} was already attempted."
            )
    return events, errors


def reconcile_phase5_state_before_teardown(
    state: OrchestratorState,
) -> Dict[str, Any]:
    """Apply the final supervisor state barrier before teardown.

    Teardown is a cleanup operation, not a routing decision.  It must receive
    a terminal task projection with no retry plans, active targets, or current
    worker inputs.  This function deliberately performs no LLM calls and uses
    the same validator as the supervisor return path, so direct teardown
    callers and graph executions share the same invariant.
    """
    task_queue: Dict[str, RemediationTask] = {
        task_id: task.model_copy()
        for task_id, task in dict(state.get("task_queue", {})).items()
    }
    snapshots_by_id: Dict[str, TaskAttemptSnapshot] = dict(
        state.get("attempt_snapshots_by_id", {})
    )
    retry_plans_by_task: Dict[str, SupervisorRetryPlan] = dict(
        state.get("retry_plans_by_task", {})
    )
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    prior_events = list(state.get("consistency_events", []) or [])
    prior_event_keys = {
        (event.task_id, event.received_attempt_id, event.error_code)
        for event in prior_events
    }

    events, errors = _validate_committed_state(
        task_queue,
        snapshots_by_id,
        retry_plans_by_task,
        retry_diagnostics_by_task,
        [],
        "teardown",
    )
    new_events = [
        event
        for event in _dedupe_consistency_events(events)
        if (event.task_id, event.received_attempt_id, event.error_code)
        not in prior_event_keys
    ]
    prior_errors = set(state.get("errors", []) or [])
    new_errors = list(dict.fromkeys(error for error in errors if error not in prior_errors))

    return {
        "task_queue": task_queue,
        "retry_plans_by_task": retry_plans_by_task,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "active_target_task_ids": [],
        "active_target_group_ids": [],
        "next_routing_step": "teardown",
        "state_revision": int(state.get("state_revision", 0)) + 1,
        "consistency_events": new_events,
        "errors": new_errors,
    }


def _parse_planner_retry_plans(
    scratchpad: str,
    task_queue: Dict[str, RemediationTask],
    diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    group_by_id: Optional[Dict[str, VulnerabilityGroup]] = None,
) -> Tuple[Dict[str, UpdateRetryDiagnostics], Dict[str, SupervisorRetryPlan]]:
    """Parse planner markers and reconcile them into typed per-task plans.

    The planner scratchpad remains useful audit evidence, but this function is
    the only place where planner output becomes routing state.  In particular,
    ``SELECTED_VERSION: NONE`` clears stale selections instead of leaving the
    previous retry candidate active.
    """
    updated = dict(diagnostics_by_task)
    plans: Dict[str, SupervisorRetryPlan] = {}
    sections: Dict[str, List[str]] = {}
    current_task: Optional[str] = None
    for line in (scratchpad or "").splitlines():
        task_match = re.search(r"(?:TASK|Task)\s*[:#]?\s*(task-[\w-]+)", line)
        if task_match and task_match.group(1) in task_queue:
            current_task = task_match.group(1)
            sections.setdefault(current_task, []).append(line)
            continue
        if current_task:
            sections[current_task].append(line)

    for task_id, task in task_queue.items():
        if task_id not in sections:
            continue
        section = "\n".join(sections.get(task_id, []))
        selected_match = re.search(
            r"(?:SELECTED[_ ]VERSION|Selected Version)\s*[:=]\s*([^\s,;]+)",
            section,
            re.IGNORECASE,
        )
        selected = None
        if not selected_match:
            # An incomplete planner section is not a state transition. Keep
            # the last committed plan rather than clearing it accidentally.
            continue
        candidate = selected_match.group(1).strip().lstrip("vV")
        selected = None if candidate.upper() == "NONE" else candidate

        effective_stage = task.strategy_stage
        stage_match = re.search(
            r"EFFECTIVE[_ ]STAGE\s*[:=]\s*([a-z0-9_-]+)",
            section,
            re.IGNORECASE,
        )
        if stage_match:
            raw_stage = stage_match.group(1).strip().lower().replace("-", "_")
            # Normalize known planner vocabulary.  Use CODE_WORKAROUND as a
            # fail-closed sentinel for unknown values: the semantic validator
            # below rejects retry_update at that stage, so an invalid token can
            # never silently inherit the task's current routing stage.
            effective_stage = _PLANNER_STAGE_ALIASES.get(
                raw_stage,
                SCARemediationStage.CODE_WORKAROUND,
            )
        action_match = re.search(
            r"ACTION\s*[:=]\s*(retry_update|pivot_workaround)",
            section,
            re.IGNORECASE,
        )
        action_hint = action_match.group(1).lower() if action_match else None

        prior = updated.get(task_id)
        if prior is None:
            group = (group_by_id or {}).get(task.parent_group_id)
            prior = UpdateRetryDiagnostics(
                task_id=task_id,
                strategy_stage=task.strategy_stage,
                security_floor=(group.fix_plan.fixed_version if group and group.fix_plan else None),
            )

        attempted = list(prior.attempted_versions)
        candidates = list(prior.candidate_versions_considered)
        if selected and selected not in candidates:
            candidates.insert(0, selected)

        # Keep the latest registry fact when the planner states it in its
        # scratchpad.  This is audit evidence as well as a safe deterministic
        # fallback if the LLM selects a version that has already been tried.
        latest_seen = prior.latest_version_seen
        latest_matches = re.findall(
            r"(?:latest\s+(?:stable\s+)?version|latest\s+stable)\s*(?:is|=|:|\(|-)\s*v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)",
            section,
            re.IGNORECASE,
        )
        if latest_matches:
            latest_seen = latest_matches[-1]
            if latest_seen not in candidates:
                candidates.append(latest_seen)

        lowered = section.lower()
        pivot_recommended = bool(
            re.search(r"pivot[^\n]*(?:workaround|child)|workaround child", lowered)
        )
        same_major_equals_latest = bool(
            re.search(
                r"same[- ]major.*(?:equal|same as).*latest\s+stable|"
                r"latest\s+stable.*same[- ]major.*(?:equal|same as)",
                lowered,
            )
        )
        latest_candidate_already_attempted = bool(
            re.search(r"(?:same[- ]major|latest\s+stable).*already\s+attempted", lowered)
            or re.search(r"already\s+attempted.*(?:same[- ]major|latest\s+stable)", lowered)
        )
        exhausted = bool(prior.exhausted_update_path)
        if selected is None and (
            action_hint == "pivot_workaround"
            or
            pivot_recommended
            or "no new version" in lowered
            or "no valid candidate" in lowered
            or "only candidate" in lowered and "already been attempted" in lowered
            or (
                effective_stage == SCARemediationStage.NPM_LATEST
                and same_major_equals_latest
                and latest_candidate_already_attempted
            )
        ):
            exhausted = True

        if (
            not stage_match
            and
            effective_stage == SCARemediationStage.NPM_SAME_MAJOR
            and "same-major" in lowered
            and (
                "same-major stage: skipped" in lowered
                or (
                    "latest" in lowered
                    and ("equal" in lowered or "same" in lowered or "skip" in lowered)
                )
            )
        ):
            effective_stage = SCARemediationStage.NPM_LATEST

        # A planner's free-form pivot language cannot bypass the ordered
        # version stages. Only an exhausted NPM_LATEST task may pivot; earlier
        # stages must remain retryable update tasks.
        if effective_stage != SCARemediationStage.NPM_LATEST:
            exhausted = bool(prior.package_abandoned)

        action = (
            "pivot_workaround"
            if (
                (action_hint == "pivot_workaround" or exhausted)
                and effective_stage == SCARemediationStage.NPM_LATEST
            )
            else "retry_update"
        )
        diagnostics = prior.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": selected,
                "candidate_versions_considered": candidates,
                "latest_version_seen": latest_seen,
                "registry_query_performed": True,
                "exhausted_update_path": exhausted,
            }
        )
        updated[task_id] = diagnostics
        group = (group_by_id or {}).get(task.parent_group_id)
        if action == "retry_update":
            instruction = _build_high_level_retry_instruction(
                task.model_copy(update={"strategy_stage": effective_stage}),
                group,
                None,
                diagnostics,
            )
        else:
            component = group.vulnerable_component if group else task.parent_group_id
            instruction = (
                f"Implement a code workaround or isolation strategy for {component} "
                "because the manifest-based update path is exhausted."
            )
        plans[task_id] = SupervisorRetryPlan(
            task_id=task_id,
            source_task_revision=task.task_revision,
            strategy_stage=effective_stage,
            selected_version=selected,
            attempted_versions=attempted,
            candidate_versions_considered=candidates,
            latest_version_seen=latest_seen,
            exhausted_update_path=exhausted,
            package_abandoned=prior.package_abandoned,
            action=action,
            exact_instruction=instruction,
        )
    return updated, plans


def _planner_plan_violations(
    plans: Dict[str, SupervisorRetryPlan],
    task_queue: Dict[str, RemediationTask],
    diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
) -> List[str]:
    """Validate planner semantics before a plan can mutate routing state.

    The planner's prose is intentionally not treated as an authority.  These
    checks enforce the small set of invariants that must hold for an exact
    worker instruction to be safe.  Returning human-readable violations also
    makes the corrective replan visible in LangSmith through the supervisor's
    accumulated ``errors`` field.
    """
    violations: List[str] = []
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        if task is None:
            violations.append(f"task {task_id}: planner returned an unknown task")
            continue
        if task.status in _TERMINAL_STATUSES:
            violations.append(f"task {task_id}: planner returned a plan for terminal task")
        if plan.source_task_revision != task.task_revision:
            violations.append(
                f"task {task_id}: planner snapshot revision {plan.source_task_revision} "
                f"does not match current revision {task.task_revision}"
            )

        attempted = {
            version.strip().lstrip("vV").lower()
            for version in plan.attempted_versions
            if version
        }
        diagnostics = diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            )

        selected = (
            plan.selected_version.strip().lstrip("vV").lower()
            if plan.selected_version
            else None
        )
        if selected and selected in attempted:
            violations.append(
                f"task {task_id}: selected version {plan.selected_version} was already attempted"
            )
        if (
            plan.strategy_stage == SCARemediationStage.NPM_LATEST
            and selected
            and plan.latest_version_seen
            and selected != plan.latest_version_seen.strip().lstrip("vV").lower()
        ):
            violations.append(
                f"task {task_id}: npm_latest selected {plan.selected_version}, "
                f"but registry latest is {plan.latest_version_seen}"
            )
        if plan.action == "retry_update" and selected is None:
            violations.append(
                f"task {task_id}: retry_update requires an unattempted exact selected_version"
            )
        if plan.action == "retry_update" and plan.exhausted_update_path:
            violations.append(
                f"task {task_id}: exhausted update path cannot retry update"
            )
        if (
            plan.action == "retry_update"
            and plan.strategy_stage == SCARemediationStage.CODE_WORKAROUND
        ):
            violations.append(
                f"task {task_id}: retry_update cannot use code_workaround stage"
            )
        if (
            plan.action == "retry_update"
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage]
            < _SCA_STAGE_ORDER[task.strategy_stage]
        ):
            violations.append(
                f"task {task_id}: planner stage {plan.strategy_stage.value} regresses "
                f"from committed stage {task.strategy_stage.value}"
            )
        if plan.action == "pivot_workaround" and plan.strategy_stage != SCARemediationStage.NPM_LATEST:
            violations.append(
                f"task {task_id}: workaround pivot must be committed at npm_latest"
            )
        if plan.action == "pivot_workaround" and selected is not None:
            violations.append(
                f"task {task_id}: workaround pivot cannot retain selected version {plan.selected_version}"
            )
    return violations


def _reconcile_registry_plan_evidence(
    plans: Dict[str, SupervisorRetryPlan],
    diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    tool_events: List[ToolEvent],
) -> Tuple[Dict[str, UpdateRetryDiagnostics], Dict[str, SupervisorRetryPlan]]:
    """Replace free-form version claims with the planner tool's result.

    The LLM still chooses which task/stage to reason about, but the version
    itself must come from ``plan_npm_version``.  This keeps a plausible prose
    hallucination from surviving merely because it matches the prompt.
    """
    if not tool_events:
        return diagnostics_by_task, plans

    updated = dict(diagnostics_by_task)
    reconciled: Dict[str, SupervisorRetryPlan] = {}
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        group = group_by_id.get(task.parent_group_id) if task else None
        package_name = group.vulnerable_component if group else None
        if not package_name:
            reconciled[task_id] = plan
            continue

        matching_events = [
            event
            for event in tool_events
            if event.name == "plan_npm_version"
            and str(event.args.get("package_name", "")).strip() == package_name
            and event.content.startswith("# NPM Version Plan:")
        ]
        if not matching_events:
            reconciled[task_id] = plan
            continue

        same_major_events = [
            event
            for event in matching_events
            if str(event.args.get("selection", "")).strip().lower() == "same_major"
        ]
        latest_events = [
            event
            for event in matching_events
            if str(event.args.get("selection", "")).strip().lower() == "latest"
        ]
        selected_event = (
            latest_events[-1]
            if plan.strategy_stage == SCARemediationStage.NPM_LATEST and latest_events
            else same_major_events[-1]
            if same_major_events
            else matching_events[-1]
        )
        if (
            selected_event in same_major_events
            and "same-major stage: skipped" in selected_event.content.lower()
            and latest_events
        ):
            selected_event = latest_events[-1]

        content = selected_event.content
        selected_match = re.search(
            r"^-\s*Selected Version:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE
        )
        latest_match = re.search(
            r"^-\s*Latest Stable:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE
        )
        eligible_match = re.search(
            r"^-\s*Eligible Candidates:\s*(.*)$", content, re.IGNORECASE | re.MULTILINE
        )
        selected_token = selected_match.group(1).strip() if selected_match else "NONE"
        selected = None if selected_token.upper() == "NONE" else selected_token.lstrip("vV")
        latest_seen = latest_match.group(1).strip().lstrip("vV") if latest_match else plan.latest_version_seen
        eligible = []
        if eligible_match:
            eligible = [
                version.strip().lstrip("vV")
                for version in eligible_match.group(1).split(",")
                if version.strip() and version.strip().upper() != "NONE"
            ]

        prior = updated.get(task_id)
        if prior is None:
            prior = UpdateRetryDiagnostics(task_id=task_id)
        attempted = {
            version.strip().lstrip("vV")
            for version in prior.attempted_versions
            if version
        }
        candidates = eligible or list(plan.candidate_versions_considered)
        if selected and selected not in candidates:
            candidates.insert(0, selected)
        if latest_seen and latest_seen not in candidates:
            candidates.append(latest_seen)

        effective_stage = plan.strategy_stage
        if selected_event in latest_events:
            effective_stage = SCARemediationStage.NPM_LATEST
        elif (
            "same-major stage: skipped" in content.lower()
            or (
                re.search(r"^-\s*Same-Major Latest:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE)
                and latest_seen
                and re.search(r"^-\s*Same-Major Latest:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE).group(1).lstrip("vV") == latest_seen
            )
        ):
            effective_stage = SCARemediationStage.NPM_LATEST

        unattempted = [version for version in candidates if version not in attempted]
        exhausted = (
            effective_stage == SCARemediationStage.NPM_LATEST
            and selected is None
            and not unattempted
        )
        action = "pivot_workaround" if exhausted else "retry_update"
        diagnostics = prior.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": selected,
                "candidate_versions_considered": candidates,
                "latest_version_seen": latest_seen,
                "registry_query_performed": True,
                "exhausted_update_path": exhausted,
            }
        )
        updated[task_id] = diagnostics
        if action == "retry_update":
            instruction = _build_high_level_retry_instruction(
                task.model_copy(update={"strategy_stage": effective_stage}),
                group,
                None,
                diagnostics,
            )
        else:
            component = group.vulnerable_component if group else task.parent_group_id
            instruction = (
                f"Implement a code workaround or isolation strategy for {component} "
                "because the manifest-based update path is exhausted."
            )
        reconciled[task_id] = plan.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": selected,
                "candidate_versions_considered": candidates,
                "latest_version_seen": latest_seen,
                "exhausted_update_path": exhausted,
                "action": action,
                "exact_instruction": instruction,
            }
        )
    return updated, reconciled


def _repair_invalid_planner_plans(
    plans: Dict[str, SupervisorRetryPlan],
    diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    violations: Optional[List[str]] = None,
) -> Tuple[Dict[str, UpdateRetryDiagnostics], Dict[str, SupervisorRetryPlan]]:
    """Apply a deterministic, fail-closed repair after corrective replanning.

    A valid unattempted candidate already present in planner evidence is safe to
    commit.  If no such candidate exists at the latest stage, the only safe
    action is the existing workaround pivot.  In particular, this function
    never preserves an invalid selected version merely to keep the graph
    moving.
    """
    repaired_diagnostics = dict(diagnostics_by_task)
    repaired_plans: Dict[str, SupervisorRetryPlan] = {}
    invalid_task_ids = {
        match.group(1)
        for violation in (violations or [])
        if (match := re.match(r"task\s+(task-[\w-]+):", violation))
    }
    for task_id, plan in plans.items():
        if invalid_task_ids and task_id not in invalid_task_ids:
            repaired_plans[task_id] = plan
            continue
        diagnostics = repaired_diagnostics.get(task_id)
        attempted = {
            version.strip().lstrip("vV").lower()
            for version in plan.attempted_versions
            if version
        }
        if diagnostics is not None:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            )

        candidate = None
        plan_regresses = (
            task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage]
            < _SCA_STAGE_ORDER[task_queue[task_id].strategy_stage]
        )
        # A correction cannot reopen an earlier stage.  In particular, a
        # stale same-major proposal must not revive a version-bump parent that
        # has already reached code_workaround.  Let the fail-closed branch
        # below create the deterministic latest-stage pivot instead.
        if not plan_regresses and plan.strategy_stage != SCARemediationStage.CODE_WORKAROUND:
            for version in [plan.latest_version_seen, *plan.candidate_versions_considered]:
                if version and version.strip().lstrip("vV").lower() not in attempted:
                    candidate = version.strip().lstrip("vV")
                    break

        if candidate:
            effective_stage = plan.strategy_stage
            if (
                effective_stage == SCARemediationStage.NPM_SAME_MAJOR
                and plan.latest_version_seen
                and candidate == plan.latest_version_seen.strip().lstrip("vV")
            ):
                effective_stage = SCARemediationStage.NPM_LATEST
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": list(
                        dict.fromkeys([*diagnostics.candidate_versions_considered, candidate])
                    ),
                    "registry_query_performed": True,
                    "exhausted_update_path": False,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
            group = group_by_id.get(task_queue[task_id].parent_group_id)
            instruction = _build_high_level_retry_instruction(
                task_queue[task_id].model_copy(update={"strategy_stage": effective_stage}),
                group,
                None,
                diagnostics,
            )
            repaired_plans[task_id] = plan.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": diagnostics.candidate_versions_considered,
                    "action": "retry_update",
                    "exact_instruction": instruction,
                    "exhausted_update_path": False,
                }
            )
            continue

        # No unattempted candidate can be proven.  Clear stale selection and
        # pivot at the terminal update stage so no guessed/old version is sent
        # to the dumb update worker.
        effective_stage = SCARemediationStage.NPM_LATEST
        if diagnostics is None:
            diagnostics = UpdateRetryDiagnostics(task_id=task_id)
        diagnostics = diagnostics.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
            }
        )
        repaired_diagnostics[task_id] = diagnostics
        group = group_by_id.get(task_queue[task_id].parent_group_id)
        component = group.vulnerable_component if group else task_queue[task_id].parent_group_id
        instruction = (
            f"Implement a code workaround or isolation strategy for {component} "
            "because the manifest-based update path is exhausted."
        )
        repaired_plans[task_id] = plan.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "action": "pivot_workaround",
                "exact_instruction": instruction,
            }
        )
    return repaired_diagnostics, repaired_plans


def _parse_planner_selected_versions(
    scratchpad: str,
    task_queue: Dict[str, RemediationTask],
    diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
) -> Dict[str, UpdateRetryDiagnostics]:
    """Backward-compatible wrapper returning reconciled diagnostics only."""
    diagnostics, _ = _parse_planner_retry_plans(
        scratchpad,
        task_queue,
        diagnostics_by_task,
    )
    return diagnostics


def _update_worker_task_ids(
    task_queue: Dict[str, RemediationTask],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    preferred_ids: Optional[List[str]] = None,
    limit: Optional[int] = UPDATE_BATCH_SIZE,
) -> List[str]:
    task_ids = _dispatchable_task_ids_for_status(
        task_queue,
        set(_WORKABLE_STATUSES),
        preferred_ids=preferred_ids,
        strategy=RoutingStrategy.VERSION_BUMP,
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


def _normalize_target_task_ids_for_node(
    next_node: str,
    target_task_ids: List[str],
    task_queue: Dict[str, RemediationTask],
    retry_diagnostics_by_task: Optional[Dict[str, UpdateRetryDiagnostics]] = None,
) -> List[str]:
    """Clamp returned active targets to the lifecycle state accepted by next_node."""
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    if next_node == "qa_critic":
        return _qa_ready_task_ids(task_queue, preferred_ids=target_task_ids)
    if next_node == "update_subagent":
        return _update_worker_task_ids(
            task_queue,
            retry_diagnostics_by_task,
            preferred_ids=target_task_ids,
            limit=UPDATE_BATCH_SIZE,
        )
    if next_node == "workaround_subagent":
        return _dispatchable_task_ids_for_status(
            task_queue,
            set(_WORKABLE_STATUSES),
            preferred_ids=target_task_ids,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            limit=1,
        )
    return []


def _resolve_task_id_from_identifier(
    identifier: str,
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Optional[str]:
    """
    Resolve a task or legacy group identifier to the most relevant task_id.

    Preference order:
    1. Exact task_id match.
    2. Active target task(s) whose parent_group_id matches the identifier.
    3. Non-terminal task(s) whose parent_group_id matches the identifier.
    4. Any task(s) whose parent_group_id matches the identifier.
    """
    if identifier in task_queue:
        return identifier

    active_matches = [
        task_id
        for task_id in active_target_task_ids
        if task_id in task_queue and task_queue[task_id].parent_group_id == identifier
    ]
    if len(active_matches) == 1:
        return active_matches[0]

    non_terminal_matches = [
        task.task_id
        for task in task_queue.values()
        if task.parent_group_id == identifier and task.status not in _TERMINAL_STATUSES
    ]
    if len(non_terminal_matches) == 1:
        return non_terminal_matches[0]

    all_matches = [
        task.task_id
        for task in task_queue.values()
        if task.parent_group_id == identifier
    ]
    if len(all_matches) == 1:
        return all_matches[0]

    return None


def _normalize_qa_evaluations_for_tasks(
    qa_evaluations: Dict[str, QAEvaluation],
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Dict[str, QAEvaluation]:
    """Re-key QA evaluations to concrete task_ids when possible."""
    normalized: Dict[str, QAEvaluation] = {}
    for identifier, evaluation in qa_evaluations.items():
        resolved_t_id = _resolve_task_id_from_identifier(
            identifier,
            task_queue,
            active_target_task_ids,
        )
        if resolved_t_id is None:
            resolved_t_id = _resolve_task_id_from_identifier(
                evaluation.task_id,
                task_queue,
                active_target_task_ids,
            )
        if resolved_t_id is None:
            continue
        normalized[resolved_t_id] = QAEvaluation(
            task_id=resolved_t_id,
            passed=evaluation.passed,
            failure_category=evaluation.failure_category,
            retry_feedback=evaluation.retry_feedback,
        )
    return normalized


def _constraint_entry_for_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    """Build a deterministic constraints-ledger entry for a QA-passed task."""
    component = (group.vulnerable_component or task.parent_group_id).strip()
    fix_plan = group.fix_plan

    if task.strategy == RoutingStrategy.VERSION_BUMP:
        fixed_version = (fix_plan.fixed_version if fix_plan else None) or "unknown"
        return f"{component}: keep resolved version at {fixed_version}"

    return f"{component}: preserve validated security workaround"


def _missing_retry_revised_instructions(
    next_node: str,
    target_task_ids: List[str],
    revised_instructions: Dict[str, str],
    task_queue: Dict[str, RemediationTask],
) -> List[str]:
    """Return retry-bound update targets that are missing exact revised instructions."""
    if next_node != "update_subagent":
        return []

    missing: List[str] = []
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


def _latest_action_summary_by_task(
    action_summaries: List[AgentActionSummary],
    task_queue: Dict[str, RemediationTask],
    active_target_task_ids: List[str],
) -> Dict[str, AgentActionSummary]:
    """Return the most recent action summary keyed by resolved task_id."""
    latest: Dict[str, AgentActionSummary] = {}
    for summary in action_summaries:
        resolved_task_id = _resolve_task_id_from_identifier(
            summary.task_id,
            task_queue,
            active_target_task_ids,
        )
        if resolved_task_id is None:
            continue
        latest[resolved_task_id] = summary
    return latest


def _build_high_level_retry_instruction(
    task: RemediationTask,
    group: Optional[VulnerabilityGroup],
    evaluation: Optional[QAEvaluation],
    diagnostics: Optional[UpdateRetryDiagnostics],
) -> str:
    """Synthesize a high-level retry instruction for the update worker."""
    component = group.vulnerable_component if group else task.parent_group_id
    category = evaluation.failure_category if evaluation else None
    if diagnostics and diagnostics.selected_version:
        manifest = (group.file_paths[0] if group and group.file_paths else "package.json")
        dependency_action = "dependency version"
        if diagnostics.used_overrides:
            dependency_action = "npm override"
        return (
            f"Apply the supervisor-selected {dependency_action} for {component} "
            f"during strategy stage {task.strategy_stage.value}: "
            f"update {manifest} to exact version {diagnostics.selected_version}; "
            "after all requested manifest edits, run the single final manifest synchronization validation."
        )
    if task.strategy_stage == SCARemediationStage.OSV_MINIMUM and group and group.fix_plan:
        floor = group.fix_plan.fixed_version
        if floor:
            manifest = group.file_paths[0] if group.file_paths else "package.json"
            return (
                f"Apply strategy stage {task.strategy_stage.value} for {component}: "
                f"update {manifest} to exact OSV minimum fixed version {floor}; "
                "after all requested manifest edits, run the single final manifest synchronization validation."
            )
    if diagnostics and task.strategy in {RoutingStrategy.VERSION_BUMP}:
        attempted = set(diagnostics.attempted_versions)
        candidates = [
            version
            for version in diagnostics.candidate_versions_considered
            if version not in attempted
        ]
        candidate = next(
            (
                version
                for version in [diagnostics.latest_version_seen, *candidates]
                if version and version not in attempted
            ),
            None,
        )
        if candidate:
            manifest = (group.file_paths[0] if group and group.file_paths else "package.json")
            return (
                f"Apply strategy stage {task.strategy_stage.value} for {component}: "
                f"update {manifest} to exact version {candidate}; "
                "after all requested manifest edits, run the single final manifest synchronization validation."
            )
    if diagnostics and diagnostics.package_abandoned:
        return (
            f"Investigate whether {component} still has a supported manifest-based update path. "
            "Use registry evidence to confirm whether the package is unpublished, abandoned, or "
            "otherwise exhausted, and report that clearly if no valid update path remains."
        )
    if category == FailureCategory.PEER_CONFLICT:
        return (
            f"Investigate compatible patched releases or override paths for {component}. "
            "Prioritize peer-compatible backports or npm overrides that preserve validation."
        )
    if category == FailureCategory.SECURITY_FLAG:
        return (
            f"Investigate patched manifest remediation paths for {component}. "
            "Use registry evidence to compare newer releases, backported patches, and override strategies."
        )
    if category == FailureCategory.BREAKING_CHANGE:
        return (
            f"Document the validated version outcome for {component} and the specific API or test regressions. "
            "Do not search for another version unless new evidence shows a safer compatible patch exists."
        )
    return (
        f"Investigate remaining safe manifest remediation paths for {component}. "
        "Use registry evidence and prior validation failures to choose the next bounded retry."
    )


def _worker_node_for_strategy(strategy: RoutingStrategy) -> str:
    """Return the worker node that handles a given routing strategy."""
    if strategy == RoutingStrategy.VERSION_BUMP:
        return "update_subagent"
    return "workaround_subagent"


def _parent_status_for_strategy_pivot(
    parent_task: RemediationTask,
    new_strategy: RoutingStrategy,
    qa_evaluations: Dict[str, QAEvaluation],
) -> TaskStatus:
    """
    Choose the terminal parent status when a strategy pivot spawns a child task.

    BREAKING_CHANGE means the version bump itself succeeded but caused regressions,
    so the parent attempt should become QA_PASSED and the child handles follow-on
    workaround work. Other pivots represent an exhausted parent strategy attempt.
    """
    evaluation = qa_evaluations.get(parent_task.task_id)
    if (
        parent_task.strategy == RoutingStrategy.VERSION_BUMP
        and new_strategy == RoutingStrategy.CODE_WORKAROUND
        and evaluation is not None
        and evaluation.failure_category == FailureCategory.BREAKING_CHANGE
    ):
        return TaskStatus.QA_PASSED
    return TaskStatus.UNFIXABLE


def _terminalize_pivot_parents(
    task_queue: Dict[str, RemediationTask],
    parent_ids: List[str],
    strategy_by_parent: Dict[str, RoutingStrategy],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Optional[Dict[str, UpdateRetryDiagnostics]] = None,
    retry_plans_by_task: Optional[Dict[str, SupervisorRetryPlan]] = None,
    group_by_id: Optional[Dict[str, VulnerabilityGroup]] = None,
) -> None:
    """Mark pivoted parent tasks terminal so they cannot be re-routed to update work."""
    for parent_id in parent_ids:
        parent_task = task_queue.get(parent_id)
        new_strategy = strategy_by_parent.get(parent_id)
        if parent_task is None or new_strategy is None:
            continue
        terminal_status = _parent_status_for_strategy_pivot(
            parent_task,
            new_strategy,
            qa_evaluations,
        )
        updates: Dict[str, Any] = {}
        if parent_task.status not in _TERMINAL_STATUSES:
            updates["status"] = terminal_status

        # Parent/child pivots are one atomic state transition.  A version-bump
        # parent remains the audit record for the exhausted update path, while
        # the newly materialized child owns workaround execution.  Do not
        # leave the parent with a code-workaround stage and an old exact
        # dependency instruction; that contradictory combination is what made
        # the router prompt in the trace authorize a stale update.
        if (
            parent_task.strategy == RoutingStrategy.VERSION_BUMP
            and new_strategy == RoutingStrategy.CODE_WORKAROUND
        ):
            group = (group_by_id or {}).get(parent_task.parent_group_id)
            component = group.vulnerable_component if group else parent_task.parent_group_id
            updates.update(
                {
                    "strategy_stage": SCARemediationStage.NPM_LATEST,
                    "selected_version": None,
                    "exhausted_update_path": True,
                    "instruction": (
                        f"The manifest-based update path for {component} is exhausted; "
                        "the workaround child task owns the remaining remediation."
                    ),
                }
            )

        # A pivot closes the update attempt. Keep the immutable attempt in
        # history, but do not leave it as the task's current worker input
        # after replacing the task with a terminal parent/child transition.
        # Otherwise the next supervisor pass sees an old update snapshot paired
        # with workaround state.
        _commit_task_transition(
            task_queue,
            parent_id,
            updates=updates,
            close_attempt=(parent_task.current_attempt_id is not None),
            clear_selected_version=(parent_task.selected_version is not None),
        )
        if retry_plans_by_task is not None:
            retry_plans_by_task.pop(parent_id, None)
        if retry_diagnostics_by_task is not None:
            diagnostics = retry_diagnostics_by_task.get(parent_id)
            if diagnostics is not None:
                committed_parent = task_queue[parent_id]
                retry_diagnostics_by_task[parent_id] = diagnostics.model_copy(
                    update={
                        "strategy_stage": committed_parent.strategy_stage,
                        "selected_version": None,
                        "exhausted_update_path": True,
                        "committed_attempt_id": None,
                        "instruction_digest": _instruction_digest(
                            committed_parent.instruction
                        ),
                    }
                )


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------


def _deterministic_routing(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: Optional[List[AgentActionSummary]] = None,
    active_target_task_ids: Optional[List[str]] = None,
    current_status: str = "",
) -> SupervisorDecision:
    """
    Pure-Python fallback routing used when the LLM call fails.

    Implements the same priority rules described in the supervisor prompt.
    """
    tasks = list(task_queue.values())
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
    latest_summaries = _latest_action_summary_by_task(
        action_summaries or [],
        task_queue,
        list(active_target_task_ids or []),
    )

    # All tasks are terminal → teardown
    if not non_terminal:
        return SupervisorDecision(
            next_node="teardown",
            target_task_ids=[],
            instructions="All tasks are terminal. Proceeding to teardown.",
            decision_reason="No actionable tasks remain.",
        )

    # If active batch is all optimistically_fixed → route to qa_critic
    current_batch_qa_ready = _qa_ready_task_ids(
        task_queue,
        preferred_ids=list(active_target_task_ids or []),
    )
    if current_status != "qa_completed" and current_batch_qa_ready:
        return SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=current_batch_qa_ready,
            instructions="Run QA on the current remediated batch before starting more remediation.",
            decision_reason=(
                f"Routing {len(current_batch_qa_ready)} optimistically fixed task(s) from the current batch to QA."
            ),
        )

    all_qa_ready = _qa_ready_task_ids(task_queue)
    if all_qa_ready:
        return SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=all_qa_ready,
            instructions="Run QA on the remaining optimistically fixed tasks.",
            decision_reason="Routing all remaining optimistically fixed tasks to QA.",
        )

    # Collect tasks that still need work
    workable = [t for t in non_terminal if t.status in _WORKABLE_STATUSES]

    # All VERSION_BUMP QA failures follow the ordered version stages. A
    # BREAKING_CHANGE is evidence for the next stage, not an immediate pivot.

    exhausted_retries = [
        task
        for task in workable
        if _is_exhausted_update_pivot_candidate(
            task,
            retry_diagnostics_by_task.get(task.task_id),
        )
    ]
    if exhausted_retries:
        spawn_requests: List[TaskSpawnRequest] = []
        for task in exhausted_retries:
            component = (
                group_by_id.get(task.parent_group_id).vulnerable_component
                if task.parent_group_id in group_by_id
                else task.parent_group_id
            )
            spawn_requests.append(
                TaskSpawnRequest(
                    parent_task_id=task.task_id,
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction=(
                        f"Implement a code workaround or isolation strategy for {component} because "
                        "manifest-based update remediation appears exhausted after bounded registry-guided retries."
                    ),
                    reason=(
                        "Deterministic fallback: exhausted manifest remediation must pivot "
                        "to a workaround child task."
                    ),
                )
            )
        return SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=[exhausted_retries[0].task_id],
            spawn_requests=spawn_requests,
            instructions="Pivot exhausted update remediation to workaround child tasks.",
            decision_reason=(
                f"Retry diagnostics show {len(exhausted_retries)} update task(s) no longer have a remaining manifest-based update path."
            ),
        )

    retry_version_bump = [
        t
        for t in workable
        if t.strategy == RoutingStrategy.VERSION_BUMP and t.status == TaskStatus.NEEDS_RETRY
    ]
    if retry_version_bump:
        batch = retry_version_bump[:UPDATE_BATCH_SIZE]
        feedback_by_task: Dict[str, str] = {}
        revised_instructions: Dict[str, str] = {}
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
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions="Route retry-bound dependency tasks back to the update worker with high-level retry goals.",
            decision_reason=(
                f"Routing {len(batch)} retry VERSION_BUMP task(s) to update_subagent for registry-guided evidence gathering."
            ),
        )

    # VERSION_BUMP tasks batch to update_subagent only for non-retry work.
    version_bump = [
        t
        for t in workable
        if t.strategy == RoutingStrategy.VERSION_BUMP and t.status != TaskStatus.NEEDS_RETRY
    ]
    if version_bump:
        batch = version_bump[:UPDATE_BATCH_SIZE]
        feedback_by_task: Dict[str, str] = {}
        for t in batch:
            eval_ = qa_evaluations.get(t.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[t.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            instructions="Apply the required version bump(s) in the package manifest(s) for this batch only.",
            decision_reason=(
                f"Routing {len(batch)} VERSION_BUMP task(s) to update_subagent (batch size cap {UPDATE_BATCH_SIZE})."
            ),
        )

    # CODE_WORKAROUND tasks: send exactly one at a time to workaround_subagent
    workaround = [t for t in workable if t.strategy == RoutingStrategy.CODE_WORKAROUND]
    if workaround:
        target = workaround[0]
        eval_ = qa_evaluations.get(target.task_id)
        feedback: Dict[str, str] = {}
        if eval_ and eval_.retry_feedback:
            feedback[target.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback,
            instructions="Apply the minimal safe code workaround for this vulnerability.",
            decision_reason=(
                f"Routing task '{target.task_id}' to workaround_subagent."
            ),
        )

    # Unexpected: no workable tasks found → teardown as safe default
    return SupervisorDecision(
        next_node="teardown",
        target_task_ids=[],
        instructions="No actionable tasks remain.",
        decision_reason=(
            "Deterministic fallback: no workable tasks found, routing to teardown."
        ),
    )


# ---------------------------------------------------------------------------
# Planner phase helpers
# ---------------------------------------------------------------------------


def _needs_planner(
    task_queue: Dict[str, RemediationTask],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    current_status: str,
) -> bool:
    """Return True when the planner should be invoked.

    The planner is reserved for retry analysis and playbook selection.
    """
    if not any(task.status == TaskStatus.NEEDS_RETRY for task in task_queue.values()):
        return False
    return current_status == "qa_completed" or bool(qa_evaluations) or bool(
        retry_diagnostics_by_task
    )


def _build_planner_prompt(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
    correction: str = "",
) -> str:
    """Build the planner system + user messages."""
    lines = [
        "You are the Planner phase of an AppSec remediation Supervisor.",
        "You own exact version planning for retry tasks.",
        "The ordered strategy is OSV minimum, highest stable same-major release, highest stable released version, then code workaround.",
        "The Update Subagent is a dumb worker and must receive an exact version instruction from you.",
        "",
        "Call plan_npm_version for NPM retry stages. Never delegate registry lookup or version choice to the worker.",
        "Respect the OSV security floor and exclude already attempted versions, even when an attempted version is the security floor.",
        "",
        "## Actionable Retry Tasks",
        "Write Strategy Scratchpad sections only for these NEEDS_RETRY tasks.",
    ]

    actionable_retry_tasks = [
        task for task in task_queue.values() if task.status == TaskStatus.NEEDS_RETRY
    ]

    for task in actionable_retry_tasks:
        group = group_by_id.get(task.parent_group_id)
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        component = group.vulnerable_component if group else task.parent_group_id
        eval_ = qa_evaluations.get(task.task_id)
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        security_floor = (
            diagnostics.security_floor if diagnostics and diagnostics.security_floor
            else (group.fix_plan.fixed_version if group and group.fix_plan else "unknown")
        )

        lines += [
            "",
            f"### Task: {task.task_id}",
            f"- Component     : {component}",
            f"- CVEs          : {cves}",
            f"- GHSAs         : {ghsas}",
            f"- Strategy      : {task.strategy.value}",
            f"- Strategy Stage: {task.strategy_stage.value}",
            f"- NPM Selection : {_selection_for_stage(task.strategy_stage) or 'none'}",
            f"- Status        : {task.status.value}",
            f"- Retries Used  : {task.retry_count}/{MAX_RETRIES}",
            f"- Ancestry Depth: {task.ancestry_depth}/{MAX_ANCESTRY_DEPTH}",
            f"- Current Instr : {task.instruction or '(none)'}",
        ]
        if eval_ and task.status == TaskStatus.NEEDS_RETRY:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines += [
                f"- Last QA Failed: category={cat}",
                f"  Feedback: {eval_.retry_feedback}",
            ]
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        if diagnostics is not None:
            attempted = ", ".join(diagnostics.attempted_versions) or "none"
            candidates = ", ".join(diagnostics.candidate_versions_considered[:10]) or "none"
            lines += [
                f"- Retry Diags  : registry={diagnostics.registry_query_performed}, exhausted={diagnostics.exhausted_update_path}, abandoned={diagnostics.package_abandoned}",
                f"  Attempted: {attempted}",
                f"  Candidates: {candidates}",
                f"  Latest Seen: {diagnostics.latest_version_seen or 'unknown'}",
                f"  Failure Reason: {diagnostics.failure_reason or 'none'}",
            ]

    if not actionable_retry_tasks:
        lines.append("- (none)")

    terminal_tasks = [
        task for task in task_queue.values() if task.status in _TERMINAL_STATUSES
    ]
    qa_ready_tasks = [
        task for task in task_queue.values() if task.status == TaskStatus.OPTIMISTICALLY_FIXED
    ]

    lines += [
        "",
        "## QA-Ready Tasks (read-only for planner)",
        "These tasks must go to QA before retry planning. Do not recommend retries, pivots, or spawn_requests for them.",
    ]
    if qa_ready_tasks:
        for task in qa_ready_tasks:
            group = group_by_id.get(task.parent_group_id)
            component = group.vulnerable_component if group else task.parent_group_id
            lines.append(f"- {task.task_id}: {component} ({task.status.value})")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Terminal History (read-only for planner)",
        "These tasks are audit/history only. Do not recommend retries, pivots, spawn_requests, revised instructions, or routing for them.",
    ]
    if terminal_tasks:
        for task in terminal_tasks:
            group = group_by_id.get(task.parent_group_id)
            component = group.vulnerable_component if group else task.parent_group_id
            lines.append(f"- {task.task_id}: {component} ({task.status.value})")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Constraints Ledger (must not violate these)",
    ]
    if constraints_ledger:
        lines.extend(f"- {c}" for c in constraints_ledger)
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Recent Action Summaries",
    ]
    for summary in _current_action_summaries(action_summaries, task_queue, 6):
        lines.append(f"- [{summary.task_id}] {summary.status.value}: {summary.summary}")
    if not action_summaries:
        lines.append("- (none)")

    lines += [
        "",
        "## Planner Playbooks",
        "- SECURITY_FLAG, PEER_CONFLICT, and BREAKING_CHANGE all advance exactly one ordered version stage.",
        "- package_abandoned=True: pivot from VERSION_BUMP to a CODE_WORKAROUND child task.",
        "- exhausted_update_path=True: pivot from VERSION_BUMP to a CODE_WORKAROUND child task.",
        "- VERSION_BUMP + NEEDS_RETRY + exhausted_update_path=True must not be routed back to update_subagent.",
        "- A strategy pivot must be expressed as a child-task recommendation, not as an in-place worker retry.",
        "",
        "## Queue Caps",
        f"- MAX_TASK_QUEUE_SIZE = {MAX_TASK_QUEUE_SIZE} (current: {len(task_queue)})",
        f"- MAX_ANCESTRY_DEPTH = {MAX_ANCESTRY_DEPTH}",
        "",
        "## Output Format",
        "Write a 'Strategy Scratchpad' with these sections for each actionable retry task only:",
        "  1. Observations",
        "  2. Playbook selected",
        "  3. Update-path assessment",
        "  4. Exact selected version and manifest instruction",
        "  5. Strategy pivot recommendation (same-task retry or workaround child)",
        "  6. Routing notes",
        "",
        "Use exact version pins in the planner output. Format each result with TASK: <task-id>, SELECTED_VERSION: <version|NONE>, EFFECTIVE_STAGE: <stage>, and ACTION: <retry_update|pivot_workaround>.",
        "When the same-major latest equals the latest stable version, skip the duplicate same-major stage. Select the latest version only if it is unattempted; if it is already attempted, emit SELECTED_VERSION: NONE, EFFECTIVE_STAGE: npm_latest, and ACTION: pivot_workaround.",
    ]

    if correction:
        lines += [
            "",
            "## Previous Planner Output Rejected",
            "The following deterministic planner invariants were violated:",
            *[f"- {violation}" for violation in correction.splitlines() if violation.strip()],
            "Correct only the affected task sections. Re-call plan_npm_version with the complete attempted-version list.",
            "Never select an attempted version. A retry_update must contain one unattempted exact version. A latest-stage task with no unattempted candidate must use ACTION: pivot_workaround.",
        ]

    return "\n".join(lines)


def _run_planner_phase(
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    qa_evaluations: Dict[str, QAEvaluation],
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics],
    action_summaries: List[AgentActionSummary],
    constraints_ledger: List[str],
    llm: Any,
    correction: str = "",
    return_tool_events: bool = False,
) -> Any:
    """Run the bounded planner ReAct loop. Returns the scratchpad text."""
    planner_prompt = _build_planner_prompt(
        task_queue,
        group_by_id,
        qa_evaluations,
        retry_diagnostics_by_task,
        action_summaries,
        constraints_ledger,
        correction=correction,
    )
    tools = [plan_npm_version]
    initial_messages = [
        SystemMessage(content=planner_prompt),
        HumanMessage(
            content=(
                "Plan every retry task. For NPM stages call plan_npm_version with the package, "
                "OSV security floor, stage selection, and attempted versions. Emit TASK and "
                "SELECTED_VERSION lines before the Strategy Scratchpad."
                + (" Apply the previous-output correction rules exactly." if correction else "")
            )
        ),
    ]
    try:
        result = run_bounded_subagent_loop(
            llm=llm,
            tools=tools,
            initial_messages=initial_messages,
            touched_files=set(),
        )
        scratchpad = result.final_text.strip()
        if scratchpad.startswith("<MagicMock"):
            scratchpad = ""
        if result.errors:
            logger.warning("supervisor planner: %d error(s): %s", len(result.errors), result.errors)
        scratchpad = scratchpad or "(Planner produced no output.)"
        if return_tool_events:
            return scratchpad, list(result.tool_events)
        return scratchpad
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervisor planner: loop failed (%s) — skipping planner.", exc)
        scratchpad = f"(Planner failed: {exc})"
        if return_tool_events:
            return scratchpad, []
        return scratchpad


# ---------------------------------------------------------------------------
# Router prompt builder
# ---------------------------------------------------------------------------


def build_supervisor_prompt(state: OrchestratorState, scratchpad: str = "") -> str:
    """Build the structured Router LLM prompt for the Supervisor decision."""
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
    constraints_ledger: List[str] = state.get("constraints_ledger", [])
    action_summaries: List[AgentActionSummary] = state.get("action_summaries", [])
    qa_evaluations: Dict[str, QAEvaluation] = state.get("qa_evaluations", {})
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics] = state.get(
        "retry_diagnostics_by_task", {}
    )
    retry_plans_by_task: Dict[str, SupervisorRetryPlan] = state.get(
        "retry_plans_by_task", {}
    )
    eval_status: str = state.get("eval_status", "")
    baseline_scan_identifiers: List[str] = state.get("baseline_scan_identifiers", [])
    post_remediation_scan_identifiers: List[str] = state.get(
        "post_remediation_scan_identifiers", []
    )
    new_vulnerability_identifiers: List[str] = state.get(
        "new_vulnerability_identifiers", []
    )
    new_vulnerability_status: str = state.get("new_vulnerability_status", "not_scanned")

    group_by_id = {g.group_id: g for g in valid_groups}

    lines = [
        "You are the Router phase of an AppSec remediation Supervisor.",
        "Produce exactly one SupervisorDecision to route the next graph step.",
        "The Planner owns playbook and exact dependency-version reasoning.",
        "You only translate planner intent and current task state into routing.",
        "Do not invent dependency versions beyond the planner scratchpad and retry diagnostics.",
        "",
    ]

    lines.append("## Remediation Tasks")

    for task in task_queue.values():
        group = group_by_id.get(task.parent_group_id)
        fix_plan = group.fix_plan if group else None
        cves = ", ".join(group.cve_ids) if group and group.cve_ids else "none"
        ghsas = ", ".join(group.ghsa_ids) if group and group.ghsa_ids else "none"
        eval_ = qa_evaluations.get(task.task_id)
        diagnostics = retry_diagnostics_by_task.get(task.task_id)
        security_floor = (
            diagnostics.security_floor
            if diagnostics and diagnostics.security_floor
            else (fix_plan.fixed_version if fix_plan else "unknown")
        )

        lines += [
            "",
            f"### Task: {task.task_id}",
            f"- Parent Group  : {task.parent_group_id}",
            f"- Component     : {group.vulnerable_component if group else 'unknown'}",
            f"- Issue Type    : {group.issue_type.value if group else 'unknown'}",
            f"- CVEs          : {cves}",
            f"- GHSAs         : {ghsas}",
            f"- Fix Plan      : {fix_plan.status.value if fix_plan else 'none'}",
            f"- Strategy      : {task.strategy.value}",
            f"- Strategy Stage: {task.strategy_stage.value}",
            f"- Task Revision : {task.task_revision}",
            f"- Current Attempt: {task.current_attempt_id or 'none'}",
            f"- Committed Selected Version: {task.selected_version or 'none'}",
            f"- OSV Security Floor: {security_floor}",
            f"- Status        : {task.status.value}",
            f"- Retries Used  : {task.retry_count}/{MAX_RETRIES}",
            f"- Ancestry Depth: {task.ancestry_depth}/{MAX_ANCESTRY_DEPTH}",
            f"- Instruction   : {task.instruction or '(none)'}",
        ]
        if task.parent_task_id:
            lines.append(f"- Parent Task   : {task.parent_task_id}")
        if eval_ and task.status != TaskStatus.OPTIMISTICALLY_FIXED:
            cat = eval_.failure_category.value if eval_.failure_category else "none"
            lines.append(
                f"- Last QA       : passed={eval_.passed}, category={cat}, "
                f"feedback={eval_.retry_feedback}"
            )
        if diagnostics is not None:
            lines.append(
                f"- Retry Diags   : registry={diagnostics.registry_query_performed}, "
                f"latest={diagnostics.latest_version_seen or 'unknown'}, "
                f"exhausted={diagnostics.exhausted_update_path}, "
                f"abandoned={diagnostics.package_abandoned}"
            )
            if diagnostics.selected_version:
                lines.append(f"- Supervisor Selected Version: {diagnostics.selected_version}")
        plan = retry_plans_by_task.get(task.task_id)
        if plan is not None:
            lines.append(
                f"- Committed Planner Action: {plan.action}; "
                f"effective_stage={plan.strategy_stage.value}; "
                f"exhausted={plan.exhausted_update_path}"
            )

    lines += [
        "",
        "## Constraints Ledger",
    ]
    if constraints_ledger:
        lines.extend(f"- {c}" for c in constraints_ledger)
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Recent Action Summaries",
    ]
    for summary in _current_action_summaries(action_summaries, task_queue, 10):
        lines.append(
            f"- [{summary.task_id}] {summary.status.value}: {summary.summary}"
        )
    if not action_summaries:
        lines.append("- (none)")

    lines += [
        "",
        f"## QA Evaluation Status: {eval_status or 'none'}",
        "",
        "## Global Post-remediation Scanner Findings",
        f"- New-vulnerability status: {new_vulnerability_status}",
        f"- Baseline identifiers: {', '.join(baseline_scan_identifiers) or '(none)'}",
        f"- Post-remediation identifiers: {', '.join(post_remediation_scan_identifiers) or '(none or unavailable)'}",
        f"- Newly introduced identifiers: {', '.join(new_vulnerability_identifiers) or '(none)'}",
        "- Newly introduced identifiers are report-only until the later triage phase; do not assign them to existing tasks or retry unrelated remediation.",
        "",
        "## Planner Scratchpad",
        scratchpad or "(none)",
        "",
        f"## Queue Caps: {len(task_queue)}/{MAX_TASK_QUEUE_SIZE} tasks used, depth cap = {MAX_ANCESTRY_DEPTH}",
        "",
        "## Router Rules (follow strictly)",
        "0. QA-ready tasks have priority: route optimistically_fixed tasks to qa_critic before planning retries or dispatching workers.",
        f"1. Send pending VERSION_BUMP tasks to update_subagent in batches of at most {UPDATE_BATCH_SIZE}.",
        f"2. Send retry VERSION_BUMP tasks to update_subagent in retry-only batches of at most {UPDATE_BATCH_SIZE}.",
        "3. Every retry task routed to update_subagent MUST have a non-empty revised_instructions entry containing the exact planned version.",
        "4. Retry revised_instructions are authoritative exact execution instructions.",
        "5. Same-strategy retries reuse the same task.",
        "6. Any strategy pivot must be represented with spawn_requests; do not rely on updated_task_strategies for new pivot decisions.",
        "7. SECURITY_FLAG, PEER_CONFLICT, and BREAKING_CHANGE advance the task by exactly one version stage.",
        "SECURITY_FLAG and PEER_CONFLICT remain update remediation first; BREAKING_CHANGE also advances through the ordered update stages.",
        "8. Only an exhausted NPM_LATEST stage may pivot to a CODE_WORKAROUND child task.",
        "9. Send exactly one pending or retry CODE_WORKAROUND task to workaround_subagent.",
        "10. After a worker succeeds for the current active batch, route that batch to qa_critic.",
        "11. When no actionable non-terminal tasks remain, route to teardown.",
        f"12. Any task with {MAX_RETRIES}+ retries may be marked unfixable.",
        "13. unfixable and qa_passed tasks must never appear in target_task_ids; optimistically_fixed tasks may only appear for qa_critic.",
        "14. task_status_updates may only set QA_PASSED or UNFIXABLE.",
        f"15. update_subagent MUST have between 1 and {UPDATE_BATCH_SIZE} target_task_ids.",
        "16. workaround_subagent MUST have exactly one target_task_id.",
        "17. instructions is audit/routing rationale only; do not use it as a substitute for revised_instructions.",
        f"18. spawn_requests must respect parent depth < {MAX_ANCESTRY_DEPTH} and queue size <= {MAX_TASK_QUEUE_SIZE}.",
        "19. When a pivot is chosen, the parent task is terminal and must not be routed back to update_subagent.",
        "20. Mixed worker batches must be split by task status before routing the next node.",
        "21. VERSION_BUMP tasks with exhausted_update_path=True or package_abandoned=True must pivot via spawn_requests, not update_subagent.",
        "22. You may include multiple spawn_requests in one decision, but workaround_subagent target_task_ids must still contain exactly one parent/child target.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spawn request materializer
# ---------------------------------------------------------------------------


def _materialize_spawn_requests(
    spawn_requests: List[TaskSpawnRequest],
    task_queue: Dict[str, RemediationTask],
    group_by_id: Dict[str, VulnerabilityGroup],
    errors: List[str],
) -> Tuple[Dict[str, RemediationTask], Dict[str, List[str]]]:
    """Validate and materialize spawn requests into new RemediationTask objects.

    Returns a dict of new task_id → RemediationTask to be merged into task_queue.
    Rejected requests are logged to errors.
    """
    next_index = len(task_queue) + 1
    new_tasks: Dict[str, RemediationTask] = {}
    child_ids_by_parent: Dict[str, List[str]] = {}
    current_queue_size = len(task_queue)

    for req in spawn_requests:
        # Guard: unknown parent task
        if req.parent_task_id not in task_queue:
            errors.append(
                f"supervisor: spawn rejected — parent task '{req.parent_task_id}' not in queue."
            )
            continue

        parent_task = task_queue[req.parent_task_id]

        # Guard: depth cap
        child_depth = parent_task.ancestry_depth + 1
        if child_depth > MAX_ANCESTRY_DEPTH:
            errors.append(
                f"supervisor: spawn rejected — parent '{req.parent_task_id}' at depth "
                f"{parent_task.ancestry_depth}, child would be depth {child_depth} "
                f"which exceeds MAX_ANCESTRY_DEPTH={MAX_ANCESTRY_DEPTH}."
            )
            continue

        # Guard: queue size cap
        if current_queue_size + len(new_tasks) + 1 > MAX_TASK_QUEUE_SIZE:
            errors.append(
                f"supervisor: spawn rejected — queue would exceed MAX_TASK_QUEUE_SIZE="
                f"{MAX_TASK_QUEUE_SIZE}. Rejected spawn for parent '{req.parent_task_id}'."
            )
            continue

        # Materialize child task
        child_task_id = f"task-{next_index}"
        next_index += 1

        new_task = RemediationTask(
            task_id=child_task_id,
            parent_group_id=parent_task.parent_group_id,
            parent_task_id=req.parent_task_id,
            strategy=req.strategy,
            strategy_stage=(
                SCARemediationStage.CODE_WORKAROUND
                if req.strategy == RoutingStrategy.CODE_WORKAROUND
                else SCARemediationStage.OSV_MINIMUM
            ),
            instruction=req.instruction,
            status=TaskStatus.PENDING,
            retry_count=0,
            ancestry_depth=child_depth,
        )
        new_tasks[child_task_id] = new_task
        child_ids_by_parent.setdefault(req.parent_task_id, []).append(child_task_id)
        logger.info(
            "supervisor: spawned child task '%s' (parent='%s', depth=%d, strategy=%s) — %s",
            child_task_id,
            req.parent_task_id,
            child_depth,
            req.strategy.value,
            req.reason,
        )

    return new_tasks, child_ids_by_parent


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def run_supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Supervisor commander for Phase 5 orchestration.

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    5. Short-circuit: if active batch is all optimistically_fixed → qa_critic.
    6. Router phase: ChatOpenAI.with_structured_output(SupervisorDecision).
    7. Guardrails: reject unknown IDs, enforce cardinality, fall back to
       deterministic routing if invalid.
    8. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
    9. Return state patch.
    """
    valid_groups: List[VulnerabilityGroup] = list(state.get("valid_groups", []))
    if not valid_groups:
        logger.info("supervisor: no valid groups — routing to teardown.")
        return {
            "status": "supervisor_routed",
            "next_routing_step": "teardown",
            "active_target_task_ids": [],
            "supervisor_instructions": "No groups to process.",
        }

    group_by_id: Dict[str, VulnerabilityGroup] = {g.group_id: g for g in valid_groups}
    existing_constraints: List[str] = list(state.get("constraints_ledger", []))
    retry_diagnostics_by_task: Dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    retry_plans_by_task: Dict[str, SupervisorRetryPlan] = dict(
        state.get("retry_plans_by_task", {})
    )
    attempt_snapshots_by_id: Dict[str, TaskAttemptSnapshot] = dict(
        state.get("attempt_snapshots_by_id", {})
    )
    worker_results_by_attempt: Dict[str, WorkerAttemptResult] = dict(
        state.get("worker_results_by_attempt", {})
    )
    qa_results_by_attempt: Dict[str, QAAttemptResult] = dict(
        state.get("qa_results_by_attempt", {})
    )
    processed_worker_attempt_ids: Set[str] = set(
        state.get("processed_worker_attempt_ids", [])
    )
    processed_qa_attempt_ids: Set[str] = set(
        state.get("processed_qa_attempt_ids", [])
    )
    prior_consistency_events: List[StateConsistencyEvent] = list(
        state.get("consistency_events", [])
    )
    consistency_events: List[StateConsistencyEvent] = []
    state_revision = int(state.get("state_revision", 0)) + 1
    # ``errors`` is an additive LangGraph reducer. A node must return only
    # errors discovered during this invocation; replaying the prior list here
    # is what caused identical planner errors to multiply across supervisor
    # loops.
    errors: List[str] = []
    prior_error_messages = set(state.get("errors", []) or [])

    # ------------------------------------------------------------------
    # 1. Normalize task_queue (copy-on-write)
    # ------------------------------------------------------------------
    raw_task_queue: Dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    # Copy-on-write: work with model copies so we never mutate state-owned objects
    task_queue: Dict[str, RemediationTask] = {
        tid: t.model_copy() for tid, t in raw_task_queue.items()
    }
    existing_group_ids = {t.parent_group_id for t in task_queue.values()}
    next_task_index = len(task_queue) + 1
    for group in valid_groups:
        if group.group_id not in existing_group_ids:
            task_id = f"task-{next_task_index}"
            task_queue[task_id] = build_initial_remediation_task(group, task_id)
            next_task_index += 1

    # Keep task-owned planner fields synchronized with the initial OSV plan.
    # Later planner commits are the only source allowed to change these fields.
    for task_id, task in list(task_queue.items()):
        group = group_by_id.get(task.parent_group_id)
        task_updates: Dict[str, Any] = {}
        if (
            task.task_revision == 0
            and
            task.status not in _TERMINAL_STATUSES
            and task.current_attempt_id is None
            and not task.instruction
            and group is not None
            and group.fix_plan is not None
            and group.fix_plan.instruction
        ):
            task_updates["instruction"] = group.fix_plan.instruction
        if (
            task.task_revision == 0
            and task.current_attempt_id is None
            and task.status == TaskStatus.PENDING
            and task.selected_version is None
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and group is not None
            and group.fix_plan is not None
            and group.fix_plan.fixed_version
        ):
            task_updates["selected_version"] = group.fix_plan.fixed_version
        if task_updates:
            task_queue[task_id] = task.model_copy(update=task_updates)

    # ------------------------------------------------------------------
    # 2. Ingest attempt-tagged worker results (active targets only)
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    active_targets = set(active_target_task_ids)
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []
    new_worker_attempt_ids: List[str] = []

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        current_attempt_id = task.current_attempt_id
        snapshot = (
            attempt_snapshots_by_id.get(current_attempt_id)
            if current_attempt_id
            else None
        )
        result = (
            worker_results_by_attempt.get(current_attempt_id)
            if current_attempt_id
            else None
        )
        for stale_result in worker_results_by_attempt.values():
            if (
                stale_result.task_id == task_id
                and stale_result.attempt_id != current_attempt_id
                and stale_result.attempt_id not in processed_worker_attempt_ids
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="STALE_WORKER_RESULT",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=stale_result.attempt_id,
                        action="ignored",
                        details="Worker result belongs to an older task attempt.",
                    )
                )
                processed_worker_attempt_ids.add(stale_result.attempt_id)
                new_worker_attempt_ids.append(stale_result.attempt_id)
        if result is not None and task.status in _TERMINAL_STATUSES:
            if current_attempt_id not in processed_worker_attempt_ids:
                consistency_events.append(
                    _build_consistency_event(
                        error_code="TERMINAL_TASK_RESULT_IGNORED",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="ignored",
                        details="A late worker result cannot reopen a terminal task.",
                    )
                )
                processed_worker_attempt_ids.add(current_attempt_id)
                new_worker_attempt_ids.append(current_attempt_id)
            continue
        if result is not None and current_attempt_id not in processed_worker_attempt_ids:
            if (
                result.task_id != task_id
                or result.task_revision != task.task_revision
                or snapshot is None
                or result.instruction_digest != snapshot.instruction_digest
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="WORKER_ATTEMPT_MISMATCH",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="ignored",
                        details=(
                            f"Expected revision {task.task_revision} and digest "
                            f"{snapshot.instruction_digest if snapshot else 'missing'}, "
                            f"received revision {result.task_revision} and digest "
                            f"{result.instruction_digest}."
                        ),
                    )
                )
                errors.append(
                    f"supervisor: ignored mismatched worker result for {task_id} "
                    f"attempt {result.attempt_id}."
                )
                processed_worker_attempt_ids.add(current_attempt_id)
                new_worker_attempt_ids.append(current_attempt_id)
                continue

            execution = result.execution_diagnostics
            attempted_versions = list(
                dict.fromkeys(execution.attempted_versions or result.executed_versions)
            )
            result_status = result.status
            normalized_executed = {
                version.strip().lstrip("vV").lower()
                for version in result.executed_versions
                if version
            }
            normalized_selected = (
                snapshot.selected_version.strip().lstrip("vV").lower()
                if snapshot.selected_version
                else None
            )
            if (
                result_status == AgentActionStatus.SUCCESS
                and normalized_selected
                and normalized_executed
                and normalized_selected not in normalized_executed
            ):
                result_status = AgentActionStatus.SURRENDER
                consistency_events.append(
                    _build_consistency_event(
                        error_code="EXECUTED_VERSION_MISMATCH",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=result.attempt_id,
                        action="replanned",
                        details=(
                            f"Committed {snapshot.selected_version}; worker executed "
                            f"{', '.join(result.executed_versions)}."
                        ),
                    )
                )
                errors.append(
                    f"supervisor: rejected worker result for {task_id} because the "
                    "executed version differed from the committed version."
                )
            prior = retry_diagnostics_by_task.get(task_id)
            if prior is None:
                prior = UpdateRetryDiagnostics(task_id=task_id)
            retry_diagnostics_by_task[task_id] = prior.model_copy(
                update={
                    "committed_attempt_id": current_attempt_id,
                    "attempted_versions": list(
                        dict.fromkeys(prior.attempted_versions + attempted_versions)
                    ),
                    "executed_versions": list(
                        dict.fromkeys(
                            prior.executed_versions + result.executed_versions
                        )
                    ),
                    "selected_version": task.selected_version,
                    "strategy_stage": task.strategy_stage,
                    "exhausted_update_path": task.exhausted_update_path,
                    "instruction_digest": snapshot.instruction_digest,
                        "failure_reason": (
                            " | ".join(result.errors)
                        if result_status == AgentActionStatus.SURRENDER
                        else prior.failure_reason
                    ),
                    "reasoning_summary": (
                        result.action_summary.summary
                        if result.action_summary is not None
                        else prior.reasoning_summary
                    ),
                }
            )
            if result_status == AgentActionStatus.SUCCESS:
                # Keep a successful attempt open because QA must evaluate the
                # exact snapshot that produced the changes.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
            elif task.strategy == RoutingStrategy.CODE_WORKAROUND:
                # A surrender is a completed worker outcome, not an active
                # worker input. Close it before the next routing decision so a
                # terminal workaround task cannot reach teardown with a live
                # current attempt.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.UNFIXABLE},
                    close_attempt=True,
                    clear_selected_version=True,
                )
            else:
                # Failed update attempts are replanned in this same
                # supervisor pass. Detach the consumed attempt first so the
                # planner cannot observe a new stage paired with an old
                # immutable snapshot.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={
                        "status": TaskStatus.NEEDS_RETRY,
                        "retry_count": task.retry_count + 1,
                    },
                    close_attempt=True,
                )
            processed_worker_attempt_ids.add(current_attempt_id)
            new_worker_attempt_ids.append(current_attempt_id)
            continue

        # Safe migration path for tests/legacy callers: untagged summaries can
        # only be consumed before the task has ever received an attempt.
        if current_attempt_id is None and task.status not in _TERMINAL_STATUSES:
            matching = [
                summary
                for summary in action_summaries
                if summary.task_id == task_id and summary.attempt_id is None
            ]
            if matching:
                summary = matching[-1]
                if summary.status == AgentActionStatus.SUCCESS:
                    task_queue[task_id] = task.model_copy(
                        update={"status": TaskStatus.OPTIMISTICALLY_FIXED}
                    )
                elif task.strategy == RoutingStrategy.CODE_WORKAROUND:
                    task_queue[task_id] = task.model_copy(
                        update={"status": TaskStatus.UNFIXABLE}
                    )
                else:
                    task_queue[task_id] = task.model_copy(
                        update={
                            "status": TaskStatus.NEEDS_RETRY,
                            "retry_count": task.retry_count + 1,
                        }
                    )
        elif action_summaries and current_attempt_id:
            if any(
                summary.task_id == task_id and summary.attempt_id is None
                for summary in action_summaries
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="UNCORRELATED_WORKER_RESULT",
                        task_id=task_id,
                        expected_attempt_id=current_attempt_id,
                        received_attempt_id=None,
                        action="ignored",
                        details="Untagged worker summary cannot mutate an attempted task.",
                    )
                )

    # ------------------------------------------------------------------
    # 3. Ingest QA results (active targets only, when qa_completed)
    # ------------------------------------------------------------------
    qa_evaluations: Dict[str, QAEvaluation] = _normalize_qa_evaluations_for_tasks(
        dict(state.get("qa_evaluations", {})),
        task_queue,
        active_target_task_ids,
    )
    qa_result_task_ids: Set[str] = set()
    new_qa_attempt_ids: List[str] = []
    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            continue
        qa_result = qa_results_by_attempt.get(task.current_attempt_id)
        for stale_result in qa_results_by_attempt.values():
            if (
                stale_result.task_id == task_id
                and stale_result.attempt_id != task.current_attempt_id
                and stale_result.attempt_id not in processed_qa_attempt_ids
            ):
                consistency_events.append(
                    _build_consistency_event(
                        error_code="STALE_QA_RESULT",
                        task_id=task_id,
                        expected_attempt_id=task.current_attempt_id,
                        received_attempt_id=stale_result.attempt_id,
                        action="ignored",
                        details="QA result belongs to an older task attempt.",
                    )
                )
                processed_qa_attempt_ids.add(stale_result.attempt_id)
                new_qa_attempt_ids.append(stale_result.attempt_id)
        if qa_result is None or task.current_attempt_id in processed_qa_attempt_ids:
            continue
        if task.status in _TERMINAL_STATUSES:
            consistency_events.append(
                _build_consistency_event(
                    error_code="TERMINAL_QA_RESULT_IGNORED",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=qa_result.attempt_id,
                    action="ignored",
                    details="A late QA result cannot reopen a terminal task.",
                )
            )
            processed_qa_attempt_ids.add(task.current_attempt_id)
            new_qa_attempt_ids.append(task.current_attempt_id)
            continue
        snapshot = attempt_snapshots_by_id.get(task.current_attempt_id)
        if (
            qa_result.task_id != task_id
            or qa_result.task_revision != task.task_revision
            or snapshot is None
        ):
            consistency_events.append(
                _build_consistency_event(
                    error_code="QA_ATTEMPT_MISMATCH",
                    task_id=task_id,
                    expected_attempt_id=task.current_attempt_id,
                    received_attempt_id=qa_result.attempt_id,
                    action="ignored",
                    details=(
                        f"Expected revision {task.task_revision}; "
                        f"received revision {qa_result.task_revision}."
                    ),
                )
            )
            errors.append(
                f"supervisor: ignored mismatched QA result for {task_id} "
                f"attempt {qa_result.attempt_id}."
            )
        else:
            qa_evaluations[task_id] = qa_result.evaluation
            qa_result_task_ids.add(task_id)
            # QA closes the worker attempt before any status or stage change.
            # The next planner proposal must observe a task with no active
            # worker input; otherwise it can see the new retry stage paired
            # with the old attempt snapshot during the same supervisor pass.
            _commit_task_transition(
                task_queue,
                task_id,
                updates={},
                close_attempt=True,
            )
        processed_qa_attempt_ids.add(task.current_attempt_id)
        new_qa_attempt_ids.append(task.current_attempt_id)
    auto_new_constraints: List[str] = []

    if state.get("status") == "qa_completed":
        for resolved_t_id, evaluation in qa_evaluations.items():
            task_for_result = task_queue.get(resolved_t_id)
            if (
                task_for_result is not None
                and task_for_result.current_attempt_id is not None
                and resolved_t_id not in qa_result_task_ids
            ):
                # A compatibility QA projection without the current attempt
                # identity cannot mutate an attempted task.
                continue
            task = task_queue[resolved_t_id]
            if task.status in (TaskStatus.UNFIXABLE, TaskStatus.QA_PASSED):
                continue
            if evaluation.passed:
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.QA_PASSED},
                )
                group = group_by_id.get(task.parent_group_id)
                if group:
                    constraint = _constraint_entry_for_task(task, group)
                    if (
                        constraint
                        and constraint not in existing_constraints
                        and constraint not in auto_new_constraints
                    ):
                        auto_new_constraints.append(constraint)
            else:
                next_stage = _next_sca_stage(task.strategy_stage)
                task_updates: Dict[str, Any] = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                if task.strategy == RoutingStrategy.VERSION_BUMP:
                    task_updates["strategy_stage"] = next_stage
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates=task_updates,
                )
                task = task_queue[resolved_t_id]
                if task.strategy == RoutingStrategy.VERSION_BUMP:
                    prior_diag = retry_diagnostics_by_task.get(resolved_t_id)
                    group = group_by_id.get(task.parent_group_id)
                    if prior_diag is None:
                        retry_diagnostics_by_task[resolved_t_id] = UpdateRetryDiagnostics(
                            task_id=resolved_t_id,
                            strategy_stage=next_stage,
                            security_floor=(
                                group.fix_plan.fixed_version
                                if group and group.fix_plan
                                else None
                            ),
                            exhausted_update_path=(
                                next_stage == SCARemediationStage.CODE_WORKAROUND
                            ),
                        )
                    else:
                        retry_diagnostics_by_task[resolved_t_id] = prior_diag.model_copy(
                            update={
                                "strategy_stage": next_stage,
                                "security_floor": prior_diag.security_floor
                                or (
                                    group.fix_plan.fixed_version
                                    if group and group.fix_plan
                                    else None
                                ),
                                "exhausted_update_path": next_stage == SCARemediationStage.CODE_WORKAROUND,
                            }
                        )

    # ------------------------------------------------------------------
    # 4. Mark UNFIXABLE tasks that hit the retry cap
    # ------------------------------------------------------------------
    for task_id, task in task_queue.items():
        if (
            task.status == TaskStatus.NEEDS_RETRY
            and task.retry_count >= MAX_RETRIES
            and not _is_exhausted_update_pivot_candidate(
                task,
                retry_diagnostics_by_task.get(task_id),
            )
        ):
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.current_attempt_id is not None,
            )
            logger.info(
                "supervisor: task '%s' marked UNFIXABLE after %d retries.",
                task_id,
                task.retry_count,
            )

    # ------------------------------------------------------------------
    # 5. Short-circuit: if active batch is all optimistically_fixed → qa_critic
    # ------------------------------------------------------------------
    decision: Optional[SupervisorDecision] = None
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing {len(active_qa_ready)} optimistically fixed task(s) from the current batch to QA."
                ),
            )

    # Short-circuit: all remaining non-terminal tasks are optimistically_fixed
    if decision is None:
        tasks = list(task_queue.values())
        non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
        if not non_terminal:
            decision = SupervisorDecision(
                next_node="teardown",
                target_task_ids=[],
                instructions="All tasks are terminal. Proceeding to teardown.",
                decision_reason="No actionable tasks remain.",
            )
        else:
            qa_ready = _qa_ready_task_ids(task_queue)
            if qa_ready:
                decision = SupervisorDecision(
                    next_node="qa_critic",
                    target_task_ids=qa_ready,
                    instructions="Run QA on the remaining optimistically fixed tasks.",
                    decision_reason="Routing remaining optimistically fixed tasks to QA.",
                )

    # ------------------------------------------------------------------
    # 6. Router phase (structured LLM call)
    # ------------------------------------------------------------------
    planner_scratchpad = ""
    if decision is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]

            model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
            router_llm = ChatOpenAI(model=model_name, temperature=0)
            if _needs_planner(
                task_queue,
                qa_evaluations,
                retry_diagnostics_by_task,
                str(state.get("status") or ""),
            ):
                logger.info("supervisor: invoking planner phase for retry analysis.")
                planner_base_diagnostics = dict(retry_diagnostics_by_task)
                planner_correction = ""
                planner_violations: List[str] = []
                parsed_diagnostics: Dict[str, UpdateRetryDiagnostics] = {}
                parsed_plans: Dict[str, SupervisorRetryPlan] = {}
                planner_tool_events: List[ToolEvent] = []

                # Planner output is an untrusted proposal.  Give it one
                # compact correction opportunity before applying anything to
                # the task queue or exposing it to the router.
                for planner_attempt in range(2):
                    planner_result = _run_planner_phase(
                        task_queue,
                        group_by_id,
                        qa_evaluations,
                        planner_base_diagnostics,
                        action_summaries,
                        existing_constraints,
                        router_llm,
                        correction=planner_correction,
                        return_tool_events=True,
                    )
                    if isinstance(planner_result, tuple):
                        planner_scratchpad, attempt_tool_events = planner_result
                        planner_tool_events.extend(attempt_tool_events)
                    else:
                        # Preserve compatibility with tests/integrations that
                        # replace the planner helper with a text-only stub.
                        planner_scratchpad = str(planner_result)
                    parsed_diagnostics, parsed_plans = _parse_planner_retry_plans(
                        planner_scratchpad,
                        task_queue,
                        planner_base_diagnostics,
                        group_by_id,
                    )
                    parsed_diagnostics, parsed_plans = _reconcile_registry_plan_evidence(
                        parsed_plans,
                        parsed_diagnostics,
                        task_queue,
                        group_by_id,
                        planner_tool_events,
                    )
                    planner_violations = _planner_plan_violations(
                        parsed_plans,
                        task_queue,
                        planner_base_diagnostics,
                    )
                    if not planner_violations:
                        break

                    errors.extend(
                        f"supervisor: planner semantic validation: {violation}"
                        for violation in planner_violations
                    )
                    if planner_attempt == 0:
                        planner_correction = "\n".join(planner_violations)
                        logger.warning(
                            "supervisor: rejecting planner output and requesting correction: %s",
                            planner_violations,
                        )
                        continue

                    # The second proposal is still untrusted.  Repair from
                    # registry facts already present in the proposal, or fail
                    # closed into the existing workaround pivot.  No stale or
                    # attempted version can survive this boundary.
                    parsed_diagnostics, parsed_plans = _repair_invalid_planner_plans(
                        parsed_plans,
                        parsed_diagnostics,
                        task_queue,
                        group_by_id,
                        violations=planner_violations,
                    )
                    repair_violations = _planner_plan_violations(
                        parsed_plans,
                        task_queue,
                        parsed_diagnostics,
                    )
                    if repair_violations:
                        errors.extend(
                            "supervisor: deterministic planner repair remained invalid: "
                            f"{violation}"
                            for violation in repair_violations
                        )
                    break

                candidate_violations = _planner_plan_violations(
                    parsed_plans,
                    task_queue,
                    parsed_diagnostics,
                )
                if not candidate_violations:
                    retry_diagnostics_by_task = dict(parsed_diagnostics)
                    retry_plans_by_task = dict(parsed_plans)
                    for task_id, plan in list(retry_plans_by_task.items()):
                        if task_id not in task_queue:
                            continue
                        committed_task = _commit_task_transition(
                            task_queue,
                            task_id,
                            updates={
                                "strategy_stage": plan.strategy_stage,
                                "instruction": plan.exact_instruction,
                                "selected_version": plan.selected_version,
                                "exhausted_update_path": plan.exhausted_update_path,
                            },
                            # A planner result supersedes any previous worker
                            # input. Closing here makes the planner commit
                            # atomic even when a stale worker result arrived
                            # in the same supervisor invocation.
                            close_attempt=True,
                        )
                        if committed_task is None:
                            continue
                        diagnostics = retry_diagnostics_by_task.get(task_id)
                        if diagnostics is not None:
                            retry_diagnostics_by_task[task_id] = diagnostics.model_copy(
                                update={
                                    "committed_attempt_id": committed_task.current_attempt_id,
                                    "strategy_stage": committed_task.strategy_stage,
                                    "selected_version": committed_task.selected_version,
                                    "exhausted_update_path": committed_task.exhausted_update_path,
                                    "instruction_digest": _instruction_digest(
                                        committed_task.instruction
                                    ),
                                }
                            )
                        retry_plans_by_task[task_id] = plan.model_copy(
                            update={"source_task_revision": committed_task.task_revision}
                        )
                else:
                    # Never commit an invalid planner proposal. Fail closed to
                    # a deterministic workaround pivot for affected retry tasks
                    # so a stale version/instruction cannot be dispatched.
                    # A malformed or incomplete planner response must not
                    # leave an older retry plan alive. Every actionable retry
                    # task is therefore moved through the same fail-closed
                    # pivot path, including tasks for which the planner
                    # emitted no parseable section at all.
                    affected_ids = {
                        task_id
                        for task_id, task in task_queue.items()
                        if task.status == TaskStatus.NEEDS_RETRY
                    }
                    affected_ids.update(
                        plan.task_id
                        for plan in parsed_plans.values()
                        if plan.task_id in task_queue
                    )
                    for task_id in affected_ids:
                        retry_plans_by_task.pop(task_id, None)
                    consistency_events.append(
                        _build_consistency_event(
                            error_code="INVALID_PLANNER_COMMIT",
                            task_id=next(iter(affected_ids), None),
                            expected_attempt_id=None,
                            received_attempt_id=None,
                            action="replanned",
                            details="Planner proposal rejected; deterministic pivot committed.",
                        )
                    )
                    for task_id in affected_ids:
                        task = task_queue[task_id]
                        group = group_by_id.get(task.parent_group_id)
                        component = group.vulnerable_component if group else task.parent_group_id
                        pivot_instruction = (
                            f"Implement a code workaround or isolation strategy for {component} "
                            "because the manifest-based update path is exhausted."
                        )
                        committed_task = _commit_task_transition(
                            task_queue,
                            task_id,
                            updates={
                                "strategy_stage": SCARemediationStage.NPM_LATEST,
                                "selected_version": None,
                                "exhausted_update_path": True,
                                "instruction": pivot_instruction,
                            },
                            close_attempt=True,
                            clear_selected_version=True,
                        )
                        retry_diagnostics_by_task[task_id] = UpdateRetryDiagnostics(
                            task_id=task_id,
                            committed_attempt_id=(
                                committed_task.current_attempt_id
                                if committed_task is not None
                                else None
                            ),
                            strategy_stage=SCARemediationStage.NPM_LATEST,
                            security_floor=(
                                group.fix_plan.fixed_version
                                if group and group.fix_plan
                                else None
                            ),
                            attempted_versions=list(
                                retry_diagnostics_by_task.get(task_id, UpdateRetryDiagnostics(task_id=task_id)).attempted_versions
                            ),
                            selected_version=None,
                            exhausted_update_path=True,
                        )
                        retry_plans_by_task[task_id] = SupervisorRetryPlan(
                            task_id=task_id,
                            source_task_revision=(
                                committed_task.task_revision
                                if committed_task is not None
                                else task.task_revision
                            ),
                            strategy_stage=SCARemediationStage.NPM_LATEST,
                            selected_version=None,
                            attempted_versions=retry_diagnostics_by_task[task_id].attempted_versions,
                            exhausted_update_path=True,
                            action="pivot_workaround",
                            exact_instruction=pivot_instruction,
                        )

                # A planner-confirmed exhausted path is deterministic. Do not
                # ask the router LLM to reinterpret a pivot as an update retry.
                if any(
                    plan.action == "pivot_workaround"
                    for plan in retry_plans_by_task.values()
                ):
                    decision = _deterministic_routing(
                        task_queue,
                        group_by_id,
                        qa_evaluations,
                        retry_diagnostics_by_task,
                        action_summaries=action_summaries,
                        active_target_task_ids=active_target_task_ids,
                        current_status=str(state.get("status") or ""),
                    )
            structured_llm = router_llm.with_structured_output(
                SupervisorDecision, method="function_calling"
            )
            if decision is None:
                prompt_state: OrchestratorState = {  # type: ignore[typeddict-item]
                    **state,
                    "task_queue": task_queue,
                    "qa_evaluations": qa_evaluations,
                    "retry_diagnostics_by_task": retry_diagnostics_by_task,
                    "retry_plans_by_task": retry_plans_by_task,
                }
                prompt_text = build_supervisor_prompt(
                    prompt_state,
                    scratchpad=planner_scratchpad,
                )
                logger.info("supervisor: invoking structured router LLM.")
                decision = invoke_with_trajectory(
                    "supervisor.router",
                    lambda: structured_llm.invoke(prompt_text),
                    prompt_text,
                )
                logger.info(
                    "supervisor: router decision → next_node=%s targets=%s reason=%s",
                    decision.next_node,
                    decision.target_task_ids,
                    decision.decision_reason,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor: LLM call failed (%s) — using deterministic fallback.", exc
            )

    # Re-apply optimistic short-circuit post-LLM (guard against LLM overriding)
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated batch before starting more remediation.",
                decision_reason=(
                    f"Routing {len(active_qa_ready)} optimistically fixed task(s) from the current batch to QA."
                ),
            )

    # ------------------------------------------------------------------
    # 7. Guardrails: validate and clamp (or deterministic fallback)
    # ------------------------------------------------------------------
    pivot_parent_status_by_parent: Dict[str, TaskStatus] = {}
    pivot_target_parent_ids: Set[str] = set()

    if decision is None:
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
        )
        logger.info(
            "supervisor: deterministic fallback → next_node=%s", decision.next_node
        )
        fallback_pivot_strategy_by_parent = {
            req.parent_task_id: req.strategy
            for req in decision.spawn_requests
            if req.parent_task_id in task_queue
            and task_queue[req.parent_task_id].strategy != req.strategy
        }
        for parent_id, new_strategy in fallback_pivot_strategy_by_parent.items():
            pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                task_queue[parent_id],
                new_strategy,
                qa_evaluations,
            )
        pivot_target_parent_ids = {
            task_id
            for task_id in decision.target_task_ids
            if task_id in fallback_pivot_strategy_by_parent
        }
    else:
        known_task_ids = set(task_queue.keys())

        raw_pivot_parent_ids = {
            req.parent_task_id
            for req in decision.spawn_requests
            if req.parent_task_id in known_task_ids
            and task_queue[req.parent_task_id].status not in _TERMINAL_STATUSES
            and task_queue[req.parent_task_id].status in _WORKABLE_STATUSES
            and task_queue[req.parent_task_id].strategy != req.strategy
            and decision.next_node == _worker_node_for_strategy(req.strategy)
        }
        raw_pivot_parent_ids.update(
            task_id
            for task_id, new_strategy in decision.updated_task_strategies.items()
            if task_id in known_task_ids
            and task_queue[task_id].status not in _TERMINAL_STATUSES
            and task_queue[task_id].status in _WORKABLE_STATUSES
            and task_queue[task_id].strategy != new_strategy
            and decision.next_node == _worker_node_for_strategy(new_strategy)
        )

        if decision.next_node == "qa_critic":
            valid_target_ids = _qa_ready_task_ids(
                task_queue,
                preferred_ids=list(decision.target_task_ids),
            )
        elif decision.next_node == "update_subagent":
            valid_target_ids = _update_worker_task_ids(
                task_queue,
                retry_diagnostics_by_task,
                preferred_ids=list(decision.target_task_ids),
                limit=UPDATE_BATCH_SIZE,
            )
        elif decision.next_node == "workaround_subagent":
            valid_target_ids = []
            for t_id in decision.target_task_ids:
                if t_id not in known_task_ids:
                    continue
                task = task_queue[t_id]
                if task.status in _TERMINAL_STATUSES or task.status not in _WORKABLE_STATUSES:
                    continue
                if task.strategy == RoutingStrategy.CODE_WORKAROUND or t_id in raw_pivot_parent_ids:
                    valid_target_ids.append(t_id)
            if not valid_target_ids and len(raw_pivot_parent_ids) == 1:
                # A pivot decision may contain the child spawn request but omit
                # the parent target. Recover the single unambiguous parent
                # before applying workaround cardinality validation.
                valid_target_ids = [next(iter(raw_pivot_parent_ids))]
        else:
            valid_target_ids = []

        valid_unfixable_ids = [
            t_id for t_id in decision.unfixable_task_ids if t_id in known_task_ids
        ]

        # Enforce cardinality constraints
        needs_fallback = False
        if decision.next_node == "workaround_subagent" and len(valid_target_ids) != 1:
            logger.warning(
                "supervisor: workaround_subagent needs 1 target, got %d — falling back.",
                len(valid_target_ids),
            )
            needs_fallback = True
        elif decision.next_node == "update_subagent" and not valid_target_ids:
            logger.warning(
                "supervisor: update_subagent needs ≥1 target, got 0 — falling back."
            )
            needs_fallback = True

        if not needs_fallback and decision.next_node == "update_subagent" and len(valid_target_ids) > UPDATE_BATCH_SIZE:
            logger.warning(
                "supervisor: update_subagent supports at most %d targets, got %d — falling back.",
                UPDATE_BATCH_SIZE,
                len(valid_target_ids),
            )
            needs_fallback = True
        if (
            not needs_fallback
            and decision.next_node == "update_subagent"
            and valid_target_ids
        ):
            has_retry_targets = any(
                task_queue[t_id].status == TaskStatus.NEEDS_RETRY
                or task_queue[t_id].retry_count > 0
                for t_id in valid_target_ids
            )
            has_first_pass_targets = any(
                task_queue[t_id].status != TaskStatus.NEEDS_RETRY
                and task_queue[t_id].retry_count == 0
                for t_id in valid_target_ids
            )
            if has_retry_targets and has_first_pass_targets:
                logger.warning(
                    "supervisor: update_subagent batch mixed first-pass and retry tasks — falling back."
                )
                needs_fallback = True
        if not needs_fallback and decision.next_node == "qa_critic" and not valid_target_ids:
            logger.warning(
                "supervisor: qa_critic needs at least 1 target, got 0 — falling back."
            )
            needs_fallback = True

        if needs_fallback:
            decision = _deterministic_routing(
                task_queue,
                group_by_id,
                qa_evaluations,
                retry_diagnostics_by_task,
                action_summaries=action_summaries,
                active_target_task_ids=active_target_task_ids,
                current_status=str(state.get("status") or ""),
            )
            fallback_pivot_strategy_by_parent = {
                req.parent_task_id: req.strategy
                for req in decision.spawn_requests
                if req.parent_task_id in task_queue
                and task_queue[req.parent_task_id].strategy != req.strategy
            }
            for parent_id, new_strategy in fallback_pivot_strategy_by_parent.items():
                pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                    task_queue[parent_id],
                    new_strategy,
                    qa_evaluations,
                )
            pivot_target_parent_ids = {
                task_id
                for task_id in decision.target_task_ids
                if task_id in fallback_pivot_strategy_by_parent
            }
        else:
            # Filter revised_instructions and feedback_by_task to known task IDs
            clean_revised_instructions = {
                k: v
                for k, v in decision.revised_instructions.items()
                if k in known_task_ids and v.strip()
            }
            # The reconciled task queue is authoritative. Preserve the public
            # revised_instructions field while filling it from committed plans
            # when the router omits the field or repeats stale text.
            for task_id in valid_target_ids:
                task = task_queue.get(task_id)
                plan = retry_plans_by_task.get(task_id)
                if (
                    task is not None
                    and task.status == TaskStatus.NEEDS_RETRY
                    and plan is not None
                    and plan.action == "retry_update"
                ):
                    clean_revised_instructions[task_id] = task.instruction
            clean_feedback = {
                k: v
                for k, v in decision.feedback_by_task.items()
                if k in known_task_ids
            }
            clean_updated_task_strategies = {
                k: v
                for k, v in decision.updated_task_strategies.items()
                if k in known_task_ids and task_queue[k].status not in _TERMINAL_STATUSES
            }
            missing_retry_revisions = _missing_retry_revised_instructions(
                decision.next_node,
                valid_target_ids,
                clean_revised_instructions,
                task_queue,
            )
            if missing_retry_revisions:
                errors.append(
                    "supervisor: rejected update_subagent retry dispatch without task-specific "
                    f"revised_instructions for {missing_retry_revisions}."
                )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)

            # Validate task_status_updates — only known tasks, only terminal statuses
            clean_status_updates: Dict[str, TaskStatus] = {}
            _allowed_statuses = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
            for t_id, new_status in decision.task_status_updates.items():
                if t_id not in known_task_ids:
                    errors.append(
                        f"supervisor: task_status_updates rejected unknown task_id '{t_id}'."
                    )
                    continue
                if new_status not in _allowed_statuses:
                    errors.append(
                        f"supervisor: task_status_updates rejected disallowed status "
                        f"'{new_status}' for task '{t_id}'."
                    )
                    continue
                clean_status_updates[t_id] = new_status

            clean_spawn_requests = [
                req
                for req in decision.spawn_requests
                if req.parent_task_id in known_task_ids
                and task_queue[req.parent_task_id].status not in _TERMINAL_STATUSES
            ]
            pivot_strategy_by_parent: Dict[str, RoutingStrategy] = {
                req.parent_task_id: req.strategy
                for req in clean_spawn_requests
                if task_queue[req.parent_task_id].strategy != req.strategy
            }
            legacy_pivot_strategy_updates = {
                task_id: new_strategy
                for task_id, new_strategy in clean_updated_task_strategies.items()
                if task_queue[task_id].strategy != new_strategy
                and task_id not in pivot_strategy_by_parent
            }
            malformed_pivot_parent_ids: List[str] = []
            for task_id, new_strategy in legacy_pivot_strategy_updates.items():
                child_instruction = clean_revised_instructions.pop(task_id, "").strip()
                if not child_instruction:
                    malformed_pivot_parent_ids.append(task_id)
                    continue
                clean_spawn_requests.append(
                    TaskSpawnRequest(
                        parent_task_id=task_id,
                        strategy=new_strategy,
                        instruction=child_instruction,
                        reason=(
                            "Auto-converted legacy strategy pivot from "
                            f"{task_queue[task_id].strategy.value} to {new_strategy.value}. "
                            f"{decision.decision_reason}"
                        ),
                    )
                )
                pivot_strategy_by_parent[task_id] = new_strategy

            targeted_pivot_ids = [
                task_id for task_id in valid_target_ids if task_id in pivot_strategy_by_parent
            ]
            incompatible_targeted_pivots = [
                task_id
                for task_id in targeted_pivot_ids
                if decision.next_node != _worker_node_for_strategy(
                    pivot_strategy_by_parent[task_id]
                )
            ]
            pivot_validation_failed = bool(malformed_pivot_parent_ids or incompatible_targeted_pivots)
            if pivot_validation_failed:
                if malformed_pivot_parent_ids:
                    errors.append(
                        "supervisor: rejected strategy pivot without task-specific child "
                        f"instructions for {malformed_pivot_parent_ids}."
                    )
                if incompatible_targeted_pivots:
                    errors.append(
                        "supervisor: rejected strategy pivot because next_node does not match "
                        f"the child strategy for {incompatible_targeted_pivots}."
                    )
                failed_parent_ids = list({
                    *malformed_pivot_parent_ids,
                    *incompatible_targeted_pivots,
                })
                _terminalize_pivot_parents(
                    task_queue,
                    failed_parent_ids,
                    pivot_strategy_by_parent | legacy_pivot_strategy_updates,
                    qa_evaluations,
                    retry_diagnostics_by_task=retry_diagnostics_by_task,
                    retry_plans_by_task=retry_plans_by_task,
                    group_by_id=group_by_id,
                )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)
                clean_status_updates = dict(decision.task_status_updates)
                clean_spawn_requests = list(decision.spawn_requests)
                pivot_strategy_by_parent = {
                    req.parent_task_id: req.strategy
                    for req in clean_spawn_requests
                    if req.parent_task_id in task_queue
                    and task_queue[req.parent_task_id].strategy != req.strategy
                }
                targeted_pivot_ids = [
                    task_id for task_id in valid_target_ids if task_id in pivot_strategy_by_parent
                ]

            for parent_id, new_strategy in pivot_strategy_by_parent.items():
                pivot_parent_status_by_parent[parent_id] = _parent_status_for_strategy_pivot(
                    task_queue[parent_id],
                    new_strategy,
                    qa_evaluations,
                )
            pivot_target_parent_ids = set(targeted_pivot_ids)

            try:
                decision = SupervisorDecision(
                    next_node=decision.next_node,
                    updated_task_strategies={},
                    target_task_ids=valid_target_ids,
                    unfixable_task_ids=valid_unfixable_ids,
                    new_constraints=decision.new_constraints,
                    feedback_by_task=clean_feedback,
                    revised_instructions=clean_revised_instructions,
                    spawn_requests=clean_spawn_requests,
                    task_status_updates=clean_status_updates,
                    instructions=decision.instructions,
                    decision_reason=decision.decision_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "supervisor: decision rebuild failed (%s) — falling back.", exc
                )
                pivot_parent_status_by_parent = {}
                pivot_target_parent_ids = set()
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )

    # ------------------------------------------------------------------
    # 8. Apply guarded updates to task_queue
    # ------------------------------------------------------------------

    # 8a. Apply revised_instructions (copy-on-write per task)
    for t_id, new_instr in decision.revised_instructions.items():
        if t_id in task_queue and new_instr.strip():
            task = task_queue[t_id]
            if task.current_attempt_id is not None:
                errors.append(
                    f"supervisor: ignored revised instruction for {t_id} because its "
                    "current attempt is still active."
                )
                continue
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"instruction": new_instr},
            )

    # 8b. Apply direct strategy pivots (currently reserved for no-op / legacy cases)
    for t_id, new_strategy in decision.updated_task_strategies.items():
        if t_id in task_queue:
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"strategy": new_strategy},
                close_attempt=task_queue[t_id].current_attempt_id is not None,
            )

    # 8c. Apply guarded task status overrides (only QA_PASSED and UNFIXABLE)
    _allowed_statuses = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
    for t_id, new_status in decision.task_status_updates.items():
        if t_id in task_queue and new_status in _allowed_statuses:
            if task_queue[t_id].status not in _TERMINAL_STATUSES:
                _commit_task_transition(
                    task_queue,
                    t_id,
                    updates={"status": new_status},
                    close_attempt=(
                        new_status in _TERMINAL_STATUSES
                        and task_queue[t_id].current_attempt_id is not None
                    ),
                    clear_selected_version=(
                        new_status in _TERMINAL_STATUSES
                        and task_queue[t_id].current_attempt_id is not None
                    ),
                )
                logger.info(
                    "supervisor: task '%s' manually set to %s via task_status_updates.",
                    t_id,
                    new_status.value,
                )

    # 8d. Apply unfixable marks from decision
    for t_id in decision.unfixable_task_ids:
        if t_id in task_queue:
            _commit_task_transition(
                task_queue,
                t_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task_queue[t_id].current_attempt_id is not None,
                clear_selected_version=task_queue[t_id].current_attempt_id is not None,
            )

    # 8e. Materialize spawn requests
    child_ids_by_parent: Dict[str, List[str]] = {}
    if decision.spawn_requests:
        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=list(decision.spawn_requests),
            task_queue=task_queue,
            group_by_id=group_by_id,
            errors=errors,
        )
        task_queue.update(new_tasks)
        # Apply the complete parent transition after children are materialized.
        # This must detach the closed update attempt as well as terminalize the
        # parent; otherwise the parent remains paired with an update snapshot
        # while routing has already moved to the workaround child.
        _terminalize_pivot_parents(
            task_queue,
            list(pivot_parent_status_by_parent),
            {
                parent_id: next(
                    (
                        request.strategy
                        for request in decision.spawn_requests
                        if request.parent_task_id == parent_id
                    ),
                    RoutingStrategy.CODE_WORKAROUND,
                )
                for parent_id in pivot_parent_status_by_parent
            },
            qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
            retry_plans_by_task=retry_plans_by_task,
            group_by_id=group_by_id,
        )

    resolved_target_task_ids: List[str] = []
    remapped_feedback_by_task: Dict[str, str] = {}
    for task_id in decision.target_task_ids:
        if task_id in pivot_target_parent_ids:
            child_ids = child_ids_by_parent.get(task_id, [])
            if child_ids:
                child_task_id = child_ids[0]
                resolved_target_task_ids.append(child_task_id)
                if task_id in decision.feedback_by_task:
                    remapped_feedback_by_task[child_task_id] = decision.feedback_by_task[task_id]
            else:
                errors.append(
                    "supervisor: dropped strategy-pivot target because child task could not "
                    f"be spawned for parent '{task_id}'."
                )
            continue
        resolved_target_task_ids.append(task_id)
        if task_id in decision.feedback_by_task:
            remapped_feedback_by_task[task_id] = decision.feedback_by_task[task_id]

    resolved_next_node = decision.next_node
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
    )
    # Status overrides and parent terminalization above are also untrusted
    # router requests. Re-clamp after those mutations so a task that became
    # terminal in this transition cannot remain in the dispatch projection.
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
    )
    remapped_feedback_by_task = {
        task_id: feedback
        for task_id, feedback in remapped_feedback_by_task.items()
        if task_id in set(resolved_target_task_ids)
    }
    if resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"} and not resolved_target_task_ids:
        errors.append(
            "supervisor: routing fell back to teardown because no dispatchable target tasks remained."
        )
        resolved_next_node = "teardown"
        resolved_target_task_ids = []
        remapped_feedback_by_task = {}

    # Commit the exact input snapshot before exposing worker targets to the
    # graph. QA reuses the current worker attempt; update/workaround dispatches
    # always receive a new attempt identity.
    if resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"}:
        for task_id in list(resolved_target_task_ids):
            task = task_queue.get(task_id)
            if task is None:
                continue
            # Normal QA follows the worker attempt and reuses its snapshot.
            # Only the safe initial-migration path needs a synthetic QA
            # snapshot for a legacy optimistic result that arrived without an
            # attempt envelope.
            if resolved_next_node == "qa_critic" and task.current_attempt_id:
                continue
            plan = retry_plans_by_task.get(task_id)
            task, snapshot = _create_attempt_snapshot(
                task,
                dispatch_node=resolved_next_node,
                snapshots_by_id=attempt_snapshots_by_id,
                state_revision=state_revision,
                plan_id=plan.plan_id if plan is not None else None,
            )
            task_queue[task_id] = task
            prior = retry_diagnostics_by_task.get(task_id)
            if prior is not None:
                retry_diagnostics_by_task[task_id] = prior.model_copy(
                    update={
                        "committed_attempt_id": snapshot.attempt_id,
                        "selected_version": task.selected_version,
                        "strategy_stage": task.strategy_stage,
                        "exhausted_update_path": task.exhausted_update_path,
                        "instruction_digest": snapshot.instruction_digest,
                    }
                )
            if plan is not None:
                # The plan was created against the pre-dispatch task revision.
                # Once its exact input is committed, keep the compatibility
                # plan projection correlated to that same revision.
                retry_plans_by_task[task_id] = plan.model_copy(
                    update={"source_task_revision": task.task_revision}
                )

    logger.info(
        "supervisor: routing to '%s' with targets=%s",
        resolved_next_node,
        resolved_target_task_ids,
    )

    # ------------------------------------------------------------------
    # 8f. Collect new constraints
    # ------------------------------------------------------------------
    returned_constraints: List[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # Build feedback_by_group for bridge node backward compat
    feedback_by_task = remapped_feedback_by_task
    feedback_by_group: Dict[str, str] = {}
    for t_id, fb in feedback_by_task.items():
        if t_id in task_queue:
            gid = task_queue[t_id].parent_group_id
            feedback_by_group[gid] = fb

    consistency_new_events, consistency_errors = _validate_committed_state(
        task_queue,
        attempt_snapshots_by_id,
        retry_plans_by_task,
        retry_diagnostics_by_task,
        resolved_target_task_ids,
        resolved_next_node,
    )
    consistency_events.extend(consistency_new_events)
    errors.extend(consistency_errors)
    existing_event_keys = {
        (event.task_id, event.received_attempt_id, event.error_code)
        for event in prior_consistency_events
    }
    consistency_events = [
        event
        for event in _dedupe_consistency_events(consistency_events)
        if (event.task_id, event.received_attempt_id, event.error_code)
        not in existing_event_keys
    ]
    # ``errors`` uses an additive reducer, so suppress both duplicate messages
    # from this invocation and exact messages already committed by an earlier
    # supervisor pass. Structured consistency events remain the detailed,
    # attempt-correlated audit record.
    errors = list(
        dict.fromkeys(
            error for error in errors if error not in prior_error_messages
        )
    )

    # ------------------------------------------------------------------
    # 9. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "active_target_task_ids": resolved_target_task_ids,
        # Keep active_target_group_ids populated for bridge nodes that still use it
        "active_target_group_ids": [
            task_queue[t].parent_group_id
            for t in resolved_target_task_ids
            if t in task_queue
        ],
        "feedback_by_task": feedback_by_task,
        "feedback_by_group": feedback_by_group,
        "supervisor_instructions": decision.instructions,
        # Compatibility projection: the attempt-tagged QA envelope remains
        # authoritative, while this task-keyed view is retained for existing
        # callers and prompt builders.
        "qa_evaluations": qa_evaluations,
        "task_queue": task_queue,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "retry_plans_by_task": retry_plans_by_task,
        "attempt_snapshots_by_id": attempt_snapshots_by_id,
        "worker_results_by_attempt": worker_results_by_attempt,
        "qa_results_by_attempt": qa_results_by_attempt,
        "processed_worker_attempt_ids": list(new_worker_attempt_ids),
        "processed_qa_attempt_ids": list(new_qa_attempt_ids),
        "consistency_events": consistency_events,
        "state_revision": state_revision,
        # constraints_ledger uses operator.add — return only NEW entries
        "constraints_ledger": returned_constraints,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Conditional-edge router
# ---------------------------------------------------------------------------


def supervisor_router(state: OrchestratorState) -> str:
    """
    Conditional-edge callable for the supervisor node.

    Reads ``state["next_routing_step"]`` and returns the target node name.
    Defaults to ``"teardown"`` for any unknown or missing value.
    """
    step = state.get("next_routing_step", "")
    if step in _VALID_NEXT_NODES:
        return step
    logger.warning(
        "supervisor_router: unknown next_routing_step '%s' — defaulting to teardown.",
        step,
    )
    return "teardown"
