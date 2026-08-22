"""
supervisor_node.py - Agentic Supervisor Node for Phase 5 hub-and-spoke orchestration.

Phase 5 architecture: deterministic Supervisor
-----------------------------------------------
The Supervisor owns the state machine and produces every routing and retry
decision in Python. Registry facts are inputs to the deterministic retry
planner; they are never selected by a model or delegated to a worker.

  Guardrails (Python):
    Validate and apply the decision: reject unknown task IDs, clamp cardinality,
    apply copy-on-write task updates, materialize spawn requests, enforce depth
    and queue-size caps.

Public API
----------
MAX_RETRIES : int
    Maximum number of QA-fail-retry cycles before a task is marked unfixable.
run_supervisor_node(state) -> Dict[str, Any]
    LangGraph node callable.
supervisor_router(state) -> str
    Conditional-edge callable: reads ``next_routing_step`` from state.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from remediation_engine.contracts.decision_codes import (
    DecisionCode,
    validate_transition,
)
from remediation_engine.contracts.schemas import (
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    NoFixMitigationStage,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorDecision,
    SupervisorRetryPlan,
    TaskAttemptSnapshot,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundPhase,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.contracts.supervisor_phases import (
    AuditRecord,
    EligibleActions,
    ReconciliationResult,
)
from remediation_engine.contracts.version_policy import select_version
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.subagent_runtime import ToolEvent
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _WORKABLE_STATUSES,
    MAX_RETRIES,
    _dispatchable_task_ids_for_status,
    _has_existing_workaround_child,
    _is_exhausted_update_pivot_candidate,
    _next_sca_stage,
    _parent_status_for_strategy_pivot,
    _qa_ready_task_ids,
    _task_sort_key,
    _worker_node_for_strategy,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage,
    build_initial_remediation_task,
    build_no_fix_package_removal_instruction,
    build_no_fix_retry_instruction,
    derive_missing_task_qa_policy,
    group_parent_context,
    is_no_fix_group,
    is_transitive_group,
)
from remediation_engine.tools.registry_tools import (
    fetch_registry_candidates,
    plan_npm_parent_version,
)

logger = logging.getLogger(__name__)

# Supervisor dispatch is intentionally per-task.  The worker and QA helpers
# remain batch-capable for direct callers and a future explicit batch mode.
UPDATE_DISPATCH_LIMIT: int = 1
QA_DISPATCH_LIMIT: int = 1

_VALID_NEXT_NODES: set[str] = {
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "triage",
    "final_full_scan",
    "teardown",
}
_WORKER_NODES = frozenset({"update_subagent", "workaround_subagent", "qa_critic"})
# Keep legacy scratchpad stage parsing and validation centralized. Historical
# planner evidence commonly uses the shorter ``same_major`` spelling. Accept
# that spelling explicitly; do not let an unrecognised value silently become
# the task's current stage.
_PLANNER_STAGE_ALIASES: dict[str, SCARemediationStage] = {
    "osv": SCARemediationStage.OSV_MINIMUM,
    "minimum": SCARemediationStage.OSV_MINIMUM,
    "osv_minimum": SCARemediationStage.OSV_MINIMUM,
    "same_major": SCARemediationStage.NPM_SAME_MAJOR,
    "npm_same_major": SCARemediationStage.NPM_SAME_MAJOR,
    "latest": SCARemediationStage.NPM_LATEST,
    "npm_latest": SCARemediationStage.NPM_LATEST,
    "package_override": SCARemediationStage.PACKAGE_OVERRIDE,
    "override": SCARemediationStage.PACKAGE_OVERRIDE,
    "code_workaround": SCARemediationStage.CODE_WORKAROUND,
}
_SCA_STAGE_ORDER: dict[SCARemediationStage, int] = {
    SCARemediationStage.OSV_MINIMUM: 0,
    SCARemediationStage.NPM_SAME_MAJOR: 1,
    SCARemediationStage.NPM_LATEST: 2,
    SCARemediationStage.PACKAGE_OVERRIDE: 3,
    SCARemediationStage.CODE_WORKAROUND: 4,
}
_OVERRIDE_DEPENDENCY_TYPES = frozenset({"overrides", "resolutions", "pnpm_overrides"})


def _instruction_digest(instruction: str) -> str:
    """Return the stable digest used to correlate worker input and output."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _attempts_for_task(
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    task_id: str,
) -> list[TaskAttemptSnapshot]:
    return sorted(
        (snapshot for snapshot in snapshots_by_id.values() if snapshot.task_id == task_id),
        key=lambda snapshot: (snapshot.attempt_number, snapshot.created_at),
    )


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
        action=action,  # type: ignore[arg-type]
        details=details,
    )


def _dedupe_consistency_events(
    events: list[StateConsistencyEvent],
) -> list[StateConsistencyEvent]:
    """Keep one consistency event per task/attempt/error tuple."""
    result: list[StateConsistencyEvent] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for event in events:
        key = (event.task_id, event.received_attempt_id, event.error_code)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def _current_action_summaries(
    action_summaries: list[AgentActionSummary],
    task_queue: dict[str, RemediationTask],
    limit: int,
) -> list[AgentActionSummary]:
    """Return only summaries belonging to each task's committed attempt."""
    relevant: list[AgentActionSummary] = []
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


def _extract_workaround_vulnerability_mechanism(group: VulnerabilityGroup) -> str:
    """Return the scanner-described security mechanism for workaround prompts."""
    for issue in getattr(group, "issues", []) or []:
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message.strip():
            continue
        mechanism = message
        for marker in ("### Am I affected?", "### How to fix that?"):
            mechanism = mechanism.split(marker, 1)[0]
        mechanism = re.sub(r"\s+", " ", mechanism).strip()
        if mechanism:
            return mechanism[:1200]

    fix_plan = getattr(group, "fix_plan", None)
    instruction = getattr(fix_plan, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return re.sub(r"\s+", " ", instruction).strip()[:1200]
    return ""


def _create_attempt_snapshot(
    task: RemediationTask,
    *,
    dispatch_node: str,
    snapshots_by_id: dict[str, TaskAttemptSnapshot],
    state_revision: int,
    plan_id: str | None = None,
    workaround_context: WorkaroundContext | None = None,
) -> tuple[RemediationTask, TaskAttemptSnapshot]:
    """Commit the exact worker input and return the revised task projection."""
    task_revision = task.task_revision + 1
    # Attempt identity is derived from committed state rather than wall-clock
    # randomness.  This makes replaying an identical OrchestratorState produce
    # the same dispatch projection while still changing identity whenever the
    # task revision or supervisor state revision changes.
    attempt_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"remediation-attempt:{task.task_id}:{task_revision}:{state_revision}:{dispatch_node}",
        )
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        state_revision=state_revision,
        task_revision=task_revision,
        attempt_number=len(_attempts_for_task(snapshots_by_id, task.task_id)) + 1,
        qa_policy=task.qa_policy,
        strategy_stage=task.strategy_stage,
        no_fix_stage=task.no_fix_stage,
        selected_version=task.selected_version,
        target_package_name=task.target_package_name,
        target_dependency_type=task.target_dependency_type,
        parent_minimum_version=task.parent_minimum_version,
        instruction=task.instruction,
        instruction_digest=_instruction_digest(task.instruction),
        dispatch_node=dispatch_node,  # type: ignore[arg-type]
        plan_id=plan_id,
        created_at=datetime.fromtimestamp(state_revision, tz=UTC),
        workaround_context=workaround_context,
    )
    snapshots_by_id[attempt_id] = snapshot
    updated_task = task.model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
        }
    )
    return updated_task, snapshot


def _qa_failure_evidence_for_workaround_retry(
    task_id: str,
    parent_group_id: str,
    qa_evaluations: dict[str, QAEvaluation],
    qa_results_by_attempt: dict[str, QAAttemptResult],
    *,
    related_task_ids: Iterable[str] = (),
    related_group_ids: Iterable[str] = (),
) -> QAFailureEvidence | None:
    """Resolve QA evidence for a workaround dispatch and its task ancestry.

    QA closes the failed worker attempt before Supervisor creates the retry
    snapshot. Therefore the failed evidence's ``attempt_id`` must not be
    compared with the new task's ``current_attempt_id`` (which is ``None`` at
    this point). Prefer the authoritative attempt envelope and fall back to
    the task-keyed compatibility projection for legacy callers.

    A workaround pivot creates a child task with a new task ID and usually a
    strategy-specific group ID. Its first QA failure still belongs to the
    parent update attempt, so the caller supplies the parent's task/group IDs
    through ``related_task_ids`` and ``related_group_ids``. Evidence for the
    current task always takes precedence over inherited evidence.
    """
    task_ids = list(dict.fromkeys([task_id, *related_task_ids]))
    group_ids = list(dict.fromkeys([parent_group_id, *related_group_ids]))

    for evaluation_key in [*task_ids, *group_ids]:
        evaluation = qa_evaluations.get(evaluation_key)
        evaluation_evidence = evaluation.failure_evidence if evaluation else None

        if evaluation_evidence and evaluation_evidence.attempt_id:
            envelope = qa_results_by_attempt.get(evaluation_evidence.attempt_id)
            if (
                envelope is not None
                and envelope.task_id in task_ids
                and envelope.evaluation.failure_evidence is not None
            ):
                return envelope.evaluation.failure_evidence

        if evaluation_evidence is not None:
            return evaluation_evidence

    candidates = [
        result.evaluation.failure_evidence
        for result in qa_results_by_attempt.values()
        if (
            result.task_id in task_ids
            and not result.evaluation.passed
            and result.evaluation.failure_evidence is not None
        )
    ]
    return candidates[-1] if candidates else None


def _qa_evidence_indicates_test_regression(
    evidence: QAFailureEvidence | None,
) -> bool:
    """Return whether inherited QA evidence represents a test/build regression."""
    if evidence is None:
        return False
    if evidence.failed_tests:
        return True

    text = " ".join(
        [
            *(evidence.exact_diagnostics or []),
            evidence.raw_excerpt or "",
            *(evidence.source_locations or []),
        ]
    ).lower()
    return any(
        marker in text
        for marker in (
            "npm test",
            "test failed",
            "tests failed",
            "typecheck",
            "compile failed",
            "build failed",
            "is not a function",
            "typeerror",
            "/test/",
            "\\test\\",
        )
    )


def _workaround_task_ancestry(
    task: RemediationTask,
    task_queue: dict[str, RemediationTask],
) -> tuple[list[str], list[str]]:
    """Return parent task/group IDs whose QA evidence may seed a child attempt.

    The task queue is authoritative for parent links. Missing or cyclic links
    are ignored defensively so malformed state cannot block workaround
    dispatch.

    Args:
        task: Workaround task being dispatched.
        task_queue: Current supervisor task queue.

    Returns:
        A tuple of ``(parent_task_ids, parent_group_ids)`` ordered from the
        immediate parent toward the root task.
    """
    parent_task_ids: list[str] = []
    parent_group_ids: list[str] = []
    seen: set[str] = set()
    parent_task_id = task.parent_task_id

    while parent_task_id and parent_task_id not in seen:
        seen.add(parent_task_id)
        parent_task = task_queue.get(parent_task_id)
        if parent_task is None:
            break
        parent_task_ids.append(parent_task.task_id)
        parent_group_ids.append(parent_task.parent_group_id)
        parent_task_id = parent_task.parent_task_id

    return parent_task_ids, parent_group_ids


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


def _commit_task_transition(
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
            and not validate_transition(current_status, new_status)
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
            logger.error(
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
    if isinstance(committed_status, TaskStatus) and committed_status in _TERMINAL_STATUSES:
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


def _validate_committed_state(
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


def reconcile_phase5_state_before_teardown(
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
        "active_target_group_ids": [],
        "next_routing_step": "teardown",
        "state_revision": int(state.get("state_revision", 0)) + 1,
        "consistency_events": new_events,
        "errors": new_errors,
    }


def _parse_planner_retry_plans(
    scratchpad: str,
    task_queue: dict[str, RemediationTask],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Parse planner markers and reconcile them into typed per-task plans.

    The planner scratchpad remains useful audit evidence, but this function is
    the only place where planner output becomes routing state.  In particular,
    ``SELECTED_VERSION: NONE`` clears stale selections instead of leaving the
    previous retry candidate active.
    """
    updated = dict(diagnostics_by_task)
    plans: dict[str, SupervisorRetryPlan] = {}
    sections: dict[str, list[str]] = {}
    current_task: str | None = None
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

        # Keep the latest registry fact when legacy scratchpad evidence states
        # it. This is audit evidence as well as a safe deterministic fallback
        # when a claimed version has already been tried.
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
            or pivot_recommended
            or "no new version" in lowered
            or "no valid candidate" in lowered
            or "only candidate" in lowered
            and "already been attempted" in lowered
            or (
                effective_stage == SCARemediationStage.NPM_LATEST
                and same_major_equals_latest
                and latest_candidate_already_attempted
            )
        ):
            exhausted = True

        if (
            not stage_match
            and effective_stage == SCARemediationStage.NPM_SAME_MAJOR
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

        # Legacy free-form pivot language cannot bypass the ordered
        # version stages. Direct tasks pivot only after NPM_LATEST. A
        # transitive task converts parent exhaustion into PACKAGE_OVERRIDE.
        group = (group_by_id or {}).get(task.parent_group_id)
        transitive = bool(group and is_transitive_group(group))
        if effective_stage != SCARemediationStage.NPM_LATEST:
            exhausted = bool(prior.package_abandoned)

        if (
            transitive
            and effective_stage == SCARemediationStage.NPM_LATEST
            and selected is None
            and exhausted
        ):
            selected = group.fix_plan.fixed_version if group and group.fix_plan else None
            effective_stage = SCARemediationStage.PACKAGE_OVERRIDE
            exhausted = False

        target_package = task.target_package_name
        target_type = task.target_dependency_type
        if effective_stage == SCARemediationStage.PACKAGE_OVERRIDE:
            target_package = group.vulnerable_component if group else task.parent_group_id
            target_type = _override_dependency_type(group)
        elif not target_package:
            target_package, _, parent_type = (
                group_parent_context(group) if group is not None else (None, None, None)
            )
            target_type = target_type or parent_type

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
                "target_package_name": target_package,
                "target_dependency_type": target_type,
                "parent_package_name": (
                    group.parent_package_name if group is not None else task.parent_package_name
                ),
                "parent_minimum_version": task.parent_minimum_version,
            }
        )
        updated[task_id] = diagnostics
        group = (group_by_id or {}).get(task.parent_group_id)
        if action == "retry_update":
            instruction = _build_high_level_retry_instruction(
                task.model_copy(
                    update={
                        "strategy_stage": effective_stage,
                        "target_package_name": target_package,
                        "target_dependency_type": target_type,
                        "selected_version": selected,
                    }
                ),
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
            target_package_name=target_package,
            target_dependency_type=target_type,
            parent_minimum_version=task.parent_minimum_version,
            action=action,
            exact_instruction=instruction,
        )
    return updated, plans


def _planner_plan_violations(
    plans: dict[str, SupervisorRetryPlan],
    task_queue: dict[str, RemediationTask],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> list[str]:
    """Validate retry-plan semantics before a plan can mutate routing state.

    Free-form plan evidence is intentionally not treated as an authority. These
    checks enforce the small set of invariants that must hold for an exact
    worker instruction to be safe.  Returning human-readable violations also
    makes the corrective replan visible in LangSmith through the supervisor's
    accumulated ``errors`` field.
    """
    violations: list[str] = []
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
            version.strip().lstrip("vV").lower() for version in plan.attempted_versions if version
        }
        diagnostics = diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in diagnostics.attempted_versions
                if version
            )

        selected = (
            plan.selected_version.strip().lstrip("vV").lower() if plan.selected_version else None
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
            violations.append(f"task {task_id}: exhausted update path cannot retry update")
        if (
            plan.action == "retry_update"
            and plan.strategy_stage == SCARemediationStage.CODE_WORKAROUND
        ):
            violations.append(f"task {task_id}: retry_update cannot use code_workaround stage")
        if (
            plan.action == "retry_update"
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage] < _SCA_STAGE_ORDER[task.strategy_stage]
        ):
            violations.append(
                f"task {task_id}: retry plan stage {plan.strategy_stage.value} regresses "
                f"from committed stage {task.strategy_stage.value}"
            )
        if (
            plan.action == "pivot_workaround"
            and plan.strategy_stage != SCARemediationStage.NPM_LATEST
        ):
            violations.append(f"task {task_id}: workaround pivot must be committed at npm_latest")
        if plan.action == "pivot_workaround" and selected is not None:
            violations.append(
                f"task {task_id}: workaround pivot cannot retain selected version {plan.selected_version}"
            )
    return violations


def _reconcile_registry_plan_evidence(
    plans: dict[str, SupervisorRetryPlan],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    tool_events: list[ToolEvent],
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Reconcile legacy retry-plan evidence with registry-tool results.

    Production retry planning is deterministic and does not call this helper.
    It remains available to direct callers that still replay historical
    scratchpad/tool-event evidence; registry output remains authoritative for
    any version retained by that compatibility path.
    """
    if not tool_events:
        return diagnostics_by_task, plans

    updated = dict(diagnostics_by_task)
    reconciled: dict[str, SupervisorRetryPlan] = {}
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        group = group_by_id.get(task.parent_group_id) if task else None
        transitive = bool(group and is_transitive_group(group))
        parent_name, _, _ = group_parent_context(group) if group is not None else (None, None, None)
        package_name = (
            parent_name
            if transitive
            and parent_name
            and plan.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE
            else (task.target_package_name if task else None)
            or (group.vulnerable_component if group else None)
        )
        if not package_name:
            reconciled[task_id] = plan
            continue

        expected_tool = "plan_npm_parent_version" if transitive else "plan_npm_version"
        matching_events = [
            event
            for event in tool_events
            if event.name == expected_tool
            and (
                str(event.args.get("parent_package_name", "")).strip() == package_name
                if transitive
                else str(event.args.get("package_name", "")).strip() == package_name
            )
            and event.content.startswith(
                "# NPM Parent Version Plan:" if transitive else "# NPM Version Plan:"
            )
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
        minimum_events = [
            event
            for event in matching_events
            if str(event.args.get("selection", "")).strip().lower() == "minimum"
        ]
        selected_event = (
            latest_events[-1]
            if plan.strategy_stage == SCARemediationStage.NPM_LATEST and latest_events
            else minimum_events[-1]
            if plan.strategy_stage == SCARemediationStage.OSV_MINIMUM and minimum_events
            else same_major_events[-1]
            if same_major_events
            else matching_events[-1]
        )
        if (
            selected_event in same_major_events
            and latest_events
            and (
                "same-major stage: skipped" in selected_event.content.lower()
                or _registry_selected_version(selected_event.content) is None
            )
        ):
            selected_event = latest_events[-1]

        content = selected_event.content
        selected_match = re.search(
            r"^-\s*Selected Version:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE
        )
        latest_match = re.search(
            r"^-\s*(?:Latest Stable|Latest Compatible):\s*(\S+)",
            content,
            re.IGNORECASE | re.MULTILINE,
        )
        eligible_match = re.search(
            r"^-\s*Eligible Candidates:\s*(.*)$", content, re.IGNORECASE | re.MULTILINE
        )
        selected_token = selected_match.group(1).strip() if selected_match else "NONE"
        selected = None if selected_token.upper() == "NONE" else selected_token.lstrip("vV")
        latest_seen = (
            latest_match.group(1).strip().lstrip("vV") if latest_match else plan.latest_version_seen
        )
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
            version.strip().lstrip("vV") for version in prior.attempted_versions if version
        }
        candidates = eligible or list(plan.candidate_versions_considered)
        if selected and selected not in candidates:
            candidates.insert(0, selected)
        if latest_seen and latest_seen not in candidates:
            candidates.append(latest_seen)

        effective_stage = plan.strategy_stage
        selected_selection = str(selected_event.args.get("selection", "")).strip().lower()
        if selected_selection == "minimum":
            effective_stage = SCARemediationStage.OSV_MINIMUM
        elif selected_selection == "same_major":
            effective_stage = SCARemediationStage.NPM_SAME_MAJOR
        elif selected_selection == "latest":
            effective_stage = SCARemediationStage.NPM_LATEST
        if selected_selection != "minimum" and (
            selected_event in latest_events
            or "same-major stage: skipped" in content.lower()
            or (
                re.search(r"^-\s*Same-Major Latest:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE)
                and latest_seen
                and re.search(
                    r"^-\s*Same-Major Latest:\s*(\S+)", content, re.IGNORECASE | re.MULTILINE
                )
                .group(1)
                .lstrip("vV")
                == latest_seen
            )
        ):
            effective_stage = SCARemediationStage.NPM_LATEST

        unattempted = [version for version in candidates if version not in attempted]
        exhausted = (
            effective_stage == SCARemediationStage.NPM_LATEST
            and selected is None
            and not unattempted
        )
        target_package = task.target_package_name if task else package_name
        target_type = task.target_dependency_type if task else None
        if transitive and effective_stage == SCARemediationStage.PACKAGE_OVERRIDE:
            target_package = group.vulnerable_component if group else package_name
            target_type = _override_dependency_type(group)
        if transitive and effective_stage == SCARemediationStage.NPM_LATEST and exhausted:
            selected = group.fix_plan.fixed_version if group and group.fix_plan else None
            effective_stage = SCARemediationStage.PACKAGE_OVERRIDE
            target_package = group.vulnerable_component if group else package_name
            target_type = _override_dependency_type(group)
            exhausted = False
        action = "pivot_workaround" if exhausted else "retry_update"
        diagnostics = prior.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": selected,
                "candidate_versions_considered": candidates,
                "latest_version_seen": latest_seen,
                "registry_query_performed": True,
                "exhausted_update_path": exhausted,
                "target_package_name": target_package,
                "target_dependency_type": target_type,
                "parent_package_name": parent_name if transitive else prior.parent_package_name,
            }
        )
        updated[task_id] = diagnostics
        if action == "retry_update":
            instruction = _build_high_level_retry_instruction(
                task.model_copy(
                    update={
                        "strategy_stage": effective_stage,
                        "target_package_name": target_package,
                        "target_dependency_type": target_type,
                        "selected_version": selected,
                    }
                ),
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
                "target_package_name": target_package,
                "target_dependency_type": target_type,
            }
        )
    return updated, reconciled


def _repair_invalid_planner_plans(
    plans: dict[str, SupervisorRetryPlan],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    violations: list[str] | None = None,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Apply a deterministic, fail-closed repair after corrective replanning.

    A valid unattempted candidate already present in planner evidence is safe to
    commit.  If no such candidate exists at the latest stage, the only safe
    action is the existing workaround pivot.  In particular, this function
    never preserves an invalid selected version merely to keep the graph
    moving.
    """
    repaired_diagnostics = dict(diagnostics_by_task)
    repaired_plans: dict[str, SupervisorRetryPlan] = {}
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
            version.strip().lstrip("vV").lower() for version in plan.attempted_versions if version
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
            # At npm_latest, only the registry's declared latest version is
            # safe to repair into a retry. Older candidates are already
            # covered by the bounded update stages and selecting one here can
            # produce a null/stale dispatch state after the guardrail clears
            # the invalid plan. If latest was attempted, fail closed into the
            # deterministic workaround pivot.
            repair_candidates = (
                [plan.latest_version_seen]
                if plan.strategy_stage == SCARemediationStage.NPM_LATEST
                else [plan.latest_version_seen, *plan.candidate_versions_considered]
            )
            for version in repair_candidates:
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
            group = group_by_id.get(task_queue[task_id].parent_group_id)
            target_package = task_queue[task_id].target_package_name or (
                group_parent_context(group)[0] if group is not None else None
            )
            target_type = task_queue[task_id].target_dependency_type
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": list(
                        dict.fromkeys([*diagnostics.candidate_versions_considered, candidate])
                    ),
                    "registry_query_performed": True,
                    "exhausted_update_path": False,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
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
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            continue

        group = group_by_id.get(task_queue[task_id].parent_group_id)
        if group is not None and is_transitive_group(group) and group.fix_plan:
            # Parent registry exhaustion is the deterministic handoff to the
            # native child override stage, not yet a code-workaround pivot.
            child_version = group.fix_plan.fixed_version
            target_type = _override_dependency_type(group)
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                    "selected_version": child_version,
                    "target_package_name": group.vulnerable_component,
                    "target_dependency_type": target_type,
                    "exhausted_update_path": False,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
            override_task = task_queue[task_id].model_copy(
                update={
                    "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                    "selected_version": child_version,
                    "target_package_name": group.vulnerable_component,
                    "target_dependency_type": target_type,
                }
            )
            instruction = _build_high_level_retry_instruction(
                override_task,
                group,
                None,
                diagnostics,
            )
            repaired_plans[task_id] = plan.model_copy(
                update={
                    "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                    "selected_version": child_version,
                    "exhausted_update_path": False,
                    "action": "retry_update",
                    "exact_instruction": instruction,
                    "target_package_name": group.vulnerable_component,
                    "target_dependency_type": target_type,
                }
            )
            continue

        # No direct unattempted candidate can be proven. Clear stale selection
        # and pivot at the terminal update stage so no guessed/old version is
        # sent to the dumb update worker.
        effective_stage = SCARemediationStage.NPM_LATEST
        if diagnostics is None:
            diagnostics = UpdateRetryDiagnostics(task_id=task_id)
        diagnostics = diagnostics.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
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
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
            }
        )
    return repaired_diagnostics, repaired_plans


def _parse_planner_selected_versions(
    scratchpad: str,
    task_queue: dict[str, RemediationTask],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> dict[str, UpdateRetryDiagnostics]:
    """Backward-compatible wrapper returning reconciled diagnostics only."""
    diagnostics, _ = _parse_planner_retry_plans(
        scratchpad,
        task_queue,
        diagnostics_by_task,
    )
    return diagnostics


def _update_worker_task_ids(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    preferred_ids: list[str] | None = None,
    limit: int | None = UPDATE_DISPATCH_LIMIT,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    task_ids = _dispatchable_task_ids_for_status(
        task_queue,
        set(_WORKABLE_STATUSES),
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


def _normalize_target_task_ids_for_node(
    next_node: str,
    target_task_ids: list[str],
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> list[str]:
    """Clamp returned active targets to the lifecycle state accepted by next_node."""
    retry_diagnostics_by_task = retry_diagnostics_by_task or {}
    if next_node == "qa_critic":
        return _qa_ready_task_ids(
            task_queue,
            preferred_ids=target_task_ids,
            group_by_id=group_by_id,
            limit=QA_DISPATCH_LIMIT,
        )
    if next_node == "update_subagent":
        return _update_worker_task_ids(
            task_queue,
            retry_diagnostics_by_task,
            preferred_ids=target_task_ids,
            limit=UPDATE_DISPATCH_LIMIT,
            group_by_id=group_by_id,
        )
    if next_node == "workaround_subagent":
        return _dispatchable_task_ids_for_status(
            task_queue,
            set(_WORKABLE_STATUSES),
            preferred_ids=target_task_ids,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            limit=1,
            group_by_id=group_by_id,
        )
    return []


def _resolve_task_id_from_identifier(
    identifier: str,
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> str | None:
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
        task.task_id for task in task_queue.values() if task.parent_group_id == identifier
    ]
    if len(all_matches) == 1:
        return all_matches[0]

    return None


def _normalize_qa_evaluations_for_tasks(
    qa_evaluations: dict[str, QAEvaluation],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, QAEvaluation]:
    """Re-key QA evaluations to concrete task_ids when possible."""
    normalized: dict[str, QAEvaluation] = {}
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
            failure_evidence=evaluation.failure_evidence,
            scan_evidence=evaluation.scan_evidence,
            deterministic_gates=evaluation.deterministic_gates,
            semantic_security_review=evaluation.semantic_security_review,
            test_attribution=evaluation.test_attribution,
            contract_error=evaluation.contract_error,
            contract_error_reason=evaluation.contract_error_reason,
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


def _latest_action_summary_by_task(
    action_summaries: list[AgentActionSummary],
    task_queue: dict[str, RemediationTask],
    active_target_task_ids: list[str],
) -> dict[str, AgentActionSummary]:
    """Return the most recent action summary keyed by resolved task_id."""
    latest: dict[str, AgentActionSummary] = {}
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


def _no_fix_failure_transition(
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
    updates = advance_no_fix_stage(task)
    next_stage = updates.get("no_fix_stage")
    if next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        updates["qa_policy"] = QAPolicy.NO_FIX_CODE_REMOVAL
    reset_workspace = next_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    if reset_workspace:
        retry_task = task.model_copy(update=updates)
        updates["instruction"] = build_no_fix_retry_instruction(
            retry_task,
            group,
            evaluation=evaluation,
            failure_feedback=failure_feedback,
        )
    return updates, reset_workspace


def _reset_no_fix_replay_plan(
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


def _build_high_level_retry_instruction(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    evaluation: QAEvaluation | None,
    diagnostics: UpdateRetryDiagnostics | None,
) -> str:
    """Synthesize a high-level retry instruction for the update worker."""
    component = group.vulnerable_component if group else task.parent_group_id
    parent_name, _, parent_type = (
        group_parent_context(group) if group is not None else (None, None, None)
    )
    target = task.target_package_name
    if not target and task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        target = parent_name
    target = target or component
    dependency_type = task.target_dependency_type or parent_type
    if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        dependency_type = dependency_type or "overrides"
    category = evaluation.failure_category if evaluation else None
    if diagnostics and diagnostics.selected_version:
        manifest = group.file_paths[0] if group and group.file_paths else "package.json"
        is_override = (
            task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
            or diagnostics.used_overrides
            or dependency_type in {"overrides", "resolutions", "pnpm_overrides"}
        )
        dependency_action = (
            f"package-manager override for {component}"
            if is_override
            else f"{target} dependency version"
        )
        target_clause = (
            f"edit only the {target} declaration" if target != component else f"update {target}"
        )
        if dependency_type:
            target_clause += f" in {dependency_type}"
        return (
            f"Apply the supervisor-selected {dependency_action} for {component}; "
            f"during strategy stage {task.strategy_stage.value}: "
            f"{target_clause} in {manifest} to exact version {diagnostics.selected_version}; "
            "do not edit any other dependency target; "
            "after all requested manifest edits, run the single final manifest synchronization validation."
        )
    if task.strategy_stage == SCARemediationStage.OSV_MINIMUM and group and group.fix_plan:
        floor = group.fix_plan.fixed_version
        if floor:
            manifest = group.file_paths[0] if group.file_paths else "package.json"
            if parent_name and target == parent_name:
                return (
                    f"Apply strategy stage {task.strategy_stage.value} for transitive package {component}: "
                    f"update only directly declared parent {parent_name} in {manifest} to the "
                    "supervisor-selected compatible parent version; do not use a child override; "
                    "after all requested manifest edits, run the single final manifest synchronization validation."
                )
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
            manifest = group.file_paths[0] if group and group.file_paths else "package.json"
            return (
                f"Apply strategy stage {task.strategy_stage.value} for {component}: "
                f"update only {target} in {manifest} to exact version {candidate}; "
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


def _registry_selected_version(report: str) -> str | None:
    """Extract a planner-selected stable version from a registry report."""
    match = re.search(
        r"^-\s*Selected Version:\s*(\S+)",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match or match.group(1).upper() == "NONE":
        return None
    return match.group(1).strip().lstrip("vV")


def _override_dependency_type(group: VulnerabilityGroup | None) -> str:
    """Return the package-manager-native override field for an SCA group."""
    managers = {
        (issue.package_manager or "").strip().lower()
        for issue in (group.localized_issues if group else [])
    }
    if "yarn" in managers:
        return "resolutions"
    if "pnpm" in managers:
        return "pnpm_overrides"
    return "overrides"


def _plan_initial_transitive_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> RemediationTask:
    """Select the first parent-first candidate before worker dispatch.

    The worker receives only the committed result of this function. Registry
    failures or an empty candidate set advance deterministically to the next
    parent stage, and only a fully exhausted parent path commits a child
    package-manager override.
    """
    if (
        task.strategy != RoutingStrategy.VERSION_BUMP
        or task.status != TaskStatus.PENDING
        or task.parent_package_name is None
        or task.strategy_stage != SCARemediationStage.OSV_MINIMUM
    ):
        return task
    child_fixed_version = group.fix_plan.fixed_version if group.fix_plan else None
    installed_parent_version = task.parent_package_version
    parent_name, _, parent_type = group_parent_context(group)
    if not child_fixed_version or not installed_parent_version or not parent_name:
        stage = SCARemediationStage.PACKAGE_OVERRIDE
        target_type = _override_dependency_type(group)
        override_task = task.model_copy(
            update={
                "strategy_stage": stage,
                "target_package_name": group.vulnerable_component,
                "target_dependency_type": target_type,
                "selected_version": child_fixed_version,
                "instruction": (
                    f"Apply package-manager override stage for {group.vulnerable_component}: "
                    f"pin the vulnerable child to exact version {child_fixed_version or 'the OSV-fixed version'} "
                    f"using {target_type}; do not edit the parent declaration."
                ),
            }
        )
        return override_task

    attempted: set[str] = set()
    for selection, stage in (
        ("minimum", SCARemediationStage.OSV_MINIMUM),
        ("same_major", SCARemediationStage.NPM_SAME_MAJOR),
        ("latest", SCARemediationStage.NPM_LATEST),
    ):
        try:
            report = plan_npm_parent_version.invoke(
                {
                    "parent_package_name": parent_name,
                    "child_package_name": group.vulnerable_component,
                    "child_fixed_version": child_fixed_version,
                    "installed_parent_version": installed_parent_version,
                    "selection": selection,
                    "attempted_versions": ",".join(sorted(attempted)),
                    "dependency_ancestry": ",".join(group.dependency_ancestry),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor: initial parent registry planning failed for %s (%s)",
                parent_name,
                exc,
            )
            report = ""
        selected = _registry_selected_version(report)
        if not selected:
            continue
        attempted.add(selected)
        target_task = task.model_copy(
            update={
                "strategy_stage": stage,
                "target_package_name": parent_name,
                "target_dependency_type": task.target_dependency_type or parent_type,
                "selected_version": selected,
                "parent_minimum_version": (
                    selected
                    if stage == SCARemediationStage.OSV_MINIMUM
                    else task.parent_minimum_version
                ),
            }
        )
        diagnostics = UpdateRetryDiagnostics(
            task_id=task.task_id,
            strategy_stage=stage,
            security_floor=child_fixed_version,
            selected_version=selected,
            target_package_name=parent_name,
            target_dependency_type=target_task.target_dependency_type,
            parent_package_name=parent_name,
            parent_minimum_version=target_task.parent_minimum_version,
            registry_query_performed=True,
            candidate_versions_considered=[selected],
        )
        return target_task.model_copy(
            update={
                "instruction": _build_high_level_retry_instruction(
                    target_task,
                    group,
                    None,
                    diagnostics,
                )
            }
        )

    target_type = _override_dependency_type(group)
    return task.model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "target_package_name": group.vulnerable_component,
            "target_dependency_type": target_type,
            "selected_version": child_fixed_version,
            "instruction": (
                f"Apply package-manager override stage for {group.vulnerable_component}: "
                f"pin the vulnerable child to exact version {child_fixed_version} using {target_type}; "
                "do not edit the parent declaration."
            ),
        }
    )


def _terminalize_pivot_parents(
    task_queue: dict[str, RemediationTask],
    parent_ids: list[str],
    strategy_by_parent: dict[str, RoutingStrategy],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    retry_plans_by_task: dict[str, SupervisorRetryPlan] | None = None,
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
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
        updates: dict[str, Any] = {}
        if parent_task.status not in _TERMINAL_STATUSES:
            updates["status"] = terminal_status

        # Parent/child pivots are one atomic state transition.  A version-bump
        # parent remains the audit record for the exhausted update path, while
        # the newly materialized child owns workaround execution.  Do not
        # leave the parent with a code-workaround stage and an old exact
        # dependency instruction; that contradictory combination is what made
        # a stale routing trace authorize a stale update.
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
            # A migration can legitimately close a NEEDS_RETRY parent as
            # QA_PASSED after the child has cleared the target.  PIVOTED is
            # already a normal transition and needs no exception.
            allow_breaking_change_pivot=(terminal_status == TaskStatus.QA_PASSED),
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
                        "instruction_digest": _instruction_digest(committed_parent.instruction),
                    }
                )


def _reconcile_terminal_pivot_parents(
    task_queue: dict[str, RemediationTask],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    group_by_id: dict[str, VulnerabilityGroup],
) -> list[str]:
    """Terminalize non-terminal update parents whose workaround child is terminal.

    Args:
        task_queue: Copy-on-write task queue to mutate.
        qa_evaluations: QA evidence used to classify the parent transition.
        retry_diagnostics_by_task: Retry diagnostics to keep aligned with the
            parent transition.
        retry_plans_by_task: Retry plans to clear for terminalized parents.
        group_by_id: Vulnerability groups used to rebuild parent instructions.

    Returns:
        Task IDs of parents terminalized during this reconciliation.

    Mutations:
        Closes any live parent attempt, clears dispatch-only fields, and
        removes retry plans through _terminalize_pivot_parents.
    """
    parent_ids: list[str] = []
    for child in task_queue.values():
        if (
            child.parent_task_id is None
            or child.strategy != RoutingStrategy.CODE_WORKAROUND
            or child.status not in _TERMINAL_STATUSES
        ):
            continue
        parent = task_queue.get(child.parent_task_id)
        if (
            parent is None
            or parent.status in _TERMINAL_STATUSES
            or parent.strategy != RoutingStrategy.VERSION_BUMP
        ):
            continue
        parent_ids.append(parent.task_id)

    parent_ids = sorted(set(parent_ids))
    if not parent_ids:
        return []

    _terminalize_pivot_parents(
        task_queue,
        parent_ids,
        {parent_id: RoutingStrategy.CODE_WORKAROUND for parent_id in parent_ids},
        qa_evaluations,
        retry_diagnostics_by_task=retry_diagnostics_by_task,
        retry_plans_by_task=retry_plans_by_task,
        group_by_id=group_by_id,
    )
    return parent_ids


# ---------------------------------------------------------------------------
# Deterministic fallback router
# ---------------------------------------------------------------------------


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
        evaluation = qa_evaluations.get(target.task_id) or qa_evaluations.get(
            target.parent_group_id
        )
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
            eval_ = qa_evaluations.get(task.task_id) or qa_evaluations.get(task.parent_group_id)
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
            evaluation = qa_evaluations.get(task.task_id) or qa_evaluations.get(
                task.parent_group_id
            )
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
            eval_ = qa_evaluations.get(t.task_id) or qa_evaluations.get(t.parent_group_id)
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
        eval_ = qa_evaluations.get(target.task_id) or qa_evaluations.get(target.parent_group_id)
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

    if state.get("status") == "qa_completed":
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
                committed, _snapshot = _create_attempt_snapshot(
                    task,
                    dispatch_node=decision.next_node,
                    snapshots_by_id=projected_snapshots,
                    state_revision=state_revision,
                    plan_id=(
                        retry_plans_by_task[task_id].plan_id
                        if task_id in retry_plans_by_task
                        else None
                    ),
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


# ---------------------------------------------------------------------------
# Planner phase helpers
# ---------------------------------------------------------------------------


def _registry_report_value(report: str, label: str) -> str | None:
    """Extract one normalized value from a deterministic registry-tool report."""
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(\S+)",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match or match.group(1).upper() == "NONE":
        return None
    return match.group(1).strip().lstrip("vV")


def _registry_report_versions(report: str, label: str) -> list[str]:
    """Extract a comma-separated version list from a registry-tool report."""
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(.*)$",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    return list(
        dict.fromkeys(
            version.strip().lstrip("vV")
            for version in match.group(1).split(",")
            if version.strip() and version.strip().upper() != "NONE"
        )
    )


def _build_deterministic_retry_plan(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics,
    group: VulnerabilityGroup | None,
    *,
    requested_stage: SCARemediationStage | None = None,
) -> SupervisorRetryPlan:
    """Build an exact retry plan from committed state and registry facts.

    The task's committed stage is authoritative for this pass. A later
    Supervisor pass may advance the stage after QA evidence; this function
    never skips an empty stage, regresses, reuses an attempted version, or
    turns an exhausted update path back into an update retry.
    """
    requested = requested_stage or task.strategy_stage
    requested_order = _SCA_STAGE_ORDER.get(requested, 99)
    current_order = _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
    effective_stage = requested if requested_order >= current_order else task.strategy_stage
    attempted = set(diagnostics.attempted_versions)
    security_floor = group.fix_plan.fixed_version if group and group.fix_plan else None
    transitive = bool(group and is_transitive_group(group))
    candidate_versions: list[str] = []
    latest_version_seen: str | None = None
    selected_version: str | None = None
    failure_reason = ""
    parent_minimum_version = task.parent_minimum_version
    target_package_name = task.target_package_name
    target_dependency_type = task.target_dependency_type

    if effective_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        selected_version = security_floor
        if group is not None:
            target_package_name = group.vulnerable_component
            target_dependency_type = _override_dependency_type(group)
    elif effective_stage != SCARemediationStage.CODE_WORKAROUND and security_floor:
        # The QA transition already advances the committed stage one step.
        # Plan only that stage here; an empty candidate set must not silently
        # skip ahead to a later stage in the same Supervisor pass.
        stages = [effective_stage]
        if transitive:
            parent_name, parent_version, parent_type = group_parent_context(group)
            target_package_name = target_package_name or parent_name
            target_dependency_type = target_dependency_type or parent_type
            if not parent_name or not parent_version or not group.vulnerable_component:
                failure_reason = "Missing parent context for deterministic transitive planning."
            else:
                for stage in stages:
                    selection = {
                        SCARemediationStage.OSV_MINIMUM: "minimum",
                        SCARemediationStage.NPM_SAME_MAJOR: "same_major",
                        SCARemediationStage.NPM_LATEST: "latest",
                    }[stage]
                    try:
                        report = plan_npm_parent_version.invoke(
                            {
                                "parent_package_name": parent_name,
                                "child_package_name": group.vulnerable_component,
                                "child_fixed_version": security_floor,
                                "installed_parent_version": parent_version,
                                "selection": selection,
                                "attempted_versions": ",".join(sorted(attempted)),
                                "dependency_ancestry": ",".join(group.dependency_ancestry),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        failure_reason = f"Deterministic parent registry planning failed: {exc}"
                        continue
                    report_candidates = _registry_report_versions(
                        report, "Eligible Candidates"
                    ) or _registry_report_versions(report, "Compatible Parent Versions")
                    candidate_versions = list(
                        dict.fromkeys([*candidate_versions, *report_candidates])
                    )
                    latest_version_seen = (
                        _registry_report_value(report, "Latest Compatible")
                        or _registry_report_value(report, "Latest Stable")
                        or latest_version_seen
                    )
                    selected = _registry_selected_version(report)
                    if selected:
                        selected_version = selected
                        effective_stage = stage
                        if stage == SCARemediationStage.OSV_MINIMUM:
                            parent_minimum_version = selected
                        break
        else:
            try:
                candidates = fetch_registry_candidates(
                    group.vulnerable_component or "",
                    security_floor,
                    attempted,
                )
                candidate_versions = [candidate.version for candidate in candidates[:30]]
                latest_version_seen = candidates[-1].version if candidates else None
                for stage in stages:
                    selected = select_version(candidates, stage, attempted)
                    if selected:
                        selected_version = selected
                        effective_stage = stage
                        break
            except Exception as exc:  # noqa: BLE001
                failure_reason = f"Deterministic registry planning failed: {exc}"
    elif effective_stage != SCARemediationStage.CODE_WORKAROUND:
        failure_reason = "No security floor is available for deterministic version selection."

    if selected_version is None and effective_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        if transitive and security_floor and effective_stage == SCARemediationStage.NPM_LATEST:
            effective_stage = SCARemediationStage.PACKAGE_OVERRIDE
            selected_version = security_floor
            target_package_name = group.vulnerable_component if group else task.parent_group_id
            target_dependency_type = _override_dependency_type(group)
            failure_reason = "Parent update stages are exhausted; entering package override."
        elif effective_stage == SCARemediationStage.NPM_LATEST:
            effective_stage = SCARemediationStage.NPM_LATEST
            target_package_name = target_package_name or task.target_package_name

    exhausted = selected_version is None and effective_stage == SCARemediationStage.NPM_LATEST
    effective_task = task.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected_version,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "parent_minimum_version": parent_minimum_version,
        }
    )
    safe_candidate_versions = candidate_versions[:30] if selected_version or exhausted else []
    safe_latest_version = latest_version_seen if selected_version or exhausted else None
    effective_diagnostics = diagnostics.model_copy(
        update={
            "strategy_stage": effective_stage,
            "security_floor": security_floor or diagnostics.security_floor,
            "selected_version": selected_version,
            "candidate_versions_considered": safe_candidate_versions,
            "latest_version_seen": safe_latest_version,
            "registry_query_performed": bool(security_floor),
            "exhausted_update_path": exhausted,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "parent_package_name": (
                group.parent_package_name if group is not None else diagnostics.parent_package_name
            ),
            "parent_minimum_version": parent_minimum_version,
            "failure_reason": failure_reason,
        }
    )
    if exhausted:
        component = group.vulnerable_component if group else task.parent_group_id
        instruction = (
            f"Implement a code workaround or isolation strategy for {component} "
            "because the deterministic registry policy found no remaining eligible update version."
        )
    else:
        instruction = _build_high_level_retry_instruction(
            effective_task,
            group,
            None,
            effective_diagnostics,
        )
    return SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=effective_stage,
        selected_version=selected_version,
        attempted_versions=list(diagnostics.attempted_versions),
        candidate_versions_considered=safe_candidate_versions,
        latest_version_seen=safe_latest_version,
        exhausted_update_path=exhausted,
        package_abandoned=diagnostics.package_abandoned,
        target_package_name=target_package_name,
        target_dependency_type=target_dependency_type,
        parent_minimum_version=parent_minimum_version,
        action="pivot_workaround" if exhausted else "retry_update",
        exact_instruction=instruction,
    )


def _needs_planner(
    task_queue: dict[str, RemediationTask],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    current_status: str,
) -> bool:
    """Return True when deterministic retry planning should run.

    Retry planning is reserved for VERSION_BUMP retry analysis and playbook selection.
    CODE_WORKAROUND tasks do not use the npm version planner.
    """
    version_bump_retries = [
        t
        for t in task_queue.values()
        if (
            t.status == TaskStatus.NEEDS_RETRY
            and t.strategy == RoutingStrategy.VERSION_BUMP
            and (
                t.strategy_stage
                in {
                    SCARemediationStage.OSV_MINIMUM,
                    SCARemediationStage.NPM_SAME_MAJOR,
                    SCARemediationStage.NPM_LATEST,
                }
                or (
                    t.strategy_stage == SCARemediationStage.CODE_WORKAROUND
                    and not t.parent_package_name
                    and t.target_dependency_type
                    not in {
                        "overrides",
                        "resolutions",
                        "pnpm_overrides",
                    }
                )
            )
        )
    ]
    if not version_bump_retries:
        return False
    return (
        current_status == "qa_completed" or bool(qa_evaluations) or bool(retry_diagnostics_by_task)
    )


def _run_deterministic_retry_planner(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Plan every actionable retry from task state and deterministic registry facts."""
    updated_diagnostics = dict(retry_diagnostics_by_task)
    plans: dict[str, SupervisorRetryPlan] = {}
    retry_tasks = sorted(
        (
            task
            for task in task_queue.values()
            if task.status == TaskStatus.NEEDS_RETRY
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.strategy_stage
            in {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
        ),
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    for task in retry_tasks:
        diagnostics = updated_diagnostics.get(
            task.task_id,
            UpdateRetryDiagnostics(task_id=task.task_id, strategy_stage=task.strategy_stage),
        )
        # The deterministic router owns the exhausted-update pivot. Never
        # reopen a path that its guardrail has already proven exhausted.
        if _is_exhausted_update_pivot_candidate(task, diagnostics):
            continue
        group = group_by_id.get(task.parent_group_id)
        plan = _build_deterministic_retry_plan(task, diagnostics, group)
        # An empty stage is deterministic evidence to advance to the next
        # bounded stage, not a request for the worker to inspect the registry.
        # Keep the worker execution-only: every update dispatch must end with
        # an exact unattempted version, or with the terminal update pivot.
        while plan.selected_version is None and not plan.exhausted_update_path:
            transitive = bool(group and is_transitive_group(group))
            next_stage = _next_sca_stage(plan.strategy_stage, transitive=transitive)
            if next_stage == SCARemediationStage.CODE_WORKAROUND:
                break
            if _SCA_STAGE_ORDER.get(next_stage, 99) <= _SCA_STAGE_ORDER.get(
                plan.strategy_stage, 99
            ):
                break
            plan = _build_deterministic_retry_plan(
                task,
                diagnostics,
                group,
                requested_stage=next_stage,
            )

        if plan.selected_version is None and not plan.exhausted_update_path:
            # This is a malformed or incomplete registry state (for example a
            # package-override stage without a security floor). Never preserve
            # it as retry_update: commit the same fail-closed latest-stage
            # pivot used by the deterministic guardrail.
            component = group.vulnerable_component if group else task.parent_group_id
            plan = plan.model_copy(
                update={
                    "strategy_stage": SCARemediationStage.NPM_LATEST,
                    "selected_version": None,
                    "exhausted_update_path": True,
                    "action": "pivot_workaround",
                    "exact_instruction": (
                        f"Implement a code workaround or isolation strategy for {component} "
                        "because the manifest-based update path is exhausted."
                    ),
                }
            )
        plans[task.task_id] = plan
        updated_diagnostics[task.task_id] = diagnostics.model_copy(
            update={
                "strategy_stage": plan.strategy_stage,
                "security_floor": diagnostics.security_floor
                or (
                    group_by_id[task.parent_group_id].fix_plan.fixed_version
                    if task.parent_group_id in group_by_id
                    and group_by_id[task.parent_group_id].fix_plan is not None
                    else None
                ),
                "selected_version": plan.selected_version,
                "candidate_versions_considered": plan.candidate_versions_considered,
                "latest_version_seen": plan.latest_version_seen,
                "registry_query_performed": bool(
                    group_by_id.get(task.parent_group_id)
                    and group_by_id[task.parent_group_id].fix_plan
                    and group_by_id[task.parent_group_id].fix_plan.fixed_version
                ),
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name,
                "target_dependency_type": plan.target_dependency_type,
                "parent_minimum_version": plan.parent_minimum_version,
            }
        )
    return updated_diagnostics, plans


def _commit_retry_plans(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    plans: dict[str, SupervisorRetryPlan],
) -> None:
    """Commit deterministic retry inputs before the next worker dispatch."""
    for task_id, plan in sorted(plans.items()):
        task = task_queue.get(task_id)
        if task is None:
            continue
        committed_task = _commit_task_transition(
            task_queue,
            task_id,
            updates={
                "strategy_stage": plan.strategy_stage,
                "instruction": plan.exact_instruction,
                "selected_version": plan.selected_version,
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name or task.target_package_name,
                "target_dependency_type": plan.target_dependency_type
                or task.target_dependency_type,
                "parent_minimum_version": plan.parent_minimum_version
                or task.parent_minimum_version,
            },
            close_attempt=True,
            clear_selected_version=plan.selected_version is None,
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
                    "target_package_name": committed_task.target_package_name,
                    "target_dependency_type": committed_task.target_dependency_type,
                    "parent_minimum_version": committed_task.parent_minimum_version,
                    "instruction_digest": _instruction_digest(committed_task.instruction),
                }
            )
        retry_plans_by_task[task_id] = plan.model_copy(
            update={"source_task_revision": committed_task.task_revision}
        )


# Spawn request materializer
# ---------------------------------------------------------------------------


def _materialize_spawn_requests(
    spawn_requests: list[TaskSpawnRequest],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    errors: list[str],
    valid_groups: list[VulnerabilityGroup] | None = None,
    qa_evaluations: dict[str, QAEvaluation] | None = None,
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] | None = None,
    consistency_events: list[StateConsistencyEvent] | None = None,
) -> tuple[dict[str, RemediationTask], dict[str, list[str]]]:
    """Validate and materialize spawn requests into new RemediationTask objects.

    Returns a dict of new task_id â†’ RemediationTask to be merged into task_queue.
    Rejected requests are logged to errors.  A repeated parent/strategy pivot
    reuses its existing child and is not treated as an error.
    """
    next_index = len(task_queue) + 1
    new_tasks: dict[str, RemediationTask] = {}
    child_ids_by_parent: dict[str, list[str]] = {}
    current_queue_size = len(task_queue)

    ordered_spawn_requests = sorted(
        spawn_requests,
        key=lambda request: (request.parent_task_id, request.strategy.value),
    )
    for req in ordered_spawn_requests:
        # Guard: unknown parent task
        if req.parent_task_id not in task_queue:
            errors.append(
                f"supervisor: spawn rejected â€” parent task '{req.parent_task_id}' not in queue."
            )
            continue

        parent_task = task_queue[req.parent_task_id]

        # Guard: CODE_WORKAROUND tasks must not spawn CODE_WORKAROUND children.
        # Workaround tasks are terminal remediation strategies â€” exhausted
        # workarounds should be marked UNFIXABLE, not recursively respawned.
        if (
            parent_task.strategy == RoutingStrategy.CODE_WORKAROUND
            and req.strategy == RoutingStrategy.CODE_WORKAROUND
        ):
            errors.append(
                f"supervisor: spawn rejected â€” CODE_WORKAROUND parent '{req.parent_task_id}' "
                f"cannot spawn another CODE_WORKAROUND child. Workaround tasks are terminal "
                f"remediation strategies."
            )
            continue

        # Spawn requests can be replayed when the supervisor revisits the same
        # exhausted parent, and an untrusted decision can also contain duplicate
        # requests in one envelope.  A workaround pivot is idempotent: one
        # parent remediation task may own at most one child for a given
        # strategy.  Check both the committed queue and children materialized
        # earlier in this call so neither replay can create sibling tasks.
        existing_child = next(
            (
                candidate
                for candidate in (*task_queue.values(), *new_tasks.values())
                if candidate.parent_task_id == req.parent_task_id
                and candidate.strategy == req.strategy
            ),
            None,
        )
        if existing_child is not None:
            existing_child_ids = child_ids_by_parent.setdefault(req.parent_task_id, [])
            if existing_child.task_id not in existing_child_ids:
                existing_child_ids.append(existing_child.task_id)
            logger.info(
                "supervisor: skipped duplicate child spawn for parent '%s'; "
                "reusing existing child '%s' (strategy=%s).",
                req.parent_task_id,
                existing_child.task_id,
                req.strategy.value,
            )
            continue

        # Guard: depth cap
        child_depth = parent_task.ancestry_depth + 1
        if child_depth > MAX_ANCESTRY_DEPTH:
            errors.append(
                f"supervisor: spawn rejected â€” parent '{req.parent_task_id}' at depth "
                f"{parent_task.ancestry_depth}, child would be depth {child_depth} "
                f"which exceeds MAX_ANCESTRY_DEPTH={MAX_ANCESTRY_DEPTH}."
            )
            continue

        # Guard: queue size cap
        if current_queue_size + len(new_tasks) + 1 > MAX_TASK_QUEUE_SIZE:
            errors.append(
                f"supervisor: spawn rejected â€” queue would exceed MAX_TASK_QUEUE_SIZE="
                f"{MAX_TASK_QUEUE_SIZE}. Rejected spawn for parent '{req.parent_task_id}'."
            )
            continue

        # Materialize child task
        child_task_id = f"task-{next_index}"
        next_index += 1

        child_policy: QAPolicy | None
        if req.strategy == RoutingStrategy.VERSION_BUMP:
            child_policy = QAPolicy.VERSION_BUMP
        elif (
            parent_task.strategy == RoutingStrategy.VERSION_BUMP
            and req.strategy == RoutingStrategy.CODE_WORKAROUND
        ):
            parent_eval = (qa_evaluations or {}).get(parent_task.task_id) or (
                qa_evaluations or {}
            ).get(parent_task.parent_group_id)
            diagnostics = (retry_diagnostics_by_task or {}).get(parent_task.task_id)
            if parent_eval is None and diagnostics is None:
                details = (
                    "A VERSION_BUMP-to-CODE_WORKAROUND pivot was requested without "
                    "authoritative parent QA provenance."
                )
                errors.append(f"supervisor: {details}")
                if consistency_events is not None:
                    consistency_events.append(
                        _build_consistency_event(
                            error_code="AMBIGUOUS_QA_POLICY_PIVOT",
                            task_id=parent_task.task_id,
                            expected_attempt_id=parent_task.current_attempt_id,
                            received_attempt_id=None,
                            action="ignored",
                            details=details,
                        )
                    )
                continue
            gates = parent_eval.deterministic_gates if parent_eval is not None else None
            scanner_cleared = bool(gates and gates.target_scanner_cleared is True)
            hard_test_failure = bool(gates and gates.tests_passed is False) or bool(
                parent_eval and parent_eval.failure_category == FailureCategory.BREAKING_CHANGE
            )
            child_policy = (
                QAPolicy.MIGRATION_CODE_WORKAROUND
                if scanner_cleared and hard_test_failure
                else QAPolicy.MITIGATION_CODE_WORKAROUND
            )
        else:
            child_policy = QAPolicy.INITIAL_CODE_WORKAROUND

        _TRIAGE_BUCKET_TO_STRATEGY: dict[str, str] = {
            "UPDATE_VERSION": RoutingStrategy.VERSION_BUMP.name,
            "WORKAROUND": RoutingStrategy.CODE_WORKAROUND.name,
            "NO_FIX": RoutingStrategy.CODE_WORKAROUND.name,
        }
        new_group_id = parent_task.parent_group_id
        if parent_task.strategy.name in new_group_id and req.strategy.name not in new_group_id:
            new_group_id = new_group_id.replace(parent_task.strategy.name, req.strategy.name)
        elif parent_task.strategy.value in new_group_id and req.strategy.value not in new_group_id:
            new_group_id = new_group_id.replace(parent_task.strategy.value, req.strategy.value)
        else:
            # Fallback: try triage-level strategy bucket tokens
            for bucket_token, mapped_strategy_name in _TRIAGE_BUCKET_TO_STRATEGY.items():
                if (
                    bucket_token in new_group_id
                    and mapped_strategy_name == parent_task.strategy.name
                    and req.strategy.name != parent_task.strategy.name
                ):
                    new_group_id = new_group_id.replace(
                        bucket_token,
                        req.strategy.name,
                    )
                    break

        if new_group_id != parent_task.parent_group_id and new_group_id not in group_by_id:
            parent_group = group_by_id.get(parent_task.parent_group_id)
            if parent_group:
                new_group = parent_group.model_copy(update={"group_id": new_group_id})
                group_by_id[new_group_id] = new_group
                if valid_groups is not None:
                    valid_groups.append(new_group)

        new_task = RemediationTask(
            task_id=child_task_id,
            parent_group_id=new_group_id,
            parent_task_id=req.parent_task_id,
            qa_policy=child_policy,
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
            "supervisor: spawned child task '%s' (parent='%s', depth=%d, strategy=%s) â€” %s",
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


def run_supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node â€” Supervisor commander for Phase 5 orchestration.

    Execution stages
    ----------------
    1. Normalize task_queue: create initial RemediationTask entries for any
       valid_groups not yet represented (copy-on-write via model_copy).
    2. Ingest subagent action summaries for current active_target_task_ids only.
    3. Ingest QA results for active task IDs only (when status == "qa_completed").
    4. Mark UNFIXABLE any task whose retry_count has reached MAX_RETRIES.
    5. Short-circuit: if an active task is optimistically_fixed â†’ qa_critic.
    6. If QA produced a parseable scan and set ``triage_required``, route to
       the post-QA triage node before any worker or teardown decision.
    7. Plan retry inputs from deterministic registry facts and committed state.
    8. Route through the deterministic state machine and validate its
       dispatch projection.
    9. Apply guarded: revised_instructions, strategy updates, status overrides,
       unfixable marks, new constraints, and materialized spawn requests.
    10. Return state patch.
    """
    valid_groups: list[VulnerabilityGroup] = list(state.get("valid_groups", []))
    if not valid_groups:
        if state.get("triage_required") and state.get("status") in {
            "qa_completed",
            "qa_failed",
            "final_scan_completed",
        }:
            decision = SupervisorDecision(
                decision_code=DecisionCode.TRIAGE_REQUIRED,
                next_node="triage",
                target_task_ids=[],
                instructions="Route the parseable post-remediation scan to triage.",
                decision_reason="Post-QA triage is required.",
            )
            return {
                "status": "supervisor_routed",
                "next_routing_step": "triage",
                "active_target_task_ids": [],
                "decision_code": decision.decision_code,
                "supervisor_audit": _emit_audit(
                    decision, [], int(state.get("state_revision", 0)) + 1
                ),
                "supervisor_instructions": "Route the parseable post-remediation scan to triage.",
            }
        logger.info("supervisor: no valid groups â€” routing to teardown.")
        decision = SupervisorDecision(
            decision_code=DecisionCode.NO_VALID_GROUPS,
            next_node="teardown",
            target_task_ids=[],
            instructions="No groups to process.",
            decision_reason="No valid vulnerability groups are available.",
        )
        return {
            "status": "supervisor_routed",
            "next_routing_step": "teardown",
            "active_target_task_ids": [],
            "decision_code": decision.decision_code,
            "supervisor_audit": _emit_audit(decision, [], int(state.get("state_revision", 0)) + 1),
            "supervisor_instructions": "No groups to process.",
        }

    group_by_id: dict[str, VulnerabilityGroup] = {g.group_id: g for g in valid_groups}
    existing_constraints: list[str] = list(state.get("constraints_ledger", []))
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = dict(
        state.get("retry_diagnostics_by_task", {})
    )
    retry_plans_by_task: dict[str, SupervisorRetryPlan] = dict(state.get("retry_plans_by_task", {}))
    workaround_replay_plans_by_task: dict[str, WorkaroundReplayPlan] = dict(
        state.get("workaround_replay_plans_by_task", {})
    )
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot] = dict(
        state.get("attempt_snapshots_by_id", {})
    )
    worker_results_by_attempt: dict[str, WorkerAttemptResult] = dict(
        state.get("worker_results_by_attempt", {})
    )
    qa_results_by_attempt: dict[str, QAAttemptResult] = dict(state.get("qa_results_by_attempt", {}))
    processed_worker_attempt_ids: set[str] = set(state.get("processed_worker_attempt_ids", []))
    processed_qa_attempt_ids: set[str] = set(state.get("processed_qa_attempt_ids", []))
    prior_consistency_events: list[StateConsistencyEvent] = list(
        state.get("consistency_events", [])
    )
    consistency_events: list[StateConsistencyEvent] = []
    state_revision = int(state.get("state_revision", 0)) + 1
    # ``errors`` is an additive LangGraph reducer. A node must return only
    # errors discovered during this invocation; replaying the prior list here
    # is what caused identical planner errors to multiply across supervisor
    # loops.
    errors: list[str] = []
    prior_error_messages = set(state.get("errors", []) or [])

    # ------------------------------------------------------------------
    # 1. Normalize task_queue (copy-on-write)
    # ------------------------------------------------------------------
    raw_task_queue: dict[str, RemediationTask] = dict(state.get("task_queue", {}))
    # Copy-on-write: work with model copies so we never mutate state-owned objects
    task_queue: dict[str, RemediationTask] = {
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
        task_updates: dict[str, Any] = {}
        if task.qa_policy is None and task.current_attempt_id is None:
            recovered_policy = derive_missing_task_qa_policy(task, group)
            if recovered_policy is not None:
                task_updates["qa_policy"] = recovered_policy
            elif task.status not in _TERMINAL_STATUSES:
                details = (
                    "Task has no recoverable QA policy provenance before dispatch; "
                    "the Supervisor will fail closed."
                )
                errors.append(f"supervisor: task {task_id} has missing QA policy provenance.")
                consistency_events.append(
                    _build_consistency_event(
                        error_code="MISSING_QA_POLICY_PROVENANCE",
                        task_id=task_id,
                        expected_attempt_id=None,
                        received_attempt_id=None,
                        action="rejected",
                        details=details,
                    )
                )
        if (
            task.status not in _TERMINAL_STATUSES
            and task.current_attempt_id is None
            and task.no_fix_stage is None
            and group is not None
            and is_no_fix_group(group)
        ):
            task_updates["no_fix_stage"] = NoFixMitigationStage.PACKAGE_REMOVAL
            if task.selected_version is not None:
                task_updates["selected_version"] = None
            if not task.instruction or task.instruction.strip().casefold() == (
                "no upstream patch or workaround was found. inform the user."
            ):
                task_updates["instruction"] = build_no_fix_package_removal_instruction(group)
        if (
            task.task_revision == 0
            and task.status not in _TERMINAL_STATUSES
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
            and not task.parent_package_name
            and group is not None
            and group.fix_plan is not None
            and group.fix_plan.fixed_version
        ):
            task_updates["selected_version"] = group.fix_plan.fixed_version
        if (
            task.task_revision == 0
            and task.current_attempt_id is None
            and group is not None
            and is_transitive_group(group)
        ):
            parent_name, parent_version, parent_type = group_parent_context(group)
            if parent_name and task.parent_package_name != parent_name:
                task_updates["parent_package_name"] = parent_name
            if parent_version and task.parent_package_version != parent_version:
                task_updates["parent_package_version"] = parent_version
            if parent_name and task.target_package_name is None:
                task_updates["target_package_name"] = parent_name
                task_updates["target_dependency_type"] = task.target_dependency_type or parent_type
        if task_updates:
            if "qa_policy" in task_updates:
                task_updates["task_revision"] = task.task_revision + 1
            task_queue[task_id] = task.model_copy(update=task_updates)

    # A transitive VERSION_BUMP task is planned against its nearest directly
    # declared parent before the first update worker is dispatched. This keeps
    # child pins/overrides out of the initial instruction.
    for task_id, task in list(task_queue.items()):
        group = group_by_id.get(task.parent_group_id)
        if (
            group is not None
            and is_transitive_group(group)
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and task.status == TaskStatus.PENDING
            and task.current_attempt_id is None
            and task.parent_package_name
            and task.strategy_stage == SCARemediationStage.OSV_MINIMUM
        ):
            task_queue[task_id] = _plan_initial_transitive_task(task, group)

    # ------------------------------------------------------------------
    # 2. Ingest attempt-tagged worker results (active targets only)
    # ------------------------------------------------------------------
    active_target_task_ids = list(state.get("active_target_task_ids") or [])
    action_summaries: list[AgentActionSummary] = state.get("action_summaries") or []
    new_worker_attempt_ids: list[str] = []

    for task_id in active_target_task_ids:
        task = task_queue.get(task_id)
        if task is None:
            continue
        current_attempt_id = task.current_attempt_id
        snapshot = attempt_snapshots_by_id.get(current_attempt_id) if current_attempt_id else None
        result = worker_results_by_attempt.get(current_attempt_id) if current_attempt_id else None
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
            if (
                result_status == AgentActionStatus.SUCCESS
                and task.strategy == RoutingStrategy.CODE_WORKAROUND
            ):
                diag = result.execution_diagnostics
                val_files_match = set(diag.validated_files) == set(result.changed_files)
                overall_status = diag.per_gate_results.get("overall_status")
                structured_validation_passed = str(overall_status) in {
                    "PASS",
                    "WorkaroundValidationStatus.PASS",
                }
                is_valid = (
                    diag.validation_passed
                    and diag.validation_calls > 0
                    and val_files_match
                    and structured_validation_passed
                )
                if not is_valid:
                    result_status = AgentActionStatus.SURRENDER
                    errors.append(
                        f"supervisor: rejected workaround worker result for {task_id} due to invalid validation state."
                    )
            prior = retry_diagnostics_by_task.get(task_id)
            if prior is None:
                prior = UpdateRetryDiagnostics(task_id=task_id)
            group = group_by_id.get(task.parent_group_id)
            parent_name, _, parent_type = (
                group_parent_context(group) if group is not None else (None, None, None)
            )
            if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
                target_package_name = (
                    snapshot.target_package_name
                    or task.target_package_name
                    or (group.vulnerable_component if group is not None else None)
                )
                target_dependency_type = (
                    snapshot.target_dependency_type
                    or task.target_dependency_type
                    or (_override_dependency_type(group) if group is not None else None)
                )
            else:
                target_package_name = (
                    snapshot.target_package_name
                    or task.target_package_name
                    or (parent_name if group is not None and is_transitive_group(group) else None)
                    or (group.vulnerable_component if group is not None else None)
                )
                target_dependency_type = (
                    snapshot.target_dependency_type
                    or task.target_dependency_type
                    or (parent_type if group is not None and is_transitive_group(group) else None)
                )
            attempted_versions_by_target = dict(prior.attempted_versions_by_target)
            if target_package_name and attempted_versions:
                attempted_versions_by_target[target_package_name] = list(
                    dict.fromkeys(
                        [
                            *attempted_versions_by_target.get(target_package_name, []),
                            *attempted_versions,
                        ]
                    )
                )
            used_overrides = (
                prior.used_overrides
                or task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
                or target_dependency_type in _OVERRIDE_DEPENDENCY_TYPES
            )
            retry_diagnostics_by_task[task_id] = prior.model_copy(
                update={
                    "committed_attempt_id": current_attempt_id,
                    "attempted_versions": list(
                        dict.fromkeys(prior.attempted_versions + attempted_versions)
                    ),
                    "executed_versions": list(
                        dict.fromkeys(prior.executed_versions + result.executed_versions)
                    ),
                    "selected_version": task.selected_version,
                    "strategy_stage": task.strategy_stage,
                    "exhausted_update_path": task.exhausted_update_path,
                    "target_package_name": target_package_name or prior.target_package_name,
                    "target_dependency_type": target_dependency_type
                    or prior.target_dependency_type,
                    "attempted_versions_by_target": attempted_versions_by_target,
                    "used_overrides": used_overrides,
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
            if result.replay_plan is not None:
                workaround_replay_plans_by_task[task_id] = result.replay_plan
            if result_status == AgentActionStatus.SUCCESS:
                # Package removal still needs the normal QA install/test
                # checks. QA deliberately skips only ODC for this stage;
                # keep every successful attempt open so the snapshot is
                # consumed by QA before the task becomes terminal.
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
            elif task.strategy == RoutingStrategy.CODE_WORKAROUND:
                if task.no_fix_stage is not None:
                    transition, reset_workspace = _no_fix_failure_transition(
                        task,
                        group_by_id.get(task.parent_group_id),
                        failure_feedback=(
                            " | ".join(result.errors)
                            or (
                                result.action_summary.summary
                                if result.action_summary is not None
                                else None
                            )
                        ),
                    )
                    _commit_task_transition(
                        task_queue,
                        task_id,
                        updates=transition,
                        close_attempt=True,
                        clear_selected_version=True,
                    )
                    if reset_workspace:
                        reset_plan = _reset_no_fix_replay_plan(
                            workaround_replay_plans_by_task.get(task_id)
                        )
                        if reset_plan is None:
                            workaround_replay_plans_by_task.pop(task_id, None)
                        else:
                            workaround_replay_plans_by_task[task_id] = reset_plan
                else:
                    # A surrender is a completed worker outcome, not an active
                    # worker input. Close it before the next routing decision so
                    # a terminal workaround task cannot reach teardown with a
                    # live current attempt.
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
                failed_group = group_by_id.get(task.parent_group_id)
                transitive_failure = bool(failed_group and is_transitive_group(failed_group))
                next_failure_stage = (
                    _next_sca_stage(task.strategy_stage, transitive=True)
                    if transitive_failure
                    else task.strategy_stage
                )
                failure_updates: dict[str, Any] = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                if transitive_failure and next_failure_stage != task.strategy_stage:
                    failure_updates["strategy_stage"] = next_failure_stage
                    if next_failure_stage == SCARemediationStage.PACKAGE_OVERRIDE:
                        failure_updates.update(
                            {
                                "target_package_name": failed_group.vulnerable_component,
                                "target_dependency_type": _override_dependency_type(failed_group),
                                "selected_version": (
                                    failed_group.fix_plan.fixed_version
                                    if failed_group.fix_plan
                                    else None
                                ),
                            }
                        )
                    elif next_failure_stage == SCARemediationStage.CODE_WORKAROUND:
                        failure_updates.update(
                            {
                                "selected_version": None,
                                "exhausted_update_path": True,
                            }
                        )
                _commit_task_transition(
                    task_queue,
                    task_id,
                    updates=failure_updates,
                    close_attempt=True,
                    clear_selected_version=(
                        next_failure_stage == SCARemediationStage.CODE_WORKAROUND
                    ),
                )
                if transitive_failure:
                    committed_failure_task = task_queue[task_id]
                    retry_diagnostics_by_task[task_id] = prior.model_copy(
                        update={
                            "strategy_stage": committed_failure_task.strategy_stage,
                            "selected_version": committed_failure_task.selected_version,
                            "target_package_name": committed_failure_task.target_package_name,
                            "target_dependency_type": committed_failure_task.target_dependency_type,
                            "exhausted_update_path": committed_failure_task.exhausted_update_path,
                        }
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
                    if task.no_fix_stage is not None:
                        transition, reset_workspace = _no_fix_failure_transition(
                            task,
                            group_by_id.get(task.parent_group_id),
                            failure_feedback=summary.summary,
                        )
                        _commit_task_transition(
                            task_queue,
                            task_id,
                            updates=transition,
                            close_attempt=True,
                            clear_selected_version=True,
                        )
                        if reset_workspace:
                            reset_plan = _reset_no_fix_replay_plan(
                                workaround_replay_plans_by_task.get(task_id)
                            )
                            if reset_plan is None:
                                workaround_replay_plans_by_task.pop(task_id, None)
                            else:
                                workaround_replay_plans_by_task[task_id] = reset_plan
                    else:
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
    qa_evaluations: dict[str, QAEvaluation] = _normalize_qa_evaluations_for_tasks(
        dict(state.get("qa_evaluations", {})),
        task_queue,
        active_target_task_ids,
    )
    qa_result_task_ids: set[str] = set()
    new_qa_attempt_ids: list[str] = []
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
                clear_selected_version=True,
            )
        processed_qa_attempt_ids.add(task.current_attempt_id)
        new_qa_attempt_ids.append(task.current_attempt_id)
    auto_new_constraints: list[str] = []

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
            if evaluation.contract_error:
                # A malformed judge response is a QA-contract failure, not a
                # remediation failure. Requeue QA without advancing the
                # strategy stage or consuming the worker retry budget.
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates={"status": TaskStatus.OPTIMISTICALLY_FIXED},
                )
                errors.append(
                    f"supervisor: QA contract error for {resolved_t_id}; "
                    "requeued QA without consuming a remediation retry."
                )
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
                if task.no_fix_stage is not None:
                    no_fix_updates, reset_workspace = _no_fix_failure_transition(
                        task,
                        group_by_id.get(task.parent_group_id),
                        evaluation=evaluation,
                    )
                    _commit_task_transition(
                        task_queue,
                        resolved_t_id,
                        updates=no_fix_updates,
                        clear_selected_version=True,
                    )
                    if reset_workspace:
                        reset_plan = _reset_no_fix_replay_plan(
                            workaround_replay_plans_by_task.get(resolved_t_id)
                        )
                        if reset_plan is None:
                            workaround_replay_plans_by_task.pop(resolved_t_id, None)
                        else:
                            workaround_replay_plans_by_task[resolved_t_id] = reset_plan
                    continue

                group = group_by_id.get(task.parent_group_id)
                next_stage = _next_sca_stage(
                    task.strategy_stage,
                    transitive=bool(group and is_transitive_group(group)),
                )
                task_updates = {
                    "status": TaskStatus.NEEDS_RETRY,
                    "retry_count": task.retry_count + 1,
                }
                if task.strategy == RoutingStrategy.VERSION_BUMP:
                    task_updates["strategy_stage"] = next_stage
                    if next_stage == SCARemediationStage.PACKAGE_OVERRIDE:
                        task_updates.update(
                            {
                                "target_package_name": (
                                    group.vulnerable_component if group else task.parent_group_id
                                ),
                                "target_dependency_type": _override_dependency_type(group),
                                "selected_version": (
                                    group.fix_plan.fixed_version
                                    if group and group.fix_plan
                                    else task.selected_version
                                ),
                            }
                        )
                _commit_task_transition(
                    task_queue,
                    resolved_t_id,
                    updates=task_updates,
                )
                task = task_queue[resolved_t_id]
                if task.strategy == RoutingStrategy.VERSION_BUMP:
                    prior_diag = retry_diagnostics_by_task.get(resolved_t_id)
                    parent_name, _, parent_type = (
                        group_parent_context(group) if group is not None else (None, None, None)
                    )
                    next_target = (
                        group.vulnerable_component
                        if next_stage == SCARemediationStage.PACKAGE_OVERRIDE
                        else task.target_package_name or parent_name
                    )
                    next_target_type = (
                        _override_dependency_type(group)
                        if next_stage == SCARemediationStage.PACKAGE_OVERRIDE
                        else task.target_dependency_type or parent_type
                    )
                    if prior_diag is None:
                        retry_diagnostics_by_task[resolved_t_id] = UpdateRetryDiagnostics(
                            task_id=resolved_t_id,
                            strategy_stage=next_stage,
                            security_floor=(
                                group.fix_plan.fixed_version if group and group.fix_plan else None
                            ),
                            exhausted_update_path=(
                                next_stage == SCARemediationStage.CODE_WORKAROUND
                            ),
                            target_package_name=next_target,
                            target_dependency_type=next_target_type,
                            parent_package_name=parent_name,
                            parent_minimum_version=task.parent_minimum_version,
                            selected_version=task.selected_version,
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
                                "exhausted_update_path": next_stage
                                == SCARemediationStage.CODE_WORKAROUND,
                                "target_package_name": next_target,
                                "target_dependency_type": next_target_type,
                                "parent_package_name": parent_name,
                                "parent_minimum_version": task.parent_minimum_version,
                                "selected_version": task.selected_version,
                            }
                        )

    # ------------------------------------------------------------------
    # 4. Mark UNFIXABLE tasks that hit the retry cap
    # ------------------------------------------------------------------
    for task_id, task in task_queue.items():
        if (
            task.no_fix_stage == NoFixMitigationStage.UNFIXABLE
            and task.status not in _TERMINAL_STATUSES
        ):
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.selected_version is not None,
            )
        if task.no_fix_stage == NoFixMitigationStage.UNFIXABLE or (
            task.no_fix_stage is not None and task.status in _TERMINAL_STATUSES
        ):
            retry_plans_by_task.pop(task_id, None)
            workaround_replay_plans_by_task.pop(task_id, None)

    for task_id, task in task_queue.items():
        if (
            task.status == TaskStatus.NEEDS_RETRY
            and task.retry_count >= MAX_RETRIES
            and task.no_fix_stage is None
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
                clear_selected_version=task.selected_version is not None,
            )
            logger.info(
                "supervisor: task '%s' marked UNFIXABLE after %d retries.",
                task_id,
                task.retry_count,
            )

    # Keep diagnostics aligned with terminal task state without emitting a
    # projection-repair event. The selected version is no longer dispatchable,
    # while target and attempt evidence remains useful in the final report.
    for task_id, task in task_queue.items():
        if task.status not in _TERMINAL_STATUSES:
            continue
        diagnostics = retry_diagnostics_by_task.get(task_id)
        if diagnostics is not None and diagnostics.selected_version is not None:
            retry_diagnostics_by_task[task_id] = diagnostics.model_copy(
                update={"selected_version": None}
            )

    # ------------------------------------------------------------------
    # 4.5. Reconcile terminal workaround children before routing.
    # ------------------------------------------------------------------
    stale_pivot_parent_ids = _reconcile_terminal_pivot_parents(
        task_queue,
        qa_evaluations,
        retry_diagnostics_by_task,
        retry_plans_by_task,
        group_by_id,
    )
    if stale_pivot_parent_ids:
        errors.append(
            "supervisor: terminalized stale pivot parent task(s) after their "
            "workaround child reached a terminal state: "
            f"{stale_pivot_parent_ids}."
        )
    # 5. Short-circuit: if an active task is optimistically_fixed â†’ qa_critic
    # ------------------------------------------------------------------
    decision: SupervisorDecision | None = None
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
            limit=QA_DISPATCH_LIMIT,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated task before starting more remediation.",
                decision_reason=(
                    f"Routing task '{active_qa_ready[0]}' to QA after a successful worker attempt."
                ),
            )

    # The Supervisor is not permitted to bypass the post-QA triage handoff. This is
    # deliberately applied after the optimistic QA guard so Supervisor stays
    # the sole routing authority while preserving the required order:
    # worker -> supervisor -> QA -> supervisor -> triage -> supervisor.
    if state.get("triage_required") and state.get("status") in {
        "qa_completed",
        "qa_failed",
        "final_scan_completed",
    }:
        decision = SupervisorDecision(
            next_node="triage",
            target_task_ids=[],
            instructions="Route the completed QA scan to post-remediation triage before dispatching more work.",
            decision_reason="Supervisor guardrail: triage_required is set after a parseable QA scan.",
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
            qa_ready = _qa_ready_task_ids(task_queue, limit=QA_DISPATCH_LIMIT)
            if qa_ready:
                decision = SupervisorDecision(
                    next_node="qa_critic",
                    target_task_ids=qa_ready,
                    instructions="Run QA on the next remaining optimistically fixed task.",
                    decision_reason=f"Routing task '{qa_ready[0]}' to QA.",
                )

    # ------------------------------------------------------------------
    # 6. Deterministic retry planning
    # ------------------------------------------------------------------
    if decision is None and _needs_planner(
        task_queue,
        qa_evaluations,
        retry_diagnostics_by_task,
        str(state.get("status") or ""),
    ):
        parsed_diagnostics, parsed_plans = _run_deterministic_retry_planner(
            task_queue,
            group_by_id,
            retry_diagnostics_by_task,
        )
        planner_violations = _planner_plan_violations(
            parsed_plans,
            task_queue,
            parsed_diagnostics,
        )
        if planner_violations:
            errors.extend(
                f"supervisor: deterministic retry planning: {violation}"
                for violation in planner_violations
            )
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
                    f"supervisor: deterministic retry repair: {violation}"
                    for violation in repair_violations
                )
        retry_diagnostics_by_task = parsed_diagnostics
        if not _planner_plan_violations(parsed_plans, task_queue, parsed_diagnostics):
            _commit_retry_plans(
                task_queue,
                retry_diagnostics_by_task,
                retry_plans_by_task,
                parsed_plans,
            )
            if any(plan.action == "pivot_workaround" for plan in parsed_plans.values()):
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                )

    # Deterministic retry planning above is the only Supervisor planning path.

    # Re-apply optimistic short-circuit after retry planning.
    if state.get("status") != "qa_completed" and active_target_task_ids:
        active_qa_ready = _qa_ready_task_ids(
            task_queue,
            preferred_ids=active_target_task_ids,
            limit=QA_DISPATCH_LIMIT,
        )
        if active_qa_ready:
            decision = SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=active_qa_ready,
                instructions="Run QA on the current remediated task before starting more remediation.",
                decision_reason=(
                    f"Routing task '{active_qa_ready[0]}' to QA after a successful worker attempt."
                ),
            )

    # ------------------------------------------------------------------
    # 6b. Deterministic authority boundary
    # ------------------------------------------------------------------
    deterministic_decision = _deterministic_routing(
        task_queue,
        group_by_id,
        qa_evaluations,
        retry_diagnostics_by_task,
        action_summaries=action_summaries,
        active_target_task_ids=active_target_task_ids,
        current_status=str(state.get("status") or ""),
        triage_required=bool(state.get("triage_required")),
    )
    # The deterministic decision is authoritative. QA feedback below is
    # carried only when Python produced it for the selected task; no model
    # can add worker feedback or constraints to this projection.
    decision = deterministic_decision

    # ------------------------------------------------------------------
    # 7. Guardrails: validate and clamp (or deterministic fallback)
    # ------------------------------------------------------------------
    pivot_parent_status_by_parent: dict[str, TaskStatus] = {}
    pivot_target_parent_ids: set[str] = set()

    if decision is not None and _no_fix_decision_requires_fallback(decision, task_queue):
        errors.append(
            "supervisor: rejected router decision that attempted to bypass the "
            "deterministic NO_FIX mitigation lifecycle."
        )
        decision = None

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
        logger.info("supervisor: deterministic fallback â†’ next_node=%s", decision.next_node)
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
                limit=QA_DISPATCH_LIMIT,
            )
        elif decision.next_node == "update_subagent":
            valid_target_ids = _update_worker_task_ids(
                task_queue,
                retry_diagnostics_by_task,
                preferred_ids=list(decision.target_task_ids),
                limit=UPDATE_DISPATCH_LIMIT,
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
        requested_target_count = len(decision.target_task_ids)
        if decision.next_node == "workaround_subagent" and len(valid_target_ids) != 1:
            logger.warning(
                "supervisor: workaround_subagent needs 1 target, got %d â€” falling back.",
                len(valid_target_ids),
            )
            needs_fallback = True
        elif decision.next_node == "update_subagent" and not valid_target_ids:
            logger.warning("supervisor: update_subagent needs â‰¥1 target, got 0 â€” falling back.")
            needs_fallback = True

        if (
            not needs_fallback
            and decision.next_node == "update_subagent"
            and requested_target_count > UPDATE_DISPATCH_LIMIT
        ):
            logger.warning(
                "supervisor: update_subagent current policy allows exactly 1 target, got %d â€” falling back.",
                requested_target_count,
            )
            needs_fallback = True
        if (
            not needs_fallback
            and decision.next_node == "qa_critic"
            and requested_target_count > QA_DISPATCH_LIMIT
        ):
            logger.warning(
                "supervisor: qa_critic current policy allows exactly 1 target, got %d â€” falling back.",
                requested_target_count,
            )
            needs_fallback = True
        if not needs_fallback and decision.next_node == "update_subagent" and valid_target_ids:
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
                    "supervisor: update_subagent request mixed first-pass and retry tasks â€” falling back."
                )
                needs_fallback = True
        if not needs_fallback and decision.next_node == "qa_critic" and not valid_target_ids:
            logger.warning("supervisor: qa_critic needs at least 1 target, got 0 â€” falling back.")
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
            for task_id in valid_target_ids:
                task = task_queue.get(task_id)
                if task is None or task.no_fix_stage is None:
                    continue
                group = group_by_id.get(task.parent_group_id)
                if task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
                    clean_revised_instructions[task_id] = build_no_fix_retry_instruction(
                        task,
                        group,
                        evaluation=qa_evaluations.get(task_id)
                        or qa_evaluations.get(task.parent_group_id),
                    )
                elif task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL and group:
                    # The package-removal instruction is supervisor-owned too;
                    # an untrusted decision cannot replace the scoped manifest capability with
                    # arbitrary prose or a generic source-only workaround.
                    clean_revised_instructions[task_id] = build_no_fix_package_removal_instruction(
                        group
                    )
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
                k: v for k, v in decision.feedback_by_task.items() if k in known_task_ids
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
                    f"revised_instructions for {missing_retry_revisions}; replanning deterministically."
                )
                recovery_tasks = {
                    task_id: task_queue[task_id]
                    for task_id in missing_retry_revisions
                    if task_id in task_queue
                    and task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
                }
                recovery_inputs = {
                    task_id: retry_diagnostics_by_task.get(
                        task_id,
                        UpdateRetryDiagnostics(
                            task_id=task_id,
                            strategy_stage=task_queue[task_id].strategy_stage,
                        ),
                    )
                    for task_id in recovery_tasks
                }
                if recovery_tasks:
                    recovered_diagnostics, recovered_plans = _run_deterministic_retry_planner(
                        recovery_tasks,
                        group_by_id,
                        recovery_inputs,
                    )
                    retry_diagnostics_by_task.update(recovered_diagnostics)
                    _commit_retry_plans(
                        task_queue,
                        retry_diagnostics_by_task,
                        retry_plans_by_task,
                        recovered_plans,
                    )
                decision = _deterministic_routing(
                    task_queue,
                    group_by_id,
                    qa_evaluations,
                    retry_diagnostics_by_task,
                    action_summaries=action_summaries,
                    active_target_task_ids=active_target_task_ids,
                    current_status=str(state.get("status") or ""),
                    triage_required=bool(state.get("triage_required")),
                )
                valid_target_ids = list(decision.target_task_ids)
                valid_unfixable_ids = list(decision.unfixable_task_ids)
                clean_revised_instructions = dict(decision.revised_instructions)
                clean_feedback = dict(decision.feedback_by_task)
                clean_updated_task_strategies = dict(decision.updated_task_strategies)

            # Validate task_status_updates â€” only known tasks, only terminal statuses
            clean_status_updates: dict[str, TaskStatus] = {}
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
                and task_queue[req.parent_task_id].no_fix_stage is None
            ]
            pivot_strategy_by_parent: dict[str, RoutingStrategy] = {
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
            malformed_pivot_parent_ids: list[str] = []
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
                if decision.next_node
                != _worker_node_for_strategy(pivot_strategy_by_parent[task_id])
            ]
            pivot_validation_failed = bool(
                malformed_pivot_parent_ids or incompatible_targeted_pivots
            )
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
                failed_parent_ids = list(
                    {
                        *malformed_pivot_parent_ids,
                        *incompatible_targeted_pivots,
                    }
                )
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
                    decision_code=decision.decision_code,
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
                logger.warning("supervisor: decision rebuild failed (%s) â€” falling back.", exc)
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
        if (
            t_id in task_queue
            and new_status in _allowed_statuses
            and task_queue[t_id].status not in _TERMINAL_STATUSES
        ):
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
                    and task_queue[t_id].selected_version is not None
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
                clear_selected_version=task_queue[t_id].selected_version is not None,
            )

    # 8e. Materialize spawn requests
    child_ids_by_parent: dict[str, list[str]] = {}
    if decision.spawn_requests:
        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=list(decision.spawn_requests),
            task_queue=task_queue,
            group_by_id=group_by_id,
            errors=errors,
            valid_groups=valid_groups,
            qa_evaluations=qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
            consistency_events=consistency_events,
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

    resolved_target_task_ids: list[str] = []
    remapped_feedback_by_task: dict[str, str] = {}
    for task_id in decision.target_task_ids:
        if task_id in pivot_target_parent_ids:
            child_ids = child_ids_by_parent.get(task_id, [])
            if child_ids:
                child_task_id = child_ids[0]
                child_task = task_queue.get(child_task_id)
                if (
                    child_task is not None
                    and child_task.strategy == RoutingStrategy.CODE_WORKAROUND
                    and child_task.status in _WORKABLE_STATUSES
                ):
                    resolved_target_task_ids.append(child_task_id)
                    if task_id in decision.feedback_by_task:
                        remapped_feedback_by_task[child_task_id] = decision.feedback_by_task[
                            task_id
                        ]
                else:
                    # A replayed pivot may find a child that is already
                    # optimistic/terminal.  It is valid evidence that the
                    # pivot exists, but it is not a worker target.  Let the
                    # deterministic reroute below choose QA, final scanning,
                    # teardown, or another actionable task.
                    logger.info(
                        "supervisor: pivot child '%s' is not dispatchable "
                        "(status=%s); recomputing the next route.",
                        child_task_id,
                        child_task.status.value if child_task is not None else "missing",
                    )
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
    all_tasks_terminal = bool(task_queue) and all(
        task.status in _TERMINAL_STATUSES for task in task_queue.values()
    )
    if (
        all_tasks_terminal
        and state.get("workspace_volume")
        and not state.get("final_full_scan_completed", False)
    ):
        decision = decision.model_copy(
            update={
                "decision_code": DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                "next_node": "final_full_scan",
                "target_task_ids": [],
                "instructions": "Run the authoritative full Dependency-Check scan before teardown.",
                "decision_reason": (
                    "Supervisor routing barrier required the final full scan before teardown."
                ),
            }
        )
        resolved_next_node = "final_full_scan"
        resolved_target_task_ids = []
    elif resolved_next_node == "final_full_scan":
        errors.append(
            "supervisor: rejected final_full_scan because the terminal workspace gate is not satisfied."
        )
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
            workspace_volume=state.get("workspace_volume"),
            final_full_scan_completed=bool(state.get("final_full_scan_completed")),
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = list(decision.target_task_ids)
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
        group_by_id,
    )
    # Status overrides and parent terminalization above are also untrusted
    # router requests. Re-clamp after those mutations so a task that became
    # terminal in this transition cannot remain in the dispatch projection.
    resolved_target_task_ids = _normalize_target_task_ids_for_node(
        resolved_next_node,
        resolved_target_task_ids,
        task_queue,
        retry_diagnostics_by_task,
        group_by_id,
    )
    remapped_feedback_by_task = {
        task_id: feedback
        for task_id, feedback in remapped_feedback_by_task.items()
        if task_id in set(resolved_target_task_ids)
    }

    # Programmatic override: forcefully inject the latest QA feedback into
    # workaround-subagent retries so the worker receives the evidence.
    if resolved_next_node == "workaround_subagent":
        for task_id in resolved_target_task_ids:
            task = task_queue.get(task_id)
            if task:
                eval_ = qa_evaluations.get(task_id) or qa_evaluations.get(task.parent_group_id)
                if eval_ and eval_.retry_feedback:
                    remapped_feedback_by_task[task_id] = eval_.retry_feedback
    if (
        resolved_next_node in {"update_subagent", "workaround_subagent", "qa_critic"}
        and not resolved_target_task_ids
    ):
        errors.append(
            "supervisor: recomputing routing because no dispatchable target tasks remained."
        )
        # This can happen when a replayed pivot reuses a child that has
        # already reached a terminal state.  Teardown would incorrectly skip
        # other non-terminal tasks, so ask the deterministic router for the
        # next eligible action after the parent/child transition is committed.
        decision = _deterministic_routing(
            task_queue,
            group_by_id,
            qa_evaluations,
            retry_diagnostics_by_task,
            action_summaries=action_summaries,
            active_target_task_ids=active_target_task_ids,
            current_status=str(state.get("status") or ""),
            triage_required=bool(state.get("triage_required")),
            workspace_volume=state.get("workspace_volume"),
            final_full_scan_completed=bool(state.get("final_full_scan_completed")),
        )
        resolved_next_node = decision.next_node
        resolved_target_task_ids = _normalize_target_task_ids_for_node(
            resolved_next_node,
            list(decision.target_task_ids),
            task_queue,
            retry_diagnostics_by_task,
            group_by_id,
        )
        remapped_feedback_by_task = {
            task_id: feedback
            for task_id, feedback in decision.feedback_by_task.items()
            if task_id in set(resolved_target_task_ids)
        }

    if resolved_next_node in _WORKER_NODES and not resolved_target_task_ids:
        stalled_task_ids = [
            task_id for task_id, task in task_queue.items() if task.status not in _TERMINAL_STATUSES
        ]
        for task_id in stalled_task_ids:
            task = task_queue[task_id]
            _commit_task_transition(
                task_queue,
                task_id,
                updates={"status": TaskStatus.UNFIXABLE},
                close_attempt=task.current_attempt_id is not None,
                clear_selected_version=task.selected_version is not None,
            )
        errors.append(
            "supervisor: fail-closed an empty worker route; no dispatchable "
            f"target remained. Marked unresolved tasks unfixable: {stalled_task_ids}."
        )

        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        if terminal_workspace:
            resolved_next_node = "final_full_scan"
            decision = SupervisorDecision(
                decision_code=DecisionCode.FINAL_FULL_SCAN_REQUIRED,
                next_node=resolved_next_node,
                target_task_ids=[],
                instructions="Run the authoritative full Dependency-Check scan before teardown.",
                decision_reason=(
                    "All tasks were terminalized after the Supervisor found no "
                    "dispatchable worker target."
                ),
            )
        else:
            resolved_next_node = "teardown"
            decision = SupervisorDecision(
                decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                next_node=resolved_next_node,
                target_task_ids=[],
                instructions="No dispatchable tasks remain; proceed to teardown.",
                decision_reason=(
                    "The Supervisor fail-closed after a worker route had no dispatchable target."
                ),
            )
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
            workaround_ctx = None
            if resolved_next_node == "workaround_subagent":
                attempts = _attempts_for_task(attempt_snapshots_by_id, task_id)
                parent_task_ids, parent_group_ids = _workaround_task_ancestry(
                    task,
                    task_queue,
                )
                evidence = _qa_failure_evidence_for_workaround_retry(
                    task_id,
                    task.parent_group_id,
                    qa_evaluations,
                    qa_results_by_attempt,
                    related_task_ids=parent_task_ids,
                    related_group_ids=parent_group_ids,
                )
                phase = (
                    WorkaroundPhase.QA_REGRESSION_REPAIR
                    if attempts or _qa_evidence_indicates_test_regression(evidence)
                    else WorkaroundPhase.INITIAL_MITIGATION
                )
                group = next(
                    (
                        candidate
                        for candidate in state.get("valid_groups", []) or []
                        if candidate.group_id == task.parent_group_id
                    ),
                    None,
                )
                vulnerability_mechanism = (
                    _extract_workaround_vulnerability_mechanism(group) if group is not None else ""
                )
                workaround_ctx = WorkaroundContext(
                    phase=phase,
                    vulnerability_mechanism=vulnerability_mechanism,
                    qa_evidence=evidence,
                    no_fix_stage=task.no_fix_stage,
                    reset_prior_stage_workspace=(
                        task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
                        and task.parent_group_id in group_by_id
                    ),
                )
            task, snapshot = _create_attempt_snapshot(
                task,
                dispatch_node=resolved_next_node,
                snapshots_by_id=attempt_snapshots_by_id,
                state_revision=state_revision,
                plan_id=plan.plan_id if plan is not None else None,
                workaround_context=workaround_ctx,
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
    returned_constraints: list[str] = list(auto_new_constraints)
    for constraint in decision.new_constraints:
        if (
            constraint
            and constraint not in existing_constraints
            and constraint not in returned_constraints
        ):
            returned_constraints.append(constraint)

    # Build feedback_by_group for bridge node backward compat
    feedback_by_task = remapped_feedback_by_task
    feedback_by_group: dict[str, str] = {}
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
        if (event.task_id, event.received_attempt_id, event.error_code) not in existing_event_keys
    ]
    # ``errors`` uses an additive reducer, so suppress both duplicate messages
    # from this invocation and exact messages already committed by an earlier
    # supervisor pass. Structured consistency events remain the detailed,
    # attempt-correlated audit record.
    errors = list(dict.fromkeys(error for error in errors if error not in prior_error_messages))

    # ------------------------------------------------------------------
    # 9. Return state patch
    # ------------------------------------------------------------------
    return {
        "status": "supervisor_routed",
        "next_routing_step": resolved_next_node,
        "decision_code": decision.decision_code,
        "supervisor_audit": _emit_audit(decision, consistency_events, state_revision),
        "active_target_task_ids": resolved_target_task_ids,
        # Keep active_target_group_ids populated for bridge nodes that still use it
        "active_target_group_ids": [
            task_queue[t].parent_group_id for t in resolved_target_task_ids if t in task_queue
        ],
        "feedback_by_task": feedback_by_task,
        "feedback_by_group": feedback_by_group,
        "supervisor_instructions": decision.instructions,
        # Compatibility projection: the attempt-tagged QA envelope remains
        # authoritative, while this task-keyed view is retained for existing
        # callers and prompt builders.
        "qa_evaluations": qa_evaluations,
        "task_queue": task_queue,
        "valid_groups": valid_groups,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "retry_plans_by_task": retry_plans_by_task,
        "workaround_replay_plans_by_task": workaround_replay_plans_by_task,
        "attempt_snapshots_by_id": attempt_snapshots_by_id,
        "worker_results_by_attempt": worker_results_by_attempt,
        "qa_results_by_attempt": qa_results_by_attempt,
        "processed_worker_attempt_ids": list(new_worker_attempt_ids),
        "processed_qa_attempt_ids": list(new_qa_attempt_ids),
        "consistency_events": consistency_events,
        "state_revision": state_revision,
        # constraints_ledger uses operator.add â€” return only NEW entries
        "constraints_ledger": returned_constraints,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Conditional-edge router
# ---------------------------------------------------------------------------


def supervisor_router(state: OrchestratorState) -> str:
    """Return the committed route, recomputing it when state is invalid."""
    step = state.get("next_routing_step", "")
    task_queue = dict(state.get("task_queue", {}) or {})
    if (
        step == "teardown"
        and state.get("workspace_volume")
        and task_queue
        and not state.get("final_full_scan_completed", False)
        and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
    ):
        logger.warning("supervisor_router: enforcing final_full_scan before teardown.")
        return "final_full_scan"
    if step == "final_full_scan":
        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        if not terminal_workspace:
            logger.warning("supervisor_router: rejecting premature final_full_scan route.")
            step = ""
    if step in _WORKER_NODES and task_queue and not state.get("active_target_task_ids"):
        logger.error(
            "supervisor_router: refusing worker route '%s' without active targets; "
            "failing closed to teardown.",
            step,
        )
        terminal_workspace = (
            bool(task_queue)
            and bool(state.get("workspace_volume"))
            and all(task.status in _TERMINAL_STATUSES for task in task_queue.values())
            and not state.get("final_full_scan_completed", False)
        )
        return "final_full_scan" if terminal_workspace else "teardown"
    if step in _VALID_NEXT_NODES:
        return step
    logger.error("supervisor_router: invalid next_routing_step '%s' - recomputing.", step)

    valid_groups = list(state.get("valid_groups", []) or [])
    group_by_id = {group.group_id: group for group in valid_groups}
    decision = _deterministic_routing(
        dict(state.get("task_queue", {}) or {}),
        group_by_id,
        dict(state.get("qa_evaluations", {}) or {}),
        dict(state.get("retry_diagnostics_by_task", {}) or {}),
        active_target_task_ids=list(state.get("active_target_task_ids", []) or []),
        current_status=str(state.get("status") or ""),
        triage_required=bool(state.get("triage_required")),
        workspace_volume=state.get("workspace_volume"),
        final_full_scan_completed=bool(state.get("final_full_scan_completed")),
    )
    if decision.next_node not in _VALID_NEXT_NODES:
        logger.critical(
            "supervisor_router: deterministic recomputation produced invalid node '%s'.",
            decision.next_node,
        )
        return "teardown"
    return decision.next_node
