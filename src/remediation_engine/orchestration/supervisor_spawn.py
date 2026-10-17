"""Spawn materialization and parent/child pivot lifecycle helpers."""

from __future__ import annotations

import logging
from typing import Any

from remediation_engine.contracts.schemas import (
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    FailureCategory,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorRetryPlan,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.supervisor_planner import (
    _build_high_level_retry_instruction,
    instruction_digest,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _parent_status_for_strategy_pivot,
)
from remediation_engine.orchestration.supervisor_routing import _build_consistency_event
from remediation_engine.orchestration.task_utils import select_package_fix_plan

logger = logging.getLogger(__name__)


def _commit_task_transition(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._commit_task_transition(*args, **kwargs)


def _plan_initial_transitive_task(
    task: RemediationTask,
    group: VulnerabilityGroup,
    *,
    candidate_versions: list[str] | None = None,
) -> RemediationTask:
    """Project an already committed outer-plan decision for a transitive task.

    Registry and parent-version selection belong to the outer Portfolio
    Orchestrator.  This compatibility helper only exposes the task's
    solver-approved values to callers that still use the historical helper.
    """
    if task.strategy != RoutingStrategy.VERSION_BUMP or task.status != TaskStatus.PENDING:
        return task
    approved = list(
        dict.fromkeys(
            str(version).strip().lstrip("vV")
            for version in (candidate_versions or task.allowed_target_versions)
            if str(version).strip()
        )
    )
    if candidate_versions is not None:
        candidate_versions[:] = approved
    selected = task.selected_version or (approved[0] if approved else None)
    if selected is None:
        return task
    return task.model_copy(
        update={
            "selected_version": selected,
            "instruction": task.instruction
            or _build_high_level_retry_instruction(
                task,
                group,
                None,
                UpdateRetryDiagnostics(
                    task_id=task.task_id,
                    strategy_stage=task.strategy_stage,
                    selected_version=selected,
                    candidate_versions_considered=approved,
                    registry_query_performed=False,
                ),
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
                        "instruction_digest": instruction_digest(committed_parent.instruction),
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
            parent_eval = (qa_evaluations or {}).get(parent_task.task_id)
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

        new_group_id = parent_task.parent_group_id
        # Strategy pivots remain child-task audit records, but the package-only
        # triage group is the stable logical portfolio item. Never manufacture
        # a second strategy-specific VulnerabilityGroup.

        package_selection = (
            select_package_fix_plan(
                group_by_id[parent_task.parent_group_id],
                req.strategy,
            )
            if group_by_id and parent_task.parent_group_id in group_by_id
            else None
        )
        new_task = RemediationTask(
            task_id=child_task_id,
            parent_group_id=new_group_id,
            parent_task_id=req.parent_task_id,
            qa_policy=child_policy,
            strategy=req.strategy,
            selected_plan_issue_ids=(
                list(package_selection.issue_ids) if package_selection is not None else []
            ),
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
