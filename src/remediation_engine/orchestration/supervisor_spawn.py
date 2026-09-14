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
    _override_dependency_type,
    _registry_report_versions,
    _registry_selected_version,
    instruction_digest,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _parent_status_for_strategy_pivot,
)
from remediation_engine.orchestration.supervisor_routing import _build_consistency_event
from remediation_engine.orchestration.task_utils import group_parent_context
from remediation_engine.tools.registry_tools import plan_npm_parent_version

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
    """Select the first parent-first candidate before worker dispatch.

    The worker receives only the committed result of this function. Registry
    failures or an empty candidate set advance deterministically to the next
    parent stage, and only a fully exhausted parent path commits a child
    package-manager override.

    Args:
        task: Pending transitive dependency task to plan.
        group: Vulnerability group containing the transitive dependency chain.
        candidate_versions: Optional mutable output list populated with the
            unfiltered registry candidates used for the selected parent stage.
    """
    if (
        task.strategy != RoutingStrategy.VERSION_BUMP
        or task.status != TaskStatus.PENDING
        or task.parent_package_name is None
        or task.strategy_stage != SCARemediationStage.OSV_MINIMUM
    ):
        return task
    child_fixed_version = group.fix_plan.fixed_version if group.fix_plan else None
    parent_name, parent_version, parent_type = group_parent_context(group)
    installed_parent_version = task.parent_package_version or parent_version
    if installed_parent_version and task.parent_package_version != installed_parent_version:
        task = task.model_copy(update={"parent_package_version": installed_parent_version})
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
        report_candidates = _registry_report_versions(report, "Eligible Candidates")
        if not report_candidates:
            report_candidates = _registry_report_versions(report, "Compatible Parent Versions")
        selected = _registry_selected_version(report)
        if selected and candidate_versions is not None:
            for candidate in [*report_candidates, selected]:
                if candidate not in candidate_versions:
                    candidate_versions.append(candidate)
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
