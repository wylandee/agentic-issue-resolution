"""Wrapper nodes bridging orchestrator state to worker and QA runtimes.

The wrapper implementation lives outside :mod:`graph` so graph topology and
entrypoint concerns remain easy to inspect.  Worker dependencies are resolved
lazily from the graph module at invocation time: this intentionally preserves
the established test and integration seam where callers monkeypatch names on
``remediation_engine.orchestration.graph``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    FailureCategory,
    MultiPackageAction,
    QAAttemptResult,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
)
from remediation_engine.orchestration import qa_test_parsing as _qa_test_parsing
from remediation_engine.orchestration.portfolio_orchestrator import isolate_delta_failure
from remediation_engine.orchestration.state import (
    OrchestratorState,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from remediation_engine.orchestration.tools_manifest import apply_multi_package_action

log = logging.getLogger(__name__)


def _graph_module():
    """Return the graph module whose collaborators should be invoked.

    Importing lazily avoids a graph/ wrapper import cycle and, importantly,
    means monkeypatches applied to graph-level dependency names are observed
    for every invocation rather than captured during module import.
    """
    from remediation_engine.orchestration import graph

    return graph


def _dispatch_boundary_rejection(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    expected_node: str,
) -> dict[str, Any] | None:
    """Reject a worker/QA invocation without a committed attempt snapshot."""
    if "attempt_snapshots_by_id" not in state:
        details = "Dispatch requires the authoritative attempt snapshot map."
        errors = [
            f"graph: rejected {expected_node} dispatch for {task.task_id}: {details}"
            for task in target_tasks
        ]
        events = [
            StateConsistencyEvent(
                error_code="DISPATCH_SNAPSHOT_MAP_MISSING",
                task_id=task.task_id,
                expected_attempt_id=task.current_attempt_id,
                received_attempt_id=task.current_attempt_id,
                action="ignored",
                details=details,
            )
            for task in target_tasks
        ]
        if not errors:
            return None
        return {
            "status": "supervisor_routed",
            "next_routing_step": "supervisor",
            "active_target_task_ids": [],
            "errors": errors,
            "consistency_events": events,
        }
    snapshots = state.get("attempt_snapshots_by_id") or {}
    errors: list[str] = []
    events: list[StateConsistencyEvent] = []
    active_cluster_id = state.get("active_cluster_id")
    active_batch_id = state.get("active_dispatch_batch_id")
    active_action = state.get("active_multi_package_action")
    active_action_digest = (
        _graph_module().instruction_digest(active_action.model_dump_json())
        if isinstance(active_action, MultiPackageAction)
        else None
    )
    if active_cluster_id and len(target_tasks) < 2:
        details = "An atomic cluster dispatch must contain every active leaf task."
        errors.append(f"graph: rejected {expected_node} dispatch: {details}")
        events.append(
            StateConsistencyEvent(
                error_code="CLUSTER_TARGET_SET_INCOMPLETE",
                action="rejected",
                details=details,
            )
        )
    if active_cluster_id:
        target_ids = {task.task_id for task in target_tasks}
        action_ids = (
            {mutation.task_id for mutation in active_action.package_mutations}
            if isinstance(active_action, MultiPackageAction)
            else set()
        )
        if not isinstance(active_action, MultiPackageAction):
            details = "Atomic cluster dispatch has no validated Supervisor action."
            errors.append(f"graph: rejected {expected_node} dispatch: {details}")
            events.append(
                StateConsistencyEvent(
                    error_code="CLUSTER_ACTION_MISSING",
                    action="rejected",
                    details=details,
                )
            )
        elif (
            active_action.cluster_id != active_cluster_id
            or active_action.dispatch_batch_id != active_batch_id
        ):
            details = (
                "Atomic cluster action provenance does not match the active Supervisor "
                "cluster or dispatch batch."
            )
            errors.append(f"graph: rejected {expected_node} dispatch: {details}")
            events.append(
                StateConsistencyEvent(
                    error_code="CLUSTER_ACTION_PROVENANCE_MISMATCH",
                    action="rejected",
                    details=details,
                )
            )
        elif action_ids != target_ids:
            details = "Atomic cluster target IDs must exactly match the committed action mutations."
            errors.append(f"graph: rejected {expected_node} dispatch: {details}")
            events.append(
                StateConsistencyEvent(
                    error_code="CLUSTER_TARGET_SET_MISMATCH",
                    action="rejected",
                    details=details,
                )
            )
    for task in target_tasks:
        attempt_id = task.current_attempt_id
        snapshot = snapshots.get(attempt_id) if attempt_id else None
        error_code: str | None = None
        details = ""
        if attempt_id is None:
            error_code = "DISPATCH_WITHOUT_ATTEMPT"
            details = "Active worker target has no committed attempt snapshot."
        elif snapshot is None:
            error_code = "DISPATCH_ATTEMPT_MISSING"
            details = "Task references an attempt that is absent from the snapshot map."
        elif task.qa_policy is None and snapshot.qa_policy is None:
            error_code = "MISSING_QA_POLICY_PROVENANCE"
            details = (
                "Task and committed attempt must carry a supervisor-owned QA policy "
                "before dispatch."
            )
        elif task.qa_policy is None or snapshot.qa_policy is None:
            error_code = "DISPATCH_SNAPSHOT_CONTRADICTION"
            details = "Task and committed attempt disagree because QA policy provenance is missing."
        elif active_cluster_id and (
            snapshot.cluster_id != active_cluster_id
            or snapshot.dispatch_batch_id != active_batch_id
            or (active_action_digest is not None and snapshot.action_digest != active_action_digest)
        ):
            error_code = "CLUSTER_SNAPSHOT_MISMATCH"
            details = "Task snapshot cluster, batch, or action provenance does not match Supervisor state."
        elif active_cluster_id and expected_node == "update_subagent" and active_action is None:
            error_code = "CLUSTER_ACTION_MISSING"
            details = "Atomic update dispatch has no Supervisor-committed multi-package action."
        elif (
            snapshot.task_id != task.task_id
            or snapshot.task_revision != task.task_revision
            or snapshot.strategy_stage != task.strategy_stage
            or snapshot.no_fix_stage != task.no_fix_stage
            or snapshot.qa_policy != task.qa_policy
            or snapshot.selected_version != task.selected_version
            or (
                snapshot.selected_plan_issue_ids
                and snapshot.selected_plan_issue_ids != task.selected_plan_issue_ids
            )
            or snapshot.instruction != task.instruction
            or snapshot.instruction_digest != _graph_module().instruction_digest(task.instruction)
            or (
                snapshot.dispatch_node == "update_subagent"
                and task.strategy != RoutingStrategy.VERSION_BUMP
            )
            or (
                snapshot.dispatch_node == "workaround_subagent"
                and task.strategy != RoutingStrategy.CODE_WORKAROUND
            )
        ):
            error_code = "DISPATCH_SNAPSHOT_CONTRADICTION"
            details = "Task fields do not match the immutable dispatch snapshot."
        elif (
            expected_node in {"update_subagent", "workaround_subagent"}
            and snapshot.dispatch_node != expected_node
        ):
            error_code = "DISPATCH_NODE_MISMATCH"
            details = f"Snapshot was committed for {snapshot.dispatch_node}, not {expected_node}."
        if error_code is not None:
            errors.append(f"graph: rejected {expected_node} dispatch for {task.task_id}: {details}")
            events.append(
                StateConsistencyEvent(
                    error_code=error_code,
                    task_id=task.task_id,
                    expected_attempt_id=attempt_id,
                    received_attempt_id=attempt_id,
                    action="ignored",
                    details=details,
                )
            )

    if not errors:
        return None
    return {
        "status": "supervisor_routed",
        "next_routing_step": "supervisor",
        "active_target_task_ids": [],
        "errors": errors,
        "consistency_events": events,
    }


def _workspace_snapshot_id(
    target_tasks: list[RemediationTask],
    snapshots_by_id: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the stable workspace snapshot ID for one worker dispatch.

    Supervisor dispatch normally contains one task, so its committed attempt
    ID is enough.  Direct batch callers are treated as one workspace
    transaction: a QA failure restores the complete batch snapshot instead of
    restoring one task over another task's changes.
    """
    attempt_ids = sorted(
        {
            task.current_attempt_id
            for task in target_tasks
            if task.current_attempt_id and task.current_attempt_id.strip()
        }
    )
    if not attempt_ids or len(attempt_ids) != len(target_tasks):
        return None
    if snapshots_by_id is not None:
        snapshots = [snapshots_by_id.get(task.current_attempt_id) for task in target_tasks]
        if any(snapshot is None for snapshot in snapshots):
            return None
        batch_ids = {
            snapshot.dispatch_batch_id
            for snapshot in snapshots
            if snapshot is not None and snapshot.dispatch_batch_id
        }
        cluster_ids = {
            snapshot.cluster_id
            for snapshot in snapshots
            if snapshot is not None and snapshot.cluster_id
        }
        if any(snapshot is not None and snapshot.cluster_id for snapshot in snapshots):
            if (
                len(cluster_ids) != 1
                or len(batch_ids) != 1
                or any(
                    snapshot is None or not snapshot.cluster_id or not snapshot.dispatch_batch_id
                    for snapshot in snapshots
                )
            ):
                return None
            return next(iter(batch_ids))
    if len(attempt_ids) == 1:
        return f"attempt-{attempt_ids[0]}"
    digest = hashlib.sha256("\n".join(attempt_ids).encode("utf-8")).hexdigest()[:24]
    return f"batch-{digest}"


def _create_workspace_attempt_snapshot(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
) -> tuple[str | None, list[str]]:
    """Snapshot a committed worker target before it mutates the shared volume.

    A dispatch without a workspace volume cannot create a transaction snapshot;
    callers with a real Phase 5 workspace are rejected earlier when the
    authoritative attempt map is absent or contradictory.
    """
    workspace_volume = state.get("workspace_volume")
    snapshot_id = _workspace_snapshot_id(
        target_tasks,
        state.get("attempt_snapshots_by_id") if "attempt_snapshots_by_id" in state else None,
    )
    if not workspace_volume or snapshot_id is None:
        return None, []

    try:
        with _graph_module().DockerSandbox(
            repo_root=None, workspace_volume=workspace_volume
        ) as sandbox:
            sandbox.create_workspace_snapshot(snapshot_id)
    except Exception as exc:  # noqa: BLE001 - boundary must prevent unsafe execution
        message = f"graph: could not snapshot workspace before attempt {snapshot_id}: {exc}"
        log.exception("Workspace snapshot creation failed for %s.", snapshot_id)
        cleanup_errors = _finish_workspace_attempt_snapshot(
            state,
            snapshot_id,
            restore=False,
        )
        return None, [message, *cleanup_errors]
    return snapshot_id, []


def _finish_workspace_attempt_snapshot(
    state: OrchestratorState,
    snapshot_id: str | None,
    *,
    restore: bool,
) -> list[str]:
    """Restore or delete a worker snapshot while preserving failed restores.

    A successful restore is followed by snapshot deletion. If restoration
    fails, the archive is retained for teardown diagnostics and recovery
    instead of being deleted by the cleanup path that reports the failure.
    """
    workspace_volume = state.get("workspace_volume")
    if not workspace_volume or snapshot_id is None:
        return []

    errors: list[str] = []
    restore_succeeded = not restore
    try:
        with _graph_module().DockerSandbox(
            repo_root=None, workspace_volume=workspace_volume
        ) as sandbox:
            if restore:
                sandbox.restore_workspace_snapshot(snapshot_id)
                restore_succeeded = True
            if restore_succeeded:
                sandbox.remove_workspace_snapshot(snapshot_id)
    except Exception as exc:  # noqa: BLE001 - preserve the original attempt outcome
        action = "remove" if restore_succeeded else "restore"
        message = f"graph: could not {action} workspace snapshot {snapshot_id}: {exc}"
        log.exception("Workspace snapshot %s failed for %s.", action, snapshot_id)
        if restore and not restore_succeeded:
            log.warning(
                "Workspace snapshot %s retained after failed restore for teardown cleanup.",
                snapshot_id,
            )
        errors.append(message)
    return errors


def _restore_workspace_snapshot(
    state: OrchestratorState,
    snapshot_id: str,
) -> list[str]:
    """Restore a retained snapshot without deleting its archive.

    Baseline snapshots remain available across update retries. This helper
    separates restoration from cleanup so a worker failure can return to the
    task baseline while preserving it for the next retry.
    """
    workspace_volume = state.get("workspace_volume")
    if not workspace_volume or not snapshot_id:
        return []

    try:
        with _graph_module().DockerSandbox(
            repo_root=None, workspace_volume=workspace_volume
        ) as sandbox:
            sandbox.restore_workspace_snapshot(snapshot_id)
    except Exception as exc:  # noqa: BLE001 - preserve the original attempt outcome
        message = f"graph: could not restore retained workspace snapshot {snapshot_id}: {exc}"
        log.exception("Retained workspace snapshot restore failed for %s.", snapshot_id)
        return [message]
    return []


def run_delta_isolation_canaries(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    action: MultiPackageAction,
    qa_probe: Callable[[Any, MultiPackageAction], Literal["PASS", "FAIL", "INCONCLUSIVE"]],
) -> dict[str, Any]:
    """Run bounded subset canaries against a shared Docker baseline.

    ``qa_probe`` is an injected deterministic QA adapter. Every invocation is
    restored to the same baseline before and after the committed mutation
    subset, and restore failures are reported as inconclusive.
    """
    task_ids = tuple(sorted(task.task_id for task in target_tasks))
    workspace_volume = state.get("workspace_volume")
    if len(task_ids) <= 1 or not workspace_volume:
        return {
            "status": "INCONCLUSIVE",
            "diagnostic": "delta isolation requires a cluster workspace",
        }
    baseline_id = (
        "delta-baseline-"
        + hashlib.sha256(f"{action.cluster_id}:{','.join(task_ids)}".encode()).hexdigest()[:24]
    )
    snapshots = state.get("attempt_snapshots_by_id") or {}
    snapshot_id = _workspace_snapshot_id(target_tasks, snapshots)
    if snapshot_id is None:
        return {"status": "INCONCLUSIVE", "diagnostic": "cluster attempt metadata is incomplete"}
    try:
        with _graph_module().DockerSandbox(
            repo_root=None, workspace_volume=workspace_volume
        ) as sandbox:
            sandbox.create_workspace_snapshot(baseline_id)

            def probe(subset: tuple[str, ...]) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]:
                mutations = [
                    mutation for mutation in action.package_mutations if mutation.task_id in subset
                ]
                subset_action = action.model_copy(update={"package_mutations": mutations})
                outcome: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "INCONCLUSIVE"
                try:
                    sandbox.restore_workspace_snapshot(baseline_id)
                    touched_files: set[str] = set()
                    applied, _error = apply_multi_package_action(
                        sandbox,
                        subset_action,
                        touched_files,
                    )
                    outcome = "INCONCLUSIVE" if not applied else qa_probe(sandbox, subset_action)
                except Exception:  # noqa: BLE001 - canary infrastructure is inconclusive
                    outcome = "INCONCLUSIVE"
                finally:
                    try:
                        sandbox.restore_workspace_snapshot(baseline_id)
                    except Exception:
                        # A canary that cannot restore its shared baseline is
                        # never safe to attribute. The outer cleanup still
                        # gets a chance to report/remove the baseline archive.
                        outcome = "INCONCLUSIVE"
                return outcome

            isolation = isolate_delta_failure(task_ids, probe)
            try:
                sandbox.restore_workspace_snapshot(baseline_id)
                sandbox.remove_workspace_snapshot(baseline_id)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "INCONCLUSIVE",
                    "diagnostic": f"delta baseline cleanup failed: {exc}",
                    "executions": isolation.executions,
                }
    except Exception as exc:  # noqa: BLE001
        return {"status": "INCONCLUSIVE", "diagnostic": f"delta isolation unavailable: {exc}"}
    return {
        "status": isolation.status,
        "responsible_task_ids": list(isolation.responsible_task_ids),
        "tested_subsets": [list(subset) for subset in isolation.tested_subsets],
        "executions": isolation.executions,
        "diagnostic": isolation.diagnostic,
    }


def _workspace_rollback_anchor_ids(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
) -> list[str]:
    """Return immutable baselines for targets and their workaround parents."""
    anchors = state.get("workspace_rollback_anchors_by_task", {}) or {}
    task_ids: list[str] = []
    for task in target_tasks:
        for task_id in (task.task_id, task.parent_task_id):
            if task_id and task_id not in task_ids:
                task_ids.append(task_id)
    return list(dict.fromkeys(anchors[task_id] for task_id in task_ids if anchors.get(task_id)))


def _finish_workspace_rollback_anchors(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    *,
    restore: bool,
) -> list[str]:
    """Restore or discard immutable task baselines associated with targets."""
    errors: list[str] = []
    for anchor_id in _workspace_rollback_anchor_ids(state, target_tasks):
        errors.extend(_finish_workspace_attempt_snapshot(state, anchor_id, restore=restore))
    return errors


def _finish_all_workspace_rollback_anchors(
    state: OrchestratorState,
    *,
    restore: bool,
) -> list[str]:
    """Restore or discard every retained workspace baseline.

    A successful workaround validates the complete live workspace, including
    the dependency candidate that caused the regression. Any rollback anchor
    retained from before that validation is therefore stale: restoring it
    later would split the validated code edit from its dependency state. The
    caller is responsible for emitting the corresponding empty anchor map
    after this cleanup completes.
    """
    anchors = state.get("workspace_rollback_anchors_by_task", {}) or {}
    errors: list[str] = []
    for anchor_id in dict.fromkeys(anchors.values()):
        errors.extend(_finish_workspace_attempt_snapshot(state, anchor_id, restore=restore))
    return errors


def _restore_retained_workspace_anchors(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
) -> list[str]:
    """Restore immutable task baselines while retaining them for another retry."""
    errors: list[str] = []
    for anchor_id in _workspace_rollback_anchor_ids(state, target_tasks):
        errors.extend(_restore_workspace_snapshot(state, anchor_id))
    return errors


def _parent_workspace_rollback_anchors(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
) -> list[str]:
    """Return retained pre-update snapshots for workaround child tasks."""
    anchors = state.get("workspace_rollback_anchors_by_task", {}) or {}
    return list(
        dict.fromkeys(
            anchor_id
            for task in target_tasks
            if task.parent_task_id
            for anchor_id in [anchors.get(task.parent_task_id)]
            if anchor_id
        )
    )


def _finish_parent_workspace_rollback_anchors(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    *,
    restore: bool,
) -> list[str]:
    """Restore or discard parent-attempt snapshots associated with a child."""
    errors: list[str] = []
    for anchor_id in _parent_workspace_rollback_anchors(state, target_tasks):
        errors.extend(_finish_workspace_attempt_snapshot(state, anchor_id, restore=restore))
    return errors


def _worker_attempts_succeeded(
    result: dict[str, Any],
    target_tasks: list[RemediationTask],
) -> bool:
    """Return whether every committed target produced a validated worker result."""
    if result.get("errors"):
        return False

    worker_results = result.get("worker_results_by_attempt") or {}
    if worker_results:
        by_task = {item.task_id: item for item in worker_results.values()}
        return all(
            (worker_result := by_task.get(task.task_id)) is not None
            and worker_result.status == AgentActionStatus.SUCCESS
            and worker_result.execution_diagnostics.validation_passed
            for task in target_tasks
        )

    summaries = {summary.task_id: summary for summary in result.get("action_summaries", []) or []}
    if not summaries:
        return False
    return all(
        (summary := summaries.get(task.task_id)) is not None
        and summary.status == AgentActionStatus.SUCCESS
        for task in target_tasks
    )


def _has_partial_update_success(
    state: OrchestratorState,
    result: dict[str, Any],
    target_tasks: list[RemediationTask],
) -> bool:
    """Return whether a direct update batch has mixed package outcomes.

    The combined update tool rolls back each failed package independently. A
    direct batch therefore must keep successful package transactions when a
    different package exhausts its retry budget. The normal Supervisor path
    dispatches one task at a time; this exception is restricted to a complete
    update-only batch without retained QA rollback anchors.
    """
    if result.get("errors") or len(target_tasks) < 2 or state.get("active_cluster_id"):
        return False
    if _workspace_rollback_anchor_ids(state, target_tasks):
        return False

    snapshots = state.get("attempt_snapshots_by_id", {}) or {}
    if any(
        task.current_attempt_id is None
        or (snapshot := snapshots.get(task.current_attempt_id)) is None
        or snapshot.dispatch_node != "update_subagent"
        for task in target_tasks
    ):
        return False

    worker_results = result.get("worker_results_by_attempt") or {}
    if not worker_results:
        return False
    by_task = {item.task_id: item for item in worker_results.values()}
    outcomes = [by_task.get(task.task_id) for task in target_tasks]
    if any(outcome is None for outcome in outcomes):
        return False
    succeeded = [
        outcome.status == AgentActionStatus.SUCCESS
        and outcome.execution_diagnostics.validation_passed
        for outcome in outcomes
    ]
    return any(succeeded) and not all(succeeded)


def _finalize_worker_workspace_snapshot(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    snapshot_id: str | None,
    result: dict[str, Any],
) -> list[str]:
    """Keep successful worker archives until QA; restore failed workers."""
    if snapshot_id is None:
        return []
    if _worker_attempts_succeeded(result, target_tasks):
        # QA owns the next decision. It must still be able to restore this
        # exact candidate if install, scanning, or tests reject it.
        return []
    if _has_partial_update_success(state, result, target_tasks):
        # Failed combined transactions have already restored their own
        # package checkpoints. Remove only the outer batch archive so the
        # successful package transactions remain available for later QA.
        return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)
    if _workspace_rollback_anchor_ids(state, target_tasks):
        # A retry may have started from a previously rejected candidate. A
        # failed worker must not preserve that candidate for the next task;
        # restore the immutable task baseline and delete this checkpoint.
        errors = _restore_retained_workspace_anchors(state, target_tasks)
        errors.extend(_finish_workspace_attempt_snapshot(state, snapshot_id, restore=False))
        return errors
    errors = _finish_workspace_attempt_snapshot(
        state,
        snapshot_id,
        restore=True,
    )
    errors.extend(_finish_parent_workspace_rollback_anchors(state, target_tasks, restore=True))
    return errors


def _qa_result_requires_non_remediation_rerun(result: Mapping[str, Any]) -> bool:
    """Return whether QA failed without rejecting the worker candidate."""
    evaluations = result.get("qa_evaluations") or {}
    if not evaluations:
        return False
    return all(
        bool(
            (evaluation.get("contract_error") or evaluation.get("evidence_inconclusive"))
            if isinstance(evaluation, Mapping)
            else (
                getattr(evaluation, "contract_error", False)
                or getattr(evaluation, "evidence_inconclusive", False)
            )
        )
        for evaluation in evaluations.values()
    )


def _finalize_qa_workspace_snapshot(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    snapshot_id: str | None,
    result: dict[str, Any],
) -> list[str]:
    """Finalize the candidate workspace after scoped QA.
    A validated dependency update that fails only because it introduced a
    runtime or test regression is still the correct base for a workaround
    child. Preserve that candidate while handing control back to the
    Supervisor. Other QA failures restore the immutable task baseline when one
    exists, so a rejected retry cannot leak into the next task.
    """
    if snapshot_id is None:
        return []
    attempt_snapshots = state.get("attempt_snapshots_by_id", {})
    candidate_snapshots = [
        attempt_snapshots.get(task.current_attempt_id)
        for task in target_tasks
        if task.current_attempt_id
    ]
    workaround_attempt = (
        bool(target_tasks)
        and len(candidate_snapshots) == len(target_tasks)
        and all(
            snapshot is not None and snapshot.dispatch_node == "workaround_subagent"
            for snapshot in candidate_snapshots
        )
    )

    def restore_failed_workaround() -> list[str]:
        errors = _finish_workspace_attempt_snapshot(state, snapshot_id, restore=True)
        errors.extend(_finish_workspace_rollback_anchors(state, target_tasks, restore=True))
        return errors

    def restore_failed_update() -> list[str]:
        if _workspace_rollback_anchor_ids(state, target_tasks):
            errors = _restore_retained_workspace_anchors(state, target_tasks)
            errors.extend(_finish_workspace_attempt_snapshot(state, snapshot_id, restore=False))
            return errors
        return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=True)

    def discard_task_rollback_anchors() -> list[str]:
        return _finish_workspace_rollback_anchors(state, target_tasks, restore=False)

    if result.get("status") != "qa_completed":
        if _qa_result_requires_non_remediation_rerun(result):
            # QA infrastructure/contract failures must rerun against the
            # candidate rather than restoring the worker's baseline.
            return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)
        return restore_failed_workaround() if workaround_attempt else restore_failed_update()
    evaluations = result.get("qa_evaluations") or {}
    update_candidate = (
        bool(target_tasks)
        and len(candidate_snapshots) == len(target_tasks)
        and all(
            snapshot is not None and snapshot.dispatch_node == "update_subagent"
            for snapshot in candidate_snapshots
        )
    )
    atomic_cluster = bool(
        len(target_tasks) > 1
        and candidate_snapshots
        and all(snapshot is not None and snapshot.cluster_id for snapshot in candidate_snapshots)
    )
    has_regression = False
    has_non_remediation_rerun = False
    has_real_failure = False
    for task in target_tasks:
        evaluation = evaluations.get(task.task_id)
        if evaluation is None:
            return restore_failed_workaround() if workaround_attempt else restore_failed_update()
        if not evaluation.passed:
            if evaluation.evidence_inconclusive or evaluation.contract_error:
                # Keep the worker candidate available for a non-remediation
                # QA rerun; this is not a candidate rejection.
                has_non_remediation_rerun = True
                continue
            has_real_failure = True
            if workaround_attempt:
                return restore_failed_workaround()
            evidence = evaluation.failure_evidence
            is_regression = evaluation.failure_category == FailureCategory.BREAKING_CHANGE or bool(
                evidence and evidence.failed_tests
            )
            if not is_regression:
                return restore_failed_update()
            has_regression = True
    if atomic_cluster:
        if has_real_failure:
            # A cluster is one acceptance unit. No member may retain a partial
            # candidate after any real QA rejection.
            return restore_failed_update()
        if has_non_remediation_rerun:
            return []
        return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)
    if update_candidate and has_regression:
        # The next Supervisor decision may dispatch a workaround child. The
        # child creates its own checkpoint from this retained candidate, while
        # the anchor map continues to point to the first pre-task baseline.
        return []
    if has_regression:
        return restore_failed_workaround()

    if workaround_attempt:
        # The workaround was validated against the live candidate workspace.
        # Promote that exact cumulative state: the parent dependency update and
        # the child source edit must remain together for all later tasks. Any
        # anchor retained before this QA run points at an older workspace and
        # must not be allowed to restore over the promoted candidate.
        errors = _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)
        if not has_non_remediation_rerun:
            errors.extend(_finish_all_workspace_rollback_anchors(state, restore=False))
        return errors

    errors = _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)
    if not has_non_remediation_rerun:
        errors.extend(discard_task_rollback_anchors())
    return errors


def _qa_workspace_rollback_anchor_updates(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    snapshot_id: str | None,
    result: dict[str, Any],
) -> dict[str, str]:
    """Project immutable task baselines and remove resolved snapshot IDs.
    The first regression snapshot becomes the task baseline. Later update
    retries retain that original ID instead of replacing it with the
    pre-retry candidate, which previously allowed a failed dependency change
    to leak into subsequent tasks.
    """
    anchors = dict(state.get("workspace_rollback_anchors_by_task", {}) or {})
    if snapshot_id is None:
        return anchors
    attempt_snapshots = state.get("attempt_snapshots_by_id", {}) or {}
    snapshots = [
        attempt_snapshots.get(task.current_attempt_id)
        for task in target_tasks
        if task.current_attempt_id
    ]
    if (
        not target_tasks
        or len(snapshots) != len(target_tasks)
        or not all(
            snapshot is not None
            and snapshot.dispatch_node in {"update_subagent", "workaround_subagent"}
            for snapshot in snapshots
        )
    ):
        return anchors
    workaround_attempt = all(
        snapshot is not None and snapshot.dispatch_node == "workaround_subagent"
        for snapshot in snapshots
    )
    if result.get("status") != "qa_completed":
        if _qa_result_requires_non_remediation_rerun(result):
            return anchors
        if workaround_attempt:
            for task in target_tasks:
                if task.parent_task_id:
                    anchors.pop(task.parent_task_id, None)
        return anchors
    evaluations = result.get("qa_evaluations") or {}
    workaround_passed = workaround_attempt
    for task in target_tasks:
        evaluation = evaluations.get(task.task_id)
        if evaluation is None:
            workaround_passed = False
            continue
        if workaround_attempt:
            if not evaluation.passed:
                workaround_passed = False
            if task.parent_task_id:
                anchors.pop(task.parent_task_id, None)
            continue
        if evaluation.passed:
            anchors.pop(task.task_id, None)
            continue
        evidence = evaluation.failure_evidence
        if evaluation.failure_category == FailureCategory.BREAKING_CHANGE or bool(
            evidence and evidence.failed_tests
        ):
            # setdefault makes this an immutable task baseline. The current
            # failed candidate remains available to a workaround child, but
            # it is never promoted to the rollback anchor for later retries.
            anchors.setdefault(task.task_id, snapshot_id)
    if workaround_passed:
        # The current workspace is now the authoritative cumulative patch.
        # Clear the complete projection so an unrelated later failure cannot
        # restore a stale pre-workaround dependency state.
        return {}
    return anchors


def run_update_subagent_from_orchestrator(state: OrchestratorState) -> dict[str, Any]:
    """
    Bridge OrchestratorState â†’ SubagentState for the dependency update subagent.

    Normal Supervisor dispatches contain one active task. The bridge retains
    generic target-list handling for direct and future batch callers, resolves
    the associated VulnerabilityGroups, calls ``run_update_subagent_node``, and
    merges results back into the orchestrator state via ``task_queue`` while
    preserving attempt snapshots, task revisions, and typed result correlation.
    """
    task_queue: dict[str, RemediationTask] = state.get("task_queue", {})
    active_task_ids = list(state.get("active_target_task_ids", []))

    group_by_id = {g.group_id: g for g in state.get("valid_groups", [])}
    target_tasks = []
    target_groups = []
    for t_id in active_task_ids:
        task = task_queue.get(t_id)
        if task is not None:
            target_tasks.append(task)
            g = group_by_id.get(task.parent_group_id)
            if g is not None:
                target_groups.append(g)

    if not target_tasks:
        msg = "update_subagent: no valid tasks found for active_target_task_ids."
        log.warning(msg)
        return {"errors": [msg]}

    boundary_rejection = _dispatch_boundary_rejection(
        state,
        target_tasks,
        "update_subagent",
    )
    if boundary_rejection is not None:
        return boundary_rejection

    feedback_by_task = dict(state.get("feedback_by_task", {}))
    attempt_snapshots = dict(state.get("attempt_snapshots_by_id", {}))
    target_attempt_snapshots = {
        task.task_id: attempt_snapshots[task.current_attempt_id]
        for task in target_tasks
        if task.current_attempt_id in attempt_snapshots
    }
    latest_action_summary_by_task: dict[str, str] = {}
    target_attempt_ids = {task.task_id: task.current_attempt_id for task in target_tasks}
    for summary in state.get("action_summaries", []) or []:
        expected_attempt_id = target_attempt_ids.get(summary.task_id)
        if expected_attempt_id and summary.attempt_id != expected_attempt_id:
            continue
        if not expected_attempt_id and summary.attempt_id is not None:
            continue
        latest_action_summary_by_task[summary.task_id] = summary.summary
    subagent_state = initial_update_subagent_state(
        repo_root=state.get("repo_root", ""),
        workspace_volume=state.get("workspace_volume", ""),
        target_tasks=target_tasks,
        target_groups=target_groups,
        constraints_ledger=list(state.get("constraints_ledger", [])),
        feedback_by_task=feedback_by_task,
        previous_action_summaries_by_task=latest_action_summary_by_task,
        retry_diagnostics_by_task=dict(state.get("retry_diagnostics_by_task", {})),
        target_attempt_snapshots=target_attempt_snapshots,
        active_cluster_id=state.get("active_cluster_id"),
        dispatch_batch_id=state.get("active_dispatch_batch_id"),
        multi_package_action=state.get("active_multi_package_action"),
    )

    workspace_snapshot_id, snapshot_errors = _create_workspace_attempt_snapshot(
        state,
        target_tasks,
    )
    if snapshot_errors:
        return {"errors": snapshot_errors}

    try:
        result = _graph_module().run_update_subagent_node(subagent_state)
    except Exception:
        if _workspace_rollback_anchor_ids(state, target_tasks):
            _restore_retained_workspace_anchors(state, target_tasks)
            _finish_workspace_attempt_snapshot(state, workspace_snapshot_id, restore=False)
        else:
            _finish_workspace_attempt_snapshot(
                state,
                workspace_snapshot_id,
                restore=True,
            )
        raise
    out: dict[str, Any] = {
        "errors": result.get("errors", []),
    }
    if result.get("changed_files"):
        out["changed_files"] = result["changed_files"]
    summaries = list(result.get("action_summaries", []) or [])
    if not summaries:
        summary = result.get("action_summary")
        if summary is not None:
            summaries = [summary]
    if summaries:
        out["action_summaries"] = summaries
    worker_results = result.get("worker_results_by_attempt") or {}
    if worker_results:
        out["worker_results_by_attempt"] = dict(worker_results)
    if result.get("retry_diagnostics_by_task"):
        # Keep cumulative version/type evidence available to the Supervisor in
        # addition to the attempt-correlated worker envelope.
        out["retry_diagnostics_by_task"] = result["retry_diagnostics_by_task"]
    out["errors"] = list(out.get("errors", [])) + _finalize_worker_workspace_snapshot(
        state,
        target_tasks,
        workspace_snapshot_id,
        result,
    )
    return out


def run_workaround_subagent_from_orchestrator(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Bridge OrchestratorState â†’ SubagentState for the single-group workaround subagent.

    Takes the first entry of ``active_target_task_ids``, resolves the associated
    VulnerabilityGroup, calls ``run_workaround_subagent_node``, then merges
    results back and updates task_queue.
    """
    task_queue: dict[str, RemediationTask] = state.get("task_queue", {})
    active_task_ids = list(state.get("active_target_task_ids", []))

    if not active_task_ids:
        msg = "workaround_subagent: active_target_task_ids is empty."
        log.warning(msg)
        return {"errors": [msg]}
    if len(active_task_ids) > 1:
        details = "Workaround dispatch accepts exactly one active task."
        msg = f"workaround_subagent: {details}"
        log.warning(msg)
        return {
            "status": "supervisor_routed",
            "next_routing_step": "supervisor",
            "active_target_task_ids": [],
            "errors": [msg],
            "consistency_events": [
                StateConsistencyEvent(
                    error_code="WORKAROUND_BATCH_UNSUPPORTED",
                    action="ignored",
                    details=details,
                )
            ],
        }

    t_id = active_task_ids[0]
    task = task_queue.get(t_id)

    if task is None:
        msg = f"workaround_subagent: could not resolve task '{t_id}'."
        log.warning(msg)
        return {"errors": [msg]}

    boundary_rejection = _dispatch_boundary_rejection(
        state,
        [task],
        "workaround_subagent",
    )
    if boundary_rejection is not None:
        return boundary_rejection

    group_by_id = {g.group_id: g for g in state.get("valid_groups", [])}
    target_group = group_by_id.get(task.parent_group_id)

    if target_group is None:
        msg = f"workaround_subagent: could not resolve group for task '{t_id}'."
        log.warning(msg)
        return {"errors": [msg]}

    feedback_by_task = dict(state.get("feedback_by_task", {}))
    attempt_snapshot = None
    if task.current_attempt_id:
        attempt_snapshot = state.get("attempt_snapshots_by_id", {}).get(task.current_attempt_id)
    current_replay_plan = state.get("workaround_replay_plans_by_task", {}).get(task.task_id)
    subagent_state = initial_workaround_subagent_state(
        repo_root=state.get("repo_root", ""),
        workspace_volume=state.get("workspace_volume", ""),
        target_task=task,
        target_group=target_group,
        constraints_ledger=list(state.get("constraints_ledger", [])),
        previous_feedback=feedback_by_task.get(task.task_id),
        attempt_snapshot=attempt_snapshot,
        current_replay_plan=current_replay_plan,
    )

    workspace_snapshot_id, snapshot_errors = _create_workspace_attempt_snapshot(
        state,
        [task],
    )
    if snapshot_errors:
        return {"errors": snapshot_errors}

    try:
        result = _graph_module().run_workaround_subagent_node(subagent_state)
    except Exception:
        _finish_workspace_attempt_snapshot(
            state,
            workspace_snapshot_id,
            restore=True,
        )
        _finish_parent_workspace_rollback_anchors(state, [task], restore=True)
        raise

    out: dict[str, Any] = {
        "errors": result.get("errors", []),
    }
    if result.get("changed_files"):
        out["changed_files"] = result["changed_files"]
    summaries = list(result.get("action_summaries", []) or [])
    if not summaries:
        summary = result.get("action_summary")
        if summary is not None:
            summaries = [summary]
    if summaries:
        out["action_summaries"] = summaries
    worker_results = result.get("worker_results_by_attempt") or {}
    if worker_results:
        out["worker_results_by_attempt"] = dict(worker_results)
    out["errors"] = list(out.get("errors", [])) + _finalize_worker_workspace_snapshot(
        state,
        [task],
        workspace_snapshot_id,
        result,
    )
    return out


def _maybe_run_delta_isolation(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Run canary attribution when a cluster test failure is ambiguous."""
    if len(target_tasks) <= 1 or state.get("active_multi_package_action") is None:
        return None
    evaluations = result.get("qa_evaluations") or {}
    if not evaluations:
        return None
    if any(
        evaluation.failure_category == FailureCategory.PEER_CONFLICT
        for evaluation in evaluations.values()
    ):
        return None
    gates = [evaluation.deterministic_gates for evaluation in evaluations.values()]
    if any(gate is None or not gate.install_passed for gate in gates):
        return None
    if any(gate.scanner_execution_status.value not in {"success", "skipped"} for gate in gates):
        return None
    if not any(
        gate.tests_passed is False
        and (
            evaluation.test_attribution is None
            or evaluation.test_attribution.verdict.value == "inconclusive"
        )
        for evaluation, gate in zip(evaluations.values(), gates, strict=False)
    ):
        return None

    def qa_probe(
        sandbox: Any, _action: MultiPackageAction
    ) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]:
        install = _qa_test_parsing._run_install(sandbox)
        if not install.ok:
            return "INCONCLUSIVE"
        tests = _qa_test_parsing._run_unit_tests(sandbox)
        return "PASS" if tests.ok else "FAIL"

    isolation = run_delta_isolation_canaries(
        state,
        target_tasks,
        state["active_multi_package_action"],
        qa_probe,
    )
    return isolation


def run_qa_critic_from_orchestrator(state: OrchestratorState) -> dict[str, Any]:
    """
    Run the QA Critic against the current OrchestratorState.

    When ``active_target_task_ids`` is populated, QA is scoped to the
    corresponding VulnerabilityGroups only. The wrapper does NOT re-emit
    ``changed_files`` in its return dict to avoid double-counting via the
    ``operator.add`` reducer.
    """
    task_queue: dict[str, RemediationTask] = state.get("task_queue", {})
    active_task_ids = list(state.get("active_target_task_ids", []))
    missing_task_ids = [task_id for task_id in active_task_ids if task_id not in task_queue]
    if missing_task_ids:
        details = "QA active task IDs are missing from task_queue: " + ", ".join(missing_task_ids)
        return {
            "status": "supervisor_routed",
            "next_routing_step": "supervisor",
            "active_target_task_ids": [],
            "qa_evaluations": {},
            "eval_status": "state_inconsistent",
            "qa_investigation_report": "",
            "errors": [details],
            "consistency_events": [
                StateConsistencyEvent(
                    error_code="QA_ACTIVE_TASK_MISSING",
                    task_id=task_id,
                    action="ignored",
                    details=details,
                )
                for task_id in missing_task_ids
            ],
        }
    target_tasks = [task_queue[task_id] for task_id in active_task_ids]
    boundary_rejection = _dispatch_boundary_rejection(
        state,
        target_tasks,
        "qa_critic",
    )
    if boundary_rejection is not None:
        return {
            **boundary_rejection,
            "qa_evaluations": {},
            "eval_status": "state_inconsistent",
            "qa_investigation_report": "",
        }
    scoped_state = state
    if active_task_ids:
        target_group_ids = {
            task.parent_group_id
            for t_id in active_task_ids
            if (task := task_queue.get(t_id)) is not None
        }
        scoped_groups = [
            group for group in state.get("valid_groups", []) if group.group_id in target_group_ids
        ]
        if not scoped_groups:
            details = (
                "QA active tasks reference parent groups absent from valid_groups: "
                + ", ".join(sorted(target_group_ids))
            )
            return {
                "status": "supervisor_routed",
                "next_routing_step": "supervisor",
                "active_target_task_ids": [],
                "qa_evaluations": {},
                "eval_status": "state_inconsistent",
                "qa_investigation_report": "",
                "errors": [details],
                "consistency_events": [
                    StateConsistencyEvent(
                        error_code="QA_PARENT_GROUP_MISSING",
                        task_id=task_id,
                        action="ignored",
                        details=details,
                    )
                    for task_id in active_task_ids
                ],
            }
        scoped_state = {
            **state,
            "valid_groups": scoped_groups,
        }

    workspace_snapshot_id = _workspace_snapshot_id(
        target_tasks,
        state.get("attempt_snapshots_by_id") if "attempt_snapshots_by_id" in state else None,
    )
    try:
        result = _graph_module().run_qa_critic_node(scoped_state)
    except Exception:
        snapshots = state.get("attempt_snapshots_by_id", {}) or {}
        workaround_attempt = bool(target_tasks) and all(
            (snapshot := snapshots.get(task.current_attempt_id)) is not None
            and snapshot.dispatch_node == "workaround_subagent"
            for task in target_tasks
        )
        if workaround_attempt:
            _finish_workspace_attempt_snapshot(state, workspace_snapshot_id, restore=True)
            _finish_workspace_rollback_anchors(state, target_tasks, restore=True)
        elif _workspace_rollback_anchor_ids(state, target_tasks):
            _restore_retained_workspace_anchors(state, target_tasks)
            _finish_workspace_attempt_snapshot(state, workspace_snapshot_id, restore=False)
        else:
            _finish_workspace_attempt_snapshot(
                state,
                workspace_snapshot_id,
                restore=True,
            )
        raise
    delta_isolation = _maybe_run_delta_isolation(state, target_tasks, result)
    if delta_isolation is not None:
        result = {**result, "delta_isolation": delta_isolation}
        if delta_isolation.get("status") == "INCONCLUSIVE":
            result["qa_evaluations"] = {
                task_id: (
                    evaluation.model_copy(
                        update={
                            "evidence_inconclusive": True,
                            "retry_feedback": (
                                f"Delta-isolation QA was inconclusive: "
                                f"{delta_isolation.get('diagnostic', 'unknown reason')}."
                            ),
                        }
                    )
                    if not evaluation.passed
                    else evaluation
                )
                for task_id, evaluation in (result.get("qa_evaluations") or {}).items()
            }
    snapshot_cleanup_errors = _finalize_qa_workspace_snapshot(
        state,
        target_tasks,
        workspace_snapshot_id,
        result,
    )
    rollback_anchor_updates = _qa_workspace_rollback_anchor_updates(
        state,
        target_tasks,
        workspace_snapshot_id,
        result,
    )
    scan_evidence = result.get("scan_evidence")
    scan_was_skipped = bool(result.get("scan_skipped"))
    attempt_scan_is_authoritative = not scan_was_skipped and (
        scan_evidence is None or bool(getattr(scan_evidence, "authoritative", False))
    )
    scan_status = (
        result.get("new_vulnerability_status", state.get("new_vulnerability_status", "not_scanned"))
        if attempt_scan_is_authoritative
        else state.get("new_vulnerability_status", "not_scanned")
    )
    scan_snapshot_available = attempt_scan_is_authoritative and (
        "post_remediation_scan_issues" in result or "post_remediation_scan_issues" in state
    )
    settings = _graph_module().get_runtime_settings()
    disable_retriage = settings.remedy_disable_post_qa_triage
    triage_required = (
        not disable_retriage
        and attempt_scan_is_authoritative
        and result.get("status") in {"qa_completed", "qa_failed"}
        and scan_status in {"none", "detected"}
        and scan_snapshot_available
    )
    out: dict[str, Any] = {
        "qa_evaluations": result.get("qa_evaluations", {}),
        "eval_status": result.get("eval_status", ""),
        "qa_investigation_report": result.get("qa_investigation_report", ""),
        "baseline_scan_identifiers": result.get(
            "baseline_scan_identifiers",
            state.get("baseline_scan_identifiers", []),
        ),
        "post_remediation_scan_identifiers": (
            result.get(
                "post_remediation_scan_identifiers",
                state.get("post_remediation_scan_identifiers", []),
            )
            if attempt_scan_is_authoritative
            else state.get("post_remediation_scan_identifiers", [])
        ),
        "post_remediation_scan_issues": (
            result.get(
                "post_remediation_scan_issues", state.get("post_remediation_scan_issues", [])
            )
            if attempt_scan_is_authoritative
            else state.get("post_remediation_scan_issues", [])
        ),
        "new_vulnerability_identifiers": (
            result.get(
                "new_vulnerability_identifiers", state.get("new_vulnerability_identifiers", [])
            )
            if attempt_scan_is_authoritative
            else state.get("new_vulnerability_identifiers", [])
        ),
        "new_vulnerability_status": (
            result.get(
                "new_vulnerability_status", state.get("new_vulnerability_status", "not_scanned")
            )
            if attempt_scan_is_authoritative
            else state.get("new_vulnerability_status", "not_scanned")
        ),
        "triage_required": triage_required,
        "status": result.get("status", "qa_completed"),
        "errors": list(result.get("errors", []) or []) + snapshot_cleanup_errors,
    }
    if delta_isolation is not None:
        out["delta_isolation_by_cluster"] = {
            str(state.get("active_cluster_id") or "unknown"): delta_isolation
        }
        if delta_isolation.get("status") == "IDENTIFIED":
            out["portfolio_escalation"] = {
                "reason": "DELTA_ISOLATION_ATTRIBUTION",
                "forced_singleton_task_ids": list(delta_isolation.get("responsible_task_ids", [])),
            }
            out["portfolio_dirty"] = True
            out["portfolio_plan"] = None
    qa_results_by_attempt: dict[str, QAAttemptResult] = {}
    task_queue = state.get("task_queue", {})
    evaluations = result.get("qa_evaluations", {}) or {}
    reports_by_task = result.get("qa_investigation_reports_by_task", {}) or {}
    errors_by_task = result.get("qa_errors_by_task", {}) or {}
    snapshots = state.get("attempt_snapshots_by_id") or {}
    scan_evidence_by_task: dict[str, Any] = {}
    for task_id in active_task_ids:
        task = task_queue[task_id]
        attempt_id = task.current_attempt_id
        if not attempt_id:
            raise ValueError(f"qa_critic: active task {task_id} has no current attempt")
        evaluation = evaluations.get(task_id)
        if evaluation is None:
            raise ValueError(f"qa_critic: missing evaluation for task {task_id}")
        if evaluation.task_id != task_id:
            raise ValueError(
                f"qa_critic: evaluation task identity mismatch for {task_id}: {evaluation.task_id}"
            )
        attempt_snapshot = snapshots.get(attempt_id)
        if attempt_snapshot is None:
            raise ValueError(f"qa_critic: missing attempt snapshot for task {task_id}")
        attempt_policy = (
            attempt_snapshot.get("qa_policy")
            if isinstance(attempt_snapshot, Mapping)
            else attempt_snapshot.qa_policy
        )
        if attempt_policy is None or task.qa_policy != attempt_policy:
            raise ValueError(f"qa_critic: missing or contradictory QA policy for task {task_id}")
        qa_results_by_attempt[attempt_id] = QAAttemptResult(
            attempt_id=attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            cluster_id=attempt_snapshot.cluster_id,
            dispatch_batch_id=attempt_snapshot.dispatch_batch_id,
            action_digest=attempt_snapshot.action_digest,
            qa_policy=attempt_policy,
            qa_policy_source="attempt_snapshot",
            evaluation=evaluation,
            investigation_report=reports_by_task.get(task_id, ""),
            errors=list(errors_by_task.get(task_id, []) or []),
        )
        if evaluation.scan_evidence is not None:
            scan_evidence_by_task[task_id] = evaluation.scan_evidence
    if qa_results_by_attempt:
        out["qa_results_by_attempt"] = qa_results_by_attempt
    if scan_evidence_by_task:
        out["scan_evidence_by_task"] = scan_evidence_by_task
    # Emit the complete projection so the replace reducer can clear anchors
    # after a successful update or an abandoned workaround.
    out["workspace_rollback_anchors_by_task"] = rollback_anchor_updates
    return out
