"""
graph.py - LangGraph remediation orchestrator for the current Phase 5 runtime.

Phase 5 graph topology (hub-and-spoke)
---------------------------------------
::

    START
      |
    initial_triage (one preprocessing pass)
      | triage_completed / skipped -> workspace_builder
      | failed | no_work -> teardown
    workspace_builder
      | workspace_ready / failed -> teardown
    supervisor  <-----------------------------------+
      |                                            |
      +-> update_subagent ----------------------->-+
      |                                            |
      +-> workaround_subagent ------------------->-+
      |                                            |
      +-> qa_critic ------------------------------>+
      |                                            |
      +-> triage (post-QA reconciliation) -------->+
      |                                            |
      +-> final_full_scan ------------------------>+
      |
      +-> teardown
           |
          END

Public API:
``build_orchestrator_graph()``
``orchestrator_engine``
``run_orchestrator(...)``

Typed request/result models are exposed from ``remediation_engine.api`` so
callers do not need to construct LangGraph state directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    IssueSource,
    QAAttemptResult,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
    SystemContext,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.langsmith_config import (
    build_phase5_runnable_config,
    resolve_phase5_trace_url,
)
from remediation_engine.orchestration.qa_critic import (
    _group_target_identifiers,
    run_final_full_scan_node,
    run_qa_critic_node,
)
from remediation_engine.orchestration.state import (
    OrchestratorState,
    initial_orchestrator_state,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
    normalize_group_paths,
)
from remediation_engine.orchestration.supervisor_node import (
    _instruction_digest,
    run_supervisor_node,
    supervisor_router,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.orchestration.teardown_node import run_teardown_node
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    export_phase5_trajectory,
    invoke_with_trajectory,
    use_trajectory_recorder,
)
from remediation_engine.orchestration.update_subagent import run_update_subagent_node
from remediation_engine.orchestration.workaround_subagent import run_workaround_subagent_node
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.triage.pipeline import run_triage_pipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 5 triage node and routing
# ---------------------------------------------------------------------------


def triage_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the one-time preprocessing triage pass."""
    # ``valid_groups`` is the explicit compatibility contract for callers
    # that already performed preprocessing (for example, the cached-group
    # Juice Shop driver).  Do not silently replace that caller-selected
    # scope with a second full triage pass.
    if state.get("valid_groups"):
        log.info("triage_node: pre-triaged groups supplied; skipping preprocessing triage.")
        return {
            "status": "triage_skipped",
            "initial_triage_status": "skipped_preprocessed_groups",
            "initial_triage_executed": False,
        }

    issues = state.get("issues")
    system_context = state.get("system_context")
    repo_root = state.get("repo_root")

    if not issues or not system_context:
        log.info("triage_node: issues or system_context not found, skipping triage.")
        return {
            "status": "triage_skipped",
            "initial_triage_status": "skipped_missing_input",
            "initial_triage_executed": False,
        }

    log.info("triage_node: running triage on %d issues.", len(issues))

    try:
        results = invoke_with_trajectory(
            "triage.pipeline",
            lambda: run_triage_pipeline(issues, system_context, repo_root),
            {
                "issue_count": len(issues),
                "repo_root": repo_root,
            },
            run_type="chain",
        )
        valid_groups = [group for group, result in results if result.is_valid]
        valid_groups = normalize_group_paths(valid_groups, repo_root)
        log.info("triage_node: produced %d valid groups.", len(valid_groups))

        if not valid_groups:
            return {
                "valid_groups": [],
                "status": "triage_completed_no_work",
                "initial_triage_status": "completed_no_work",
                "initial_triage_executed": True,
            }

        return {
            "valid_groups": valid_groups,
            "status": "triage_completed",
            "initial_triage_status": "completed",
            "initial_triage_executed": True,
        }
    except Exception as exc:
        log.exception("triage_node: triage pipeline raised")
        return {
            "valid_groups": [],
            "status": "failed",
            "initial_triage_status": "failed",
            "initial_triage_executed": True,
            "errors": [f"triage_node raised: {exc}"],
        }


def _stable_issue_fingerprint(issue: VulnerabilityIssue) -> str:
    """Return an issue fingerprint that ignores generated ingestion metadata."""
    payload = issue.model_dump(mode="json")
    payload.pop("id", None)
    payload.pop("ingested_at", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _stable_group_fingerprint(group: VulnerabilityGroup) -> str:
    """Compare meaningful group content while ignoring volatile triage metadata."""
    payload = {
        "group_id": group.group_id,
        "issue_type": group.issue_type.value,
        "vulnerable_component": group.vulnerable_component,
        "file_path": group.file_path,
        "file_paths": sorted(group.file_paths or []),
        "cve_ids": sorted(group.cve_ids or []),
        "ghsa_ids": sorted(group.ghsa_ids or []),
        "versions": sorted(group.versions or []),
        "sources": sorted(source.value for source in (group.sources or [])),
        "issues": sorted(_stable_issue_fingerprint(issue) for issue in (group.issues or [])),
        "fix_plan": group.fix_plan.model_dump(mode="json") if group.fix_plan else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _post_triage_issue_input(
    state: OrchestratorState,
) -> list[VulnerabilityIssue] | None:
    """Build the current issue universe from the parseable QA scan snapshot."""
    if "post_remediation_scan_issues" not in state:
        return None

    post_scan_issues = list(state.get("post_remediation_scan_issues") or [])
    baseline_issues = list(state.get("issues") or [])
    retained_non_odc = [issue for issue in baseline_issues if issue.source != IssueSource.ODC]

    # Skip-triage callers may omit the full initial issue set.  Preserve any
    # non-ODC findings carried by the supplied groups in that compatibility
    # mode, while the post-remediation ODC snapshot remains authoritative for
    # dependency findings.
    if not baseline_issues:
        seen_issue_fingerprints = {_stable_issue_fingerprint(issue) for issue in retained_non_odc}
        for group in state.get("valid_groups", []) or []:
            for issue in group.issues or []:
                if issue.source == IssueSource.ODC:
                    continue
                fingerprint = _stable_issue_fingerprint(issue)
                if fingerprint not in seen_issue_fingerprints:
                    retained_non_odc.append(issue)
                    seen_issue_fingerprints.add(fingerprint)

    return retained_non_odc + post_scan_issues


def _reconcile_triaged_groups(
    state: OrchestratorState,
    candidate_groups: list[VulnerabilityGroup],
) -> tuple[list[VulnerabilityGroup], dict[str, list[str]]]:
    """Reuse unchanged groups and retain active removed groups for QA handoff."""
    previous_groups = list(state.get("valid_groups", []) or [])
    previous_by_id = {group.group_id: group for group in previous_groups}
    task_queue = state.get("task_queue", {}) or {}
    active_task_ids = set(state.get("active_target_task_ids", []) or [])

    reused: list[str] = []
    changed: list[str] = []
    added: list[str] = []
    reappeared: list[str] = []
    result: list[VulnerabilityGroup] = []
    candidate_ids = {group.group_id for group in candidate_groups}

    for candidate in candidate_groups:
        previous = previous_by_id.get(candidate.group_id)
        if previous is not None:
            if _stable_group_fingerprint(previous) == _stable_group_fingerprint(candidate):
                result.append(previous)
                reused.append(candidate.group_id)
            else:
                result.append(candidate)
                changed.append(candidate.group_id)
            continue

        existing_task = next(
            (task for task in task_queue.values() if task.parent_group_id == candidate.group_id),
            None,
        )
        result.append(candidate)
        if existing_task is not None:
            reappeared.append(candidate.group_id)
        else:
            added.append(candidate.group_id)

    retained: list[str] = []
    for previous in previous_groups:
        if previous.group_id in candidate_ids:
            continue
        matching_tasks = [
            (task_id, task)
            for task_id, task in task_queue.items()
            if task.parent_group_id == previous.group_id
        ]
        if any(
            task_id in active_task_ids
            or task.status not in {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
            for task_id, task in matching_tasks
        ):
            result.append(previous)
            retained.append(previous.group_id)

    reconciliation = {
        "reused_group_ids": sorted(reused),
        "changed_group_ids": sorted(changed),
        "new_group_ids": sorted(added),
        "reappeared_group_ids": sorted(reappeared),
        "retained_removed_group_ids": sorted(retained),
        "removed_group_ids": sorted(set(previous_by_id) - candidate_ids - set(retained)),
    }
    return sorted(result, key=lambda group: group.group_id), reconciliation


def post_qa_triage_node(state: OrchestratorState) -> dict[str, Any]:
    """Re-triage the complete parseable post-remediation scan snapshot."""
    disable_retriage = os.environ.get("REMEDY_DISABLE_POST_QA_TRIAGE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("REMEDY_DISABLE_RETRIAGE", "").lower() in ("1", "true", "yes")
    if disable_retriage or not state.get("triage_required"):
        return {
            "status": "triage_skipped",
            "triage_reconciliation": {},
            "active_target_task_ids": [],
            "active_target_group_ids": [],
        }

    if state.get("new_vulnerability_status") == "scan_failed":
        log.info("post_qa_triage_node: scan failed; preserving current groups.")
        return {
            "status": "triage_skipped",
            "triage_required": False,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
            "active_target_group_ids": [],
        }

    issues = _post_triage_issue_input(state)
    if issues is None:
        return {
            "status": "triage_skipped",
            "triage_required": False,
            "triage_reconciliation": {},
            "active_target_task_ids": [],
            "active_target_group_ids": [],
        }

    system_context = state.get("system_context") or SystemContext()
    repo_root = state.get("repo_root")
    log.info("post_qa_triage_node: re-triaging %d current issues.", len(issues))

    try:
        results = run_triage_pipeline(issues, system_context, repo_root)
        candidate_groups = [group for group, triage_result in results if triage_result.is_valid]
        valid_groups, reconciliation = _reconcile_triaged_groups(
            state,
            candidate_groups,
        )
        task_queue = dict(state.get("task_queue", {}) or {})
        changed_group_ids = set(reconciliation["changed_group_ids"])
        changed_group_ids.update(reconciliation["reappeared_group_ids"])
        changed_group_ids.update(reconciliation["new_group_ids"])
        final_scan = state.get("final_full_scan_result")
        final_scan_identifiers = set(
            (
                final_scan.get("remaining_target_identifiers", [])
                if isinstance(final_scan, dict)
                else getattr(final_scan, "remaining_target_identifiers", [])
            )
            or []
        )
        final_scan_reopened_group_ids = sorted(
            group.group_id
            for group in state.get("valid_groups", []) or []
            if _group_target_identifiers(group) & final_scan_identifiers
        )
        changed_group_ids.update(final_scan_reopened_group_ids)
        if final_scan_reopened_group_ids:
            reconciliation["final_scan_reopened_group_ids"] = final_scan_reopened_group_ids
        groups_by_id = {group.group_id: group for group in valid_groups}
        reopened_task_ids: set[str] = set()
        preserved_unfixable_task_ids: set[str] = set()
        prior_qa_evaluations = dict(state.get("qa_evaluations", {}) or {})
        for task_id, task in list(task_queue.items()):
            if task.parent_group_id not in changed_group_ids:
                continue
            if task.status == TaskStatus.UNFIXABLE:
                preserved_unfixable_task_ids.add(task_id)
                continue
            group = groups_by_id.get(task.parent_group_id)
            if group is None:
                continue
            fresh_task = build_initial_remediation_task(group, task_id)
            task_queue[task_id] = task.model_copy(
                update={
                    "task_revision": task.task_revision + 1,
                    "current_attempt_id": None,
                    "strategy": fresh_task.strategy,
                    "strategy_stage": fresh_task.strategy_stage,
                    "selected_version": fresh_task.selected_version,
                    "exhausted_update_path": False,
                    "instruction": fresh_task.instruction,
                    "status": TaskStatus.PENDING,
                    "retry_count": 0,
                }
            )
            reopened_task_ids.add(task_id)

        # A previous QA result belongs to the pre-retriage task revision. It
        # remains available in the attempt history, but must not close the
        # newly reopened task.
        qa_evaluations = {
            key: value
            for key, value in prior_qa_evaluations.items()
            if key not in reopened_task_ids
            and not any(
                task.parent_group_id in changed_group_ids and key == task.parent_group_id
                for task in task_queue.values()
            )
        }
        log.info(
            "post_qa_triage_node: produced %d valid groups (%d reused, %d new, %d changed).",
            len(valid_groups),
            len(reconciliation["reused_group_ids"]),
            len(reconciliation["new_group_ids"]),
            len(reconciliation["changed_group_ids"]),
        )
        work_reopened = bool(
            reopened_task_ids
            or reconciliation["new_group_ids"]
            or reconciliation["reappeared_group_ids"]
        )
        if preserved_unfixable_task_ids:
            reconciliation["preserved_unfixable_task_ids"] = sorted(preserved_unfixable_task_ids)
        return {
            "valid_groups": valid_groups,
            "status": "triage_completed" if valid_groups else "triage_completed_no_work",
            "triage_required": False,
            "triage_reconciliation": reconciliation,
            "task_queue": task_queue,
            "qa_evaluations": qa_evaluations,
            "active_target_task_ids": [],
            "active_target_group_ids": [],
            "final_full_scan_completed": False
            if work_reopened
            else state.get("final_full_scan_completed", False),
            "final_full_scan_result": None
            if work_reopened
            else state.get("final_full_scan_result"),
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("post_qa_triage_node: triage pipeline raised")
        return {
            "status": "triage_failed",
            "triage_required": False,
            "triage_reconciliation": {},
            "errors": [f"post_qa_triage_node raised: {exc}"],
        }


def route_after_triage(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the triage node."""
    status = state.get("status")
    if status in ("triage_completed_no_work", "failed"):
        return "teardown"
    return "workspace_builder"


# ---------------------------------------------------------------------------
# Phase 5 wrapper nodes
# ---------------------------------------------------------------------------


def _tag_attempt_summaries(
    summaries: list[AgentActionSummary],
    target_tasks: list[RemediationTask],
    snapshots: dict[str, Any],
) -> list[AgentActionSummary]:
    """Attach the committed attempt identity to compatibility summaries.

    Some worker exit paths return an ordinary ``AgentActionSummary`` while
    their structured ``WorkerAttemptResult`` is correctly tagged.  Re-emitting
    the ordinary summary would create a second, uncorrelated source of truth
    in the graph state.  The bridge therefore tags summaries only from the
    committed snapshot for that target task.
    """
    task_by_id = {task.task_id: task for task in target_tasks}
    tagged: list[AgentActionSummary] = []
    for summary in summaries:
        task = task_by_id.get(summary.task_id)
        snapshot = (
            snapshots.get(task.current_attempt_id) or snapshots.get(task.task_id)
            if task is not None and task.current_attempt_id
            else None
        )
        if snapshot is not None and summary.attempt_id is None:
            summary = summary.model_copy(
                update={
                    "attempt_id": snapshot.attempt_id,
                    "task_revision": snapshot.task_revision,
                    "instruction_digest": snapshot.instruction_digest,
                }
            )
        tagged.append(summary)
    return tagged


def _dispatch_boundary_rejection(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    expected_node: str,
) -> dict[str, Any] | None:
    """Reject a worker/QA invocation whose input is not a committed snapshot.

    Direct legacy bridge callers do not carry ``attempt_snapshots_by_id`` and
    remain supported.  Every Phase 5 supervisor-produced state does carry the
    field, so real graph execution is strict: no worker or QA node may run
    without a matching task revision, instruction, and snapshot.
    """
    if "attempt_snapshots_by_id" not in state:
        return None

    snapshots = state.get("attempt_snapshots_by_id") or {}
    errors: list[str] = []
    events: list[StateConsistencyEvent] = []
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
        elif (
            snapshot.task_id != task.task_id
            or snapshot.task_revision != task.task_revision
            or snapshot.strategy_stage != task.strategy_stage
            or snapshot.no_fix_stage != task.no_fix_stage
            or snapshot.selected_version != task.selected_version
            or snapshot.instruction != task.instruction
            or snapshot.instruction_digest != _instruction_digest(task.instruction)
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
        "active_target_group_ids": [],
        "errors": errors,
        "consistency_events": events,
    }


def _workspace_snapshot_id(target_tasks: list[RemediationTask]) -> str | None:
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
    if len(attempt_ids) == 1:
        return f"attempt-{attempt_ids[0]}"
    digest = hashlib.sha256("\n".join(attempt_ids).encode("utf-8")).hexdigest()[:24]
    return f"batch-{digest}"


def _create_workspace_attempt_snapshot(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
) -> tuple[str | None, list[str]]:
    """Snapshot a committed worker target before it mutates the shared volume.

    Legacy bridge callers without committed attempts, and calls without a
    workspace volume, retain their pre-transaction behavior.  A real Phase 5
    dispatch with a workspace is fail-closed if snapshot creation fails.
    """
    workspace_volume = state.get("workspace_volume")
    if "attempt_snapshots_by_id" not in state:
        return None, []
    snapshot_id = _workspace_snapshot_id(target_tasks)
    if not workspace_volume or snapshot_id is None:
        return None, []

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
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
    """Restore or delete a worker snapshot and always attempt cleanup."""
    workspace_volume = state.get("workspace_volume")
    if not workspace_volume or snapshot_id is None:
        return []

    errors: list[str] = []
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            try:
                if restore:
                    sandbox.restore_workspace_snapshot(snapshot_id)
            finally:
                # Do not leave a failed restore archive in the shared volume;
                # teardown also performs a final best-effort sweep.
                sandbox.remove_workspace_snapshot(snapshot_id)
    except Exception as exc:  # noqa: BLE001 - preserve the original attempt outcome
        action = "restore" if restore else "remove"
        message = f"graph: could not {action} workspace snapshot {snapshot_id}: {exc}"
        log.exception("Workspace snapshot %s failed for %s.", action, snapshot_id)
        errors.append(message)
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
        # QA owns the next decision.  It must still be able to restore this
        # exact candidate if install, scanning, or tests reject it.
        return []
    return _finish_workspace_attempt_snapshot(
        state,
        snapshot_id,
        restore=True,
    )


def _finalize_qa_workspace_snapshot(
    state: OrchestratorState,
    target_tasks: list[RemediationTask],
    snapshot_id: str | None,
    result: dict[str, Any],
) -> list[str]:
    """Keep a candidate only when all scoped QA evaluations pass."""
    if snapshot_id is None:
        return []
    if result.get("status") != "qa_completed":
        return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=True)

    evaluations = result.get("qa_evaluations") or {}
    for task in target_tasks:
        evaluation = evaluations.get(task.parent_group_id) or evaluations.get(task.task_id)
        if evaluation is None or not evaluation.passed:
            return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=True)
    return _finish_workspace_attempt_snapshot(state, snapshot_id, restore=False)


def _ensure_worker_attempt_results(
    result: dict[str, Any],
    target_tasks: list[RemediationTask],
    snapshots: dict[str, Any],
) -> dict[str, WorkerAttemptResult]:
    """Normalize worker compatibility output into attempt-tagged envelopes."""
    existing = result.get("worker_results_by_attempt") or {}
    if existing:
        return dict(existing)
    summaries = list(result.get("action_summaries", []) or [])
    summary_by_task = {summary.task_id: summary for summary in summaries}
    errors = list(result.get("errors", []) or [])
    output: dict[str, WorkerAttemptResult] = {}
    for task in target_tasks:
        snapshot = (
            snapshots.get(task.current_attempt_id) or snapshots.get(task.task_id)
            if task.current_attempt_id
            else None
        )
        if snapshot is None:
            continue
        summary = summary_by_task.get(task.task_id)
        status = summary.status if summary is not None else AgentActionStatus.SURRENDER
        output[snapshot.attempt_id] = WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=task.task_id,
            task_revision=snapshot.task_revision,
            status=status,
            action_summary=summary,
            execution_diagnostics=WorkerExecutionDiagnostics(
                validation_passed=status == AgentActionStatus.SUCCESS,
                failure_reason=" | ".join(errors),
            ),
            instruction_digest=snapshot.instruction_digest,
            errors=errors,
        )
    return output


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

    # Fall back to active_target_group_ids for backward compat during migration
    if not active_task_ids:
        active_task_ids = list(state.get("active_target_group_ids", []))

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
    )

    workspace_snapshot_id, snapshot_errors = _create_workspace_attempt_snapshot(
        state,
        target_tasks,
    )
    if snapshot_errors:
        return {"errors": snapshot_errors}

    try:
        result = run_update_subagent_node(subagent_state)
    except Exception:
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
    summaries = _tag_attempt_summaries(
        list(result.get("action_summaries", [])),
        target_tasks,
        target_attempt_snapshots,
    )
    if not summaries:
        summary = result.get("action_summary")
        if summary is not None:
            summaries = [summary]
    if summaries:
        out["action_summaries"] = summaries
    worker_results = _ensure_worker_attempt_results(
        {**result, "action_summaries": summaries},
        target_tasks,
        attempt_snapshots,
    )
    if worker_results:
        out["worker_results_by_attempt"] = worker_results
    elif result.get("retry_diagnostics_by_task") and not target_attempt_snapshots:
        # Compatibility for legacy direct bridge callers that do not provide
        # attempt envelopes. Real Phase 5 dispatches always use envelopes.
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

    # Fall back to active_target_group_ids for backward compat
    if not active_task_ids:
        active_task_ids = list(state.get("active_target_group_ids", []))

    if not active_task_ids:
        msg = "workaround_subagent: active_target_task_ids is empty."
        log.warning(msg)
        return {"errors": [msg]}

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
        result = run_workaround_subagent_node(subagent_state)
    except Exception:
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
    target_snapshots = {
        task.current_attempt_id: attempt_snapshot
        for task in [task]
        if attempt_snapshot is not None and task.current_attempt_id
    }
    summaries = _tag_attempt_summaries(
        list(result.get("action_summaries", [])),
        [task],
        target_snapshots,
    )
    if not summaries:
        summary = result.get("action_summary")
        if summary is not None:
            summaries = [summary]
    if summaries:
        out["action_summaries"] = summaries
    worker_results = _ensure_worker_attempt_results(
        {**result, "action_summaries": summaries},
        [task],
        target_snapshots,
    )
    if worker_results:
        out["worker_results_by_attempt"] = worker_results
    out["errors"] = list(out.get("errors", [])) + _finalize_worker_workspace_snapshot(
        state,
        [task],
        workspace_snapshot_id,
        result,
    )
    return out


def run_qa_critic_from_orchestrator(state: OrchestratorState) -> dict[str, Any]:
    """
    Run the QA Critic against the current OrchestratorState.

    When ``active_target_task_ids`` is populated, QA is scoped to the
    corresponding VulnerabilityGroups only. The wrapper does NOT re-emit
    ``changed_files`` in its return dict to avoid double-counting via the
    ``operator.add`` reducer.
    """
    task_queue: dict[str, RemediationTask] = state.get("task_queue", {})
    active_task_ids = set(state.get("active_target_task_ids", []))

    # Fall back to active_target_group_ids
    if not active_task_ids:
        active_task_ids = set(state.get("active_target_group_ids", []))

    target_tasks = [state.get("task_queue", {}).get(task_id) for task_id in active_task_ids]
    target_tasks = [task for task in target_tasks if task is not None]
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
        # Resolve parent group IDs from tasks; fallback treats IDs as group IDs
        target_group_ids: set[str] = set()
        for t_id in active_task_ids:
            task = task_queue.get(t_id)
            if task is not None:
                target_group_ids.add(task.parent_group_id)
            else:
                target_group_ids.add(t_id)  # fallback: treat as group ID
        scoped_groups = [
            group for group in state.get("valid_groups", []) if group.group_id in target_group_ids
        ]
        if scoped_groups:
            scoped_state = {
                **state,
                "valid_groups": scoped_groups,
            }

    workspace_snapshot_id = (
        _workspace_snapshot_id(target_tasks) if "attempt_snapshots_by_id" in state else None
    )
    try:
        result = run_qa_critic_node(scoped_state)
    except Exception:
        _finish_workspace_attempt_snapshot(
            state,
            workspace_snapshot_id,
            restore=True,
        )
        raise
    snapshot_cleanup_errors = _finalize_qa_workspace_snapshot(
        state,
        target_tasks,
        workspace_snapshot_id,
        result,
    )
    scan_evidence = result.get("scan_evidence")
    attempt_scan_is_authoritative = scan_evidence is None or bool(
        getattr(scan_evidence, "authoritative", False)
    )
    scan_status = (
        result.get("new_vulnerability_status", state.get("new_vulnerability_status", "not_scanned"))
        if attempt_scan_is_authoritative
        else state.get("new_vulnerability_status", "not_scanned")
    )
    scan_snapshot_available = attempt_scan_is_authoritative and (
        "post_remediation_scan_issues" in result or "post_remediation_scan_issues" in state
    )
    disable_retriage = os.environ.get("REMEDY_DISABLE_POST_QA_TRIAGE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("REMEDY_DISABLE_RETRIAGE", "").lower() in ("1", "true", "yes")
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
    qa_results_by_attempt: dict[str, QAAttemptResult] = {}
    task_queue = state.get("task_queue", {})
    evaluations = result.get("qa_evaluations", {}) or {}
    for task_id in active_task_ids:
        task = task_queue.get(task_id)
        if task is None or not task.current_attempt_id:
            continue
        evaluation = evaluations.get(task.parent_group_id) or evaluations.get(task_id)
        if evaluation is None:
            continue
        if evaluation.task_id != task_id:
            # QA workers may still return the legacy group-keyed projection.
            # The attempt envelope is task-keyed, so normalize the nested
            # evaluation before it enters the authoritative correlation map.
            evaluation = evaluation.model_copy(update={"task_id": task_id})
        qa_results_by_attempt[task.current_attempt_id] = QAAttemptResult(
            attempt_id=task.current_attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            evaluation=evaluation,
            investigation_report=result.get("qa_investigation_report", ""),
            errors=list(result.get("errors", []) or []),
        )
    if qa_results_by_attempt:
        out["qa_results_by_attempt"] = qa_results_by_attempt
    if scan_evidence is not None:
        out["scan_evidence_by_task"] = {task_id: scan_evidence for task_id in active_task_ids}
    return out


# ---------------------------------------------------------------------------
# Phase 5 routing
# ---------------------------------------------------------------------------


def route_after_workspace_builder(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the workspace builder node."""
    if state.get("status") == "workspace_ready":
        return "supervisor"
    return "teardown"


# ---------------------------------------------------------------------------
# Phase 5 graph construction
# ---------------------------------------------------------------------------


def build_orchestrator_graph():
    """Compile and return the Phase 5 orchestrator StateGraph."""
    workflow = StateGraph(OrchestratorState)

    # ``initial_triage`` is the one preprocessing pass.  The node named
    # ``triage`` is reserved for Supervisor-dispatched post-QA re-triage.
    workflow.add_node("initial_triage", triage_node)
    workflow.add_node("triage", post_qa_triage_node)
    workflow.add_node("workspace_builder", run_workspace_builder_node)
    workflow.add_node("supervisor", run_supervisor_node)
    workflow.add_node("update_subagent", run_update_subagent_from_orchestrator)
    workflow.add_node("workaround_subagent", run_workaround_subagent_from_orchestrator)
    workflow.add_node("qa_critic", run_qa_critic_from_orchestrator)
    workflow.add_node("final_full_scan", run_final_full_scan_node)
    workflow.add_node("teardown", run_teardown_node)

    workflow.add_edge(START, "initial_triage")
    workflow.add_conditional_edges("initial_triage", route_after_triage)
    workflow.add_conditional_edges("workspace_builder", route_after_workspace_builder)
    workflow.add_conditional_edges("supervisor", supervisor_router)
    workflow.add_edge("update_subagent", "supervisor")
    workflow.add_edge("workaround_subagent", "supervisor")
    workflow.add_edge("qa_critic", "supervisor")
    workflow.add_edge("triage", "supervisor")
    workflow.add_edge("final_full_scan", "supervisor")
    workflow.add_edge("teardown", END)

    return workflow.compile()


orchestrator_engine = build_orchestrator_graph()


def run_orchestrator(
    repo_root: str,
    valid_groups: list[VulnerabilityGroup],
    issues: list[VulnerabilityIssue] | None = None,
    system_context: SystemContext | None = None,
) -> OrchestratorState:
    """Convenience entry point for the Phase 5 orchestrator graph."""
    initial_state = initial_orchestrator_state(
        repo_root=repo_root,
        valid_groups=valid_groups,
        issues=issues,
        system_context=system_context,
    )
    config, run_id = build_phase5_runnable_config(repo_root, valid_groups)
    recorder = TrajectoryRecorder()
    langsmith_enabled = config is not None and run_id is not None
    trace_id = run_id if run_id is not None else uuid.uuid4()
    if config is None:
        runnable_config: dict[str, Any] = {
            "run_id": trace_id,
            "run_name": "phase5_orchestrator_local",
            "tags": ["phase-5", "orchestrator", "langgraph", "local-trajectory"],
            "metadata": {
                "repo_name": Path(repo_root).name,
                "repo_root": repo_root,
                "vulnerability_group_count": len(valid_groups),
            },
            "callbacks": [recorder],
        }
    else:
        runnable_config = config
        runnable_config["callbacks"] = list(config.get("callbacks") or []) + [recorder]

    recorder.record_manual(
        name="phase5.root_input",
        run_type="state",
        inputs=initial_state,
    )
    result: OrchestratorState | None = None
    run_error: BaseException | None = None
    trace_url: str | None = None
    try:
        with use_trajectory_recorder(recorder):
            result = orchestrator_engine.invoke(initial_state, runnable_config)
        if langsmith_enabled and run_id is not None:
            result["langsmith_run_id"] = str(run_id)
            trace_url = resolve_phase5_trace_url(run_id)
            if trace_url:
                result["langsmith_trace_url"] = trace_url
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        recorder.record_manual(
            name="phase5.root_output",
            run_type="state",
            inputs={"error": str(run_error)} if run_error else None,
            outputs=result
            if result is not None
            else {"error": str(run_error) if run_error else "no result"},
            error=run_error,
        )
        try:
            trajectory_path = export_phase5_trajectory(
                trace_id=trace_id,
                repo_root=repo_root,
                initial_state=initial_state,
                final_state=result
                if result is not None
                else {"error": str(run_error) if run_error else "no result"},
                recorder=recorder,
                langsmith_enabled=langsmith_enabled,
                langsmith_url=trace_url,
                run_error=run_error,
            )
            if result is not None:
                result["trajectory_path"] = str(trajectory_path)
        except Exception as export_error:  # noqa: BLE001 - never mask remediation
            log.warning("run_orchestrator: trajectory export failed: %s", export_error)
            if result is not None:
                result.setdefault("errors", []).append(f"trajectory export failed: {export_error}")

    log.info(
        "run_orchestrator: repo_root=%s groups=%d final_status=%s",
        repo_root,
        len(result.get("valid_groups", [])) if result is not None else 0,
        result.get("status") if result is not None else "failed",
    )
    return result  # type: ignore[return-value]
