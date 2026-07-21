"""
graph.py - LangGraph remediation orchestrators for Phase 4.1 and Phase 5.

Phase 4.1 graph topology
------------------------
::

    START
      |
    locator_node
      | (confidence > 0)   / (confidence == 0 -> status="failed")
    planner_node                END
      | SCA + version_found  / SAST | no_fix | workaround_found
    edit_request_builder_node           END
      | edit_request built   / status="planned_manual_edit_required"
    editor_node                         END
      |
    END

Phase 5 graph topology (hub-and-spoke)
---------------------------------------
::

    START
      |
    triage
      | triage_completed / failed | no_work -> teardown
    workspace_builder
      | workspace_ready / failed -> teardown
    supervisor  <-----------------------------------+
      |                                            |
      +-> update_subagent ----------------------->-+
      |                                            |
      +-> workaround_subagent ------------------->-+
      |                                            |
      +-> qa_critic ------------------------------>+
      |
      +-> teardown
           |
          END

Public API
----------
Phase 4.1:
``build_remediation_graph()``
``remediation_engine``
``run_remediation(...)``

Phase 5:
``build_orchestrator_graph()``
``orchestrator_engine``
``run_orchestrator(...)``
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    QAAttemptResult,
    EditStatus,
    FixPlan,
    FixPlanStatus,
    IssueType,
    LocalizedIssue,
    RemediationTask,
    RoutingStrategy,
    StateConsistencyEvent,
    SystemContext,
    TaskStatus,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.edit_request_builder import build_edit_request
from src.orchestrator.langsmith_config import (
    build_phase5_runnable_config,
    resolve_phase5_trace_url,
)
from src.orchestrator.qa_critic import run_qa_critic_node
from src.orchestrator.state import (
    OrchestratorState,
    RemediationState,
    initial_orchestrator_state,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from src.orchestrator.supervisor_node import (
    _instruction_digest,
    run_supervisor_node,
    supervisor_router,
)
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.trajectory_exporter import (
    TrajectoryRecorder,
    export_phase5_trajectory,
    use_trajectory_recorder,
)
from src.orchestrator.update_subagent import run_update_subagent_node
from src.orchestrator.workaround_subagent import run_workaround_subagent_node
from src.tools.edit_tools import apply_edit
from src.triage.pipeline import run_triage_pipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 4.1 node implementations
# ---------------------------------------------------------------------------


def locator_node(state: RemediationState) -> Dict[str, Any]:
    """Locate the finding within the repository."""
    issue = state["issue"]
    repo_root = state["repo_root"]

    log.info(
        "locator_node: issue=%s type=%s file=%s",
        issue.id,
        issue.issue_type,
        issue.file_path,
    )

    try:
        if issue.issue_type == IssueType.SCA:
            from src.tools.manifest_locator import locate_from_issue

            localized: LocalizedIssue = locate_from_issue(issue, Path(repo_root))
        else:
            from src.tools.code_locator import locate_sast

            localized = locate_sast(issue, repo_root)
    except Exception as exc:  # pragma: no cover
        log.exception("locator_node: unexpected error")
        return {
            "localized_issue": None,
            "status": "failed",
            "errors": [f"locator_node raised: {exc}"],
        }

    if localized.localization_confidence == 0.0:
        msg = (
            f"locator_node: zero confidence for issue={issue.id} "
            f"file={issue.file_path}"
        )
        log.warning(msg)
        return {
            "localized_issue": localized,
            "status": "failed",
            "errors": [msg],
        }

    log.info(
        "locator_node: confidence=%.2f manifest=%s symbol=%s",
        localized.localization_confidence,
        localized.manifest_file,
        localized.enclosing_symbol,
    )
    return {"localized_issue": localized, "status": "located"}


def planner_node(state: RemediationState) -> Dict[str, Any]:
    """Run the fix-planner waterfall for SCA issues."""
    issue = state["issue"]
    localized = state.get("localized_issue")

    if issue.issue_type == IssueType.SAST:
        log.info(
            "planner_node: SAST issue=%s - deferring to Phase 4.2 remedy agent",
            issue.id,
        )
        return {"status": "localized_needs_remedy_agent"}

    if localized is None:
        return {
            "status": "failed",
            "errors": ["planner_node: localized_issue is None"],
        }

    log.info("planner_node: running fix waterfall for issue=%s", issue.id)
    try:
        from src.tools.fix_planner import plan_fix

        raw_plan: dict = plan_fix(localized)
        fix_plan = FixPlan(**raw_plan)
    except Exception as exc:  # pragma: no cover
        log.exception("planner_node: fix_planner raised")
        return {
            "fix_plan": None,
            "status": "failed",
            "errors": [f"planner_node raised: {exc}"],
        }

    log.info(
        "planner_node: status=%s strategy=%s fixed_version=%s",
        fix_plan.status,
        fix_plan.strategy_used,
        fix_plan.fixed_version,
    )

    if fix_plan.status == FixPlanStatus.VERSION_FOUND:
        return {"fix_plan": fix_plan, "status": "planned_version_found"}

    if fix_plan.status == FixPlanStatus.WORKAROUND_FOUND:
        return {"fix_plan": fix_plan, "status": "planned_workaround_found"}

    return {"fix_plan": fix_plan, "status": "planned_no_auto_edit"}


def edit_request_builder_node(state: RemediationState) -> Dict[str, Any]:
    """Build an EditRequest from the localized issue and fix plan."""
    localized = state.get("localized_issue")
    fix_plan = state.get("fix_plan")
    repo_root = state["repo_root"]
    dry_run: bool = state.get("dry_run", True)

    if localized is None or fix_plan is None:
        return {
            "status": "failed",
            "errors": ["edit_request_builder_node: missing localized_issue or fix_plan"],
        }

    edit_request, reason = build_edit_request(
        localized, fix_plan, repo_root, dry_run=dry_run
    )

    if edit_request is None:
        log.warning("edit_request_builder_node: %s", reason)
        return {
            "edit_request": None,
            "status": "planned_manual_edit_required",
            "errors": [reason or "edit_request_builder_node: could not build EditRequest"],
        }

    log.info(
        "edit_request_builder_node: built EditRequest for %s dry_run=%s",
        edit_request.file_path,
        edit_request.dry_run,
    )
    return {"edit_request": edit_request, "status": "edit_request_ready"}


def editor_node(state: RemediationState) -> Dict[str, Any]:
    """Apply or dry-run the Phase 4.1 EditRequest."""
    edit_request = state.get("edit_request")

    if edit_request is None:
        log.debug("editor_node: no edit_request - skipping")
        return {}

    log.info(
        "editor_node: applying edit to %s (dry_run=%s)",
        edit_request.file_path,
        edit_request.dry_run,
    )

    try:
        edit_result = apply_edit(edit_request)
    except Exception as exc:  # pragma: no cover
        log.exception("editor_node: apply_edit raised")
        return {
            "status": "failed",
            "errors": [f"editor_node raised: {exc}"],
        }

    log.info("editor_node: edit_result.status=%s", edit_result.status)

    status_map = {
        EditStatus.APPLIED: "edited",
        EditStatus.DRY_RUN: "dry_run",
        EditStatus.REJECTED: "failed",
        EditStatus.ERROR: "failed",
    }
    graph_status = status_map.get(edit_result.status, "failed")

    update: Dict[str, Any] = {"edit_result": edit_result, "status": graph_status}

    if graph_status == "failed":
        update["errors"] = [
            f"editor_node: edit {edit_result.status.value} - "
            f"{edit_result.rejection_reason or 'no reason'}"
        ]

    return update


# ---------------------------------------------------------------------------
# Phase 4.1 routing
# ---------------------------------------------------------------------------


def _route_after_locator(state: RemediationState) -> str:
    """Route: locator -> planner or END."""
    if state.get("status") == "failed":
        return END
    return "planner"


def _route_after_planner(state: RemediationState) -> str:
    """Route: planner -> edit_request_builder or END."""
    if state.get("status", "") == "planned_version_found":
        return "edit_request_builder"
    return END


def _route_after_edit_request_builder(state: RemediationState) -> str:
    """Route: edit_request_builder -> editor or END."""
    if state.get("status") == "edit_request_ready":
        return "editor"
    return END


# ---------------------------------------------------------------------------
# Phase 4.1 graph construction
# ---------------------------------------------------------------------------


def build_remediation_graph():
    """Compile and return the Phase 4.1 remediation StateGraph."""
    builder = StateGraph(RemediationState)

    builder.add_node("locator", locator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("edit_request_builder", edit_request_builder_node)
    builder.add_node("editor", editor_node)

    builder.add_edge(START, "locator")
    builder.add_conditional_edges("locator", _route_after_locator)
    builder.add_conditional_edges("planner", _route_after_planner)
    builder.add_conditional_edges(
        "edit_request_builder",
        _route_after_edit_request_builder,
    )
    builder.add_edge("editor", END)

    return builder.compile()


remediation_engine = build_remediation_graph()


def run_remediation(
    issue,
    repo_root: str,
    *,
    dry_run: bool = True,
) -> RemediationState:
    """Convenience entry point for the Phase 4.1 graph."""
    initial_state: RemediationState = {
        "issue": issue,
        "repo_root": repo_root,
        "dry_run": dry_run,
        "status": "pending",
        "errors": [],
    }
    result: RemediationState = remediation_engine.invoke(initial_state)
    log.info(
        "run_remediation: issue=%s final_status=%s",
        issue.id,
        result.get("status"),
    )
    return result


# ---------------------------------------------------------------------------
# Phase 5 triage node and routing
# ---------------------------------------------------------------------------


def triage_node(state: OrchestratorState) -> Dict[str, Any]:
    """Run the Phase 4 triage pipeline."""
    issues = state.get("issues")
    system_context = state.get("system_context")
    repo_root = state.get("repo_root")

    if not issues or not system_context:
        log.info("triage_node: issues or system_context not found, skipping triage.")
        return {"status": "triage_skipped"}

    log.info("triage_node: running triage on %d issues.", len(issues))

    try:
        results = run_triage_pipeline(issues, system_context, repo_root)
        valid_groups = [group for group, result in results if result.is_valid]
        log.info("triage_node: produced %d valid groups.", len(valid_groups))

        if not valid_groups:
            return {"valid_groups": [], "status": "triage_completed_no_work"}

        return {"valid_groups": valid_groups, "status": "triage_completed"}
    except Exception as exc:
        log.exception("triage_node: triage pipeline raised")
        return {
            "valid_groups": [],
            "status": "failed",
            "errors": [f"triage_node raised: {exc}"],
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
    summaries: List[AgentActionSummary],
    target_tasks: List[RemediationTask],
    snapshots: Dict[str, Any],
) -> List[AgentActionSummary]:
    """Attach the committed attempt identity to compatibility summaries.

    Some worker exit paths return an ordinary ``AgentActionSummary`` while
    their structured ``WorkerAttemptResult`` is correctly tagged.  Re-emitting
    the ordinary summary would create a second, uncorrelated source of truth
    in the graph state.  The bridge therefore tags summaries only from the
    committed snapshot for that target task.
    """
    task_by_id = {task.task_id: task for task in target_tasks}
    tagged: List[AgentActionSummary] = []
    for summary in summaries:
        task = task_by_id.get(summary.task_id)
        snapshot = (
            snapshots.get(task.current_attempt_id)
            or snapshots.get(task.task_id)
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
    target_tasks: List[RemediationTask],
    expected_node: str,
) -> Optional[Dict[str, Any]]:
    """Reject a worker/QA invocation whose input is not a committed snapshot.

    Direct legacy bridge callers do not carry ``attempt_snapshots_by_id`` and
    remain supported.  Every Phase 5 supervisor-produced state does carry the
    field, so real graph execution is strict: no worker or QA node may run
    without a matching task revision, instruction, and snapshot.
    """
    if "attempt_snapshots_by_id" not in state:
        return None

    snapshots = state.get("attempt_snapshots_by_id") or {}
    errors: List[str] = []
    events: List[StateConsistencyEvent] = []
    for task in target_tasks:
        attempt_id = task.current_attempt_id
        snapshot = snapshots.get(attempt_id) if attempt_id else None
        error_code: Optional[str] = None
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
        elif expected_node in {"update_subagent", "workaround_subagent"} and snapshot.dispatch_node != expected_node:
            error_code = "DISPATCH_NODE_MISMATCH"
            details = (
                f"Snapshot was committed for {snapshot.dispatch_node}, "
                f"not {expected_node}."
            )
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


def _ensure_worker_attempt_results(
    result: Dict[str, Any],
    target_tasks: List[RemediationTask],
    snapshots: Dict[str, Any],
) -> Dict[str, WorkerAttemptResult]:
    """Normalize worker compatibility output into attempt-tagged envelopes."""
    existing = result.get("worker_results_by_attempt") or {}
    if existing:
        return dict(existing)
    summaries = list(result.get("action_summaries", []) or [])
    summary_by_task = {summary.task_id: summary for summary in summaries}
    errors = list(result.get("errors", []) or [])
    output: Dict[str, WorkerAttemptResult] = {}
    for task in target_tasks:
        snapshot = (
            snapshots.get(task.current_attempt_id)
            or snapshots.get(task.task_id)
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


def run_update_subagent_from_orchestrator(state: OrchestratorState) -> Dict[str, Any]:
    """
    Bridge OrchestratorState → SubagentState for the batch dependency update subagent.

    Reads ``active_target_task_ids`` from state to select the target tasks,
    resolves the associated VulnerabilityGroups, calls ``run_update_subagent_node``,
    then merges results back into the orchestrator state via task_queue.
    """
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
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
    latest_action_summary_by_task: Dict[str, str] = {}
    target_attempt_ids = {
        task.task_id: task.current_attempt_id for task in target_tasks
    }
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

    result = run_update_subagent_node(subagent_state)

    out: Dict[str, Any] = {
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
    return out


def run_workaround_subagent_from_orchestrator(
    state: OrchestratorState,
) -> Dict[str, Any]:
    """
    Bridge OrchestratorState → SubagentState for the single-group workaround subagent.

    Takes the first entry of ``active_target_task_ids``, resolves the associated
    VulnerabilityGroup, calls ``run_workaround_subagent_node``, then merges
    results back and updates task_queue.
    """
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
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
        attempt_snapshot = state.get("attempt_snapshots_by_id", {}).get(
            task.current_attempt_id
        )
    subagent_state = initial_workaround_subagent_state(
        repo_root=state.get("repo_root", ""),
        workspace_volume=state.get("workspace_volume", ""),
        target_task=task,
        target_group=target_group,
        constraints_ledger=list(state.get("constraints_ledger", [])),
        previous_feedback=feedback_by_task.get(task.task_id),
        attempt_snapshot=attempt_snapshot,
    )

    result = run_workaround_subagent_node(subagent_state)

    out: Dict[str, Any] = {
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
    return out


def run_qa_critic_from_orchestrator(state: OrchestratorState) -> Dict[str, Any]:
    """
    Run the QA Critic against the current OrchestratorState.

    When ``active_target_task_ids`` is populated, QA is scoped to the
    corresponding VulnerabilityGroups only. The wrapper does NOT re-emit
    ``changed_files`` in its return dict to avoid double-counting via the
    ``operator.add`` reducer.
    """
    task_queue: Dict[str, RemediationTask] = state.get("task_queue", {})
    active_task_ids = set(state.get("active_target_task_ids", []))

    # Fall back to active_target_group_ids
    if not active_task_ids:
        active_task_ids = set(state.get("active_target_group_ids", []))

    target_tasks = [
        state.get("task_queue", {}).get(task_id)
        for task_id in active_task_ids
    ]
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

    result = run_qa_critic_node(scoped_state)
    out: Dict[str, Any] = {
        "qa_evaluations": result.get("qa_evaluations", {}),
        "eval_status": result.get("eval_status", ""),
        "qa_investigation_report": result.get("qa_investigation_report", ""),
        "status": result.get("status", "qa_completed"),
        "errors": result.get("errors", []),
    }
    qa_results_by_attempt: Dict[str, QAAttemptResult] = {}
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

    workflow.add_node("triage", triage_node)
    workflow.add_node("workspace_builder", run_workspace_builder_node)
    workflow.add_node("supervisor", run_supervisor_node)
    workflow.add_node("update_subagent", run_update_subagent_from_orchestrator)
    workflow.add_node("workaround_subagent", run_workaround_subagent_from_orchestrator)
    workflow.add_node("qa_critic", run_qa_critic_from_orchestrator)
    workflow.add_node("teardown", run_teardown_node)

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", route_after_triage)
    workflow.add_conditional_edges("workspace_builder", route_after_workspace_builder)
    workflow.add_conditional_edges("supervisor", supervisor_router)
    workflow.add_edge("update_subagent", "supervisor")
    workflow.add_edge("workaround_subagent", "supervisor")
    workflow.add_edge("qa_critic", "supervisor")
    workflow.add_edge("teardown", END)

    return workflow.compile()


orchestrator_engine = build_orchestrator_graph()


def run_orchestrator(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    issues: Optional[List[VulnerabilityIssue]] = None,
    system_context: Optional[SystemContext] = None,
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
        runnable_config: Dict[str, Any] = {
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
    result: Optional[OrchestratorState] = None
    run_error: Optional[BaseException] = None
    trace_url: Optional[str] = None
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
            outputs=result if result is not None else {"error": str(run_error) if run_error else "no result"},
            error=run_error,
        )
        try:
            trajectory_path = export_phase5_trajectory(
                trace_id=trace_id,
                repo_root=repo_root,
                initial_state=initial_state,
                final_state=result if result is not None else {"error": str(run_error) if run_error else "no result"},
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
                result.setdefault("errors", []).append(
                    f"trajectory export failed: {export_error}"
                )

    log.info(
        "run_orchestrator: repo_root=%s groups=%d final_status=%s",
        repo_root,
        len(result.get("valid_groups", [])) if result is not None else 0,
        result.get("status") if result is not None else "failed",
    )
    return result  # type: ignore[return-value]
