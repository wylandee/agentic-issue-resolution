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
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from src.contracts.schemas import (
    AgentActionStatus,
    EditStatus,
    FixPlan,
    FixPlanStatus,
    GroupRemediationStatus,
    IssueType,
    LocalizedIssue,
    SystemContext,
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
from src.orchestrator.supervisor_node import run_supervisor_node, supervisor_router
from src.orchestrator.teardown_node import run_teardown_node
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


def run_update_subagent_from_orchestrator(state: OrchestratorState) -> Dict[str, Any]:
    """
    Bridge OrchestratorState → SubagentState for the batch dependency update subagent.

    Reads ``active_target_group_ids`` from state to select the target groups,
    calls ``run_update_subagent_node``, then merges results back into the
    orchestrator state and updates ``group_statuses``.
    """
    active_ids = set(state.get("active_target_group_ids", []))
    target_groups = [
        g for g in state.get("valid_groups", []) if g.group_id in active_ids
    ]

    if not target_groups:
        msg = "update_subagent: no valid groups found for active_target_group_ids."
        log.warning(msg)
        return {"errors": [msg]}

    subagent_state = initial_update_subagent_state(
        repo_root=state.get("repo_root", ""),
        workspace_volume=state.get("workspace_volume", ""),
        target_groups=target_groups,
        constraints_ledger=list(state.get("constraints_ledger", [])),
        feedback_by_group=dict(state.get("feedback_by_group", {})),
    )

    result = run_update_subagent_node(subagent_state)

    from src.contracts.schemas import AgentActionSummary  # local import avoids cycle
    summary = result.get("action_summary")
    succeeded = (
        isinstance(summary, AgentActionSummary)
        and summary.status == AgentActionStatus.SUCCESS
    )
    new_statuses = {
        g.group_id: (
            GroupRemediationStatus.OPTIMISTICALLY_FIXED
            if succeeded
            else GroupRemediationStatus.NEEDS_RETRY
        )
        for g in target_groups
    }

    out: Dict[str, Any] = {
        "group_statuses": new_statuses,
        "errors": result.get("errors", []),
    }
    if result.get("changed_files"):
        out["changed_files"] = result["changed_files"]
    if summary is not None:
        out["action_summaries"] = [summary]
    return out


def run_workaround_subagent_from_orchestrator(
    state: OrchestratorState,
) -> Dict[str, Any]:
    """
    Bridge OrchestratorState → SubagentState for the single-group workaround subagent.

    Takes the first entry of ``active_target_group_ids``, builds a
    ``SubagentState``, calls ``run_workaround_subagent_node``, then merges
    results back and updates ``group_statuses``.
    """
    active_ids = list(state.get("active_target_group_ids", []))
    if not active_ids:
        msg = "workaround_subagent: active_target_group_ids is empty."
        log.warning(msg)
        return {"errors": [msg]}

    group_id = active_ids[0]
    group_by_id = {g.group_id: g for g in state.get("valid_groups", [])}
    target_group = group_by_id.get(group_id)

    if target_group is None:
        msg = f"workaround_subagent: group '{group_id}' not found in valid_groups."
        log.warning(msg)
        return {"errors": [msg]}

    feedback_by_group = dict(state.get("feedback_by_group", {}))
    subagent_state = initial_workaround_subagent_state(
        repo_root=state.get("repo_root", ""),
        workspace_volume=state.get("workspace_volume", ""),
        target_group=target_group,
        constraints_ledger=list(state.get("constraints_ledger", [])),
        previous_feedback=feedback_by_group.get(group_id),
    )

    result = run_workaround_subagent_node(subagent_state)

    from src.contracts.schemas import AgentActionSummary  # local import avoids cycle
    summary = result.get("action_summary")
    succeeded = (
        isinstance(summary, AgentActionSummary)
        and summary.status == AgentActionStatus.SUCCESS
    )
    new_statuses = {
        group_id: (
            GroupRemediationStatus.OPTIMISTICALLY_FIXED
            if succeeded
            else GroupRemediationStatus.NEEDS_RETRY
        )
    }

    out: Dict[str, Any] = {
        "group_statuses": new_statuses,
        "errors": result.get("errors", []),
    }
    if result.get("changed_files"):
        out["changed_files"] = result["changed_files"]
    if summary is not None:
        out["action_summaries"] = [summary]
    return out


def run_qa_critic_from_orchestrator(state: OrchestratorState) -> Dict[str, Any]:
    """
    Run the QA Critic against the current OrchestratorState.

    ``changed_files`` accumulated in state are passed transparently to the
    QA Critic (it reads them from state).  The wrapper does NOT re-emit
    ``changed_files`` in its return dict to avoid double-counting via the
    ``operator.add`` reducer.
    """
    result = run_qa_critic_node(state)
    return {
        "qa_evaluations": result.get("qa_evaluations", {}),
        "eval_status": result.get("eval_status", ""),
        "status": result.get("status", "qa_completed"),
        "errors": result.get("errors", []),
    }


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
    if config is None:
        result: OrchestratorState = orchestrator_engine.invoke(initial_state)
    else:
        result = orchestrator_engine.invoke(initial_state, config)
        result["langsmith_run_id"] = str(run_id)
        trace_url = resolve_phase5_trace_url(run_id)
        if trace_url:
            result["langsmith_trace_url"] = trace_url

    log.info(
        "run_orchestrator: repo_root=%s groups=%d final_status=%s",
        repo_root,
        len(result.get("valid_groups", [])),
        result.get("status"),
    )
    return result
