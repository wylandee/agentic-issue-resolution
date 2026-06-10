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

Phase 5 graph topology
----------------------
::

    START
      |
    workspace_builder
      | workspace_ready
      v
    remedy_agent
      | edits_completed
      v
    workspace_sync
      | dependencies_ready + SCA version bump
      |------------------------------> scanner
      | dependencies_ready without SCA version bump
      /------------------------------> tester

    workspace_sync
      \\ dependency_sync_failed -----> remedy_agent

    scanner
      | scanned
      |-----------------------> tester
      / scan_failed ----------> remedy_agent

    tester
      | tested
      |-----------------------> teardown -> END
      / test_failed ----------> remedy_agent

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
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from src.contracts.schemas import (
    EditStatus,
    FixPlan,
    FixPlanStatus,
    IssueType,
    LocalizedIssue,
    VulnerabilityGroup,
)
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.edit_request_builder import build_edit_request
from src.orchestrator.remedy_agent import run_remedy_agent
from src.orchestrator.scanner_node import run_scanner_node
from src.orchestrator.state import (
    DEFAULT_MAX_RETRIES,
    OrchestratorState,
    RemediationState,
    initial_orchestrator_state,
)
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.tester_node import run_tester_node
from src.orchestrator.workspace_sync_node import run_workspace_sync_node
from src.tools.edit_tools import apply_edit

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
# Phase 5 routing
# ---------------------------------------------------------------------------


def route_after_workspace_builder(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the workspace builder node."""
    if state.get("status") == "workspace_ready":
        return "remedy_agent"
    return "teardown"


def route_after_remedy_agent(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the Remedy Agent."""
    if state.get("status") == "edits_completed":
        return "workspace_sync"
    return "teardown"


def route_after_workspace_sync(state: OrchestratorState) -> str:
    """Route Phase 5 flow after dependency sync."""
    status = state.get("status")
    if status == "dependency_sync_failed":
        return "remedy_agent"
    if status != "dependencies_ready":
        return "teardown"

    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    requires_scan = any(
        group.issue_type == IssueType.SCA
        and group.fix_plan is not None
        and group.fix_plan.status == FixPlanStatus.VERSION_FOUND
        for group in valid_groups
    )
    if requires_scan:
        return "scanner"
    return "tester"


def route_after_scanner(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the scanner node."""
    status = state.get("status")
    if status == "scanned":
        return "tester"
    if status == "scan_failed":
        return "remedy_agent"
    return "teardown"


def route_after_tester(state: OrchestratorState) -> str:
    """Route Phase 5 flow after the tester node."""
    status = state.get("status")
    if status == "tested":
        return "teardown"
    if status == "test_failed":
        return "remedy_agent"
    return "teardown"


# ---------------------------------------------------------------------------
# Phase 5 graph construction
# ---------------------------------------------------------------------------


def build_orchestrator_graph():
    """Compile and return the Phase 5 orchestrator StateGraph."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("workspace_builder", run_workspace_builder_node)
    workflow.add_node("remedy_agent", run_remedy_agent)
    workflow.add_node("workspace_sync", run_workspace_sync_node)
    workflow.add_node("scanner", run_scanner_node)
    workflow.add_node("tester", run_tester_node)
    workflow.add_node("teardown", run_teardown_node)

    workflow.add_edge(START, "workspace_builder")
    workflow.add_conditional_edges("workspace_builder", route_after_workspace_builder)
    workflow.add_conditional_edges("remedy_agent", route_after_remedy_agent)
    workflow.add_conditional_edges("workspace_sync", route_after_workspace_sync)
    workflow.add_conditional_edges("scanner", route_after_scanner)
    workflow.add_conditional_edges("tester", route_after_tester)
    workflow.add_edge("teardown", END)

    return workflow.compile()


orchestrator_engine = build_orchestrator_graph()


def run_orchestrator(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OrchestratorState:
    """Convenience entry point for the Phase 5 orchestrator graph."""
    initial_state = initial_orchestrator_state(
        repo_root=repo_root,
        valid_groups=valid_groups,
        max_retries=max_retries,
    )
    result: OrchestratorState = orchestrator_engine.invoke(initial_state)
    log.info(
        "run_orchestrator: repo_root=%s groups=%d final_status=%s",
        repo_root,
        len(valid_groups),
        result.get("status"),
    )
    return result
