"""
graph.py — Phase 4.1 LangGraph remediation orchestrator.

Graph topology
--------------
::

    START
      ↓
    locator_node
      ↓ (confidence > 0)  ↘ (confidence == 0 → status="failed")
    planner_node                END
      ↓ SCA + version_found  ↘ SAST | no_fix | workaround_found
    edit_request_builder_node           END
      ↓ edit_request built   ↘ status="planned_manual_edit_required"
    editor_node                         END
      ↓
    END

Design constraints
------------------
* All nodes are pure functions: they receive state, return a state *update* dict.
* No LLM calls in Phase 4.1 — all routing is deterministic.
* Default execution is dry-run (``dry_run=True``) to protect the workspace.
* Logging, not printing.

Public API
----------
``build_remediation_graph() -> CompiledGraph``
``remediation_engine``  — module-level compiled graph (import and call directly)
``run_remediation(issue, repo_root, dry_run) -> RemediationState``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.contracts.schemas import (
    EditStatus,
    FixPlan,
    FixPlanStatus,
    IssueType,
    LocalizedIssue,
)
from src.orchestrator.edit_request_builder import build_edit_request
from src.orchestrator.state import RemediationState
from src.tools.edit_tools import apply_edit

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def locator_node(state: RemediationState) -> Dict[str, Any]:
    """Locate the finding within the repository.

    * SCA → ``manifest_locator.locate_from_issue``
    * SAST → ``code_locator.locate_sast``

    Returns
    -------
    State update with ``localized_issue`` and ``status``.
    Appends to ``errors`` on failure; sets ``status="failed"`` when confidence is 0.
    """
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
            from src.tools.manifest_locator import locate_from_issue  # lazy import

            localized: LocalizedIssue = locate_from_issue(issue, Path(repo_root))
        else:  # SAST
            from src.tools.code_locator import locate_sast  # lazy import

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
    """Run the fix-planner waterfall (SCA only).

    SAST issues skip planning and set ``status="localized_needs_remedy_agent"`` to
    signal that a Phase 4.2 LLM Remedy agent must produce the ``EditRequest``.

    For SCA:
    * ``version_found`` → continue to edit_request_builder_node
    * ``workaround_found`` / ``no_fix`` → set status and stop

    Returns
    -------
    State update with ``fix_plan`` (SCA only) and ``status``.
    """
    issue = state["issue"]
    localized = state.get("localized_issue")

    if issue.issue_type == IssueType.SAST:
        log.info(
            "planner_node: SAST issue=%s — deferring to Phase 4.2 remedy agent",
            issue.id,
        )
        return {"status": "localized_needs_remedy_agent"}

    if localized is None:  # should never happen if graph wiring is correct
        return {
            "status": "failed",
            "errors": ["planner_node: localized_issue is None"],
        }

    log.info("planner_node: running fix waterfall for issue=%s", issue.id)
    try:
        from src.tools.fix_planner import plan_fix  # lazy import

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

    # NO_FIX
    return {"fix_plan": fix_plan, "status": "planned_no_auto_edit"}


def edit_request_builder_node(state: RemediationState) -> Dict[str, Any]:
    """Deterministically build an ``EditRequest`` from the localized issue + fix plan.

    Only direct-dependency version bumps are supported in Phase 4.1.  Transitive
    overrides and other cases result in ``status="planned_manual_edit_required"``.
    """
    localized = state.get("localized_issue")
    fix_plan = state.get("fix_plan")
    repo_root = state["repo_root"]
    dry_run: bool = state.get("dry_run", True)  # default safe: dry_run=True

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
    """Apply (or dry-run) the ``EditRequest`` using ``edit_tools.apply_edit``.

    Translates ``EditResult.status`` → graph ``status``:

    * ``APPLIED``  → ``"edited"``
    * ``DRY_RUN``  → ``"dry_run"``
    * ``REJECTED`` → ``"failed"``
    * ``ERROR``    → ``"failed"``
    """
    edit_request = state.get("edit_request")

    if edit_request is None:
        log.debug("editor_node: no edit_request — skipping")
        return {}  # leave status unchanged

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
            f"editor_node: edit {edit_result.status.value} — "
            f"{edit_result.rejection_reason or 'no reason'}"
        ]

    return update


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------


def _route_after_locator(state: RemediationState) -> str:
    """Route: locator → planner or END."""
    if state.get("status") == "failed":
        return END
    return "planner"


def _route_after_planner(state: RemediationState) -> str:
    """Route: planner → edit_request_builder or END."""
    status = state.get("status", "")
    if status == "planned_version_found":
        return "edit_request_builder"
    # SAST, no_fix, workaround, or failed → stop
    return END


def _route_after_edit_request_builder(state: RemediationState) -> str:
    """Route: edit_request_builder → editor or END."""
    if state.get("status") == "edit_request_ready":
        return "editor"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_remediation_graph():
    """Compile and return the Phase 4.1 remediation StateGraph.

    The returned object is callable: ``graph.invoke(initial_state)``.
    """
    builder = StateGraph(RemediationState)

    # Register nodes
    builder.add_node("locator", locator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("edit_request_builder", edit_request_builder_node)
    builder.add_node("editor", editor_node)

    # Edges
    builder.add_edge(START, "locator")
    builder.add_conditional_edges("locator", _route_after_locator)
    builder.add_conditional_edges("planner", _route_after_planner)
    builder.add_conditional_edges("edit_request_builder", _route_after_edit_request_builder)
    builder.add_edge("editor", END)

    return builder.compile()


# Module-level compiled graph (import directly and call .invoke / run_remediation)
remediation_engine = build_remediation_graph()


def run_remediation(
    issue,
    repo_root: str,
    *,
    dry_run: bool = True,
) -> RemediationState:
    """Convenience entry point: run the remediation graph for a single issue.

    Args:
        issue:     ``VulnerabilityIssue`` to remediate.
        repo_root: Absolute path to the cloned repository workspace.
        dry_run:   When ``True`` (default), edits are validated but not written to disk.

    Returns:
        The final ``RemediationState`` after graph execution.
    """
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
