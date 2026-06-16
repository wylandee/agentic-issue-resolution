"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

Phase 4.1 (Legacy):
    ``RemediationState`` - single-issue, kept for the current Phase 4.1 graph.

Phase 5:
    ``OrchestratorState`` - group-level state for the linear
    workspace_builder -> remedy_agent -> teardown pipeline.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its new error strings and LangGraph will append them.
* ``messages`` in ``OrchestratorState`` uses ``add_messages`` so the remedy
  agent appends only the newly produced LLM and tool transcript.
* ``changed_files`` uses ``operator.add`` so each node can report only the
  files it newly observed as changed.
* All other fields use the default "last writer wins" semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from src.contracts.schemas import (
    EditRequest,
    EditResult,
    FixPlan,
    LocalizedIssue,
    SystemContext,
    VulnerabilityGroup,
    VulnerabilityIssue,
)


class RemediationState(TypedDict, total=False):
    """
    Full state schema for the Phase 4.1 remediation graph.

    Fields with ``total=False`` are optional. Only ``issue`` and ``repo_root``
    are required in the initial invocation dict.
    """

    issue: VulnerabilityIssue
    repo_root: str

    localized_issue: Optional[LocalizedIssue]
    fix_plan: Optional[FixPlan]
    edit_request: Optional[EditRequest]
    edit_result: Optional[EditResult]

    status: str
    dry_run: bool
    errors: Annotated[list[str], operator.add]


class OrchestratorState(TypedDict, total=False):
    """
    Full state schema for the Phase 5 linear remediation pipeline.

    Required inputs
    ---------------
    repo_root:
        Absolute path to the cloned repository on disk.
    valid_groups:
        Non-empty list of triaged ``VulnerabilityGroup`` records.

    Orchestration fields
    --------------------
    workspace_volume:
        Docker named volume shared across builder, remedy agent, and teardown.

    Remedy transcript / outputs
    ---------------------------
    messages:
        Accumulated conversation history including tool calls and tool results.
    changed_files:
        Repo-relative files successfully modified inside the workspace.
    """

    repo_root: str
    valid_groups: List[VulnerabilityGroup]
    
    issues: List[VulnerabilityIssue]
    system_context: SystemContext

    messages: Annotated[List[AnyMessage], add_messages]
    changed_files: Annotated[List[str], operator.add]

    workspace_volume: Optional[str]

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    errors: Annotated[List[str], operator.add]


def initial_orchestrator_state(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    issues: Optional[List[VulnerabilityIssue]] = None,
    system_context: Optional[SystemContext] = None,
) -> Dict[str, Any]:
    """Build a well-formed initial ``OrchestratorState`` dict."""
    state: Dict[str, Any] = {
        "repo_root": repo_root,
        "valid_groups": valid_groups,
        "messages": [],
        "changed_files": [],
        "workspace_volume": None,
        "status": "pending",
        "diff": "",
        "errors": [],
    }
    if issues is not None:
        state["issues"] = issues
    if system_context is not None:
        state["system_context"] = system_context
    return state
