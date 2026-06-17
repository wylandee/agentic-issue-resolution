"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

Phase 4.1 (Legacy):
    ``RemediationState`` - single-issue, kept for the current Phase 4.1 graph.

Phase 5:
    ``OrchestratorState`` - supervisor master state for the hub-and-spoke
    remedy architecture.
    ``SubagentState`` - ephemeral private state for specialist subagents.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its new error strings and LangGraph will append them.
* ``messages`` exists only in ``SubagentState`` so subagent ReAct transcripts
  stay isolated from the long-lived supervisor state.
* ``changed_files`` uses ``operator.add`` so each node can report only the
  files it newly observed as changed.
* All other fields use the default "last writer wins" semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Mapping, Optional, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from src.contracts.schemas import (
    AgentActionSummary,
    EditRequest,
    EditResult,
    FixPlan,
    LocalizedIssue,
    QAEvaluation,
    RoutingStrategy,
    SystemContext,
    VulnerabilityGroup,
    VulnerabilityIssue,
)

K = TypeVar("K")
V = TypeVar("V")


def merge_dict_reducer(
    left: Mapping[K, V] | None,
    right: Mapping[K, V] | None,
) -> Dict[K, V]:
    """Merge dict-like values without mutating either input."""
    merged: Dict[K, V] = dict(left or {})
    if right:
        merged.update(right)
    return merged


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
    Full state schema for the Phase 5 supervisor master state.

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

    Supervisor memory / outputs
    ---------------------------
    changed_files:
        Repo-relative files successfully modified across subagent runs.
    """

    repo_root: str
    valid_groups: List[VulnerabilityGroup]

    issues: List[VulnerabilityIssue]
    system_context: SystemContext

    constraints_ledger: Annotated[List[str], operator.add]
    retry_counts: Annotated[Dict[str, int], merge_dict_reducer]
    group_strategies: Annotated[Dict[str, RoutingStrategy], merge_dict_reducer]
    qa_evaluations: Annotated[Dict[str, QAEvaluation], merge_dict_reducer]
    action_summaries: Annotated[List[AgentActionSummary], operator.add]
    changed_files: Annotated[List[str], operator.add]

    workspace_volume: Optional[str]

    status: str
    diff: str
    langsmith_run_id: str
    langsmith_trace_url: str
    errors: Annotated[List[str], operator.add]


class SubagentState(TypedDict, total=False):
    """
    Ephemeral private state for one specialist subagent run.

    This is the only Phase 5 state that carries localized ReAct messages.
    """

    repo_root: str
    workspace_volume: str

    target_groups: List[VulnerabilityGroup]
    feedback_by_group: Dict[str, str]

    target_group: VulnerabilityGroup
    constraints_ledger: List[str]
    previous_feedback: Optional[str]

    messages: Annotated[List[AnyMessage], add_messages]

    action_summary: AgentActionSummary
    changed_files: Annotated[List[str], operator.add]
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
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "qa_evaluations": {},
        "action_summaries": [],
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


def initial_update_subagent_state(
    repo_root: str,
    workspace_volume: str,
    target_groups: List[VulnerabilityGroup],
    constraints_ledger: List[str],
    feedback_by_group: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a well-formed initial batch ``SubagentState`` dict."""
    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "target_groups": list(target_groups),
        "feedback_by_group": dict(feedback_by_group or {}),
        "constraints_ledger": list(constraints_ledger),
        "messages": [],
        "changed_files": [],
        "errors": [],
    }


def initial_workaround_subagent_state(
    repo_root: str,
    workspace_volume: str,
    target_group: VulnerabilityGroup,
    constraints_ledger: List[str],
    previous_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a well-formed single-group workaround ``SubagentState`` dict."""
    return {
        "repo_root": repo_root,
        "workspace_volume": workspace_volume,
        "target_group": target_group,
        "constraints_ledger": list(constraints_ledger),
        "previous_feedback": previous_feedback,
        "messages": [],
        "changed_files": [],
        "errors": [],
    }
