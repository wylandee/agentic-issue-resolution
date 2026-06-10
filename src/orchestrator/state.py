"""
state.py - LangGraph state schemas for the AppSec Remediation Engine.

Phase 4.1 (Legacy):
    ``RemediationState`` - single-issue, kept for the current Phase 4.1 graph.

Phase 5:
    ``OrchestratorState`` - group-level, looping state machine for the new
    Triage -> Remedy Agent -> Sandbox -> Scan -> PR flow.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its new error strings and LangGraph will append them.
* ``messages`` in ``OrchestratorState`` uses ``add_messages`` so each remedy
  pass appends only the newly produced LLM and tool transcript.
* ``changed_files`` uses ``operator.add`` so each remedy pass can report only
  the files it newly modified.
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


DEFAULT_MAX_RETRIES: int = 3
"""Default number of self-correction iterations the Remedy Agent may attempt."""


class OrchestratorState(TypedDict, total=False):
    """
    Full state schema for the Phase 5 autonomous remediation state machine.

    Required inputs
    ---------------
    repo_root:
        Absolute path to the cloned repository on disk.
    valid_groups:
        Non-empty list of triaged ``VulnerabilityGroup`` records.

    Retry / orchestration fields
    ----------------------------
    retry_count:
        Incremented by the Remedy Agent when retry feedback is present.
    max_retries:
        Ceiling for ``retry_count``.
    workspace_volume:
        Docker named volume shared across builder, remedy, scan, test, teardown.

    Remedy transcript / outputs
    ---------------------------
    messages:
        Accumulated conversation history including tool calls and tool results.
    changed_files:
        Repo-relative files successfully modified inside the workspace.

    Validation feedback
    -------------------
    install_failures:
        Captured stdout/stderr from failed dependency sync.
    test_failures:
        Captured stdout/stderr from failed unit tests.
    scan_failures:
        Captured stdout/stderr from failed ODC scans.
    """

    repo_root: str
    valid_groups: List[VulnerabilityGroup]

    retry_count: int
    max_retries: int

    messages: Annotated[List[AnyMessage], add_messages]
    changed_files: Annotated[List[str], operator.add]

    workspace_volume: Optional[str]

    install_failures: Optional[str]
    test_failures: Optional[str]
    scan_failures: Optional[str]

    status: str
    errors: Annotated[List[str], operator.add]


def initial_orchestrator_state(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Dict[str, Any]:
    """Build a well-formed initial ``OrchestratorState`` dict."""
    return {
        "repo_root": repo_root,
        "valid_groups": valid_groups,
        "retry_count": 0,
        "max_retries": max_retries,
        "messages": [],
        "changed_files": [],
        "workspace_volume": None,
        "install_failures": None,
        "test_failures": None,
        "scan_failures": None,
        "status": "pending",
        "errors": [],
    }
