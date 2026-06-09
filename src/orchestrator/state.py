"""
state.py — LangGraph state schemas for the AppSec Remediation Engine.

Phase 4.1 (Legacy):
    ``RemediationState`` — single-issue, kept for the current Phase 4.1 graph.

Phase 5:
    ``OrchestratorState`` — group-level, looping state machine for the new
    Triage → Remedy Agent → Sandbox → Scan → PR flow.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` in both states so each node can return
  only its *new* error strings — LangGraph automatically appends them to the
  accumulated list.
* All other fields use the default "last writer wins" semantics (no
  annotation), so retry loops can replace ``edit_requests`` cleanly.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

from src.contracts.schemas import (
    EditRequest,
    EditResult,
    FixPlan,
    LocalizedIssue,
    VulnerabilityGroup,
    VulnerabilityIssue,
)

# ---------------------------------------------------------------------------
# Phase 4.1 — Legacy single-issue state (preserved; do not remove)
# ---------------------------------------------------------------------------


class RemediationState(TypedDict, total=False):
    """
    Full state schema for the Phase 4.1 remediation graph.

    Fields with ``total=False`` are optional (i.e. do not need to be present in
    the initial invocation dict).  Only ``issue`` and ``repo_root`` are required.

    .. deprecated::
        Use ``OrchestratorState`` for all Phase 5+ work.  This class is kept
        temporarily so that ``src/orchestrator/graph.py`` and its tests do not
        require simultaneous migration.
    """

    # ---- Required inputs ----
    issue: VulnerabilityIssue
    repo_root: str

    # ---- Outputs set by nodes ----
    localized_issue: Optional[LocalizedIssue]
    fix_plan: Optional[FixPlan]
    edit_request: Optional[EditRequest]
    edit_result: Optional[EditResult]

    # ---- Routing / metadata ----
    status: str
    dry_run: bool

    # ---- Error accumulator (reducer: append-only) ----
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Phase 5 — Group-level orchestrator state
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 3
"""Default number of self-correction iterations the Remedy Agent may attempt."""


class OrchestratorState(TypedDict, total=False):
    """
    Full state schema for the Phase 5 autonomous remediation state machine.

    The orchestrator consumes ``valid_groups`` produced by the triage pipeline
    and loops: Remedy Agent → Sandbox Edits → ODC Scan → Unit Tests → PR.
    If a downstream validation step fails, the loop routes back to the Remedy
    Agent with ``test_failures`` / ``scan_failures`` so it can self-correct.

    Reducer notes
    -------------
    * ``errors``        — append-only via ``operator.add``; each node returns
                          only its new errors.
    * ``edit_requests`` — last-writer-wins; retry loops replace the previous
                          failed edit batch rather than appending to it.

    Required inputs (must be present in the initial state dict)
    -----------------------------------------------------------
    repo_root   — absolute path to the cloned repository on disk
    valid_groups — non-empty list of ``VulnerabilityGroup`` objects that
                   survived triage (``TriageResult.is_valid == True``).

    Optional / node-populated fields
    ---------------------------------
    retry_count     — incremented by the Remedy Agent on each self-correction.
    max_retries     — ceiling for ``retry_count``; defaults to 3.
    edit_requests   — the current batch of file edits proposed by the Remedy Agent.
    test_failures   — stdout/stderr from the failing unit-test run (if any).
    scan_failures   — stdout/stderr from the failing ODC scan (if any).
    status          — human-readable / routing label for the current node outcome.
    errors          — accumulated error strings (append-only via reducer).
    """

    # ---- Required inputs ----
    repo_root: str
    valid_groups: List[VulnerabilityGroup]

    # ---- Retry loop counters ----
    retry_count: int
    max_retries: int

    # ---- Remedy Agent outputs ----
    edit_requests: List[EditRequest]

    # ---- Shared Docker workspace (last-writer-wins) ----
    workspace_volume: Optional[str]

    # ---- Feedback from downstream validation ----
    test_failures: Optional[str]
    scan_failures: Optional[str]

    # ---- Routing / metadata ----
    status: str

    # ---- Error accumulator (reducer: append-only) ----
    errors: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def initial_orchestrator_state(
    repo_root: str,
    valid_groups: List[VulnerabilityGroup],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Build a well-formed initial ``OrchestratorState`` dict.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository on disk.
    valid_groups:
        Triaged vulnerability groups to remediate.
    max_retries:
        Maximum self-correction iterations (defaults to ``DEFAULT_MAX_RETRIES``).

    Returns
    -------
    dict
        A ``OrchestratorState``-compatible initial state dict with all
        required fields populated and optional fields set to safe defaults.
    """
    return {
        "repo_root": repo_root,
        "valid_groups": valid_groups,
        "retry_count": 0,
        "max_retries": max_retries,
        "edit_requests": [],
        "workspace_volume": None,
        "test_failures": None,
        "scan_failures": None,
        "status": "pending",
        "errors": [],
    }
