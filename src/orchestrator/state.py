"""
state.py — LangGraph state schema for the Phase 4.1 remediation graph.

``RemediationState`` is a ``TypedDict`` consumed and produced by every node in
the graph.  LangGraph merges node return values into the running state via the
reducers declared here.

Reducer notes
-------------
* ``errors`` uses ``operator.add`` so each node can return only its *new* error
  strings — LangGraph automatically appends them to the accumulated list.
* All other fields use the default "last writer wins" semantics (no annotation).

Input vs. output fields
-----------------------
Required inputs (must be provided to ``graph.invoke``):
    issue       — the ``VulnerabilityIssue`` to remediate
    repo_root   — absolute path to the cloned repository on disk

Output fields (populated by nodes during execution):
    localized_issue   — result from locator_node
    fix_plan          — result from planner_node (SCA only)
    edit_request      — result from edit_request_builder_node (SCA version_found only)
    edit_result       — result from editor_node
    status            — routing / human-readable outcome string
    dry_run           — controls whether edit_tools writes to disk
    errors            — accumulated error strings (append-only via reducer)
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict

from src.contracts.schemas import (
    EditRequest,
    EditResult,
    FixPlan,
    LocalizedIssue,
    VulnerabilityIssue,
)


class RemediationState(TypedDict, total=False):
    """
    Full state schema for the remediation graph.

    Fields with ``total=False`` are optional (i.e. do not need to be present in
    the initial invocation dict).  Only ``issue`` and ``repo_root`` are required.
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
