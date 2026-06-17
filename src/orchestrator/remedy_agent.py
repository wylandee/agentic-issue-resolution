"""
Deprecated Phase 5 monolithic remedy agent shim.

The monolithic Remedy Agent has been removed from active orchestration in favor
of specialized update and workaround subagents. This module remains only as an
explicit transitional failure surface for stale callers.
"""

from __future__ import annotations

from typing import Any, Dict

from src.orchestrator.state import OrchestratorState

PHASE5_REFACTOR_BLOCKED_STATUS = "phase5_refactor_blocked"
PHASE5_REFACTOR_BLOCKED_MESSAGE = (
    "Phase 5 supervisor refactor in progress: the monolithic remedy agent has "
    "been removed and Supervisor/QA routing is not implemented yet."
)


def run_remedy_agent(_state: OrchestratorState) -> Dict[str, Any]:
    """Deprecated compatibility shim for stale Phase 5 callers."""
    return {
        "status": PHASE5_REFACTOR_BLOCKED_STATUS,
        "errors": [PHASE5_REFACTOR_BLOCKED_MESSAGE],
    }
