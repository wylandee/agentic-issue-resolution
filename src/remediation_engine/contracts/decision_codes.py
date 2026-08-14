"""Machine-readable decision codes and task-transition rules.

The supervisor uses :class:`DecisionCode` as an audit-friendly explanation of
the deterministic rule that selected a route.  The transition table is kept
here, next to the other state-machine contracts, so worker and QA reducers can
share one predicate instead of reimplementing lifecycle rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class DecisionCode(str, Enum):  # noqa: UP042
    """Machine-readable rule identifier for every supervisor decision."""

    TRIAGE_REQUIRED = "TRIAGE_REQUIRED"
    QA_READY = "QA_READY"
    QA_READY_BATCH = "QA_READY_BATCH"
    NO_FIX_LIFECYCLE = "NO_FIX_LIFECYCLE"
    EXHAUSTED_UPDATE_PIVOT = "EXHAUSTED_UPDATE_PIVOT"
    RETRY_VERSION_BUMP = "RETRY_VERSION_BUMP"
    NEW_VERSION_BUMP = "NEW_VERSION_BUMP"
    WORKAROUND_DISPATCH = "WORKAROUND_DISPATCH"
    NO_ACTIONABLE_TASKS = "NO_ACTIONABLE_TASKS"
    FINAL_FULL_SCAN_REQUIRED = "FINAL_FULL_SCAN_REQUIRED"
    NO_VALID_GROUPS = "NO_VALID_GROUPS"
    INVALID_LLM_DECISION = "INVALID_LLM_DECISION"
    PIVOT_TO_WORKAROUND = "PIVOT_TO_WORKAROUND"


# Store enum values rather than importing ``schemas.TaskStatus`` at module
# import time.  ``schemas`` imports ``DecisionCode`` after defining
# ``TaskStatus``; using values here keeps both direct import orders safe while
# remaining equivalent to enum members because TaskStatus is a str Enum.
VALID_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pending", "optimistically_fixed"),
        ("pending", "needs_retry"),
        ("pending", "unfixable"),
        ("optimistically_fixed", "qa_passed"),
        ("optimistically_fixed", "needs_retry"),
        ("optimistically_fixed", "inconclusive"),
        ("needs_retry", "optimistically_fixed"),
        ("needs_retry", "unfixable"),
        ("needs_retry", "inconclusive"),
    }
)


def _status_value(status: Any) -> str:
    """Normalize a TaskStatus-like value for transition comparison."""

    return str(getattr(status, "value", status))


def validate_transition(current: Any, target: Any) -> bool:
    """Return whether a task lifecycle transition is permitted.

    Terminal statuses are absorbing states because they have no outgoing
    entries in :data:`VALID_TRANSITIONS`.  The predicate is intentionally
    side-effect free; callers decide how to record rejected transitions.
    """

    return (_status_value(current), _status_value(target)) in VALID_TRANSITIONS


__all__ = ["DecisionCode", "VALID_TRANSITIONS", "validate_transition"]
