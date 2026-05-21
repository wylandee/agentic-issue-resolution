"""
Shared Pydantic v2 contracts for the Agentic AppSec Remediation Engine.

All data that flows between agents, tools, and storage layers is typed
through these models. Agents NEVER pass raw dicts across module boundaries.
"""

from .schemas import (
    # Enumerations
    Severity,
    IssueSource,
    IssueType,
    ASTNodeType,
    EditStatus,
    ValidationStatus,
    TrajectoryEventKind,
    # Core domain models
    VulnerabilityIssue,
    LocalizedIssue,
    EditRequest,
    EditResult,
    ValidationResult,
    PatchAttempt,
    TrajectoryEvent,
)

__all__ = [
    "Severity",
    "IssueSource",
    "IssueType",
    "ASTNodeType",
    "EditStatus",
    "ValidationStatus",
    "TrajectoryEventKind",
    "VulnerabilityIssue",
    "LocalizedIssue",
    "EditRequest",
    "EditResult",
    "ValidationResult",
    "PatchAttempt",
    "TrajectoryEvent",
]
