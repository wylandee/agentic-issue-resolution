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
    FixPlanStatus,
    FailureCategory,
    RoutingStrategy,
    SCARemediationStage,
    AgentActionStatus,
    GroupRemediationStatus,
    TaskStatus,
    # Constants
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    # Sub-models
    CWEEntry,
    LineRange,
    # Sandbox contracts (Phase 3)
    CommandResult,
    # Core domain models
    VulnerabilityIssue,
    LocalizedIssue,
    FixPlan,
    EditRequest,
    EditResult,
    ValidationResult,
    PatchAttempt,
    TrajectoryEvent,
    # Triage layer (Phase 4.0)
    SystemContext,
    CVEEnrichment,
    VulnerabilityGroup,
    TriageResult,
    QAEvaluation,
    BatchQAResult,
    AgentActionSummary,
    TaskAttemptSnapshot,
    UpdateRetryDiagnostics,
    WorkerExecutionDiagnostics,
    WorkerAttemptResult,
    QAAttemptResult,
    StateConsistencyEvent,
    SupervisorRetryPlan,
    SupervisorDecision,
    # Phase 5 Task Queue
    TaskSpawnRequest,
    RemediationTask,
    # Phase 5 Remedy Agent
    RemedyAgentOutput,
)

__all__ = [
    "Severity",
    "IssueSource",
    "IssueType",
    "ASTNodeType",
    "EditStatus",
    "ValidationStatus",
    "TrajectoryEventKind",
    "FixPlanStatus",
    "FailureCategory",
    "RoutingStrategy",
    "SCARemediationStage",
    "AgentActionStatus",
    "GroupRemediationStatus",
    "TaskStatus",
    # Constants
    "MAX_ANCESTRY_DEPTH",
    "MAX_TASK_QUEUE_SIZE",
    "CWEEntry",
    "LineRange",
    # Sandbox contracts (Phase 3)
    "CommandResult",
    "VulnerabilityIssue",
    "LocalizedIssue",
    "FixPlan",
    "EditRequest",
    "EditResult",
    "ValidationResult",
    "PatchAttempt",
    "TrajectoryEvent",
    # Triage layer (Phase 4.0)
    "SystemContext",
    "CVEEnrichment",
    "VulnerabilityGroup",
    "TriageResult",
    "QAEvaluation",
    "BatchQAResult",
    "AgentActionSummary",
    "TaskAttemptSnapshot",
    "UpdateRetryDiagnostics",
    "WorkerExecutionDiagnostics",
    "WorkerAttemptResult",
    "QAAttemptResult",
    "StateConsistencyEvent",
    "SupervisorRetryPlan",
    "SupervisorDecision",
    # Phase 5 Task Queue
    "TaskSpawnRequest",
    "RemediationTask",
    # Phase 5 Remedy Agent
    "RemedyAgentOutput",
]
