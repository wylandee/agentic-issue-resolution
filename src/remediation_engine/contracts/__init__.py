"""
Shared Pydantic v2 contracts for the Agentic AppSec Remediation Engine.

All data that flows between agents, tools, and storage layers is typed
through these models. Agents NEVER pass raw dicts across module boundaries.
"""

from .decision_codes import VALID_TRANSITIONS, DecisionCode, validate_transition
from .llm_advisory import LLMAdvisory
from .planner_advice import PlannerAdvice, PlannerBatchAdvice
from .schemas import (
    # Constants
    MAX_ANCESTRY_DEPTH,
    MAX_TASK_QUEUE_SIZE,
    AgentActionStatus,
    AgentActionSummary,
    ASTNodeType,
    BatchQAResult,
    # Sandbox contracts
    CommandResult,
    CVEEnrichment,
    # Sub-models
    CWEEntry,
    EditRequest,
    EditResult,
    EditStatus,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    GroupRemediationStatus,
    IssueSource,
    IssueType,
    LineRange,
    LocalizedIssue,
    NoFixMitigationStage,
    PatchAttempt,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    # Enumerations
    Severity,
    StateConsistencyEvent,
    SupervisorDecision,
    SupervisorRetryPlan,
    # Triage and QA contracts
    SystemContext,
    TaskAttemptSnapshot,
    # Task queue
    TaskSpawnRequest,
    TaskStatus,
    TrajectoryEvent,
    TrajectoryEventKind,
    TriageResult,
    UpdateRetryDiagnostics,
    ValidationResult,
    ValidationStatus,
    VulnerabilityGroup,
    # Core domain models
    VulnerabilityIssue,
    WorkaroundContext,
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundPhase,
    WorkaroundPlannedReplacement,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from .supervisor_phases import AuditRecord, EligibleActions, ReconciliationResult
from .version_policy import RegistryCandidate, is_version_space_exhausted, select_version

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
    "NoFixMitigationStage",
    "AgentActionStatus",
    "GroupRemediationStatus",
    "TaskStatus",
    "DecisionCode",
    "VALID_TRANSITIONS",
    "validate_transition",
    # Constants
    "MAX_ANCESTRY_DEPTH",
    "MAX_TASK_QUEUE_SIZE",
    "CWEEntry",
    "LineRange",
    # Sandbox contracts
    "CommandResult",
    "VulnerabilityIssue",
    "LocalizedIssue",
    "FixPlan",
    "EditRequest",
    "EditResult",
    "ValidationResult",
    "PatchAttempt",
    "TrajectoryEvent",
    # Triage and QA contracts
    "SystemContext",
    "CVEEnrichment",
    "VulnerabilityGroup",
    "TriageResult",
    "QAEvaluation",
    "QAFailureEvidence",
    "WorkaroundPhase",
    "WorkaroundContext",
    "WorkaroundPlannedReplacement",
    "WorkaroundEdit",
    "WorkaroundEditSet",
    "WorkaroundReplayPlan",
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
    # Task queue
    "TaskSpawnRequest",
    "RemediationTask",
    "PlannerAdvice",
    "PlannerBatchAdvice",
    "RegistryCandidate",
    "select_version",
    "is_version_space_exhausted",
    "LLMAdvisory",
    "ReconciliationResult",
    "EligibleActions",
    "AuditRecord",
]
