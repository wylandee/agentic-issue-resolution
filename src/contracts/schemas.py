"""
Pydantic v2 schemas for the Agentic AppSec Remediation Engine.

Design principles
-----------------
* All fields are strictly typed; no plain ``dict`` or ``Any`` at API boundaries.
* Optional fields use ``None`` as the sentinel (no empty strings as NULL).
* Validators are declared with ``@field_validator`` (Pydantic v2 style).
* Every model is serialisable to/from JSON with ``.model_dump_json()`` /
  ``.model_validate_json()``.
* JSONL round-trip is the canonical storage format; CSV is a human export.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Canonical severity levels across SAST and SCA findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class IssueSource(str, Enum):
    """The originating scanner / data source."""

    SEMGREP = "semgrep"
    ODC = "odc"              # OWASP Dependency-Check
    MANUAL = "manual"        # Injected by a human or eval harness
    SYNTHETIC = "synthetic"  # Generated for testing / benchmarks


class IssueType(str, Enum):
    """Broad classification of the finding."""

    SAST = "sast"  # Static Application Security Testing code finding
    SCA = "sca"    # Software Composition Analysis dependency finding


class ASTNodeType(str, Enum):
    """Abstract Syntax Tree node kinds used by the SAST code locator."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    ARROW_FUNCTION = "arrow_function"
    CALL_EXPRESSION = "call_expression"
    VARIABLE_DECLARATION = "variable_declaration"
    IMPORT_STATEMENT = "import_statement"
    UNKNOWN = "unknown"


class EditStatus(str, Enum):
    """Outcome of an attempted file edit."""

    APPLIED = "applied"      # Patch applied successfully
    DRY_RUN = "dry_run"      # Validated only; no disk write
    REJECTED = "rejected"    # Validation failed before write
    ERROR = "error"          # Unexpected error during application


class ValidationStatus(str, Enum):
    """Outcome of a sandbox validation run."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"


class TrajectoryEventKind(str, Enum):
    """Discrete steps recorded in a remediation trajectory."""

    INGEST = "ingest"
    LOCALIZE = "localize"
    PLAN = "plan"
    APPLY_EDIT = "apply_edit"
    VALIDATE = "validate"
    RETRY = "retry"
    DELIVER = "deliver"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class LineRange(BaseModel):
    """Inclusive, 1-indexed line range within a file."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(..., ge=1, description="First line of the range (1-indexed).")
    end: int = Field(..., ge=1, description="Last line of the range (1-indexed, inclusive).")

    @model_validator(mode="after")
    def _end_gte_start(self) -> "LineRange":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    @property
    def line_count(self) -> int:
        return self.end - self.start + 1


class CWEEntry(BaseModel):
    """A single CWE weakness identifier with optional display name."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., pattern=r"^CWE-\d+$", description="e.g. 'CWE-79'")
    name: Optional[str] = Field(None, description="Human-readable weakness name.")


# ---------------------------------------------------------------------------
# VulnerabilityIssue
# ---------------------------------------------------------------------------


class VulnerabilityIssue(BaseModel):
    """
    Canonical representation of a single security finding.

    This is the first typed object produced by ingestion (Semgrep, ODC).
    Agents consume this model; raw scanner payloads are stored in
    ``raw_payload`` for audit purposes.
    """

    model_config = ConfigDict(
        frozen=False,
        populate_by_name=True,
    )

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Stable internal finding UUID.")
    finding_id: Optional[str] = Field(
        None,
        description="Scanner-native finding identifier (e.g. Semgrep finding ID).",
    )

    # Provenance
    source: IssueSource = Field(..., description="Which scanner produced this finding.")
    issue_type: IssueType = Field(..., description="SAST or SCA classification.")

    # Repository context
    repo_url: Optional[str] = Field(
        None, description="HTTPS clone URL of the target repository."
    )
    base_ref: Optional[str] = Field(
        None,
        description="Git branch name or commit SHA that was scanned.",
    )

    # Rule / advisory identification
    rule_id: Optional[str] = Field(
        None,
        description="Semgrep rule ID or equivalent scanner rule key.",
    )
    cve_id: Optional[str] = Field(
        None,
        pattern=r"^CVE-\d{4}-\d{4,}$",
        description="CVE identifier (SCA findings).",
    )
    cwe: List[CWEEntry] = Field(
        default_factory=list,
        description="Associated CWE weakness entries.",
    )
    owasp: List[str] = Field(
        default_factory=list,
        description="OWASP Top-10 category labels (e.g. 'A03:2021').",
    )

    # Severity
    severity: Severity = Field(default=Severity.UNKNOWN)
    confidence: Optional[str] = Field(
        None, description="Scanner confidence label (HIGH/MEDIUM/LOW)."
    )

    # Location (populated for SAST; optional for SCA)
    file_path: Optional[str] = Field(
        None,
        description="Repo-relative path to the affected source file.",
    )
    line_range: Optional[LineRange] = Field(
        None, description="Affected line range within ``file_path``."
    )

    # SCA-specific
    package_name: Optional[str] = Field(None, description="Vulnerable package name.")
    package_version: Optional[str] = Field(
        None, description="Installed version of the package."
    )
    fixed_version: Optional[str] = Field(
        None, description="Earliest non-vulnerable version, if known."
    )
    purl: Optional[str] = Field(
        None, description="Package URL (PURL) per the PURL spec."
    )
    ecosystem: Optional[str] = Field(
        None, description="Package ecosystem: npm, pypi, maven, etc."
    )

    # Content
    message: Optional[str] = Field(None, description="Human-readable finding message.")
    finding_url: Optional[str] = Field(
        None, description="Deep-link to the scanner UI for this finding."
    )

    # Validation profile (which sandbox ruleset to use)
    validation_profile: Optional[str] = Field(
        None,
        description="Key into config/rules.yaml selecting the validation suite.",
    )

    # Audit
    raw_payload: Optional[Dict[str, Any]] = Field(
        None,
        description="Original scanner JSON payload; preserved for auditability.",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this issue was ingested.",
    )

    @field_serializer("ingested_at")
    def _serialise_ingested_at(self, v: datetime) -> str:
        return v.isoformat()

    @field_validator("cve_id", mode="before")
    @classmethod
    def _normalise_cve(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s if s else None

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v: Any) -> Severity:
        if isinstance(v, Severity):
            return v
        try:
            return Severity(str(v).strip().upper())
        except ValueError:
            return Severity.UNKNOWN

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalise_path(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().lstrip("/")
        return s if s else None


# ---------------------------------------------------------------------------
# LocalizedIssue
# ---------------------------------------------------------------------------


class LocalizedIssue(BaseModel):
    """
    A ``VulnerabilityIssue`` enriched with AST-level code localization context.

    Produced by the SAST code locator (``locate_sast``) and the SCA manifest
    locator (``locate_dependency``). Agents use this to plan targeted edits.
    """

    model_config = ConfigDict(frozen=False)

    issue: VulnerabilityIssue = Field(..., description="The originating finding.")

    # AST / symbol context (SAST)
    enclosing_symbol: Optional[str] = Field(
        None, description="Name of the enclosing function, method, or class."
    )
    enclosing_node_type: ASTNodeType = Field(
        default=ASTNodeType.UNKNOWN,
        description="AST node type of the enclosing symbol.",
    )
    sink_expression: Optional[str] = Field(
        None, description="The specific sink call expression at the finding location."
    )
    imports: List[str] = Field(
        default_factory=list,
        description="Relevant import statements from the affected file.",
    )
    data_flow_hints: List[str] = Field(
        default_factory=list,
        description="Brief data-flow notes (e.g. 'taint source: req.params.id').",
    )
    snippet: Optional[str] = Field(
        None, description="Bounded code snippet around the finding (≤ 30 lines)."
    )

    # Manifest context (SCA)
    manifest_file: Optional[str] = Field(
        None, description="Repo-relative path to the resolved manifest / lockfile."
    )
    is_direct_dependency: Optional[bool] = Field(
        None,
        description="True if the package appears as a direct dependency.",
    )
    manifest_line: Optional[int] = Field(
        None,
        ge=1,
        description="1-indexed line in the manifest where the dependency is declared.",
    )
    manifest_snippet: Optional[str] = Field(
        None, description="3-line snippet centred on the manifest declaration."
    )
    fix_instruction: Optional[str] = Field(
        None,
        description="Short natural-language instruction for a Remedy agent.",
    )

    # Confidence
    localization_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0–1 confidence score for the localization result.",
    )

    localized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# EditRequest / EditResult
# ---------------------------------------------------------------------------


class EditRequest(BaseModel):
    """
    A structured, agent-proposed file mutation.

    Agents MUST supply ``old_text`` as an exact-match anchor. The editor
    tool rejects requests where the old text is absent, ambiguous, or stale.
    """

    model_config = ConfigDict(frozen=True)

    # Target
    repo_root: str = Field(
        ..., description="Absolute path to the repository workspace root."
    )
    file_path: str = Field(
        ..., description="Repo-relative path to the file to be edited."
    )

    # Edit specification
    old_text: str = Field(
        ...,
        min_length=1,
        description="The exact text block to be replaced (must match uniquely).",
    )
    new_text: str = Field(
        ..., description="Replacement text for the matched block."
    )

    # Safety controls
    dry_run: bool = Field(
        default=False,
        description="If True, validate only; do not write to disk.",
    )
    max_deletion_lines: int = Field(
        default=200,
        ge=1,
        description="Reject edits that delete more than this many lines.",
    )

    # Traceability
    issue_id: Optional[UUID] = Field(
        None, description="UUID of the ``VulnerabilityIssue`` this edit addresses."
    )
    rationale: Optional[str] = Field(
        None, description="Agent-provided explanation for the change."
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def _no_traversal(cls, v: Any) -> str:
        path = str(v)
        if ".." in path.split("/") or ".." in path.split("\\"):
            raise ValueError("Path traversal detected in file_path.")
        return path


class EditResult(BaseModel):
    """
    The outcome of applying (or dry-running) an ``EditRequest``.
    """

    model_config = ConfigDict(frozen=True)

    request: EditRequest
    status: EditStatus
    unified_diff: Optional[str] = Field(
        None, description="Unified diff of the change (populated on APPLIED/DRY_RUN)."
    )
    lines_added: int = Field(default=0, ge=0)
    lines_removed: int = Field(default=0, ge=0)
    rejection_reason: Optional[str] = Field(
        None, description="Human-readable reason for REJECTED / ERROR status."
    )
    applied_at: Optional[datetime] = Field(
        None, description="UTC timestamp of the write (None for dry-run/rejected)."
    )


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class FailingTest(BaseModel):
    """Details of a single failing test case."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: Optional[str] = None
    line: Optional[int] = Field(None, ge=1)
    message: Optional[str] = None


class ValidationResult(BaseModel):
    """
    Structured outcome of a sandbox validation run (tests, security scans, etc.).

    The Feedback Loop agent parses this to decide whether to retry or deliver.
    """

    model_config = ConfigDict(frozen=False)

    # Identity
    patch_attempt_id: Optional[UUID] = Field(
        None, description="UUID of the ``PatchAttempt`` this validates."
    )
    phase: str = Field(
        ...,
        description="Phase label from the validation profile (install/unit/security_scan).",
    )

    # Result
    status: ValidationStatus
    exit_code: Optional[int] = None
    command: List[str] = Field(
        default_factory=list,
        description="The command array that was executed.",
    )
    stdout_tail: Optional[str] = Field(
        None, description="Last N lines of stdout (truncated for context)."
    )
    stderr_tail: Optional[str] = Field(
        None, description="Last N lines of stderr."
    )

    # Structured failure analysis
    failing_tests: List[FailingTest] = Field(default_factory=list)
    dependency_conflict_hints: List[str] = Field(
        default_factory=list,
        description="Extracted dependency conflict messages for retry planning.",
    )
    changed_files: List[str] = Field(
        default_factory=list,
        description="Repo-relative paths of files modified during validation.",
    )

    duration_seconds: Optional[float] = Field(None, ge=0.0)
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# PatchAttempt
# ---------------------------------------------------------------------------


class PatchAttempt(BaseModel):
    """
    A single end-to-end attempt to remediate one ``VulnerabilityIssue``.

    Bundles the edit(s) applied and the validation results for that attempt.
    The Remedy agent uses this to decide whether to retry (max 2–3 retries).
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    issue_id: UUID = Field(..., description="UUID of the ``VulnerabilityIssue`` being remediated.")
    attempt_number: int = Field(..., ge=1, description="1-indexed attempt counter.")

    edits: List[EditResult] = Field(
        default_factory=list,
        description="All file edits applied in this attempt.",
    )
    validations: List[ValidationResult] = Field(
        default_factory=list,
        description="Ordered validation results for this attempt.",
    )

    # Summary
    succeeded: Optional[bool] = Field(
        None,
        description="True if all validation phases passed; False if any failed; None if still running.",
    )
    failure_summary: Optional[str] = Field(
        None,
        description="Natural-language summary of why this attempt failed (for the next retry prompt).",
    )

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def all_validations_passed(self) -> bool:
        return bool(self.validations) and all(
            v.status == ValidationStatus.PASSED for v in self.validations
        )


# ---------------------------------------------------------------------------
# TrajectoryEvent
# ---------------------------------------------------------------------------


class TrajectoryEvent(BaseModel):
    """
    A single step in the remediation trajectory / audit log.

    Written by every agent node in the LangGraph StateGraph so the full
    execution can be replayed, evaluated, and billed.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    issue_id: Optional[UUID] = Field(None, description="Related ``VulnerabilityIssue`` UUID.")
    patch_attempt_id: Optional[UUID] = Field(None)

    kind: TrajectoryEventKind
    agent: Optional[str] = Field(None, description="Agent node name (e.g. 'remedy_agent').")

    # Payload
    summary: str = Field(..., description="Short human-readable description of the step.")
    detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured detail payload (tool call args, diff stats, etc.).",
    )

    # Cost tracking
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_seconds: Optional[float] = Field(None, ge=0.0)

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
