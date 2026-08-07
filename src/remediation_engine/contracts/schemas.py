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
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
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
    ODC = "odc"  # OWASP Dependency-Check
    MANUAL = "manual"  # Injected by a human or eval harness
    SYNTHETIC = "synthetic"  # Generated for testing / benchmarks


class IssueType(str, Enum):
    """Broad classification of the finding."""

    SAST = "sast"  # Static Application Security Testing code finding
    SCA = "sca"  # Software Composition Analysis dependency finding


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

    APPLIED = "applied"  # Patch applied successfully
    DRY_RUN = "dry_run"  # Validated only; no disk write
    REJECTED = "rejected"  # Validation failed before write
    ERROR = "error"  # Unexpected error during application


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


class FixPlanStatus(str, Enum):
    """Outcome of the fix-planner waterfall for one SCA finding."""

    VERSION_FOUND = "version_found"  # A safe pinned version was identified
    WORKAROUND_FOUND = "workaround_found"  # No upstream fix; web snippets found
    NO_FIX = "no_fix"  # All strategies exhausted, nothing found


class FailureCategory(str, Enum):
    """Strict QA failure categories used for supervisor retry routing."""

    SECURITY_FLAG = "security_flag"
    PEER_CONFLICT = "peer_conflict"
    BREAKING_CHANGE = "breaking_change"


class RoutingStrategy(str, Enum):
    """Strict supervisor routing strategies for vulnerability groups."""

    VERSION_BUMP = "version_bump"
    CODE_WORKAROUND = "code_workaround"


class SCARemediationStage(str, Enum):
    """Ordered remediation stages for an SCA version-bump task."""

    OSV_MINIMUM = "osv_minimum"
    NPM_SAME_MAJOR = "npm_same_major"
    NPM_LATEST = "npm_latest"
    CODE_WORKAROUND = "code_workaround"


class NoFixMitigationStage(str, Enum):
    """Ordered mitigation stages for a ``NO_FIX`` vulnerability group.

    ``PACKAGE_REMOVAL`` is the initial best-effort mitigation.  If that
    attempt fails validation, the same task advances to
    ``VULNERABLE_CODE_REMOVAL``.  A second failed attempt is terminalized as
    ``UNFIXABLE``.
    """

    PACKAGE_REMOVAL = "package_removal"
    VULNERABLE_CODE_REMOVAL = "vulnerable_code_removal"
    UNFIXABLE = "unfixable"


# ---------------------------------------------------------------------------
# Phase 5 orchestrator caps
# ---------------------------------------------------------------------------

MAX_ANCESTRY_DEPTH: int = 3
MAX_TASK_QUEUE_SIZE: int = 20


class AgentActionStatus(str, Enum):
    """Strict terminal statuses returned by subagents."""

    SUCCESS = "success"
    SURRENDER = "surrender"


class GroupRemediationStatus(str, Enum):
    """Lifecycle status for one vulnerability group in the remediation pipeline."""

    PENDING = "pending"  # Not yet touched by any subagent
    OPTIMISTICALLY_FIXED = "optimistically_fixed"  # Subagent succeeded; awaiting QA
    QA_PASSED = "qa_passed"  # QA explicitly passed; terminal success
    NEEDS_RETRY = "needs_retry"  # QA failed; will be re-routed
    UNFIXABLE = "unfixable"  # Max retries exhausted; terminal failure


class TaskStatus(str, Enum):
    """Lifecycle status for one RemediationTask in the task queue."""

    PENDING = "pending"  # Not yet dispatched to any worker
    OPTIMISTICALLY_FIXED = "optimistically_fixed"  # Worker succeeded; awaiting QA verdict
    QA_PASSED = "qa_passed"  # QA explicitly passed; terminal success
    NEEDS_RETRY = "needs_retry"  # QA failed; will be re-routed by supervisor
    UNFIXABLE = "unfixable"  # Max retries exhausted; terminal failure


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


class CommandResult(BaseModel):
    """
    Output of a command executed inside a ``DockerSandbox``.

    Produced by ``remediation_engine.runtime.sandbox_mgr.DockerSandbox.run``.
    """

    model_config = ConfigDict(frozen=True)

    exit_code: int = Field(
        ...,
        description="Process exit code.  0 = success; 124 = timeout; other = failure.",
    )
    stdout: str = Field(
        default="",
        description="Captured standard output from the command.",
    )
    stderr: str = Field(
        default="",
        description="Captured standard error from the command.",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time in seconds that the command ran inside the sandbox.",
    )


# ---------------------------------------------------------------------------
# FixPlan
# ---------------------------------------------------------------------------


class FixPlan(BaseModel):
    """
    Structured output of the fix-planner waterfall.

    Produced by ``remediation_engine.tools.fix_planner`` and consumed by the downstream
    Remedy agent to apply the correct type of edit.

    Invariants (enforced by model_validator):
    - ``version_found``    â†’ ``fixed_version`` is set, ``workaround_snippets`` is None.
    - ``workaround_found`` â†’ ``workaround_snippets`` is non-empty, ``fixed_version`` is None.
    - ``no_fix``           â†’ both ``fixed_version`` and ``workaround_snippets`` are None.
    """

    model_config = ConfigDict(frozen=True)

    status: FixPlanStatus = Field(..., description="Outcome of the waterfall.")
    fixed_version: str | None = Field(
        None,
        description="Safe pinned version to upgrade to (set iff status=version_found).",
    )
    workaround_snippets: list[str] | None = Field(
        None,
        description="Ordered list of workaround text snippets from web search "
        "(set iff status=workaround_found).",
    )
    instruction: str = Field(
        ...,
        min_length=1,
        description="Natural-language action for the Remedy agent.",
    )
    strategy_used: str = Field(
        ...,
        min_length=1,
        description="Which waterfall step produced this plan "
        "(local_regex | osv_api | npm_registry | serper | serper_llm | none).",
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> FixPlan:
        if self.status == FixPlanStatus.VERSION_FOUND:
            if not self.fixed_version:
                raise ValueError("status='version_found' requires a non-empty fixed_version.")
            if self.workaround_snippets is not None:
                raise ValueError("status='version_found' must have workaround_snippets=None.")
        elif self.status == FixPlanStatus.WORKAROUND_FOUND:
            if not self.workaround_snippets:
                raise ValueError(
                    "status='workaround_found' requires a non-empty workaround_snippets list."
                )
            if self.fixed_version is not None:
                raise ValueError("status='workaround_found' must have fixed_version=None.")
        else:  # NO_FIX
            if self.fixed_version is not None:
                raise ValueError("status='no_fix' must have fixed_version=None.")
            if self.workaround_snippets is not None:
                raise ValueError("status='no_fix' must have workaround_snippets=None.")
        return self


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class LineRange(BaseModel):
    """Inclusive, 1-indexed line range within a file."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(..., ge=1, description="First line of the range (1-indexed).")
    end: int = Field(..., ge=1, description="Last line of the range (1-indexed, inclusive).")

    @model_validator(mode="after")
    def _end_gte_start(self) -> LineRange:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    @property
    def line_count(self) -> int:
        """Return the number of lines covered by the inclusive range."""
        return self.end - self.start + 1


class CWEEntry(BaseModel):
    """A single CWE weakness identifier with optional display name."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., pattern=r"^CWE-\d+$", description="e.g. 'CWE-79'")
    name: str | None = Field(None, description="Human-readable weakness name.")


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
    finding_id: str | None = Field(
        None,
        description="Scanner-native finding identifier (e.g. Semgrep finding ID).",
    )

    # Provenance
    source: IssueSource = Field(..., description="Which scanner produced this finding.")
    issue_type: IssueType = Field(..., description="SAST or SCA classification.")

    # Repository context
    repo_url: str | None = Field(None, description="HTTPS clone URL of the target repository.")
    base_ref: str | None = Field(
        None,
        description="Git branch name or commit SHA that was scanned.",
    )

    # Rule / advisory identification
    rule_id: str | None = Field(
        None,
        description="Semgrep rule ID or equivalent scanner rule key.",
    )
    cve_id: str | None = Field(
        None,
        pattern=r"^CVE-\d{4}-\d{4,}$",
        description="CVE identifier (SCA findings).",
    )
    ghsa_id: str | None = Field(
        None,
        pattern=r"^GHSA-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$",
        description="GitHub Security Advisory identifier (SCA findings).",
    )
    cwe: list[CWEEntry] = Field(
        default_factory=list,
        description="Associated CWE weakness entries.",
    )
    owasp: list[str] = Field(
        default_factory=list,
        description="OWASP Top-10 category labels (e.g. 'A03:2021').",
    )

    # Severity
    severity: Severity = Field(default=Severity.UNKNOWN)
    confidence: str | None = Field(None, description="Scanner confidence label (HIGH/MEDIUM/LOW).")

    # Location (populated for SAST; optional for SCA)
    file_path: str | None = Field(
        None,
        description=(
            "Location string for the affected artifact. For SAST this is the "
            "repo-relative source file path; for SCA this may be the raw "
            "scanner-native lockfile or package path used later for manifest "
            "localization."
        ),
    )
    line_range: LineRange | None = Field(
        None, description="Affected line range within ``file_path``."
    )

    # SCA-specific
    package_name: str | None = Field(None, description="Vulnerable package name.")
    package_version: str | None = Field(None, description="Installed version of the package.")
    fixed_version: str | None = Field(
        None, description="Earliest non-vulnerable version, if known."
    )
    purl: str | None = Field(None, description="Package URL (PURL) per the PURL spec.")
    ecosystem: str | None = Field(None, description="Package ecosystem: npm, pypi, maven, etc.")

    # Content
    message: str | None = Field(None, description="Human-readable finding message.")
    finding_url: str | None = Field(
        None, description="Deep-link to the scanner UI for this finding."
    )
    dataflow_trace: dict[str, Any] | None = Field(
        None,
        description="Raw Semgrep dataflow trace object for SAST findings, if present.",
    )

    # Validation profile (which sandbox ruleset to use)
    validation_profile: str | None = Field(
        None,
        description="Key into config/rules.yaml selecting the validation suite.",
    )

    # Audit
    raw_payload: dict[str, Any] | None = Field(
        None,
        description="Original scanner JSON payload; preserved for auditability.",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this issue was ingested.",
    )

    @field_serializer("ingested_at")
    def _serialise_ingested_at(self, v: datetime) -> str:
        return v.isoformat()

    @field_validator("cve_id", mode="before")
    @classmethod
    def _normalise_cve(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s if s else None

    @field_validator("ghsa_id", mode="before")
    @classmethod
    def _normalise_ghsa(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s if s else None

    @model_validator(mode="after")
    def _backfill_ghsa_from_rule_id(self) -> VulnerabilityIssue:
        """Populate ``ghsa_id`` when legacy/scanner inputs only set ``rule_id``."""
        if self.ghsa_id or not self.rule_id:
            return self

        match = re.search(
            r"\b(GHSA-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4})\b",
            self.rule_id,
            re.IGNORECASE,
        )
        if match:
            self.ghsa_id = match.group(1).upper()
        return self

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
    def _normalise_path(cls, v: Any) -> str | None:
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
    locator (``locate_dependency``). This model is a *pure localization result* â€”
    it contains no fix instructions or remediation planning. Fix planning is the
    responsibility of the downstream "Plan Fix" agent.
    """

    model_config = ConfigDict(frozen=False)

    issue: VulnerabilityIssue = Field(..., description="The originating finding.")

    # AST / symbol context (SAST)
    enclosing_symbol: str | None = Field(
        None, description="Name of the enclosing function, method, or class."
    )
    enclosing_node_type: ASTNodeType = Field(
        default=ASTNodeType.UNKNOWN,
        description="AST node type of the enclosing symbol.",
    )
    sink_expression: str | None = Field(
        None, description="The specific sink call expression at the finding location."
    )
    imports: list[str] = Field(
        default_factory=list,
        description="Relevant import statements from the affected file.",
    )
    data_flow_hints: list[str] = Field(
        default_factory=list,
        description="Brief data-flow notes (e.g. 'taint source: req.params.id').",
    )
    snippet: str | None = Field(
        None, description="Bounded code snippet around the finding (â‰¤ 30 lines)."
    )

    # Manifest context (SCA)
    manifest_file: str | None = Field(
        None, description="Repo-relative path to the resolved manifest / lockfile."
    )
    is_direct_dependency: bool | None = Field(
        None,
        description="True if the package appears as a direct dependency.",
    )
    manifest_line: int | None = Field(
        None,
        ge=1,
        description="1-indexed line in the manifest where the dependency is declared.",
    )
    manifest_snippet: str | None = Field(
        None, description="3-line snippet centred on the manifest declaration."
    )
    package_manager: str | None = Field(
        None,
        description="Detected package manager (npm / yarn / pnpm) for the manifest.",
    )

    # Confidence
    localization_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0â€“1 confidence score for the localization result.",
    )

    localized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("manifest_file", mode="before")
    @classmethod
    def _normalise_manifest_file(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().replace("\\", "/").lstrip("/")
        return s if s else None


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
    repo_root: str = Field(..., description="Absolute path to the repository workspace root.")
    file_path: str = Field(..., description="Repo-relative path to the file to be edited.")

    # Edit specification
    old_text: str = Field(
        ...,
        min_length=1,
        description="The exact text block to be replaced (must match uniquely).",
    )
    new_text: str = Field(..., description="Replacement text for the matched block.")

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
    issue_id: UUID | None = Field(
        None, description="UUID of the ``VulnerabilityIssue`` this edit addresses."
    )
    rationale: str | None = Field(None, description="Agent-provided explanation for the change.")

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
    unified_diff: str | None = Field(
        None, description="Unified diff of the change (populated on APPLIED/DRY_RUN)."
    )
    lines_added: int = Field(default=0, ge=0)
    lines_removed: int = Field(default=0, ge=0)
    rejection_reason: str | None = Field(
        None, description="Human-readable reason for REJECTED / ERROR status."
    )
    applied_at: datetime | None = Field(
        None, description="UTC timestamp of the write (None for dry-run/rejected)."
    )


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class FailingTest(BaseModel):
    """Details of a single failing test case."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: str | None = None
    line: int | None = Field(None, ge=1)
    message: str | None = None


class ValidationResult(BaseModel):
    """
    Structured outcome of a sandbox validation run (tests, security scans, etc.).

    The Feedback Loop agent parses this to decide whether to retry or deliver.
    """

    model_config = ConfigDict(frozen=False)

    # Identity
    patch_attempt_id: UUID | None = Field(
        None, description="UUID of the ``PatchAttempt`` this validates."
    )
    phase: str = Field(
        ...,
        description="Phase label from the validation profile (install/unit/security_scan).",
    )

    # Result
    status: ValidationStatus
    exit_code: int | None = None
    command: list[str] = Field(
        default_factory=list,
        description="The command array that was executed.",
    )
    stdout_tail: str | None = Field(
        None, description="Last N lines of stdout (truncated for context)."
    )
    stderr_tail: str | None = Field(None, description="Last N lines of stderr.")

    # Structured failure analysis
    failing_tests: list[FailingTest] = Field(default_factory=list)
    dependency_conflict_hints: list[str] = Field(
        default_factory=list,
        description="Extracted dependency conflict messages for retry planning.",
    )
    changed_files: list[str] = Field(
        default_factory=list,
        description="Repo-relative paths of files modified during validation.",
    )

    duration_seconds: float | None = Field(None, ge=0.0)
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# PatchAttempt
# ---------------------------------------------------------------------------


class PatchAttempt(BaseModel):
    """
    A single end-to-end attempt to remediate one ``VulnerabilityIssue``.

    Bundles the edit(s) applied and the validation results for that attempt.
    The Remedy agent uses this to decide whether to retry (max 2â€“3 retries).
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    issue_id: UUID = Field(..., description="UUID of the ``VulnerabilityIssue`` being remediated.")
    attempt_number: int = Field(..., ge=1, description="1-indexed attempt counter.")

    edits: list[EditResult] = Field(
        default_factory=list,
        description="All file edits applied in this attempt.",
    )
    validations: list[ValidationResult] = Field(
        default_factory=list,
        description="Ordered validation results for this attempt.",
    )

    # Summary
    succeeded: bool | None = Field(
        None,
        description="True if all validation phases passed; False if any failed; None if still running.",
    )
    failure_summary: str | None = Field(
        None,
        description="Natural-language summary of why this attempt failed (for the next retry prompt).",
    )

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @property
    def all_validations_passed(self) -> bool:
        """Return whether at least one validation ran and every validation passed."""
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
    issue_id: UUID | None = Field(None, description="Related ``VulnerabilityIssue`` UUID.")
    patch_attempt_id: UUID | None = Field(None)

    kind: TrajectoryEventKind
    agent: str | None = Field(None, description="Agent node name (e.g. 'remedy_agent').")

    # Payload
    summary: str = Field(..., description="Short human-readable description of the step.")
    detail: dict[str, Any] | None = Field(
        None,
        description="Structured detail payload (tool call args, diff stats, etc.).",
    )

    # Cost tracking
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_seconds: float | None = Field(None, ge=0.0)

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def total_tokens(self) -> int:
        """Return the total input and output token count for the event."""
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Triage and QA contracts
# ---------------------------------------------------------------------------


class SystemContext(BaseModel):
    """
    Caller-supplied metadata describing the scan session.

    Passed through the triage pipeline so agents can contextualise their
    verdicts (e.g. prod vs. dev environment, target language, org policies).
    """

    model_config = ConfigDict(frozen=True)

    repo_url: str | None = Field(None, description="HTTPS clone URL of the target repository.")
    base_ref: str | None = Field(None, description="Git branch or commit SHA that was scanned.")
    scanned_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the scan was initiated.",
    )
    environment: str | None = Field(
        None,
        description="Deployment environment label (e.g. 'production', 'staging', 'dev').",
    )
    deployment_os: str | None = Field(
        None, description="Operating system where the app is deployed."
    )
    public_facing: bool | None = Field(
        None, description="Whether the app is public-facing (Internet-accessible)."
    )
    primary_language: str | None = Field(
        None, description="Primary programming language of the codebase."
    )
    deployment_architecture: str | None = Field(
        None, description="Architecture layout, e.g. serverless, containerized, monolith."
    )
    data_sensitivity: str | None = Field(
        None, description="Data sensitivity level, e.g. high, medium, low, public."
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (e.g. team, project, cost-center).",
    )


class CVEEnrichment(BaseModel):
    """
    External threat-intelligence enrichment for a single CVE identifier.

    Populated by ``remediation_engine.triage.enrichment`` from the FIRST EPSS API and the
    CISA Known Exploited Vulnerabilities (KEV) catalogue.  Always returned,
    even when upstream APIs fail â€” safe defaults indicate "unknown risk".
    """

    model_config = ConfigDict(frozen=True)

    cve_id: str = Field(..., description="CVE identifier this record enriches.")

    # FIRST EPSS
    epss: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="EPSS probability score (0â€“1).  0.0 = unknown / API failure.",
    )
    epss_percentile: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="EPSS percentile rank (0â€“1).  0.0 = unknown / API failure.",
    )

    # CISA KEV
    in_kev: bool = Field(
        default=False,
        description="True if the CVE appears in the CISA KEV catalogue.",
    )
    kev_date_added: str | None = Field(
        None,
        description="ISO-8601 date the CVE was added to KEV (e.g. '2023-04-03').",
    )

    # Provenance
    enriched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when enrichment was fetched.",
    )
    enrichment_source: str = Field(
        default="none",
        description="Which data sources contributed: 'epss', 'kev', 'epss+kev', or 'none'.",
    )


class VulnerabilityGroup(BaseModel):
    """
    A set of ``VulnerabilityIssue`` records that share the same vulnerable
    component (SCA) or code location (SAST).

    Produced by ``remediation_engine.triage.grouper``.  The grouper deduplicates
    cross-tool findings and merges overlapping CVE/component/file triples into
    a single authoritative group for downstream triage.
    """

    model_config = ConfigDict(frozen=False)

    # Stable deterministic key
    group_id: str = Field(
        ...,
        description=(
            "Deterministic group key.  "
            "SCA: 'sca:{manifest_file}:{package_name}:{fix_strategy}'. "
            "SAST: 'sast:{file_path}:{rule_id}:{line_start}-{line_end}'."
        ),
    )

    issue_type: IssueType = Field(..., description="SAST or SCA classification of this group.")

    # Component identity
    vulnerable_component: str | None = Field(
        None,
        description="Package name (SCA) or Semgrep rule ID (SAST) shared by all members.",
    )
    file_path: str | None = Field(
        None,
        description="Repo-relative path to the affected file (may be None for SCA without location).",
    )
    file_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Deduplicated repo-relative file paths associated with this group. "
            "For SCA groups this is the set of resolved manifest paths; for SAST "
            "groups this is typically a singleton list containing file_path."
        ),
    )

    # CVE / version metadata (SCA-oriented; empty for pure SAST groups)
    cve_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of CVE identifiers affecting this component.",
    )
    ghsa_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of GHSA identifiers affecting this component.",
    )
    versions: list[str] = Field(
        default_factory=list,
        description="Deduplicated installed versions of the vulnerable package.",
    )

    # Scanner provenance
    sources: list[IssueSource] = Field(
        default_factory=list,
        description="Which scanners contributed at least one issue to this group.",
    )

    # Representative issue
    representative_issue_id: UUID = Field(
        ...,
        description=(
            "UUID of the member issue chosen as the canonical finding for triage. "
            "Chosen by: fixed_version present > most fields populated > first seen."
        ),
    )

    # All member issues
    issues: list[VulnerabilityIssue] = Field(
        default_factory=list,
        description="All ``VulnerabilityIssue`` records that belong to this group.",
    )
    localized_issues: list[LocalizedIssue] = Field(
        default_factory=list,
        description=(
            "All pre-group localization results associated with this group. "
            "Primarily populated for SCA groups in the shift-left flow."
        ),
    )
    fix_plan: FixPlan | None = Field(
        None,
        description=(
            "Unified remediation plan for the group. For SCA groups this is the "
            "group-level plan derived from member issue fix plans."
        ),
    )

    # Enrichment (attached after grouping, before triage)
    enrichment: CVEEnrichment | None = Field(
        None,
        description="Threat-intel enrichment for the primary CVE of this group.",
    )
    is_reachable: bool | None = Field(
        default=None,
        description=(
            "Reachability analysis result for SCA groups. "
            "True when the package is imported in app code; False when it is a direct "
            "dependency but never imported; None when reachability is unknown."
        ),
    )

    grouped_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalise_group_file_path(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().replace("\\", "/").lstrip("/")
        return s if s else None

    @field_validator("file_paths", mode="before")
    @classmethod
    def _normalise_group_file_paths(cls, value: Any) -> list[str]:
        if value is None:
            return []

        raw_values = value if isinstance(value, list) else [value]
        normalised: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            if raw is None:
                continue
            path = str(raw).strip().replace("\\", "/").lstrip("/")
            if not path or path in seen:
                continue
            normalised.append(path)
            seen.add(path)
        return normalised

    @model_validator(mode="after")
    def _sync_group_file_path_fields(self) -> VulnerabilityGroup:
        if self.file_path and self.file_path not in self.file_paths:
            self.file_paths = [self.file_path, *self.file_paths]
        elif not self.file_path and self.file_paths:
            self.file_path = self.file_paths[0]
        return self


class TriageResult(BaseModel):
    """
    Deterministic or LLM-assisted triage verdict for one ``VulnerabilityGroup``.

    Produced by ``remediation_engine.triage.agent``.  The triage agent may use an LLM for
    initial reasoning, but deterministic guardrails (KEV, EPSS, original
    severity) are always applied afterwards to prevent the LLM from
    under-ranking exploitable issues.

    Invariant: ``false_positive_reason`` MUST be set when ``is_valid=False``.
    """

    model_config = ConfigDict(frozen=False)

    chain_of_thought: str = Field(
        default="",
        description="Step-by-step chain of thought reasoning before deciding the triage fields.",
    )
    group_id: str = Field(..., description="Matches ``VulnerabilityGroup.group_id``.")

    # Validity
    is_valid: bool = Field(
        ...,
        description=(
            "False if the group is assessed as a false positive or out-of-scope. "
            "Only set to False when there is explicit, specific evidence."
        ),
    )
    false_positive_reason: str | None = Field(
        None,
        description="Required when is_valid=False.  Must explain the specific evidence.",
    )

    # Priority
    original_severity: Severity = Field(
        default=Severity.UNKNOWN,
        description=(
            "Original scanner-reported severity retained from the source finding "
            "before any contextual triage or guardrail adjustments."
        ),
    )
    revised_priority: Severity = Field(
        ...,
        description=(
            "Revised priority using the canonical Severity enum.  "
            "Guardrails clamp: KEV â†’ CRITICAL; EPSSâ‰¥0.5 or HIGH/CRITICAL original â†’ at least HIGH."
        ),
    )
    is_unreachable_code: bool = Field(
        default=False,
        description=(
            "True when reachability analysis shows the vulnerable package is not "
            "imported by the application source code."
        ),
    )
    priority_reasoning: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the revised priority.",
    )
    validity_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 to 1.0 representing the certainty of the is_valid decision based on hard evidence."
        ),
    )
    priority_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 to 1.0 representing the certainty of the revised_priority "
            "based on context and threat intel."
        ),
    )

    # Recommendation
    recommended_issue_id: UUID = Field(
        ...,
        description="UUID of the ``VulnerabilityIssue`` recommended for remediation.",
    )

    # Provenance
    triage_method: str = Field(
        ...,
        description="'deterministic' if no LLM was used; 'llm' if structured LLM output was applied.",
    )

    triaged_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @model_validator(mode="after")
    def _require_fp_reason(self) -> TriageResult:
        if not self.is_valid and not self.false_positive_reason:
            raise ValueError("false_positive_reason is required when is_valid=False.")
        return self


# ---------------------------------------------------------------------------
# Phase 5 Remedy refactor contracts
# ---------------------------------------------------------------------------


class WorkaroundPhase(str, Enum):
    """Workflow phase for workaround subagent execution."""

    INITIAL_MITIGATION = "initial_mitigation"
    QA_REGRESSION_REPAIR = "qa_regression_repair"


class WorkaroundExecutionPhase(str, Enum):
    """Execution lifecycle phase for workaround subagent iterations."""

    INVESTIGATE = "INVESTIGATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VALIDATE = "VALIDATE"


class WorkaroundValidationStatus(str, Enum):
    """Outcome status of a workaround validation run."""

    PASS = "PASS"
    CODE_FAILURE = "CODE_FAILURE"
    INFRA_FAILURE = "INFRA_FAILURE"
    BLOCKED = "BLOCKED"


class WorkaroundValidationResult(BaseModel):
    """Structured validation result for workaround gate runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_status: WorkaroundValidationStatus
    syntax: str = ""
    typecheck: str = ""
    lint: str = ""
    runtime_smoke: str = ""
    targeted_test: str = ""
    targeted_test_file: str | None = None
    alternative_used: bool = False
    alternative_test_mapping_details: dict[str, dict[str, str]] = Field(default_factory=dict)
    validated_files: list[str] = Field(default_factory=list)
    failure_category: FailureCategory | None = None
    infrastructure_diagnostics: str | None = None


class QAFailureEvidence(BaseModel):
    """Exact diagnostic evidence extracted from a failed QA evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_diagnostics: list[str] = Field(default_factory=list)
    failed_tests: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    raw_excerpt: str = Field(default="")
    attempt_id: str = Field(default="")
    task_revision: int = Field(default=0, ge=0)


class WorkaroundContext(BaseModel):
    """Supervisor-provided phase and evidence context for workaround attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: WorkaroundPhase = Field(default=WorkaroundPhase.INITIAL_MITIGATION)
    vulnerability_mechanism: str = Field(default="")
    qa_evidence: QAFailureEvidence | None = Field(default=None)
    no_fix_stage: NoFixMitigationStage | None = Field(default=None)
    reset_prior_stage_workspace: bool = Field(
        default=False,
        description=(
            "Restore the task-local pre-stage workspace before executing this "
            "attempt. Used when a NO_FIX package-removal attempt advances to "
            "vulnerable-code removal."
        ),
    )


class QAEvaluation(BaseModel):
    """Structured QA Critic verdict for a single remediation task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    passed: bool
    failure_category: FailureCategory | None = Field(
        None,
        description="Required when passed=False.",
    )
    retry_feedback: str | None = Field(
        None,
        description="Required retry guidance when passed=False.",
    )
    failure_evidence: QAFailureEvidence | None = Field(
        None,
        description="Structured failure evidence when passed=False.",
    )

    @model_validator(mode="after")
    def _check_pass_fail_payload(self) -> QAEvaluation:
        if self.passed:
            if (
                self.failure_category is not None
                or self.retry_feedback is not None
                or self.failure_evidence is not None
            ):
                raise ValueError(
                    "passed=True requires failure_category=None, retry_feedback=None, and failure_evidence=None."
                )
            return self

        if self.failure_category is None:
            raise ValueError("passed=False requires a non-null failure_category.")
        if not self.retry_feedback or not self.retry_feedback.strip():
            raise ValueError("passed=False requires a non-empty retry_feedback.")
        return self


class BatchQAResult(BaseModel):
    """Structured output from the map-reduce batch judge phase."""

    model_config = ConfigDict(frozen=True)

    holistic_report: str = Field(
        ...,
        min_length=1,
        description=(
            "Free-form markdown holistic report synthesizing all individual "
            "investigations into a unified narrative, listing responsible, "
            "possibly responsible, and exonerated groups."
        ),
    )
    evaluations: list[QAEvaluation] = Field(
        default_factory=list,
        description="Exactly one QAEvaluation per vulnerability group in the batch.",
    )


class AgentActionSummary(BaseModel):
    """Condensed subagent outcome stored in supervisor state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    attempt_id: str | None = None
    task_revision: int | None = Field(default=None, ge=0)
    instruction_digest: str | None = None
    status: AgentActionStatus
    summary: str = Field(..., min_length=1)

    @field_validator("summary")
    @classmethod
    def _summary_must_be_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary must be a non-empty natural-language string.")
        return cleaned


class TaskAttemptSnapshot(BaseModel):
    """Immutable supervisor commit describing one worker/QA attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    task_id: str = Field(..., min_length=1)
    state_revision: int = Field(default=0, ge=0)
    task_revision: int = Field(default=0, ge=0)
    attempt_number: int = Field(default=1, ge=1)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    no_fix_stage: NoFixMitigationStage | None = Field(default=None)
    selected_version: str | None = None
    instruction: str = Field(..., min_length=1)
    instruction_digest: str = Field(..., min_length=1)
    dispatch_node: Literal["update_subagent", "workaround_subagent", "qa_critic"]
    plan_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    workaround_context: WorkaroundContext | None = Field(default=None)


class UpdateRetryDiagnostics(BaseModel):
    """Structured retry evidence emitted by the update subagent per task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    committed_attempt_id: str | None = None
    security_floor: str | None = None
    registry_query_performed: bool = False
    attempted_versions: list[str] = Field(default_factory=list)
    executed_versions: list[str] = Field(default_factory=list)
    candidate_versions_considered: list[str] = Field(default_factory=list)
    selected_version: str | None = None
    latest_version_seen: str | None = None
    used_overrides: bool = False
    package_abandoned: bool = False
    exhausted_update_path: bool = False
    failure_reason: str = ""
    reasoning_summary: str = ""
    instruction_digest: str | None = None

    @field_validator(
        "attempted_versions",
        "executed_versions",
        "candidate_versions_considered",
        mode="before",
    )
    @classmethod
    def _normalize_version_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("version lists must be lists of strings.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("version lists must contain only strings.")
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @field_validator("selected_version", "latest_version_seen", mode="before")
    @classmethod
    def _normalize_optional_version(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("version fields must be strings when provided.")
        cleaned = value.strip()
        return cleaned or None

    @field_validator("failure_reason", "reasoning_summary", mode="before")
    @classmethod
    def _normalize_text_field(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("text fields must be strings.")
        return value.strip()


class WorkerExecutionDiagnostics(BaseModel):
    """Execution-only evidence reported by the update/workaround worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempted_versions: list[str] = Field(default_factory=list)
    executed_versions: list[str] = Field(default_factory=list)
    validation_calls: int = Field(default=0, ge=0)
    validation_input_errors: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of validation requests rejected during preflight before any validation gate ran."
        ),
    )
    validation_passed: bool = False
    failure_reason: str = ""
    per_gate_results: dict[str, Any] = Field(default_factory=dict)
    final_selected_targeted_test: str | None = None
    original_to_alternative_test_mapping: dict[str, str] = Field(default_factory=dict)
    alternative_test_mapping_evidence: dict[str, list[str]] = Field(default_factory=dict)
    alternative_test_mapping_details: dict[str, dict[str, str]] = Field(default_factory=dict)
    validated_files: list[str] = Field(default_factory=list)
    infrastructure_failure_details: str | None = None


class WorkaroundPlannedReplacement(BaseModel):
    """A typed single replacement planned within a WorkaroundEditSet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1)
    old_text: str = Field(...)
    new_text: str = Field(...)
    expected_occurrences: int = Field(default=1, ge=1)
    symbol_name: str | None = None


class WorkaroundEdit(BaseModel):
    """Recorded deterministic search-replace edit for workaround replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1)
    old_text: str = Field(...)
    new_text: str = Field(...)
    symbol_name: str | None = None
    patch_id: str = Field(default="")
    replacement_index: int = Field(default=0, ge=0)
    expected_occurrences: int = Field(default=1, ge=1)
    edit_index: int = Field(default=0, ge=0)
    timestamp: str = Field(default="")


class WorkaroundEditSet(BaseModel):
    """An atomic collection of edits applied in one workaround iteration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    patch_id: str = Field(..., min_length=1)
    plan_revision: int = Field(default=1, ge=1)
    iteration: int = Field(default=1, ge=1)
    affected_files: list[str] = Field(default_factory=list)
    replacements: list[WorkaroundEdit] = Field(default_factory=list)


class WorkaroundReplayPlan(BaseModel):
    """Cumulative replay plan for workaround retries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    pre_attempt_snapshots: dict[str, str] = Field(default_factory=dict)
    pre_attempt_absent_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Repository-relative files that did not exist at the stage baseline. "
            "They are removed before replay when a later stage resets the workspace."
        ),
    )
    successful_edit_sets: list[WorkaroundEditSet] = Field(default_factory=list)
    investigation_findings: dict[str, Any] = Field(default_factory=dict)
    source_attempt_id: str = Field(default="")
    security_invariants: list[str] = Field(default_factory=list)
    diagnosed_root_causes: list[str] = Field(default_factory=list)
    planned_targets: list[str] = Field(default_factory=list)
    validated_files: list[str] = Field(default_factory=list)
    validation_calls: int = Field(default=0, ge=0)
    validation_input_errors: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of validation requests rejected during preflight before any validation gate ran."
        ),
    )
    per_gate_results: dict[str, Any] = Field(default_factory=dict)
    final_selected_targeted_test: str | None = None
    original_to_alternative_test_mapping: dict[str, str] = Field(default_factory=dict)
    alternative_test_mapping_evidence: dict[str, list[str]] = Field(default_factory=dict)
    alternative_test_mapping_details: dict[str, dict[str, str]] = Field(default_factory=dict)
    infrastructure_failure_details: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_successful_edits(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data_copy = dict(data)
            legacy_edits = data_copy.pop("successful_edits", None)
            if legacy_edits and not data_copy.get("successful_edit_sets"):
                sets: list[dict[str, Any]] = []
                for idx, item in enumerate(legacy_edits):
                    edit_dict = (
                        item.model_dump()
                        if isinstance(item, BaseModel)
                        else dict(item)
                        if isinstance(item, dict)
                        else {}
                    )
                    if edit_dict:
                        p_id = edit_dict.get("patch_id") or f"patch_legacy_{idx}"
                        sets.append(
                            {
                                "patch_id": p_id,
                                "plan_revision": 1,
                                "iteration": idx + 1,
                                "affected_files": [edit_dict.get("file_path", "")],
                                "replacements": [edit_dict],
                            }
                        )
                data_copy["successful_edit_sets"] = sets
            return data_copy
        return data

    @property
    def successful_edits(self) -> list[WorkaroundEdit]:
        return [edit for edit_set in self.successful_edit_sets for edit in edit_set.replacements]


class WorkerAttemptResult(BaseModel):
    """Worker result correlated to the supervisor's committed attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    task_revision: int = Field(default=0, ge=0)
    status: AgentActionStatus
    executed_versions: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    action_summary: AgentActionSummary | None = None
    execution_diagnostics: WorkerExecutionDiagnostics = Field(
        default_factory=WorkerExecutionDiagnostics
    )
    instruction_digest: str = Field(..., min_length=1)
    replay_plan: WorkaroundReplayPlan | None = None
    errors: list[str] = Field(default_factory=list)


class QAAttemptResult(BaseModel):
    """QA result correlated to the attempt that produced the changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    task_revision: int = Field(default=0, ge=0)
    evaluation: QAEvaluation
    investigation_report: str = ""
    errors: list[str] = Field(default_factory=list)


class StateConsistencyEvent(BaseModel):
    """Deduplicated state-reconciliation diagnostic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    error_code: str = Field(..., min_length=1)
    task_id: str | None = None
    expected_attempt_id: str | None = None
    received_attempt_id: str | None = None
    action: Literal["ignored", "repaired", "replanned"]
    details: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SupervisorRetryPlan(BaseModel):
    """Authoritative planner decision committed before supervisor routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    plan_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_task_revision: int = Field(default=0, ge=0)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    selected_version: str | None = None
    attempted_versions: list[str] = Field(default_factory=list)
    candidate_versions_considered: list[str] = Field(default_factory=list)
    latest_version_seen: str | None = None
    exhausted_update_path: bool = False
    package_abandoned: bool = False
    action: Literal["retry_update", "pivot_workaround"] = "retry_update"
    exact_instruction: str = Field(default="", min_length=1)

    @field_validator(
        "attempted_versions",
        "candidate_versions_considered",
        mode="before",
    )
    @classmethod
    def _normalize_plan_version_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("planner version lists must be lists of strings.")
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("planner version lists must contain only strings.")
            normalized = item.strip().lstrip("vV")
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @field_validator("selected_version", "latest_version_seen", mode="before")
    @classmethod
    def _normalize_plan_optional_version(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("planner version fields must be strings when provided.")
        normalized = value.strip().lstrip("vV")
        return normalized or None

    @field_validator("exact_instruction")
    @classmethod
    def _normalize_plan_instruction(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("exact_instruction must be non-empty.")
        return normalized


class SupervisorDecision(BaseModel):
    """
    Structured LLM output from the Supervisor Node.

    Controls hub-and-spoke routing in the Phase 5 orchestrator graph.
    Pydantic validators enforce routing invariants:
    - ``workaround_subagent`` requires exactly one ``target_task_id``.
    - ``update_subagent`` requires 1-10 ``target_task_ids``.
    - ``qa_critic`` requires one or more ``target_task_ids`` for batch QA.
    - ``triage`` is a graph-level handoff and does not target a task.
    - ``teardown`` requires empty ``target_task_ids``.
    - ``unfixable_task_ids`` and ``target_task_ids`` must not overlap.
    """

    model_config = ConfigDict(frozen=True)

    next_node: Literal[
        "update_subagent", "workaround_subagent", "qa_critic", "triage", "teardown"
    ] = Field(
        ...,
        description="The next node to route to in the orchestrator graph.",
    )
    updated_task_strategies: dict[str, RoutingStrategy] = Field(
        default_factory=dict,
        description=(
            "Requested strategy pivots keyed by parent task_id. "
            "The supervisor realizes these as child-task spawns rather than mutating the parent task in place."
        ),
    )
    target_task_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Task IDs to send to the next worker node. "
            "One or more for qa_critic. "
            "Empty for teardown. "
            "Exactly one entry for workaround_subagent. "
            "One to ten entries for update_subagent."
        ),
    )
    unfixable_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs that have hit MAX_RETRIES and should be marked unfixable.",
    )
    new_constraints: list[str] = Field(
        default_factory=list,
        description="New constraint strings to append to the constraints ledger.",
    )
    feedback_by_task: dict[str, str] = Field(
        default_factory=dict,
        description="Task-specific retry guidance keyed by task_id.",
    )
    instructions: str = Field(
        ...,
        min_length=1,
        description=(
            "Supervisor audit/routing rationale for this decision. "
            "This is not the authoritative worker instruction for retry-bound tasks."
        ),
    )
    decision_reason: str = Field(
        ...,
        min_length=1,
        description="Concise audit explanation of why this routing decision was made.",
    )
    revised_instructions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Supervisor-revised instruction text per task_id. "
            "Replaces the task's current instruction when non-empty. "
            "All values must be non-empty strings. "
            "Retry-bound worker dispatches rely on these task-specific instructions."
        ),
    )
    spawn_requests: list[TaskSpawnRequest] = Field(
        default_factory=list,
        description=(
            "Requests to spawn new child tasks. "
            "The Python guardrail layer materializes actual RemediationTask objects."
        ),
    )
    task_status_updates: dict[str, TaskStatus] = Field(
        default_factory=dict,
        description=(
            "Manual status overrides keyed by task_id. "
            "Only QA_PASSED and UNFIXABLE are permitted; all other values are rejected."
        ),
    )

    @model_validator(mode="after")
    def _validate_routing_invariants(self) -> SupervisorDecision:
        node = self.next_node
        targets = self.target_task_ids
        unfixable = self.unfixable_task_ids

        if node == "workaround_subagent" and len(targets) != 1:
            raise ValueError(
                f"workaround_subagent requires exactly 1 target_task_id, got {len(targets)}."
            )
        if node == "update_subagent" and len(targets) < 1:
            raise ValueError("update_subagent requires at least 1 target_task_id.")
        if node == "update_subagent" and len(targets) > 10:
            raise ValueError(
                f"update_subagent supports at most 10 target_task_ids, got {len(targets)}."
            )
        if node == "qa_critic" and len(targets) < 1:
            raise ValueError("qa_critic requires at least 1 target_task_id.")
        if node == "teardown" and targets:
            raise ValueError(f"{node} must have empty target_task_ids, got {targets}.")
        overlap = set(unfixable) & set(targets)
        if overlap:
            raise ValueError(
                f"unfixable_task_ids and target_task_ids must not overlap. Overlap: {overlap}"
            )
        # Validate revised_instructions keys are non-empty strings
        for k, v in self.revised_instructions.items():
            if not k.strip():
                raise ValueError("revised_instructions keys must be non-empty task IDs.")
            if not v.strip():
                raise ValueError(f"revised_instructions['{k}'] must be a non-empty instruction.")
        # Validate task_status_updates values are only QA_PASSED or UNFIXABLE
        _allowed = {TaskStatus.QA_PASSED, TaskStatus.UNFIXABLE}
        for tid, status in self.task_status_updates.items():
            if status not in _allowed:
                raise ValueError(
                    f"task_status_updates['{tid}'] = '{status}' is not allowed; "
                    "only QA_PASSED and UNFIXABLE may be set by the LLM."
                )
        return self


# ---------------------------------------------------------------------------
# Phase 5 Task Queue contracts
# ---------------------------------------------------------------------------


class TaskSpawnRequest(BaseModel):
    """
    A request from the supervisor to spawn a new child ``RemediationTask``.

    The supervisor submits these inside ``SupervisorDecision.spawn_requests``;
    the Python guardrail layer materializes the actual ``RemediationTask`` by
    assigning ``task_id``, ``parent_group_id``, ``status``, ``retry_count``,
    and ``ancestry_depth``. The LLM never creates raw ``RemediationTask``
    objects directly.
    """

    model_config = ConfigDict(frozen=True)

    parent_task_id: str = Field(
        ...,
        min_length=1,
        description="The task_id of the parent task that is spawning this child.",
    )
    strategy: RoutingStrategy = Field(
        ...,
        description="Routing strategy for the child task.",
    )
    instruction: str = Field(
        ...,
        min_length=1,
        description="Exact instruction for the child task worker agent.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Audit reason explaining why this child task is being spawned.",
    )


# ---------------------------------------------------------------------------
# Phase 5 Task Queue contracts
# ---------------------------------------------------------------------------


class RemediationTask(BaseModel):
    """
    A single unit of work in the Phase 5 task queue.

    Created by the supervisor from a ``VulnerabilityGroup`` and carried
    through the orchestrator lifecycle.  Tasks are the primary key for
    supervisor decisions, QA evaluations, and action summaries.
    """

    model_config = ConfigDict(frozen=False)

    task_id: str = Field(
        ...,
        min_length=1,
        description="Unique task identifier (e.g. 'task-1').",
    )
    task_revision: int = Field(
        default=0,
        ge=0,
        description="Monotonic revision of the supervisor-committed task input.",
    )
    current_attempt_id: str | None = Field(
        default=None,
        description="Attempt snapshot currently authorized to produce results.",
    )
    parent_group_id: str = Field(
        ...,
        min_length=1,
        description="The ``VulnerabilityGroup.group_id`` this task remediates.",
    )
    parent_task_id: str | None = Field(
        default=None,
        description=(
            "The task_id of the parent task that spawned this task. "
            "None for initial (depth-0) tasks."
        ),
    )
    strategy: RoutingStrategy = Field(
        ...,
        description="Routing strategy: VERSION_BUMP or CODE_WORKAROUND.",
    )
    strategy_stage: SCARemediationStage = Field(
        default=SCARemediationStage.OSV_MINIMUM,
        description="Current ordered SCA remediation stage for this task.",
    )
    no_fix_stage: NoFixMitigationStage | None = Field(
        default=None,
        description=(
            "Current mitigation stage for a NO_FIX vulnerability group. None for non-NO_FIX tasks."
        ),
    )
    selected_version: str | None = Field(
        default=None,
        description="Supervisor-selected version for the current update stage.",
    )
    exhausted_update_path: bool = Field(
        default=False,
        description="True when all registry-guided update stages are exhausted.",
    )
    instruction: str = Field(
        default="",
        description="Exact supervisor-written instruction for the worker agent.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current lifecycle status of this task.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of retry dispatches after a worker or QA failure. "
            "For NO_FIX tasks, a failed mitigation stage consumes one retry "
            "even when the worker fails before producing a QA envelope."
        ),
    )
    ancestry_depth: int = Field(
        default=0,
        ge=0,
        description="How many parent tasks spawned this task (0 = initial task).",
    )
