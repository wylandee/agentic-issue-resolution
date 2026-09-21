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
from enum import Enum, StrEnum
from pathlib import PureWindowsPath
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

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.solver_models import SolverRemediationPlan

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    """Canonical severity levels across SAST and SCA findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class IssueSource(StrEnum):
    """The originating scanner / data source."""

    SEMGREP = "semgrep"
    ODC = "odc"  # OWASP Dependency-Check
    MANUAL = "manual"  # Injected by a human or eval harness
    SYNTHETIC = "synthetic"  # Generated for testing / benchmarks


class IssueType(StrEnum):
    """Broad classification of the finding."""

    SAST = "sast"  # Static Application Security Testing code finding
    SCA = "sca"  # Software Composition Analysis dependency finding


class ASTNodeType(StrEnum):
    """Abstract Syntax Tree node kinds used by the SAST code locator."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    ARROW_FUNCTION = "arrow_function"
    CALL_EXPRESSION = "call_expression"
    VARIABLE_DECLARATION = "variable_declaration"
    IMPORT_STATEMENT = "import_statement"
    UNKNOWN = "unknown"


class FixPlanStatus(StrEnum):
    """Outcome of the fix-planner waterfall for one SCA finding."""

    VERSION_FOUND = "version_found"  # A safe pinned version was identified
    WORKAROUND_FOUND = "workaround_found"  # No upstream fix; web snippets found
    NO_FIX = "no_fix"  # All strategies exhausted, nothing found


class FailureCategory(StrEnum):
    """Strict QA failure categories used for supervisor retry routing."""

    SECURITY_FLAG = "security_flag"
    PEER_CONFLICT = "peer_conflict"
    BREAKING_CHANGE = "breaking_change"


class ScanScope(str, Enum):  # noqa: UP042
    """Requested or effective scope of an OWASP Dependency-Check run."""

    TARGETED = "targeted"
    FULL = "full"


class ScanFallbackReason(str, Enum):  # noqa: UP042
    """Reason a requested targeted scan used the full-scan fallback."""

    UNSUPPORTED_PACKAGE_MANAGER = "unsupported_package_manager"
    MISSING_LOCKFILE = "missing_lockfile"
    INVALID_LOCKFILE = "invalid_lockfile"
    NO_MATCHING_TARGET = "no_matching_target"
    MULTIPLE_TARGETS = "multiple_targets"
    INCOMPLETE_CLOSURE = "incomplete_closure"
    TARGETED_SCAN_FAILED = "targeted_scan_failed"
    TARGETED_REPORT_UNPARSEABLE = "targeted_report_unparseable"


class RoutingStrategy(StrEnum):
    """Strict supervisor routing strategies for vulnerability groups."""

    VERSION_BUMP = "version_bump"
    CODE_WORKAROUND = "code_workaround"


class TacticalStrategy(StrEnum):
    """Action strategies proposed by the tactical Supervisor."""

    VERSION_BUMP = "version_bump"
    PACKAGE_OVERRIDE = "package_override"
    CODE_WORKAROUND = "code_workaround"


class QAPolicy(StrEnum):
    """Supervisor-owned policy that defines the QA gates for an attempt."""

    VERSION_BUMP = "version_bump"
    INITIAL_CODE_WORKAROUND = "initial_code_workaround"
    MIGRATION_CODE_WORKAROUND = "migration_code_workaround"
    MITIGATION_CODE_WORKAROUND = "mitigation_code_workaround"
    NO_FIX_PACKAGE_REMOVAL = "no_fix_package_removal"
    NO_FIX_CODE_REMOVAL = "no_fix_code_removal"


class ScannerExecutionStatus(StrEnum):
    """Trust status of the deterministic Dependency-Check execution."""

    SUCCESS = "success"
    DOCKER_UNAVAILABLE = "docker_unavailable"
    TIMEOUT = "timeout"
    UNPARSEABLE = "unparseable"
    NOT_RUN = "not_run"


class DependencyEvidenceStatus(StrEnum):
    """Deterministic status of dependency manifest and graph evidence."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    INCONCLUSIVE = "inconclusive"


class SecurityReviewVerdict(StrEnum):
    """Verdict emitted by the semantic security review."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class TestAttributionVerdict(StrEnum):
    """LLM attribution of a shared test-suite failure."""

    __test__ = False

    RESPONSIBLE = "responsible"
    EXONERATED = "exonerated"
    INCONCLUSIVE = "inconclusive"


class SCARemediationStage(StrEnum):
    """Ordered remediation stages for an SCA version-bump task."""

    OSV_MINIMUM = "osv_minimum"
    NPM_SAME_MAJOR = "npm_same_major"
    NPM_LATEST = "npm_latest"
    PACKAGE_OVERRIDE = "package_override"
    CODE_WORKAROUND = "code_workaround"


class NoFixMitigationStage(StrEnum):
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
MAX_MULTI_PACKAGE_ACTION_SIZE: int = 30


class AgentActionStatus(StrEnum):
    """Strict terminal statuses returned by subagents."""

    SUCCESS = "success"
    SURRENDER = "surrender"


class TaskStatus(StrEnum):
    """Lifecycle status for one RemediationTask in the task queue."""

    PENDING = "pending"  # Not yet dispatched to any worker
    OPTIMISTICALLY_FIXED = "optimistically_fixed"  # Worker succeeded; awaiting QA verdict
    QA_PASSED = "qa_passed"  # QA explicitly passed; terminal success
    NEEDS_RETRY = "needs_retry"  # QA failed; will be re-routed by supervisor
    UNFIXABLE = "unfixable"  # Max retries exhausted; terminal failure
    INCONCLUSIVE = "inconclusive"  # Evidence was invalid or could not be classified
    PIVOTED = "pivoted"  # Parent attempt was superseded by a spawned child task


class TaskDependencyKind(StrEnum):
    """Relationship types used by strategic task clusters."""

    PEER = "peer"
    WORKSPACE = "workspace"
    RUNTIME = "runtime"


def _trim_required_contract_text(value: Any, field_name: str) -> str:
    """Return a trimmed non-empty contract string.

    Args:
        value: Candidate value supplied to a contract field.
        field_name: Field name used in validation errors.

    Returns:
        The trimmed string.

    Raises:
        ValueError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _trim_optional_contract_text(value: Any, field_name: str) -> str | None:
    """Return a trimmed optional contract string, treating blank as absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided.")
    normalized = value.strip()
    return normalized or None


def _normalize_contract_string_list(
    value: Any,
    field_name: str,
    *,
    reject_duplicates: bool = True,
) -> list[str]:
    """Normalize a list of non-empty strings while preserving input order."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _trim_required_contract_text(item, field_name)
        if normalized in seen:
            if reject_duplicates:
                raise ValueError(f"{field_name} must not contain duplicates: {normalized!r}.")
            result.append(normalized)
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_relative_file_hint(value: Any) -> str:
    """Normalize and validate one relative POSIX-style file hint.

    File hints are deliberately validated only as safe relative paths.  They
    do not authorize a worker to access or mutate a file.
    """
    normalized = _trim_required_contract_text(value, "target_files_hint")
    normalized = normalized.replace("\\", "/")
    if "\x00" in normalized:
        raise ValueError("target_files_hint must not contain NUL bytes.")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("target_files_hint must contain relative paths only.")
    segments = normalized.split("/")
    if ".." in segments:
        raise ValueError("target_files_hint must not contain parent traversal.")
    segments = [segment for segment in segments if segment not in {"", "."}]
    if not segments:
        raise ValueError("target_files_hint must contain a non-empty relative path.")
    return "/".join(segments)


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

    # Dependency ancestry / parent-first transitive remediation context
    dependency_ancestry: list[str] = Field(
        default_factory=list,
        description=(
            "Package names in the resolved dependency chain, ordered from the "
            "outermost package toward the vulnerable leaf."
        ),
    )
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved versions keyed by package name when the scanner supplied them.",
    )
    declaration_type: str | None = Field(
        None,
        description=(
            "Manifest declaration section for the localized leaf dependency "
            "(dependencies, devDependencies, peerDependencies, or optionalDependencies)."
        ),
    )
    parent_package_name: str | None = Field(
        None,
        description=(
            "Nearest ancestor in dependency_ancestry that is directly declared "
            "in the editable manifest."
        ),
    )
    parent_package_version: str | None = Field(
        None,
        description="Resolved version of parent_package_name, when available.",
    )
    parent_declaration_type: str | None = Field(
        None,
        description="Manifest declaration section containing parent_package_name.",
    )
    parent_manifest_line: int | None = Field(
        None,
        ge=1,
        description="1-indexed manifest line for the directly declared parent dependency.",
    )
    parent_manifest_snippet: str | None = Field(
        None,
        description="Manifest snippet centred on the directly declared parent dependency.",
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

    @field_validator("dependency_ancestry", mode="before")
    @classmethod
    def _normalise_dependency_ancestry(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            normalized = str(item).strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result

    @field_validator("dependency_versions", mode="before")
    @classmethod
    def _normalise_dependency_versions(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("dependency_versions must be a mapping of package names to versions.")
        return {
            str(package).strip(): str(version).strip()
            for package, version in value.items()
            if str(package).strip() and str(version).strip()
        }


class DependencyParentContext(BaseModel):
    """Typed context for a directly declared parent of a transitive finding.

    Parent context is evidence used by the Supervisor to choose the first
    remediation stage. It is deliberately separate from the group's identity
    so duplicate scanner paths for the same vulnerable child do not create
    independent remediation tasks.
    """

    model_config = ConfigDict(frozen=True)

    package_name: str = Field(..., min_length=1, description="Directly declared parent package.")
    package_version: str | None = Field(
        None, description="Resolved installed version of the parent package, when available."
    )
    declaration_type: str | None = Field(
        None, description="Manifest declaration section containing the parent package."
    )
    manifest_file: str | None = Field(
        None, description="Repo-relative manifest containing the parent declaration."
    )
    manifest_line: int | None = Field(
        None, ge=1, description="1-indexed manifest line for the parent declaration."
    )
    manifest_snippet: str | None = Field(
        None, description="Bounded manifest snippet around the parent declaration."
    )
    dependency_ancestry: list[str] = Field(
        default_factory=list, description="Resolved ancestry from parent to vulnerable leaf."
    )
    dependency_versions: dict[str, str] = Field(
        default_factory=dict, description="Resolved versions keyed by dependency name."
    )


# ---------------------------------------------------------------------------
# EditRequest / EditResult
# ---------------------------------------------------------------------------


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


class PackageFixPlanCandidate(BaseModel):
    """One issue-level fix plan retained inside a package-centric group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue_id: UUID = Field(..., description="Issue whose planner produced this candidate.")
    plan: FixPlan = Field(..., description="Issue-level remediation plan.")


class VulnerabilityGroup(BaseModel):
    """
    A set of ``VulnerabilityIssue`` records that share the same vulnerable
    component (SCA) or code location (SAST).

    Produced by ``remediation_engine.triage.grouper``.  The grouper deduplicates
    cross-tool findings and merges overlapping CVE/component/file triples into
    a single authoritative group for downstream triage. The Supervisor may
    additionally create an ``is_synthetic`` coordination group for a direct
    dependency occurrence that has no active scanner finding.
    """

    model_config = ConfigDict(frozen=False)

    # Stable deterministic key
    group_id: str = Field(
        ...,
        description=(
            "Deterministic group key.  "
            "SCA: 'sca:{manifest_file}:{package_name}'. Parent "
            "contexts are retained as evidence, not included in the identity. "
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
    is_synthetic: bool = Field(
        default=False,
        description=(
            "True for a Supervisor-generated coordination group representing a "
            "package with no active scanner finding. Synthetic groups are included "
            "in portfolio actions but are not security findings."
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
    dependency_ancestry: list[str] = Field(
        default_factory=list,
        description="Resolved dependency chain for the grouped SCA finding.",
    )
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved dependency versions from the grouped SCA finding.",
    )
    parent_package_name: str | None = Field(
        None,
        description="Nearest directly declared parent package for a transitive finding.",
    )
    parent_package_version: str | None = Field(
        None,
        description="Installed/resolved version of the directly declared parent package.",
    )
    parent_declaration_type: str | None = Field(
        None,
        description="Manifest declaration section for the directly declared parent package.",
    )
    parent_contexts: list[DependencyParentContext] = Field(
        default_factory=list,
        description=(
            "All directly declared parent contexts observed for this vulnerable child. "
            "The group identity is independent of this list."
        ),
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
    fix_plan_candidates: list[PackageFixPlanCandidate] = Field(
        default_factory=list,
        description=(
            "All issue-level remediation plans retained for this package group. "
            "The Supervisor selects the active candidate; grouping does not discard "
            "alternative strategies."
        ),
    )
    fix_plan: FixPlan | None = Field(
        None,
        description=(
            "Compatibility summary of the package plans. New orchestration code "
            "must use fix_plan_candidates for strategy selection."
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


class WorkaroundPhase(StrEnum):
    """Workflow phase for workaround subagent execution."""

    INITIAL_MITIGATION = "initial_mitigation"
    QA_REGRESSION_REPAIR = "qa_regression_repair"


class WorkaroundExecutionPhase(StrEnum):
    """Execution lifecycle phase for workaround subagent iterations."""

    INVESTIGATE = "INVESTIGATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VALIDATE = "VALIDATE"


class ScratchpadScope(StrEnum):
    """Scope for ephemeral specialist scratchpad entries."""

    WORKAROUND = "WORKAROUND"
    QA = "QA"


class ScratchpadEntry(BaseModel):
    """Bounded deterministic memory captured from one specialist tool round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ScratchpadScope = ScratchpadScope.WORKAROUND
    phase: WorkaroundExecutionPhase | None = None
    round_number: int = Field(ge=1)
    key_findings: list[str] = Field(default_factory=list)
    files_inspected: list[str] = Field(default_factory=list)
    plan_summary: str = ""
    validation_outcome: str = ""
    critical_outcome: str = Field(default="", max_length=1300)

    @model_validator(mode="after")
    def _require_workaround_phase(self) -> ScratchpadEntry:
        """Require an execution phase for workaround-scoped entries."""
        if self.scope == ScratchpadScope.WORKAROUND and self.phase is None:
            raise ValueError("workaround scratchpad entries require a phase.")
        return self


class WorkaroundValidationStatus(StrEnum):
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


class TacticalSupervisorAction(BaseModel):
    """Proposal envelope for one tactical remediation action.

    The envelope describes a proposal only.  It does not identify the active
    task or authorize file access; task identity and provenance remain owned by
    the committed ``TaskAttemptSnapshot`` and its surrounding orchestration
    state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_strategy: TacticalStrategy
    target_version: str | None = None
    workaround_hypothesis: str | None = None
    target_files_hint: list[str] = Field(
        default_factory=list,
        description="Normalized relative path hints, not worker authorization.",
    )
    rationale: str = Field(..., min_length=1)

    @field_validator("selected_strategy", mode="before")
    @classmethod
    def _normalize_tactical_strategy(cls, value: Any) -> Any:
        """Trim a tactical strategy string before enum validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("target_version", "workaround_hypothesis", mode="before")
    @classmethod
    def _normalize_tactical_optional_text(cls, value: Any, info: Any) -> str | None:
        """Normalize optional tactical action text fields."""
        return _trim_optional_contract_text(value, info.field_name)

    @field_validator("target_files_hint", mode="before")
    @classmethod
    def _normalize_tactical_file_hints(cls, value: Any) -> list[str]:
        """Normalize safe relative file hints and reject duplicates."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("target_files_hint must be a list of relative paths.")
        result = [_normalize_relative_file_hint(item) for item in value]
        if len(result) != len(set(result)):
            raise ValueError("target_files_hint must not contain duplicates.")
        return result

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_tactical_rationale(cls, value: Any) -> str:
        """Require a trimmed tactical rationale."""
        return _trim_required_contract_text(value, "rationale")

    @model_validator(mode="after")
    def _validate_tactical_strategy_fields(self) -> TacticalSupervisorAction:
        """Reject strategy fields that contradict the selected action."""
        if self.selected_strategy == TacticalStrategy.CODE_WORKAROUND:
            if self.target_version is not None:
                raise ValueError("code_workaround actions must not specify target_version.")
            if self.workaround_hypothesis is None:
                raise ValueError("code_workaround actions require workaround_hypothesis.")
        else:
            if self.target_version is None:
                raise ValueError(f"{self.selected_strategy.value} actions require target_version.")
            if self.workaround_hypothesis is not None:
                raise ValueError(
                    f"{self.selected_strategy.value} actions must not specify "
                    "workaround_hypothesis."
                )
        return self


class QAFailureEvidence(BaseModel):
    """Exact diagnostic evidence extracted from a failed QA evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_diagnostics: list[str] = Field(default_factory=list, max_length=15)
    failed_tests: list[str] = Field(default_factory=list, max_length=10)
    source_locations: list[str] = Field(default_factory=list, max_length=10)
    affected_files: list[str] = Field(default_factory=list, max_length=10)
    raw_excerpt: str = Field(default="", max_length=2000)
    attempt_id: str = Field(default="")
    task_revision: int = Field(default=0, ge=0)

    @field_validator(
        "exact_diagnostics",
        "failed_tests",
        "source_locations",
        "affected_files",
        mode="before",
    )
    @classmethod
    def _normalize_failure_evidence_lists(cls, value: Any, info: Any) -> list[str]:
        """Trim diagnostic entries and preserve their evidence order."""
        return _normalize_contract_string_list(
            value,
            info.field_name,
            reject_duplicates=False,
        )

    @field_validator("raw_excerpt", "attempt_id", mode="before")
    @classmethod
    def _normalize_failure_evidence_text(cls, value: Any, info: Any) -> str:
        """Trim bounded failure excerpts and attempt identifiers."""
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string.")
        return value.strip()


class QASemanticSecurityReview(BaseModel):
    """Evidence-backed semantic review for a code-remediation policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: SecurityReviewVerdict
    reasoning: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class QATestAttribution(BaseModel):
    """Structured attribution of failed tests to remediation groups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: TestAttributionVerdict
    responsible_group_ids: list[str] = Field(default_factory=list)
    failed_tests: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("responsible_group_ids", "failed_tests", mode="before")
    @classmethod
    def _normalize_attribution_lists(cls, value: Any) -> list[str]:
        """Trim attribution evidence without weakening fail-closed checks."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("attribution evidence fields must be lists of strings.")
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("attribution evidence fields must contain only strings.")
            result.append(item.strip())
        return result

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_attribution_reasoning(cls, value: Any) -> str:
        """Trim attribution reasoning while preserving empty reasoning as invalid evidence."""
        if not isinstance(value, str):
            raise ValueError("reasoning must be a string.")
        return value.strip()

    @model_validator(mode="after")
    def _check_attribution_evidence(self) -> QATestAttribution:
        """Require evidence for responsible and exonerated attribution."""
        if self.verdict == TestAttributionVerdict.INCONCLUSIVE:
            return self
        if not self.responsible_group_ids or any(
            not identifier.strip() for identifier in self.responsible_group_ids
        ):
            raise ValueError(
                "responsible or exonerated test attribution requires non-empty "
                "responsible_group_ids."
            )
        if not self.failed_tests or any(not test.strip() for test in self.failed_tests):
            raise ValueError("responsible or exonerated test attribution requires failed_tests.")
        if not self.reasoning.strip():
            raise ValueError("responsible or exonerated test attribution requires reasoning.")
        return self


class QADependencyEvidence(BaseModel):
    """Compact Python-owned evidence for one task's dependency state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: DependencyEvidenceStatus
    target_package: str = Field(default="")
    expected_version: str | None = None
    manifest_paths: list[str] = Field(default_factory=list)
    lockfile_paths: list[str] = Field(default_factory=list)
    declarations: dict[str, str] = Field(default_factory=dict)
    resolved_versions: list[str] = Field(default_factory=list)
    lockfile_versions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class PeerConflictEvidence(BaseModel):
    """Deterministic npm peer-conflict evidence used for cluster expansion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requester_package: str = ""
    peer_package: str = ""
    required_range: str | None = None
    observed_version: str | None = None
    evidence: str = Field(default="", max_length=2000)

    @field_validator(
        "requester_package",
        "peer_package",
        "required_range",
        "observed_version",
        "evidence",
        mode="before",
    )
    @classmethod
    def _normalize_peer_evidence_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("peer conflict evidence fields must be strings.")
        return value.strip()


class QADeterministicGates(BaseModel):
    """Raw Python-owned QA evidence before policy-specific decision rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["pass", "fail"]
    install_passed: bool
    scanner_execution_status: ScannerExecutionStatus
    target_remaining_identifiers: list[str] = Field(default_factory=list)
    target_scanner_cleared: bool | None = None
    tests_passed: bool | None = None
    package_manifest_state: str | None = None
    package_graph_state: str | None = None
    install_error_category: str | None = None
    peer_conflicts: list[PeerConflictEvidence] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    dependency_evidence: QADependencyEvidence | None = Field(
        default=None,
        description=(
            "Python-owned manifest and resolved dependency evidence. "
            "Raw file contents are not required for the QA decision."
        ),
    )


class ODCScanEvidence(BaseModel):
    """Typed evidence describing one ODC scan and its authority boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_scope: ScanScope
    effective_scope: ScanScope
    authoritative: bool = False
    covered_task_ids: list[str] = Field(default_factory=list)
    closure_package_names: list[str] = Field(default_factory=list)
    closure_lockfile_keys: list[str] = Field(default_factory=list)
    found_identifiers: list[str] = Field(default_factory=list)
    remaining_target_identifiers: list[str] = Field(default_factory=list)
    complete: bool = False
    fallback_reason: ScanFallbackReason | None = None


class FinalFullScanResult(BaseModel):
    """Authoritative full-workspace ODC result used before teardown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    completed: bool = False
    authoritative: bool = True
    found_identifiers: list[str] = Field(default_factory=list)
    remaining_target_identifiers: list[str] = Field(default_factory=list)
    new_identifiers: list[str] = Field(default_factory=list)
    found_issues: list[VulnerabilityIssue] = Field(default_factory=list)
    status: str = "not_scanned"
    triage_required: bool = False
    error: str | None = None


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


class QACriticLLMOutput(BaseModel):
    """LLM-owned fields returned by one task-scoped QA evaluator.

    Deterministic gates, scan evidence, failure excerpts, and attempt
    provenance are attached by Python after this model is validated.
    """

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
    semantic_security_review: QASemanticSecurityReview | None = Field(default=None)
    test_attribution: QATestAttribution | None = Field(default=None)

    @model_validator(mode="after")
    def _check_pass_fail_payload(self) -> QACriticLLMOutput:
        """Require the minimum structured decision fields from the evaluator."""
        if self.passed:
            if self.failure_category is not None or self.retry_feedback is not None:
                raise ValueError(
                    "passed=True requires failure_category=None and retry_feedback=None."
                )
            return self
        if self.failure_category is None:
            raise ValueError("passed=False requires a non-null failure_category.")
        if not self.retry_feedback or not self.retry_feedback.strip():
            raise ValueError("passed=False requires a non-empty retry_feedback.")
        return self


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
    deterministic_gates: QADeterministicGates | None = Field(default=None)
    semantic_security_review: QASemanticSecurityReview | None = Field(default=None)
    test_attribution: QATestAttribution | None = Field(default=None)
    contract_error: bool = Field(
        default=False,
        description=(
            "True when the structured QA result could not be validated. This is "
            "an inconclusive QA outcome and must not consume the task retry budget."
        ),
    )
    contract_error_reason: str = Field(default="")
    evidence_inconclusive: bool = Field(
        default=False,
        description=(
            "True when deterministic evidence collection was unavailable. "
            "This must be rechecked without consuming a remediation retry."
        ),
    )

    @model_validator(mode="after")
    def _check_contract_error(self) -> QAEvaluation:
        if self.contract_error:
            if self.passed:
                raise ValueError("contract_error=True cannot accompany passed=True.")
            if not self.contract_error_reason.strip():
                raise ValueError("contract_error=True requires a non-empty contract_error_reason.")
        return self

    scan_evidence: ODCScanEvidence | None = Field(
        None,
        description=(
            "Attempt-local ODC evidence. Only the final full scan is authoritative "
            "for repo-wide status."
        ),
    )

    @model_validator(mode="after")
    def _check_pass_fail_payload(self) -> QAEvaluation:
        if self.passed:
            if (
                self.failure_category is not None
                or self.retry_feedback is not None
                or self.failure_evidence is not None
                or self.evidence_inconclusive
            ):
                raise ValueError(
                    "passed=True requires failure_category=None, retry_feedback=None, "
                    "failure_evidence=None, and evidence_inconclusive=False."
                )
            return self

        if self.contract_error:
            return self

        if self.failure_category is None:
            raise ValueError("passed=False requires a non-null failure_category.")
        if not self.retry_feedback or not self.retry_feedback.strip():
            raise ValueError("passed=False requires a non-empty retry_feedback.")
        return self


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
    is_synthetic: bool = Field(
        default=False,
        description="Whether the committed attempt belongs to a no-CVE coordination task.",
    )
    state_revision: int = Field(default=0, ge=0)
    task_revision: int = Field(default=0, ge=0)
    attempt_number: int = Field(default=1, ge=1)
    cluster_id: str | None = Field(default=None)
    dispatch_batch_id: str | None = Field(default=None)
    action_digest: str | None = Field(default=None)
    manifest_path: str | None = Field(default=None)
    selected_plan_issue_ids: list[str] = Field(default_factory=list)
    qa_policy: QAPolicy | None = Field(default=None)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    no_fix_stage: NoFixMitigationStage | None = Field(default=None)
    selected_version: str | None = None
    allowed_target_versions: list[str] = Field(default_factory=list)
    target_package_name: str | None = None
    target_dependency_type: str | None = None
    allowed_dependency_types: list[str] = Field(default_factory=list)
    parent_minimum_version: str | None = None
    instruction: str = Field(..., min_length=1)
    instruction_digest: str = Field(..., min_length=1)
    dispatch_node: Literal["update_subagent", "workaround_subagent", "qa_critic"]
    plan_id: str | None = None
    portfolio_plan_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    workaround_context: WorkaroundContext | None = Field(default=None)

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _normalize_snapshot_manifest_path(cls, value: Any) -> str | None:
        """Keep the committed manifest target repository-relative and safe."""
        if value is None:
            return None
        normalized = _trim_optional_contract_text(value, "manifest_path")
        if normalized is None:
            return None
        normalized = normalized.replace("\\", "/")
        if (
            normalized.startswith("/")
            or PureWindowsPath(normalized).drive
            or ".." in normalized.split("/")
        ):
            raise ValueError("manifest_path must be repository-relative and traversal-free.")
        return normalized


class UpdateRetryDiagnostics(BaseModel):
    """Structured retry evidence emitted by the update subagent per task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    committed_attempt_id: str | None = None
    security_floor: str | None = None
    target_package_name: str | None = None
    target_dependency_type: str | None = None
    parent_package_name: str | None = None
    parent_minimum_version: str | None = None
    registry_query_performed: bool = False
    attempted_versions: list[str] = Field(default_factory=list)
    executed_versions: list[str] = Field(default_factory=list)
    attempted_versions_by_target: dict[str, list[str]] = Field(default_factory=dict)
    candidate_versions_considered: list[str] = Field(default_factory=list)
    attempted_dependency_types: list[str] = Field(default_factory=list)
    candidate_dependency_types: list[str] = Field(default_factory=list)
    selected_version: str | None = None
    effective_target_version: str | None = None
    effective_dependency_type: str | None = None
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

    @field_validator(
        "attempted_dependency_types",
        "candidate_dependency_types",
        mode="before",
    )
    @classmethod
    def _normalize_dependency_type_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("dependency type lists must be lists of strings.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("dependency type lists must contain only strings.")
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @field_validator("attempted_versions_by_target", mode="before")
    @classmethod
    def _normalize_attempted_versions_by_target(cls, value: Any) -> dict[str, list[str]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("attempted_versions_by_target must be a mapping.")
        normalized: dict[str, list[str]] = {}
        for package, versions in value.items():
            package_name = str(package).strip()
            if not package_name:
                continue
            if not isinstance(versions, list):
                raise ValueError("attempted_versions_by_target values must be lists.")
            seen: set[str] = set()
            cleaned: list[str] = []
            for version in versions:
                if not isinstance(version, str):
                    raise ValueError("attempted target versions must be strings.")
                item = version.strip()
                if item and item not in seen:
                    seen.add(item)
                    cleaned.append(item)
            normalized[package_name] = cleaned
        return normalized

    @field_validator(
        "selected_version",
        "effective_target_version",
        "latest_version_seen",
        mode="before",
    )
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
    effective_target_version: str | None = None
    effective_dependency_type: str | None = None
    manifest_transaction_attempts: int = Field(default=0, ge=0)
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


class WorkerAttemptResult(BaseModel):
    """Worker result correlated to the supervisor's committed attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    task_revision: int = Field(default=0, ge=0)
    cluster_id: str | None = Field(default=None)
    dispatch_batch_id: str | None = Field(default=None)
    action_digest: str | None = Field(default=None)
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
    cluster_id: str | None = Field(default=None)
    dispatch_batch_id: str | None = Field(default=None)
    action_digest: str | None = Field(default=None)
    qa_policy: QAPolicy = Field(
        ...,
        description="Supervisor-owned policy copied from the immutable attempt snapshot.",
    )
    qa_policy_source: Literal["attempt_snapshot"] = Field(
        ...,
        description="Policy provenance must be the immutable attempt snapshot.",
    )
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
    action: Literal["ignored", "repaired", "replanned", "rejected"]
    details: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SupervisorRetryPlan(BaseModel):
    """Deterministic retry plan committed before Supervisor routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    plan_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_task_revision: int = Field(default=0, ge=0)
    strategy_stage: SCARemediationStage = SCARemediationStage.OSV_MINIMUM
    selected_version: str | None = None
    target_package_name: str | None = None
    target_dependency_type: str | None = None
    parent_minimum_version: str | None = None
    attempted_versions: list[str] = Field(default_factory=list)
    candidate_versions_considered: list[str] = Field(default_factory=list)
    candidate_dependency_types: list[str] = Field(default_factory=list)
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

    @field_validator("candidate_dependency_types", mode="before")
    @classmethod
    def _normalize_plan_dependency_types(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("planner dependency type candidates must be a list of strings.")
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("planner dependency type candidates must contain only strings.")
            normalized = item.strip()
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


class TaskDependency(BaseModel):
    """One directed prerequisite edge in a strategic task cluster.

    ``upstream_task_id`` is the prerequisite and
    ``downstream_task_id`` is the task that depends on it.  An upstream task
    may belong to another cluster; cluster membership is required only for the
    downstream task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    upstream_task_id: str = Field(..., min_length=1)
    downstream_task_id: str = Field(..., min_length=1)
    edge_type: TaskDependencyKind
    version_constraint: str | None = None

    @field_validator("upstream_task_id", "downstream_task_id", mode="before")
    @classmethod
    def _normalize_dependency_task_id(cls, value: Any, info: Any) -> str:
        """Require trimmed task identifiers."""
        return _trim_required_contract_text(value, info.field_name)

    @field_validator("edge_type", mode="before")
    @classmethod
    def _normalize_dependency_kind(cls, value: Any) -> Any:
        """Trim dependency-kind strings before enum validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("version_constraint", mode="before")
    @classmethod
    def _normalize_dependency_constraint(cls, value: Any) -> str | None:
        """Normalize an optional version constraint."""
        return _trim_optional_contract_text(value, "version_constraint")

    @model_validator(mode="after")
    def _reject_self_dependency(self) -> TaskDependency:
        """Reject an edge whose prerequisite and dependent are identical."""
        if self.upstream_task_id == self.downstream_task_id:
            raise ValueError("task dependency endpoints must be distinct.")
        return self


class TaskCluster(BaseModel):
    """Bounded group of tasks that may be planned as one strategic unit.

    Dependency validation checks local membership and duplicate edges only.
    It intentionally does not detect cycles because peer relationships can be
    bidirectional and the Phase 3 graph solver owns cycle handling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str = Field(..., min_length=1)
    task_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_MULTI_PACKAGE_ACTION_SIZE,
    )
    dependencies: list[TaskDependency] = Field(default_factory=list)
    reason: str = Field(..., min_length=1)
    atomic: bool = True
    dispatchable: bool = True

    @field_validator("cluster_id", "reason", mode="before")
    @classmethod
    def _normalize_cluster_text(cls, value: Any, info: Any) -> str:
        """Require trimmed cluster identity and formation reason."""
        return _trim_required_contract_text(value, info.field_name)

    @field_validator("task_ids", mode="before")
    @classmethod
    def _normalize_cluster_task_ids(cls, value: Any) -> list[str]:
        """Normalize unique task identifiers while preserving their order."""
        return _normalize_contract_string_list(value, "task_ids")

    @model_validator(mode="after")
    def _validate_cluster_dependencies(self) -> TaskCluster:
        """Require local downstream endpoints and unique directed edges."""
        task_ids = set(self.task_ids)
        seen_edges: set[tuple[str, str]] = set()
        for dependency in self.dependencies:
            if dependency.downstream_task_id not in task_ids:
                raise ValueError(
                    "dependency downstream_task_id must belong to the cluster: "
                    f"{dependency.downstream_task_id!r}."
                )
            edge_key = (
                dependency.upstream_task_id,
                dependency.downstream_task_id,
            )
            if edge_key in seen_edges:
                raise ValueError(f"duplicate dependency edge: {edge_key!r}.")
            seen_edges.add(edge_key)
        return self


class PortfolioPlan(BaseModel):
    """Deterministic package-group ordering and solver-backed execution plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(..., min_length=1)
    portfolio_plan_id: str = Field(
        default="",
        description="Immutable outer-plan identity retained across task-local retries.",
    )
    repository_fingerprint: str = Field(..., min_length=1)
    graph_digest: str = Field(..., min_length=1)
    plan_digest: str = Field(..., min_length=1)
    solver_input_digest: str | None = None
    solver_plan: SolverRemediationPlan | None = None
    portfolio_iteration: int = Field(default=0, ge=0)
    task_ids: list[str] = Field(..., min_length=1)
    clusters: list[TaskCluster] = Field(..., min_length=1)
    cluster_order: list[str] = Field(..., min_length=1)
    task_order: list[str] = Field(..., min_length=1)
    task_to_cluster: dict[str, str] = Field(default_factory=dict)
    task_revisions: dict[str, int] = Field(default_factory=dict)
    planned_task_revisions: dict[str, int] = Field(default_factory=dict)
    task_strategies: dict[str, RoutingStrategy] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "plan_id",
        "repository_fingerprint",
        "graph_digest",
        "plan_digest",
        mode="before",
    )
    @classmethod
    def _normalize_plan_text(cls, value: Any, info: Any) -> str:
        return _trim_required_contract_text(value, info.field_name)

    @field_validator("portfolio_plan_id", mode="before")
    @classmethod
    def _normalize_portfolio_plan_id(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("portfolio_plan_id must be a string.")
        return value.strip()

    @field_validator("solver_input_digest", mode="before")
    @classmethod
    def _normalize_solver_input_digest(cls, value: Any) -> str | None:
        return _trim_optional_contract_text(value, "solver_input_digest")

    @field_validator("task_ids", "cluster_order", "task_order", mode="before")
    @classmethod
    def _normalize_plan_lists(cls, value: Any, info: Any) -> list[str]:
        return _normalize_contract_string_list(value, info.field_name)

    @model_validator(mode="after")
    def _validate_plan_membership(self) -> PortfolioPlan:
        task_ids = set(self.task_ids)
        clusters_by_id = {cluster.cluster_id: cluster for cluster in self.clusters}
        if len(clusters_by_id) != len(self.clusters):
            raise ValueError("PortfolioPlan cluster IDs must be unique.")
        if set(self.cluster_order) != set(clusters_by_id):
            raise ValueError("cluster_order must contain every cluster exactly once.")
        if set(self.task_order) != task_ids or len(self.task_order) != len(task_ids):
            raise ValueError("task_order must contain every task exactly once.")
        members: dict[str, str] = {}
        for cluster in self.clusters:
            for task_id in cluster.task_ids:
                if task_id in members:
                    raise ValueError(f"task {task_id!r} belongs to multiple clusters.")
                members[task_id] = cluster.cluster_id
        if set(members) != task_ids:
            raise ValueError("clusters must cover every planned task exactly once.")
        if self.task_to_cluster != members:
            raise ValueError("task_to_cluster must match cluster membership.")
        if set(self.task_revisions) != task_ids:
            raise ValueError("task_revisions must cover every planned task.")
        if not self.planned_task_revisions:
            object.__setattr__(self, "planned_task_revisions", dict(self.task_revisions))
        elif set(self.planned_task_revisions) != task_ids:
            raise ValueError("planned_task_revisions must cover every planned task.")
        if set(self.task_strategies) != task_ids:
            raise ValueError("task_strategies must cover every planned task.")
        if not self.portfolio_plan_id:
            object.__setattr__(self, "portfolio_plan_id", self.plan_id)
        return self


class PackageMutation(BaseModel):
    """One package/version mutation within a multi-package proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    manifest_path: str = Field(
        default="package.json",
        min_length=1,
        description="Repository-relative package manifest containing the mutation.",
    )
    target_version: str = Field(..., min_length=1)
    dependency_type: Literal[
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
        "overrides",
        "resolutions",
        "pnpm_overrides",
    ]

    @field_validator("task_id", "package_name", "manifest_path", "target_version", mode="before")
    @classmethod
    def _normalize_package_mutation_text(cls, value: Any, info: Any) -> str:
        """Require trimmed package mutation text fields."""
        normalized = _trim_required_contract_text(value, info.field_name)
        if info.field_name == "manifest_path":
            normalized = normalized.replace("\\", "/")
            if (
                normalized.startswith("/")
                or PureWindowsPath(normalized).drive
                or ".." in normalized.split("/")
            ):
                raise ValueError("manifest_path must be repository-relative and traversal-free.")
        return normalized

    @field_validator("dependency_type", mode="before")
    @classmethod
    def _normalize_package_dependency_type(cls, value: Any) -> Any:
        """Trim dependency labels before literal validation."""
        return value.strip() if isinstance(value, str) else value


class MultiPackageAction(BaseModel):
    """Atomic proposal for one to ten package version mutations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str | None = None
    dispatch_batch_id: str | None = None
    selected_strategy: TacticalStrategy
    package_mutations: list[PackageMutation] = Field(
        ...,
        min_length=1,
        max_length=MAX_MULTI_PACKAGE_ACTION_SIZE,
    )
    rationale: str = Field(..., min_length=1)

    @field_validator("cluster_id", "dispatch_batch_id", mode="before")
    @classmethod
    def _normalize_action_provenance(cls, value: Any, info: Any) -> str | None:
        """Normalize optional cluster and dispatch-batch provenance."""
        return _trim_optional_contract_text(value, info.field_name)

    @field_validator("selected_strategy", mode="before")
    @classmethod
    def _normalize_action_strategy(cls, value: Any) -> Any:
        """Trim the tactical strategy before enum validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("package_mutations", mode="before")
    @classmethod
    def _require_mutation_list(cls, value: Any) -> Any:
        """Require an explicit list so the batch boundary stays unambiguous."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("package_mutations must be a list.")
        return value

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_action_rationale(cls, value: Any) -> str:
        """Require a trimmed action rationale."""
        return _trim_required_contract_text(value, "rationale")

    @model_validator(mode="after")
    def _validate_multi_package_action(self) -> MultiPackageAction:
        """Enforce strategy, package uniqueness, and cluster invariants."""
        if self.selected_strategy == TacticalStrategy.CODE_WORKAROUND:
            raise ValueError("MultiPackageAction supports only version_bump or package_override.")
        package_names = [mutation.package_name for mutation in self.package_mutations]
        if len(package_names) != len(set(package_names)):
            raise ValueError("package_mutations must contain unique package names.")
        if len(self.package_mutations) > 1 and self.cluster_id is None:
            raise ValueError("cluster_id is required for multi-package actions.")
        return self


class SupervisorDecision(BaseModel):
    """
    Typed transition decision produced by the deterministic Supervisor.

    Controls hub-and-spoke routing in the Phase 5 orchestrator graph.
    Pydantic validators enforce routing invariants:
    - ``workaround_subagent`` requires exactly one ``target_task_id``.
    - ``update_subagent`` accepts 1-10 ``target_task_ids`` for reusable batch callers;
      the current Supervisor dispatch policy narrows this to one target.
    - ``qa_critic`` accepts one or more ``target_task_ids`` for reusable batch QA;
      the current Supervisor dispatch policy narrows this to one target.
    - ``triage`` is a graph-level handoff and does not target a task.
    - ``final_full_scan`` is a graph-level handoff and does not target a task.
    - ``teardown`` requires empty ``target_task_ids``.
    - ``unfixable_task_ids`` and ``target_task_ids`` must not overlap.
    """

    model_config = ConfigDict(frozen=True)

    decision_code: DecisionCode | None = Field(
        default=None,
        description=(
            "Machine-readable deterministic rule that produced this decision. "
            "Python owns this field; it is advisory metadata and does not affect "
            "routing validation."
        ),
    )

    next_node: Literal[
        "portfolio",
        "update_subagent",
        "workaround_subagent",
        "qa_critic",
        "triage",
        "final_full_scan",
        "teardown",
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
            "One or more for direct/batch qa_critic callers. "
            "Empty for teardown. "
            "Exactly one entry for workaround_subagent. "
            "One to ten entries for direct/batch update_subagent callers; Supervisor routing currently sends one."
        ),
    )
    cluster_id: str | None = Field(default=None)
    multi_package_action: MultiPackageAction | None = Field(default=None)
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
        description="QA-derived retry feedback keyed by task_id.",
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
            "Python-owned status transitions; untrusted callers cannot set this field."
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
        if node == "update_subagent" and len(targets) > MAX_MULTI_PACKAGE_ACTION_SIZE:
            raise ValueError(
                "update_subagent supports at most "
                f"{MAX_MULTI_PACKAGE_ACTION_SIZE} target_task_ids, got {len(targets)}."
            )
        if node == "portfolio" and (targets or self.cluster_id or self.multi_package_action):
            raise ValueError("portfolio decisions must not carry dispatch targets or actions.")
        if self.multi_package_action is not None:
            if node != "update_subagent":
                raise ValueError("multi_package_action is only valid for update_subagent.")
            if self.multi_package_action.cluster_id is None:
                raise ValueError("multi_package_action.cluster_id is required.")
            if self.cluster_id != self.multi_package_action.cluster_id:
                raise ValueError("cluster_id must match multi_package_action.cluster_id.")
            action_task_ids = {
                mutation.task_id for mutation in self.multi_package_action.package_mutations
            }
            if action_task_ids != set(targets):
                raise ValueError("multi_package_action must cover exactly target_task_ids.")
        if self.cluster_id is not None and node not in {"update_subagent", "qa_critic"}:
            raise ValueError("cluster_id is only valid for update_subagent or qa_critic.")
        if node == "qa_critic" and len(targets) < 1:
            raise ValueError("qa_critic requires at least 1 target_task_id.")
        if node in {"final_full_scan", "teardown"} and targets:
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
                    "only QA_PASSED and UNFIXABLE may be set by the Supervisor guardrail."
                )
        return self


# ---------------------------------------------------------------------------
# Phase 5 Task Queue contracts
# ---------------------------------------------------------------------------


class TaskSpawnRequest(BaseModel):
    """
    A request from the supervisor to spawn a new child ``RemediationTask``.

    The Supervisor submits these inside ``SupervisorDecision.spawn_requests``;
    the Python guardrail layer materializes the actual ``RemediationTask`` by
    assigning ``task_id``, ``parent_group_id``, ``status``, ``retry_count``,
    and ``ancestry_depth``. Callers never create raw ``RemediationTask``
    objects through this contract.
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
    through the orchestrator lifecycle. Synthetic coordination tasks use the
    same lifecycle while representing a related package without an active CVE.
    Tasks are the primary key for supervisor decisions, QA evaluations, and
    action summaries.
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
    is_synthetic: bool = Field(
        default=False,
        description=(
            "True when this is a Supervisor-generated coordination task for a "
            "package with no active CVE. Synthetic tasks retain the same audit and "
            "atomic-dispatch lifecycle as finding-backed tasks."
        ),
    )
    parent_task_id: str | None = Field(
        default=None,
        description=(
            "The task_id of the parent task that spawned this task. "
            "None for initial (depth-0) tasks."
        ),
    )
    qa_policy: QAPolicy | None = Field(
        default=None,
        description=(
            "Supervisor-owned QA gate policy. None is retained only for legacy "
            "state so missing provenance can fail closed."
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
    target_package_name: str | None = Field(
        default=None,
        description=(
            "Package the current stage is allowed to edit. For transitive tasks this "
            "is the direct parent until PACKAGE_OVERRIDE, then the vulnerable child."
        ),
    )
    target_dependency_type: str | None = Field(
        default=None,
        description=(
            "Manifest dependency section or package-manager override mechanism for "
            "the current target package."
        ),
    )
    parent_package_name: str | None = Field(
        default=None,
        description="Direct parent package for a transitive vulnerability, if known.",
    )
    parent_package_version: str | None = Field(
        default=None,
        description="Installed/resolved direct parent version for a transitive vulnerability.",
    )
    parent_minimum_version: str | None = Field(
        default=None,
        description="Lowest compatible parent version selected by the parent planner.",
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
    selected_plan_issue_ids: list[str] = Field(
        default_factory=list,
        description="Issue IDs whose retained fix plans support the active strategy.",
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
    allowed_target_versions: list[str] = Field(
        default_factory=list,
        description="Solver-approved target versions the inner Supervisor may attempt.",
    )
    allowed_dependency_types: list[str] = Field(
        default_factory=list,
        description="Solver-approved manifest dependency sections for this task.",
    )
    portfolio_plan_id: str | None = Field(
        default=None,
        description="Committed outer portfolio plan authorizing the task input.",
    )
