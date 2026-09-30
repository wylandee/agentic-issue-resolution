"""Immutable contracts shared by the portfolio solver and orchestration layer.

The models in this module deliberately do not import orchestration or graph-state
code.  They are the stable, occurrence-aware boundary between candidate
preparation, CP-SAT, and portfolio projection.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SolverStatus(StrEnum):
    """Outcome of a portfolio solve."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"
    FALLBACK = "FALLBACK"


# A compatibility spelling useful to callers that call this the plan status.
SolverPlanStatus = SolverStatus


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    value = _required_text(value, field_name)
    return value or None


def _unique_strings(value: Any, field_name: str) -> list[str]:
    """Trim a list and reject duplicate values while retaining its order."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        item = _required_text(item, field_name)
        if item in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {item!r}.")
        seen.add(item)
        result.append(item)
    return result


def _normalise_version(value: Any, field_name: str = "version") -> str:
    value = _required_text(value, field_name)
    if value[:1] in {"v", "V"}:
        value = value[1:]
    if not value:
        raise ValueError(f"{field_name} must contain a version.")
    return value


def _optional_version(value: Any, field_name: str = "version") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided.")
    value = value.strip()
    return _normalise_version(value, field_name) if value else None


def _relative_manifest_path(value: Any) -> str:
    """Return a safe repository-relative POSIX path.

    Backslashes are accepted as input separators so a Windows-produced manifest
    cannot bypass traversal checks on a POSIX worker.
    """
    value = _required_text(value, "manifest_path").replace("\\", "/")
    if "\x00" in value:
        raise ValueError("manifest_path must not contain NUL bytes.")
    if value.startswith("/") or re.match(r"^[A-Za-z]:/", value) or re.match(r"^[A-Za-z]:$", value):
        raise ValueError("manifest_path must be relative.")
    if value.startswith("//") or value.startswith("~/") or value == "~":
        raise ValueError("manifest_path must be relative.")
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError("manifest_path must not contain parent traversal.")
    return "/".join(parts)


def _normalise_mapping(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping of strings.")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_required_text(key, field_name)] = _required_text(item, field_name)
    return dict(sorted(result.items()))


def _normalise_pairs(value: Any, field_name: str) -> list[tuple[str, str]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of two-item pairs.")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{field_name} entries must contain exactly two strings.")
        left = _required_text(pair[0], field_name)
        right = _required_text(pair[1], field_name)
        if left == right:
            raise ValueError(f"{field_name} entries must not be self-pairs.")
        key = (left, right)
        reverse = (right, left)
        if key in seen or reverse in seen:
            raise ValueError(f"{field_name} must not contain duplicate pairs.")
        seen.add(key)
        pairs.append(key)
    return pairs


class _SolverModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    _text_fields: ClassVar[tuple[str, ...]] = ()


class SolverPackageOccurrence(_SolverModel):
    """One physical or direct package occurrence retained as solver evidence."""

    occurrence_id: str = Field(..., min_length=1)
    manifest_path: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    lockfile_package_key: str | None = None
    installed_version: str | None = None
    dependency_type: str | None = None
    is_direct: bool
    ancestry: list[str] = Field(default_factory=list)

    @field_validator("occurrence_id", "package_name", mode="before")
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> str:
        return _relative_manifest_path(value)

    @field_validator("lockfile_package_key", "installed_version", "dependency_type", mode="before")
    @classmethod
    def _optional(cls, value: Any, info: Any) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return _optional_text(value, info.field_name)

    @field_validator("ancestry", mode="before")
    @classmethod
    def _ancestry(cls, value: Any) -> list[str]:
        return _unique_strings(value, "ancestry")


class SolverTarget(_SolverModel):
    """One occurrence of a package eligible for solver consideration."""

    occurrence_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    group_id: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    target_package_name: str = Field(..., min_length=1)
    manifest_path: str = Field(..., min_length=1)
    lockfile_package_key: str = Field(..., min_length=1)
    installed_version: str = Field(..., min_length=1)
    dependency_type: str = Field(default="dependencies", min_length=1)
    strategy: str = Field(default="version_bump", min_length=1)
    is_synthetic: bool = False
    is_finding_backed: bool = True
    eligible_for_atomic_update: bool = True
    workspace_id: str | None = None
    dependency_ancestry: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    is_terminal: bool = False
    has_open_attempt: bool = False

    @field_validator(
        "occurrence_id",
        "task_id",
        "group_id",
        "package_name",
        "target_package_name",
        "dependency_type",
        "strategy",
        "lockfile_package_key",
        mode="before",
    )
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> str:
        return _relative_manifest_path(value)

    @field_validator("installed_version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str:
        return _normalise_version(value, "installed_version")

    @field_validator("workspace_id", mode="before")
    @classmethod
    def _workspace(cls, value: Any) -> str | None:
        return _optional_text(value, "workspace_id")

    @field_validator("dependency_ancestry", "finding_ids", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)


class SolverFindingRequirement(_SolverModel):
    """Finding coverage and remediation requirements for an occurrence."""

    finding_id: str = Field(..., min_length=1)
    coverage_id: str = ""
    vulnerable_occurrence_id: str = Field(..., min_length=1)
    cve_id: str | None = None
    ghsa_id: str | None = None
    severity: str = "UNKNOWN"
    vulnerable_package: str = Field(..., min_length=1)
    target_occurrence_id: str = Field(..., min_length=1)
    fixed_version: str | None = None
    direct_parent_name: str | None = None
    direct_parent_minimum_version: str | None = None
    strategy_stage: str = "osv_minimum"
    workaround_available: bool = False
    workaround_plan_ids: list[str] = Field(default_factory=list)
    is_transitive: bool = False

    @field_validator(
        "finding_id",
        "vulnerable_occurrence_id",
        "vulnerable_package",
        "target_occurrence_id",
        "severity",
        "strategy_stage",
        mode="before",
    )
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("cve_id", "ghsa_id", "direct_parent_name", mode="before")
    @classmethod
    def _optional(cls, value: Any, info: Any) -> str | None:
        return _optional_text(value, info.field_name)

    @field_validator("fixed_version", "direct_parent_minimum_version", mode="before")
    @classmethod
    def _versions(cls, value: Any, info: Any) -> str | None:
        return _optional_version(value, info.field_name)

    @field_validator("workaround_plan_ids", mode="before")
    @classmethod
    def _plan_ids(cls, value: Any) -> list[str]:
        return _unique_strings(value, "workaround_plan_ids")

    @field_validator("coverage_id", mode="before")
    @classmethod
    def _coverage(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @model_validator(mode="after")
    def _coverage_identity(self) -> SolverFindingRequirement:
        expected = (
            "coverage:"
            + hashlib.sha256(
                (self.finding_id + "\0" + self.vulnerable_occurrence_id).encode("utf-8")
            ).hexdigest()
        )
        if self.coverage_id and self.coverage_id != expected:
            raise ValueError("coverage_id must identify its finding and vulnerable occurrence.")
        object.__setattr__(self, "coverage_id", expected)
        return self

    @model_validator(mode="after")
    def _has_authoritative_identifier(self) -> SolverFindingRequirement:
        if not self.cve_id and not self.ghsa_id:
            # The stable scanner finding identity is still useful for unresolved
            # findings, but a requirement without either public identifier is not
            # a seed for the bounded SCA solver universe.
            return self
        return self


class SolverDependencyRequirement(_SolverModel):
    """One dependency declaration published by a package candidate."""

    package_name: str = Field(..., min_length=1)
    version_range: str = Field(..., min_length=1)
    kind: Literal["dependency", "optional_dependency", "peer"]
    is_optional: bool = False
    is_range_supported: bool = True

    @field_validator("package_name", "version_range", mode="before")
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)


class SolverVersionCandidate(_SolverModel):
    """Bounded, validated registry candidate for one solver target."""

    version: str = Field(..., min_length=1)
    semver_key: tuple[int, ...] = Field(default_factory=tuple)
    source: str = "registry"
    meets_security_floor: bool = True
    attempted: bool = False
    requirements: list[SolverDependencyRequirement] = Field(default_factory=list)
    engines: dict[str, str] = Field(default_factory=dict)
    os: list[str] = Field(default_factory=list)
    cpu: list[str] = Field(default_factory=list)

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str:
        return _normalise_version(value)

    @field_validator("source", mode="before")
    @classmethod
    def _source(cls, value: Any) -> str:
        return _required_text(value, "source")

    @field_validator("semver_key", mode="before")
    @classmethod
    def _semver_key(cls, value: Any) -> tuple[int, ...]:
        if value is None or value == () or value == []:
            return ()
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in value
            )
        ):
            raise ValueError("semver_key must be a non-empty sequence of non-negative integers.")
        return tuple(value)

    @field_validator("engines", mode="before")
    @classmethod
    def _engines(cls, value: Any) -> dict[str, str]:
        return _normalise_mapping(value, "engines")

    @field_validator("os", "cpu", mode="before")
    @classmethod
    def _platform_lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @model_validator(mode="after")
    def _unique_requirements(self) -> SolverVersionCandidate:
        identities = [
            (item.package_name, item.version_range, item.kind, item.is_optional)
            for item in self.requirements
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("candidate requirements must not contain duplicates.")
        return self

    @model_validator(mode="after")
    def _derive_semver_key(self) -> SolverVersionCandidate:
        if self.semver_key:
            return self
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", self.version)
        if match:
            object.__setattr__(self, "semver_key", tuple(int(part) for part in match.groups()))
        return self


class SolverEvidenceDomain(_SolverModel):
    """Complete supported versions for one non-mutable npm witness."""

    variable_id: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    source_occurrence_id: str = Field(..., min_length=1)
    manifest_path: str = Field(..., min_length=1)
    workspace_id: str | None = None
    dependency_kind: Literal["dependency", "optional_dependency", "peer"]
    candidate_versions: list[str] = Field(default_factory=list)

    @field_validator("variable_id", "package_name", "source_occurrence_id", mode="before")
    @classmethod
    def _required(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("variable_id")
    @classmethod
    def _namespaced_id(cls, value: str) -> str:
        if not value.startswith("evidence:"):
            raise ValueError("evidence variable IDs must use the evidence namespace.")
        return value

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _manifest_path(cls, value: Any) -> str:
        return _relative_manifest_path(value)

    @field_validator("workspace_id", mode="before")
    @classmethod
    def _workspace_id(cls, value: Any) -> str | None:
        return _optional_text(value, "workspace_id")

    @field_validator("candidate_versions", mode="before")
    @classmethod
    def _candidate_versions(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("candidate_versions must be a list.")
        versions = [_normalise_version(item, "candidate_versions") for item in value]
        if len(versions) != len(set(versions)):
            raise ValueError("candidate_versions must not contain duplicates.")
        return sorted(versions)


class SolverRuntimeFingerprint(_SolverModel):
    """Node runtime identity used to certify package-manager compatibility."""

    node_version: str = Field(..., min_length=1)
    npm_version: str = Field(..., min_length=1)
    platform: str = Field(..., min_length=1)
    architecture: str = Field(..., min_length=1)

    @field_validator("node_version", "npm_version", "platform", "architecture", mode="before")
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)


class PackageResolutionStatus(StrEnum):
    """Outcome of package-manager certification for a solver assignment."""

    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class SolverCandidateRejectionReason(StrEnum):
    """Structured evidence category for one strict-resolver rejection."""

    RUNTIME_ENGINE = "runtime_engine"
    RUNTIME_PLATFORM = "runtime_platform"
    DEPENDENCY_RANGE = "dependency_range"
    PEER_CONFLICT = "peer_conflict"
    REQUIRED_DEPENDENCY_MISSING = "required_dependency_missing"
    NEW_VULNERABLE_COPY = "new_vulnerable_copy"
    COVERAGE_UNRESOLVED = "coverage_unresolved"
    UNCLASSIFIED = "unclassified"


class SolverCandidateCutKind(StrEnum):
    """Shape of a safe resolver-learned no-good."""

    UNARY = "unary"
    PAIR = "pair"
    EXACT_ASSIGNMENT = "exact_assignment"


class SolverCandidateLiteral(_SolverModel):
    """One task-backed mutation variable/value in a rejected conjunction."""

    variable_id: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)

    @field_validator("variable_id", mode="before")
    @classmethod
    def _variable_id(cls, value: Any) -> str:
        return _required_text(value, "variable_id")

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str:
        return _normalise_version(value)


class SolverCandidateConflict(_SolverModel):
    """Typed, digest-bound evidence that a candidate conjunction was rejected."""

    assignment_digest: str = Field(..., min_length=1)
    reason_code: SolverCandidateRejectionReason
    cut_kind: SolverCandidateCutKind
    literals: list[SolverCandidateLiteral] = Field(..., min_length=1)
    evidence_digest: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1, max_length=500)

    @field_validator("assignment_digest", "evidence_digest", mode="before")
    @classmethod
    def _sha256_digest(cls, value: Any, info: Any) -> str:
        value = _required_text(value, info.field_name)
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, value: Any) -> str:
        return _required_text(value, "summary")

    @field_validator("literals")
    @classmethod
    def _canonical_literals(
        cls, value: list[SolverCandidateLiteral]
    ) -> list[SolverCandidateLiteral]:
        variable_ids = [literal.variable_id for literal in value]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("conflict literals must have unique variable IDs.")
        return sorted(value, key=lambda literal: (literal.variable_id, literal.version))

    @model_validator(mode="after")
    def _cut_shape_matches_evidence(self) -> SolverCandidateConflict:
        if self.cut_kind == SolverCandidateCutKind.UNARY and (
            len(self.literals) != 1
            or self.reason_code
            not in {
                SolverCandidateRejectionReason.RUNTIME_ENGINE,
                SolverCandidateRejectionReason.RUNTIME_PLATFORM,
            }
        ):
            raise ValueError("unary cuts require one runtime engine or platform literal.")
        if self.cut_kind == SolverCandidateCutKind.PAIR and (
            len(self.literals) != 2
            or self.reason_code
            not in {
                SolverCandidateRejectionReason.DEPENDENCY_RANGE,
                SolverCandidateRejectionReason.PEER_CONFLICT,
            }
        ):
            raise ValueError("pair cuts require two dependency or peer literals.")
        return self


class CertificationStatistics(_SolverModel):
    """Bounded aggregate measurements for one certification call."""

    solver_calls: int = Field(default=0, ge=0, le=10_000_000)
    candidate_attempts: int = Field(default=0, ge=0, le=10_000_000)
    strict_install_invocations: int = Field(default=0, ge=0, le=10_000_000)
    rejection_counts_by_reason: dict[str, int] = Field(default_factory=dict, max_length=8)
    unknown_count: int = Field(default=0, ge=0, le=10_000_000)
    evidence_relations_modeled: int = Field(default=0, ge=0, le=10_000_000)
    evidence_relations_unmodeled: int = Field(default=0, ge=0, le=10_000_000)
    metadata_files_read: int = Field(default=0, ge=0, le=10_000_000)
    metadata_bytes_read: int = Field(default=0, ge=0, le=1_000_000_000_000)
    archive_bytes: int = Field(default=0, ge=0, le=1_000_000_000_000)
    archive_build_time_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    extraction_time_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    package_manager_time_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    lockfile_read_time_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    total_certification_time_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)

    @field_validator("rejection_counts_by_reason", mode="before")
    @classmethod
    def _rejection_counts(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("rejection_counts_by_reason must be a mapping.")
        known = {reason.value for reason in SolverCandidateRejectionReason}
        result: dict[str, int] = {}
        for raw_reason, count in value.items():
            reason = _required_text(raw_reason, "rejection_counts_by_reason")
            if reason not in known:
                raise ValueError(f"unknown rejection reason {reason!r}.")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 10_000_000
            ):
                raise ValueError("rejection counts must be bounded non-negative integers.")
            result[reason] = count
        return dict(sorted(result.items()))


class PackageResolutionCertificate(_SolverModel):
    """Immutable evidence for one package-resolved assignment and its coverage."""

    status: PackageResolutionStatus
    portfolio_plan_id: str = Field(..., min_length=1)
    candidate_plan_id: str | None = None
    solver_input_digest: str = Field(..., min_length=1)
    repository_fingerprint: str = Field(..., min_length=1)
    task_revisions: dict[str, int] = Field(default_factory=dict)
    workspace_graph_digest: str = Field(..., min_length=1)
    candidate_catalog_digest: str = Field(..., min_length=1)
    candidate_assignment_digest: str = Field(..., min_length=1)
    resolved_graph_digest: str = Field(..., min_length=1)
    resolved_lockfile_digests: dict[str, str] = Field(default_factory=dict)
    runtime_fingerprint: SolverRuntimeFingerprint
    covered_coverage_ids: list[str] = Field(default_factory=list)
    workaround_coverage_ids: list[str] = Field(default_factory=list)
    unresolved_coverage_ids: list[str] = Field(default_factory=list)
    rejected_assignment_digests: list[str] = Field(default_factory=list)
    rejection_conflicts: list[SolverCandidateConflict] = Field(default_factory=list)
    certification_statistics: CertificationStatistics = Field(
        default_factory=CertificationStatistics
    )
    batch_prefix_graph_digests: dict[str, str] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "portfolio_plan_id",
        "solver_input_digest",
        "repository_fingerprint",
        "workspace_graph_digest",
        "candidate_catalog_digest",
        "candidate_assignment_digest",
        "resolved_graph_digest",
        mode="before",
    )
    @classmethod
    def _required(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("candidate_plan_id", mode="before")
    @classmethod
    def _optional_candidate_plan_id(cls, value: Any) -> str | None:
        return _optional_text(value, "candidate_plan_id")

    @field_validator(
        "resolved_lockfile_digests",
        "batch_prefix_graph_digests",
        mode="before",
    )
    @classmethod
    def _digest_mappings(cls, value: Any, info: Any) -> dict[str, str]:
        return _normalise_mapping(value, info.field_name)

    @field_validator("task_revisions", mode="before")
    @classmethod
    def _revisions(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("task_revisions must be a mapping.")
        result: dict[str, int] = {}
        for task_id, revision in value.items():
            task_id = _required_text(task_id, "task_revisions")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("task_revisions values must be non-negative integers.")
            result[task_id] = revision
        return dict(sorted(result.items()))

    @field_validator(
        "covered_coverage_ids",
        "workaround_coverage_ids",
        "unresolved_coverage_ids",
        "rejected_assignment_digests",
        "diagnostics",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @model_validator(mode="after")
    def _coverage_is_consistent(self) -> PackageResolutionCertificate:
        coverage_sets = (
            set(self.covered_coverage_ids),
            set(self.workaround_coverage_ids),
            set(self.unresolved_coverage_ids),
        )
        if any(
            coverage_sets[left] & coverage_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("certificate coverage IDs must be classified exactly once.")
        return self


class SolverTaskDecision(_SolverModel):
    """Solver-approved execution decision for one task."""

    task_id: str = Field(..., min_length=1)
    selected_strategy: str = "version_bump"
    selected_route: str | None = None
    selected_version: str | None = None
    allowed_alternative_versions: list[str] = Field(default_factory=list)
    allowed_dependency_types: list[str] = Field(default_factory=list)
    strategy_stage: str = "osv_minimum"
    selected_plan_issue_ids: list[str] = Field(default_factory=list)
    instruction_source: str = "deterministic_solver"
    exact_instruction: str | None = None
    requires_source_migration: bool = False
    installed_version: str | None = None
    target_occurrence_id: str | None = None
    target_group_id: str | None = None
    target_package_name: str | None = None
    manifest_path: str | None = None
    lockfile_package_key: str | None = None
    dependency_type: str | None = None

    @field_validator(
        "task_id", "selected_strategy", "strategy_stage", "instruction_source", mode="before"
    )
    @classmethod
    def _required(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator(
        "selected_route",
        "dependency_type",
        "target_occurrence_id",
        "target_group_id",
        "target_package_name",
        "manifest_path",
        "lockfile_package_key",
        "exact_instruction",
        "installed_version",
        mode="before",
    )
    @classmethod
    def _optional(cls, value: Any, info: Any) -> str | None:
        return _optional_text(value, info.field_name)

    @field_validator("selected_version", mode="before")
    @classmethod
    def _selected_version(cls, value: Any) -> str | None:
        return _optional_version(value, "selected_version")

    @field_validator("allowed_alternative_versions", mode="before")
    @classmethod
    def _versions(cls, value: Any) -> list[str]:
        return _unique_strings(
            [_normalise_version(item) for item in (value or [])], "allowed_alternative_versions"
        )

    @field_validator("allowed_dependency_types", "selected_plan_issue_ids", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)


class SolverPeerConstraint(_SolverModel):
    """Candidate compatibility constraint between package occurrences."""

    source_occurrence_id: str = Field(..., min_length=1)
    target_occurrence_id: str = Field(..., min_length=1)
    version_range: str = Field(..., min_length=1)
    is_strict: bool = True
    is_optional: bool = False
    candidate_specific_source_version: str | None = None

    @field_validator("source_occurrence_id", "target_occurrence_id", "version_range", mode="before")
    @classmethod
    def _required(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("candidate_specific_source_version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str | None:
        return _optional_version(value, "candidate_specific_source_version")

    @model_validator(mode="after")
    def _not_self(self) -> SolverPeerConstraint:
        if self.source_occurrence_id == self.target_occurrence_id:
            raise ValueError("peer constraint endpoints must differ.")
        return self


class SolverCandidateRelation(_SolverModel):
    """One candidate-specific runtime or peer dependency requirement."""

    source_occurrence_id: str = Field(..., min_length=1)
    source_candidate_version: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    version_range: str = Field(..., min_length=1)
    kind: Literal["dependency", "optional_dependency", "peer"]
    is_optional: bool = False
    is_modelled: bool = True
    is_range_supported: bool = True
    target_occurrence_id: str | None = None
    evidence_variable_id: str | None = None

    @field_validator("source_occurrence_id", "package_name", "version_range", mode="before")
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("source_candidate_version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str:
        return _normalise_version(value, "source_candidate_version")

    @field_validator("target_occurrence_id", "evidence_variable_id", mode="before")
    @classmethod
    def _optional_id(cls, value: Any, info: Any) -> str | None:
        return _optional_text(value, info.field_name)


class SolverMutation(_SolverModel):
    """Exact package mutation approved by the outer solver."""

    task_id: str = Field(..., min_length=1)
    occurrence_id: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    manifest_path: str = Field(..., min_length=1)
    target_version: str = Field(..., min_length=1)
    dependency_type: str = Field(..., min_length=1)

    @field_validator("task_id", "occurrence_id", "package_name", "dependency_type", mode="before")
    @classmethod
    def _text(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> str:
        return _relative_manifest_path(value)

    @field_validator("target_version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> str:
        return _normalise_version(value, "target_version")


class SolverBatch(_SolverModel):
    """Atomic dispatch batch projected into one ``TaskCluster``."""

    batch_id: str = Field(..., min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    mutations: list[SolverMutation] = Field(default_factory=list)
    resolved_finding_ids: list[str] = Field(default_factory=list)
    workaround_finding_ids: list[str] = Field(default_factory=list)
    unresolved_finding_ids: list[str] = Field(default_factory=list)
    resolved_coverage_ids: list[str] = Field(default_factory=list)
    workaround_coverage_ids: list[str] = Field(default_factory=list)
    unresolved_coverage_ids: list[str] = Field(default_factory=list)
    coverage_finding_ids: dict[str, str] = Field(default_factory=dict)
    severity_rank: int = Field(default=0, ge=0)
    atomic: bool = True
    dispatchable: bool = True
    diagnostic: str | None = None

    @field_validator("diagnostic", mode="before")
    @classmethod
    def _diagnostic(cls, value: Any) -> str | None:
        return _optional_text(value, "diagnostic")

    @field_validator("batch_id", mode="before")
    @classmethod
    def _batch_id(cls, value: Any) -> str:
        return _required_text(value, "batch_id")

    @field_validator(
        "task_ids",
        "resolved_finding_ids",
        "workaround_finding_ids",
        "unresolved_finding_ids",
        "resolved_coverage_ids",
        "workaround_coverage_ids",
        "unresolved_coverage_ids",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @field_validator("coverage_finding_ids", mode="before")
    @classmethod
    def _coverage_finding_ids(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("coverage_finding_ids must be a mapping.")
        result: dict[str, str] = {}
        for coverage_id, finding_id in value.items():
            coverage_id = _required_text(coverage_id, "coverage_finding_ids")
            if coverage_id in result:
                raise ValueError(f"duplicate coverage_finding_ids key: {coverage_id!r}")
            result[coverage_id] = _required_text(finding_id, "coverage_finding_ids")
        return dict(sorted(result.items()))

    @model_validator(mode="after")
    def _unique_mutations_and_membership(self) -> SolverBatch:
        occurrence_ids: set[str] = set()
        task_ids = set(self.task_ids)
        for mutation in self.mutations:
            if mutation.occurrence_id in occurrence_ids:
                raise ValueError(
                    f"mutations must not contain duplicate occurrence IDs: {mutation.occurrence_id!r}."
                )
            occurrence_ids.add(mutation.occurrence_id)
            if task_ids and mutation.task_id not in task_ids:
                raise ValueError("every mutation task_id must belong to task_ids.")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique.")
        coverage_sets = (
            set(self.resolved_coverage_ids),
            set(self.workaround_coverage_ids),
            set(self.unresolved_coverage_ids),
        )
        if any(
            coverage_sets[left] & coverage_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("coverage IDs must be classified exactly once per batch.")
        classified_coverage = set().union(*coverage_sets)
        if self.coverage_finding_ids and set(self.coverage_finding_ids) != classified_coverage:
            raise ValueError("coverage_finding_ids must map every classified coverage ID.")
        return self


class SolverEdge(_SolverModel):
    """Occurrence or batch dependency edge."""

    source_occurrence_id: str = Field(..., min_length=1)
    target_occurrence_id: str = Field(..., min_length=1)
    edge_kind: str = Field(..., min_length=1)
    source_task_id: str | None = None
    target_task_id: str | None = None
    version_range: str | None = None
    is_optional: bool = False
    is_peer_coupling: bool = False

    @field_validator("source_occurrence_id", "target_occurrence_id", "edge_kind", mode="before")
    @classmethod
    def _required(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("source_task_id", "target_task_id", "version_range", mode="before")
    @classmethod
    def _optional(cls, value: Any, info: Any) -> str | None:
        return _optional_text(value, info.field_name)

    @model_validator(mode="after")
    def _not_self(self) -> SolverEdge:
        if self.source_occurrence_id == self.target_occurrence_id:
            raise ValueError("edge endpoints must differ.")
        return self


class SolverPhase(_SolverModel):
    """One deterministic scheduling phase."""

    phase_number: int = Field(..., ge=1)
    batch_ids: list[str] = Field(default_factory=list)
    scc_collapsed: bool = False
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator("batch_ids", "diagnostics", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)


class SolverSubgraph(_SolverModel):
    """Bounded occurrence graph passed to CP-SAT."""

    targets: list[SolverTarget] = Field(default_factory=list)
    occurrences: list[SolverPackageOccurrence] = Field(default_factory=list)
    findings: list[SolverFindingRequirement] = Field(default_factory=list)
    peer_constraints: list[SolverPeerConstraint] = Field(default_factory=list)
    candidate_relations: list[SolverCandidateRelation] = Field(default_factory=list)
    evidence_domains: list[SolverEvidenceDomain] = Field(default_factory=list)
    edges: list[SolverEdge] = Field(default_factory=list)
    forced_singleton_task_ids: list[str] = Field(default_factory=list)
    external_prerequisite_task_ids: list[str] = Field(default_factory=list)
    valid: bool = True
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "forced_singleton_task_ids", "external_prerequisite_task_ids", "diagnostics", mode="before"
    )
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @model_validator(mode="after")
    def _unique_targets_and_findings(self) -> SolverSubgraph:
        occurrence_ids = [occurrence.occurrence_id for occurrence in self.occurrences]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("evidence occurrences must not contain duplicate occurrence IDs.")
        target_occurrence_ids = [target.occurrence_id for target in self.targets]
        if len(target_occurrence_ids) != len(set(target_occurrence_ids)):
            raise ValueError("targets must not contain duplicate occurrence IDs.")
        task_ids = [target.task_id for target in self.targets]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("targets must not contain duplicate task IDs.")
        evidence_ids = [domain.variable_id for domain in self.evidence_domains]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence domains must have unique variable IDs.")
        relation_keys = [
            (
                relation.source_occurrence_id,
                relation.source_candidate_version,
                relation.package_name,
                relation.version_range,
                relation.kind,
                relation.is_optional,
                relation.target_occurrence_id,
                relation.evidence_variable_id,
                relation.is_modelled,
                relation.is_range_supported,
            )
            for relation in self.candidate_relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("candidate relations must not contain duplicates.")
        evidence_domains_by_id = {domain.variable_id: domain for domain in self.evidence_domains}
        relation_evidence_ids = {
            relation.evidence_variable_id
            for relation in self.candidate_relations
            if relation.evidence_variable_id is not None
        }
        if relation_evidence_ids != set(evidence_domains_by_id):
            raise ValueError("every evidence ID must have exactly one referenced domain.")
        if not self.valid:
            return self
        evidence_by_id = {occurrence.occurrence_id: occurrence for occurrence in self.occurrences}
        targets_by_id = {target.occurrence_id: target for target in self.targets}
        mutation_target_ids = {
            target.occurrence_id for target in self.targets if target.eligible_for_atomic_update
        }
        for relation in self.candidate_relations:
            if relation.source_occurrence_id not in targets_by_id:
                raise ValueError(
                    f"candidate relation source {relation.source_occurrence_id!r} "
                    "is not a mutation target."
                )
            if (
                relation.target_occurrence_id is not None
                and relation.target_occurrence_id not in evidence_by_id
            ):
                raise ValueError(
                    f"candidate relation target {relation.target_occurrence_id!r} is not evidence."
                )
            if relation.evidence_variable_id is not None:
                if relation.target_occurrence_id in mutation_target_ids:
                    raise ValueError(
                        "a candidate relation cannot bind a mutation target and evidence variable."
                    )
                domain = evidence_domains_by_id[relation.evidence_variable_id]
                if (
                    domain.source_occurrence_id != relation.source_occurrence_id
                    or domain.package_name != relation.package_name
                    or domain.dependency_kind != relation.kind
                ):
                    raise ValueError(
                        f"candidate relation does not match evidence domain "
                        f"{relation.evidence_variable_id!r}."
                    )
        for target in self.targets:
            occurrence = evidence_by_id.get(target.occurrence_id)
            if occurrence is None:
                raise ValueError(
                    f"mutation target {target.occurrence_id!r} must exist in evidence occurrences."
                )
            if occurrence.package_name != target.target_package_name:
                raise ValueError(
                    f"mutation target {target.occurrence_id!r} package does not match evidence."
                )
        for finding in self.findings:
            if finding.target_occurrence_id not in targets_by_id:
                raise ValueError(
                    f"finding {finding.coverage_id!r} references an unknown mutation target."
                )
            occurrence = evidence_by_id.get(finding.vulnerable_occurrence_id)
            if occurrence is None:
                raise ValueError(
                    f"finding {finding.coverage_id!r} references an unknown vulnerable occurrence."
                )
            if occurrence.package_name != finding.vulnerable_package:
                raise ValueError(
                    f"finding {finding.coverage_id!r} package does not match vulnerable occurrence."
                )
        return self


class SolverCandidatePlan(_SolverModel):
    """One validated CP-SAT candidate portfolio."""

    candidate_plan_id: str = Field(..., min_length=1)
    objective_vector: tuple[int, ...] = Field(default_factory=tuple)
    selected_candidate_versions: dict[str, str] = Field(default_factory=dict)
    coverage_ids: list[str] = Field(default_factory=list)
    unresolved_ids: list[str] = Field(default_factory=list)
    task_decisions: list[SolverTaskDecision] = Field(default_factory=list)
    batches: list[SolverBatch] = Field(default_factory=list)
    phases: list[SolverPhase] = Field(default_factory=list)
    status: SolverStatus = SolverStatus.FEASIBLE
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator("candidate_plan_id", mode="before")
    @classmethod
    def _id(cls, value: Any) -> str:
        return _required_text(value, "candidate_plan_id")

    @field_validator("objective_vector", mode="before")
    @classmethod
    def _objective(cls, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ValueError("objective_vector must contain integers.")
        return tuple(value)

    @field_validator("selected_candidate_versions", mode="before")
    @classmethod
    def _selected_candidates(cls, value: Any) -> dict[str, str]:
        return _normalise_mapping(value, "selected_candidate_versions")

    @field_validator("coverage_ids", "unresolved_ids", "diagnostics", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @model_validator(mode="after")
    def _unique_plan_members(self) -> SolverCandidatePlan:
        task_ids = [decision.task_id for decision in self.task_decisions]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_decisions must not contain duplicate task IDs.")
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("batches must not contain duplicate batch IDs.")
        return self


class ArbitrationResult(_SolverModel):
    """Bounded metadata returned by optional candidate arbitration."""

    selected_plan_index: int = Field(..., ge=0)
    rationale: str = Field(default="", max_length=2000)
    fallback: bool = False
    model_name: str = Field(default="", max_length=200)
    input_digest: str = Field(..., min_length=1)

    @field_validator("rationale", "model_name", mode="before")
    @classmethod
    def _optional_text_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("arbitration text fields must be strings.")
        return value.strip()

    @field_validator("input_digest", mode="before")
    @classmethod
    def _input_digest(cls, value: Any) -> str:
        return _required_text(value, "input_digest")


class SolverStatistics(_SolverModel):
    """Bounded raw statistics collected from CP-SAT solve calls.

    The solver may perform several calls while projecting a top-K
    lexicographic result.  Keeping the status sequence and the final raw
    status name makes ``UNKNOWN`` distinguishable from statuses such as
    ``MODEL_INVALID`` that otherwise collapse into the public
    :class:`SolverStatus.UNKNOWN` value.
    """

    solve_calls: int = Field(default=0, ge=0)
    raw_status_name: str | None = None
    raw_status_code: int | None = None
    status_sequence: list[str] = Field(default_factory=list)
    status_code_sequence: list[int] = Field(default_factory=list)
    wall_time_seconds: float = Field(default=0.0, ge=0.0)
    user_time_seconds: float = Field(default=0.0, ge=0.0)
    deterministic_time_seconds: float = Field(default=0.0, ge=0.0)
    num_conflicts: int = Field(default=0, ge=0)
    num_branches: int = Field(default=0, ge=0)
    objective_value: float | None = None
    best_objective_bound: float | None = None

    @field_validator("raw_status_name", mode="before")
    @classmethod
    def _status_name(cls, value: Any) -> str | None:
        return _optional_text(value, "raw_status_name")

    @field_validator("status_sequence", mode="before")
    @classmethod
    def _status_sequence(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("status_sequence must be a list of strings.")
        return [_required_text(item, "status_sequence") for item in value]


class SolverRemediationPlan(_SolverModel):
    """Complete solver output, including failures that must remain visible."""

    status: SolverStatus
    input_digest: str = Field(..., min_length=1)
    domain_digest: str = Field(..., min_length=1)
    repository_digest: str = Field(..., min_length=1)
    candidate_catalog_complete: bool = False
    candidate_catalog_digest: str = Field(
        default_factory=lambda: hashlib.sha256(b"").hexdigest(), min_length=1
    )
    task_revisions: dict[str, int] = Field(default_factory=dict)
    candidate_plans: list[SolverCandidatePlan] = Field(default_factory=list)
    selected_plan: SolverCandidatePlan | None = None
    arbitration: ArbitrationResult | None = None
    # Runtime-only observability must not alter immutable plan identity or
    # deterministic model serialization. The dedicated solver span records it.
    solver_statistics: SolverStatistics | None = Field(default=None, exclude=True)
    unresolved_finding_ids: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "input_digest",
        "domain_digest",
        "repository_digest",
        "candidate_catalog_digest",
        mode="before",
    )
    @classmethod
    def _digests(cls, value: Any, info: Any) -> str:
        return _required_text(value, info.field_name)

    @field_validator("task_revisions", mode="before")
    @classmethod
    def _revisions(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("task_revisions must be a mapping.")
        out: dict[str, int] = {}
        for key, revision in value.items():
            key = _required_text(key, "task_revisions")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("task_revisions values must be non-negative integers.")
            out[key] = revision
        return dict(sorted(out.items()))

    @field_validator("unresolved_finding_ids", "diagnostics", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @model_validator(mode="after")
    def _selected_candidate_is_known(self) -> SolverRemediationPlan:
        if self.selected_plan is not None and self.candidate_plans:
            known = {candidate.candidate_plan_id for candidate in self.candidate_plans}
            if self.selected_plan.candidate_plan_id not in known:
                raise ValueError("selected_plan must be one of candidate_plans.")
        if (
            self.status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}
            and self.selected_plan is not None
        ):
            raise ValueError("infeasible or unknown plans cannot carry a selected plan.")
        return self


class DAGBuildResult(_SolverModel):
    """Validated dependency-DAG projection and external prerequisite set."""

    batches: list[SolverBatch] = Field(default_factory=list)
    edges: list[SolverEdge] = Field(default_factory=list)
    batch_edges: list[tuple[str, str]] = Field(default_factory=list)
    task_to_batch: dict[str, str] = Field(default_factory=dict)
    external_prerequisite_task_ids: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    valid: bool = True

    @field_validator("external_prerequisite_task_ids", "diagnostics", mode="before")
    @classmethod
    def _lists(cls, value: Any, info: Any) -> list[str]:
        return _unique_strings(value, info.field_name)

    @field_validator("batch_edges", mode="before")
    @classmethod
    def _batch_edges(cls, value: Any) -> list[tuple[str, str]]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("batch_edges must be a list of directed pairs.")
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("batch_edges must contain pairs.")
            left = _required_text(pair[0], "batch_edges")
            right = _required_text(pair[1], "batch_edges")
            if left == right:
                raise ValueError("batch_edges cannot contain self-edges.")
            key = (left, right)
            if key in seen:
                raise ValueError("batch_edges must not contain duplicate directed pairs.")
            seen.add(key)
            pairs.append(key)
        return sorted(pairs)

    @field_validator("task_to_batch", mode="before")
    @classmethod
    def _task_to_batch(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("task_to_batch must be a mapping.")
        return {
            _required_text(task_id, "task_to_batch"): _required_text(batch_id, "task_to_batch")
            for task_id, batch_id in sorted(value.items())
        }

    @model_validator(mode="after")
    def _unique_edges_and_batch_membership(self) -> DAGBuildResult:
        keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            key = (edge.source_occurrence_id, edge.target_occurrence_id, edge.edge_kind)
            if key in keys:
                raise ValueError("edges must not contain duplicate directed relationships.")
            keys.add(key)
        batch_ids = {batch.batch_id for batch in self.batches}
        if any(left not in batch_ids or right not in batch_ids for left, right in self.batch_edges):
            raise ValueError("batch_edges must reference known batch IDs.")
        if any(batch_id not in batch_ids for batch_id in self.task_to_batch.values()):
            raise ValueError("task_to_batch must reference known batch IDs.")
        return self


class PortfolioReplanRequest(_SolverModel):
    """Typed request to leave the inner loop and rebuild the outer portfolio."""

    reason: str = Field(..., min_length=1)
    peer_conflict_pairs: list[tuple[str, str]] = Field(default_factory=list)
    forced_singleton_task_ids: list[str] = Field(default_factory=list)
    triggering_attempt_id: str | None = None
    triggering_scan_id: str | None = None
    source_portfolio_plan_id: str | None = None

    @field_validator("reason", mode="before")
    @classmethod
    def _reason(cls, value: Any) -> str:
        return _required_text(value, "reason")

    @field_validator("peer_conflict_pairs", mode="before")
    @classmethod
    def _pairs(cls, value: Any) -> list[tuple[str, str]]:
        return _normalise_pairs(value, "peer_conflict_pairs")

    @field_validator("forced_singleton_task_ids", mode="before")
    @classmethod
    def _singletons(cls, value: Any) -> list[str]:
        return _unique_strings(value, "forced_singleton_task_ids")

    @field_validator(
        "triggering_attempt_id", "triggering_scan_id", "source_portfolio_plan_id", mode="before"
    )
    @classmethod
    def _optional_ids(cls, value: Any, info: Any) -> str | None:
        return _optional_text(value, info.field_name)


__all__ = [
    "ArbitrationResult",
    "PackageResolutionCertificate",
    "PackageResolutionStatus",
    "SolverPackageOccurrence",
    "SolverCandidateRelation",
    "SolverDependencyRequirement",
    "SolverRuntimeFingerprint",
    "SolverBatch",
    "SolverCandidatePlan",
    "SolverEdge",
    "SolverFindingRequirement",
    "SolverMutation",
    "SolverPeerConstraint",
    "SolverPhase",
    "SolverPlanStatus",
    "SolverRemediationPlan",
    "SolverStatus",
    "SolverStatistics",
    "SolverSubgraph",
    "SolverTarget",
    "SolverTaskDecision",
    "SolverVersionCandidate",
    "CertificationStatistics",
    "SolverCandidateConflict",
    "SolverCandidateCutKind",
    "SolverCandidateLiteral",
    "SolverCandidateRejectionReason",
    "SolverEvidenceDomain",
]
