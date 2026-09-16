"""Shared typed records and small helpers for QA execution modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from remediation_engine.contracts.schemas import (
    ODCScanEvidence,
    PeerConflictEvidence,
    QADependencyEvidence,
    RemediationTask,
    ScannerExecutionStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.runtime.path_policy import normalize_workspace_path


@dataclass(frozen=True)
class _QALogRecord:
    """Private raw execution evidence for one QA command or suite."""

    phase: str
    label: str
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None = None


@dataclass(frozen=True)
class _SecurityScanResult:
    """Complete deterministic security-scan outcome."""

    ok: bool
    summary: str
    remaining_identifiers: set[str]
    found_identifiers: set[str]
    new_identifiers: set[str]
    found_issues: list[VulnerabilityIssue] = field(default_factory=list)
    execution_status: ScannerExecutionStatus = ScannerExecutionStatus.NOT_RUN
    exit_code: int | None = None
    raw_stdout: str | None = None
    raw_stderr: str | None = None
    diagnostic_log_path: str | None = None
    scan_records: tuple[_QALogRecord, ...] = ()


@dataclass
class _QAPackageState:
    """Per-task manifest and resolved-graph evidence collected by Python."""

    manifest_state: str | None = None
    graph_state: str | None = None
    diagnostics: tuple[str, ...] = ()
    dependency_evidence: QADependencyEvidence | None = None


@dataclass
class _QAExecutionResults:
    """Cache for global execution phase results."""

    install: tuple[bool, str] | None = None
    scan: _SecurityScanResult | None = None
    tests: tuple[bool, str] | None = None
    install_exit_code: int | None = None
    install_error_category: str | None = None
    install_raw_stdout: str | None = None
    install_raw_stderr: str | None = None
    peer_conflicts: list[PeerConflictEvidence] = field(default_factory=list)
    test_exit_code: int | None = None
    test_failure_count: int | None = None
    test_raw_stdout: str | None = None
    test_raw_stderr: str | None = None
    log_records: dict[str, tuple[_QALogRecord, ...]] = field(default_factory=dict)
    scan_evidence: ODCScanEvidence | None = None
    package_state_by_task: dict[str, _QAPackageState] = field(default_factory=dict)
    scan_skipped: bool = False
    scan_skip_reason: str | None = None


@dataclass(frozen=True)
class QAScanTarget:
    """Task-owned package target and live lockfile context for QA scanning."""

    task_id: str
    group_id: str
    target_package: str
    expected_version: str | None
    manifest_paths: tuple[str, ...]
    dependency_ancestry: tuple[str, ...]
    target_identifiers: frozenset[str]


@dataclass(frozen=True)
class QATaskContext:
    """Task-owned group context passed through deterministic and LLM QA."""

    task: RemediationTask
    group: VulnerabilityGroup

    @property
    def task_id(self) -> str:
        """Return the authoritative task identifier."""
        return self.task.task_id

    @property
    def group_id(self) -> str:
        """Return the parent vulnerability-group identifier."""
        return self.group.group_id


_ScanValue = TypeVar("_ScanValue")


def _subprocess_text(value: Any) -> str:
    """Convert subprocess output, including timeout byte output, to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _exception_stream(error: BaseException, name: str) -> str:
    """Read a text stream attached to a command exception."""
    return _subprocess_text(getattr(error, name, None))


def _append_qa_log_records(
    results: _QAExecutionResults,
    phase: str,
    records: Sequence[_QALogRecord],
) -> None:
    """Append immutable command records to one phase of the results cache."""
    if not records:
        return
    current = results.log_records.get(phase, ())
    results.log_records[phase] = (*current, *tuple(records))


def _validate_qa_path(file_path: str) -> str:
    """Validate a repo-relative path for QA read-only review tools."""
    if not str(file_path or "").strip():
        raise ValueError("file_path is required.")
    return normalize_workspace_path(file_path)


def _scan_result_value(
    scan_result: _SecurityScanResult | None,
    field: str,
    default: _ScanValue,
) -> _ScanValue:
    """Read a field from a full security-scan result."""
    if scan_result is None:
        return default
    return cast(_ScanValue, getattr(scan_result, field, default))
