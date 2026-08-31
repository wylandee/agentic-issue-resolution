"""Typed deterministic context shared by report extraction and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReportContext:
    """Normalized, deterministic evidence supplied to report renderers."""

    run_id: str
    repo_root: str
    run_started_at: str | None
    run_ended_at: str | None
    duration_seconds: float | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    status: str
    overall_label: str
    targeted_qa_total: int
    targeted_qa_passed: int
    recorded_qa_total: int
    has_patch: bool
    original_scanner_findings: int
    actionable_groups: int
    groups_fixed: int
    groups_unresolved: int
    groups_inconclusive: int
    groups_pending: int
    groups_retriage_discovered: int | None
    consistency_events: list[Any]
    error_strings: list[str]
    error_records: list[ErrorRecord]
    initial_valid_groups: list[Any]
    final_valid_groups: list[Any]
    issues: list[Any]
    task_queue: dict[str, Any]
    action_summaries: list[Any]
    triage_reconciliation: dict[str, Any]
    group_strategies: dict[str, Any]
    retry_plans: dict[str, Any]
    qa_evaluations: dict[str, Any]
    group_statuses: dict[str, Any]
    worker_results: dict[str, Any]
    qa_results: dict[str, Any]
    attempt_snapshots: dict[str, Any]
    retry_diagnostics: dict[str, Any]
    final_full_scan_result: Any
    triage_required: bool
    post_remediation_scan_issues: list[Any]
    post_remediation_scan_identifiers: list[str]
    remaining_target_identifiers: list[str]
    new_vulnerability_identifiers: list[str]
    new_vulnerability_status: str
    diff: str
    changed_files: list[str]
    trajectory_path: str | None
    langsmith_trace_url: str | None
    new_groups_discovered: int | None = None
    new_groups_unresolved: int | None = None
    new_groups_inconclusive: int | None = None
    new_groups_pending: int | None = None
    workaround_replay_plans: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackageChange:
    """One direct or lockfile package version change extracted from a diff."""

    name: str
    old: str
    new: str
    file: str
    scope: str
    section: str = ""


@dataclass(frozen=True)
class ErrorRecord:
    """One deduplicated, source-aware critical error for the report."""

    source: str
    code: str
    message: str
    occurrences: int


__all__ = ["ErrorRecord", "PackageChange", "ReportContext"]
