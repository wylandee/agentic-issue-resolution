"""Grounded report facts, package changes, and diff evidence."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from remediation_engine.contracts.schemas import model_or_dict_value
from remediation_engine.orchestration.state import (
    effective_group_status,
    task_group_lineage,
    terminal_outcome_issues,
)
from remediation_engine.runtime.telemetry import TrajectoryRecorder
from remediation_engine.tools.identity import canonical_issue_components


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
    canonical_finding_count: int
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
    initial_group_evidence_available: bool
    final_valid_groups: list[Any]
    issues: list[Any]
    task_queue: dict[str, Any]
    action_summaries: list[Any]
    triage_reconciliation: dict[str, Any]
    group_strategies: dict[str, Any]
    retry_plans: dict[str, Any]
    qa_evaluations: dict[str, Any]
    resolved_statuses_by_group: dict[str, Any]
    worker_results: dict[str, Any]
    qa_results: dict[str, Any]
    attempt_snapshots: dict[str, Any]
    retry_diagnostics: dict[str, Any]
    final_full_scan_result: Any
    scan_coverage_scope: str
    scan_coverage_complete: bool | None
    scan_evidence_state: str
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
    group_reconciliation_incomplete: bool = False


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


_KNOWN_ERROR_SOURCES = {
    "docker",
    "incremental_scan",
    "post_remediation_scan",
    "odc",
    "qa_critic",
    "report_node",
    "scan",
    "supervisor",
    "teardown",
    "update_subagent",
    "workaround_subagent",
    "workspace_builder",
}

_KNOWN_ERROR_CODES = {
    "INVALID_PLANNER_COMMIT",
    "ODC_TIMEOUT",
    "PLANNER_SEMANTIC_VALIDATION",
    "QA_OUTPUT_VALIDATION",
    "STALE_ATTEMPT_RESULT",
    "STALE_PIVOT_REPAIR",
    "VALIDATION_INPUT_LIMIT_REACHED",
}

_SCAN_COMPLETE_STATUSES = {
    "clear",
    "completed",
    "detected",
    "none",
    "not_detected",
    "scan_completed",
    "success",
    "unresolved",
}

_value = model_or_dict_value

_USER_FRIENDLY_STATUS_LABELS = {
    "qa_passed": "Fixed",
    "unfixable": "Unresolved",
    "needs_retry": "Retry needed",
    "inconclusive": "Inconclusive",
    "pending": "Pending",
    "optimistically_fixed": "Awaiting validation",
    "awaiting_validation": "Awaiting validation",
    "pivoted": "Unresolved",
}


def _text(value: Any, default: str = "") -> str:
    """Return a display-safe string, unwrapping enum values when present."""
    if value is None:
        return default
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _items(value: Any) -> list[Any]:
    """Normalize an optional sequence into a list."""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def _mapping(value: Any) -> dict[str, Any]:
    """Normalize an optional mapping and preserve deterministic key order."""
    return {str(key): item for key, item in (value.items() if isinstance(value, Mapping) else [])}


def _iso(value: Any) -> str | None:
    """Normalize a timestamp value to ISO-8601 text."""
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    """Calculate elapsed seconds when both timestamps are parseable."""
    if not start or not end:
        return None
    try:
        started = datetime.fromisoformat(start.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=UTC)
        return max(0.0, (ended - started).total_seconds())
    except (TypeError, ValueError):
        return None


def _reconciliation_ids(reconciliation: Mapping[str, Any], *names: str) -> list[str]:
    """Read reconciliation IDs while accepting the plan's and graph's aliases."""
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        for item in _items(reconciliation.get(name)):
            text = str(item)
            if text and text not in seen:
                result.append(text)
                seen.add(text)
    return result


def _group_tree(task_queue: Mapping[str, Any], group_id: str) -> list[Any]:
    """Return the root task and all pivot descendants for one group."""
    return task_group_lineage(task_queue, group_id)


def _task_queue_fallback_groups(
    task_queue: Mapping[str, Any],
    represented_group_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build report group projections for root tasks missing from triage output.

    Args:
        task_queue: Authoritative task-ID to task mapping.
        represented_group_ids: Group IDs already represented by triage groups.

    Returns:
        Stable synthetic group projections for task roots without a triage
        group. Child tasks are represented through their root's task lineage.
    """
    tasks = {str(task_id): task for task_id, task in task_queue.items()}
    represented = set(represented_group_ids or ())
    groups: list[dict[str, Any]] = []
    for _task_id, task in sorted(tasks.items()):
        parent_task_id = _text(_value(task, "parent_task_id"))
        if parent_task_id and parent_task_id in tasks:
            continue
        group_id = _text(_value(task, "parent_group_id"))
        if not group_id or group_id in represented:
            continue
        package = (
            _text(_value(task, "target_package_name")).strip()
            or _text(_value(task, "parent_package_name")).strip()
            or "Unspecified finding"
        )
        groups.append(
            {
                "group_id": group_id,
                "vulnerable_component": package,
                "issue_type": "sca",
                "sources": ["task_queue"],
                "file_path": "unavailable (task queue fallback)",
                "issues": [],
            }
        )
        represented.add(group_id)
    return groups


def _group_status(task_queue: Mapping[str, Any], group_id: str) -> str:
    """Collapse a root task and pivot descendants with failure-first rules."""
    return effective_group_status(task_queue, group_id)


def _report_group_status(context: ReportContext, group_id: str) -> str:
    """Return a report group's effective status derived from task queue lineage."""
    status = context.resolved_statuses_by_group.get(group_id)
    if status is not None:
        return _text(status)
    return _group_status(context.task_queue, group_id)


def _overall_label(
    status: str,
    counts: Mapping[str, int],
    new_vulnerability_status: str,
    new_identifier_count: int,
    remaining_identifier_count: int = 0,
) -> str:
    """Map run state and post-scan evidence to a reader-facing outcome label."""
    if status in {"failed", "error"}:
        base = "Failed"
    elif status == "completed_with_errors":
        base = "Completed with errors"
    elif counts.get("unresolved", 0):
        base = "Partial"
    elif counts.get("inconclusive", 0):
        base = "Inconclusive"
    elif status in {"completed", "triage_completed_no_work"}:
        base = "Successful"
    else:
        base = "Inconclusive"

    if new_vulnerability_status in {"scan_failed", "failed"} and base == "Successful":
        base = "Completed with errors"
    if remaining_identifier_count:
        if base == "Successful":
            base = "Completed with unresolved findings"
        elif base not in {"Failed", "Completed with errors"} and "unresolved" not in base:
            base = f"{base}; unresolved findings"
    if new_vulnerability_status == "detected" or new_identifier_count:
        if base == "Successful":
            return "Completed with new findings"
        if base not in {"Failed", "Completed with errors"}:
            return f"{base}; new findings detected"
    return base


def _extract_token_summary(recorder: TrajectoryRecorder | None) -> dict[str, Any]:
    """Return token metrics without confusing unavailable usage with zero usage."""
    if recorder is None or not getattr(recorder, "token_data_available", False):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "available": False,
        }
    input_tokens = int(getattr(recorder, "total_prompt_tokens", 0))
    output_tokens = int(getattr(recorder, "total_completion_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "available": True,
    }


def _error_source_and_message(candidate: Any) -> tuple[str, str, str | None]:
    """Extract a source, message, and optional explicit code from an error."""
    explicit_code = _text(_value(candidate, "error_code")).strip() or None
    if explicit_code:
        source = _text(_value(candidate, "source"), "consistency").strip() or "consistency"
        message = _text(_value(candidate, "details"), explicit_code).strip()
        return source, message, explicit_code

    source = "run"
    message = _text(candidate).strip()
    while message:
        match = re.match(r"^(?P<source>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<message>.+)$", message)
        if not match:
            break
        prefix = match.group("source")
        if prefix.upper() in _KNOWN_ERROR_CODES:
            return source, match.group("message").strip(), prefix.upper()
        if prefix.lower() not in _KNOWN_ERROR_SOURCES:
            break
        source = prefix.lower()
        message = match.group("message").strip()
    return source, message, None


def _classify_error(source: str, message: str, explicit_code: str | None) -> str:
    """Map known failure wording to a stable report error code."""
    if explicit_code:
        return explicit_code.upper()
    lowered = message.lower()
    if ("timeout" in lowered or "timed out" in lowered) and (
        source in {"incremental_scan", "post_remediation_scan", "odc", "qa_critic"}
        or "odc" in lowered
        or "dependency-check" in lowered
        or "dependency check" in lowered
    ):
        return "ODC_TIMEOUT"
    if "invalid planner commit" in lowered:
        return "INVALID_PLANNER_COMMIT"
    if "planner" in lowered and "semantic" in lowered and "valid" in lowered:
        return "PLANNER_SEMANTIC_VALIDATION"
    if "validation input" in lowered and "limit" in lowered:
        return "VALIDATION_INPUT_LIMIT_REACHED"
    if (
        "stale" in lowered
        and ("attempt" in lowered or "instruction" in lowered)
        or "ignored revised instruction" in lowered
    ):
        return "STALE_ATTEMPT_RESULT"
    if "structured" in lowered and "qa" in lowered and "valid" in lowered:
        return "QA_OUTPUT_VALIDATION"
    if "terminalized" in lowered and "pivot" in lowered:
        return "STALE_PIVOT_REPAIR"
    return "ERROR"


def _error_records(state: Mapping[str, Any]) -> list[ErrorRecord]:
    """Collect critical errors by normalized message/code with source counts."""
    candidates = _items(state.get("errors"))
    scan_summary = state.get("final_full_scan_result")
    scan_error = _value(scan_summary, "error")
    if scan_error:
        candidates.append(f"post_remediation_scan: {scan_error}")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        source, message, explicit_code = _error_source_and_message(candidate)
        message = re.sub(r"\s+", " ", message).strip()
        if not message:
            continue
        code = _classify_error(source, message, explicit_code)
        key = (code, message.casefold())
        record = grouped.setdefault(key, {"source": [], "message": message, "occurrences": 0})
        record["occurrences"] += 1
        if source not in record["source"]:
            record["source"].append(source)

    return [
        ErrorRecord(
            source=", ".join(data["source"]),
            code=code,
            message=data["message"],
            occurrences=int(data["occurrences"]),
        )
        for (code, _), data in grouped.items()
    ]


def _scan_evidence_state(final_scan: Any, new_status: str = "") -> str:
    """Classify full-scan authority or incremental-input coverage separately."""
    if final_scan is None:
        normalized_new_status = new_status.strip().lower()
        if normalized_new_status in {"scan_failed", "failed"}:
            return "failed"
        return "not_scanned"

    status = _text(_value(final_scan, "status")).strip().lower()
    coverage_scope = _text(_value(final_scan, "coverage_scope")).strip().lower()
    if coverage_scope == "incremental_dependency_inputs":
        coverage_complete = _value(final_scan, "coverage_complete")
        completed = _value(final_scan, "completed")
        if coverage_complete is True and completed is not False:
            return "incremental_complete"
        return "incremental_incomplete"

    normalized_new_status = new_status.strip().lower()
    if status in {"scan_failed", "failed", "error", "timeout"} or normalized_new_status in {
        "scan_failed",
        "failed",
    }:
        return "failed"
    if _value(final_scan, "authoritative") is False:
        return "not_scanned"
    completed = _value(final_scan, "completed")
    if completed is False:
        return "not_scanned" if status in {"", "not_scanned"} else "failed"
    if completed is True or status in _SCAN_COMPLETE_STATUSES:
        return "complete"
    return "not_scanned"


def _unique_texts(values: Sequence[Any]) -> list[str]:
    """Return non-empty text values once, preserving their first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _user_friendly_status(status: Any) -> str:
    """Convert an internal lifecycle status into reader-facing language.

    Args:
        status: Enum or string status from a task or group projection.

    Returns:
        A concise status label that does not expose orchestration terminology.
    """
    normalized = _text(status).strip().casefold()
    if not normalized:
        return "Pending"
    return _USER_FRIENDLY_STATUS_LABELS.get(
        normalized,
        normalized.replace("_", " ").capitalize(),
    )


def _format_duration(seconds: float | None) -> str:
    """Format an elapsed run duration for the summary table.

    Args:
        seconds: Elapsed seconds, or ``None`` while a preliminary report is
            waiting for finalization.

    Returns:
        ``Pending finalization`` when unavailable, a two-decimal seconds value
        for short runs, or a compact hours/minutes/seconds value.
    """
    if seconds is None:
        return "Pending finalization"
    try:
        elapsed = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "Pending finalization"
    if elapsed < 60:
        return f"{elapsed:.2f}s"

    whole_seconds = int(round(elapsed))
    minutes, remainder = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remainder}s"
    return f"{minutes}m {remainder}s"


def _format_token_summary(context: ReportContext) -> str:
    """Format total token usage and its prompt/completion breakdown."""
    total = context.total_tokens
    if (
        total is None
        and context.total_input_tokens is not None
        and context.total_output_tokens is not None
    ):
        total = context.total_input_tokens + context.total_output_tokens
    if total is None:
        return "Unavailable"

    total_text = f"{int(total):,}"
    if context.total_input_tokens is None or context.total_output_tokens is None:
        return total_text
    return (
        f"{total_text} (Input: {int(context.total_input_tokens):,}, "
        f"Output: {int(context.total_output_tokens):,})"
    )


def _qa_record_summary(
    qa_evaluations: Mapping[str, Any],
    qa_results_by_attempt: Mapping[str, Any],
) -> int:
    """Count distinct task-keyed QA verdicts with committed provenance.

    Attempt envelopes are authoritative because they retain the task and
    attempt identity. The derived ``qa_evaluations`` projection contributes
    only task IDs not already represented by an attempt envelope.
    """
    records: dict[str, Any] = {}
    for result in qa_results_by_attempt.values():
        evaluation = _value(result, "evaluation")
        identifier = _text(_value(result, "task_id")) or _text(_value(evaluation, "task_id"))
        if identifier and evaluation is not None:
            records[identifier] = evaluation
    for key, evaluation in qa_evaluations.items():
        identifier = _text(_value(evaluation, "task_id"), str(key))
        if identifier and evaluation is not None:
            records.setdefault(identifier, evaluation)
    return len(records)


def _issue_identifiers(issue: Any) -> list[str]:
    """Return scanner-facing identifiers for one typed vulnerability issue."""
    identifiers: list[str] = []
    seen: set[str] = set()
    for value in (
        _value(issue, "cve_id"),
        _value(issue, "ghsa_id"),
        _value(issue, "rule_id"),
        _value(issue, "finding_id"),
    ):
        identifier = _text(value).strip()
        if not identifier:
            continue
        if re.match(r"^(?:CVE|GHSA)-", identifier, flags=re.IGNORECASE):
            identifier = identifier.upper()
        key = identifier.casefold()
        if key in seen:
            continue
        identifiers.append(identifier)
        seen.add(key)
    return identifiers


def _group_identifiers(group: Any) -> set[str]:
    """Return normalized scanner identifiers associated with a group."""
    identifiers: set[str] = set()
    for identifier_field in ("cve_ids", "ghsa_ids", "finding_ids", "rule_ids", "identifiers"):
        identifiers.update(
            _text(value).strip().casefold()
            for value in _items(_value(group, identifier_field))
            if _text(value).strip()
        )
    for issue in _items(_value(group, "issues")):
        identifiers.update(identifier.casefold() for identifier in _issue_identifiers(issue))
    return identifiers


def _reconcile_authoritative_statuses(
    statuses: Mapping[str, str],
    groups: Sequence[Any],
    final_scan: Any,
    new_status: str,
    remaining_identifiers: Sequence[str],
) -> dict[str, str]:
    """Reopen targeted successes contradicted by a repository-wide scan.

    Incremental dependency scans are intentionally excluded: their evidence
    cannot determine whether unrelated groups outside scanned closures remain
    vulnerable.
    """
    result = dict(statuses)
    if _scan_evidence_state(final_scan, new_status) != "complete":
        return result
    if _text(_value(final_scan, "coverage_scope"), "full_workspace") != "full_workspace":
        return result
    if _value(final_scan, "authoritative") is False:
        return result
    remaining = {identifier.casefold() for identifier in remaining_identifiers}
    if not remaining:
        return result
    for group in groups:
        group_id = _text(_value(group, "group_id"))
        if result.get(group_id) != "qa_passed":
            continue
        if _group_identifiers(group).intersection(remaining):
            result[group_id] = "needs_retry"
    return result


def _build_context(
    state: Mapping[str, Any],
    *,
    trajectory_path: str | None = None,
    trace_url: str | None = None,
    token_summary: Mapping[str, Any] | None = None,
    run_ended_at: datetime | None = None,
) -> ReportContext:
    """Normalize graph state into the renderer's evidence contract."""
    initial_groups = [
        group
        for group in _items(state.get("initial_valid_groups"))
        if not bool(_value(group, "is_synthetic"))
    ]
    if not initial_groups and "initial_valid_groups" not in state:
        initial_groups = [
            group
            for group in _items(state.get("valid_groups"))
            if not bool(_value(group, "is_synthetic"))
        ]
    final_groups = [
        group
        for group in _items(state.get("valid_groups"))
        if not bool(_value(group, "is_synthetic"))
    ]
    task_queue = _mapping(state.get("task_queue"))
    issues = _items(state.get("issues"))
    if not issues:
        issues = [issue for group in initial_groups for issue in _items(_value(group, "issues"))]
    initial_group_ids = {_text(_value(group, "group_id")) for group in initial_groups}
    represented_group_ids = initial_group_ids | {
        _text(_value(group, "group_id")) for group in final_groups
    }
    all_groups = initial_groups + [
        group for group in final_groups if _text(_value(group, "group_id")) not in initial_group_ids
    ]
    all_groups.extend(_task_queue_fallback_groups(task_queue, represented_group_ids))
    statuses = {
        _text(_value(group, "group_id")): _group_status(
            task_queue, _text(_value(group, "group_id"))
        )
        for group in all_groups
    }
    reconciliation = _mapping(state.get("triage_reconciliation"))
    ended = _iso(run_ended_at)
    started = _iso(state.get("run_started_at"))
    tokens = dict(token_summary or {})
    status = _text(state.get("status"), "unknown")
    outcome_issues = terminal_outcome_issues(state)
    if outcome_issues and status == "completed":
        status = "completed_with_errors"
    final_scan = state.get("final_full_scan_result")
    final_scan_issues = _value(final_scan, "found_issues")
    post_scan_issues = (
        _items(final_scan_issues)
        if final_scan_issues is not None
        else _items(state.get("post_remediation_scan_issues"))
    )
    final_scan_identifiers = _value(final_scan, "found_identifiers")
    post_scan_identifiers = _unique_texts(
        _items(final_scan_identifiers)
        if final_scan_identifiers is not None
        else _items(state.get("post_remediation_scan_identifiers"))
    )
    if not post_scan_identifiers:
        post_scan_identifiers = _unique_texts(
            identifier for issue in post_scan_issues for identifier in _issue_identifiers(issue)
        )
    final_new_identifiers = _value(final_scan, "new_identifiers")
    new_identifiers = _unique_texts(
        _items(final_new_identifiers)
        if final_new_identifiers is not None
        else _items(state.get("new_vulnerability_identifiers"))
    )
    final_remaining_identifiers = _value(final_scan, "remaining_target_identifiers")
    remaining_identifiers = _unique_texts(
        _items(final_remaining_identifiers)
        if final_remaining_identifiers is not None
        else _items(state.get("remaining_target_identifiers"))
    )
    final_scan_status = _value(final_scan, "status")
    new_status = _text(
        final_scan_status
        if final_scan_status is not None
        else state.get("new_vulnerability_status"),
        "not_scanned",
    )
    scan_evidence_state = _scan_evidence_state(final_scan, new_status)
    final_scan_triage_required = _value(final_scan, "triage_required")
    triage_required_value = (
        final_scan_triage_required
        if final_scan_triage_required is not None
        else state.get("triage_required")
    )
    triage_required = bool(
        _value(final_scan, "triage_required")
        if triage_required_value is None
        else triage_required_value
    )
    statuses = _reconcile_authoritative_statuses(
        statuses,
        all_groups,
        final_scan,
        new_status,
        remaining_identifiers,
    )
    status_counts = Counter(
        statuses.get(_text(_value(group, "group_id")), "pending") for group in all_groups
    )
    counts = {
        "fixed": status_counts.get("qa_passed", 0),
        "unresolved": status_counts.get("unfixable", 0) + status_counts.get("needs_retry", 0),
        "inconclusive": status_counts.get("inconclusive", 0),
        "pending": status_counts.get("pending", 0) + status_counts.get("optimistically_fixed", 0),
    }
    discovered_ids = _reconciliation_ids(
        reconciliation,
        "added",
        "new_group_ids",
        "reappeared",
        "reappeared_group_ids",
        "final_scan_reopened_group_ids",
    )
    scan_findings_usable = scan_evidence_state in {
        "complete",
        "incremental_complete",
    } or (scan_evidence_state == "incremental_incomplete" and bool(new_identifiers))
    if scan_findings_usable and (discovered_ids or (not triage_required and not new_identifiers)):
        discovered_status_counts = Counter(
            statuses.get(group_id, "pending") for group_id in discovered_ids
        )
        new_group_metrics: dict[str, int | None] = {
            "discovered": len(discovered_ids),
            "unresolved": discovered_status_counts.get("unfixable", 0)
            + discovered_status_counts.get("needs_retry", 0),
            "inconclusive": discovered_status_counts.get("inconclusive", 0),
            "pending": discovered_status_counts.get("pending", 0)
            + discovered_status_counts.get("optimistically_fixed", 0),
        }
    else:
        new_group_metrics = {
            "discovered": None,
            "unresolved": None,
            "inconclusive": None,
            "pending": None,
        }
    report_state = {
        **state,
        "errors": [
            *(list(state.get("errors", []) or [])),
            *(f"remediation outcome: {issue}" for issue in outcome_issues),
        ],
    }
    error_records = _error_records(report_state)
    qa_evaluations = _mapping(state.get("qa_evaluations"))
    qa_results = _mapping(state.get("qa_results_by_attempt"))
    recorded_qa_total = _qa_record_summary(qa_evaluations, qa_results)
    return ReportContext(
        run_id=_text(state.get("run_id") or state.get("langsmith_run_id"), "local-run"),
        repo_root=_text(state.get("repo_root")),
        run_started_at=started,
        run_ended_at=ended,
        duration_seconds=_duration_seconds(started, ended),
        total_input_tokens=tokens.get("input_tokens"),
        total_output_tokens=tokens.get("output_tokens"),
        total_tokens=tokens.get("total_tokens"),
        status=status,
        overall_label=_overall_label(
            status,
            counts,
            new_status,
            len(new_identifiers),
            len(remaining_identifiers),
        ),
        targeted_qa_total=len(initial_groups),
        targeted_qa_passed=sum(
            statuses.get(_text(_value(group, "group_id")), "pending") == "qa_passed"
            for group in initial_groups
        ),
        recorded_qa_total=recorded_qa_total,
        has_patch=bool(_text(state.get("diff"))),
        canonical_finding_count=len(canonical_issue_components(issues)),
        actionable_groups=len(all_groups),
        groups_fixed=counts["fixed"],
        groups_unresolved=counts["unresolved"],
        groups_inconclusive=counts["inconclusive"],
        groups_pending=counts["pending"],
        groups_retriage_discovered=new_group_metrics["discovered"],
        group_reconciliation_incomplete=(
            bool(issues)
            and not initial_groups
            and state.get("initial_triage_status") != "completed_no_work"
        ),
        consistency_events=_items(state.get("consistency_events")),
        error_strings=[
            f"{record.source}/{record.code}: {record.message}" for record in error_records
        ],
        error_records=error_records,
        initial_valid_groups=initial_groups,
        initial_group_evidence_available="initial_valid_groups" in state,
        final_valid_groups=final_groups,
        issues=issues,
        task_queue=task_queue,
        action_summaries=_items(state.get("action_summaries")),
        triage_reconciliation=reconciliation,
        group_strategies=_mapping(state.get("group_strategies")),
        retry_plans=_mapping(state.get("retry_plans_by_task")),
        qa_evaluations=qa_evaluations,
        resolved_statuses_by_group=statuses,
        worker_results=_mapping(state.get("worker_results_by_attempt")),
        qa_results=qa_results,
        attempt_snapshots=_mapping(state.get("attempt_snapshots_by_id")),
        retry_diagnostics=_mapping(state.get("retry_diagnostics_by_task")),
        final_full_scan_result=final_scan,
        scan_coverage_scope=_text(
            _value(final_scan, "coverage_scope"),
            "full_workspace" if final_scan is not None else "unknown",
        ),
        scan_coverage_complete=_value(final_scan, "coverage_complete"),
        scan_evidence_state=scan_evidence_state,
        triage_required=triage_required,
        post_remediation_scan_issues=post_scan_issues,
        post_remediation_scan_identifiers=post_scan_identifiers,
        new_vulnerability_identifiers=new_identifiers,
        new_vulnerability_status=new_status,
        remaining_target_identifiers=remaining_identifiers,
        diff=_text(state.get("diff")),
        changed_files=[_text(item) for item in _items(state.get("changed_files"))],
        trajectory_path=trajectory_path or _text(state.get("trajectory_path")) or None,
        langsmith_trace_url=trace_url or _text(state.get("langsmith_trace_url")) or None,
        new_groups_discovered=new_group_metrics["discovered"],
        new_groups_unresolved=new_group_metrics["unresolved"],
        new_groups_inconclusive=new_group_metrics["inconclusive"],
        new_groups_pending=new_group_metrics["pending"],
        workaround_replay_plans=_mapping(state.get("workaround_replay_plans_by_task")),
    )


def _inline_code_text(value: Any, limit: int = 420) -> str:
    """Compact a code fragment without losing its leading change semantics."""
    text = re.sub(r"\s+", " ", _text(value).strip())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 13)].rstrip() + "... (truncated)"


_PACKAGE_FILE_RE = re.compile(
    r"(?:^|/)(?:package\.json|package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|"
    r"pnpm-lock\.yaml|pom\.xml|requirements\.txt)$",
    re.IGNORECASE,
)

_POM_FILE_RE = re.compile(r"(?:^|/)pom\.xml$", re.IGNORECASE)

_PACKAGE_LINE_RE = re.compile(r'^\s*"(?P<name>(?:@[^" ]+/)?[^" ]+)"\s*:\s*"(?P<version>[^"\n]+)"')

_LOCKFILE_PACKAGE_RE = re.compile(r'^\s*"(?P<name>(?:node_modules/)?(?:@[^" ]+/)?[^" ]+)"\s*:\s*\{')

_LOCKFILE_RESOLVED_RE = re.compile(r'^\s*"resolved"\s*:\s*"(?P<url>[^"\n]+)"')

_MANIFEST_SECTION_RE = re.compile(
    r'^\s*"(?P<section>dependencies|devDependencies|optionalDependencies|'
    r'peerDependencies|overrides|resolutions)"\s*:\s*\{'
)

_NON_PACKAGE_KEYS = {
    "author",
    "bin",
    "browser",
    "bundleDependencies",
    "bundled",
    "contributors",
    "cpu",
    "description",
    "name",
    "deprecated",
    "directories",
    "engines",
    "engineStrict",
    "files",
    "funding",
    "hasInstallScript",
    "homepage",
    "keywords",
    "license",
    "main",
    "man",
    "module",
    "node",
    "optional",
    "os",
    "peer",
    "peerDependenciesMeta",
    "publishConfig",
    "readme",
    "repository",
    "version",
    "lockfileVersion",
    "requires",
    "integrity",
    "resolved",
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
    "packages",
    "snapshots",
    "scripts",
    "sideEffects",
    "type",
    "types",
    "workspaces",
}


def _normalise_replay_text(value: str) -> str:
    """Normalize line endings before replaying an attempt replacement."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _diff_file_paths(diff: str) -> list[str]:
    """Return repository paths represented by unified-diff file headers."""
    return _unique_texts(
        line[6:].strip() for line in diff.splitlines() if line.startswith("+++ b/")
    )


def _normalized_path(value: Any) -> str:
    """Normalize a repository path for comparisons while preserving display text elsewhere."""
    return _text(value).strip().replace("\\", "/").removeprefix("./")


def _path_matches_any(path: Any, candidates: Sequence[Any]) -> bool:
    """Return whether a path matches one of several relative path candidates."""
    normalized = _normalized_path(path)
    if not normalized:
        return False
    for candidate in candidates:
        candidate_normalized = _normalized_path(candidate)
        if not candidate_normalized:
            continue
        if normalized == candidate_normalized:
            return True
        if normalized.endswith(f"/{candidate_normalized}") or candidate_normalized.endswith(
            f"/{normalized}"
        ):
            return True
    return False


def _package_name_from_resolved_url(value: Any) -> str:
    """Extract an npm package name from a registry tarball URL."""
    match = re.search(
        r"(?:https?://[^/]+/)(?P<package>(?:@[^/]+/)?[^/]+)/-/",
        _text(value),
        re.IGNORECASE,
    )
    return match.group("package") if match else ""


def _diff_line_changes(diff: str) -> dict[str, list[tuple[str, str]]]:
    """Collect added and removed lines by file from a unified diff."""
    changes: dict[str, list[tuple[str, str]]] = {}
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if line.startswith("--- a/"):
            continue
        if not current_file or line.startswith(("+++", "---")):
            continue
        if line[:1] in {"+", "-"}:
            content = _inline_code_text(_diff_content(line).strip())
            changes.setdefault(current_file, []).append((line[:1], content))
    return changes


def _diff_code_change_details(diff: str, files: Sequence[Any] = ()) -> list[str]:
    """Summarize source-line additions and removals for selected files."""
    details: list[str] = []
    for path, line_changes in _diff_line_changes(diff).items():
        if _PACKAGE_FILE_RE.search(path) or (files and not _path_matches_any(path, files)):
            continue
        index = 0
        while index < len(line_changes):
            prefix, content = line_changes[index]
            if (
                prefix == "-"
                and index + 1 < len(line_changes)
                and line_changes[index + 1][0] == "+"
            ):
                details.append(
                    f"{path}: removed {content or 'blank line'}; added {line_changes[index + 1][1] or 'blank line'}"
                )
                index += 2
                continue
            verb = "added" if prefix == "+" else "removed"
            details.append(f"{path}: {verb} {content or 'blank line'}")
            index += 1
    return details


def _unified_diff_blocks(
    diff: str,
    files: Sequence[Any] | None = None,
    *,
    include_package_files: bool = False,
) -> list[str]:
    """Return complete unified-diff file blocks for selected source files.

    The report uses these blocks for code workarounds so a reviewer can inspect
    the exact source edit. Package manifests are normally excluded because
    their compact version transition is already rendered as prose. Package
    removal attempts opt in so their manifest edits are visible beside the
    source changes.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_path = ""
    current_lines: list[str] = []

    def finish() -> None:
        if current_path and current_lines:
            blocks.append((current_path, list(current_lines)))

    for line in diff.splitlines():
        if line.startswith("--- a/"):
            finish()
            current_path = ""
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)
            if line.startswith("+++ b/"):
                current_path = line[6:].strip()
    finish()

    selected = (
        [_normalized_path(path) for path in files if _normalized_path(path)]
        if files is not None
        else None
    )
    result: list[str] = []
    for path, lines in blocks:
        if not include_package_files and _PACKAGE_FILE_RE.search(path):
            continue
        if selected is not None and (not selected or not _path_matches_any(path, selected)):
            continue
        result.append("\n".join(lines).strip())
    return result


def _replay_diff_blocks(
    metadata: Any,
    files: Sequence[Any] | None = None,
    *,
    include_package_files: bool = False,
) -> list[str]:
    """Build per-attempt unified-diff blocks from committed replacements.

    A final workspace diff can contain edits from several workaround tasks that
    touched the same source file. Replay replacements are attempt-scoped, so
    they are the authoritative source for isolating those edits in a report.
    Multiple replacements for one file are kept in one code block.
    """
    replay_plan = _value(metadata, "replay_plan")
    if replay_plan is None:
        return []

    selected = (
        [_normalized_path(path) for path in files if _normalized_path(path)]
        if files is not None
        else None
    )
    replacements_by_path: dict[str, list[Any]] = {}
    for edit_set in _items(_value(replay_plan, "successful_edit_sets")):
        for replacement in _items(_value(edit_set, "replacements")):
            path = _normalized_path(_value(replacement, "file_path"))
            if not path or (not include_package_files and _PACKAGE_FILE_RE.search(path)):
                continue
            if selected is not None and (not selected or not _path_matches_any(path, selected)):
                continue
            replacements_by_path.setdefault(path, []).append(replacement)

    snapshots = _value(replay_plan, "pre_attempt_snapshots")
    snapshot_by_path = (
        {
            _normalized_path(path): content
            for path, content in snapshots.items()
            if _normalized_path(path)
        }
        if isinstance(snapshots, Mapping)
        else {}
    )
    blocks: list[str] = []
    for path, replacements in replacements_by_path.items():
        baseline = snapshot_by_path.get(path)
        if isinstance(baseline, str):
            current = baseline
            replay_valid = True
            for replacement in replacements:
                old_text = _normalise_replay_text(_text(_value(replacement, "old_text")))
                new_text = _normalise_replay_text(_text(_value(replacement, "new_text")))
                expected = _value(replacement, "expected_occurrences", 1) or 1
                count = _normalise_replay_text(current).count(old_text)
                if count == expected:
                    current = _normalise_replay_text(current).replace(old_text, new_text, expected)
                elif count == 0 and _normalise_replay_text(current).count(new_text) >= expected:
                    # The edit is already present in a cumulative retry
                    # workspace; it contributes no new net hunk.
                    continue
                else:
                    replay_valid = False
                    break

            if replay_valid and current != baseline:
                before_lines = baseline.splitlines()
                after_lines = current.splitlines()
                hunks: list[str] = []
                matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
                for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    hunks.extend(
                        [
                            "@@",
                            *(f"-{line}" for line in before_lines[before_start:before_end]),
                            *(f"+{line}" for line in after_lines[after_start:after_end]),
                        ]
                    )
                if hunks:
                    blocks.append("\n".join([f"--- a/{path}", f"+++ b/{path}", *hunks]))
                    continue

        # Legacy/sparse replay plans may not carry a baseline snapshot. Keep
        # their exact replacement evidence, but group all replacements for a
        # file into one code block.
        hunks = []
        for replacement in replacements:
            old_lines = _text(_value(replacement, "old_text")).splitlines() or [""]
            new_lines = _text(_value(replacement, "new_text")).splitlines() or [""]
            hunks.extend(
                [
                    "@@",
                    *(f"-{line}" for line in old_lines),
                    *(f"+{line}" for line in new_lines),
                ]
            )
        if hunks:
            blocks.append("\n".join([f"--- a/{path}", f"+++ b/{path}", *hunks]))
    return blocks


def _diff_block_paths(blocks: Sequence[str]) -> set[str]:
    """Return normalized file paths represented by unified-diff blocks."""
    paths: set[str] = set()
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("+++ b/"):
                paths.add(_normalized_path(line[6:].strip()))
                break
    return paths


def _diff_change_counts(block: str) -> tuple[Counter[str], Counter[str]]:
    """Return removed and added diff lines from one unified-diff block."""
    removed: Counter[str] = Counter()
    added: Counter[str] = Counter()
    for line in block.splitlines():
        if line.startswith(("--- a/", "+++ b/", "@@")):
            continue
        if line.startswith("-"):
            removed[line[1:]] += 1
        elif line.startswith("+"):
            added[line[1:]] += 1
    return removed, added


def _replay_blocks_are_backed_by_diff(
    final_diff: str,
    replay_blocks: Sequence[str],
) -> bool:
    """Return whether each replay change is present in the emitted final diff."""
    if not replay_blocks:
        return False

    final_blocks_by_path = {
        path: block
        for block in _unified_diff_blocks(final_diff, None, include_package_files=True)
        for path in _diff_block_paths([block])
    }
    for replay_block in replay_blocks:
        replay_paths = _diff_block_paths([replay_block])
        if not replay_paths:
            return False
        expected_removed, expected_added = _diff_change_counts(replay_block)
        for path in replay_paths:
            final_block = final_blocks_by_path.get(path)
            if final_block is None:
                return False
            actual_removed, actual_added = _diff_change_counts(final_block)
            if expected_removed - actual_removed or expected_added - actual_added:
                return False
    return True


def _diff_content(line: str) -> str:
    """Remove the unified-diff prefix from one line."""
    return line[1:] if line[:1] in {"+", "-", " "} else line


def _line_indent(line: str) -> int:
    """Return the leading-space count for diff content."""
    return len(line) - len(line.lstrip())


def _record_package_change(
    records: dict[str, dict[str, str]],
    name: str,
    version: str,
    prefix: str,
    file_path: str,
    section: str = "",
) -> None:
    """Record one added or removed package version from a diff line."""
    if name in _NON_PACKAGE_KEYS:
        return
    record = records.setdefault(
        name,
        {"old": "", "new": "", "file": file_path, "section": section},
    )
    recorded_files = [item.strip() for item in record.get("file", "").split(";")]
    if file_path and file_path not in recorded_files:
        record["file"] = "; ".join([*recorded_files, file_path])
    if section and not record.get("section"):
        record["section"] = section
    if prefix == "+":
        record["new"] = version
    elif prefix == "-":
        record["old"] = version


def _package_changes(diff: str) -> list[PackageChange]:
    """Extract direct manifest changes and classified lockfile changes.

    Only dependency-section entries from ``package.json`` are considered
    direct changes. Lockfiles contribute only package ``version`` fields under
    package entries; metadata such as ``engines.node`` or ``deprecated`` is
    deliberately excluded.
    """
    manifest_records: dict[str, dict[str, str]] = {}
    lockfile_records: dict[str, dict[str, str]] = {}
    current_file = ""
    manifest_section: tuple[str, int] | None = None
    lockfile_package: tuple[str, int] | None = None
    lockfile_version_candidates: list[tuple[int, str, str, str, str]] = []

    for line_number, line in enumerate(diff.splitlines()):
        if line.startswith("@@"):
            manifest_section = None
            lockfile_package = None
            lockfile_version_candidates.clear()
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            manifest_section = None
            lockfile_package = None
            lockfile_version_candidates.clear()
            continue
        if line.startswith("--- a/") or not _PACKAGE_FILE_RE.search(current_file):
            continue
        # POM XML is a package manifest, but it must never enter the npm
        # JSON/lockfile line parser below.
        if _POM_FILE_RE.search(current_file):
            continue
        prefix = line[:1]
        content = _diff_content(line)
        if current_file.casefold().endswith("requirements.txt"):
            match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)", content)
            if prefix in {"+", "-"} and match:
                _record_package_change(
                    manifest_records, match[1], match[2], prefix, current_file, "requirements"
                )
            continue
        indent = _line_indent(content)

        if current_file.lower().endswith("package.json") and not current_file.lower().endswith(
            "package-lock.json"
        ):
            section_match = _MANIFEST_SECTION_RE.match(content)
            if section_match:
                manifest_section = (section_match.group("section"), indent)
                continue
            if manifest_section is not None:
                section_name, section_indent = manifest_section
                if content.strip() and indent <= section_indent:
                    manifest_section = None
                elif prefix in {"+", "-"}:
                    package_match = _PACKAGE_LINE_RE.match(content)
                    if package_match:
                        _record_package_change(
                            manifest_records,
                            package_match.group("name"),
                            package_match.group("version"),
                            prefix,
                            current_file,
                            section_name,
                        )
            elif prefix in {"+", "-"}:
                # Sparse diffs may omit the surrounding dependency-section
                # context. The metadata deny-list keeps this fallback bounded.
                package_match = _PACKAGE_LINE_RE.match(content)
                if package_match:
                    _record_package_change(
                        manifest_records,
                        package_match.group("name"),
                        package_match.group("version"),
                        prefix,
                        current_file,
                        "",
                    )
            continue

        if "lock" in current_file.lower():
            package_match = _LOCKFILE_PACKAGE_RE.match(content)
            if package_match:
                raw_name = package_match.group("name")
                name = raw_name.removeprefix("node_modules/")
                if name and name not in _NON_PACKAGE_KEYS:
                    lockfile_package = (name, indent)
                    lockfile_version_candidates.clear()
                elif lockfile_package and indent <= lockfile_package[1]:
                    lockfile_package = None
            elif (
                lockfile_package
                and content.strip().startswith("}")
                and indent <= lockfile_package[1]
            ):
                lockfile_package = None

            if prefix in {"+", "-"} and lockfile_package:
                version_match = _PACKAGE_LINE_RE.match(content)
                if (
                    version_match
                    and version_match.group("name") == "version"
                    and indent > lockfile_package[1]
                ):
                    version = version_match.group("version")
                    _record_package_change(
                        lockfile_records,
                        lockfile_package[0],
                        version,
                        prefix,
                        current_file,
                        "lockfile",
                    )
                    lockfile_version_candidates.append(
                        (line_number, prefix, current_file, lockfile_package[0], version)
                    )

            resolved_match = _LOCKFILE_RESOLVED_RE.match(content)
            if prefix in {"+", "-"} and resolved_match:
                resolved_name = _package_name_from_resolved_url(resolved_match.group("url"))
                candidate = next(
                    (
                        item
                        for item in reversed(lockfile_version_candidates)
                        if item[1] == prefix
                        and item[2] == current_file
                        and 0 < line_number - item[0] <= 4
                    ),
                    None,
                )
                if resolved_name and candidate is not None and candidate[3] != resolved_name:
                    _, _, _, previous_name, version = candidate
                    previous_record = lockfile_records.get(previous_name)
                    version_key = "new" if prefix == "+" else "old"
                    if previous_record is not None and previous_record.get(version_key) == version:
                        previous_record[version_key] = ""
                    _record_package_change(
                        lockfile_records,
                        resolved_name,
                        version,
                        prefix,
                        current_file,
                        "lockfile",
                    )

    direct_names = set(manifest_records)
    changes: list[PackageChange] = []
    for name in sorted(manifest_records):
        record = manifest_records[name]
        if record["old"] or record["new"]:
            evidence_file = record["file"]
            if name in lockfile_records:
                evidence_file += f"; {lockfile_records[name]['file']}"
            changes.append(
                PackageChange(
                    name,
                    record["old"],
                    record["new"],
                    evidence_file,
                    "direct",
                    record.get("section", ""),
                )
            )
    for name in sorted(lockfile_records):
        if name in direct_names:
            continue
        record = lockfile_records[name]
        if record["old"] or record["new"]:
            changes.append(
                PackageChange(
                    name,
                    record["old"],
                    record["new"],
                    record["file"],
                    "transitive",
                    record.get("section", "lockfile"),
                )
            )
    return sorted(changes, key=lambda change: (change.scope != "direct", change.name))
