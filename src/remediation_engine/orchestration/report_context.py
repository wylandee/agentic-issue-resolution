"""Typed deterministic context shared by report extraction and rendering."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from remediation_engine.contracts.accessors import model_or_dict_value

from .task_utils import effective_group_status, task_group_lineage, terminal_outcome_issues
from .trajectory_exporter import TrajectoryRecorder


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
    resolved_statuses_by_group: dict[str, Any]
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


_KNOWN_ERROR_SOURCES = {
    "docker",
    "final_full_scan",
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
        source in {"final_full_scan", "odc", "qa_critic"}
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
    final_scan = state.get("final_full_scan_result")
    scan_error = _value(final_scan, "error")
    if scan_error:
        candidates.append(f"final_full_scan: {scan_error}")

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
    """Classify authoritative post-remediation scan evidence.

    Returns `complete` only when the scan produced a usable result,
    `failed` when it explicitly failed, and `not_scanned` when no
    authoritative result is available.
    """
    status = _text(_value(final_scan, "status")).strip().lower()
    normalized_new_status = new_status.strip().lower()
    if status in {"scan_failed", "failed", "error", "timeout"} or normalized_new_status in {
        "scan_failed",
        "failed",
    }:
        return "failed"
    if final_scan is None:
        return "not_scanned"
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
    """Reopen targeted successes contradicted by an authoritative scan.

    Targeted QA can cover one manifest or dependency closure while the final
    scan covers the whole repository.  A successful task therefore cannot
    remain a successful report row when the authoritative scan still contains
    one of that group's original vulnerability identifiers.
    """
    result = dict(statuses)
    if _scan_evidence_state(final_scan, new_status) != "complete":
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
    if not initial_groups:
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
    initial_group_ids = {_text(_value(group, "group_id")) for group in initial_groups}
    all_groups = initial_groups + [
        group for group in final_groups if _text(_value(group, "group_id")) not in initial_group_ids
    ]
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
    issues = _items(state.get("issues"))
    if not issues:
        issues = [issue for group in initial_groups for issue in _items(_value(group, "issues"))]
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
    if scan_evidence_state == "complete" and (
        discovered_ids or (not triage_required and not new_identifiers)
    ):
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
        original_scanner_findings=len(issues),
        actionable_groups=len(all_groups),
        groups_fixed=counts["fixed"],
        groups_unresolved=counts["unresolved"],
        groups_inconclusive=counts["inconclusive"],
        groups_pending=counts["pending"],
        groups_retriage_discovered=new_group_metrics["discovered"],
        consistency_events=_items(state.get("consistency_events")),
        error_strings=[
            f"{record.source}/{record.code}: {record.message}" for record in error_records
        ],
        error_records=error_records,
        initial_valid_groups=initial_groups,
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


__all__ = [
    "ErrorRecord",
    "PackageChange",
    "ReportContext",
]
