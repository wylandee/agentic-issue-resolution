"""Human-readable reporting for a Phase 5 remediation run.

The graph node renders the state available after teardown and before the graph
exits. The orchestrator then finalizes and persists that report before the
final trajectory export, so both artifacts contain the same report metadata.
Report facts, statuses, findings, and file changes are derived
deterministically; the report contains no model-generated narrative or error
telemetry.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from remediation_engine.settings import AppSettings

from .report_context import ErrorRecord, PackageChange, ReportContext
from .report_persistence import (
    report_filename,
    resolve_report_dir,
    write_report_atomic,
)
from .runtime_context import get_runtime_settings
from .task_utils import effective_group_status, task_group_lineage, terminal_outcome_issues
from .trajectory_exporter import TrajectoryRecorder

log = logging.getLogger(__name__)

# Private aliases preserve imports used by older report-focused integrations
# while the typed context definitions live in their own boundary module.
_ReportContext = ReportContext
_PackageChange = PackageChange
_ErrorRecord = ErrorRecord

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "data" / "reports"
_PACKAGE_FILE_RE = re.compile(
    r"(?:^|/)(?:package\.json|package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml)$",
    re.IGNORECASE,
)
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


def _value(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic object or a mapping."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _text(value: Any, default: str = "") -> str:
    """Return a display-safe string, unwrapping enum values when present."""
    if value is None:
        return default
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _full_text(value: Any, default: str = "—") -> str:
    """Return complete report prose without truncating or normalizing it.

    Markdown table escaping is applied later by :func:`_escape_cell`, so this
    helper deliberately preserves internal whitespace and line breaks. Empty
    values still receive the report's standard placeholder.
    """
    text = _text(value)
    return text if text.strip() else default


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


def _error_strings(state: Mapping[str, Any]) -> list[str]:
    """Return stable error strings for bounded narrative evidence."""
    return [f"{record.source}/{record.code}: {record.message}" for record in _error_records(state)]


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


_USER_FRIENDLY_STATUS_LABELS = {
    "qa_passed": "Fixed",
    "mitigated": "Fixed",
    "unfixable": "Unresolved",
    "needs_retry": "Retry needed",
    "inconclusive": "Inconclusive",
    "pending": "Pending",
    "optimistically_fixed": "Awaiting validation",
    "awaiting_validation": "Awaiting validation",
    "pivoted": "Unresolved",
}


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


def _diff_file_paths(diff: str) -> list[str]:
    """Return repository paths represented by unified-diff file headers."""
    return _unique_texts(
        line[6:].strip() for line in diff.splitlines() if line.startswith("+++ b/")
    )


def _patch_file_paths(context: ReportContext) -> list[str]:
    """Return changed files from the patch projection and diff headers."""
    return _unique_texts([*context.changed_files, *_diff_file_paths(context.diff)])


def _format_patch_status(context: ReportContext) -> str:
    """Format whether a unified patch is available and how many files changed."""
    if not context.has_patch:
        return "Not available"
    count = len(_patch_file_paths(context))
    noun = "file" if count == 1 else "files"
    return f"Available ({count} {noun} changed)"


def _qa_record_summary(
    qa_evaluations: Mapping[str, Any],
    qa_results_by_attempt: Mapping[str, Any],
) -> int:
    """Count distinct recorded QA verdicts.

    Attempt envelopes are preferred because they are task-keyed and preserve
    provenance. The legacy evaluation map is used as a fallback for callers
    that construct state without attempt envelopes.
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
    for field in ("cve_ids", "ghsa_ids", "finding_ids", "rule_ids", "identifiers"):
        identifiers.update(
            _text(value).strip().casefold()
            for value in _items(_value(group, field))
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
        if result.get(group_id) not in {"qa_passed", "mitigated"}:
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
    initial_groups = _items(state.get("initial_valid_groups"))
    if not initial_groups:
        initial_groups = _items(state.get("valid_groups"))
    final_groups = _items(state.get("valid_groups"))
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
        "fixed": status_counts.get("qa_passed", 0) + status_counts.get("mitigated", 0),
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
        group_statuses=statuses,
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


def _escape_cell(value: Any) -> str:
    """Escape Markdown table delimiters and compact multiline values."""
    text = _text(value, "—")
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")
    return text.replace("\r\n", "<br>").replace("\n", "<br>") or "—"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a small Markdown table with deterministic escaping."""
    header_line = "| " + " | ".join(_escape_cell(header) for header in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _group_issue(group: Any) -> Any | None:
    """Select the representative issue for a vulnerability group."""
    issues = _items(_value(group, "issues"))
    representative_id = _value(group, "representative_issue_id")
    if representative_id:
        for issue in issues:
            if _text(_value(issue, "id")) == _text(representative_id):
                return issue
    return issues[0] if issues else None


def _group_severity(group: Any, issue: Any | None) -> str:
    """Return the original scanner severity for a group."""
    return (
        _text(
            _value(issue, "severity") if issue is not None else _value(group, "severity"),
            "UNKNOWN",
        )
        .strip()
        .upper()
    )


def _group_package_or_target(context: ReportContext, group: Any) -> str:
    """Return the package a remediation action should edit.

    A transitive finding may have a different editable target: for example,
    the finding can be for ``lodash`` while the worker updates its directly
    declared parent ``sanitize-html``.  Callers that identify the finding
    itself must use :func:`_group_finding_package` instead.
    """
    group_id = _text(_value(group, "group_id"))
    for task in reversed(_group_tree(context.task_queue, group_id)):
        target = _text(_value(task, "target_package_name")).strip()
        if target:
            return target

    issue = _group_issue(group)
    for value in (
        _value(issue, "package_name"),
        _value(group, "package_name"),
        _value(group, "vulnerable_component"),
    ):
        candidate = _text(value).strip()
        if not candidate:
            continue
        # Group components may carry a parent/package suffix. The first
        # component is the clearest package label when no typed package field
        # is available.
        return re.split(r"\s*[|,]\s*", candidate, maxsplit=1)[0]
    return "Unspecified target"


def _group_finding_package(group: Any) -> str:
    """Return the package or component identified by the scanner finding.

    The vulnerable component is the stable finding identity.  It must not be
    replaced by a remediation target selected for a transitive dependency.
    """
    issue = _group_issue(group)
    for value in (
        _value(group, "vulnerable_component"),
        _value(issue, "package_name"),
        _value(group, "package_name"),
    ):
        candidate = _text(value).strip()
        if not candidate:
            continue
        return re.split(r"\s*[|,]\s*", candidate, maxsplit=1)[0]
    return "Unspecified finding"


def _finding_identifier(group: Any, issue: Any | None) -> str:
    """Return a stable human-facing identifier for a finding row."""
    identifiers = (
        _items(_value(group, "cve_ids"))
        + _items(_value(group, "ghsa_ids"))
        + _items(_value(group, "finding_ids"))
        + _items(_value(group, "rule_ids"))
    )
    if issue is not None:
        identifiers.extend(
            [
                _value(issue, "cve_id"),
                _value(issue, "ghsa_id"),
                _value(issue, "finding_id"),
                _value(issue, "rule_id"),
            ]
        )
    labels = sorted({_text(identifier) for identifier in identifiers if identifier})
    return ", ".join(labels) or _text(_value(group, "group_id"), "unknown")


_OUTSTANDING_GROUP_STATUSES = frozenset(
    {
        "unfixable",
        "needs_retry",
        "inconclusive",
        "pending",
        "optimistically_fixed",
        "awaiting_validation",
        "pivoted",
    }
)


def _report_groups(context: ReportContext) -> list[Any]:
    """Return initial and final groups once, preserving stable report order."""
    groups: list[Any] = []
    seen: set[str] = set()
    for group in [*context.initial_valid_groups, *context.final_valid_groups]:
        group_id = _text(_value(group, "group_id"))
        if not group_id or group_id in seen:
            continue
        seen.add(group_id)
        groups.append(group)
    return sorted(groups, key=lambda item: _text(_value(item, "group_id")))


def _lineage_root_group_id(context: ReportContext, group_id: str) -> str:
    """Return the original group ID for a pivot child group."""
    task_queue = context.task_queue
    candidates = sorted(
        (str(task_id), task)
        for task_id, task in task_queue.items()
        if _text(_value(task, "parent_group_id")) == group_id
    )
    if not candidates:
        return group_id

    task_id, task = candidates[0]
    visited: set[str] = set()
    while task_id not in visited:
        visited.add(task_id)
        parent_task_id = _text(_value(task, "parent_task_id"))
        if not parent_task_id:
            return _text(_value(task, "parent_group_id"), group_id)
        parent_task = task_queue.get(parent_task_id)
        if parent_task is None:
            return _text(_value(task, "parent_group_id"), group_id)
        task_id = parent_task_id
        task = parent_task
    return group_id


def _synthetic_scan_group(context: ReportContext, identifier: str) -> dict[str, Any]:
    """Create a display-only group for an untriaged authoritative finding."""
    issue = _scan_issue_for_identifier(context, identifier)
    issue_data: dict[str, Any] = {}
    if issue is not None:
        for field_name in (
            "cve_id",
            "ghsa_id",
            "finding_id",
            "rule_id",
            "source",
            "severity",
            "file_path",
            "package_name",
        ):
            value = _value(issue, field_name)
            if value is not None:
                issue_data[field_name] = value
    normalized_identifier = identifier.strip()
    if normalized_identifier.casefold().startswith("cve-"):
        for field_name in ("ghsa_id", "finding_id", "rule_id"):
            issue_data.pop(field_name, None)
        issue_data.setdefault("cve_id", normalized_identifier.upper())
    elif normalized_identifier.casefold().startswith("ghsa-"):
        for field_name in ("cve_id", "finding_id", "rule_id"):
            issue_data.pop(field_name, None)
        issue_data.setdefault("ghsa_id", normalized_identifier.upper())
    else:
        for field_name in ("cve_id", "ghsa_id", "rule_id"):
            issue_data.pop(field_name, None)
        issue_data.setdefault("finding_id", normalized_identifier)
    package = _text(issue_data.get("package_name")).strip() or "Untriaged finding"
    return {
        "group_id": normalized_identifier,
        "vulnerable_component": package,
        "issue_type": "sca",
        "sources": [_text(issue_data.get("source"), "final_full_scan")],
        "file_path": _text(issue_data.get("file_path"), "authoritative final scan"),
        "issues": [issue_data],
    }


def _follow_up_groups(context: ReportContext) -> list[tuple[str, Any]]:
    """Return every open finding, including untriaged final-scan findings."""
    discovered_ids = set(_discovered_group_ids(context))
    initial_ids = {_text(_value(group, "group_id")) for group in context.initial_valid_groups}
    selected: dict[str, tuple[str, Any]] = {}
    for group in _report_groups(context):
        group_id = _text(_value(group, "group_id"))
        canonical_id = (
            group_id if group_id in discovered_ids else _lineage_root_group_id(context, group_id)
        )
        current = selected.get(canonical_id)
        if current is None:
            selected[canonical_id] = (canonical_id, group)
            continue
        current_group_id = _text(_value(current[1], "group_id"))
        if group_id in initial_ids and current_group_id not in initial_ids:
            selected[canonical_id] = (canonical_id, group)

    scan_state = _scan_evidence_state(
        context.final_full_scan_result, context.new_vulnerability_status
    )
    if scan_state == "complete":
        for group_id in discovered_ids:
            selected.setdefault(group_id, (group_id, {"group_id": group_id}))

        represented_identifiers = (
            set().union(*(_group_identifiers(group) for _, group in selected.values()))
            if selected
            else set()
        )
        for identifier in context.new_vulnerability_identifiers:
            if identifier.casefold() not in represented_identifiers:
                selected.setdefault(
                    identifier,
                    (identifier, _synthetic_scan_group(context, identifier)),
                )

    # A malformed or partially projected state can contain a status without a
    # corresponding group object. Keep that open work visible instead of
    # silently dropping it from the user report.
    selected_ids = set(selected)
    for group_id, status in context.group_statuses.items():
        lineage_id = _lineage_root_group_id(context, group_id)
        if (
            status in _OUTSTANDING_GROUP_STATUSES
            and group_id not in selected_ids
            and lineage_id not in selected_ids
        ):
            selected[group_id] = (group_id, {"group_id": group_id})
    return sorted(selected.values(), key=lambda item: item[0])


def _discovered_group_ids(context: ReportContext) -> list[str]:
    """Return groups added, reappeared, or reopened by the authoritative scan."""
    return _reconciliation_ids(
        context.triage_reconciliation,
        "added",
        "new_group_ids",
        "reappeared",
        "reappeared_group_ids",
        "final_scan_reopened_group_ids",
    )


def _task_ids_for_group(context: ReportContext, group_id: str) -> list[str]:
    """Return task-lineage IDs associated with a vulnerability group."""
    return [
        task_id
        for task in _group_tree(context.task_queue, group_id)
        if (task_id := _text(_value(task, "task_id")))
    ]


def _group_package_names(group: Any) -> set[str]:
    """Return package names that can identify a group in a manifest diff."""
    names: set[str] = set()
    issue = _group_issue(group)
    for value in (
        _value(group, "vulnerable_component"),
        _value(group, "package_name"),
        _value(group, "parent_package_name"),
        _value(issue, "package_name"),
    ):
        value_text = _text(value).strip()
        if not value_text:
            continue
        names.add(value_text)
        names.update(part.strip() for part in re.split(r"[|,]", value_text) if part.strip())
    for parent_context in _items(_value(group, "parent_contexts")):
        parent_name = _text(_value(parent_context, "package_name")).strip()
        if parent_name:
            names.add(parent_name)
    return names


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


def _package_change_files(change: PackageChange) -> list[str]:
    """Return displayable file paths recorded for one package change."""
    files: list[str] = []
    for value in _text(change.file).split(";"):
        path = value.strip()
        if not path or "synchronized" in path.casefold():
            continue
        if path not in files:
            files.append(path)
    return files


def _package_name_from_resolved_url(value: Any) -> str:
    """Extract an npm package name from a registry tarball URL."""
    match = re.search(
        r"(?:https?://[^/]+/)(?P<package>(?:@[^/]+/)?[^/]+)/-/",
        _text(value),
        re.IGNORECASE,
    )
    return match.group("package") if match else ""


def _dependency_mechanism(
    context: ReportContext,
    group_id: str,
    change: PackageChange | None = None,
) -> str:
    """Resolve the manifest mechanism used for a package change."""
    section = _text(_value(change, "section")).strip().casefold()
    if section:
        if "override" in section:
            return "overrides"
        if "resolution" in section:
            return "resolutions"
        if section == "lockfile":
            return "lockfile"
        return section

    for task_id in _task_ids_for_group(context, group_id):
        task = context.task_queue.get(task_id)
        diagnostic = context.retry_diagnostics.get(task_id)
        for value in (
            _value(_value(task, "target_dependency_type"), "value"),
            _value(task, "target_dependency_type"),
            _value(diagnostic, "target_dependency_type"),
        ):
            value_text = _text(value).strip().casefold()
            if not value_text:
                continue
            if "override" in value_text:
                return "overrides"
            if "resolution" in value_text:
                return "resolutions"
            return value_text
        if _text(_value(task, "strategy_stage")).casefold() == "package_override":
            return "overrides"
        if bool(_value(diagnostic, "used_overrides", False)):
            return "overrides"

    if change is not None and change.scope == "transitive":
        return "lockfile"
    return "dependencies"


def _package_change_matches_group(change: PackageChange, group: Any) -> bool:
    """Return whether a package change identifies the supplied finding group."""
    names = _group_package_names(group)
    return change.name in names or change.name.casefold() in {name.casefold() for name in names}


def _package_change_detail(change: PackageChange, mechanism: str) -> str:
    """Describe the exact manifest entry changed by a package remediation."""
    previous = change.old or "not present"
    current = change.new or "removed"
    return f"{change.name}: {previous} → {current} via {mechanism}"


def _package_attempt_text(change: PackageChange, mechanism: str) -> str:
    """Render a compact package transition for an attempted remediation."""
    previous = change.old or "not present"
    current = change.new or "removed"
    return f"Updated {change.name} {previous} → {current} via {mechanism}."


def _required_follow_up_action(
    context: ReportContext,
    group: Any,
    status: str,
    *,
    group_id: str | None = None,
) -> str:
    """Describe the next user-facing action for an outstanding group."""
    package = _group_package_or_target(context, group)
    target = f" for {package}" if package and package != "Unspecified target" else ""
    if status == "needs_retry":
        return f"Retry the remediation{target}, then rerun validation and the security scan."
    elif status == "unfixable":
        return f"Apply an alternative remediation{target}, then rerun validation and the security scan."
    elif status == "inconclusive":
        return f"Review the attempted change{target} and complete validation, then rerun the security scan."
    elif status in {"optimistically_fixed", "awaiting_validation"}:
        return f"Complete validation for the attempted remediation{target}."
    elif status == "pivoted":
        return f"Apply an alternative remediation{target}, then rerun validation and the security scan."
    else:
        return f"Complete the planned remediation{target}, then rerun validation and the security scan."


_ATTEMPT_DETAIL_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:what changed|changes?|validation|validated|"
    r"verification|tests?|notes?|note|evidence|diagnostics?|retry details?)\s*:?.*$",
    re.IGNORECASE,
)
_ATTEMPT_DETAIL_INLINE_RE = re.compile(
    r"\s+(?:what changed|changes?|validation|verification|tests?|"
    r"notes?|note|evidence|diagnostics?|retry details?)\s*:",
    re.IGNORECASE,
)
_FINAL_NOTE_RE = re.compile(r"\s+(?:final note|final conclusion)\s*:\s*", re.IGNORECASE)
_CHANGED_FILES_RE = re.compile(r"\bchanged files?\s*:\s*", re.IGNORECASE)
_NO_CHANGED_FILES_RE = re.compile(
    r"\bno files?(?: were)? changed\b|\bno file changes\b", re.IGNORECASE
)
_NO_CODE_CHANGE_RE = re.compile(
    r"\b(?:no (?:validated )?(?:code )?change|without applying a change|"
    r"nothing (?:was )?changed)\b",
    re.IGNORECASE,
)
_ATTEMPT_SECTION_LABELS = (
    "specific code changes",
    "what changed",
    "code changes",
    "implementation",
    "changes",
    "current modified source",
    "modified source",
    "file changes",
    "files changed",
    "changed files",
    "what i found",
    "final conclusion",
    "final note",
    "final outcome",
    "outcome",
    "validation status",
    "validation",
    "verification",
    "test status",
    "tests",
    "notes",
    "note",
)
_ATTEMPT_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?P<label>"
    + "|".join(re.escape(label) for label in _ATTEMPT_SECTION_LABELS)
    + r")\s*(?::\s*(?P<value>.*))?$",
    re.IGNORECASE,
)
_INLINE_ATTEMPT_SECTION_RE = re.compile(
    r"\s+(?P<label>final note|final conclusion|final outcome|outcome)\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_CHANGE_LANGUAGE_RE = re.compile(
    r"\b(?:add(?:ed|s)?|chang(?:e|ed|es|ing)|implement(?:ed|s|ing)?|"
    r"modif(?:y|ied|ies|ying)|patch(?:ed|es|ing)?|remov(?:e|ed|es|ing)|"
    r"replac(?:e|ed|es|ing)|updat(?:e|ed|es|ing)|workaround|guard|annotation)\b",
    re.IGNORECASE,
)


def _attempt_sections(value: Any) -> dict[str, list[str]]:
    """Split an action summary into lead, change, outcome, and validation sections."""
    sections: dict[str, list[str]] = {"lead": []}
    current = "lead"
    raw = _text(value).strip()
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        inline_match = _INLINE_ATTEMPT_SECTION_RE.search(stripped)
        if inline_match:
            prefix = stripped[: inline_match.start()].strip()
            if prefix:
                sections.setdefault(current, []).append(prefix)
            current = inline_match.group("label").casefold()
            sections.setdefault(current, [])
            value_text = inline_match.group("value").strip()
            if value_text:
                sections[current].append(value_text)
            continue

        section_match = _ATTEMPT_SECTION_RE.match(stripped)
        if section_match:
            current = section_match.group("label").casefold()
            sections.setdefault(current, [])
            value_text = (section_match.group("value") or "").strip()
            if value_text:
                sections[current].append(value_text)
            continue

        sections.setdefault(current, []).append(stripped)
    return sections


def _clean_detail_text(lines: Sequence[Any]) -> str:
    """Normalize summary-detail lines for one Markdown table cell."""
    cleaned: list[str] = []
    for line in lines:
        text = _text(line).strip()
        if not text or text in {"```", "~~~"}:
            continue
        text = re.sub(r"^[-*]\s+", "", text)
        text = re.sub(r"^\d+[.)]\s+", "", text)
        if text:
            cleaned.append(text)
    return " ".join(cleaned)


def _summarize_outcome_text(value: Any) -> str:
    """Keep natural outcome prose while removing embedded code blocks and markup."""
    text = _clean_detail_text([value])
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2])
    return text


def _section_text(sections: Mapping[str, Sequence[Any]], *labels: str) -> str:
    """Return normalized text from the first populated named summary sections."""
    for label in labels:
        text = _clean_detail_text(sections.get(label.casefold(), []))
        if text:
            return text
    return ""


def _inline_code_text(value: Any, limit: int = 420) -> str:
    """Compact a code fragment without losing its leading change semantics."""
    text = re.sub(r"\s+", " ", _text(value).strip())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 13)].rstrip() + "... (truncated)"


def _attempt_lead_action(value: Any) -> str:
    """Return the lead action sentence without changed-file or note metadata."""
    sections = _attempt_sections(value)
    action = _clean_detail_text(sections.get("lead", []))
    final_note = _FINAL_NOTE_RE.search(action)
    if final_note is not None:
        action = action[: final_note.start()].strip(" ;:-")
    changed_clause = _CHANGED_FILES_RE.search(action)
    if changed_clause is not None:
        action = action[: changed_clause.start()].strip(" ;:-.")
    return action.rstrip(" ;:.")


def _attempt_final_note(value: Any) -> str:
    """Return the concise final-note or outcome prose from an action summary."""
    sections = _attempt_sections(value)
    return _section_text(
        sections,
        "final note",
        "final conclusion",
        "final outcome",
        "outcome",
    )


def _attempt_primary_text(value: Any) -> str:
    """Keep the first remediation paragraph and discard agent detail sections."""
    raw = _text(value).strip()
    if not raw:
        return ""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if _ATTEMPT_DETAIL_LINE_RE.match(stripped):
            break
        lines.append(stripped)
    primary = " ".join(lines)
    final_note = _FINAL_NOTE_RE.search(primary)
    if final_note is not None:
        note = _ATTEMPT_DETAIL_INLINE_RE.split(primary[final_note.end() :], maxsplit=1)[0].strip()
        prefix = primary[: final_note.start()].strip()
        if note:
            return f"{prefix} Final note: {note}".strip()
        return prefix
    return _ATTEMPT_DETAIL_INLINE_RE.split(primary, maxsplit=1)[0].strip()


def _attempt_changed_files(value: Any) -> tuple[list[str], bool]:
    """Extract changed-file evidence from an attempt summary."""
    text = _attempt_primary_text(value)
    if _NO_CHANGED_FILES_RE.search(text):
        return [], True
    match = _CHANGED_FILES_RE.search(text)
    if match is not None:
        file_text = text[match.end() :]
        file_text = _FINAL_NOTE_RE.split(file_text, maxsplit=1)[0]
        file_text = file_text.strip().rstrip(".;")
        if file_text.casefold() in {"", "none", "no files", "n/a"}:
            return [], True
        files = [part.strip().strip("`") for part in file_text.split(",")]
        return [part for part in files if part], False

    sections = _attempt_sections(value)
    for label in ("file changes", "files changed", "changed files", "current modified source"):
        lines = sections.get(label, [])
        if not lines:
            continue
        section_text = _clean_detail_text(lines)
        if section_text.casefold() in {
            "",
            "none",
            "no files",
            "n/a",
        } or _NO_CHANGED_FILES_RE.search(section_text):
            return [], True
        files = [
            part.strip().strip("`")
            for line in lines
            for part in re.split(r"[,;]", _text(line))
            if part.strip()
        ]
        if files:
            return files, False
    return [], False


def _attempt_replay_change_details(metadata: Any) -> tuple[list[str], list[str]]:
    """Extract exact source replacements and affected files from a replay plan."""
    replay_plan = _value(metadata, "replay_plan")
    if replay_plan is None:
        return [], []

    details: list[str] = []
    files = [_text(path) for path in _items(_value(replay_plan, "validated_files"))]

    edit_sets = _items(_value(replay_plan, "successful_edit_sets"))
    for edit_set in edit_sets:
        files.extend(_text(path) for path in _items(_value(edit_set, "affected_files")))
        for replacement in _items(_value(edit_set, "replacements")):
            path = _text(_value(replacement, "file_path"))
            if path:
                files.append(path)
            old_text = _inline_code_text(_value(replacement, "old_text"))
            new_text = _inline_code_text(_value(replacement, "new_text"))
            if not path:
                continue
            if old_text or new_text:
                details.append(
                    f"{path}: replaced {old_text or 'nothing'} with {new_text or 'nothing'}"
                )

    return details, _unique_texts(files)


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


def _unified_diff_blocks(diff: str, files: Sequence[Any] = ()) -> list[str]:
    """Return complete unified-diff file blocks for selected source files.

    The report uses these blocks for code workarounds so a reviewer can inspect
    the exact source edit. Package manifests are excluded because their
    compact version transition is already rendered as prose.
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

    selected = [_normalized_path(path) for path in files if _normalized_path(path)]
    result: list[str] = []
    for path, lines in blocks:
        if _PACKAGE_FILE_RE.search(path):
            continue
        if selected and not _path_matches_any(path, selected):
            continue
        result.append("\n".join(lines).strip())
    return result


def _replay_diff_blocks(metadata: Any) -> list[str]:
    """Build unified-diff blocks from committed workaround replacements."""
    replay_plan = _value(metadata, "replay_plan")
    if replay_plan is None:
        return []

    blocks: list[str] = []
    for edit_set in _items(_value(replay_plan, "successful_edit_sets")):
        for replacement in _items(_value(edit_set, "replacements")):
            path = _text(_value(replacement, "file_path")).strip()
            if not path or _PACKAGE_FILE_RE.search(path):
                continue
            old_text = _text(_value(replacement, "old_text"))
            new_text = _text(_value(replacement, "new_text"))
            old_lines = old_text.splitlines() or [""]
            new_lines = new_text.splitlines() or [""]
            block = [f"--- a/{path}", f"+++ b/{path}", "@@"]
            block.extend(f"-{line}" for line in old_lines)
            block.extend(f"+{line}" for line in new_lines)
            blocks.append("\n".join(block))
    return blocks


def _attempt_diff_blocks(
    context: ReportContext,
    summary: Any,
    metadata: Any,
    files: Sequence[Any],
) -> list[str]:
    """Return exact source diff blocks for one workaround attempt."""
    source_files = [path for path in files if not _PACKAGE_FILE_RE.search(_text(path))]
    blocks = _unified_diff_blocks(context.diff, source_files)
    if blocks:
        return blocks
    return _replay_diff_blocks(metadata)


def _attempt_package_metadata(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    attempt_id: str,
) -> tuple[str, str, str, str, str] | None:
    """Recover a package/version operation from committed attempt metadata."""
    task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
    task = context.task_queue.get(task_id)
    snapshot = context.attempt_snapshots.get(attempt_id)
    diagnostic = context.retry_diagnostics.get(task_id)
    candidates = [snapshot, metadata, task, diagnostic]

    package = ""
    for item in candidates:
        package = _text(_value(item, "target_package_name")).strip()
        if package:
            break
    if not package:
        return None

    selected_version = ""
    for item in candidates:
        selected_version = _text(_value(item, "selected_version")).strip()
        if selected_version:
            break
    if not selected_version:
        for item in candidates:
            executed = [_text(version) for version in _items(_value(item, "executed_versions"))]
            if executed:
                selected_version = executed[-1]
                break
    if not selected_version:
        return None

    previous_version = ""
    for item in (task, diagnostic):
        previous_version = _text(_value(item, "parent_package_version")).strip()
        if previous_version:
            break
    group = next(
        (
            item
            for item in [*context.initial_valid_groups, *context.final_valid_groups]
            if _text(_value(item, "group_id")) == group_id
        ),
        None,
    )
    if group is not None and not previous_version:
        package_names = _group_package_names(group)
        for issue in _items(_value(group, "issues")):
            issue_package = _text(_value(issue, "package_name")).strip()
            issue_version = _text(_value(issue, "package_version")).strip()
            if issue_version and (not issue_package or issue_package in package_names):
                previous_version = issue_version
                break

    dependency_type = ""
    for item in candidates:
        dependency_type = _text(_value(item, "target_dependency_type")).strip()
        if dependency_type:
            break
    synthetic_change = PackageChange(
        package,
        previous_version,
        selected_version,
        "",
        "direct",
        dependency_type,
    )
    mechanism = _dependency_mechanism(context, group_id, synthetic_change)
    file_path = ""
    for item in candidates:
        instruction = _text(_value(item, "instruction"))
        file_match = _PACKAGE_FILE_RE.search(instruction)
        if file_match:
            file_path = file_match.group(0)
            break
    if not file_path and group is not None:
        for path in _items(_value(group, "file_paths")) + [_value(group, "file_path")]:
            if _PACKAGE_FILE_RE.search(_text(path)):
                file_path = _text(path)
                break
    return package, previous_version, selected_version, mechanism, file_path


def _attempt_kind(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
) -> str:
    """Classify an attempt as a version update or source-code workaround."""
    task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
    task = context.task_queue.get(task_id)
    snapshot = context.attempt_snapshots.get(
        _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
    )
    explicit_strategy = False
    for item in (metadata, snapshot, task):
        for field_name in ("strategy", "dispatch_node", "strategy_stage", "no_fix_stage"):
            value = _text(_value(item, field_name)).strip().casefold()
            if "workaround" in value or "code" in value or "no_fix" in value:
                return "Code Workaround"
            if "version" in value or "update" in value:
                explicit_strategy = True
    replay_plan = _value(metadata, "replay_plan")
    if replay_plan is not None and (
        _items(_value(replay_plan, "successful_edit_sets"))
        or _items(_value(replay_plan, "validated_files"))
    ):
        return "Code Workaround"

    if explicit_strategy:
        return "Version Update"
    summary_text = _text(_value(summary, "summary"))
    if re.search(r"\b(?:code\s+workaround|source\s+edit|source\s+change)\b", summary_text, re.I):
        return "Code Workaround"
    return "Version Update"


def _attempt_result_status(context: ReportContext, summary: Any, metadata: Any) -> str:
    """Return a reader-facing outcome label for one recorded attempt."""
    attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
    qa_result = context.qa_results.get(attempt_id)
    evaluation = _value(qa_result, "evaluation")
    if evaluation is not None:
        if bool(_value(evaluation, "passed", False)):
            return "Succeeded"
        return "Failed"

    status = (
        (_text(_value(metadata, "status")) or _text(_value(summary, "status")) or "pending")
        .strip()
        .casefold()
    )
    if status in {"success", "mitigated", "qa_passed"}:
        return "Succeeded"
    if status in {"pending", "optimistically_fixed", "awaiting_validation"}:
        return "Pending"
    return "Failed"


def _sanitize_report_outcome(value: Any) -> str:
    """Remove internal prefixes and lifecycle codes from attempt outcome prose."""
    text = _summarize_outcome_text(value)
    if not text:
        return ""
    text = re.sub(r"\bODC\s+(?:FAILURE|ERROR)\b\s*:?\s*", "security scan issue: ", text, flags=re.I)
    text = re.sub(r"\b(?:supervisor|planner|subagent|worker)\s*:\s*", "", text, flags=re.I)
    text = re.sub(
        r"^(?:worker|agent)(?:\s+(?:summary|result|outcome|report))?\s*[-:]*\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:qa_passed|unfixable|needs_retry|optimistically_fixed|awaiting_validation|pivoted|inconclusive)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:stack trace|traceback|crash diagnostics?)\b", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ;:")
    if text:
        text = text[0].upper() + text[1:]
    return _inline_code_text(text, 260)


def _attempt_package_changes(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    *,
    attempt_id: str,
    files: Sequence[Any],
    explicit_no_files: bool,
) -> list[tuple[PackageChange, str]]:
    """Collect package transitions evidenced by one attempt."""
    if explicit_no_files:
        return []
    group = next(
        (
            item
            for item in [*context.initial_valid_groups, *context.final_valid_groups]
            if _text(_value(item, "group_id")) == group_id
        ),
        {},
    )
    changes: list[tuple[PackageChange, str]] = []
    for change in _package_changes(context.diff):
        if not _package_change_matches_group(change, group):
            continue
        change_files = _package_change_files(change)
        if files and not any(_path_matches_any(path, files) for path in change_files):
            continue
        changes.append((change, _dependency_mechanism(context, group_id, change)))
    if changes:
        return changes

    metadata_change = _attempt_package_metadata(
        context,
        group_id,
        summary,
        metadata,
        attempt_id,
    )
    if metadata_change is None:
        return []
    package, previous, selected, mechanism, file_path = metadata_change
    if files and file_path and not any(_path_matches_any(file_path, files) for _ in files):
        return []
    return [
        (
            PackageChange(package, previous, selected, file_path, "direct", mechanism),
            mechanism,
        )
    ]


def _attempt_code_details(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    *,
    attempt_id: str = "",
    files: Sequence[Any] = (),
    include_package_diff: bool = True,
    include_source_diff: bool = True,
    include_metadata_fallback: bool = True,
) -> list[str]:
    """Build specific code-change descriptions for one remediation attempt."""
    summary_text = _text(_value(summary, "summary")) if summary is not None else ""
    status = _text(_value(summary, "status")) or _text(_value(metadata, "status"), "recorded")
    details: list[str] = []
    group = next(
        (
            item
            for item in [*context.initial_valid_groups, *context.final_valid_groups]
            if _text(_value(item, "group_id")) == group_id
        ),
        {},
    )

    if include_package_diff:
        for change in _package_changes(context.diff):
            if not _package_change_matches_group(change, group):
                continue
            if files and not any(
                _path_matches_any(path, files) for path in _package_change_files(change)
            ):
                continue
            mechanism = _dependency_mechanism(context, group_id, change)
            details.append(_package_change_detail(change, mechanism))

    replay_details, _ = _attempt_replay_change_details(metadata)
    details.extend(replay_details)

    for field_name in ("specific_code_changes", "code_changes"):
        structured_changes = _value(summary, field_name)
        if isinstance(structured_changes, (str, bytes)):
            structured_text = _text(structured_changes).strip()
        else:
            structured_text = _clean_detail_text(_items(structured_changes))
        if structured_text:
            details.append(structured_text)

    sections = _attempt_sections(summary_text)
    for label in (
        "specific code changes",
        "what changed",
        "code changes",
        "implementation",
        "changes",
    ):
        text = _section_text(sections, label)
        if text:
            details.append(text)
    if not any(details):
        found_lines = [
            line
            for line in sections.get("what i found", [])
            if _CHANGE_LANGUAGE_RE.search(_text(line))
        ]
        if found_lines:
            details.append(_clean_detail_text(found_lines))

    if not details:
        final_note = _attempt_final_note(summary_text)
        if (
            final_note
            and _CHANGE_LANGUAGE_RE.search(final_note)
            and not _NO_CODE_CHANGE_RE.search(final_note)
        ):
            details.append(final_note)

    if not details and status.casefold() not in {"surrender", "unfixable", "failed"}:
        lead_action = _attempt_lead_action(summary_text)
        if lead_action and _CHANGE_LANGUAGE_RE.search(lead_action):
            details.append(lead_action)

    if include_source_diff and files:
        details.extend(_diff_code_change_details(context.diff, files))

    metadata_change = _attempt_package_metadata(
        context,
        group_id,
        summary,
        metadata,
        attempt_id,
    )
    if not details and include_metadata_fallback and metadata_change is not None:
        package, previous, selected, mechanism, _ = metadata_change
        details.append(f"{package}: {previous or 'unknown'} → {selected} via {mechanism}")

    unique: list[str] = []
    seen: set[str] = set()
    for detail in details:
        normalized = re.sub(r"\s+", " ", _text(detail).strip())
        if normalized and normalized.casefold() not in seen:
            unique.append(normalized)
            seen.add(normalized.casefold())
    if unique:
        return unique
    if status.casefold() in {"surrender", "unfixable", "failed"}:
        return ["No validated code change was applied"]
    return ["No specific code change recorded"]


def _attempt_files(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    *,
    attempt_id: str = "",
) -> tuple[list[str], bool]:
    """Collect file evidence for one attempt and report explicit no-file claims."""
    summary_text = _text(_value(summary, "summary")) if summary is not None else ""
    parsed_files, explicit_no_files = _attempt_changed_files(summary_text)
    values: list[Any] = [
        *_items(_value(summary, "changed_files")),
        *_items(_value(summary, "file_changes")),
        *_items(_value(summary, "files_changed")),
        *_items(_value(metadata, "changed_files")),
        *_items(_value(metadata, "file_changes")),
        *_items(_value(metadata, "files_changed")),
    ]
    diagnostics = _value(metadata, "execution_diagnostics")
    values.extend(_items(_value(diagnostics, "validated_files")))
    _, replay_files = _attempt_replay_change_details(metadata)
    values.extend(replay_files)
    values.extend(parsed_files)
    files = _unique_texts(values)
    if not files and not explicit_no_files:
        metadata_change = _attempt_package_metadata(
            context,
            group_id,
            summary,
            metadata,
            attempt_id,
        )
        if metadata_change is not None and metadata_change[4]:
            files.append(metadata_change[4])
    return _unique_texts(files), explicit_no_files


def _attempt_outcome(
    summary: Any,
    metadata: Any,
    *,
    status: str,
) -> str:
    """Summarize the final outcome of one attempt without exposing retry diagnostics."""
    summary_text = _text(_value(summary, "summary")) if summary is not None else ""
    sections = _attempt_sections(summary_text)
    outcome = (
        _text(_value(summary, "final_outcome")).strip() or _text(_value(summary, "outcome")).strip()
    )
    outcome = outcome or _attempt_final_note(summary_text) or _section_text(sections, "lead")
    outcome = _summarize_outcome_text(outcome)
    if not outcome:
        if status.casefold() in {"success", "mitigated"}:
            outcome = "Attempt completed successfully"
        else:
            outcome = "Attempt ended without a validated remediation"

    verification = _section_text(
        sections,
        "validation status",
        "validation",
        "verification",
        "test status",
        "tests",
    )
    diagnostics = _value(metadata, "execution_diagnostics")
    validation_passed = bool(_value(diagnostics, "validation_passed", False))
    if verification and not re.search(
        r"\b(?:pass(?:ed)?|fail(?:ed)?|success(?:ful)?|unable|could not|couldn't|not claim)\b",
        outcome,
        re.IGNORECASE,
    ):
        outcome = (
            f"{outcome.rstrip('.; ')}. Verification result: {_summarize_outcome_text(verification)}"
        )
    if status.casefold() in {"success", "mitigated"}:
        if validation_passed and not re.search(
            r"\b(?:pass(?:ed)?|success(?:ful)?|validat(?:e|ed|ion))\b", outcome, re.IGNORECASE
        ):
            outcome = (
                f"{outcome.rstrip('.; ')}. Attempt completed successfully with validation passed"
            )
    elif not re.search(
        r"\b(?:without|stopp(?:ed|ing)|fail(?:ed|ure)|could not|couldn't|unfixable|not claim)\b",
        outcome,
        re.IGNORECASE,
    ):
        outcome = f"{outcome.rstrip('.; ')}. Attempt ended without a validated remediation"
    cleaned = _sanitize_report_outcome(outcome.rstrip(" ;.") + ".")
    return cleaned or "Attempt ended without a validated remediation."


def _attempt_change_summary(summary: Any, details: Sequence[Any]) -> str:
    """Return short natural prose describing the change in a successful attempt."""
    summary_text = _text(_value(summary, "summary")) if summary is not None else ""
    sections = _attempt_sections(summary_text)
    candidate = (
        _section_text(
            sections,
            "specific code changes",
            "what changed",
            "code changes",
            "implementation",
            "changes",
        )
        or _attempt_final_note(summary_text)
        or _attempt_lead_action(summary_text)
    )
    if candidate:
        candidate = _summarize_outcome_text(candidate)
        return candidate.rstrip(" ;.") + "."
    for detail in details:
        text = _text(detail).strip()
        if text:
            text = _summarize_outcome_text(text)
            return text.rstrip(" ;.") + "."
    return "Validated remediation recorded."


def _attempt_records_for_group(
    context: ReportContext,
    group_id: str,
) -> list[tuple[Any, Any]]:
    """Return deduplicated worker/action evidence for a group in stable order."""
    task_ids = set(_task_ids_for_group(context, group_id))
    worker_results = sorted(
        context.worker_results.values(),
        key=lambda item: _text(_value(item, "attempt_id")),
    )
    worker_by_attempt = {
        _text(_value(result, "attempt_id")): result
        for result in worker_results
        if _text(_value(result, "attempt_id"))
    }
    records: list[tuple[Any, Any]] = []
    seen_records: set[tuple[str, str, str]] = set()

    def metadata_with_replay_plan(summary: Any, metadata: Any) -> Any:
        """Attach the task-keyed replay plan when an attempt lacks one."""
        if metadata is not None and _value(metadata, "replay_plan") is not None:
            return metadata
        task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
        replay_plan = context.workaround_replay_plans.get(task_id)
        if replay_plan is None:
            return metadata
        if metadata is None:
            return {"task_id": task_id, "replay_plan": replay_plan}
        if isinstance(metadata, Mapping):
            enriched = dict(metadata)
        elif hasattr(metadata, "model_dump"):
            enriched = metadata.model_dump()
        else:
            enriched = {
                field_name: getattr(metadata, field_name)
                for field_name in dir(metadata)
                if not field_name.startswith("_")
                and not callable(getattr(metadata, field_name, None))
            }
        enriched["replay_plan"] = replay_plan
        return enriched

    def add(summary: Any, metadata: Any = None) -> None:
        metadata = metadata_with_replay_plan(summary, metadata)
        task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
        if task_id not in task_ids:
            return
        attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
        status = _text(_value(summary, "status")) or _text(_value(metadata, "status"), "recorded")
        record_key = (
            attempt_id,
            task_id,
            status.casefold() if attempt_id else _text(_value(summary, "summary")),
        )
        if record_key in seen_records:
            return
        seen_records.add(record_key)
        records.append((summary, metadata))

    for summary in context.action_summaries:
        attempt_id = _text(_value(summary, "attempt_id"))
        add(summary, worker_by_attempt.get(attempt_id))
    for result in worker_results:
        add(_value(result, "action_summary"), result)
    return records


def _attempt_summary_key(summary: Any, metadata: Any) -> str:
    """Return the stable key used to attach an optional summary to an attempt."""
    attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
    if attempt_id:
        return attempt_id
    task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"), "attempt")
    status = _text(_value(summary, "status")) or _text(_value(metadata, "status"), "recorded")
    summary_text = _inline_code_text(_value(summary, "summary"), 160)
    return f"{task_id}:{status}:{summary_text}"


def _successful_attempts_for_group(
    context: ReportContext,
    group_id: str,
) -> list[tuple[Any, Any]]:
    """Return successful worker/action evidence for a group in stable order."""
    records: list[tuple[Any, Any]] = []
    for summary, metadata in _attempt_records_for_group(context, group_id):
        result_status = _text(_value(metadata, "status")).casefold()
        summary_status = _text(_value(summary, "status")).casefold()
        diagnostics = _value(metadata, "execution_diagnostics")
        if (
            result_status in {"success", "mitigated"}
            or summary_status in {"success", "mitigated"}
            or bool(_value(diagnostics, "validation_passed", False))
        ):
            records.append((summary, metadata))
    return records


def _attempted_fixes_for_group(context: ReportContext, group_id: str) -> str:
    """Render concise, numbered attempt history for one follow-up finding."""
    entries: list[str] = []
    seen_entries: set[tuple[str, str, str]] = set()
    for number, (summary, metadata) in enumerate(
        _attempt_records_for_group(context, group_id),
        start=1,
    ):
        attempt_id = _attempt_summary_key(summary, metadata)
        kind = _attempt_kind(context, group_id, summary, metadata)
        outcome_status = _attempt_result_status(context, summary, metadata)
        files, explicit_no_files = _attempt_files(
            context,
            group_id,
            summary,
            metadata,
            attempt_id=attempt_id,
        )
        package_changes = _attempt_package_changes(
            context,
            group_id,
            summary,
            metadata,
            attempt_id=attempt_id,
            files=files,
            explicit_no_files=explicit_no_files,
        )
        change_lines: list[str] = []
        if package_changes:
            for change, mechanism in package_changes:
                change_lines.append(_package_attempt_text(change, mechanism))
        elif kind == "Code Workaround":
            if files:
                change_lines.append(f"Attempted a code workaround in {', '.join(files)}.")
            else:
                change_lines.append("Attempted a code workaround; source files were not recorded.")
        elif explicit_no_files:
            change_lines.append("No validated package change was applied.")
        else:
            change_lines.append("Attempted a package version update.")

        diff_blocks = (
            _attempt_diff_blocks(context, summary, metadata, files)
            if kind == "Code Workaround"
            else []
        )
        raw_attempt_status = (
            _text(_value(metadata, "status")) or _text(_value(summary, "status")) or "pending"
        )
        outcome = _attempt_outcome(summary, metadata, status=raw_attempt_status)
        # The task result can be marked successful even when its QA envelope
        # failed. Keep the report aligned with the evidence available to the
        # reviewer.
        if outcome_status == "Failed" and not re.search(
            r"\b(?:fail|timed out|without a validated)\b", outcome, re.I
        ):
            outcome = "Validation failed."
        key = (kind.casefold(), outcome_status.casefold(), "\n".join(change_lines))
        if key in seen_entries:
            continue
        seen_entries.add(key)

        lines = [f"{number}. **Attempt {number} ({kind} — {outcome_status}):**"]
        lines.extend(f"   - {line}" for line in change_lines)
        for block in diff_blocks:
            lines.extend(["   ```diff", *block.splitlines(), "   ```"])
        lines.append(f"   - Outcome: {outcome}")
        entries.append("\n".join(lines))

    return "\n\n".join(entries) or "No remediation attempt recorded."


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
        prefix = line[:1]
        content = _diff_content(line)
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


def _render_summary(context: ReportContext) -> str:
    """Render the seven-metric, user-facing run summary."""
    follow_up = context.groups_unresolved + context.groups_inconclusive + context.groups_pending
    visible_open_groups = sum(
        context.group_statuses.get(group_id, "pending") in _OUTSTANDING_GROUP_STATUSES
        for group_id, _ in _follow_up_groups(context)
    )
    follow_up = max(follow_up, visible_open_groups)
    fixed = context.groups_fixed
    actionable = max(context.actionable_groups, fixed + follow_up)
    sentence = (
        f"**{fixed} of {actionable}** vulnerability groups were successfully remediated. "
        f"**{follow_up} require follow-up review.**"
    )
    metrics = [
        ("Run ID", context.run_id),
        ("Total findings scanned", context.original_scanner_findings),
        ("Successfully remediated", fixed),
        ("Require follow-up", follow_up),
        ("Run duration", _format_duration(context.duration_seconds)),
        ("Total tokens", _format_token_summary(context)),
        ("Patch status", _format_patch_status(context)),
    ]
    return "\n".join(
        [
            "# Remediation Run Report",
            "",
            "## 1. Summary",
            "",
            sentence,
            "",
            _table(("Metric", "Value"), metrics),
        ]
    )


def _scan_issue_for_identifier(context: ReportContext, identifier: str) -> Any | None:
    """Find the authoritative post-scan issue associated with an identifier."""
    normalized = identifier.casefold()
    for issue in context.post_remediation_scan_issues:
        if normalized in {item.casefold() for item in _issue_identifiers(issue)}:
            return issue
    return None


def _package_transition(change: PackageChange, mechanism: str) -> str:
    """Render only the version transition for a successful package change."""
    previous = change.old or "not present"
    current = change.new or "removed"
    return f"{previous} → {current} via {mechanism}"


def _successful_remediation_evidence(
    context: ReportContext,
    group: Any,
) -> tuple[str, list[str], list[str], list[str]]:
    """Collect table text and source diff evidence for one fixed group.

    Returns:
        A tuple containing the remediation-change text, changed files, compact
        code-workaround summaries, and complete unified-diff blocks.
    """
    group_id = _text(_value(group, "group_id"))
    package_changes = [
        (change, _dependency_mechanism(context, group_id, change))
        for change in _package_changes(context.diff)
        if _package_change_matches_group(change, group)
    ]
    package_text = [_package_transition(change, mechanism) for change, mechanism in package_changes]
    files: list[str] = []
    for change, _ in package_changes:
        files.extend(_package_change_files(change))

    code_summaries: list[str] = []
    diff_blocks: list[str] = []
    for summary, metadata in _successful_attempts_for_group(context, group_id):
        if _attempt_kind(context, group_id, summary, metadata) != "Code Workaround":
            continue
        attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
        attempt_files, _ = _attempt_files(
            context,
            group_id,
            summary,
            metadata,
            attempt_id=attempt_id,
        )
        files.extend(attempt_files)
        details = _attempt_code_details(
            context,
            group_id,
            summary,
            metadata,
            attempt_id=attempt_id,
            files=attempt_files,
            include_package_diff=False,
        )
        detail_summary = _sanitize_report_outcome(_attempt_change_summary(summary, details))
        if detail_summary and detail_summary.casefold() not in {
            "validated remediation recorded.",
            "no specific code change recorded.",
        }:
            code_summaries.append(detail_summary)
        diff_blocks.extend(_attempt_diff_blocks(context, summary, metadata, attempt_files))

    files = _unique_texts(files)
    if not files and len(context.initial_valid_groups) == 1:
        files = list(context.changed_files)
    diff_blocks = _unique_texts(diff_blocks)

    changes: list[str] = []
    if package_text:
        changes.extend(package_text)
    if code_summaries:
        changes.append(f"Code workaround: {'; '.join(_unique_texts(code_summaries))}")
    elif diff_blocks:
        changes.append("Code workaround: source changes applied")
    if not changes:
        changes.append("Validated remediation recorded")
    return "; ".join(_unique_texts(changes)), files, _unique_texts(code_summaries), diff_blocks


def _render_follow_up_actions(context: ReportContext) -> str:
    """Render one action-oriented block for every open finding."""
    status_order = {
        "unfixable": 0,
        "needs_retry": 1,
        "inconclusive": 2,
        "pivoted": 3,
        "pending": 4,
        "optimistically_fixed": 5,
        "awaiting_validation": 5,
    }
    groups = [
        (group_id, group)
        for group_id, group in _follow_up_groups(context)
        if context.group_statuses.get(group_id, "pending") in _OUTSTANDING_GROUP_STATUSES
    ]
    groups.sort(
        key=lambda item: (
            status_order.get(context.group_statuses.get(item[0], "pending"), 99),
            item[0],
        )
    )

    lines = ["## 2. Follow up Actions", ""]
    if not groups:
        lines.append("No follow-up actions are required.")
        return "\n".join(lines)

    for index, (group_id, group) in enumerate(groups):
        issue = _group_issue(group)
        status = context.group_statuses.get(group_id, "pending")
        finding = _finding_identifier(group, issue)
        package = _group_finding_package(group)
        severity = _group_severity(group, issue)
        lines.extend(
            [
                f"### {finding} — {package} ({severity})",
                "",
                f"- **Status:** {_user_friendly_status(status)}",
                f"- **Recommended action:** {_required_follow_up_action(context, group, status, group_id=group_id)}",
                "- **Attempted remediations:**",
            ]
        )
        attempt_text = _attempted_fixes_for_group(context, group_id)
        if attempt_text == "No remediation attempt recorded.":
            lines.append("  No remediation attempt recorded.")
        else:
            # The attempt renderer already owns numbering and indentation. The
            # first item is kept under the labelled list for readable Markdown.
            in_diff = False
            for line in attempt_text.splitlines():
                stripped = line.strip()
                if stripped == "```diff":
                    lines.append("   ```diff")
                    in_diff = True
                elif in_diff and stripped == "```":
                    lines.append("   ```")
                    in_diff = False
                elif in_diff:
                    lines.append(line)
                else:
                    lines.append(f"  {line}")
        if index != len(groups) - 1:
            lines.append("")
    return "\n".join(lines)


def _render_successful_remediations(context: ReportContext) -> str:
    """Render one consolidated table of successful package and code fixes."""
    groups = [
        group
        for group in _report_groups(context)
        if context.group_statuses.get(_text(_value(group, "group_id")), "pending")
        in {"qa_passed", "mitigated"}
    ]
    lines = ["## 3. Successful Remediations", ""]
    if not groups:
        lines.append("No successful remediations were produced during this run.")
        return "\n".join(lines)

    rows: list[tuple[str, ...]] = []
    code_evidence: list[tuple[str, str, str, str, list[str], list[str]]] = []
    for group in groups:
        issue = _group_issue(group)
        change_text, files, code_summaries, diff_blocks = _successful_remediation_evidence(
            context,
            group,
        )
        rows.append(
            (
                _finding_identifier(group, issue),
                _group_finding_package(group),
                _group_severity(group, issue),
                change_text,
                ", ".join(files) or "Not recorded",
            )
        )
        if code_summaries or diff_blocks:
            code_evidence.append(
                (
                    _finding_identifier(group, issue),
                    _group_finding_package(group),
                    _group_severity(group, issue),
                    "; ".join(code_summaries) or "Source changes applied.",
                    files,
                    diff_blocks,
                )
            )

    lines.append(
        _table(
            ("Finding", "Package / Target", "Severity", "Remediation Change", "Files Changed"),
            rows,
        )
    )
    if code_evidence:
        lines.extend(["", "### Code workaround details", ""])
        for finding, package, severity, summary, files, diff_blocks in code_evidence:
            lines.extend(
                [
                    f"#### {finding} — {package} ({severity})",
                    "",
                    f"- **Summary:** {summary}",
                    f"- **Files changed:** {', '.join(files) or 'Not recorded'}",
                ]
            )
            for block in diff_blocks:
                lines.extend(["", "```diff", block, "```"])
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
    return "\n".join(lines)


def _render_references(context: ReportContext) -> str:
    """Render the three artifact references useful to an end user."""
    return "\n".join(
        [
            "## 4. References",
            "",
            _table(
                ("Artifact", "Reference"),
                [
                    ("Trajectory", context.trajectory_path or "Not available"),
                    ("LangSmith trace", context.langsmith_trace_url or "Not available"),
                    ("Patch", "Included in run result" if context.has_patch else "No unified diff"),
                ],
            ),
        ]
    )


def generate_report(
    state: Mapping[str, Any],
    *,
    trajectory_path: str | None = None,
    trace_url: str | None = None,
    token_summary: Mapping[str, Any] | None = None,
    run_ended_at: datetime | None = None,
) -> str:
    """Render a deterministic Markdown report from graph state.

    Args:
        state: Graph state or a compatible mapping of remediation evidence.
        trajectory_path: Optional exported trajectory reference.
        trace_url: Optional LangSmith trace reference.
        token_summary: Optional prompt/completion token totals.
        run_ended_at: Optional final timestamp; omitted values remain pending.

    Returns:
        Markdown text. This function does not write files or call external
        services.
    """
    context = _build_context(
        state,
        trajectory_path=trajectory_path,
        trace_url=trace_url,
        token_summary=token_summary,
        run_ended_at=run_ended_at,
    )
    sections = [
        _render_follow_up_actions(context),
        _render_successful_remediations(context),
        _render_references(context),
    ]
    return _render_summary(context) + "\n\n" + "\n\n".join(sections) + "\n"


def _resolve_report_dir(settings: AppSettings | None = None) -> Path:
    """Resolve the canonical report directory from validated settings."""
    return resolve_report_dir(settings or get_runtime_settings(), _DEFAULT_REPORT_DIR)


def _report_filename(run_id: str) -> str:
    """Return a filesystem-safe canonical report filename."""
    return report_filename(run_id)


def _write_report_atomic(path: Path, markdown: str) -> None:
    """Write Markdown via a sibling temporary file and atomic replacement."""
    write_report_atomic(path, markdown)


def run_report_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Render the preliminary report as the terminal graph node."""
    try:
        return {
            "report_markdown": generate_report(state),
            "report_path": None,
            "report_status": "rendered",
            "report_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - reporting must not mask remediation
        log.exception("report_node: deterministic rendering failed")
        return {
            "report_markdown": "",
            "report_path": None,
            "report_status": "failed",
            "report_error": str(exc),
            "errors": [f"report_node: failed to render report: {exc}"],
        }


def finalize_report(
    state: Mapping[str, Any],
    *,
    recorder: TrajectoryRecorder | None,
    trajectory_path: str | None,
    trace_url: str | None,
    settings: AppSettings | None = None,
) -> tuple[str, Path | None]:
    """Render and atomically persist the final report.

    Args:
        state: Final graph state to summarize.
        recorder: Remediation trajectory recorder used for token totals.
        trajectory_path: Path to the exported trajectory, when available.
        trace_url: Remote trace URL, when available.
        settings: Optional validated settings for report persistence and the
            report directory.

    Returns:
        A tuple containing Markdown and the canonical path. The path is
        ``None`` if persistence fails; the Markdown remains available.

    Side Effects:
        Writes the canonical Markdown report through an atomic sibling-file
        replacement.
    """
    token_summary = _extract_token_summary(recorder)
    ended_at = datetime.now(UTC)
    settings = settings or get_runtime_settings()
    base_context = _build_context(
        state,
        trajectory_path=trajectory_path,
        trace_url=trace_url,
        token_summary=token_summary,
        run_ended_at=ended_at,
    )
    markdown = generate_report(
        state,
        trajectory_path=trajectory_path,
        trace_url=trace_url,
        token_summary=token_summary,
        run_ended_at=ended_at,
    )
    try:
        report_dir = _resolve_report_dir(settings)
        path = report_dir / _report_filename(base_context.run_id)
        _write_report_atomic(path, markdown)
        return markdown, path
    except Exception as exc:  # noqa: BLE001 - return usable preliminary report
        log.exception("final report persistence failed: %s", exc)
        return markdown, None


__all__ = [
    "finalize_report",
    "generate_report",
    "run_report_node",
]
