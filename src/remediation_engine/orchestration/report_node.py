"""Deterministic human-readable reporting for a Phase 5 remediation run.

The graph node renders the state available after teardown and before the graph
exits.  The orchestrator then finalizes and persists that report before the
final trajectory export, so both artifacts contain the same report metadata.
The optional LLM call is limited to an executive narrative and cannot
determine any report facts or statuses.
"""

from __future__ import annotations

import json
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
    executive_narrative: str | None = None,
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
    added_ids = _reconciliation_ids(reconciliation, "added", "new_group_ids")
    reappeared_ids = _reconciliation_ids(
        reconciliation,
        "reappeared",
        "reappeared_group_ids",
    )
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
    post_scan_issues = _items(state.get("post_remediation_scan_issues"))
    if not post_scan_issues:
        post_scan_issues = _items(_value(final_scan, "found_issues"))
    post_scan_identifiers = _unique_texts(
        _items(state.get("post_remediation_scan_identifiers"))
        or _items(_value(final_scan, "found_identifiers"))
        or [identifier for issue in post_scan_issues for identifier in _issue_identifiers(issue)]
    )
    new_identifiers = _unique_texts(
        _items(state.get("new_vulnerability_identifiers"))
        or _items(_value(final_scan, "new_identifiers"))
    )
    remaining_identifiers = _unique_texts(
        _items(state.get("remaining_target_identifiers"))
        or _items(_value(final_scan, "remaining_target_identifiers"))
    )
    new_status = _text(
        state.get("new_vulnerability_status") or _value(final_scan, "status"),
        "not_scanned",
    )
    scan_evidence_state = _scan_evidence_state(final_scan, new_status)
    triage_required_value = state.get("triage_required")
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
        statuses.get(_text(_value(group, "group_id")), "pending") for group in initial_groups
    )
    counts = {
        "fixed": status_counts.get("qa_passed", 0) + status_counts.get("mitigated", 0),
        "unresolved": status_counts.get("unfixable", 0) + status_counts.get("needs_retry", 0),
        "inconclusive": status_counts.get("inconclusive", 0),
        "pending": status_counts.get("pending", 0) + status_counts.get("optimistically_fixed", 0),
    }
    discovered_ids = _unique_texts([*added_ids, *reappeared_ids])
    if scan_evidence_state == "complete" and (discovered_ids or not triage_required):
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
        actionable_groups=len(initial_groups),
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
        executive_narrative=executive_narrative,
        new_groups_discovered=new_group_metrics["discovered"],
        new_groups_unresolved=new_group_metrics["unresolved"],
        new_groups_inconclusive=new_group_metrics["inconclusive"],
        new_groups_pending=new_group_metrics["pending"],
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


def _group_location(group: Any) -> str:
    """Return the most useful source location for a group."""
    paths = _items(_value(group, "file_paths"))
    if paths:
        return ", ".join(_text(path) for path in paths)
    return _text(_value(group, "file_path"), "—")


def _group_sources(group: Any, issue: Any | None) -> str:
    """Return scanner/source labels for a group."""
    sources = [_text(source) for source in _items(_value(group, "sources"))]
    if not sources and issue is not None:
        sources = [_text(_value(issue, "source"))]
    return ", ".join(sorted(set(sources))) or "—"


def _group_severity(group: Any, issue: Any | None) -> str:
    """Return the original scanner severity for a group."""
    return _text(
        _value(issue, "severity") if issue is not None else _value(group, "severity"),
        "unknown",
    )


def _finding_identifier(group: Any, issue: Any | None) -> str:
    """Return a stable human-facing identifier for a finding row."""
    identifiers = _items(_value(group, "cve_ids")) + _items(_value(group, "ghsa_ids"))
    if issue is not None:
        identifiers.extend(
            [_value(issue, "cve_id"), _value(issue, "ghsa_id"), _value(issue, "rule_id")]
        )
    labels = sorted({_text(identifier) for identifier in identifiers if identifier})
    return ", ".join(labels) or _text(_value(group, "group_id"), "unknown")


def _validation_for_group(context: ReportContext, group_id: str) -> str:
    """Summarize the latest QA evaluation associated with a group."""
    task_ids = [
        _text(_value(task, "task_id")) for task in _group_tree(context.task_queue, group_id)
    ]
    evaluations = [context.qa_evaluations.get(key) for key in [group_id, *task_ids]]
    for evaluation in reversed([item for item in evaluations if item is not None]):
        if bool(_value(evaluation, "passed", False)):
            return "Passed"
        category = _text(_value(evaluation, "failure_category"), "failed")
        return f"Failed ({category})"
    status = context.group_statuses.get(group_id, "pending")
    return {
        "qa_passed": "Passed",
        "inconclusive": "Inconclusive",
        "unfixable": "Not fixed",
    }.get(status, "Not evaluated")


_FINDING_HEADERS = (
    "Finding",
    "Source",
    "Location",
    "Package/component",
    "Severity",
    "Remediation",
    "Final change",
    "Final status",
    "Validation",
)
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
_FOLLOW_UP_STATUS_LABELS = {
    "unfixable": "Unresolved",
    "needs_retry": "Unresolved — retry required",
    "inconclusive": "Inconclusive",
    "pending": "Pending",
    "optimistically_fixed": "Awaiting validation",
    "awaiting_validation": "Awaiting validation",
    "pivoted": "Unresolved — no active task",
}


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


def _follow_up_groups(context: ReportContext) -> list[tuple[str, Any]]:
    """Return one follow-up row per issue lineage plus discovered groups."""
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
    return sorted(selected.values(), key=lambda item: item[0])


def _discovered_group_ids(context: ReportContext) -> list[str]:
    """Return newly added and reappeared group IDs from reconciliation."""
    return _reconciliation_ids(
        context.triage_reconciliation,
        "added",
        "new_group_ids",
        "reappeared",
        "reappeared_group_ids",
    )


def _new_group_metric(context: ReportContext, value: int | None) -> int | str:
    """Render a discovered-group metric without treating missing scans as zero."""
    if value is not None:
        return value
    if (
        _scan_evidence_state(context.final_full_scan_result, context.new_vulnerability_status)
        == "complete"
        and context.triage_required
    ):
        return "Not assessed — post-scan triage required"
    return _scan_assessment_text(context)


def _task_ids_for_group(context: ReportContext, group_id: str) -> list[str]:
    """Return task-lineage IDs associated with a vulnerability group."""
    return [
        task_id
        for task in _group_tree(context.task_queue, group_id)
        if (task_id := _text(_value(task, "task_id")))
    ]


def _package_change_matches_group(change: PackageChange, group: Any) -> bool:
    """Return whether a package change identifies the supplied finding group."""
    names: set[str] = set()
    issue = _group_issue(group)
    for value in (
        _value(group, "vulnerable_component"),
        _value(group, "package_name"),
        _value(group, "parent_package_name"),
        _value(issue, "package_name"),
    ):
        text = _text(value).strip()
        if not text:
            continue
        names.add(text)
        names.update(part.strip() for part in re.split(r"[|,]", text) if part.strip())
    return change.name in names


def _final_change_for_group(context: ReportContext, group: Any) -> str:
    """Render only the final validated change for a successful group."""
    group_id = _text(_value(group, "group_id"))
    status = context.group_statuses.get(group_id, "pending")
    if status not in {"qa_passed", "mitigated"}:
        return "No validated change"

    for change in _package_changes(context.diff):
        if not _package_change_matches_group(change, group):
            continue
        previous = change.old or "not present"
        current = change.new or "removed"
        return f"{change.name}: {previous} → {current} ({change.file})"

    task_ids = set(_task_ids_for_group(context, group_id))
    changed_files: list[str] = []
    successful_summaries: list[Any] = []
    for result in context.worker_results.values():
        if _text(_value(result, "task_id")) not in task_ids:
            continue
        result_status = _text(_value(result, "status")).lower()
        diagnostics = _value(result, "execution_diagnostics")
        if result_status != "success" and not bool(_value(diagnostics, "validation_passed")):
            continue
        changed_files.extend(_text(path) for path in _items(_value(result, "changed_files")))
        changed_files.extend(_text(path) for path in _items(_value(diagnostics, "validated_files")))
        action_summary = _value(result, "action_summary")
        if action_summary is not None and (
            result_status == "success"
            or _text(_value(action_summary, "status")).lower() == "success"
        ):
            successful_summaries.append(action_summary)

    for summary in reversed(context.action_summaries):
        if _text(_value(summary, "task_id")) not in task_ids:
            continue
        if _text(_value(summary, "status")).lower() != "success":
            continue
        successful_summaries.append(summary)

    final_summary = ""
    for summary in reversed(successful_summaries):
        text = _compact_remediation_attempt(
            _value(summary, "summary"),
            status=_text(_value(summary, "status"), "success"),
            include_changed_files=False,
        )
        if text:
            final_summary = text
            break
    final_files = _unique_texts([*changed_files, *context.changed_files])
    if final_files and final_summary:
        return f"Changed files: {', '.join(final_files)}; {final_summary}"
    if final_files:
        return "Changed files: " + ", ".join(final_files)
    if final_summary:
        return final_summary
    return "Validated remediation recorded"


def _required_follow_up_action(
    context: ReportContext,
    group: Any,
    status: str,
    *,
    group_id: str | None = None,
) -> str:
    """Describe the next user-facing action for an outstanding group."""
    group_id = group_id or _text(_value(group, "group_id"))
    if status == "needs_retry":
        action_prefix = "Retry the remediation using the current plan"
    elif status == "unfixable":
        action_prefix = "Apply an alternative remediation"
    elif status == "inconclusive":
        action_prefix = "Resolve the incomplete or invalid validation evidence"
    elif status in {"optimistically_fixed", "awaiting_validation"}:
        action_prefix = "Complete QA for the attempted remediation"
    elif status == "pivoted":
        action_prefix = "Resume remediation for the issue"
    else:
        action_prefix = "Complete the planned remediation"

    if status in {"needs_retry", "pending"}:
        for task_id in reversed(_task_ids_for_group(context, group_id)):
            plan = context.retry_plans.get(task_id)
            task = context.task_queue.get(task_id)
            instruction = (
                _value(plan, "exact_instruction")
                or _value(plan, "instructions")
                or _value(task, "instruction")
            )
            if _text(instruction).strip():
                instruction_text = _full_text(instruction).rstrip(" .")
                return f"{action_prefix}: {instruction_text}. Then rerun QA."

    if group_id in set(_discovered_group_ids(context)) and not _task_ids_for_group(
        context, group_id
    ):
        return "Remediate the newly discovered issue, then rerun QA and the authoritative security scan."
    return f"{action_prefix}, then rerun QA and the authoritative security scan."


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
    """Extract a concise changed-file list from an attempt's lead paragraph."""
    text = _attempt_primary_text(value)
    match = _CHANGED_FILES_RE.search(text)
    if match is None:
        return [], False
    file_text = text[match.end() :]
    file_text = _FINAL_NOTE_RE.split(file_text, maxsplit=1)[0]
    file_text = file_text.strip().rstrip(".;")
    if file_text.casefold() in {"", "none", "no files", "n/a"}:
        return [], True
    files = [part.strip().strip("`") for part in file_text.split(",")]
    return [part for part in files if part], False


def _compact_remediation_attempt(
    value: Any,
    *,
    status: str,
    changed_files: Sequence[Any] = (),
    include_changed_files: bool = True,
) -> str:
    """Render only the concise remediation action from an agent summary.

    The worker summary is run-specific prose and may contain separate
    validation, notes, evidence, or diagnostic sections.  The report keeps
    the lead remediation action and changed files, while dropping those
    sections without truncating the retained action text.
    """
    raw = _text(value).strip()
    if not raw:
        return ""
    primary = _attempt_primary_text(raw)
    if not primary:
        return ""

    final_note = _FINAL_NOTE_RE.search(primary)
    status_name = status.strip().casefold()
    if final_note is not None:
        prefix = primary[: final_note.start()].strip(" ;:-")
        note = primary[final_note.end() :].strip()
        action = note if status_name in {"success", "mitigated"} and note else prefix or note
    else:
        action = primary

    parsed_files, explicit_no_files = _attempt_changed_files(raw)
    files = _unique_texts([*changed_files, *parsed_files])
    changed_clause = _CHANGED_FILES_RE.search(action)
    if changed_clause is not None:
        action = action[: changed_clause.start()].strip(" ;:-.")
    action = " ".join(action.split()).rstrip(" ;.:")
    if not action:
        action = "Remediation attempt recorded"

    parts = [action]
    if include_changed_files:
        if files:
            parts.append(f"changed files: {', '.join(files)}")
        elif explicit_no_files:
            parts.append("no files changed")
    return "; ".join(parts).rstrip(".") + "."


def _attempted_fixes_for_group(context: ReportContext, group_id: str) -> str:
    """Aggregate remediation-attempt summaries for one follow-up row."""
    task_ids = set(_task_ids_for_group(context, group_id))
    entries: list[str] = []
    seen_attempt_ids: set[str] = set()
    seen_entries: set[tuple[str, str]] = set()
    worker_results = sorted(
        context.worker_results.values(),
        key=lambda item: _text(_value(item, "attempt_id")),
    )
    worker_by_attempt = {
        _text(_value(result, "attempt_id")): result
        for result in worker_results
        if _text(_value(result, "attempt_id"))
    }

    def add_attempt(summary: Any, metadata: Any = None) -> None:
        if summary is None:
            return
        summary_text = summary if isinstance(summary, (str, bytes)) else _value(summary, "summary")
        if not _text(summary_text).strip():
            return
        attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
        if attempt_id and attempt_id in seen_attempt_ids:
            return
        status = _text(_value(summary, "status")) or _text(_value(metadata, "status"), "recorded")
        structured_files = [
            *_items(_value(summary, "changed_files")),
            *_items(_value(metadata, "changed_files")),
        ]
        compact = _compact_remediation_attempt(
            summary_text,
            status=status,
            changed_files=structured_files,
        )
        if not compact:
            return
        key = (status.casefold(), compact.casefold())
        if key in seen_entries:
            return
        seen_entries.add(key)
        if attempt_id:
            seen_attempt_ids.add(attempt_id)
        entries.append(f"Remediation attempt ({status}): {compact}")

    for summary in context.action_summaries:
        if _text(_value(summary, "task_id")) not in task_ids:
            continue
        attempt_id = _text(_value(summary, "attempt_id"))
        add_attempt(summary, worker_by_attempt.get(attempt_id))

    for result in worker_results:
        if _text(_value(result, "task_id")) not in task_ids:
            continue
        action_summary = _value(result, "action_summary")
        add_attempt(action_summary, result)

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
) -> None:
    """Record one added or removed package version from a diff line."""
    if name in _NON_PACKAGE_KEYS:
        return
    record = records.setdefault(name, {"old": "", "new": "", "file": file_path})
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
    manifest_section_indent: int | None = None
    lockfile_package: tuple[str, int] | None = None

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            manifest_section_indent = None
            lockfile_package = None
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
                manifest_section_indent = indent
                continue
            if manifest_section_indent is not None:
                if content.strip() and indent <= manifest_section_indent:
                    manifest_section_indent = None
                elif prefix in {"+", "-"}:
                    package_match = _PACKAGE_LINE_RE.match(content)
                    if package_match:
                        _record_package_change(
                            manifest_records,
                            package_match.group("name"),
                            package_match.group("version"),
                            prefix,
                            current_file,
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
                    )
            continue

        if "lock" in current_file.lower():
            package_match = _LOCKFILE_PACKAGE_RE.match(content)
            if package_match:
                raw_name = package_match.group("name")
                name = raw_name.removeprefix("node_modules/")
                if name and name not in _NON_PACKAGE_KEYS:
                    lockfile_package = (name, indent)
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
                    _record_package_change(
                        lockfile_records,
                        lockfile_package[0],
                        version_match.group("version"),
                        prefix,
                        current_file,
                    )

    direct_names = set(manifest_records)
    changes: list[PackageChange] = []
    for name in sorted(manifest_records):
        record = manifest_records[name]
        if record["old"] or record["new"]:
            evidence_file = record["file"]
            if name in lockfile_records:
                evidence_file += "; lockfile synchronized"
            changes.append(
                PackageChange(name, record["old"], record["new"], evidence_file, "direct")
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
                )
            )
    return sorted(changes, key=lambda change: (change.scope != "direct", change.name))


def _scan_assessment_text(context: ReportContext) -> str:
    """Describe unavailable post-scan evidence without implying zero findings."""
    if (
        _scan_evidence_state(context.final_full_scan_result, context.new_vulnerability_status)
        == "failed"
    ):
        return "Unknown — authoritative scan failed"
    return "Not assessed — no authoritative scan"


def _render_summary(context: ReportContext) -> str:
    """Render the compact user-facing run summary."""
    duration = (
        f"{context.duration_seconds:.2f} seconds"
        if context.duration_seconds is not None
        else "Pending finalization"
    )
    token_rows = (
        [
            ("Input tokens", context.total_input_tokens),
            ("Output tokens", context.total_output_tokens),
            ("Total tokens", context.total_tokens),
        ]
        if context.total_tokens is not None
        else [("Tokens", "Unavailable")]
    )
    metrics = [
        ("Run ID", context.run_id),
        ("Repository", context.repo_root),
        ("Status", f"{context.overall_label} ({context.status})"),
        ("Total time taken", duration),
        ("Original scanner findings", context.original_scanner_findings),
        ("Actionable groups", context.actionable_groups),
        ("Groups fixed", context.groups_fixed),
        ("Groups unresolved", context.groups_unresolved),
        ("Groups inconclusive", context.groups_inconclusive),
        ("Groups pending", context.groups_pending),
        ("New groups discovered", _new_group_metric(context, context.new_groups_discovered)),
        ("New unresolved groups", _new_group_metric(context, context.new_groups_unresolved)),
        (
            "New inconclusive groups",
            _new_group_metric(context, context.new_groups_inconclusive),
        ),
        ("New pending groups", _new_group_metric(context, context.new_groups_pending)),
        *token_rows,
    ]
    lines = ["# Remediation Run Report", "", "## 1. Summary", ""]
    if context.executive_narrative:
        narrative = "\n".join(
            line
            for line in context.executive_narrative.splitlines()
            if not line.lstrip().startswith("#")
        ).strip()
        if narrative:
            lines.extend([narrative, ""])
    lines.append(_table(("Metric", "Value"), metrics))
    return "\n".join(lines)


def _finding_row(context: ReportContext, group: Any) -> tuple[str, ...]:
    """Build one compact findings-overview row."""
    issue = _group_issue(group)
    group_id = _text(_value(group, "group_id"))
    return (
        _finding_identifier(group, issue),
        _group_sources(group, issue),
        _group_location(group),
        _text(_value(group, "vulnerable_component"), "—"),
        _group_severity(group, issue),
        _text(_value(_value(group, "fix_plan"), "status"), "—"),
        _final_change_for_group(context, group),
        context.group_statuses.get(group_id, "pending"),
        _validation_for_group(context, group_id),
    )


def _newly_discovered_finding_rows(context: ReportContext) -> list[tuple[str, ...]]:
    """Build findings-overview rows for new and reappeared groups."""
    scan_state = _scan_evidence_state(
        context.final_full_scan_result, context.new_vulnerability_status
    )
    if scan_state != "complete":
        assessment = _scan_assessment_text(context)
        return [
            (
                "Not assessed",
                "—",
                "—",
                "—",
                "—",
                "—",
                "No validated change",
                "—",
                assessment,
            )
        ]

    discovered_ids = set(_discovered_group_ids(context))
    groups = [
        group
        for group in _report_groups(context)
        if _text(_value(group, "group_id")) in discovered_ids
    ]
    if groups:
        return [_finding_row(context, group) for group in groups]
    if discovered_ids:
        return [
            (
                group_id,
                "—",
                "—",
                "—",
                "unknown",
                "—",
                "No validated change",
                "pending",
                "Not evaluated",
            )
            for group_id in sorted(discovered_ids)
        ]

    if context.triage_required:
        return [
            (
                "Not assessed",
                "—",
                "—",
                "—",
                "—",
                "—",
                "No validated change",
                "pending",
                "Not assessed — post-scan triage required",
            )
        ]
    return [("None", "—", "—", "—", "—", "—", "—", "—", "No newly discovered groups")]


def _render_follow_up_actions(context: ReportContext) -> str:
    """Render required actions and complete attempt evidence for open groups."""
    status_order = {
        "unfixable": 0,
        "needs_retry": 1,
        "pivoted": 2,
        "inconclusive": 3,
        "pending": 4,
        "optimistically_fixed": 5,
    }
    groups = [
        (group_id, group)
        for group_id, group in _follow_up_groups(context)
        if context.group_statuses.get(group_id, "pending") in _OUTSTANDING_GROUP_STATUSES
    ]
    groups.sort(
        key=lambda item: (
            status_order.get(
                context.group_statuses.get(item[0], "pending"),
                99,
            ),
            item[0],
        )
    )
    rows: list[tuple[str, ...]] = []
    for group_id, group in groups:
        status = context.group_statuses.get(group_id, "pending")
        rows.append(
            (
                _finding_identifier(group, _group_issue(group)),
                _text(_value(group, "vulnerable_component"), "—"),
                _FOLLOW_UP_STATUS_LABELS.get(status, status),
                _required_follow_up_action(context, group, status, group_id=group_id),
                _attempted_fixes_for_group(context, group_id),
            )
        )
    if not rows:
        rows = [("None", "—", "—", "No follow-up actions required", "—")]

    return "\n".join(
        [
            "## {number}. Follow up Actions",
            "",
            _table(
                (
                    "Finding",
                    "Package/component",
                    "Status",
                    "Required follow-up action",
                    "Attempted fixes",
                ),
                rows,
            ),
        ]
    )


def _render_findings(context: ReportContext) -> str:
    """Render original and newly discovered findings with shared columns."""
    original_rows = [
        _finding_row(context, group)
        for group in sorted(
            context.initial_valid_groups,
            key=lambda item: _text(_value(item, "group_id")),
        )
    ]
    original_table = _table(_FINDING_HEADERS, original_rows or [("None",) * 9])
    discovered_table = _table(_FINDING_HEADERS, _newly_discovered_finding_rows(context))
    return "\n".join(
        [
            "## {number}. Findings Overview",
            "",
            "### Original Findings",
            "",
            original_table,
            "",
            "### Newly Discovered Groups",
            "",
            discovered_table,
        ]
    )


def _render_references(context: ReportContext) -> str:
    """Render lightweight artifact references."""
    return "\n".join(
        [
            "## {number}. References",
            "",
            _table(
                ("Artifact", "Reference"),
                [
                    ("Trajectory", context.trajectory_path or "Not available"),
                    ("LangSmith trace", context.langsmith_trace_url or "Not available"),
                    ("Patch", "Included in run result" if context.has_patch else "No unified diff"),
                    ("Changed files", ", ".join(context.changed_files) or "None"),
                ],
            ),
        ]
    )


def _evidence_payload(context: ReportContext) -> dict[str, Any]:
    """Build the bounded deterministic evidence supplied to an optional LLM."""
    return {
        "status": context.status,
        "overall_label": context.overall_label,
        "metrics": {
            "original_scanner_findings": context.original_scanner_findings,
            "actionable_groups": context.actionable_groups,
            "groups_fixed": context.groups_fixed,
            "groups_unresolved": context.groups_unresolved,
            "groups_inconclusive": context.groups_inconclusive,
            "groups_pending": context.groups_pending,
            "new_groups_discovered": context.new_groups_discovered,
            "new_groups_unresolved": context.new_groups_unresolved,
            "new_groups_inconclusive": context.new_groups_inconclusive,
            "new_groups_pending": context.new_groups_pending,
        },
        "group_statuses": context.group_statuses,
        "strategies": {key: _text(value) for key, value in context.group_strategies.items()},
        "reconciliation": context.triage_reconciliation,
        "changed_files": context.changed_files,
    }


def _generate_executive_narrative(
    context: ReportContext,
    settings: AppSettings,
) -> str | None:
    """Ask an optional LLM to summarize supplied evidence without making decisions."""
    if not settings.report_llm_enabled:
        return None
    try:
        from langchain_openai import ChatOpenAI

        evidence = json.dumps(_evidence_payload(context), sort_keys=True, default=str)
        prompt = (
            "Write a concise executive narrative for a human reader of a software security remediation run.\n"
            "Use only the deterministic evidence below. Do not add facts, calculate metrics, change statuses, "
            "or recommend actions. Do not use a heading. Write 3 to 6 short paragraphs.\n\n"
            f"Deterministic evidence:\n{evidence[:16000]}"
        )
        response = ChatOpenAI(
            model=settings.report_llm_model,
            temperature=0.3,
            max_tokens=1500,
        ).invoke(prompt)
        content = _value(response, "content")
        narrative = "\n".join(
            line for line in _text(content).splitlines() if not line.lstrip().startswith("#")
        ).strip()
        return narrative or None
    except Exception as exc:  # noqa: BLE001 - optional narrative must not block reporting
        log.warning("report narrative generation failed: %s", exc)
        return None


def generate_report(
    state: Mapping[str, Any],
    *,
    trajectory_path: str | None = None,
    trace_url: str | None = None,
    token_summary: Mapping[str, Any] | None = None,
    run_ended_at: datetime | None = None,
    executive_narrative: str | None = None,
) -> str:
    """Render a deterministic Markdown report from graph state.

    Args:
        state: Graph state or a compatible mapping of remediation evidence.
        trajectory_path: Optional exported trajectory reference.
        trace_url: Optional LangSmith trace reference.
        token_summary: Optional prompt/completion token totals.
        run_ended_at: Optional final timestamp; omitted values remain pending.
        executive_narrative: Optional bounded narrative supplied by the caller.

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
        executive_narrative=executive_narrative,
    )
    sections = [
        _render_follow_up_actions(context),
        _render_findings(context),
        _render_references(context),
    ]
    report = _render_summary(context)
    section_number = 2
    for section in sections:
        report += "\n\n" + section.replace("{number}", str(section_number), 1)
        section_number += 1
    return report + "\n"


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
    """Enrich, optionally narrate, and atomically persist the final report.

    Args:
        state: Final graph state to summarize.
        recorder: Remediation trajectory recorder used for token totals.
        trajectory_path: Path to the exported trajectory, when available.
        trace_url: Remote trace URL, when available.
        settings: Optional validated settings for report persistence and the
            optional narrative model.

    Returns:
        A tuple containing Markdown and the canonical path. The path is
        ``None`` if persistence fails; the Markdown remains available.

    Side Effects:
        May invoke the optional report narrative LLM and writes the canonical
        Markdown report through an atomic sibling-file replacement.
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
    narrative = _generate_executive_narrative(base_context, settings)
    markdown = generate_report(
        state,
        trajectory_path=trajectory_path,
        trace_url=trace_url,
        token_summary=token_summary,
        run_ended_at=ended_at,
        executive_narrative=narrative,
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
