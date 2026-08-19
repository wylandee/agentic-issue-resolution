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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from remediation_engine.settings import AppSettings

from .trajectory_exporter import TrajectoryRecorder

log = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class _ReportContext:
    """Normalized, deterministic evidence used by the Markdown renderer."""

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
    groups_retriage_discovered: int
    consistency_events: list[Any]
    error_strings: list[str]
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
    post_remediation_scan_issues: list[Any]
    post_remediation_scan_identifiers: list[str]
    new_vulnerability_identifiers: list[str]
    new_vulnerability_status: str
    diff: str
    changed_files: list[str]
    trajectory_path: str | None
    langsmith_trace_url: str | None
    executive_narrative: str | None = None


@dataclass(frozen=True)
class _PackageChange:
    """One direct or lockfile package version change extracted from a diff."""

    name: str
    old: str
    new: str
    file: str
    scope: str


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
    """Return the root task and all descendants for one original group."""
    roots = [
        task
        for task in task_queue.values()
        if _value(task, "parent_group_id") == group_id and _value(task, "parent_task_id") is None
    ]
    result: list[Any] = []
    children: dict[str, list[Any]] = defaultdict(list)
    for task in task_queue.values():
        parent = _value(task, "parent_task_id")
        if parent:
            children[str(parent)].append(task)
    stack = sorted(roots, key=lambda task: _text(_value(task, "task_id")))
    while stack:
        task = stack.pop(0)
        result.append(task)
        stack[0:0] = sorted(
            children.get(_text(_value(task, "task_id")), []),
            key=lambda child: _text(_value(child, "task_id")),
        )
    return result


def _group_status(task_queue: Mapping[str, Any], group_id: str) -> str:
    """Collapse a root task and any pivot children to one group status."""
    statuses = [
        _text(_value(task, "status"), "pending") for task in _group_tree(task_queue, group_id)
    ]
    if not statuses:
        return "pending"
    if "qa_passed" in statuses:
        return "qa_passed"
    if "inconclusive" in statuses:
        return "inconclusive"
    if "unfixable" in statuses:
        return "unfixable"
    if "mitigated" in statuses:
        return "mitigated"
    if "needs_retry" in statuses:
        return "needs_retry"
    if "optimistically_fixed" in statuses:
        return "optimistically_fixed"
    return "pending"


def _overall_label(
    status: str,
    counts: Mapping[str, int],
    new_vulnerability_status: str,
    new_identifier_count: int,
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


def _error_strings(state: Mapping[str, Any]) -> list[str]:
    """Collect critical run errors and final-scan errors in stable order."""
    result: list[str] = []
    seen: set[str] = set()
    candidates = _items(state.get("errors"))
    final_scan = state.get("final_full_scan_result")
    scan_error = _value(final_scan, "error")
    if scan_error:
        candidates.append(f"final_full_scan: {scan_error}")
    for candidate in candidates:
        text = _text(candidate).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


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
    return _unique_texts(
        [
            _value(issue, "cve_id"),
            _value(issue, "ghsa_id"),
            _value(issue, "rule_id"),
            _value(issue, "finding_id"),
        ]
    )


def _build_context(
    state: Mapping[str, Any],
    *,
    trajectory_path: str | None = None,
    trace_url: str | None = None,
    token_summary: Mapping[str, Any] | None = None,
    run_ended_at: datetime | None = None,
    executive_narrative: str | None = None,
) -> _ReportContext:
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
    status_counts = Counter(
        statuses.get(_text(_value(group, "group_id")), "pending") for group in initial_groups
    )
    counts = {
        "fixed": status_counts.get("qa_passed", 0) + status_counts.get("mitigated", 0),
        "unresolved": status_counts.get("unfixable", 0) + status_counts.get("needs_retry", 0),
        "inconclusive": status_counts.get("inconclusive", 0),
        "pending": status_counts.get("pending", 0) + status_counts.get("optimistically_fixed", 0),
    }
    reconciliation = _mapping(state.get("triage_reconciliation"))
    added_ids = _unique_texts(
        _items(
            reconciliation.get("added")
            if "added" in reconciliation
            else reconciliation.get("new_group_ids")
        )
    )
    reappeared_ids = _unique_texts(
        _items(
            reconciliation.get("reappeared")
            if "reappeared" in reconciliation
            else reconciliation.get("reappeared_group_ids")
        )
    )
    ended = _iso(run_ended_at)
    started = _iso(state.get("run_started_at"))
    tokens = dict(token_summary or {})
    status = _text(state.get("status"), "unknown")
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
    new_status = _text(
        state.get("new_vulnerability_status") or _value(final_scan, "status"),
        "not_scanned",
    )
    qa_evaluations = _mapping(state.get("qa_evaluations"))
    qa_results = _mapping(state.get("qa_results_by_attempt"))
    recorded_qa_total = _qa_record_summary(qa_evaluations, qa_results)
    return _ReportContext(
        run_id=_text(state.get("run_id") or state.get("langsmith_run_id"), "local-run"),
        repo_root=_text(state.get("repo_root")),
        run_started_at=started,
        run_ended_at=ended,
        duration_seconds=_duration_seconds(started, ended),
        total_input_tokens=tokens.get("input_tokens"),
        total_output_tokens=tokens.get("output_tokens"),
        total_tokens=tokens.get("total_tokens"),
        status=status,
        overall_label=_overall_label(status, counts, new_status, len(new_identifiers)),
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
        groups_retriage_discovered=len(added_ids) + len(reappeared_ids),
        consistency_events=_items(state.get("consistency_events")),
        error_strings=_error_strings(state),
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
        post_remediation_scan_issues=post_scan_issues,
        post_remediation_scan_identifiers=post_scan_identifiers,
        new_vulnerability_identifiers=new_identifiers,
        new_vulnerability_status=new_status,
        diff=_text(state.get("diff")),
        changed_files=[_text(item) for item in _items(state.get("changed_files"))],
        trajectory_path=trajectory_path or _text(state.get("trajectory_path")) or None,
        langsmith_trace_url=trace_url or _text(state.get("langsmith_trace_url")) or None,
        executive_narrative=executive_narrative,
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


def _group_action(context: _ReportContext, group: Any) -> str:
    """Describe the committed action for one group."""
    group_id = _text(_value(group, "group_id"))
    tasks = _group_tree(context.task_queue, group_id)
    if tasks:
        task = tasks[-1]
        strategy = _text(_value(task, "strategy"), "unknown")
        instruction = _text(_value(task, "instruction"), "")
        if instruction:
            return f"{strategy}: {instruction}"
        return strategy
    plan = _value(group, "fix_plan")
    if plan is not None:
        return f"{_text(_value(plan, 'strategy_used'), 'planned')}: {_text(_value(plan, 'instruction'))}"
    return "No remediation task recorded"


def _validation_for_group(context: _ReportContext, group_id: str) -> str:
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


def _package_change_kind(change: _PackageChange) -> str:
    """Classify a package change for compact report summaries."""
    if change.old and change.new and change.old != change.new:
        return "changed"
    if change.new and not change.old:
        return "added"
    if change.old and not change.new:
        return "removed"
    return "unchanged"


def _package_changes(diff: str) -> list[_PackageChange]:
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
    changes: list[_PackageChange] = []
    for name in sorted(manifest_records):
        record = manifest_records[name]
        if record["old"] or record["new"]:
            evidence_file = record["file"]
            if name in lockfile_records:
                evidence_file += "; lockfile synchronized"
            changes.append(
                _PackageChange(name, record["old"], record["new"], evidence_file, "direct")
            )
    for name in sorted(lockfile_records):
        if name in direct_names:
            continue
        record = lockfile_records[name]
        if record["old"] or record["new"]:
            changes.append(
                _PackageChange(
                    name,
                    record["old"],
                    record["new"],
                    record["file"],
                    "transitive",
                )
            )
    return sorted(changes, key=lambda change: (change.scope != "direct", change.name))


def _targeted_remediation_summary(context: _ReportContext) -> str:
    """Summarize the outcome for the original actionable groups."""
    return (
        f"{context.groups_fixed}/{context.actionable_groups} actionable groups fixed; "
        f"{context.groups_unresolved} unresolved, "
        f"{context.groups_inconclusive} inconclusive, "
        f"{context.groups_pending} pending"
    )


def _post_scan_summary(context: _ReportContext) -> str:
    """Summarize post-remediation scan evidence without conflating its counts."""
    status = context.new_vulnerability_status
    if status == "detected" or context.new_vulnerability_identifiers:
        return (
            f"{len(context.post_remediation_scan_issues)} findings / "
            f"{len(context.post_remediation_scan_identifiers)} unique identifiers; "
            f"{len(context.new_vulnerability_identifiers)} new identifiers detected"
        )
    if status in {"scan_failed", "failed"}:
        return "Scanner validation failed"
    if status in {"none", "clear", "not_detected"}:
        return "No post-remediation findings detected"
    if status == "not_scanned":
        return "Not scanned"
    return status


def _render_summary(context: _ReportContext) -> str:
    """Render the compact run summary and critical errors."""
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
        ("Targeted remediation", _targeted_remediation_summary(context)),
        ("Post-remediation security status", _post_scan_summary(context)),
        ("Total time taken", duration),
        ("Original scanner findings", context.original_scanner_findings),
        ("Actionable groups", context.actionable_groups),
        ("Groups fixed", context.groups_fixed),
        ("Groups unresolved", context.groups_unresolved),
        ("Groups inconclusive", context.groups_inconclusive),
        ("Groups pending", context.groups_pending),
        ("Re-triage groups discovered", context.groups_retriage_discovered),
        (
            "Targeted QA coverage",
            f"{context.targeted_qa_passed} passed / {context.targeted_qa_total} targeted groups",
        ),
        ("Patch present", "yes" if context.has_patch else "no"),
        *token_rows,
    ]
    errors = (
        _table(("Critical error",), [(error,) for error in context.error_strings])
        if context.error_strings
        else "No critical errors recorded."
    )
    return "\n".join(
        [
            "# Remediation Run Report",
            "",
            "## 1. Run Summary",
            "",
            _table(("Metric", "Value"), metrics),
            "",
            "### Critical Errors Encountered",
            "",
            errors,
        ]
    )


def _render_overview(context: _ReportContext) -> str:
    """Render deterministic counts, strategies, reconciliation, and outcome."""
    issue_types = Counter(_text(_value(issue, "issue_type"), "unknown") for issue in context.issues)
    severity_counts = Counter(
        _group_severity(group, _group_issue(group)) for group in context.initial_valid_groups
    )
    type_text = ", ".join(f"{key}: {value}" for key, value in sorted(issue_types.items())) or "None"
    severity_text = (
        ", ".join(f"{key}: {value}" for key, value in sorted(severity_counts.items())) or "None"
    )
    strategy_counts = Counter(
        _text(_value(task, "strategy"), "unknown") for task in context.task_queue.values()
    )
    strategy_text = (
        ", ".join(f"{key}: {value}" for key, value in sorted(strategy_counts.items())) or "None"
    )
    reconciliation_rows = [
        (
            label,
            ", ".join(_text(item) for item in _items(context.triage_reconciliation.get(key)))
            or "None",
        )
        for label, key in (
            ("Added/new groups", "new_group_ids"),
            ("Reappeared groups", "reappeared_group_ids"),
            ("Changed groups", "changed_group_ids"),
            ("Removed groups", "removed_group_ids"),
        )
    ]
    return "\n".join(
        [
            "## {number}. Run Overview",
            "",
            f"The run processed {context.original_scanner_findings} scanner findings and {context.actionable_groups} initial actionable groups.",
            "",
            _table(
                ("Overview", "Value"),
                [
                    ("Findings by type", type_text),
                    ("Initial groups by severity", severity_text),
                    ("Task strategies", strategy_text),
                    ("Targeted remediation", _targeted_remediation_summary(context)),
                    ("Post-remediation security status", _post_scan_summary(context)),
                    ("Re-triage discovered groups", context.groups_retriage_discovered),
                ],
            ),
            "",
            "### Triage Reconciliation",
            "",
            _table(("Category", "Group IDs"), reconciliation_rows),
            "",
            f"Overall outcome: **{context.overall_label}**.",
        ]
    )


def _render_key_decisions(context: _ReportContext) -> str:
    """Render supervisor decisions, retries, QA failures, and repairs."""
    retry_rows = []
    for task_id in sorted(context.retry_plans):
        plan = context.retry_plans[task_id]
        retry_rows.append(
            (
                task_id,
                _text(_value(plan, "action"), _text(_value(plan, "next_node"), "retry")),
                _text(_value(plan, "selected_version"), "—"),
                _text(_value(plan, "instructions"), _text(_value(plan, "reason"), "—"))[:240],
            )
        )
    if not retry_rows:
        retry_rows = [("None", "—", "—", "No retry plans recorded")]
    failed_qa = []
    for key, evaluation in sorted(context.qa_evaluations.items()):
        if not bool(_value(evaluation, "passed", False)):
            failed_qa.append(
                (
                    key,
                    _text(_value(evaluation, "failure_category"), "unknown"),
                    _text(_value(evaluation, "retry_feedback"), "—")[:240],
                )
            )
    if not failed_qa:
        failed_qa = [("None", "—", "No failed QA evaluations recorded")]
    strategies_by_group = dict(context.group_strategies)
    for task in context.task_queue.values():
        group_id = _text(_value(task, "parent_group_id"))
        if group_id and group_id not in strategies_by_group:
            strategies_by_group[group_id] = _value(task, "strategy")
    strategy_rows = [
        (group_id, _text(strategy)) for group_id, strategy in sorted(strategies_by_group.items())
    ] or [("None", "No strategy selections recorded")]
    action_rows = [
        (
            _text(_value(summary, "task_id"), "—"),
            _text(_value(summary, "status"), "—"),
            _text(
                _value(summary, "summary"),
                _text(_value(summary, "message"), _value(summary, "rationale")),
            )[:240],
        )
        for summary in context.action_summaries
    ] or [("None", "—", "No worker action summaries recorded")]
    event_rows = [
        (
            _text(_value(event, "event_type"), "consistency event"),
            _text(_value(event, "reason"), event),
        )
        for event in context.consistency_events
    ] or [("None", "No consistency repairs recorded")]
    return "\n".join(
        [
            "## {number}. Key Decisions",
            "",
            "### Retry Plans and Pivots",
            "",
            _table(("Task", "Action", "Version", "Instruction/reason"), retry_rows),
            "",
            "### Failed QA Gates",
            "",
            _table(("Task/group", "Category", "Feedback"), failed_qa),
            "",
            "### Strategy Selections and Worker Actions",
            "",
            _table(("Group", "Strategy"), strategy_rows),
            "",
            _table(("Task", "Status", "Summary"), action_rows),
            "",
            "### Consistency Repairs and Replans",
            "",
            _table(("Event", "Details"), event_rows),
        ]
    )


def _render_findings(context: _ReportContext) -> str:
    """Render original findings, package changes, and post-retriage findings."""
    rows = []
    for group in sorted(
        context.initial_valid_groups, key=lambda item: _text(_value(item, "group_id"))
    ):
        issue = _group_issue(group)
        group_id = _text(_value(group, "group_id"))
        rows.append(
            (
                _finding_identifier(group, issue),
                _group_sources(group, issue),
                _group_location(group),
                _text(_value(group, "vulnerable_component"), "—"),
                _group_severity(group, issue),
                _text(_value(_value(group, "fix_plan"), "status"), "—"),
                _group_action(context, group),
                context.group_statuses.get(group_id, "pending"),
                _validation_for_group(context, group_id),
            )
        )
    original_table = _table(
        (
            "Finding",
            "Source",
            "Location",
            "Package/component",
            "Severity",
            "Remediation",
            "Action",
            "Final status",
            "Validation",
        ),
        rows or [("None",) * 9],
    )
    package_changes = _package_changes(context.diff)
    direct_package_rows = [
        (
            change.name,
            change.old or "-",
            change.new or "-",
            f"{_package_change_kind(change)} ({change.file})",
        )
        for change in package_changes
        if change.scope == "direct"
    ]
    direct_package_table = _table(
        ("Package", "Previous", "New", "Evidence"),
        direct_package_rows or [("None", "-", "-", "No direct manifest package changes found")],
    )
    transitive_changes = [change for change in package_changes if change.scope == "transitive"]
    transitive_counts = Counter(_package_change_kind(change) for change in transitive_changes)
    transitive_summary = (
        "No transitive lockfile package changes found."
        if not transitive_changes
        else (
            f"{len(transitive_changes)} transitive lockfile package entries changed: "
            + ", ".join(f"{kind} {count}" for kind, count in sorted(transitive_counts.items()))
            + "."
        )
    )
    diagnostic_rows = []
    for attempt_id, worker_result in sorted(context.worker_results.items()):
        diagnostics = _value(worker_result, "execution_diagnostics") or _value(
            worker_result, "diagnostics"
        )
        task_id = _text(_value(worker_result, "task_id"))
        snapshot = context.attempt_snapshots.get(attempt_id)
        if not task_id:
            task_id = _text(_value(snapshot, "task_id"), attempt_id)
        package = _text(_value(snapshot, "target_package_name"), "—")
        attempted = _value(diagnostics, "attempted_versions") or _value(
            worker_result, "attempted_versions"
        )
        executed = _value(diagnostics, "executed_versions") or _value(
            worker_result, "executed_versions"
        )
        if package != "—" or attempted or executed:
            diagnostic_rows.append(
                (
                    task_id,
                    package,
                    ", ".join(_text(item) for item in _items(attempted)) or "—",
                    ", ".join(_text(item) for item in _items(executed)) or "—",
                )
            )
    diagnostics_table = _table(
        ("Task", "Package", "Attempted versions", "Executed versions"),
        diagnostic_rows or [("None", "—", "—", "No worker package diagnostics recorded")],
    )
    discovered_ids = set(
        _reconciliation_ids(
            context.triage_reconciliation,
            "added",
            "new_group_ids",
            "reappeared",
            "reappeared_group_ids",
        )
    )
    retriage_groups = [
        group
        for group in context.final_valid_groups
        if _text(_value(group, "group_id")) in discovered_ids
    ]
    retriage_rows = [
        (
            _text(_value(group, "group_id")),
            _text(_value(group, "vulnerable_component")),
            _group_sources(group, _group_issue(group)),
            _group_location(group),
            _group_severity(group, _group_issue(group)),
            context.group_statuses.get(_text(_value(group, "group_id")), "new/pending"),
        )
        for group in sorted(retriage_groups, key=lambda item: _text(_value(item, "group_id")))
    ]
    retriage_table = _table(
        ("Group ID", "Finding/package", "Source", "Location", "Severity", "Status"),
        retriage_rows or [("None", "—", "—", "—", "—", "No added or reappeared groups")],
    )
    return "\n".join(
        [
            "## {number}. Findings Overview",
            "",
            "### Original Findings",
            "",
            original_table,
            "",
            "### Packages Added or Changed",
            "",
            "Direct manifest changes are listed individually; transitive lockfile churn is summarized.",
            "",
            "#### Direct Manifest Changes",
            "",
            direct_package_table,
            "",
            "#### Transitive Lockfile Changes",
            "",
            transitive_summary,
            "",
            "### Worker Package Execution Evidence",
            "",
            diagnostics_table,
            "",
            "### Newly Added or Reappeared Findings After Re-triage",
            "",
            retriage_table,
        ]
    )


def _new_scan_finding_rows(context: _ReportContext) -> list[tuple[str, str, str, str, str, str]]:
    """Render post-scan issues once per scanner finding, not once per identifier."""
    new_ids = set(context.new_vulnerability_identifiers)
    rows: list[tuple[str, str, str, str, str, str]] = []
    for issue in context.post_remediation_scan_issues:
        identifiers = _issue_identifiers(issue)
        if new_ids and identifiers and not new_ids.intersection(identifiers):
            continue
        rows.append(
            (
                ", ".join(identifiers) or "unidentified finding",
                _text(
                    _value(issue, "package_name") or _value(issue, "vulnerable_component"),
                    "-",
                ),
                _text(_value(issue, "source"), "-"),
                _text(_value(issue, "file_path"), "-"),
                _text(_value(issue, "severity"), "unknown"),
                "new vulnerability",
            )
        )
    if not rows and new_ids:
        rows = [
            (identifier, "-", "-", "-", "unknown", "new vulnerability")
            for identifier in sorted(new_ids)
        ]
    return rows


def _render_validation(context: _ReportContext) -> str:
    """Render deterministic validation gates and remaining work."""
    scan_status = _text(_value(context.final_full_scan_result, "status"), "not recorded")
    remaining_groups = [
        (group_id, status)
        for group_id, status in sorted(context.group_statuses.items())
        if status not in {"qa_passed", "mitigated"}
    ]
    new_scan_rows = _new_scan_finding_rows(context)
    return "\n".join(
        [
            "## {number}. Validation and Remaining Issues",
            "",
            _table(
                ("Validation area", "Result"),
                [
                    (
                        "Targeted QA coverage",
                        f"{context.targeted_qa_passed} passed / {context.targeted_qa_total} targeted groups",
                    ),
                    (
                        "QA record provenance",
                        f"{context.recorded_qa_total} records retained",
                    ),
                    (
                        "Post-remediation scanner findings",
                        f"{len(context.post_remediation_scan_issues)} findings",
                    ),
                    (
                        "Post-remediation unique identifiers",
                        f"{len(context.post_remediation_scan_identifiers)} identifiers",
                    ),
                    (
                        "New vulnerability identifiers",
                        f"{len(context.new_vulnerability_identifiers)} identifiers",
                    ),
                    ("Scanner/QA evidence", scan_status),
                    ("Consistency events", len(context.consistency_events)),
                ],
            ),
            "",
            "### Remaining or Inconclusive Groups",
            "",
            _table(
                ("Group/finding", "Status"),
                remaining_groups
                or [("None", "All original groups reached a fixed or mitigated status")],
            ),
            "",
            "### Newly Detected Findings",
            "",
            _table(
                ("Identifiers", "Package/component", "Source", "Location", "Severity", "Status"),
                new_scan_rows or [("None", "-", "-", "-", "-", "No newly detected findings")],
            ),
        ]
    )


def _render_references(context: _ReportContext) -> str:
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


def _evidence_payload(context: _ReportContext) -> dict[str, Any]:
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
            "groups_retriage_discovered": context.groups_retriage_discovered,
        },
        "group_statuses": context.group_statuses,
        "strategies": {key: _text(value) for key, value in context.group_strategies.items()},
        "reconciliation": context.triage_reconciliation,
        "errors": context.error_strings,
        "changed_files": context.changed_files,
    }


def _generate_executive_narrative(
    context: _ReportContext,
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
        _render_overview(context),
        _render_key_decisions(context),
        _render_findings(context),
        _render_validation(context),
        _render_references(context),
    ]
    if context.executive_narrative:
        sections.insert(0, "## {number}. Executive Summary\n\n" + context.executive_narrative)
    report = _render_summary(context)
    section_number = 2
    for section in sections:
        report += "\n\n" + section.replace("{number}", str(section_number), 1)
        section_number += 1
    return report + "\n"


def _resolve_report_dir(settings: AppSettings | None = None) -> Path:
    """Resolve the canonical report directory from validated settings."""
    configured = (settings or AppSettings.from_env()).remediation_report_dir
    return configured or _DEFAULT_REPORT_DIR


def _report_filename(run_id: str) -> str:
    """Return a filesystem-safe canonical report filename."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-") or "local-run"
    return f"remediation_{safe_id}.md"


def _write_report_atomic(path: Path, markdown: str) -> None:
    """Write Markdown via a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    settings = settings or AppSettings.from_env()
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
