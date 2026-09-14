"""Human-readable reporting for a Phase 5 remediation run.

The graph node renders the state available after teardown and before the graph
exits. The orchestrator finalizes and persists that report before returning a
typed result. Report facts, statuses, findings, and file changes are derived
deterministically; no model-authored prose or error telemetry is included.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import VulnerabilityGroup, VulnerabilityIssue
from remediation_engine.settings import AppSettings

from .report_context import (
    _KNOWN_ERROR_CODES,
    _KNOWN_ERROR_SOURCES,
    _SCAN_COMPLETE_STATUSES,
    _USER_FRIENDLY_STATUS_LABELS,
    ErrorRecord,
    PackageChange,
    ReportContext,
    _build_context,
    _classify_error,
    _duration_seconds,
    _error_records,
    _error_source_and_message,
    _extract_token_summary,
    _format_duration,
    _format_token_summary,
    _group_identifiers,
    _group_status,
    _group_tree,
    _inline_code_text,
    _issue_identifiers,
    _items,
    _reconcile_authoritative_statuses,
    _reconciliation_ids,
    _report_group_status,
    _scan_evidence_state,
    _text,
    _unique_texts,
    _user_friendly_status,
    _value,
)
from .report_diff import (
    _LOCKFILE_PACKAGE_RE,
    _LOCKFILE_RESOLVED_RE,
    _MANIFEST_SECTION_RE,
    _NON_PACKAGE_KEYS,
    _PACKAGE_FILE_RE,
    _PACKAGE_LINE_RE,
    _diff_block_paths,
    _diff_change_counts,
    _diff_code_change_details,
    _diff_content,
    _diff_file_paths,
    _diff_line_changes,
    _line_indent,
    _normalise_replay_text,
    _normalized_path,
    _package_changes,
    _package_name_from_resolved_url,
    _path_matches_any,
    _record_package_change,
    _replay_blocks_are_backed_by_diff,
    _replay_diff_blocks,
    _unified_diff_blocks,
)
from .report_persistence import report_filename, resolve_report_dir, write_report_atomic
from .runtime_context import get_runtime_settings
from .state import OrchestratorState
from .task_utils import effective_group_status, task_group_lineage, terminal_outcome_issues
from .trajectory_exporter import TrajectoryRecorder

log = logging.getLogger(__name__)

_ReportGroup = VulnerabilityGroup | Mapping[str, Any]
_ReportIssue = VulnerabilityIssue | Mapping[str, Any]

__all__ = [
    "generate_report",
    "run_report_node",
    "finalize_report",
    "ErrorRecord",
    "PackageChange",
    "ReportContext",
    "_KNOWN_ERROR_CODES",
    "_KNOWN_ERROR_SOURCES",
    "_SCAN_COMPLETE_STATUSES",
    "_USER_FRIENDLY_STATUS_LABELS",
    "_build_context",
    "_classify_error",
    "_duration_seconds",
    "_error_records",
    "_error_source_and_message",
    "_extract_token_summary",
    "_format_duration",
    "_format_token_summary",
    "_group_identifiers",
    "_group_status",
    "_group_tree",
    "_issue_identifiers",
    "_inline_code_text",
    "_reconcile_authoritative_statuses",
    "_report_group_status",
    "_scan_evidence_state",
    "_text",
    "_unique_texts",
    "_user_friendly_status",
    "_value",
    "_LOCKFILE_PACKAGE_RE",
    "_LOCKFILE_RESOLVED_RE",
    "_MANIFEST_SECTION_RE",
    "_NON_PACKAGE_KEYS",
    "_PACKAGE_FILE_RE",
    "_PACKAGE_LINE_RE",
    "_diff_block_paths",
    "_diff_change_counts",
    "_diff_code_change_details",
    "_diff_content",
    "_diff_file_paths",
    "_diff_line_changes",
    "_line_indent",
    "_normalise_replay_text",
    "_normalized_path",
    "_package_changes",
    "_package_name_from_resolved_url",
    "_path_matches_any",
    "_record_package_change",
    "_replay_blocks_are_backed_by_diff",
    "_replay_diff_blocks",
    "_unified_diff_blocks",
    "effective_group_status",
    "task_group_lineage",
    "terminal_outcome_issues",
]


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "data" / "reports"


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


def _group_issue(group: _ReportGroup) -> _ReportIssue | None:
    """Select the representative issue for a vulnerability group."""
    issues = _items(_value(group, "issues"))
    representative_id = _value(group, "representative_issue_id")
    if representative_id:
        for issue in issues:
            if _text(_value(issue, "id")) == _text(representative_id):
                return issue
    return issues[0] if issues else None


def _group_severity(group: _ReportGroup, issue: _ReportIssue | None) -> str:
    """Return the original scanner severity for a group."""
    return (
        _text(
            _value(issue, "severity") if issue is not None else _value(group, "severity"),
            "UNKNOWN",
        )
        .strip()
        .upper()
    )


def _group_package_or_target(context: ReportContext, group: _ReportGroup) -> str:
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


def _group_finding_package(group: _ReportGroup) -> str:
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


def _finding_identifier(group: _ReportGroup, issue: _ReportIssue | None) -> str:
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


def _report_groups(context: ReportContext) -> list[_ReportGroup]:
    """Return report groups once, preserving stable report order.

    Successful groups discovered after the initial triage can be removed from
    ``valid_groups`` once their remediation passes. Keep a lightweight group
    projection for those task-lineage records so successful discoveries remain
    visible in the report.
    """
    groups: list[_ReportGroup] = []
    seen: set[str] = set()
    for group in [*context.initial_valid_groups, *context.final_valid_groups]:
        group_id = _text(_value(group, "group_id"))
        if not group_id or group_id in seen:
            continue
        seen.add(group_id)
        groups.append(group)

    for _task_id, task in sorted(context.task_queue.items(), key=lambda item: str(item[0])):
        group_id = _text(_value(task, "parent_group_id"))
        if not group_id or group_id in seen:
            continue
        if _group_status(context.task_queue, group_id) != "qa_passed":
            continue
        package = _text(_value(task, "target_package_name")).strip()
        if not package:
            package = _text(_value(task, "parent_package_name")).strip()
        groups.append(
            {
                "group_id": group_id,
                "vulnerable_component": package or "Unspecified finding",
                "issue_type": "sca",
                "sources": ["task_queue"],
                "file_path": "package.json",
                "issues": [],
            }
        )
        seen.add(group_id)
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


def _report_group_identity(
    context: ReportContext,
    group_id: str,
    discovered_ids: set[str] | None = None,
) -> str:
    """Return the report identity for a group and its pivot descendants."""
    discovered_ids = discovered_ids or set()
    if group_id in discovered_ids:
        return group_id
    return _lineage_root_group_id(context, group_id)


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


def _follow_up_groups(context: ReportContext) -> list[tuple[str, _ReportGroup]]:
    """Return every open finding, including untriaged final-scan findings."""
    discovered_ids = set(_discovered_group_ids(context))
    initial_ids = {_text(_value(group, "group_id")) for group in context.initial_valid_groups}
    selected: dict[str, tuple[str, _ReportGroup]] = {}
    for group in _report_groups(context):
        group_id = _text(_value(group, "group_id"))
        canonical_id = _report_group_identity(context, group_id, discovered_ids)
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
    represented_raw_group_ids = {_text(_value(g, "group_id")) for g in _report_groups(context)}
    selected_ids = set(selected)
    for _task_id, task in context.task_queue.items():
        group_id = _text(_value(task, "parent_group_id"))
        if not group_id or group_id in selected_ids or group_id in represented_raw_group_ids:
            continue
        status = _report_group_status(context, group_id)
        if status in _OUTSTANDING_GROUP_STATUSES:
            selected[group_id] = (group_id, {"group_id": group_id})
    return sorted(selected.values(), key=lambda item: item[0])


def _unique_vulnerability_group_ids(context: ReportContext) -> set[str]:
    """Return unique report group IDs, including authoritative new findings."""
    discovered_ids = set(_discovered_group_ids(context))
    selected: dict[str, Any] = {}
    for group in _report_groups(context):
        group_id = _text(_value(group, "group_id"))
        if not group_id:
            continue
        canonical_id = _report_group_identity(context, group_id, discovered_ids)
        selected.setdefault(canonical_id, group)

    if (
        _scan_evidence_state(context.final_full_scan_result, context.new_vulnerability_status)
        == "complete"
    ):
        for group_id in discovered_ids:
            selected.setdefault(group_id, {"group_id": group_id})

        represented_identifiers = (
            set().union(*(_group_identifiers(group) for group in selected.values()))
            if selected
            else set()
        )
        for identifier in context.new_vulnerability_identifiers:
            if identifier.casefold() not in represented_identifiers:
                selected.setdefault(identifier, {"group_id": identifier})

    return set(selected)


def _unique_vulnerability_statuses_by_group(context: ReportContext) -> dict[str, str]:
    """Return effective statuses keyed by the unique report group IDs."""
    discovered_ids = set(_discovered_group_ids(context))
    statuses: dict[str, str] = {}
    for group in _report_groups(context):
        group_id = _text(_value(group, "group_id"))
        if not group_id:
            continue
        canonical_id = _report_group_identity(context, group_id, discovered_ids)
        statuses[canonical_id] = _report_group_status(context, canonical_id)
    for group_id in _unique_vulnerability_group_ids(context):
        statuses.setdefault(group_id, _report_group_status(context, group_id))
    return statuses


def _discovered_group_ids(context: ReportContext) -> list[str]:
    """Return groups added or reappeared by the authoritative scan.

    A final-scan reopening is evidence that an existing group remains
    unresolved, not evidence that a new vulnerability group was discovered.
    """
    return _reconciliation_ids(
        context.triage_reconciliation,
        "added",
        "new_group_ids",
        "reappeared",
        "reappeared_group_ids",
    )


def _task_ids_for_group(context: ReportContext, group_id: str) -> list[str]:
    """Return task-lineage IDs associated with a vulnerability group."""
    return [
        task_id
        for task in _group_tree(context.task_queue, group_id)
        if (task_id := _text(_value(task, "task_id")))
    ]


def _group_package_names(group: _ReportGroup) -> set[str]:
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


def _package_change_matches_group(change: PackageChange, group: _ReportGroup) -> bool:
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
    group: _ReportGroup,
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


def _attempt_evidence_candidates(
    context: ReportContext,
    summary: Any,
    metadata: Any,
) -> list[Any]:
    """Return attempt records in the order used for deterministic recovery."""
    task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
    return [
        context.attempt_snapshots.get(
            _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
        ),
        summary,
        metadata,
        context.task_queue.get(task_id),
        context.retry_diagnostics.get(task_id),
    ]


def _attempt_package_name(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
) -> str:
    """Recover the package name associated with one remediation attempt."""
    for item in _attempt_evidence_candidates(context, summary, metadata):
        for field_name in ("target_package_name", "no_fix_package_name", "package_name"):
            package = _text(_value(item, field_name)).strip()
            if package:
                return package

    group = next(
        (
            item
            for item in [*context.initial_valid_groups, *context.final_valid_groups]
            if _text(_value(item, "group_id")) == group_id
        ),
        None,
    )
    if group is not None:
        package = _group_finding_package(group).strip()
        if package and package != "Unspecified finding":
            return package

    removal_pattern = re.compile(
        r"\b(?:remove|removed|delete|deleted)\s+(?:the\s+)?(?:configured\s+|"
        r"vulnerable\s+|direct\s+)?(?:package\s+|dependency\s+)?"
        r"[`'\"]?(?P<package>[@A-Za-z0-9][A-Za-z0-9._/-]*)",
        re.IGNORECASE,
    )
    for item in _attempt_evidence_candidates(context, summary, metadata):
        for field_name in ("summary", "instruction", "final_note", "outcome"):
            match = removal_pattern.search(_text(_value(item, field_name)))
            if match:
                return match.group("package").strip("`'\"")
    return ""


def _is_package_removal_attempt(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
) -> bool:
    """Return whether an attempt is authorized to remove a package manifest entry."""
    for item in _attempt_evidence_candidates(context, summary, metadata):
        for field_name in ("qa_policy", "no_fix_stage"):
            value = _text(_value(item, field_name)).strip().casefold().replace("-", "_")
            if "package_removal" in value:
                return True

    text_parts: list[str] = []
    for item in _attempt_evidence_candidates(context, summary, metadata):
        for field_name in ("summary", "instruction", "final_note", "outcome"):
            value = _text(_value(item, field_name)).strip()
            if value:
                text_parts.extend(value.splitlines())

    removal_patterns = (
        re.compile(
            r"\b(?:remove|removed|delete|deleted)\b[^\r\n.]{0,160}"
            r"\b(?:package|dependency|manifest|lockfile)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:package|dependency|manifest|lockfile)\b[^\r\n.]{0,160}"
            r"\b(?:remove|removed|delete|deleted)\b",
            re.IGNORECASE,
        ),
    )
    for line in text_parts:
        if re.search(
            r"\b(?:do not|don't|never|without)\b[^\r\n.]{0,80}"
            r"\b(?:remov|delet|chang|edit|modif|touch)\w*\b[^\r\n.]{0,80}"
            r"\b(?:package|dependenc\w*|manifest\w*|lockfile\w*)\b",
            line,
            re.I,
        ):
            continue
        if any(pattern.search(line) for pattern in removal_patterns):
            return True
    return False


def _manifest_removal_files(files: Sequence[Any]) -> list[str]:
    """Return normalized package manifest paths from attempt file evidence."""
    return _unique_texts(
        _normalized_path(path) for path in files if _PACKAGE_FILE_RE.search(_normalized_path(path))
    )


def _manifest_removal_lines(path: str, package: str, content: str) -> list[str]:
    """Extract compact pre-removal lines for one package manifest."""
    lines = content.splitlines()
    package_pattern = re.compile(rf'^\s*"{re.escape(package)}"\s*:')
    if Path(path).name.casefold() != "package-lock.json":
        return [line for line in lines if package_pattern.search(line)]

    node_pattern = re.compile(rf'^\s*"node_modules/{re.escape(package)}"\s*:\s*\{{')
    result: list[str] = []
    consumed_until = -1
    for index, line in enumerate(lines):
        if index <= consumed_until:
            continue
        node_match = node_pattern.match(line)
        if node_match:
            indent = line[: len(line) - len(line.lstrip())]
            block = [line]
            end_index = index
            for candidate_index in range(index + 1, len(lines)):
                candidate = lines[candidate_index]
                block.append(candidate)
                end_index = candidate_index
                if re.match(rf"^{re.escape(indent)}\}},?\s*$", candidate):
                    break
            result.extend(block)
            consumed_until = end_index
        elif package_pattern.search(line):
            result.append(line)
    return result


def _manifest_removal_diff_blocks(
    metadata: Any,
    package: str,
    files: Sequence[Any],
) -> list[str]:
    """Build compact manifest-removal diff blocks from the attempt baseline."""
    replay_plan = _value(metadata, "replay_plan")
    snapshots = _value(replay_plan, "pre_attempt_snapshots")
    if not package or not isinstance(snapshots, Mapping):
        return []

    candidate_paths = _manifest_removal_files(files)
    if not candidate_paths:
        candidate_paths = _manifest_removal_files(list(snapshots))

    snapshot_by_path = {
        _normalized_path(path): _text(content)
        for path, content in snapshots.items()
        if _normalized_path(path)
    }
    blocks: list[str] = []
    for path in candidate_paths:
        content = snapshot_by_path.get(path)
        if not content:
            continue
        removed_lines = _manifest_removal_lines(path, package, content)
        if not removed_lines:
            continue
        blocks.append(
            "\n".join(
                [
                    f"--- a/{path}",
                    f"+++ b/{path}",
                    f"@@ package removal: {package}",
                    *(f"-{line}" for line in removed_lines),
                ]
            )
        )
    return blocks


def _attempt_diff_blocks(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    files: Sequence[Any],
) -> list[str]:
    """Return exact attempt-scoped diff blocks, including removal manifests."""
    # An explicit empty file set means the attempt produced no file-level
    # evidence. Do not reinterpret it as permission to render the entire run
    # diff (or to synthesize a manifest removal from an old baseline).
    if not files:
        return []
    package_removal = _is_package_removal_attempt(
        context,
        group_id,
        summary,
        metadata,
    )
    source_files = [path for path in files if not _PACKAGE_FILE_RE.search(_text(path))]
    selected_files = files if package_removal else source_files
    replay_blocks = _replay_diff_blocks(
        metadata,
        selected_files,
        include_package_files=package_removal,
    )
    blocks = replay_blocks or _unified_diff_blocks(
        context.diff,
        selected_files,
        include_package_files=package_removal,
    )
    if not package_removal:
        return blocks

    manifest_files = _manifest_removal_files(files)
    manifest_blocks = _unified_diff_blocks(
        context.diff,
        manifest_files,
        include_package_files=True,
    )
    represented_paths = _diff_block_paths(blocks)
    blocks.extend(
        block
        for block in manifest_blocks
        if _diff_block_paths([block]).isdisjoint(represented_paths)
    )
    if not manifest_blocks:
        package = _attempt_package_name(
            context,
            group_id,
            summary,
            metadata,
        )
        fallback_blocks = _manifest_removal_diff_blocks(metadata, package, manifest_files)
        blocks.extend(
            block
            for block in fallback_blocks
            if _diff_block_paths([block]).isdisjoint(represented_paths)
        )
    return _unique_texts(blocks)


def _final_attempt_diff_blocks(
    context: ReportContext,
    summary: Any,
    metadata: Any,
    files: Sequence[Any],
) -> list[str]:
    """Return source diff blocks proven to belong to one successful attempt.

    Replay plans provide attempt-level isolation when several workers edit the
    same file. A replay block is emitted only when the corresponding change is
    also represented in the final diff, preventing unrelated workspace edits
    from being attributed to this attempt.
    """
    source_files = [
        _normalized_path(path)
        for path in files
        if _normalized_path(path) and not _PACKAGE_FILE_RE.search(_normalized_path(path))
    ]
    if not source_files:
        return []

    replay_blocks = _replay_diff_blocks(
        metadata,
        source_files,
        include_package_files=False,
    )
    final_blocks = _unified_diff_blocks(
        context.diff,
        source_files,
        include_package_files=False,
    )
    if not replay_blocks:
        return final_blocks
    if _replay_blocks_are_backed_by_diff(context.diff, replay_blocks):
        return replay_blocks

    # Current graph runs always emit source changes for accepted workaround
    # attempts. Keep the old sparse-fixture behavior only when no source diff
    # exists at all; never display a stale replay hunk beside an unrelated
    # final source change.
    has_source_final_diff = bool(
        _unified_diff_blocks(context.diff, None, include_package_files=False)
    )
    if not has_source_final_diff and not final_blocks:
        return replay_blocks
    return []


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
    candidates = [snapshot, summary, metadata, task, diagnostic]

    group = next(
        (
            item
            for item in [*context.initial_valid_groups, *context.final_valid_groups]
            if _text(_value(item, "group_id")) == group_id
        ),
        None,
    )

    package = ""
    for item in candidates:
        package = _text(_value(item, "target_package_name")).strip()
        if package:
            break
    if not package and group is not None:
        package = _group_finding_package(group).strip()
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
        for item in candidates:
            instruction = _text(_value(item, "instruction"))
            version_match = re.search(
                r"\bversion\s+[\"'`]?v?(?P<version>[0-9][^\"'`\s,;)]+)",
                instruction,
                re.IGNORECASE,
            )
            if version_match:
                selected_version = version_match.group("version").rstrip(".")
                break
    if not selected_version:
        return None

    previous_version = ""
    for item in (task, diagnostic):
        previous_version = _text(_value(item, "parent_package_version")).strip()
        if previous_version:
            break
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
        file_match = re.search(
            r"(?P<file>package\.json|package-lock\.json|npm-shrinkwrap\.json|"
            r"yarn\.lock|pnpm-lock\.yaml)\b",
            instruction,
            re.IGNORECASE,
        )
        if file_match:
            file_path = file_match.group("file")
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
    if status in {"success", "qa_passed"}:
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
    metadata_change = _attempt_package_metadata(
        context,
        group_id,
        summary,
        metadata,
        attempt_id,
    )
    if metadata_change is not None:
        package, previous, selected, mechanism, file_path = metadata_change
        if not files or not file_path or _path_matches_any(file_path, files):
            return [
                (
                    PackageChange(package, previous, selected, file_path, "direct", mechanism),
                    mechanism,
                )
            ]

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
    return []


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
    attempt_kind = _attempt_kind(
        context,
        group_id,
        summary,
        metadata,
    )
    source_files = [path for path in files if not _PACKAGE_FILE_RE.search(path)]
    if not explicit_no_files and attempt_kind == "Code Workaround" and not source_files:
        # Legacy summaries sometimes omit ``changed_files``. Only infer a
        # source path when the final patch contains exactly one source file;
        # never broaden an attempt to every file in a shared patch.
        source_paths = [
            path for path in _diff_file_paths(context.diff) if not _PACKAGE_FILE_RE.search(path)
        ]
        if len(source_paths) == 1:
            files.append(source_paths[0])
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
        if status.casefold() in {"success", "qa_passed"}:
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
    if status.casefold() in {"success", "qa_passed"}:
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


def _attempt_qa_verdict(
    context: ReportContext,
    summary: Any,
    metadata: Any,
) -> bool | None:
    """Return the authoritative QA verdict for one committed attempt.

    Attempt envelopes are preferred; the task-keyed evaluation projection
    supplies the same verdict when the envelope does not carry an evaluation.
    ``None`` means neither projection contains a verdict, and remains distinct
    from ``False`` so worker success cannot override an explicit QA failure.
    """
    attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
    qa_result = context.qa_results.get(attempt_id) if attempt_id else None
    evaluation = _value(qa_result, "evaluation")
    if evaluation is None:
        task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
        evaluation = context.qa_evaluations.get(task_id) if task_id else None
    if evaluation is None:
        return None
    return bool(_value(evaluation, "passed", False))


def _attempt_has_qa_evidence(
    context: ReportContext,
    summary: Any,
    metadata: Any,
) -> bool:
    """Return whether QA evidence is correlated to this attempt or task."""
    attempt_id = _text(_value(summary, "attempt_id")) or _text(_value(metadata, "attempt_id"))
    if attempt_id and attempt_id in context.qa_results:
        return True
    task_id = _text(_value(summary, "task_id")) or _text(_value(metadata, "task_id"))
    return bool(task_id and task_id in context.qa_evaluations)


def _successful_attempts_for_group(
    context: ReportContext,
    group_id: str,
) -> list[tuple[Any, Any]]:
    """Return successful worker/action evidence for a group in stable order."""
    records: list[tuple[Any, Any]] = []
    for summary, metadata in _attempt_records_for_group(context, group_id):
        qa_verdict = _attempt_qa_verdict(context, summary, metadata)
        if qa_verdict is not None:
            if qa_verdict:
                records.append((summary, metadata))
            continue
        # A state with any QA evidence but no verdict for this attempt is
        # incomplete. Fail closed instead of promoting a worker-only success
        # into the successful-remediation table.
        if _attempt_has_qa_evidence(context, summary, metadata) or context.qa_results:
            continue
        result_status = _text(_value(metadata, "status")).casefold()
        summary_status = _text(_value(summary, "status")).casefold()
        diagnostics = _value(metadata, "execution_diagnostics")
        if (
            result_status in {"success", "qa_passed"}
            or summary_status in {"success", "qa_passed"}
            or bool(_value(diagnostics, "validation_passed", False))
        ):
            records.append((summary, metadata))
    return records


def _attempt_package_removal_text(
    context: ReportContext,
    group_id: str,
    summary: Any,
    metadata: Any,
    files: Sequence[Any],
    explicit_no_files: bool,
) -> str:
    """Describe an evidenced package-removal action for one attempt."""
    if explicit_no_files or not _is_package_removal_attempt(
        context,
        group_id,
        summary,
        metadata,
    ):
        return ""

    package = (
        _attempt_package_name(context, group_id, summary, metadata) or "the vulnerable package"
    )
    manifest_files = _manifest_removal_files(files)
    manifest_names = {Path(path).name.casefold() for path in manifest_files}
    if {"package.json", "package-lock.json"}.issubset(manifest_names):
        return f"Removed {package} from package.json and synchronized package-lock.json."
    if "package.json" in manifest_names:
        return f"Removed {package} from package.json."
    if "package-lock.json" in manifest_names:
        return f"Removed {package} from package-lock.json."
    if manifest_files:
        return f"Removed {package} from the dependency manifest."
    return f"Removed {package} from the dependency manifests."


def _attempted_fixes_for_group(context: ReportContext, group_id: str) -> str:
    """Render concise, numbered attempt history for one follow-up finding."""
    entries: list[str] = []
    seen_entries: set[tuple[str, str, str, str]] = set()
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
        package_changes = (
            _attempt_package_changes(
                context,
                group_id,
                summary,
                metadata,
                attempt_id=attempt_id,
                files=files,
                explicit_no_files=explicit_no_files,
            )
            if kind == "Version Update"
            else []
        )
        change_lines: list[str] = []
        package_removal_text = _attempt_package_removal_text(
            context,
            group_id,
            summary,
            metadata,
            files,
            explicit_no_files,
        )
        if package_changes:
            for change, mechanism in package_changes:
                change_lines.append(_package_attempt_text(change, mechanism))
        elif package_removal_text:
            change_lines.append(package_removal_text)
            source_files = [path for path in files if not _PACKAGE_FILE_RE.search(_text(path))]
            if source_files:
                change_lines.append(f"Attempted a code workaround in {', '.join(source_files)}.")
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
            _attempt_diff_blocks(context, group_id, summary, metadata, files)
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
        key = (
            attempt_id,
            kind.casefold(),
            outcome_status.casefold(),
            "\n".join(change_lines),
        )
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


def _render_summary(context: ReportContext) -> str:
    """Render the eight-metric, user-facing run summary."""
    unique_statuses = _unique_vulnerability_statuses_by_group(context)
    fixed = sum(status == "qa_passed" for status in unique_statuses.values())
    follow_up = sum(
        _report_group_status(context, group_id) in _OUTSTANDING_GROUP_STATUSES
        for group_id, _ in _follow_up_groups(context)
    )
    total_vulnerability_groups = len(unique_statuses)
    actionable = max(total_vulnerability_groups, fixed + follow_up)
    sentence = (
        f"**{fixed} of {actionable}** vulnerability groups were successfully remediated. "
        f"**{follow_up} require follow-up review.**"
    )
    metrics = [
        ("Run ID", context.run_id),
        ("Total findings (CVEs and GHSAs)", context.original_scanner_findings),
        ("Total vulnerability groups", total_vulnerability_groups),
        ("Successfully remediated vulnerability groups", fixed),
        ("Vulnerability groups requiring follow-up", follow_up),
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


def _scan_issue_for_identifier(context: ReportContext, identifier: str) -> _ReportIssue | None:
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
    group: _ReportGroup,
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
    successful_attempts = _successful_attempts_for_group(context, group_id)
    if not package_text and not context.diff.strip():
        # A successful discovered group may no longer have a final workspace
        # diff after teardown. Recover the latest QA-accepted package change
        # from committed attempt metadata only when no final diff exists.
        for summary, metadata in reversed(successful_attempts):
            if _attempt_kind(context, group_id, summary, metadata) != "Version Update":
                continue
            attempt_id = _text(_value(summary, "attempt_id")) or _text(
                _value(metadata, "attempt_id")
            )
            attempt_files, explicit_no_files = _attempt_files(
                context,
                group_id,
                summary,
                metadata,
                attempt_id=attempt_id,
            )
            for change, mechanism in _attempt_package_changes(
                context,
                group_id,
                summary,
                metadata,
                attempt_id=attempt_id,
                files=attempt_files,
                explicit_no_files=explicit_no_files,
            ):
                package_text.append(_package_transition(change, mechanism))
                files.extend(_package_change_files(change))
            if package_text:
                break

    generic_code_summaries = {
        "validated remediation recorded.",
        "no specific code change recorded.",
        "no validated code change was applied",
        "source changes applied.",
    }
    for summary, metadata in successful_attempts:
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
        attempt_blocks = _final_attempt_diff_blocks(
            context,
            summary,
            metadata,
            attempt_files,
        )
        if not attempt_blocks:
            # A replay plan that is not represented in the final patch is
            # stale evidence. Do not promote its summary or file list into a
            # successful remediation row.
            continue
        attempt_paths = sorted(_diff_block_paths(attempt_blocks))
        files.extend(attempt_paths)
        details = _attempt_code_details(
            context,
            group_id,
            summary,
            metadata,
            attempt_id=attempt_id,
            files=attempt_paths,
            include_package_diff=False,
            include_source_diff=False,
            include_metadata_fallback=False,
        )
        detail_summary = _sanitize_report_outcome(_attempt_change_summary(summary, details))
        exact_details = _diff_code_change_details("\n".join(attempt_blocks), attempt_paths)
        summary_parts = []
        if detail_summary and detail_summary.casefold() not in generic_code_summaries:
            summary_parts.append(detail_summary)
        summary_parts.extend(exact_details)
        if summary_parts:
            code_summaries.append("; ".join(_unique_texts(summary_parts)))
        diff_blocks.extend(attempt_blocks)

    files = _unique_texts(files)
    if not files and not context.diff.strip() and len(context.initial_valid_groups) == 1:
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
        changes.append(
            "No validated change in emitted patch"
            if context.diff.strip()
            else "Validated remediation recorded"
        )
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
        if _report_group_status(context, group_id) in _OUTSTANDING_GROUP_STATUSES
    ]
    groups.sort(
        key=lambda item: (
            status_order.get(_report_group_status(context, item[0]), 99),
            item[0],
        )
    )

    lines = ["## 2. Follow up Actions", ""]
    if not groups:
        lines.append("No follow-up actions are required.")
        return "\n".join(lines)

    for index, (group_id, group) in enumerate(groups):
        issue = _group_issue(group)
        status = _report_group_status(context, group_id)
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
                    lines.extend(["", "```diff"])
                    in_diff = True
                elif in_diff and stripped == "```":
                    lines.extend(["```", ""])
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
    discovered_ids = set(_discovered_group_ids(context))
    initial_ids = {_text(_value(group, "group_id")) for group in context.initial_valid_groups}
    selected_groups: dict[str, Any] = {}
    semantic_aliases: dict[tuple[str, str], str] = {}
    for group in _report_groups(context):
        group_id = _text(_value(group, "group_id"))
        if _report_group_status(context, group_id) != "qa_passed":
            continue
        # Group IDs can be regenerated when triage rehydrates a finding. Use
        # the user-visible finding/package identity as the final dedupe key so
        # the same successful package (for example sanitize-html) is not
        # rendered twice under two equivalent group IDs.
        semantic_key = (
            _finding_identifier(group, _group_issue(group)).casefold(),
            _group_finding_package(group).casefold(),
        )
        canonical_id = _report_group_identity(context, group_id, discovered_ids)
        selection_key = semantic_aliases.setdefault(semantic_key, canonical_id)
        current = selected_groups.get(selection_key)
        if current is None:
            selected_groups[selection_key] = group
            continue
        current_id = _text(_value(current, "group_id"))
        if group_id in initial_ids and current_id not in initial_ids:
            selected_groups[selection_key] = group
    groups = list(selected_groups.values())
    lines = ["## 3. Successful Remediations", ""]
    if not groups:
        lines.append("No successful remediations were produced during this run.")
        return "\n".join(lines)

    rows: list[tuple[str, ...]] = []
    seen_row_keys: set[tuple[str, str, str]] = set()
    code_evidence: list[tuple[str, str, str, str, list[str], list[str]]] = []
    seen_code_evidence: set[tuple[str, str, str]] = set()
    for group in groups:
        issue = _group_issue(group)
        change_text, files, code_summaries, diff_blocks = _successful_remediation_evidence(
            context,
            group,
        )
        row = (
            _finding_identifier(group, issue),
            _group_finding_package(group),
            _group_severity(group, issue),
            change_text,
            ", ".join(files) or "Not recorded",
        )
        row_key = (row[0].casefold(), row[1].casefold(), row[3].casefold())
        if row_key in seen_row_keys:
            continue
        seen_row_keys.add(row_key)
        rows.append(row)
        if code_summaries or diff_blocks:
            code_key = (
                row[0].casefold(),
                row[1].casefold(),
                "\n".join(diff_blocks).casefold(),
            )
            if code_key in seen_code_evidence:
                continue
            seen_code_evidence.add(code_key)
            code_evidence.append(
                (
                    row[0],
                    row[1],
                    row[2],
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


def run_report_node(state: OrchestratorState) -> OrchestratorState:
    """Render a preliminary report and return its state projection.

    Args:
        state: Current orchestrator state after teardown.

    Returns:
        A partial :class:`OrchestratorState` containing report fields.  On
        rendering failure, the projection includes a report-node error while
        preserving the remediation graph's failure isolation.
    """
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
