"""Context, model access, and bounded execution for specialist agents.

This module owns ephemeral model context, provider construction, and the
bounded tool loop shared by remediation workers. Durable task and attempt state
remains owned by the orchestration state layer.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from remediation_engine.contracts.schemas import (
    ScratchpadEntry,
    ScratchpadScope,
    WorkaroundExecutionPhase,
)
from remediation_engine.runtime.telemetry import invoke_with_trajectory
from remediation_engine.settings import (
    MAX_SUBAGENT_TOOL_CALL_ROUNDS,
    AppSettings,
    LunaReasoningEffort,
    openai_api_key_kwargs,
)
from remediation_engine.tools.identity import normalize_python_package_name

DEFAULT_COMPACTION_INTERVAL = 4
MAX_SCRATCHPAD_CHARS = 4000

_SCRATCHPAD_MARKER = "remediation_engine_scratchpad"
_MAX_FACT_CHARS = 520
_MAX_FINDINGS_PER_ENTRY = 6
_MAX_FILES_PER_ENTRY = 8
_COMPACTION_TOOLS = frozenset(
    {
        "read_workspace_file",
        "search_codebase_pattern",
        "read_web_page",
        "inspect_ast_symbol",
        "search_web",
        "read_repository_map",
        "list_changed_files",
        "generate_workspace_diff",
        "read_file_context",
        "query_qa_logs",
    }
)
_CRITICAL_SCRATCHPAD_TOOLS = frozenset(
    {
        "record_plan",
        "remove_no_fix_dependency",
        "remove_no_fix_python_dependency",
        "modify_and_validate_python_dependency",
        "deterministic_apply_edit_set",
        "validate_workaround",
        "validate_python_workaround",
        "run_targeted_python_test",
        "validate_python_syntax",
        "emit_qa_evaluation",
    }
)

PHASE_TOOL_REGISTRY: dict[WorkaroundExecutionPhase, frozenset[str]] = {
    WorkaroundExecutionPhase.INVESTIGATE: frozenset(
        {
            "read_repository_map",
            "read_workspace_file",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "search_web",
            "read_web_page",
            "record_plan",
        }
    ),
    WorkaroundExecutionPhase.PLAN: frozenset(
        {
            "record_plan",
            "read_workspace_file",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "search_web",
            "read_web_page",
        }
    ),
    WorkaroundExecutionPhase.EXECUTE: frozenset(
        {
            "deterministic_apply_edit_set",
            "revert_workspace_file",
            "read_workspace_file",
            "search_codebase_pattern",
            "modify_and_validate_python_dependency",
            "remove_no_fix_dependency",
            "remove_no_fix_python_dependency",
        }
    ),
    WorkaroundExecutionPhase.VALIDATE: frozenset(
        {
            "read_repository_map",
            "validate_workaround",
            "validate_python_workaround",
            "validate_python_syntax",
            "run_targeted_python_test",
            "record_targeted_test_substitution",
            "read_workspace_file",
            "remove_no_fix_dependency",
            "remove_no_fix_python_dependency",
        }
    ),
}

_PHASE_PROMPTS: dict[WorkaroundExecutionPhase, str] = {
    WorkaroundExecutionPhase.INVESTIGATE: (
        "INVESTIGATE: Inspect the local repository map, files, searches, and AST first; "
        "collect exact files, symbols, and anchors. Use web evidence only after local "
        "evidence is insufficient. Do not edit."
    ),
    WorkaroundExecutionPhase.PLAN: (
        "PLAN: Finish one complete record_plan call with the security invariant, causal "
        "hypothesis, evidence source, affected files and symbols, and every exact "
        "replacement or package-removal intent. Continue only the allowed research needed "
        "to make that plan complete."
    ),
    WorkaroundExecutionPhase.EXECUTE: (
        "EXECUTE: Use the exact recorded replacements in one atomic edit set, or the "
        "configured package-removal tool. Use reads and searches only to verify anchors. "
        "A successful mutation must lead to cumulative validation."
    ),
    WorkaroundExecutionPhase.VALIDATE: (
        "VALIDATE: Call validate_workaround with the complete modified-file set and "
        "separate source runtime-smoke and targeted-test paths. If deterministic preflight "
        "rejects a path, use read_repository_map or read_workspace_file to resolve valid "
        "repository-relative paths, then retry validate_workaround. After an infrastructure-only "
        "targeted-test failure, inspect and register one valid substitution. Do not edit or "
        "re-plan. For a NO_FIX package-removal plan with source replacements, both "
        "remove_no_fix_dependency and deterministic_apply_edit_set must have succeeded before "
        "validation; either operation may happen first, and the removal tool remains available "
        "to complete that requirement. Once all required operations are complete and the inputs "
        "are valid, your next action must be validate_workaround."
    ),
}


def normalize_phase(
    phase: WorkaroundExecutionPhase | str | None,
) -> WorkaroundExecutionPhase:
    """Normalize a worker phase and fail closed for unknown values.

    Args:
        phase: A phase enum, its serialized value, or ``None``.

    Returns:
        The normalized execution phase. Missing values default to
        :attr:`WorkaroundExecutionPhase.INVESTIGATE`.

    Raises:
        ValueError: If ``phase`` is not a known workaround execution phase.
    """
    if phase is None or (isinstance(phase, str) and not phase.strip()):
        return WorkaroundExecutionPhase.INVESTIGATE
    if isinstance(phase, WorkaroundExecutionPhase):
        return phase
    if isinstance(phase, str):
        candidate = phase.strip()
        try:
            return WorkaroundExecutionPhase(candidate)
        except ValueError:
            try:
                return WorkaroundExecutionPhase[candidate.upper()]
            except KeyError as exc:
                raise ValueError(f"Invalid workaround execution phase: {phase!r}") from exc
    raise ValueError(f"Invalid workaround execution phase: {phase!r}")


def get_tools_for_phase(
    phase: WorkaroundExecutionPhase | str | None, all_tools: Sequence[Any]
) -> list[Any]:
    """Filter a flat toolbelt to the tools allowed in one execution phase.

    Args:
        phase: Requested worker execution phase.
        all_tools: Flat tool sequence returned by the production builder.

    Returns:
        Tools in their original order. Tools absent from the builder are simply
        omitted, including the optional package-removal tool.
    """
    normalized = normalize_phase(phase)
    allowed = PHASE_TOOL_REGISTRY[normalized]
    return [tool for tool in all_tools if str(getattr(tool, "name", "")) in allowed]


def get_phase_prompt(
    phase: WorkaroundExecutionPhase | str | None,
    base_context: str = "",
) -> str:
    """Return the short transition instruction for a worker phase.

    Args:
        phase: Requested worker execution phase.
        base_context: Optional context to place before the phase fragment.

    Returns:
        A deterministic prompt containing the phase-specific instruction.
    """
    fragment = _PHASE_PROMPTS[normalize_phase(phase)]
    context = str(base_context or "").strip()
    return f"{context}\n\n{fragment}" if context else fragment


def _normalize_path(value: Any) -> str:
    """Normalize a workspace-relative path for scratchpad facts."""
    return str(value or "").replace("\\", "/").strip().lstrip("/")


def _clean_text(value: Any, limit: int = _MAX_FACT_CHARS) -> str:
    """Return bounded single-line text with obvious credentials redacted."""
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 14)].rstrip() + "... (truncated)"
    return text


def _first_meaningful_line(value: Any, limit: int = _MAX_FACT_CHARS) -> str:
    """Extract one bounded non-empty line from tool output."""
    for raw_line in str(value or "").splitlines():
        line = _clean_text(raw_line, limit)
        if line:
            return line
    return ""


def _bounded_lines(value: Any, limit: int = 3) -> list[str]:
    """Extract a few bounded meaningful output lines."""
    result: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = _clean_text(raw_line)
        if line and line not in result:
            result.append(line)
        if len(result) >= limit:
            break
    return result


def _arg(args: Mapping[str, Any], *names: str) -> Any:
    """Return the first present argument value."""
    for name in names:
        if name in args and args[name] not in (None, ""):
            return args[name]
    return None


def _json_payload(content: str) -> dict[str, Any] | None:
    """Parse a JSON object following a tool result's ``JSON:`` marker."""
    if "JSON:" not in content:
        return None
    try:
        payload = json.loads(content.split("JSON:", 1)[1].strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _unique(values: Sequence[str], seen: set[str], *, normalize_paths: bool = False) -> list[str]:
    """Deduplicate facts against both the current event and prior entries."""
    result: list[str] = []
    for value in values:
        normalized = _normalize_path(value) if normalize_paths else _clean_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


class ScratchpadMemory:
    """Deterministic, bounded memory for one specialist invocation."""

    def __init__(self, scope: ScratchpadScope | str = ScratchpadScope.WORKAROUND) -> None:
        """Initialize empty scratchpad history and deduplication indexes.

        Args:
            scope: Specialist scope whose entries this memory stores.
        """
        self.scope = scope if isinstance(scope, ScratchpadScope) else ScratchpadScope(scope)
        self._entries: list[ScratchpadEntry] = []
        self._seen_findings: set[str] = set()
        self._seen_files: set[str] = set()
        self._seen_file_symbols: set[tuple[str, str]] = set()
        self._seen_plan_summaries: set[str] = set()
        self._seen_validation_outcomes: set[str] = set()

    @property
    def entries(self) -> tuple[ScratchpadEntry, ...]:
        """Return the complete chronological entry history."""
        return tuple(self._entries)

    def update_from_tool_event(
        self,
        event: Any,
        phase: WorkaroundExecutionPhase | str | None,
        round_number: int,
        *,
        scope: ScratchpadScope | str | None = None,
    ) -> None:
        """Extract bounded facts from one tool event.

        Args:
            event: Tool-event-like object with ``name``, ``args``, and
                ``content`` attributes.
            phase: Workaround execution phase, or ``None`` for QA entries.
            round_number: One-based model loop round.
            scope: Optional entry scope override. The memory's scope is used
                when omitted.

        Returns:
            None. The event is retained only when it adds a deterministic fact.
        """
        event_name = str(getattr(event, "name", "") or "synthetic")
        args = getattr(event, "args", {}) or {}
        if not isinstance(args, Mapping):
            args = {}
        content = str(getattr(event, "content", "") or "")
        entry_scope = (
            self.scope
            if scope is None
            else (scope if isinstance(scope, ScratchpadScope) else ScratchpadScope(scope))
        )
        normalized_phase = (
            normalize_phase(phase) if entry_scope == ScratchpadScope.WORKAROUND else None
        )
        key_findings: list[str] = []
        files_inspected: list[str] = []
        plan_summary = ""
        validation_outcome = ""
        critical_outcome = ""
        pair_seen = False
        file_path = ""
        symbol = ""

        if event_name == "read_repository_map":
            status = _first_meaningful_line(content)
            if status:
                key_findings.append(f"repository map: {status}")
        elif event_name == "list_changed_files":
            reported_files = [
                _normalize_path(line.strip()[2:])
                for line in content.splitlines()
                if line.strip().startswith(("-", "*")) and line.strip()[1:].strip()
            ]
            files_inspected.extend(value for value in reported_files if value)
            key_findings.append(
                f"changed files: {len(reported_files)}"
                + (
                    f" ({', '.join(reported_files[:_MAX_FILES_PER_ENTRY])})"
                    if reported_files
                    else ""
                )
            )
        elif event_name == "generate_workspace_diff":
            diff_paths: list[str] = []
            for raw_line in content.splitlines():
                line = raw_line.strip()
                path: str | None = None
                if line.startswith("diff --git "):
                    match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
                    if match:
                        path = match.group(2)
                elif line.startswith("+++ "):
                    path = line[4:].strip()
                    if path.startswith("b/"):
                        path = path[2:]
                elif line.startswith("--- "):
                    path = line[4:].strip()
                    if path.startswith("a/"):
                        path = path[2:]
                if path and path != "/dev/null":
                    normalized_path = _normalize_path(path.split("\t", 1)[0])
                    if normalized_path and normalized_path not in diff_paths:
                        diff_paths.append(normalized_path)
            files_inspected.extend(diff_paths)
            key_findings.append(
                f"diff files: {len(diff_paths)}"
                + (f" ({', '.join(diff_paths[:_MAX_FILES_PER_ENTRY])})" if diff_paths else "")
            )
        elif event_name in {"read_workspace_file", "read_file_context"}:
            file_path = _normalize_path(_arg(args, "file_path", "path"))
            if file_path:
                files_inspected.append(file_path)
                start = _arg(args, "start_line", "line_start", "from_line")
                end = _arg(args, "end_line", "line_end", "to_line")
                range_text = ""
                if start is not None or end is not None:
                    range_text = f" lines {start or 1}-{end or 'end'}"
                excerpt = _first_meaningful_line(content, 260)
                key_findings.append(
                    f"read {file_path}{range_text}" + (f": {excerpt}" if excerpt else "")
                )
        elif event_name == "search_codebase_pattern":
            directory = _normalize_path(_arg(args, "directory", "path", "search_directory")) or "."
            pattern = _clean_text(_arg(args, "search_pattern", "pattern", "query"), 220)
            key_findings.append(f"search {directory} for {pattern or '[pattern omitted]'}")
            key_findings.extend(f"match: {line}" for line in _bounded_lines(content, 3))
        elif event_name == "inspect_ast_symbol":
            file_path = _normalize_path(_arg(args, "file_path", "path"))
            symbol = _clean_text(_arg(args, "symbol_name", "symbol"), 180)
            if file_path:
                files_inspected.append(file_path)
            pair_seen = bool(
                file_path and symbol and (file_path, symbol) in self._seen_file_symbols
            )
            if file_path and symbol and not pair_seen:
                self._seen_file_symbols.add((file_path, symbol))
                key_findings.append(
                    f"AST {file_path}::{symbol}"
                    f" node={_clean_text(_arg(args, 'node_type', 'expected_node_type') or 'unknown', 100)}"
                    f" location={_clean_text(_arg(args, 'line_number', 'line', 'location') or _first_meaningful_line(content), 180)}"
                )
            elif (file_path or symbol) and not pair_seen:
                key_findings.append(
                    f"AST {file_path or '[file omitted]'}::{symbol or '[symbol omitted]'}"
                )
        elif event_name == "query_qa_logs":
            phase_name = _clean_text(_arg(args, "log_type", "phase") or "unknown", 80)
            view = _clean_text(_arg(args, "view") or "summary", 80)
            filter_pattern = _clean_text(_arg(args, "filter_pattern") or "", 160)
            diagnostic = _first_meaningful_line(content, 320)
            key_findings.append(
                f"QA logs phase={phase_name} view={view}"
                + (f" filter={filter_pattern}" if filter_pattern else "")
                + (f": {diagnostic}" if diagnostic else "")
            )
        elif event_name == "emit_qa_evaluation":
            semantic = args.get("semantic_security_review")
            attribution = args.get("test_attribution")
            semantic_verdict = (
                semantic.get("verdict", "")
                if isinstance(semantic, Mapping)
                else getattr(semantic, "verdict", "")
            )
            attribution_verdict = (
                attribution.get("verdict", "")
                if isinstance(attribution, Mapping)
                else getattr(attribution, "verdict", "")
            )
            critical_outcome = _clean_text(
                f"task_id={args.get('task_id', '')}; passed={args.get('passed', '')}; "
                f"category={args.get('failure_category', '')}; "
                f"retry_feedback={args.get('retry_feedback', '')}; "
                f"semantic_review={semantic_verdict}; "
                f"test_attribution={attribution_verdict}",
                1250,
            )
        elif event_name in {"search_web", "read_web_page"}:
            query_or_url = _clean_text(
                _arg(args, "query", "url", "page_url", "search_query") or "", 260
            )
            label = "web search" if event_name == "search_web" else "web page"
            if query_or_url:
                key_findings.append(f"{label}: {query_or_url}")
            finding = _first_meaningful_line(content, 320)
            if finding:
                key_findings.append(f"{label} finding: {finding}")
        elif event_name == "record_plan":
            affected_files = [
                _normalize_path(value) for value in args.get("affected_files", []) or []
            ]
            files_inspected.extend(value for value in affected_files if value)
            replacements = args.get("planned_replacements", []) or []
            replacement_files = []
            if isinstance(replacements, list):
                replacement_files = [
                    _normalize_path(item.get("file_path"))
                    for item in replacements
                    if isinstance(item, Mapping) and item.get("file_path")
                ]
            package_intent = (
                "package removal"
                if args.get("package_removal_requested")
                else "source replacements"
            )
            plan_summary = _clean_text(
                f"invariant={args.get('security_invariant', '')}; "
                f"cause={args.get('causal_hypothesis', '')}; "
                f"evidence={args.get('evidence_source', '')}; "
                f"targets={','.join(value for value in affected_files + replacement_files if value)}; "
                f"intent={package_intent}",
                1000,
            )
        elif event_name == "deterministic_apply_edit_set":
            payload = _json_payload(content) or {}
            affected = payload.get("affected_files", []) or []
            if not affected and isinstance(args.get("replacements"), list):
                affected = [
                    item.get("file_path")
                    for item in args["replacements"]
                    if isinstance(item, Mapping)
                ]
            files_inspected.extend(_normalize_path(value) for value in affected if value)
            status = _first_meaningful_line(content)
            summary = (
                f" files={','.join(_normalize_path(value) for value in affected if value)}"
                if affected
                else ""
            )
            key_findings.append(_clean_text(f"atomic edit: {status}{summary}", 500))
        elif event_name == "modify_and_validate_maven_dependency":
            package = _clean_text(_arg(args, "package_name"), 180)
            version = _clean_text(_arg(args, "target_version"), 100)
            dependency_type = _clean_text(_arg(args, "dependency_type"), 80)
            manifest = _normalize_path(_arg(args, "manifest_path", "file_path"))
            if manifest:
                files_inspected.append(manifest)
            key_findings.append(
                _clean_text(
                    f"Maven update {package or '[GAV omitted]'} version={version or '[version omitted]'} "
                    f"target={dependency_type or '[type omitted]'}: {_first_meaningful_line(content)} "
                    f"POM={manifest or '[POM omitted]'}",
                    650,
                )
            )
        elif event_name in {"remove_no_fix_dependency", "remove_no_fix_python_dependency"}:
            package = _clean_text(_arg(args, "requested_package", "package_name"), 180)
            manifest = _normalize_path(_arg(args, "manifest_path", "file_path"))
            if manifest:
                files_inspected.append(manifest)
            if manifest.casefold().endswith("pom.xml"):
                key_findings.append(
                    _clean_text(
                        f"Maven direct dependency removal {package or '[GAV omitted]'}: "
                        f"{_first_meaningful_line(content)} POM={manifest}",
                        500,
                    )
                )
            else:
                key_findings.append(
                    _clean_text(
                        f"package removal {package or '[package omitted]'}: {_first_meaningful_line(content)}"
                        f" manifest={manifest or '[manifest omitted]'}",
                        500,
                    )
                )
        elif event_name == "validate_workaround":
            payload = _json_payload(content)
            if payload is not None:
                validated = payload.get("validated_files", []) or []
                files_inspected.extend(_normalize_path(value) for value in validated if value)
                validation_outcome = _clean_text(
                    f"status={payload.get('overall_status', '')}; "
                    f"syntax={payload.get('syntax', '')}; "
                    f"typecheck={payload.get('typecheck', '')}; "
                    f"lint={payload.get('lint', '')}; "
                    f"runtime_smoke={payload.get('runtime_smoke', '')}; "
                    f"targeted_test={payload.get('targeted_test', '')}; "
                    f"targeted_test_file={payload.get('targeted_test_file', '')}; "
                    f"validated_files={','.join(_normalize_path(value) for value in validated if value)}; "
                    f"diagnostics={payload.get('infrastructure_diagnostics', '')}",
                    1300,
                )
            else:
                validation_outcome = _clean_text(_first_meaningful_line(content), 1300)
        elif event_name == "record_targeted_test_substitution":
            key_findings.append(
                _clean_text(
                    f"targeted test substitution original={_arg(args, 'original_test')}; "
                    f"alternative={_arg(args, 'alternative_test')}; "
                    f"evidence={_arg(args, 'infrastructure_failure_evidence') or _first_meaningful_line(content)}",
                    900,
                )
            )
        else:
            status = _first_meaningful_line(content)
            key_findings.append(_clean_text(f"{event_name}: {status or '[no status]'}", 520))

        files_inspected = _unique(files_inspected, self._seen_files, normalize_paths=True)
        if event_name == "inspect_ast_symbol" and pair_seen:
            # The AST pair index is authoritative; a duplicate pair should not
            # create another finding even when the tool emitted a new location.
            key_findings = [
                finding
                for finding in key_findings
                if not finding.startswith(f"AST {file_path}::{symbol}")
            ]
        key_findings = _unique(key_findings, self._seen_findings)

        if plan_summary:
            if plan_summary in self._seen_plan_summaries:
                plan_summary = ""
            else:
                self._seen_plan_summaries.add(plan_summary)
        if validation_outcome:
            # Keep repeated validation outcomes only when status or diagnostics
            # differ; the complete bounded outcome is the deduplication key.
            if validation_outcome in self._seen_validation_outcomes:
                validation_outcome = ""
            else:
                self._seen_validation_outcomes.add(validation_outcome)

        if (
            not key_findings
            and not files_inspected
            and not plan_summary
            and not validation_outcome
            and not critical_outcome
        ):
            return
        self._entries.append(
            ScratchpadEntry(
                scope=entry_scope,
                phase=normalized_phase,
                round_number=max(1, int(round_number)),
                key_findings=key_findings[:_MAX_FINDINGS_PER_ENTRY],
                files_inspected=files_inspected[:_MAX_FILES_PER_ENTRY],
                plan_summary=plan_summary,
                validation_outcome=validation_outcome,
                critical_outcome=critical_outcome,
            )
        )

    def _entry_lines(self, entry: ScratchpadEntry) -> list[str]:
        """Render one entry without headings."""
        facts: list[str] = []
        if entry.files_inspected:
            facts.append(f"files: {', '.join(entry.files_inspected)}")
        if entry.key_findings:
            facts.append(f"findings: {'; '.join(entry.key_findings)}")
        if entry.plan_summary:
            facts.append(f"plan: {entry.plan_summary}")
        if entry.validation_outcome:
            facts.append(f"validation: {entry.validation_outcome}")
        if entry.critical_outcome:
            facts.append(f"critical: {entry.critical_outcome}")
        return (
            [f"- round {entry.round_number}: {_clean_text(' | '.join(facts), 1500)}"]
            if facts
            else []
        )

    def _render_entries(self, entries: Sequence[ScratchpadEntry], truncated: bool = False) -> str:
        """Render selected entries in fixed scope and phase order."""
        lines = ["## Scratchpad Memory"]
        qa_entries = [entry for entry in entries if entry.scope == ScratchpadScope.QA]
        if qa_entries:
            lines.append("### QA_REVIEW")
            for entry in qa_entries:
                lines.extend(self._entry_lines(entry))
        for phase in WorkaroundExecutionPhase:
            phase_entries = [
                entry
                for entry in entries
                if entry.scope == ScratchpadScope.WORKAROUND and entry.phase == phase
            ]
            if not phase_entries:
                continue
            lines.append(f"### {phase.value}")
            for entry in phase_entries:
                lines.extend(self._entry_lines(entry))
        if truncated:
            lines.append("... older scratchpad facts omitted")
        return "\n".join(lines)

    def render(self) -> str:
        """Render bounded deterministic Markdown while retaining full history."""
        if not self._entries:
            return "## Scratchpad Memory\n(no retained facts)"
        entries = list(self._entries)
        full = self._render_entries(entries)
        if len(full) <= MAX_SCRATCHPAD_CHARS:
            return full

        critical_indexes = [
            index
            for index, entry in enumerate(entries)
            if entry.plan_summary or entry.validation_outcome or entry.critical_outcome
        ]
        finding_indexes = [
            index for index, entry in enumerate(entries) if index not in critical_indexes
        ]
        priority_indexes = list(reversed(critical_indexes)) + list(reversed(finding_indexes))
        selected: list[int] = []
        for index in priority_indexes:
            candidate = selected + [index]
            candidate_entries = [entries[item] for item in sorted(candidate)]
            if len(self._render_entries(candidate_entries, truncated=True)) <= MAX_SCRATCHPAD_CHARS:
                selected.append(index)
        selected_entries = [entries[index] for index in sorted(selected)]
        rendered = self._render_entries(selected_entries, truncated=len(selected) != len(entries))
        if len(rendered) <= MAX_SCRATCHPAD_CHARS:
            return rendered
        marker = "... older scratchpad facts omitted"
        available = max(0, MAX_SCRATCHPAD_CHARS - len(marker) - 1)
        return rendered[:available].rstrip() + "\n" + marker

    def to_system_message(self) -> SystemMessage:
        """Return the manager-owned scratchpad system message."""
        return SystemMessage(
            content=self.render(),
            additional_kwargs={_SCRATCHPAD_MARKER: True},
        )


def _is_manager_scratchpad(message: BaseMessage) -> bool:
    """Return whether a system message belongs to this context manager."""
    return isinstance(message, SystemMessage) and bool(
        getattr(message, "additional_kwargs", {}).get(_SCRATCHPAD_MARKER)
    )


def _compact_tool_message(message: ToolMessage) -> ToolMessage:
    """Replace only a compactable tool body while preserving message metadata."""
    content = str(getattr(message, "content", "") or "")
    if content.startswith("[COMPACTED]"):
        return message
    first_line = _first_meaningful_line(content, 300)
    marker = (
        f"[COMPACTED] {getattr(message, 'name', '') or 'tool'}: "
        f"{first_line or '[no status]'} (see scratchpad for key findings)"
    )
    try:
        return message.model_copy(update={"content": marker})
    except (AttributeError, TypeError):
        data = message.model_dump()
        data["content"] = marker
        return ToolMessage(**data)


def compact_conversation(
    conversation: Sequence[BaseMessage],
    current_round: int,
) -> list[BaseMessage]:
    """Compact eligible old tool bodies without mutating the input messages.

    Args:
        conversation: Current model conversation.
        current_round: One-based round about to continue.

    Returns:
        A new ordered message list with eligible old tool bodies replaced by
        bounded markers. Message headers, IDs, metadata, and ordering remain.
    """
    compacted: list[BaseMessage] = []
    message_round = 0
    cutoff = int(current_round) - 1
    for message in conversation:
        if isinstance(message, AIMessage):
            message_round += 1
        if (
            isinstance(message, ToolMessage)
            and message_round < cutoff
            and str(getattr(message, "name", "") or "") in _COMPACTION_TOOLS
        ):
            compacted.append(_compact_tool_message(message))
        else:
            compacted.append(message)
    return compacted


class ContextManager:
    """Manage phase-filtered bindings and ephemeral specialist context."""

    def __init__(
        self,
        all_tools: Sequence[Any],
        *,
        scratchpad: ScratchpadMemory | None = None,
        compaction_interval: int = DEFAULT_COMPACTION_INTERVAL,
        skip_phase_gating: bool = False,
        scratchpad_scope: ScratchpadScope = ScratchpadScope.WORKAROUND,
    ) -> None:
        """Initialize a context manager for one node invocation.

        Args:
            all_tools: Flat toolbelt returned by the worker builder.
            scratchpad: Optional memory object shared for this invocation.
            compaction_interval: Positive one-based model-round interval.
            skip_phase_gating: If true, expose every supplied tool in every
                phase and keep the initial binding for the whole run.
            scratchpad_scope: Scope assigned to entries in this manager's
                ephemeral scratchpad.

        Raises:
            ValueError: If ``compaction_interval`` is not positive or the
                scratchpad scope is invalid.
        """
        if compaction_interval <= 0:
            raise ValueError("compaction_interval must be positive")
        self._all_tools = tuple(all_tools)
        self.scratchpad_scope = (
            scratchpad_scope
            if isinstance(scratchpad_scope, ScratchpadScope)
            else ScratchpadScope(scratchpad_scope)
        )
        self.scratchpad = scratchpad or ScratchpadMemory(self.scratchpad_scope)
        self.compaction_interval = int(compaction_interval)
        self.skip_phase_gating = bool(skip_phase_gating)

    @property
    def all_tools(self) -> tuple[Any, ...]:
        """Return the immutable full toolbelt used for internal cleanup."""
        return self._all_tools

    def phase_from_state(
        self, execution_state: Mapping[str, Any] | None
    ) -> WorkaroundExecutionPhase:
        """Read and normalize the current execution phase."""
        return normalize_phase((execution_state or {}).get("phase"))

    def maybe_advance_to_plan(self, execution_state: dict[str, Any]) -> WorkaroundExecutionPhase:
        """Advance completed investigation to PLAN and never overwrite later phases."""
        phase = self.phase_from_state(execution_state)
        if phase == WorkaroundExecutionPhase.INVESTIGATE and bool(
            execution_state.get("local_investigation_complete")
        ):
            execution_state["phase"] = WorkaroundExecutionPhase.PLAN.value
            return WorkaroundExecutionPhase.PLAN
        return phase

    def get_tools_for_phase(self, phase: WorkaroundExecutionPhase | str | None) -> list[Any]:
        """Return the manager's ordered model-visible tool list for a phase."""
        if self.skip_phase_gating:
            return list(self._all_tools)
        return get_tools_for_phase(phase, self._all_tools)

    def get_phase_prompt(
        self,
        phase: WorkaroundExecutionPhase | str | None,
        base_context: str = "",
    ) -> str:
        """Return the manager's phase transition fragment."""
        return get_phase_prompt(phase, base_context)

    def update_scratchpad(
        self, event: Any, phase: WorkaroundExecutionPhase | str | None, round_number: int
    ) -> None:
        """Record one tool event in the node-local scratchpad."""
        self.scratchpad.update_from_tool_event(
            event,
            phase,
            round_number,
            scope=self.scratchpad_scope,
        )

    def _with_scratchpad_message(self, conversation: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Replace or insert only this manager's scratchpad system message."""
        updated = list(conversation)
        scratchpad_message = self.scratchpad.to_system_message()
        for index, message in enumerate(updated):
            if _is_manager_scratchpad(message):
                updated[index] = scratchpad_message
                return updated
        insertion_index = 0
        while insertion_index < len(updated) and isinstance(
            updated[insertion_index], SystemMessage
        ):
            insertion_index += 1
        updated.insert(insertion_index, scratchpad_message)
        return updated

    def compact_conversation(
        self,
        conversation: Sequence[BaseMessage],
        current_round: int,
    ) -> list[BaseMessage]:
        """Compact at configured boundaries and refresh the manager marker."""
        copied = list(conversation)
        if int(current_round) % self.compaction_interval != 0:
            return copied
        compacted = compact_conversation(copied, current_round)
        if self.scratchpad.entries:
            compacted = self._with_scratchpad_message(compacted)
        return compacted


def _request_parameters(
    model_name: str,
    reasoning_effort: LunaReasoningEffort | None,
) -> dict[str, Any]:
    """Build compatible Responses API parameters for a configured model.

    GPT-5.4 mini/nano explicitly use ``none`` reasoning to retain their
    low-latency behavior. GPT-6 Luna defaults to ``medium`` and does not accept
    temperature when reasoning effort is active. Other configured models keep
    the service's historical deterministic temperature setting.
    """
    normalized_model = model_name.strip().lower()
    parameters: dict[str, Any] = {"use_responses_api": True}

    if normalized_model == "gpt-6-luna":
        parameters["reasoning_effort"] = reasoning_effort or "medium"
    elif normalized_model.startswith(("gpt-5.4-mini", "gpt-5.4-nano")):
        parameters["reasoning_effort"] = "none"
        parameters["temperature"] = 0
    else:
        parameters["temperature"] = 0

    return parameters


def create_chat_model(
    model_name: str,
    *,
    settings: AppSettings | None = None,
    reasoning_effort: LunaReasoningEffort | None = None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    chat_openai_class: Callable[..., Any] | None = None,
) -> Any:
    """Create a ChatOpenAI client configured to use the Responses API.

    Args:
        model_name: OpenAI model ID selected for the agent.
        settings: Runtime settings supplying an explicit API key when present.
        reasoning_effort: Optional validated GPT-6 Luna reasoning effort.
        request_timeout: Optional bounded client request timeout in seconds.
        max_retries: Optional retry count; callers may set zero for one-shot
            provider requests.
        max_tokens: Optional output-token ceiling for bounded generations.
        chat_openai_class: Optional client constructor, primarily for tests and
            existing dependency-injection seams.

    Returns:
        A LangChain ChatOpenAI client using the Responses API.

    Raises:
        ImportError: If ``langchain-openai`` is unavailable and no constructor
            was provided.
        ValueError: If the selected model/effort combination or a supplied
            request bound is invalid.
    """
    if model_name.strip().lower() == "gpt-6-luna":
        if reasoning_effort is not None and reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("GPT-6 Luna reasoning effort is invalid")
    elif reasoning_effort is not None:
        raise ValueError("Reasoning effort only applies when the model is gpt-6-luna")
    if request_timeout is not None and request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    if max_retries is not None and max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    if chat_openai_class is None:
        from langchain_openai import ChatOpenAI

        chat_openai_class = ChatOpenAI

    client_kwargs: dict[str, Any] = {
        "model": model_name,
        **_request_parameters(model_name, reasoning_effort),
        **openai_api_key_kwargs(settings),
    }
    if request_timeout is not None:
        client_kwargs["request_timeout"] = request_timeout
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    if max_tokens is not None:
        client_kwargs["max_tokens"] = max_tokens
    return chat_openai_class(**client_kwargs)


MAX_VALIDATION_GATE_ATTEMPTS = 3
MAX_VALIDATION_INPUT_ERRORS = 3
MAX_UPDATE_MANIFEST_ATTEMPTS = 3
SANDBOX_NOT_RUNNING_MARKER = "sandbox is not running"
STAGNATION_REPETITION_THRESHOLD = 2
_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_npm_dependency"
_MAVEN_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_maven_dependency"
_PYTHON_UPDATE_MANIFEST_TOOL_NAME = "modify_and_validate_python_dependency"
_UPDATE_MANIFEST_TOOL_NAMES = frozenset(
    {
        _UPDATE_MANIFEST_TOOL_NAME,
        _MAVEN_UPDATE_MANIFEST_TOOL_NAME,
        _PYTHON_UPDATE_MANIFEST_TOOL_NAME,
    }
)
_PYTHON_MANIFEST_TRANSACTION_TOOL_NAMES = frozenset(
    {_PYTHON_UPDATE_MANIFEST_TOOL_NAME, "remove_no_fix_python_dependency"}
)

# Manifest mutations and validation form one ordered transaction. Maven removal
# shares the barrier; ordinary Node package removal retains its existing behavior.
_SERIAL_MANIFEST_TOOL_NAMES = _UPDATE_MANIFEST_TOOL_NAMES | frozenset(
    {"remove_no_fix_python_dependency"}
)
_WORKAROUND_VALIDATION_TOOL_NAMES = frozenset({"validate_workaround", "validate_python_workaround"})
_TARGETED_TEST_TOOL_NAMES = frozenset({"run_targeted_test", "run_targeted_python_test"})


def _response_text_content(content: Any) -> str:
    """Return only human-readable text from a model response content value.

    Args:
        content: LangChain message content, which may be plain text or a list of
            provider content blocks.

    Returns:
        Plain text blocks joined with newlines. Non-text blocks, including
        reasoning and encrypted payloads, are omitted.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, Mapping) or block.get("type") not in {
            "text",
            "output_text",
            "input_text",
        }:
            continue
        text_value = block.get("text")
        if isinstance(text_value, str):
            text_parts.append(text_value)
    return "\n".join(part for part in text_parts if part)


def _is_serial_manifest_tool_call(tool_name: str, args: dict[str, Any]) -> bool:
    """Serialize language-specific manifest transactions without changing Node removal."""
    if tool_name in _SERIAL_MANIFEST_TOOL_NAMES:
        return True
    if tool_name != "remove_no_fix_dependency":
        return False
    manifest_path = str(args.get("manifest_path", "") or "").replace("\\", "/").casefold()
    return manifest_path == "pom.xml" or manifest_path.endswith("/pom.xml")


def _is_workaround_validation_tool(tool_name: str) -> bool:
    return tool_name in _WORKAROUND_VALIDATION_TOOL_NAMES


def _preferred_validation_tool_name(
    tool_names: Mapping[str, Any] | Sequence[str] | set[str],
) -> str:
    if "validate_python_workaround" in tool_names and "validate_workaround" not in tool_names:
        return "validate_python_workaround"
    return "validate_workaround"


def _preferred_removal_tool_name(
    tool_names: Mapping[str, Any] | Sequence[str] | set[str],
) -> str:
    """Return the package-removal tool name matching the worker language."""
    if "remove_no_fix_dependency" not in tool_names and (
        "remove_no_fix_python_dependency" in tool_names
        or "validate_python_workaround" in tool_names
    ):
        return "remove_no_fix_python_dependency"
    return "remove_no_fix_dependency"


def _manifest_package_key(tool_name: str, package_name: Any) -> str:
    package = str(package_name or "").strip()
    if tool_name in _PYTHON_MANIFEST_TRANSACTION_TOOL_NAMES:
        return normalize_python_package_name(package)
    return package


@dataclass(frozen=True)
class ToolEvent:
    """Record one tool invocation observed by a specialist worker."""

    name: str
    args: dict[str, Any]
    content: str


@dataclass(frozen=True)
class SubagentRuntimeResult:
    """Summarize a bounded worker run and its observable side effects.

    ``structured_output`` contains the typed terminal result when a worker
    configures a terminal tool. ``final_text`` records the terminal textual
    response for diagnostics and callers that do not request a typed payload.
    ``changed_files`` and ``errors`` are collected from observed tool events.
    """

    final_text: str
    tool_events: list[ToolEvent]
    changed_files: list[str]
    errors: list[str]
    structured_output: BaseModel | None = None
    terminal_validation_passed: bool = False


def _contains_sandbox_not_running(content: str) -> bool:
    """Return whether a tool result reports that the workspace sandbox stopped."""
    return SANDBOX_NOT_RUNNING_MARKER in (content or "").lower()


def _infer_changed_files(tool_event: ToolEvent) -> list[str]:
    """Infer changed file paths from a successful edit-like tool event."""
    if not tool_event.content.startswith("SUCCESS:"):
        return []

    if tool_event.name == "deterministic_apply_edit_set":
        if "JSON:" in tool_event.content:
            try:
                json_part = tool_event.content.split("JSON:", 1)[1].strip()
                data = json.loads(json_part)
                if isinstance(data, dict) and "affected_files" in data:
                    return [
                        str(path).replace("\\", "/").lstrip("/") for path in data["affected_files"]
                    ]
            except (TypeError, ValueError):
                pass
        replacements = tool_event.args.get("replacements", [])
        if isinstance(replacements, str):
            try:
                replacements = json.loads(replacements)
            except (TypeError, ValueError):
                replacements = []
        if isinstance(replacements, list):
            return [
                str(item["file_path"]).replace("\\", "/").lstrip("/")
                for item in replacements
                if isinstance(item, dict) and isinstance(item.get("file_path"), str)
            ]

    if tool_event.name in _PYTHON_MANIFEST_TRANSACTION_TOOL_NAMES:
        if "JSON:" not in tool_event.content:
            return []
        try:
            payload = json.loads(tool_event.content.split("JSON:", 1)[1].strip())
        except (TypeError, ValueError):
            return []
        changed_files = payload.get("changed_files") if isinstance(payload, dict) else None
        if not isinstance(changed_files, list):
            return []
        return [
            path.replace("\\", "/").lstrip("/")
            for path in changed_files
            if isinstance(path, str) and path.strip()
        ]

    if tool_event.name in _UPDATE_MANIFEST_TOOL_NAMES:
        # Deferred and failed transactions did not leave a committed file change.
        manifest_path = tool_event.args.get(
            "manifest_path",
            "pom.xml" if tool_event.name == _MAVEN_UPDATE_MANIFEST_TOOL_NAME else "package.json",
        )
        if isinstance(manifest_path, str) and manifest_path.strip():
            return [manifest_path.replace("\\", "/")]
        return [
            "pom.xml" if tool_event.name == _MAVEN_UPDATE_MANIFEST_TOOL_NAME else "package.json"
        ]
    if tool_event.name in {
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
        "revert_workspace_file",
    }:
        file_path = tool_event.args.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            return [file_path.replace("\\", "/")]

    return []


def _invoke_bound_tool(tool_map: dict[str, Any], tool_call: dict[str, Any]) -> ToolMessage:
    """Execute one bound tool call and wrap the result in a ToolMessage."""
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}
    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        tool_call_id = "invalid-tool-call-id"

    if tool_name not in tool_map:
        content = f"ERROR: Unknown tool '{tool_name}'."
    else:
        try:
            result = invoke_with_trajectory(
                f"tool.{tool_name}",
                lambda: tool_map[tool_name].invoke(tool_args),
                tool_args,
                run_type="tool",
            )
            content = result if isinstance(result, str) else str(result)
        except Exception as exc:  # noqa: BLE001
            content = f"ERROR: Tool '{tool_name}' failed - {exc}"

    return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)


def _tool_call_signature(tool_call: dict[str, Any]) -> tuple[str, str]:
    """Return a stable signature for comparing repeated tool calls."""
    tool_name = str(tool_call.get("name", ""))
    tool_args = tool_call.get("args", {}) or {}

    try:
        serialized_args = json.dumps(
            tool_args,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        serialized_args = repr(tool_args)
    return tool_name, serialized_args


def _is_failed_tool_result(content: str) -> bool:
    """Return whether a tool response indicates that the requested operation failed."""
    lowered = content.lstrip().lower()
    return lowered.startswith(("error:", "error_code:", "not found:", "failed:", "failure"))


def _is_validation_gate_failure(content: str) -> bool:
    """Return whether a validation tool response requires retry recovery."""
    normalized = (content or "").lstrip()
    return normalized.startswith(
        (
            "FAILURE:",
            "ERROR: [SYNTAX_FAILURE]",
            "ERROR: [WRITE_FAILURE]",
        )
    )


def _is_successful_workaround_validation(content: str) -> bool:
    """Return whether a validation response contains a valid terminal PASS."""
    if _is_invalid_validation_request(content) or _is_validation_gate_failure(content):
        return False
    if (content or "").lstrip().startswith(
        "SUCCESS: Workaround validation gate passed"
    ) and "JSON:" not in content:
        # Production validation includes the structured JSON payload.  Keep
        # the success marker authoritative for deterministic replay doubles
        # that intentionally return only the bounded gate summary.
        return True
    marker = "JSON:"
    if marker not in content:
        return False
    try:
        payload = json.loads(content.split(marker, 1)[1].strip())
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and str(payload.get("overall_status", "")) == "PASS"


def _manifest_retry_recovery_instruction(
    tool_event: ToolEvent,
    *,
    attempts: int,
) -> str:
    """Build the next-step instruction after a failed update transaction."""
    package_name = str(tool_event.args.get("package_name", "")).strip() or "the same package"
    transaction_name = (
        tool_event.name
        if tool_event.name in _UPDATE_MANIFEST_TOOL_NAMES
        else "manifest transaction"
    )
    current_version = (
        str(tool_event.args.get("target_version", "")).strip() or "the current version"
    )
    current_type = (
        str(tool_event.args.get("dependency_type", "")).strip() or "the current dependency type"
    )
    evidence = (
        tool_event.content.strip()[:2_000]
        or "The combined update transaction failed without diagnostics."
    )
    if attempts >= MAX_UPDATE_MANIFEST_ATTEMPTS:
        return (
            f"The update transaction for {package_name} has exhausted its "
            f"{MAX_UPDATE_MANIFEST_ATTEMPTS}-attempt limit. Do not call "
            f"{transaction_name} for this package again; continue with the next "
            "independent task or surrender this package.\n"
            f"Exact transaction result:\n{evidence}"
        )
    return (
        f"The combined manifest transaction for {package_name} failed on attempt {attempts} with "
        f"target_version={current_version!r}, dependency_type={current_type!r}.\n"
        f"Exact transaction result:\n{evidence}\n"
        f"Call {transaction_name} again for the same Supervisor-owned package and manifest, "
        "but change target_version or dependency_type to a different value from the "
        "Supervisor-approved candidate lists. Do not query the registry, edit source files, "
        "or repeat the exact same call signature."
    )


def _stagnation_recovery_instruction(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_content: str = "",
) -> str:
    """Build a detailed recovery instruction after a repeated failed tool call."""
    formatted_args = json.dumps(tool_args)
    reason = tool_content.strip()[:300] or "Unknown failure"

    alternative = "Use search_codebase_pattern or read_workspace_file to inspect code context."
    if tool_name == "inspect_ast_symbol":
        alternative = (
            "An imported identifier or package binding is not an AST symbol. Use search_codebase_pattern "
            "to find the call site, then inspect the enclosing declared function/class, or use "
            "read_workspace_file and document the fallback in record_plan."
        )
    elif tool_name == "run_targeted_python_test":
        alternative = (
            "Verify the test path is relative and the pytest runner is supported, or proceed with "
            "AST inspection and syntax validation."
        )
    elif tool_name == "run_targeted_test":
        alternative = (
            "Verify the test path is relative and the runner is supported, or proceed with AST "
            "inspection and syntax validation."
        )
    elif tool_name in {
        "deterministic_apply_edit_set",
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
    }:
        alternative = (
            "Inspect the target files using read_workspace_file or inspect_ast_symbol to verify "
            "exact anchor lines, expected_occurrences, and line_hints before recording a new complete replacement plan."
        )

    return (
        f"Recovery instruction: tool '{tool_name}' with arguments {formatted_args} has failed repeatedly.\n"
        f"Concrete failure reason: {reason}\n"
        f"Confirmed next action / alternative path: {alternative}\n"
        f"PROHIBITION: You are explicitly prohibited from retrying tool '{tool_name}' with the exact same signature. Reassess your hypothesis and choose a different valid tool or argument set."
    )


def _targeted_test_recovery_instruction(tool_content: str) -> str:
    """Build a structured next-step instruction from a failed targeted test."""
    evidence = tool_content.strip()[:1800] or "The targeted test failed without diagnostic output."
    return (
        "Targeted test feedback requires a hypothesis update before the next edit.\n"
        "Exact targeted-test result:\n"
        f"{evidence}\n"
        "Trace the failure to the cumulative replayed patch and record a revised complete plan "
        "before editing. Preserve all prior security fixes, and include every causally related "
        "import, declaration, call site, and control-flow change in that plan. Do not fix an "
        "unrelated package or file merely because it appears in the stack trace."
    )


def _validation_gate_recovery_instruction(
    tool_content: str,
    *,
    validation_tool_name: str = "validate_workaround",
) -> str:
    """Build a recovery message from a failed combined validation gate."""
    evidence = (
        tool_content.strip()[:2400] or "The validation gate failed without diagnostic output."
    )

    if "[INVALID_RUNTIME_SMOKE]" in evidence:
        return (
            "The validation request used an invalid runtime smoke target. Do not edit the code again. "
            "Choose a lightweight repository source module (for example, the changed source module), "
            "never a test/spec file or build/dist artifact, and keep it separate from the targeted test. "
            f"Call {validation_tool_name} again with the corrected runtime_smoke_file.\n"
            f"Exact validation-gate result:\n{evidence}"
        )
    if "Runtime smoke gate unavailable" in evidence or "Command timed out after" in evidence:
        return (
            "Runtime smoke could not complete because the execution environment timed out or stopped. "
            "Do not reapply the same patch. Preserve the failure diagnostics and stop for an infrastructure "
            "retry; an alternative targeted test cannot replace the required runtime-smoke gate.\n"
            f"Exact validation-gate result:\n{evidence}"
        )
    return (
        "The combined workaround validation gate failed; do not declare success.\n"
        "Exact validation-gate result:\n"
        f"{evidence}\n"
        "Treat the first failed gate as the current root cause. If the result is CODE_FAILURE, "
        "the entire pending edit set has been reverted to the pre-iteration checkpoint and is no "
        "longer present: return to local INVESTIGATE, re-read every related file, and re-include "
        "every required change from that failed set in one complete revised plan and atomic semantic "
        "patch. Previously validated edits remain. If the result is INFRA_FAILURE for the targeted "
        "test, the pending edit set is retained; do not re-apply it. Instead inspect a repository "
        "alternative and record the original-to-alternative mapping with evidence before retrying validation. "
        "Do not use web research as generic recovery, and never declare success until every "
        "required gate passes."
    )


def _is_invalid_validation_request(tool_content: str) -> bool:
    """Return whether validation stopped during deterministic request preflight.

    These responses indicate that no syntax, typecheck, lint, runtime-smoke,
    or targeted-test gate ran, so they do not consume the executed-validation
    budget. The marker and text checks recognize the structured preflight
    errors emitted by the validation tools.
    """
    content = str(tool_content or "")
    if "[INVALID_VALIDATION_INPUT]" in content:
        return True
    if content.startswith("ERROR: [INVALID_RUNTIME_SMOKE]"):
        return True
    return (
        ("Targeted test path '" in content and "could not be verified" in content)
        or "is not the QA-recommended target" in content
        or "Compiled test path '" in content
    )


def _validation_request_signature(args: dict[str, Any]) -> str:
    """Return a stable signature for validation target arguments.

    The bounded worker loop uses this to stop an LLM from issuing the same
    invalid smoke/test selection repeatedly when deterministic preflight could
    not canonicalize it.

    Args:
        args: Tool-call arguments emitted by the worker.

    Returns:
        A JSON-stable signature containing the validation paths and selector.
    """
    if not any(key in args for key in ("runtime_smoke_file", "targeted_test_file")):
        return ""

    normalized: dict[str, Any] = {}
    for key in ("modified_files", "runtime_smoke_file", "targeted_test_file", "targeted_test_name"):
        value = args.get(key)
        if isinstance(value, list):
            normalized[key] = sorted(
                str(item).replace("\\", "/").strip().lstrip("/") for item in value
            )
        elif value is None:
            normalized[key] = None
        else:
            normalized[key] = str(value).replace("\\", "/").strip().lstrip("/")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _validation_input_recovery_instruction(
    tool_content: str,
    *,
    validation_tool_name: str = "validate_workaround",
    removal_tool_name: str = "remove_no_fix_dependency",
) -> str:
    """Build recovery guidance for a validation request rejected before gate execution."""
    evidence = str(tool_content or "").strip()[:2400]
    if "[PACKAGE_REMOVAL]" in evidence and (
        "remove_no_fix_dependency" in evidence
        or "remove_no_fix_python_dependency" in evidence
        or "configured dependency" in evidence
    ):
        return (
            "The validation request was rejected because the NO_FIX package-removal plan is "
            "incomplete. Do not edit or re-plan the source patch. Ensure both required successful "
            f"operations have been performed: call {removal_tool_name} for the configured "
            "package and deterministic_apply_edit_set for the recorded source replacements when "
            "they exist. Either operation may happen first. After both succeed, call "
            f"{validation_tool_name} with the complete cumulative file list.\n"
            f"Exact preflight result:\n{evidence}"
        )
    instruction = (
        "The validation request was rejected during deterministic preflight; no validation gate ran and "
        "the pending edit set was not judged. Do not edit or re-plan the source patch. Correct the validation "
        "arguments, resolving the repository-relative runtime smoke and targeted test paths from the workspace "
        "map. The smoke target must be a lightweight source module, and the targeted test must be an existing "
        "test/spec file separate from the smoke target."
    )
    if validation_tool_name != "validate_workaround":
        instruction += f" Then retry with {validation_tool_name}."
    return f"{instruction}\nExact preflight result:\n{evidence}"


def _manifest_call_deferred_instruction(deferred_tool_names: Sequence[str]) -> str:
    """Tell the worker to resume after deferred manifest calls."""
    names = ", ".join(deferred_tool_names)
    transaction_names = ", ".join(sorted(_SERIAL_MANIFEST_TOOL_NAMES))
    return (
        "Manifest transaction sequencing barrier: only one of "
        f"{transaction_names} may execute per assistant turn. "
        f"Deferred call(s): {names}. The prior manifest transaction has already "
        "been executed. On the next turn, inspect its result and continue with the "
        "next authorized package or a changed candidate; do not repeat a successful "
        "transaction."
    )


def run_bounded_subagent_loop(
    llm: Any,
    tools: Sequence[Any],
    initial_messages: Sequence[Any],
    touched_files: set[str],
    planning_state: dict[str, bool] | None = None,
    execution_state: dict[str, Any] | None = None,
    structured_output_model: type[BaseModel] | None = None,
    structured_output_tool_name: str | None = None,
    context_manager: ContextManager | None = None,
    skip_phase_gating: bool = False,
) -> SubagentRuntimeResult:
    """Run a bounded tool-calling loop for one specialized subagent.

    Args:
        llm: Chat model that supports tool binding and invocation.
        tools: Tools available to the bounded loop.
        initial_messages: Initial conversation messages.
        touched_files: Files already known to be changed by the worker.
        planning_state: Optional mutable planning-phase state.
        execution_state: Optional mutable validation/execution state.
        structured_output_model: Optional Pydantic model for a terminal result.
        structured_output_tool_name: Tool name whose arguments are validated as
            ``structured_output_model``. Both structured arguments are required
            together and the terminal tool body is not executed.
        context_manager: Optional phase-aware workaround context manager. When
            absent, the supplied full tool list is bound exactly once.
        skip_phase_gating: Bypass manager phase filtering and rebinding for a
            read-only workflow such as structured QA. A manager's
            ``skip_phase_gating`` flag also enables this mode.

    Returns:
        The bounded run result, including the validated terminal model when
        structured output is configured.

    Raises:
        ValueError: If structured-output configuration is incomplete or names
            a tool that is not in tools.
    """
    all_tools = tuple(tools)
    all_tool_map = {str(getattr(tool, "name", "")): tool for tool in all_tools}
    validation_tool_name = _preferred_validation_tool_name(all_tool_map)
    removal_tool_name = _preferred_removal_tool_name(all_tool_map)
    # This is only a provider hint; the runtime barrier below remains
    # authoritative for providers that ignore it or still emit multiple calls.
    if (structured_output_model is None) != (structured_output_tool_name is None):
        raise ValueError(
            "structured_output_model and structured_output_tool_name must be provided together."
        )
    phase_gating_bypassed = bool(
        skip_phase_gating or (context_manager is not None and context_manager.skip_phase_gating)
    )
    if context_manager is None or phase_gating_bypassed:
        bound_tools = all_tools
    else:
        initial_phase = context_manager.phase_from_state(execution_state)
        bound_tools = tuple(context_manager.get_tools_for_phase(initial_phase))
    tool_map = {str(getattr(tool, "name", "")): tool for tool in bound_tools}
    bound_tool_names = tuple(tool_map)
    llm_with_tools = llm.bind_tools(list(bound_tools), parallel_tool_calls=False)
    conversation = list(initial_messages)
    tool_events: list[ToolEvent] = []
    final_text = ""
    errors: list[str] = []
    observed_changed_files = set(touched_files)
    failed_tool_call_counts: dict[tuple[str, str], int] = {}
    recovery_signatures: set[tuple[str, str]] = set()
    last_failed_tool_content: dict[tuple[str, str], str] = {}
    manifest_attempts_by_package: dict[str, int] = {}
    manifest_retry_exhausted_packages: set[str] = set()
    consecutive_validation_failures = 0
    validation_gate_call_count = int((execution_state or {}).get("validation_calls", 0))
    validation_input_error_count = int((execution_state or {}).get("validation_input_errors", 0))
    scope_violation_count = 0
    invalid_validation_signatures: set[str] = set()
    # This flag is scoped to this bounded worker run.  A prior replay state is
    # evidence, but cannot by itself terminate a fresh worker invocation.
    terminal_validation_passed = False
    for loop_index in range(MAX_SUBAGENT_TOOL_CALL_ROUNDS):
        round_number = loop_index + 1
        if context_manager is not None:
            turn_phase = context_manager.phase_from_state(execution_state)
            if not phase_gating_bypassed:
                phase_tools = tuple(context_manager.get_tools_for_phase(turn_phase))
                phase_tool_names = tuple(str(getattr(tool, "name", "")) for tool in phase_tools)
                if phase_tool_names != bound_tool_names:
                    bound_tools = phase_tools
                    tool_map = {str(getattr(tool, "name", "")): tool for tool in bound_tools}
                    bound_tool_names = phase_tool_names
                    llm_with_tools = llm.bind_tools(
                        list(bound_tools),
                        parallel_tool_calls=False,
                    )
        else:
            turn_phase = None
        turn_tool_map = dict(tool_map)
        try:
            invoke_messages = list(conversation)
            response = invoke_with_trajectory(
                "react.llm",
                lambda bound_llm=llm_with_tools, messages=invoke_messages: bound_llm.invoke(
                    messages
                ),
                invoke_messages,
                run_type="llm",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Subagent LLM/tool-loop invocation failed: {exc}")
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )
        conversation.append(response)
        final_text = _response_text_content(getattr(response, "content", ""))
        if planning_state is not None:
            lowered = final_text.lower()
            if "planning answers" in lowered or "1. what version" in lowered:
                planning_state["submitted"] = True

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            if structured_output_model is None:
                return SubagentRuntimeResult(
                    final_text=final_text,
                    tool_events=tool_events,
                    changed_files=sorted(observed_changed_files | set(touched_files)),
                    errors=errors,
                )
            conversation.append(
                HumanMessage(
                    content=(
                        "This evaluator may not emit free-form text. Continue reviewing evidence "
                        f"and finish by calling {structured_output_tool_name} with the complete "
                        "structured QA result."
                    )
                )
            )
            continue

        if structured_output_model is not None:
            terminal_calls = [
                call
                for call in tool_calls
                if str(call.get("name", "")) == structured_output_tool_name
            ]
            if terminal_calls:
                terminal_call = terminal_calls[0]
                try:
                    structured_output = structured_output_model.model_validate(
                        terminal_call.get("args", {}) or {}
                    )
                except ValidationError as exc:
                    terminal_call_id = terminal_call.get("id")
                    if not isinstance(terminal_call_id, str) or not terminal_call_id.strip():
                        terminal_call_id = "invalid-structured-output-call-id"
                    conversation.append(
                        ToolMessage(
                            content=(
                                "ERROR: Structured QA output failed validation. "
                                f"Correct the fields and call {structured_output_tool_name} again: {exc}"
                            ),
                            tool_call_id=terminal_call_id,
                            name=structured_output_tool_name,
                        )
                    )
                    errors.append(
                        "Structured QA output validation failed; evaluator was asked to retry."
                    )
                    continue
                event = ToolEvent(
                    name=structured_output_tool_name,
                    args=terminal_call.get("args", {}) or {},
                    content="STRUCTURED_OUTPUT_ACCEPTED",
                )
                tool_events.append(event)
                if context_manager is not None:
                    context_manager.update_scratchpad(event, turn_phase, round_number)
                return SubagentRuntimeResult(
                    final_text="",
                    tool_events=tool_events,
                    changed_files=sorted(observed_changed_files | set(touched_files)),
                    errors=list(dict.fromkeys(errors)),
                    structured_output=structured_output,
                )

        recovery_instruction: str | None = None
        sandbox_stopped = False
        malformed_tool_call = False
        infrastructure_blocked = False
        scope_loop_error: str | None = None
        validation_limit_error: str | None = None
        validation_input_limit_error: str | None = None
        blocker_errors: list[str] = []
        manifest_call_seen = False
        deferred_manifest_calls: list[str] = []
        manifest_retry_instructions: dict[str, str] = {}
        turn_event_start = len(tool_events)
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", ""))
            call_signature = _tool_call_signature(tool_call)
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                malformed_tool_call = True
            phase_violation = (
                context_manager is not None
                and not phase_gating_bypassed
                and tool_name in all_tool_map
                and tool_name not in turn_tool_map
            )
            if phase_violation:
                phase_name = getattr(turn_phase, "value", str(turn_phase or "INVESTIGATE"))
                if phase_name == "VALIDATE" and tool_name in {
                    "record_plan",
                    "search_web",
                    "read_web_page",
                }:
                    violation_content = (
                        "ERROR: [PHASE_VIOLATION] The worker is in VALIDATE. "
                        f"Call {validation_tool_name} before planning or web research."
                    )
                else:
                    violation_content = (
                        f"ERROR: [PHASE_VIOLATION] Tool '{tool_name}' is not available "
                        f"in the {phase_name} phase. Follow the phase instructions and "
                        "use only the currently bound tools."
                    )
                tool_message = ToolMessage(
                    content=violation_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            else:
                if _is_serial_manifest_tool_call(tool_name, tool_call.get("args", {}) or {}):
                    # Set the barrier before handling suppressed or synthetic
                    # failures as well. Every manifest call in an assistant turn
                    # must be deferred after the first one, regardless of whether
                    # the first call reached the underlying tool.
                    if manifest_call_seen:
                        deferred_manifest_calls.append(tool_name)
                        tool_message = ToolMessage(
                            content=(
                                "DEFERRED: manifest tools are serialized. This call was not "
                                "executed; continue it in the next assistant turn after the "
                                "current package operation has been observed."
                            ),
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                        conversation.append(tool_message)
                        event = ToolEvent(
                            name=tool_name,
                            args=tool_call.get("args", {}) or {},
                            content=tool_message.content,
                        )
                        tool_events.append(event)
                        if context_manager is not None:
                            context_manager.update_scratchpad(event, turn_phase, round_number)
                        continue
                    manifest_call_seen = True
                if call_signature in recovery_signatures and not _is_workaround_validation_tool(
                    tool_name
                ):
                    prior_failure = last_failed_tool_content.get(call_signature, "")[:500]
                    # Validation retries have their own bounded gate counter; do not
                    # hide a legitimate gate attempt behind generic stagnation suppression.
                    tool_message = ToolMessage(
                        content=(
                            "ERROR: [REPEATED_INVALID_CALL] Repeated failed tool call suppressed. Choose a different "
                            "tool or argument set; the exact signature has already received a "
                            f"recovery instruction. Prior failure: {prior_failure}"
                        ),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                else:
                    if tool_name in _UPDATE_MANIFEST_TOOL_NAMES:
                        # A synthetic retry-limit response is still the one
                        # serialized manifest call for this assistant turn.
                        raw_package_name = str(
                            (tool_call.get("args", {}) or {}).get("package_name", "")
                        ).strip()
                        package_name = _manifest_package_key(tool_name, raw_package_name)
                        state_attempts = int(
                            (
                                (execution_state or {}).get(
                                    "manifest_transaction_attempts_by_package", {}
                                )
                            ).get(package_name, 0)
                        )
                        runtime_attempts = manifest_attempts_by_package.get(package_name, 0)
                        attempts = max(state_attempts, runtime_attempts)
                        if (
                            package_name in manifest_retry_exhausted_packages
                            or attempts >= MAX_UPDATE_MANIFEST_ATTEMPTS
                        ):
                            tool_message = ToolMessage(
                                content=(
                                    "ERROR_CODE: RETRY_LIMIT_REACHED: At most three combined "
                                    f"update attempts are allowed for package {package_name or '(unknown)'}."
                                ),
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                        else:
                            tool_message = _invoke_bound_tool(turn_tool_map, tool_call)
                            manifest_attempts_by_package[package_name] = attempts + 1
                            if execution_state is not None:
                                runtime_attempts_by_package = execution_state.setdefault(
                                    "manifest_runtime_attempts_by_package", {}
                                )
                                runtime_attempts_by_package[package_name] = max(
                                    int(runtime_attempts_by_package.get(package_name, 0)),
                                    manifest_attempts_by_package[package_name],
                                )
                                execution_state["manifest_transaction_attempts"] = max(
                                    int(execution_state.get("manifest_transaction_attempts", 0)),
                                    sum(manifest_attempts_by_package.values()),
                                )
                    else:
                        if (
                            execution_state is not None
                            and not phase_gating_bypassed
                            and execution_state.get("phase") == "VALIDATE"
                            and tool_name
                            in {
                                "record_plan",
                                "search_web",
                                "read_web_page",
                            }
                        ):
                            tool_message = ToolMessage(
                                content=(
                                    "ERROR: [PHASE_VIOLATION] The worker is in VALIDATE. "
                                    f"Call {validation_tool_name} before planning or web research."
                                ),
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                        else:
                            tool_message = _invoke_bound_tool(turn_tool_map, tool_call)
            conversation.append(tool_message)
            event = ToolEvent(
                name=tool_name,
                args=tool_call.get("args", {}) or {},
                content=tool_message.content,
            )
            tool_events.append(event)
            if context_manager is not None:
                context_manager.update_scratchpad(event, turn_phase, round_number)

            if (
                "[PLAN_VIOLATION]" in tool_message.content
                or "[PROHIBITED_TARGET]" in tool_message.content
            ):
                scope_violation_count += 1
                if scope_violation_count >= 2:
                    scope_loop_error = f"SCOPE_LOOP_ERROR: Subagent stopped due to repeated scope/plan violations: {tool_message.content}"

            preflight_validation_error = _is_workaround_validation_tool(
                event.name
            ) and _is_invalid_validation_request(tool_message.content)
            if not preflight_validation_error and (
                tool_message.content.startswith("BLOCKED:") or "BLOCKED:" in tool_message.content
            ):
                infrastructure_blocked = True
                b_msg = tool_message.content.strip()
                if not b_msg.startswith("INFRASTRUCTURE_BLOCKER:"):
                    b_msg = f"INFRASTRUCTURE_BLOCKER: {b_msg}"
                blocker_errors.append(b_msg)

            if _is_failed_tool_result(tool_message.content):
                failed_tool_call_counts[call_signature] = (
                    failed_tool_call_counts.get(call_signature, 0) + 1
                )
                last_failed_tool_content[call_signature] = tool_message.content
                if (
                    failed_tool_call_counts[call_signature] >= STAGNATION_REPETITION_THRESHOLD
                    and call_signature not in recovery_signatures
                ):
                    recovery_instruction = _stagnation_recovery_instruction(
                        event.name,
                        event.args,
                        event.content,
                    )
                    recovery_signatures.add(call_signature)
            else:
                failed_tool_call_counts.pop(call_signature, None)

            if event.name in _UPDATE_MANIFEST_TOOL_NAMES and _is_failed_tool_result(
                tool_message.content
            ):
                raw_package_name = str(event.args.get("package_name", "")).strip()
                package_name = _manifest_package_key(event.name, raw_package_name)
                attempts_by_package = (execution_state or {}).get(
                    "manifest_transaction_attempts_by_package", {}
                )
                attempts = max(
                    int(attempts_by_package.get(package_name, 0)),
                    manifest_attempts_by_package.get(package_name, 0),
                )
                if attempts >= MAX_UPDATE_MANIFEST_ATTEMPTS:
                    manifest_retry_exhausted_packages.add(package_name)
                    if execution_state is not None:
                        exhausted = list(
                            execution_state.get("manifest_retry_exhausted_packages", []) or []
                        )
                        if package_name not in exhausted:
                            exhausted.append(package_name)
                        execution_state["manifest_retry_exhausted_packages"] = exhausted
                manifest_retry_instructions[package_name or "__unknown__"] = (
                    _manifest_retry_recovery_instruction(event, attempts=attempts)
                )

            if event.name in _TARGETED_TEST_TOOL_NAMES and event.content.startswith("FAILURE:"):
                conversation.append(
                    HumanMessage(content=_targeted_test_recovery_instruction(event.content))
                )
            if _is_workaround_validation_tool(event.name):
                if _is_invalid_validation_request(event.content):
                    validation_signature = _validation_request_signature(event.args)
                    repeated_invalid_request = bool(
                        validation_signature
                        and validation_signature in invalid_validation_signatures
                    )
                    if validation_signature:
                        invalid_validation_signatures.add(validation_signature)
                    validation_input_error_count += 1
                    if execution_state is not None:
                        execution_state["validation_input_errors"] = max(
                            int(execution_state.get("validation_input_errors", 0)),
                            validation_input_error_count,
                        )
                    if repeated_invalid_request:
                        validation_input_limit_error = (
                            "VALIDATION_INPUT_LIMIT_REACHED: VALIDATION_INPUT_RETRY_LOOP: "
                            "The same invalid validation target request was repeated. "
                            "Do not retry the same smoke/test paths; use the canonical paths "
                            "provided by deterministic preflight."
                        )
                    elif validation_input_error_count >= MAX_VALIDATION_INPUT_ERRORS:
                        validation_input_limit_error = "VALIDATION_INPUT_LIMIT_REACHED: Maximum 3 invalid validation requests reached."
                    else:
                        conversation.append(
                            HumanMessage(
                                content=_validation_input_recovery_instruction(
                                    event.content,
                                    validation_tool_name=validation_tool_name,
                                    removal_tool_name=removal_tool_name,
                                )
                            )
                        )

                else:
                    validation_gate_call_count += 1
                    if execution_state is not None:
                        # Real validation tools update this state directly. Also
                        # mirror the count here so wrappers and deterministic
                        # test doubles retain the executed-gate accounting.
                        execution_state["validation_calls"] = max(
                            int(execution_state.get("validation_calls", 0)),
                            validation_gate_call_count,
                        )
                        json_marker = "JSON:"
                        if json_marker in event.content:
                            try:
                                payload = json.loads(event.content.split(json_marker, 1)[1].strip())
                            except (TypeError, ValueError):
                                payload = None
                            if isinstance(payload, dict):
                                execution_state["last_validation_result"] = payload
                                execution_state["validated_files"] = list(
                                    payload.get("validated_files", []) or []
                                )
                                execution_state["validation_passed"] = (
                                    payload.get("overall_status") == "PASS"
                                )
                                if execution_state["validation_passed"]:
                                    terminal_validation_passed = True
                    if _is_successful_workaround_validation(event.content):
                        terminal_validation_passed = True
                    if (
                        validation_gate_call_count >= MAX_VALIDATION_GATE_ATTEMPTS
                        and _is_validation_gate_failure(event.content)
                    ):
                        validation_limit_error = (
                            "VALIDATION_LIMIT_REACHED: Maximum 3 validation gate attempts reached."
                        )
                    elif _is_validation_gate_failure(event.content):
                        consecutive_validation_failures += 1
                        if consecutive_validation_failures >= 3:
                            conversation.append(
                                HumanMessage(
                                    content=(
                                        "CRITICAL: You have failed the validation gate 3 times in a row. "
                                        "Stop repeating the same action. Classify the latest structured result: "
                                        "for CODE_FAILURE restart at INVESTIGATE; for INFRA_FAILURE inspect and "
                                        "register one alternative targeted test. Do not search the web unless "
                                        "you are back in INVESTIGATE and local evidence is insufficient."
                                    )
                                )
                            )
                            consecutive_validation_failures = 0
                        else:
                            conversation.append(
                                HumanMessage(
                                    content=_validation_gate_recovery_instruction(
                                        event.content,
                                        validation_tool_name=validation_tool_name,
                                    )
                                )
                            )
                    else:
                        consecutive_validation_failures = 0

            inferred_paths = _infer_changed_files(event)
            if inferred_paths:
                if event.name == "revert_workspace_file":
                    for path in inferred_paths:
                        if path not in touched_files:
                            observed_changed_files.discard(path)
                else:
                    observed_changed_files.update(inferred_paths)

            if (
                context_manager is None
                and execution_state is not None
                and event.name
                in {
                    "deterministic_apply_edit_set",
                    "deterministic_search_replace",
                    "deterministic_replace_ast_symbol",
                }
                and event.content.startswith("SUCCESS:")
            ):
                conversation.append(
                    HumanMessage(
                        content=(
                            f"The patch was applied atomically; call {validation_tool_name} now "
                            "with the complete modified-file list, an explicit runtime_smoke_file, "
                            "where runtime_smoke_file is a lightweight source module (not a test/spec "
                            "or build/dist artifact) and is separate from the required targeted test. "
                            f"Your next action must be {validation_tool_name}. "
                            "Do not edit, re-plan, or browse first."
                        )
                    )
                )

            if _contains_sandbox_not_running(tool_message.content):
                # Do not return from inside this loop. The model may have
                # emitted several tool calls in one assistant message and
                # every call needs a matching ToolMessage before the next LLM
                # request. Returning here was the source of malformed OpenAI
                # transcripts in which later tool_call_ids had no response.
                sandbox_stopped = True

        if deferred_manifest_calls:
            conversation.append(
                HumanMessage(content=_manifest_call_deferred_instruction(deferred_manifest_calls))
            )

        for instruction in manifest_retry_instructions.values():
            conversation.append(HumanMessage(content=instruction))
        if context_manager is not None:
            if recovery_instruction:
                conversation.append(HumanMessage(content=recovery_instruction))
            if not phase_gating_bypassed:
                if execution_state is not None:
                    final_phase = context_manager.maybe_advance_to_plan(execution_state)
                else:
                    final_phase = turn_phase
                if final_phase != turn_phase:
                    phase_prompt = context_manager.get_phase_prompt(final_phase)
                    if validation_tool_name == "validate_python_workaround":
                        phase_prompt = phase_prompt.replace(
                            "validate_workaround",
                            validation_tool_name,
                        ).replace(
                            "remove_no_fix_dependency",
                            removal_tool_name,
                        )
                    conversation.append(HumanMessage(content=phase_prompt))
            conversation = context_manager.compact_conversation(conversation, round_number)

        if malformed_tool_call:
            errors.append(
                "Subagent stopped because the model emitted a tool call without a valid tool_call_id."
            )
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=errors,
            )

        # A valid validation PASS is a terminal worker result.  We still
        # processed every tool call emitted in the current assistant turn
        # above, so each call has a matching ToolMessage.  Do not issue a
        # follow-up LLM round and, importantly, do not append the generic
        # max-round error that used to invalidate this successful gate.
        if terminal_validation_passed:
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
                terminal_validation_passed=True,
            )

        if infrastructure_blocked:
            errors.extend(blocker_errors)
            revert_tool = all_tool_map.get("revert_workspace_file")
            for f_path in list(observed_changed_files):
                if revert_tool and f_path not in touched_files:
                    with contextlib.suppress(Exception):
                        revert_tool.invoke({"file_path": f_path})
            observed_changed_files.clear()
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=[],
                errors=list(dict.fromkeys(errors)),
            )

        if scope_loop_error:
            errors.append(scope_loop_error)
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
            )

        if validation_limit_error:
            errors.append(validation_limit_error)
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
            )

        if validation_input_limit_error:
            errors.append(validation_input_limit_error)
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
            )

        if sandbox_stopped:
            sandbox_errors = [
                event.content
                for event in tool_events[turn_event_start:]
                if _contains_sandbox_not_running(event.content)
            ]
            errors.extend(f"INFRASTRUCTURE_FAILURE: {message}" for message in sandbox_errors)
            errors.append(
                "INFRASTRUCTURE_FAILURE: Subagent stopped because the sandbox is no longer running."
            )
            return SubagentRuntimeResult(
                final_text=final_text,
                tool_events=tool_events,
                changed_files=sorted(observed_changed_files | set(touched_files)),
                errors=list(dict.fromkeys(errors)),
            )

        if recovery_instruction and context_manager is None:
            conversation.append(HumanMessage(content=recovery_instruction))

    errors.append(
        "MAX_SUBAGENT_TOOL_CALL_ROUNDS: "
        "Subagent exceeded the maximum tool-call rounds without reaching a final answer."
    )
    return SubagentRuntimeResult(
        final_text=final_text,
        tool_events=tool_events,
        changed_files=sorted(observed_changed_files | set(touched_files)),
        errors=errors,
        terminal_validation_passed=terminal_validation_passed,
    )


def has_all_modified_files_validated_after_last_edit(
    tool_events: Sequence[ToolEvent],
    edit_tool_names: Sequence[str] = (
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
    ),
    validation_tool_name: str = "validate_code_syntax",
) -> bool:
    """Return whether every file modified by a successful edit had a successful syntax validation after its last edit."""
    last_edit_by_file: dict[str, int] = {}
    for index, event in enumerate(tool_events):
        if event.name in edit_tool_names and event.content.startswith("SUCCESS:"):
            file_path = event.args.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                norm_path = file_path.replace("\\", "/").strip()
                last_edit_by_file[norm_path] = index

    if not last_edit_by_file:
        return False

    for norm_path, last_edit_idx in last_edit_by_file.items():
        validated = False
        for event in tool_events[last_edit_idx + 1 :]:
            if event.name == validation_tool_name and event.content.startswith("SUCCESS:"):
                val_path = event.args.get("file_path")
                if not val_path or val_path.replace("\\", "/").strip() == norm_path:
                    validated = True
                    break
        if not validated:
            return False

    return True


def has_successful_validation_gate(
    tool_events: Sequence[ToolEvent],
    validation_tool_name: str | None = None,
    edit_tool_names: Sequence[str] = (
        "deterministic_apply_edit_set",
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
        "revert_workspace_file",
    ),
    has_prior_edits: bool = False,
) -> bool:
    """Return whether the combined gate passed after the final cumulative edit.

    A gate result from before a later edit is stale and cannot establish worker
    success. ``has_prior_edits`` covers replayed edits, which are restored into
    the sandbox before the current attempt and therefore do not have edit events
    in the current runtime trace.
    """
    last_edit_index = max(
        (
            index
            for index, event in enumerate(tool_events)
            if event.name in edit_tool_names and event.content.startswith("SUCCESS:")
        ),
        default=-1,
    )
    if last_edit_index < 0 and not has_prior_edits:
        return False

    validation_tool_names = (
        _WORKAROUND_VALIDATION_TOOL_NAMES
        if validation_tool_name is None
        else {validation_tool_name}
    )
    return any(
        index > last_edit_index
        and event.name in validation_tool_names
        and event.content.startswith("SUCCESS:")
        for index, event in enumerate(tool_events)
    )


def has_tool_call_before_first_successful_edit(
    tool_events: Sequence[ToolEvent],
    lookup_tool_name: str,
    edit_tool_names: Sequence[str] | str = (
        "deterministic_apply_edit_set",
        "deterministic_search_replace",
        "deterministic_replace_ast_symbol",
    ),
) -> bool:
    """Return whether a given tool was called before the first successful edit."""
    names = (edit_tool_names,) if isinstance(edit_tool_names, str) else tuple(edit_tool_names)
    first_successful_edit_index = -1
    for index, event in enumerate(tool_events):
        if event.name in names and event.content.startswith("SUCCESS:"):
            first_successful_edit_index = index
            break

    if first_successful_edit_index < 0:
        return False

    for event in tool_events[:first_successful_edit_index]:
        if event.name == lookup_tool_name:
            return True
    return False
