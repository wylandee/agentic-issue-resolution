"""Phase-aware context management for specialist subagents.

The manager in this module owns only ephemeral, node-local model context. The
Supervisor task state, attempt snapshots, and replay plans remain the durable
source of truth for remediation progress.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from remediation_engine.contracts.schemas import (
    ScratchpadEntry,
    ScratchpadScope,
    WorkaroundExecutionPhase,
)

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
        "record_targeted_test_substitution",
        "deterministic_apply_edit_set",
        "validate_workaround",
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
            "remove_no_fix_dependency",
        }
    ),
    WorkaroundExecutionPhase.VALIDATE: frozenset(
        {
            "read_repository_map",
            "validate_workaround",
            "record_targeted_test_substitution",
            "read_workspace_file",
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
        "re-plan. Once the inputs are valid, your next action must be validate_workaround."
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
        elif event_name == "remove_no_fix_dependency":
            package = _clean_text(_arg(args, "requested_package", "package_name"), 180)
            manifest = _normalize_path(_arg(args, "manifest_path", "file_path"))
            if manifest:
                files_inspected.append(manifest)
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


__all__ = [
    "DEFAULT_COMPACTION_INTERVAL",
    "MAX_SCRATCHPAD_CHARS",
    "PHASE_TOOL_REGISTRY",
    "ContextManager",
    "ScratchpadMemory",
    "ScratchpadScope",
    "compact_conversation",
    "get_phase_prompt",
    "get_tools_for_phase",
    "normalize_phase",
]
