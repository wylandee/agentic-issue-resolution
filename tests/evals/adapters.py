"""Trajectory recorder to DeepEval test case adapter and parser.

This module provides data models, markdown/dict parsers, and conversion utilities
to bridge recorded Phase 5 trajectories and in-memory trace spans into DeepEval
`LLMTestCase` instances without requiring network access or modifying production code.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepEval Fallbacks for Offline & Optional Dependency Environments
# ---------------------------------------------------------------------------

try:
    from deepeval.test_case import LLMTestCase as DeepEvalLLMTestCase
    from deepeval.test_case import ToolCall as DeepEvalToolCall

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False

    @dataclass
    class DeepEvalToolCall:  # type: ignore[no-redef]
        """Fallback ToolCall dataclass when deepeval is not installed."""

        name: str
        input_parameters: dict[str, Any] = field(default_factory=dict)
        output: Any = None

    @dataclass
    class DeepEvalLLMTestCase:  # type: ignore[no-redef]
        """Fallback LLMTestCase dataclass when deepeval is not installed."""

        input: str
        actual_output: str
        expected_output: str | None = None
        context: list[str] | None = None
        retrieval_context: list[str] | None = None
        tools_called: list[DeepEvalToolCall] | None = None
        expected_tools: list[DeepEvalToolCall] | None = None
        latency: float | None = None
        cost: float | None = None
        additional_metadata: dict[str, Any] | None = None


ToolCall = DeepEvalToolCall
LLMTestCase = DeepEvalLLMTestCase


# ---------------------------------------------------------------------------
# Trajectory Data Models
# ---------------------------------------------------------------------------

_KNOWN_LLM_SPAN_NAMES = frozenset(
    {
        "triage.llm",
        "react.llm",
        "qa_critic.batch_judge",
        "ChatOpenAI",
        "report.narrative",
        "fix_planner.llm",
    }
)

_KNOWN_TOOL_NAMES = frozenset(
    {
        "modify_npm_dependency",
        "validate_manifest_sync",
        "list_changed_files",
        "read_file_context",
        "query_qa_logs",
        "generate_workspace_diff",
        "revert_workspace_file",
        "read_repository_map",
        "plan_npm_version",
        "docker_sandbox.run",
        "docker_sandbox.start",
        "docker_sandbox.read_file",
        "docker_sandbox.teardown",
        "record_plan",
        "deterministic_apply_edit_set",
        "validate_workaround",
        "emit_qa_evaluation",
    }
)


@dataclass
class TrajectorySpan:
    """Structured representation of an execution span from a Phase 5 trajectory."""

    run_id: str
    name: str
    run_type: str
    parent_run_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    serialized: dict[str, Any] | None = None
    sequence: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: str | None = None
    parent_name: str | None = None
    duration_seconds: float | None = None

    @property
    def is_llm(self) -> bool:
        """Return True if this span represents an LLM call."""
        return (
            self.run_type == "llm"
            or self.name in _KNOWN_LLM_SPAN_NAMES
            or self.name.endswith(".llm")
            or self.name in {"ChatOpenAI", "PydanticToolsParser"}
        )

    @property
    def is_tool(self) -> bool:
        """Return True if this span represents a tool invocation."""
        return (
            self.run_type == "tool"
            or self.name.startswith("tool.")
            or self.name in _KNOWN_TOOL_NAMES
            or self.name.startswith("docker_sandbox.")
        )


@dataclass
class AttemptSnapshotRow:
    """Row parsed from the attempt snapshot summary table."""

    task_id: str
    attempt_id: str
    revision: str = ""
    stage: str = ""
    qa_policy: str = ""
    policy_source: str = ""
    selected_version: str = ""
    dispatch: str = ""
    executed_versions: list[str] = field(default_factory=list)
    worker: str = ""
    qa: str = ""
    final_task_status: str = ""
    instruction: str = ""


@dataclass
class TrajectoryDocument:
    """Top-level container for a parsed Phase 5 remediation trajectory."""

    trace_id: str
    repo_root: str = ""
    export_source: str = ""
    span_count: int = 0
    langsmith_trace_url: str | None = None
    exported_at: datetime | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    spans: list[TrajectorySpan] = field(default_factory=list)
    attempt_snapshots: list[AttemptSnapshotRow] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_path: Path | None = None

    @property
    def llm_spans(self) -> list[TrajectorySpan]:
        """Return all spans classified as LLM calls."""
        return [span for span in self.spans if span.is_llm]

    @property
    def tool_spans(self) -> list[TrajectorySpan]:
        """Return all spans classified as tool executions."""
        return [span for span in self.spans if span.is_tool]

    def spans_for_agent(self, agent_name: str) -> list[TrajectorySpan]:
        """Filter spans by agent category or prefix."""
        normalized = agent_name.strip().lower()
        if normalized in {"triage", "initial_triage"}:
            return [
                s
                for s in self.spans
                if s.name in {"triage.llm", "initial_triage", "triage"} or "triage" in s.tags
            ]
        if normalized in {"update", "update_subagent"}:
            return [
                s
                for s in self.spans
                if "update_subagent" in s.name.lower()
                or (s.parent_name and "update_subagent" in s.parent_name.lower())
            ]
        if normalized in {"workaround", "workaround_subagent"}:
            return [
                s
                for s in self.spans
                if "workaround_subagent" in s.name.lower()
                or (s.parent_name and "workaround_subagent" in s.parent_name.lower())
            ]
        if normalized in {"qa", "qa_critic"}:
            return [
                s
                for s in self.spans
                if "qa_critic" in s.name.lower()
                or (s.parent_name and "qa_critic" in s.parent_name.lower())
            ]
        if normalized in {"report", "report_node"}:
            return [
                s
                for s in self.spans
                if "report" in s.name.lower()
                or (s.parent_name and "report" in s.parent_name.lower())
            ]
        return [s for s in self.spans if normalized in s.name.lower()]


# ---------------------------------------------------------------------------
# Markdown & JSON Parsing Utilities
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(
    r"^`{3,}(?:json|text)?\r?\n(.*?)\r?\n`{3,}$",
    re.MULTILINE | re.DOTALL,
)

_METADATA_ITEM_PATTERN = re.compile(
    r"^-\s+(?P<key>[^:]+):\s+(?P<value>.*)$",
    re.MULTILINE,
)

_SPAN_HEADER_PATTERN = re.compile(
    r"^###\s+(?P<index>\d+)\.\s+(?P<name>[^\s\(]+)(?:\s+\(`(?P<type>[^`]+)`\))?",
    re.MULTILINE,
)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Safely parse an ISO format datetime string."""
    if not value or value.strip() in {"", "none", "null", "unavailable"}:
        return None
    cleaned = value.strip().strip("`")
    try:
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_fenced_json(text: str) -> Any:
    """Extract and deserialize JSON from a fenced Markdown code block."""
    if not text:
        return None
    match = _FENCE_PATTERN.search(text.strip())
    raw_content = match.group(1).strip() if match else text.strip()
    if not raw_content or raw_content in {"null", "None"}:
        return None
    try:
        return json.loads(raw_content)
    except Exception:
        return raw_content


def _parse_markdown_table_rows(table_text: str) -> list[list[str]]:
    """Parse pipe-separated Markdown table lines into rows of stripped strings."""
    rows: list[list[str]] = []
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line[1:-1].split("|")]
        # Skip divider row
        if all(re.match(r"^:?-+:?$", cell) for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def parse_trajectory_markdown(
    path_or_content: Path | str,
    content: str | None = None,
) -> TrajectoryDocument:
    """Parse a Phase 5 trajectory Markdown document into a structured TrajectoryDocument.

    Args:
        path_or_content: Path object or path string or markdown string if content is None.
        content: Explicit markdown string content. If omitted and path_or_content is a file,
            the file will be read from disk.

    Returns:
        A fully populated TrajectoryDocument instance.
    """
    raw_path: Path | None = None
    if content is None:
        if isinstance(path_or_content, Path) or (
            isinstance(path_or_content, str) and Path(path_or_content).exists()
        ):
            raw_path = Path(path_or_content)
            markdown_text = raw_path.read_text(encoding="utf-8", errors="replace")
        else:
            markdown_text = str(path_or_content)
    else:
        markdown_text = content
        if isinstance(path_or_content, Path) or (
            isinstance(path_or_content, str) and Path(path_or_content).exists()
        ):
            raw_path = Path(path_or_content)

    # 1. Parse Metadata
    trace_id = ""
    repo_root = ""
    export_source = "unknown"
    span_count = 0
    langsmith_trace_url: str | None = None
    exported_at: datetime | None = None

    metadata_match = re.search(r"## Run Metadata\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL)
    if metadata_match:
        for line in metadata_match.group(1).splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            if "Trace ID:" in line:
                trace_id = line.split("Trace ID:", 1)[1].strip().strip("`")
            elif "Repository:" in line:
                repo_root = line.split("Repository:", 1)[1].strip().strip("`")
            elif "Export source:" in line:
                export_source = line.split("Export source:", 1)[1].strip().strip("`")
            elif "Span count:" in line:
                try:
                    span_count = int(line.split("Span count:", 1)[1].strip().strip("`"))
                except ValueError:
                    span_count = 0
            elif "LangSmith trace:" in line:
                url = line.split("LangSmith trace:", 1)[1].strip()
                langsmith_trace_url = url if url != "unavailable" else None
            elif "Exported at:" in line:
                exported_at = _parse_iso_datetime(line.split("Exported at:", 1)[1])

    # 2. Parse Root States
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}

    input_state_match = re.search(
        r"## Root Input State\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL
    )
    if input_state_match:
        parsed_input = _extract_fenced_json(input_state_match.group(1))
        if isinstance(parsed_input, dict):
            initial_state = parsed_input

    final_state_match = re.search(
        r"## Root Final State\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL
    )
    if final_state_match:
        parsed_final = _extract_fenced_json(final_state_match.group(1))
        if isinstance(parsed_final, dict):
            final_state = parsed_final

    # 3. Parse Attempt Snapshot Summary Table
    attempt_snapshots: list[AttemptSnapshotRow] = []
    snapshots_match = re.search(
        r"## Attempt Snapshot Summary\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL
    )
    if snapshots_match:
        snapshot_rows = _parse_markdown_table_rows(snapshots_match.group(1))
        if snapshot_rows and len(snapshot_rows) > 1:
            for row in snapshot_rows[1:]:
                if not row or row[0].lower() in {"none", "task"}:
                    continue
                task_id = row[0] if len(row) > 0 else ""
                attempt_id = row[1] if len(row) > 1 else ""
                revision = row[2] if len(row) > 2 else ""
                stage = row[3] if len(row) > 3 else ""
                qa_policy = row[4] if len(row) > 4 else ""
                policy_source = row[5] if len(row) > 5 else ""
                selected_version = row[6] if len(row) > 6 else ""
                dispatch = row[7] if len(row) > 7 else ""
                executed_raw = row[8] if len(row) > 8 else ""
                executed_versions = [v.strip() for v in executed_raw.split(",") if v.strip()]
                worker = row[9] if len(row) > 9 else ""
                qa = row[10] if len(row) > 10 else ""
                final_task_status = row[11] if len(row) > 11 else ""
                instruction = row[12] if len(row) > 12 else ""
                attempt_snapshots.append(
                    AttemptSnapshotRow(
                        task_id=task_id,
                        attempt_id=attempt_id,
                        revision=revision,
                        stage=stage,
                        qa_policy=qa_policy,
                        policy_source=policy_source,
                        selected_version=selected_version,
                        dispatch=dispatch,
                        executed_versions=executed_versions,
                        worker=worker,
                        qa=qa,
                        final_task_status=final_task_status,
                        instruction=instruction,
                    )
                )

    # 4. Parse Execution Timeline Table
    timeline_by_sequence: dict[int, dict[str, Any]] = {}
    timeline_match = re.search(
        r"## Execution Timeline\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL
    )
    if timeline_match:
        timeline_rows = _parse_markdown_table_rows(timeline_match.group(1))
        if timeline_rows and len(timeline_rows) > 1:
            for row in timeline_rows[1:]:
                if not row or not row[0].isdigit():
                    continue
                seq = int(row[0])
                name = row[1] if len(row) > 1 else ""
                run_type = row[2] if len(row) > 2 else ""
                parent = row[3] if len(row) > 3 else None
                start_dt = _parse_iso_datetime(row[4]) if len(row) > 4 else None
                end_dt = _parse_iso_datetime(row[5]) if len(row) > 5 else None
                duration: float | None = None
                if len(row) > 6 and row[6]:
                    try:
                        duration = float(row[6])
                    except ValueError:
                        duration = None
                status = row[7] if len(row) > 7 else None
                total_tokens: int | None = None
                if len(row) > 8 and row[8].isdigit():
                    total_tokens = int(row[8])

                timeline_by_sequence[seq] = {
                    "sequence": seq,
                    "name": name,
                    "run_type": run_type,
                    "parent_run_id": parent if parent and parent != "none" else None,
                    "started_at": start_dt,
                    "ended_at": end_dt,
                    "duration_seconds": duration,
                    "status": status,
                    "total_tokens": total_tokens,
                }

    # 5. Parse Span Details
    spans: list[TrajectorySpan] = []
    spans_match = re.search(r"## Span Details\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL)
    if spans_match:
        span_blocks = re.split(r"(?=\n###\s+\d+\.)", "\n" + spans_match.group(1))
        for block in span_blocks:
            block = block.strip()
            if not block.startswith("###"):
                continue

            header_match = _SPAN_HEADER_PATTERN.search(block)
            if not header_match:
                continue

            seq = int(header_match.group("index"))
            name = header_match.group("name")
            run_type = header_match.group("type") or "chain"

            # Parse Run ID / Parent / Tags
            run_id = f"span-{seq}"
            parent_run_id: str | None = None
            tags: list[str] = []

            for line in block.splitlines()[:10]:
                line = line.strip()
                if line.startswith("- Run ID:"):
                    run_id = line.split("- Run ID:", 1)[1].strip().strip("`")
                elif line.startswith("- Parent Run ID:"):
                    parent_val = line.split("- Parent Run ID:", 1)[1].strip().strip("`")
                    parent_run_id = parent_val if parent_val != "none" else None
                elif line.startswith("- Tags:"):
                    tags_val = line.split("- Tags:", 1)[1].strip().strip("`")
                    if tags_val and tags_val != "none":
                        tags = [t.strip() for t in tags_val.split(",") if t.strip()]

            # Parse Inputs / Outputs / Metadata / Errors
            inputs_match = re.search(r"#### Inputs\s*\n(.*?)(?=\n####|\Z)", block, re.DOTALL)
            outputs_match = re.search(r"#### Outputs\s*\n(.*?)(?=\n####|\Z)", block, re.DOTALL)
            serialized_match = re.search(
                r"#### Serialized Metadata\s*\n(.*?)(?=\n####|\Z)", block, re.DOTALL
            )
            metadata_match_sub = re.search(
                r"#### Metadata\s*\n(.*?)(?=\n####|\Z)", block, re.DOTALL
            )
            error_match = re.search(r"#### Error\s*\n(.*?)(?=\n####|\Z)", block, re.DOTALL)

            inputs = _extract_fenced_json(inputs_match.group(1)) if inputs_match else None
            outputs = _extract_fenced_json(outputs_match.group(1)) if outputs_match else None
            serialized = (
                _extract_fenced_json(serialized_match.group(1)) if serialized_match else None
            )
            meta = _extract_fenced_json(metadata_match_sub.group(1)) if metadata_match_sub else {}
            error_text = error_match.group(1).strip() if error_match else None

            # Merge with timeline info
            t_info = timeline_by_sequence.get(seq, {})
            started_at = t_info.get("started_at")
            ended_at = t_info.get("ended_at")
            duration_seconds = t_info.get("duration_seconds")
            status = t_info.get("status")
            total_tokens = t_info.get("total_tokens")

            if parent_run_id is None and t_info.get("parent_run_id"):
                parent_run_id = t_info["parent_run_id"]

            spans.append(
                TrajectorySpan(
                    run_id=run_id,
                    name=name,
                    run_type=run_type,
                    parent_run_id=parent_run_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    inputs=inputs,
                    outputs=outputs,
                    error=error_text,
                    metadata=meta if isinstance(meta, dict) else {},
                    tags=tags,
                    serialized=serialized if isinstance(serialized, dict) else None,
                    sequence=seq,
                    total_tokens=total_tokens,
                    status=status,
                    duration_seconds=duration_seconds,
                )
            )

    # If span details were empty but timeline existed, populate basic spans from timeline
    if not spans and timeline_by_sequence:
        for seq, t_info in sorted(timeline_by_sequence.items()):
            spans.append(
                TrajectorySpan(
                    run_id=f"span-{seq}",
                    name=t_info.get("name", "unknown"),
                    run_type=t_info.get("run_type", "chain"),
                    parent_run_id=t_info.get("parent_run_id"),
                    started_at=t_info.get("started_at"),
                    ended_at=t_info.get("ended_at"),
                    duration_seconds=t_info.get("duration_seconds"),
                    status=t_info.get("status"),
                    total_tokens=t_info.get("total_tokens"),
                    sequence=seq,
                )
            )

    # Resolve parent_name for all spans
    span_name_by_id = {s.run_id: s.name for s in spans}
    for s in spans:
        if s.parent_run_id and s.parent_run_id in span_name_by_id:
            s.parent_name = span_name_by_id[s.parent_run_id]

    # 6. Parse Diagnostics and Warnings
    diagnostics: list[str] = []
    warnings: list[str] = []
    diag_match = re.search(
        r"## Diagnostics and Guardrails\s*\n(.*?)(?=\n### Export Warnings|\Z)",
        markdown_text,
        re.DOTALL,
    )
    if diag_match:
        for line in diag_match.group(1).splitlines():
            line = line.strip()
            if line.startswith("-") and line != "- No explicit diagnostics recorded.":
                diagnostics.append(line[1:].strip())

    warn_match = re.search(r"### Export Warnings\s*\n(.*?)(?=\n##|\Z)", markdown_text, re.DOTALL)
    if warn_match:
        for line in warn_match.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                warnings.append(line[1:].strip())

    if span_count == 0 and spans:
        span_count = len(spans)

    return TrajectoryDocument(
        trace_id=trace_id,
        repo_root=repo_root,
        export_source=export_source,
        span_count=span_count,
        langsmith_trace_url=langsmith_trace_url,
        exported_at=exported_at,
        initial_state=initial_state,
        final_state=final_state,
        spans=spans,
        attempt_snapshots=attempt_snapshots,
        diagnostics=diagnostics,
        warnings=warnings,
        raw_path=raw_path,
    )


def parse_trajectory_dict(data: dict[str, Any]) -> TrajectoryDocument:
    """Parse in-memory dictionary data (e.g. from TrajectoryRecorder) into a TrajectoryDocument.

    Args:
        data: Dict containing trajectory data or spans.

    Returns:
        A populated TrajectoryDocument instance.
    """
    trace_id = str(data.get("trace_id") or data.get("run_id") or "")
    repo_root = str(data.get("repo_root") or "")
    export_source = str(data.get("source") or "dict")
    raw_spans = data.get("spans") or []

    initial_state = data.get("initial_state") or {}
    final_state = data.get("final_state") or {}
    if not isinstance(initial_state, dict):
        initial_state = {}
    if not isinstance(final_state, dict):
        final_state = {}

    spans: list[TrajectorySpan] = []
    for idx, s in enumerate(raw_spans, start=1):
        if not isinstance(s, dict):
            continue
        started = s.get("started_at")
        ended = s.get("ended_at")
        if isinstance(started, str):
            started = _parse_iso_datetime(started)
        if isinstance(ended, str):
            ended = _parse_iso_datetime(ended)

        duration = s.get("duration_seconds")
        if duration is None and isinstance(started, datetime) and isinstance(ended, datetime):
            duration = round((ended - started).total_seconds(), 3)

        spans.append(
            TrajectorySpan(
                run_id=str(s.get("run_id") or f"span-{idx}"),
                name=str(s.get("name") or "unknown"),
                run_type=str(s.get("run_type") or "chain"),
                parent_run_id=s.get("parent_run_id"),
                started_at=started,
                ended_at=ended,
                inputs=s.get("inputs"),
                outputs=s.get("outputs"),
                error=s.get("error"),
                metadata=s.get("metadata") or {},
                tags=list(s.get("tags") or []),
                serialized=s.get("serialized"),
                sequence=int(s.get("sequence") or idx),
                prompt_tokens=s.get("prompt_tokens"),
                completion_tokens=s.get("completion_tokens"),
                total_tokens=s.get("total_tokens"),
                status=s.get("status"),
                duration_seconds=duration,
            )
        )

    span_name_by_id = {s.run_id: s.name for s in spans}
    for s in spans:
        if s.parent_run_id and s.parent_run_id in span_name_by_id:
            s.parent_name = span_name_by_id[s.parent_run_id]

    return TrajectoryDocument(
        trace_id=trace_id,
        repo_root=repo_root,
        export_source=export_source,
        span_count=len(spans),
        initial_state=initial_state,
        final_state=final_state,
        spans=spans,
    )


# ---------------------------------------------------------------------------
# Tool Call Extraction & LLMTestCase Conversion
# ---------------------------------------------------------------------------


def extract_tool_calls(
    spans: list[TrajectorySpan],
    parent_span_id: str | None = None,
) -> list[ToolCall]:
    """Extract tool invocations from a sequence of spans.

    Args:
        spans: List of TrajectorySpan objects.
        parent_span_id: Optional parent span ID to constrain tool extraction.

    Returns:
        List of ToolCall objects representing tool executions.
    """
    tool_calls: list[ToolCall] = []

    for span in spans:
        if parent_span_id is not None and span.parent_run_id != parent_span_id:
            continue

        if span.is_tool:
            clean_name = span.name.removeprefix("tool.")
            params: dict[str, Any] = {}
            if isinstance(span.inputs, dict):
                params = span.inputs
            elif span.inputs is not None:
                params = {"input": span.inputs}

            tool_calls.append(
                ToolCall(
                    name=clean_name,
                    input_parameters=params,
                    output=span.outputs,
                )
            )
        elif span.is_llm and isinstance(span.outputs, dict):
            # Check for embedded tool calls in chat model output messages
            embedded_calls = span.outputs.get("tool_calls")
            if isinstance(embedded_calls, list):
                for tc in embedded_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        tool_calls.append(
                            ToolCall(
                                name=tc.get("name", ""),
                                input_parameters=tc.get("args") or {},
                                output=None,
                            )
                        )

    return tool_calls


def _serialize_for_test_case(value: Any) -> str:
    """Format an input or output object into a string for LLMTestCase."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # If it is a list of chat messages, format cleanly
        formatted: list[str] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = item.get("type", "message")
                formatted.append(f"[{role}]: {item.get('content')}")
            else:
                formatted.append(str(item))
        return "\n\n".join(formatted)
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(value)


def trajectory_to_test_case(
    span: TrajectorySpan,
    doc: TrajectoryDocument | None = None,
    expected_tools: list[ToolCall] | None = None,
) -> LLMTestCase:
    """Convert a single TrajectorySpan into a DeepEval LLMTestCase.

    Args:
        span: The TrajectorySpan to convert.
        doc: Optional parent TrajectoryDocument for extracting child tools and context.
        expected_tools: Optional expected tools list for tool correctness evaluation.

    Returns:
        A typed LLMTestCase instance.
    """
    input_str = _serialize_for_test_case(span.inputs)
    output_str = _serialize_for_test_case(span.outputs)

    context_list: list[str] = []
    if doc and doc.initial_state:
        valid_groups = doc.initial_state.get("valid_groups") or []
        if valid_groups:
            context_list.append(json.dumps(valid_groups, default=str))

    # Extract child tool calls if doc is present
    tools_called: list[ToolCall] | None = None
    if doc:
        child_tools = extract_tool_calls(doc.spans, parent_span_id=span.run_id)
        if child_tools:
            tools_called = child_tools

    if tools_called is None:
        tools_called = extract_tool_calls([span])

    meta: dict[str, Any] = {
        "run_id": span.run_id,
        "span_name": span.name,
        "run_type": span.run_type,
        "parent_name": span.parent_name,
        "trace_id": doc.trace_id if doc else "",
        "status": span.status,
    }

    return LLMTestCase(
        input=input_str,
        actual_output=output_str,
        context=context_list if context_list else None,
        tools_called=tools_called if tools_called else None,
        expected_tools=expected_tools,
        latency=span.duration_seconds,
        cost=None,
        additional_metadata=meta,
    )


def spans_to_test_cases(
    spans: list[TrajectorySpan],
    agent_filter: str | None = None,
    doc: TrajectoryDocument | None = None,
) -> list[LLMTestCase]:
    """Convert a sequence of spans into DeepEval LLMTestCase objects with optional filtering.

    Args:
        spans: List of TrajectorySpan objects.
        agent_filter: Filter category (e.g. 'triage', 'update_subagent',
            'workaround_subagent', 'qa_critic', 'report', or None for all LLM spans).
        doc: Optional parent TrajectoryDocument.

    Returns:
        List of LLMTestCase objects.
    """
    target_spans = spans
    if agent_filter:
        normalized = agent_filter.strip().lower()
        if doc is not None:
            target_spans = doc.spans_for_agent(normalized)
        else:
            if normalized in {"triage", "initial_triage"}:
                target_spans = [
                    s for s in spans if s.name in {"triage.llm", "initial_triage", "triage"}
                ]
            elif normalized in {"update", "update_subagent"}:
                target_spans = [
                    s
                    for s in spans
                    if "update_subagent" in s.name.lower()
                    or (s.parent_name and "update_subagent" in s.parent_name.lower())
                ]
            elif normalized in {"workaround", "workaround_subagent"}:
                target_spans = [
                    s
                    for s in spans
                    if "workaround_subagent" in s.name.lower()
                    or (s.parent_name and "workaround_subagent" in s.parent_name.lower())
                ]
            elif normalized in {"qa", "qa_critic"}:
                target_spans = [
                    s
                    for s in spans
                    if "qa_critic" in s.name.lower()
                    or (s.parent_name and "qa_critic" in s.parent_name.lower())
                ]
            elif normalized in {"report", "report_node"}:
                target_spans = [
                    s
                    for s in spans
                    if "report" in s.name.lower()
                    or (s.parent_name and "report" in s.parent_name.lower())
                ]
            else:
                target_spans = [s for s in spans if normalized in s.name.lower()]

    test_cases: list[LLMTestCase] = []
    for s in target_spans:
        if s.is_llm or s.outputs is not None:
            test_cases.append(trajectory_to_test_case(s, doc=doc))

    return test_cases
