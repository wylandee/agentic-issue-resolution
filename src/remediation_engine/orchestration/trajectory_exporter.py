"""Local and LangSmith-backed trajectory export for Phase 5 runs.

The exporter deliberately keeps the trace format plain Markdown with JSON
payloads.  This makes the resulting artifact easy for another LLM (or a human)
to inspect without requiring access to LangSmith.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TRAJECTORY_DIR = _DEFAULT_PROJECT_ROOT / "data" / "trajectories"
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|password|passwd|secret|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token)$",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?P<prefix>bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(?P<labeled>(?:api[_ -]?key|authorization|password|passwd|secret)\s*[:=]\s*)\S+|"
    r"\b(?:sk|pk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9._-]{10,}"
)
_ACTIVE_RECORDER: ContextVar[TrajectoryRecorder | None] = ContextVar(
    "phase5_trajectory_recorder",
    default=None,
)


def _redact_string(value: str) -> str:
    """Redact common credential formats without altering ordinary prompt text."""

    def replace(match: re.Match[str]) -> str:
        """Replace a matched secret with a redaction marker."""
        prefix = match.group("prefix") or match.group("labeled") or ""
        return f"{prefix}[REDACTED]"

    return _SECRET_VALUE_RE.sub(replace, value)


def to_jsonable(value: Any, *, key: str | None = None) -> Any:
    """Convert common LangChain/Pydantic/runtime objects into safe JSON data."""
    if key and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (uuid.UUID, Path)):
        return str(value)
    if isinstance(value, BaseMessage):
        return {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": to_jsonable(getattr(value, "content", "")),
            "additional_kwargs": to_jsonable(
                getattr(value, "additional_kwargs", {}), key="additional_kwargs"
            ),
            "response_metadata": to_jsonable(
                getattr(value, "response_metadata", {}), key="response_metadata"
            ),
            "tool_calls": to_jsonable(getattr(value, "tool_calls", [])),
        }
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json"), key=key)
        except Exception:  # pragma: no cover - defensive serialization path
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return to_jsonable(value.dict(), key=key)
        except Exception:  # pragma: no cover - defensive serialization path
            pass
    if is_dataclass(value):
        return to_jsonable(asdict(value), key=key)
    if isinstance(value, Mapping):
        return {
            str(item_key): to_jsonable(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value), key=key)
        except Exception:  # pragma: no cover - defensive serialization path
            pass
    return _redact_string(str(value))


def json_text(value: Any) -> str:
    """Serialize a payload for a Markdown JSON code block."""
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


@dataclass
class _LocalSpan:
    run_id: str
    name: str
    run_type: str
    parent_run_id: str | None
    started_at: datetime
    ended_at: datetime | None = None
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    serialized: Any = None
    sequence: int = 0


def _identifier(value: Any) -> str:
    return str(value) if value is not None else ""


def _span_name(serialized: Any, fallback: str) -> str:
    if isinstance(serialized, Mapping):
        name = serialized.get("name") or serialized.get("id")
        if isinstance(name, list) and name:
            name = name[-1]
        if name:
            return str(name).split(".")[-1]
    return fallback


def _token_usage_from_response(response: Any) -> tuple[int, int, bool]:
    """Extract prompt/completion token usage from a LangChain LLM response."""
    candidates: list[Any] = []
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        candidates.append(usage_metadata)
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        candidates.extend([llm_output.get("token_usage"), llm_output.get("usage"), llm_output])
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        candidates.extend([metadata.get("token_usage"), metadata.get("usage"), metadata])
    for generation_group in getattr(response, "generations", []) or []:
        for generation in generation_group or []:
            message = getattr(generation, "message", None)
            message_usage = getattr(message, "usage_metadata", None)
            if isinstance(message_usage, Mapping):
                candidates.append(message_usage)
            generation_metadata = getattr(message, "response_metadata", None)
            if isinstance(generation_metadata, Mapping):
                candidates.extend(
                    [
                        generation_metadata.get("token_usage"),
                        generation_metadata.get("usage"),
                        generation_metadata,
                    ]
                )

    for usage in candidates:
        if not isinstance(usage, Mapping):
            continue
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            prompt_value = max(0, int(prompt)) if prompt is not None else 0
            completion_value = max(0, int(completion)) if completion is not None else 0
        except (TypeError, ValueError):
            continue
        available = prompt is not None or completion is not None
        if available:
            return prompt_value, completion_value, True
    return 0, 0, False


class TrajectoryRecorder(BaseCallbackHandler):
    """Capture callback spans for a single Phase 5 invocation.

    Callback failures are intentionally swallowed so observability can never
    change remediation behavior.
    """

    raise_error = False

    def __init__(self) -> None:
        """Initialize an empty span and token-usage recorder."""
        super().__init__()
        self._lock = RLock()
        self._spans: dict[str, _LocalSpan] = {}
        self._order: list[str] = []
        self._sequence = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._token_data_available = False

    def _start(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        run_type: str,
        inputs: Any = None,
        serialized: Any = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            identifier = _identifier(run_id)
            with self._lock:
                self._sequence += 1
                self._spans[identifier] = _LocalSpan(
                    run_id=identifier,
                    name=name,
                    run_type=run_type,
                    parent_run_id=_identifier(parent_run_id) or None,
                    started_at=datetime.now(UTC),
                    inputs=to_jsonable(inputs),
                    serialized=to_jsonable(serialized),
                    tags=list(tags or []),
                    metadata=to_jsonable(dict(metadata or {})),
                    sequence=self._sequence,
                )
                self._order.append(identifier)
        except Exception:  # pragma: no cover - telemetry must not interrupt a run
            logger.debug("trajectory recorder failed to start span", exc_info=True)

    def _end(
        self,
        run_id: Any,
        *,
        outputs: Any = None,
        error: Any | None = None,
    ) -> None:
        try:
            identifier = _identifier(run_id)
            with self._lock:
                span = self._spans.get(identifier)
                if span is None:
                    self._sequence += 1
                    span = _LocalSpan(
                        run_id=identifier,
                        name="unknown",
                        run_type="chain",
                        parent_run_id=None,
                        started_at=datetime.now(UTC),
                        sequence=self._sequence,
                    )
                    self._spans[identifier] = span
                    self._order.append(identifier)
                span.ended_at = datetime.now(UTC)
                if outputs is not None:
                    span.outputs = to_jsonable(outputs)
                if error is not None:
                    span.error = _redact_string(str(error))
        except Exception:  # pragma: no cover - telemetry must not interrupt a run
            logger.debug("trajectory recorder failed to finish span", exc_info=True)

    def record_manual(
        self,
        *,
        name: str,
        run_type: str,
        inputs: Any = None,
        outputs: Any = None,
        error: Any | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        """Record an explicit root/state/runtime event when callbacks are absent."""
        identifier = f"manual-{uuid.uuid4()}"
        self._start(
            run_id=identifier,
            parent_run_id=parent_run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
        )
        self._end(identifier, outputs=outputs, error=error)
        return identifier

    def sequence_count(self) -> int:
        """Return the number of spans recorded so far."""
        with self._lock:
            return self._sequence

    def has_span_since(self, sequence: int, run_type: str | None = None) -> bool:
        """Return whether a span of the requested type started after a sequence."""
        with self._lock:
            return any(
                span.sequence > sequence and (run_type is None or span.run_type == run_type)
                for span in self._spans.values()
            )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of a LangChain chain callback."""
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_span_name(serialized, "chain"),
            run_type="chain",
            inputs=inputs,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> None:
        """Record successful completion of a chain callback."""
        self._end(run_id, outputs=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        """Record a failed chain callback."""
        self._end(run_id, error=error)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of an LLM callback."""
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_span_name(serialized, "llm"),
            run_type="llm",
            inputs=prompts,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of a chat-model callback."""
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_span_name(serialized, "chat_model"),
            run_type="llm",
            inputs=messages,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        """Record successful completion of an LLM callback."""
        self._end(run_id, outputs=response)
        try:
            prompt_tokens, completion_tokens, available = _token_usage_from_response(response)
            if available:
                with self._lock:
                    self._total_prompt_tokens += prompt_tokens
                    self._total_completion_tokens += completion_tokens
                    self._token_data_available = True
        except Exception:  # pragma: no cover - telemetry must not interrupt a run
            logger.debug("trajectory recorder failed to extract token usage", exc_info=True)

    @property
    def total_prompt_tokens(self) -> int:
        """Return accumulated prompt tokens from LLM callbacks."""
        with self._lock:
            return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Return accumulated completion tokens from LLM callbacks."""
        with self._lock:
            return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        """Return accumulated prompt and completion tokens."""
        with self._lock:
            return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def token_data_available(self) -> bool:
        """Return whether at least one LLM callback supplied token usage."""
        with self._lock:
            return self._token_data_available

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        """Record a failed LLM callback."""
        self._end(run_id, error=error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of a tool callback."""
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_span_name(serialized, "tool"),
            run_type="tool",
            inputs=inputs if inputs is not None else input_str,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        """Record successful completion of a tool callback."""
        self._end(run_id, outputs=output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        """Record a failed tool callback."""
        self._end(run_id, error=error)

    def spans(self) -> list[dict[str, Any]]:
        """Return recorded spans in deterministic insertion order."""
        with self._lock:
            result = []
            for identifier in self._order:
                span = self._spans[identifier]
                result.append(
                    {
                        "run_id": span.run_id,
                        "name": span.name,
                        "run_type": span.run_type,
                        "parent_run_id": span.parent_run_id,
                        "started_at": span.started_at,
                        "ended_at": span.ended_at,
                        "inputs": span.inputs,
                        "outputs": span.outputs,
                        "error": span.error,
                        "metadata": span.metadata,
                        "tags": span.tags,
                        "serialized": span.serialized,
                        "sequence": span.sequence,
                    }
                )
            return result


@contextmanager
def use_trajectory_recorder(recorder: TrajectoryRecorder) -> Iterator[TrajectoryRecorder]:
    """Make the current recorder available to nested runtime helpers."""
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def get_active_trajectory_recorder() -> TrajectoryRecorder | None:
    """Return the recorder bound to the current execution context, if any."""
    return _ACTIVE_RECORDER.get()


def invoke_with_trajectory(
    name: str,
    invoke: Any,
    inputs: Any,
    *,
    run_type: str = "llm",
) -> Any:
    """Invoke a callable and add a manual span only if callbacks missed it."""
    recorder = get_active_trajectory_recorder()
    if recorder is None:
        return invoke()
    before = recorder.sequence_count()
    try:
        outputs = invoke()
    except BaseException as exc:
        if not recorder.has_span_since(before, run_type):
            recorder.record_manual(
                name=name,
                run_type=run_type,
                inputs=inputs,
                error=exc,
            )
        raise
    if not recorder.has_span_since(before, run_type):
        recorder.record_manual(
            name=name,
            run_type=run_type,
            inputs=inputs,
            outputs=outputs,
        )
    return outputs


def default_trajectory_dir() -> Path:
    """Return the configured directory for local trajectory exports."""
    configured = os.environ.get("REMEDIATION_TRAJECTORY_DIR", "").strip()
    return Path(configured) if configured else _DEFAULT_TRAJECTORY_DIR


def _remote_span(run: Any, sequence: int) -> dict[str, Any]:
    return {
        "run_id": _identifier(getattr(run, "id", "")),
        "name": str(getattr(run, "name", "unknown")),
        "run_type": str(getattr(run, "run_type", "chain")),
        "parent_run_id": _identifier(getattr(run, "parent_run_id", "")) or None,
        "started_at": getattr(run, "start_time", None),
        "ended_at": getattr(run, "end_time", None),
        "inputs": to_jsonable(getattr(run, "inputs", {})),
        "outputs": to_jsonable(getattr(run, "outputs", {})),
        "error": _redact_string(str(getattr(run, "error", "")))
        if getattr(run, "error", None)
        else None,
        "metadata": to_jsonable(getattr(run, "extra", {}) or {}),
        "tags": list(getattr(run, "tags", []) or []),
        "serialized": to_jsonable(getattr(run, "serialized", {})),
        "sequence": sequence,
        "prompt_tokens": getattr(run, "prompt_tokens", None),
        "completion_tokens": getattr(run, "completion_tokens", None),
        "total_tokens": getattr(run, "total_tokens", None),
        "status": getattr(run, "status", None),
    }


def _flatten_runs(run: Any) -> list[Any]:
    result = [run]
    for child in getattr(run, "child_runs", None) or []:
        result.extend(_flatten_runs(child))
    return result


def fetch_langsmith_spans(run_id: uuid.UUID | str) -> list[dict[str, Any]]:
    """Fetch all spans for a LangSmith trace, including nested child runs."""
    wait_for_all_tracers()
    client = Client()
    runs = list(client.list_runs(trace_id=run_id, limit=None))
    if not runs:
        root = client.read_run(run_id, load_child_runs=True)
        runs = _flatten_runs(root)
    if not runs:
        raise RuntimeError(f"LangSmith returned no spans for trace {run_id}")
    unique: dict[str, Any] = {}
    for run in runs:
        unique[_identifier(getattr(run, "id", ""))] = run
    spans = [_remote_span(run, index) for index, run in enumerate(unique.values(), start=1)]
    return sorted(spans, key=lambda item: (str(item.get("started_at") or ""), item["sequence"]))


def _format_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _duration_seconds(span: Mapping[str, Any]) -> float | None:
    start = span.get("started_at")
    end = span.get("ended_at")
    if isinstance(start, datetime) and isinstance(end, datetime):
        return round((end - start).total_seconds(), 3)
    return None


def _fence(payload: Any) -> str:
    text = json_text(payload)
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(4, longest + 1)
    return f"{fence}json\n{text}\n{fence}"


def _table_value(value: Any) -> str:
    # Preserve meaningful false-y values such as ``False`` and ``0`` in the
    # audit table.  Treating them as empty strings made failed QA results look
    # like QA had not run at all.
    return str("" if value is None else value).replace("|", "\\|").replace("\n", " ")


def _render_markdown(
    *,
    trace_id: str,
    repo_root: str,
    initial_state: Any,
    final_state: Any,
    spans: list[dict[str, Any]],
    source: str,
    langsmith_url: str | None,
    warnings: Sequence[str],
    run_error: str | None,
    trajectory_path: Path,
) -> str:
    root_output = dict(final_state) if isinstance(final_state, Mapping) else final_state
    if isinstance(root_output, dict):
        root_output["trajectory_path"] = str(trajectory_path)
    errors = []
    if isinstance(final_state, Mapping):
        errors.extend(final_state.get("errors") or [])
    if run_error:
        errors.append(run_error)

    lines = [
        "# Phase 5 Remediation Trajectory",
        "",
        "## Run Metadata",
        "",
        f"- Trace ID: `{trace_id}`",
        f"- Repository: `{repo_root}`",
        f"- Export source: `{source}`",
        f"- Span count: `{len(spans)}`",
        f"- LangSmith trace: {langsmith_url or 'unavailable'}",
        f"- Exported at: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Root Input State",
        "",
        _fence(initial_state),
        "",
        "## Root Final State",
        "",
        _fence(root_output),
        "",
        "## Attempt Snapshot Summary",
        "",
        "| Task | Attempt | Revision | Stage | Selected Version | Dispatch | Executed Versions | Worker | QA | Final Task Status | Instruction |",
        "|---|---|---:|---|---|---|---|---|---|---|---|",
    ]
    if isinstance(final_state, Mapping):
        task_queue = final_state.get("task_queue") or {}
        snapshots = final_state.get("attempt_snapshots_by_id") or {}
        worker_results = final_state.get("worker_results_by_attempt") or {}
        qa_results = final_state.get("qa_results_by_attempt") or {}
        ordered_snapshots = sorted(
            snapshots.values(),
            key=lambda snapshot: (
                getattr(snapshot, "task_id", "")
                if not isinstance(snapshot, Mapping)
                else snapshot.get("task_id", ""),
                getattr(snapshot, "attempt_number", 0)
                if not isinstance(snapshot, Mapping)
                else snapshot.get("attempt_number", 0),
            ),
        )
        for snapshot in ordered_snapshots:
            data = to_jsonable(snapshot)
            attempt_id = data.get("attempt_id", "")
            task_id = data.get("task_id", "")
            worker = worker_results.get(attempt_id)
            qa = qa_results.get(attempt_id)
            task = task_queue.get(task_id)
            worker_data = to_jsonable(worker) if worker is not None else {}
            qa_data = to_jsonable(qa) if qa is not None else {}
            task_data = to_jsonable(task) if task is not None else {}
            instruction = str(data.get("instruction", ""))
            if len(instruction) > 180:
                instruction = instruction[:177] + "..."
            lines.append(
                "| "
                + " | ".join(
                    [
                        _table_value(task_id),
                        _table_value(attempt_id),
                        _table_value(data.get("task_revision")),
                        _table_value(data.get("strategy_stage")),
                        _table_value(data.get("selected_version") or "none"),
                        _table_value(data.get("dispatch_node")),
                        _table_value(", ".join(worker_data.get("executed_versions", []) or [])),
                        _table_value(worker_data.get("status", "pending")),
                        _table_value((qa_data.get("evaluation") or {}).get("passed", "pending")),
                        _table_value(task_data.get("status", "unknown")),
                        _table_value(instruction),
                    ]
                )
                + " |"
            )
        if not snapshots:
            lines.append("| none | none |  |  |  |  |  |  |  |  | No attempt snapshots recorded |")
    else:
        lines.append("| none | none |  |  |  |  |  |  |  |  | No attempt snapshots recorded |")

    lines.extend(
        [
            "",
            "## Execution Timeline",
            "",
            "| # | Span | Type | Parent | Start | End | Duration (s) | Status | Tokens |",
            "|---:|---|---|---|---|---|---:|---|---:|",
        ]
    )
    for index, span in enumerate(spans, start=1):
        tokens = span.get("total_tokens")
        if tokens is None:
            prompt = span.get("prompt_tokens")
            completion = span.get("completion_tokens")
            tokens = (
                (prompt or 0) + (completion or 0)
                if prompt is not None or completion is not None
                else ""
            )
        status = "error" if span.get("error") else span.get("status") or "completed"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _table_value(span.get("name")),
                    _table_value(span.get("run_type")),
                    _table_value(span.get("parent_run_id")),
                    _table_value(_format_time(span.get("started_at"))),
                    _table_value(_format_time(span.get("ended_at"))),
                    _table_value(_duration_seconds(span)),
                    _table_value(status),
                    _table_value(tokens),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Span Details", ""])
    for index, span in enumerate(spans, start=1):
        lines.extend(
            [
                f"### {index}. {span.get('name')} (`{span.get('run_type')}`)",
                "",
                f"- Run ID: `{span.get('run_id')}`",
                f"- Parent Run ID: `{span.get('parent_run_id') or 'none'}`",
                f"- Tags: `{', '.join(span.get('tags') or []) or 'none'}`",
                "",
                "#### Inputs",
                "",
                _fence(span.get("inputs")),
                "",
                "#### Outputs",
                "",
                _fence(span.get("outputs")),
            ]
        )
        if span.get("serialized"):
            lines.extend(["", "#### Serialized Metadata", "", _fence(span["serialized"])])
        if span.get("metadata"):
            lines.extend(["", "#### Metadata", "", _fence(span["metadata"])])
        if span.get("error"):
            lines.extend(["", "#### Error", "", f"```text\n{span['error']}\n```"])
        lines.append("")

    lines.extend(["## Diagnostics and Guardrails", ""])
    diagnostic_items = list(errors)
    if isinstance(final_state, Mapping):
        plans = final_state.get("retry_plans_by_task") or {}
        for task_id, plan in plans.items():
            plan_data = to_jsonable(plan)
            diagnostic_items.append(
                f"{task_id}: committed planner plan {json.dumps(plan_data, sort_keys=True)}"
            )
        for event in final_state.get("consistency_events") or []:
            event_data = to_jsonable(event)
            diagnostic_items.append("consistency event: " + json.dumps(event_data, sort_keys=True))
    if diagnostic_items:
        lines.extend(f"- {item}" for item in diagnostic_items)
    else:
        lines.append("- No explicit diagnostics recorded.")
    if warnings:
        lines.extend(["", "### Export Warnings", "", *[f"- {warning}" for warning in warnings]])
    lines.append("")
    return "\n".join(lines)


def export_phase5_trajectory(
    *,
    trace_id: uuid.UUID | str,
    repo_root: str,
    initial_state: Any,
    final_state: Any,
    recorder: TrajectoryRecorder,
    langsmith_enabled: bool,
    langsmith_url: str | None = None,
    run_error: BaseException | None = None,
) -> Path:
    """Write one Markdown trajectory, preferring LangSmith spans when possible."""
    warnings: list[str] = []
    spans = recorder.spans()
    source = "local-fallback"
    if langsmith_enabled:
        try:
            spans = fetch_langsmith_spans(trace_id)
            source = "langsmith"
        except Exception as exc:  # noqa: BLE001 - fallback is intentional
            warning = f"LangSmith trace retrieval failed: {exc}"
            warnings.append(warning)
            logger.warning(warning)

    output_dir = default_trajectory_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"phase5_{timestamp}_{trace_id}.md"
    markdown = _render_markdown(
        trace_id=str(trace_id),
        repo_root=repo_root,
        initial_state=initial_state,
        final_state=final_state,
        spans=spans,
        source=source,
        langsmith_url=langsmith_url,
        warnings=warnings,
        run_error=str(run_error) if run_error else None,
        trajectory_path=output_path,
    )
    temporary_path = output_path.with_suffix(".md.tmp")
    temporary_path.write_text(markdown, encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path
