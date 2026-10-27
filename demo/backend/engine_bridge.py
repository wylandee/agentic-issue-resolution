"""Bridge connecting remediation_engine execution to the demo SSE stream and console log view."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import remediation_engine.orchestration.graph as graph_module
from remediation_engine.api import RemediationRequest, RemediationResult, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue
from remediation_engine.orchestration.trajectory_exporter import TrajectoryRecorder, to_jsonable

from .findings_tracker import FindingsTracker
from .models import ConsoleEvent, DashboardUpdate, RunState

logger = logging.getLogger(__name__)

# Active runs registry
_RUN_SESSIONS: dict[str, RunSession] = {}

GRAPH_NODES = {
    "initial_triage",
    "workspace_builder",
    "workspace_setup_scan",
    "supervisor",
    "update_subagent",
    "workaround_subagent",
    "qa_critic",
    "triage",
    "final_full_scan",
    "teardown",
    "report",
}

_GRAPH_NODE_SPAN_ALIASES = {
    "initial_triage": {"triage_node"},
    "workspace_builder": {"run_workspace_builder_node"},
    "workspace_setup_scan": {"run_workspace_setup_scan_node"},
    "supervisor": {"run_supervisor_node"},
    "update_subagent": {"run_update_subagent_from_orchestrator"},
    "workaround_subagent": {"run_workaround_subagent_from_orchestrator"},
    "qa_critic": {"run_qa_critic_from_orchestrator", "run_qa_critic_node"},
    "triage": {"post_qa_triage_node"},
    "final_full_scan": {"run_final_full_scan_node"},
    "teardown": {"run_teardown_node"},
    "report": {"run_report_node"},
}
_TOOL_ARGUMENT_OMIT_KEYS = frozenset(
    {"content", "diff", "file_contents", "messages", "new_text", "old_text", "raw_content"}
)
_TASK_ID_KEYS = frozenset(
    {"active_target_task_ids", "task_id", "task_ids", "target_task_id", "target_task_ids"}
)
_PACKAGE_NAME_KEYS = frozenset({"package", "package_name", "target_package_name"})


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    """Read one field from either a mapping or a typed runtime record."""
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _compact_trace_value(value: Any, omitted_keys: frozenset[str]) -> Any:
    """Redact secrets and remove verbose content fields from trace payloads."""
    if isinstance(value, Mapping):
        sanitized = to_jsonable(value)
        return {
            str(key): _compact_trace_value(item, omitted_keys)
            for key, item in sanitized.items()
            if str(key).casefold() not in omitted_keys
        }
    if isinstance(value, (list, tuple)):
        return [_compact_trace_value(item, omitted_keys) for item in value]
    return to_jsonable(value)


def _compact_trace_text(
    value: Any,
    *,
    max_chars: int,
    omitted_keys: frozenset[str] = _TOOL_ARGUMENT_OMIT_KEYS,
) -> str:
    """Render a sanitized trace payload within a concise character budget."""
    safe_value = _compact_trace_value(value, omitted_keys)
    rendered = (
        safe_value
        if isinstance(safe_value, str)
        else json.dumps(safe_value, ensure_ascii=False, sort_keys=True, default=str)
    )
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 1] + "…"


def _concise_tool_arguments(inputs: Any) -> str:
    """Serialize tool arguments after dropping verbose payloads and redacting secrets."""
    value = inputs
    if isinstance(inputs, str):
        with suppress(TypeError, ValueError):
            value = json.loads(inputs)
    return _compact_trace_text(value, max_chars=400)


def _concise_tool_outcome(outputs: Any) -> str:
    """Return a bounded, sanitized summary of a tool's output."""
    if outputs is None:
        return "Tool executed successfully."
    return _compact_trace_text(outputs, max_chars=320)


def _tool_output_failed(outputs: Any) -> bool:
    """Return whether a completed tool call reported an application-level failure."""
    if isinstance(outputs, Mapping):
        status = str(outputs.get("status") or "").strip().casefold()
        return (
            status in {"error", "failed", "failure"}
            or outputs.get("success") is False
            or bool(outputs.get("error"))
        )
    if isinstance(outputs, str):
        return outputs.lstrip().upper().startswith(("ERROR:", "FAILED:", "FAILURE:"))
    return False


def _collect_trace_values(value: Any, keys: frozenset[str]) -> list[str]:
    """Collect string values stored under matching keys in nested span data."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, item in value.items():
            if str(key).casefold() in keys:
                candidates = item if isinstance(item, (list, tuple, set)) else [item]
                found.extend(
                    str(candidate).strip()
                    for candidate in candidates
                    if isinstance(candidate, (str, int)) and str(candidate).strip()
                )
            else:
                found.extend(_collect_trace_values(item, keys))
        return found
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _collect_trace_values(item, keys)]
    return []


def _graph_node_name(
    name: str,
    metadata: Any,
    active_node_names: set[str] | frozenset[str] = frozenset(),
) -> str | None:
    """Return the first span for a graph node, not its inherited child spans."""
    candidate = (
        name
        if name in GRAPH_NODES
        else next(
            (
                node_name
                for node_name, aliases in _GRAPH_NODE_SPAN_ALIASES.items()
                if name in aliases
            ),
            None,
        )
    )
    if candidate is not None:
        return None if candidate in active_node_names else candidate
    if isinstance(metadata, Mapping):
        inherited_name = metadata.get("langgraph_node")
        if (
            isinstance(inherited_name, str)
            and inherited_name in GRAPH_NODES
            and inherited_name not in active_node_names
        ):
            return inherited_name
    return None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


class RunSession:
    """Encapsulates one remediation run session and its event broadcasting."""

    def __init__(
        self,
        run_id: str,
        workspace_id: str,
        repo_root: str,
        issues: list[VulnerabilityIssue],
        target_packages: list[str],
    ) -> None:
        """Initialize session state without modifying the original issue records.

        Args:
            run_id: Unique identifier for the run.
            workspace_id: Attached workspace identifier.
            repo_root: Host path to the repository being remediated.
            issues: Original scanner issues passed to the remediation engine.
            target_packages: Package allowlist for this run.
        """
        self.run_id = run_id
        self.workspace_id = workspace_id
        self.repo_root = repo_root
        self.issues = issues
        self.target_packages = list(target_packages)
        self.tracker = FindingsTracker(issues)
        self.state = RunState(
            run_id=run_id,
            workspace_id=workspace_id,
            status="pending",
            active_findings=self.tracker.active_findings,
            remediated_count=0,
            new_count=0,
            logs=[],
        )
        self.event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.result: RemediationResult | None = None
        self.thread: threading.Thread | None = None
        self._console_lock = threading.RLock()
        self._task_context_lock = threading.RLock()
        self._task_queue: Mapping[str, Any] = {}
        self._group_components: dict[str, str] = {}
        self._task_labels_by_id: dict[str, str] = {}
        self._task_packages_by_id: dict[str, str | None] = {}
        self._active_target_task_ids: list[str] = []

    def emit_console(
        self,
        category: str,
        title: str,
        details: str | None = None,
        data: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
        status: str | None = None,
        duration_seconds: float | None = None,
        outcome: str | None = None,
        task_label: str | None = None,
    ) -> ConsoleEvent:
        """Append a console event or replace an existing event with the same ID."""
        with self._console_lock:
            existing_index = next(
                (index for index, current in enumerate(self.state.logs) if current.id == event_id),
                None,
            )
            if existing_index is None:
                identifier = event_id or f"evt_{uuid.uuid4().hex[:8]}"
                timestamp = _now_iso()
            else:
                identifier = event_id
                timestamp = self.state.logs[existing_index].timestamp

            event = ConsoleEvent(
                id=identifier,
                timestamp=timestamp,
                category=category,  # type: ignore[arg-type]
                title=title,
                details=details,
                status=status,  # type: ignore[arg-type]
                duration_seconds=duration_seconds,
                outcome=outcome,
                task_label=task_label,
                data=data,
            )
            if existing_index is None:
                self.state.logs.append(event)
            else:
                self.state.logs[existing_index] = event
            self.event_queue.put({"event": "console_event", "data": event.model_dump(mode="json")})
        return event

    def update_task_context(self, state_patch: Mapping[str, Any]) -> None:
        """Cache task/package labels from an orchestration state patch."""
        with self._task_context_lock:
            groups = state_patch.get("valid_groups")
            if isinstance(groups, (list, tuple)):
                self._group_components = {
                    str(_record_value(group, "group_id")): str(component).strip()
                    for group in groups
                    if (component := _record_value(group, "vulnerable_component"))
                    and str(_record_value(group, "group_id") or "").strip()
                    and str(component).strip()
                }

            task_queue = state_patch.get("task_queue")
            if isinstance(task_queue, Mapping):
                self._task_queue = task_queue

            active_targets = state_patch.get("active_target_task_ids")
            if isinstance(active_targets, (list, tuple)):
                self._active_target_task_ids = [
                    str(task_id).strip() for task_id in active_targets if str(task_id).strip()
                ]

            labels_by_id: dict[str, str] = {}
            packages_by_id: dict[str, str | None] = {}
            for queue_id, task in self._task_queue.items():
                task_id = str(_record_value(task, "task_id", queue_id) or queue_id).strip()
                parent_group_id = str(_record_value(task, "parent_group_id", "") or "")
                target_package = _record_value(task, "target_package_name")
                package_name = str(target_package).strip() if target_package else ""
                if not package_name:
                    package_name = self._group_components.get(parent_group_id, "")
                packages_by_id[task_id] = package_name or None
                labels_by_id[task_id] = f"{task_id} — {package_name}" if package_name else task_id
            self._task_labels_by_id = labels_by_id
            self._task_packages_by_id = packages_by_id

    def task_labels_for_ids(self, task_ids: list[str]) -> list[str]:
        """Render task IDs with their authorized package targets when available."""
        with self._task_context_lock:
            return [self._task_labels_by_id.get(task_id, task_id) for task_id in task_ids]

    def resolve_tool_task_label(self, inputs: Any, metadata: Any) -> str | None:
        """Associate a tool span with explicit or currently dispatched task targets."""
        task_ids = list(
            dict.fromkeys(
                _collect_trace_values(metadata, _TASK_ID_KEYS)
                + _collect_trace_values(inputs, _TASK_ID_KEYS)
            )
        )
        package_names = {
            package.casefold() for package in _collect_trace_values(inputs, _PACKAGE_NAME_KEYS)
        }

        with self._task_context_lock:
            if not task_ids and package_names:
                matching_active_tasks = [
                    task_id
                    for task_id in self._active_target_task_ids
                    if (package := self._task_packages_by_id.get(task_id))
                    and package.casefold() in package_names
                ]
                task_ids = matching_active_tasks or [
                    task_id
                    for task_id, package in self._task_packages_by_id.items()
                    if package and package.casefold() in package_names
                ]
            if not task_ids:
                task_ids = list(self._active_target_task_ids)
            labels = [
                self._task_labels_by_id.get(task_id, task_id) for task_id in dict.fromkeys(task_ids)
            ]
        return ", ".join(labels) or None

    def emit_dashboard_update(self, update: DashboardUpdate) -> None:
        """Broadcast dashboard update and sync state."""
        self.state.active_findings = update.active_findings
        self.state.remediated_count = update.remediated_count
        self.state.new_count = update.new_count
        self.event_queue.put({"event": "dashboard_update", "data": update.model_dump(mode="json")})

    def emit_run_status(self, status: str, current_node: str | None = None) -> None:
        """Update run status and notify frontend."""
        self.state.status = status  # type: ignore[assignment]
        if current_node:
            self.state.current_node = current_node
        self.event_queue.put(
            {
                "event": "run_status",
                "data": {
                    "status": self.state.status,
                    "current_node": self.state.current_node,
                    "report_available": self.state.report_available,
                },
            }
        )


class DemoTelemetryRecorder(TrajectoryRecorder):
    """Subclass of TrajectoryRecorder that streams filtered events to the active RunSession."""

    def __init__(self, session: RunSession) -> None:
        """Initialize callback span tracking for one run session.

        Args:
            session: Run session receiving console and dashboard events.
        """
        super().__init__()
        self.session = session
        self._active_spans: dict[str, dict[str, Any]] = {}

    def _start(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        run_type: str,
        inputs: Any = None,
        serialized: Any = None,
        tags: Any = None,
        metadata: Any = None,
    ) -> None:
        super()._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )
        try:
            self._handle_span_start(str(run_id), name, run_type, inputs, metadata)
        except Exception as exc:
            logger.debug("Demo telemetry start handler failed: %s", exc)

    def _end(self, run_id: Any, *, outputs: Any = None, error: Any | None = None) -> None:
        super()._end(run_id, outputs=outputs, error=error)
        try:
            self._handle_span_end(str(run_id), outputs, error)
        except Exception as exc:
            logger.debug("Demo telemetry end handler failed: %s", exc)

    def _handle_span_start(
        self, run_id: str, name: str, run_type: str, inputs: Any, metadata: Any
    ) -> None:
        if str(run_type).casefold() == "tool" or name.startswith("tool."):
            tool_name = name.removeprefix("tool.")
            arguments = _concise_tool_arguments(inputs)
            task_label = self.session.resolve_tool_task_label(inputs, metadata)
            event = self.session.emit_console(
                category="TOOL_CALL",
                title=f"Tool: {tool_name}",
                details=f"Args: {arguments}",
                status="running",
                task_label=task_label,
                data={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "status": "running",
                    "task_label": task_label,
                },
            )
            self._active_spans[run_id] = {
                "kind": "tool",
                "name": tool_name,
                "event_id": event.id,
                "arguments": arguments,
                "task_label": task_label,
                "started_at": time.perf_counter(),
            }
            return

        active_node_names = {
            str(span["name"]) for span in self._active_spans.values() if span.get("kind") == "node"
        }
        node_name = _graph_node_name(name, metadata, active_node_names)
        if node_name:
            self._active_spans[run_id] = {"kind": "node", "name": node_name}
            self.session.emit_run_status("running", current_node=node_name)
            return

        if name in {"triage.pipeline", "triage.grouping", "triage.reachability"}:
            self._active_spans[run_id] = {"kind": "det", "name": name}
            self.session.emit_console(
                category="DETERMINISTIC_FN",
                title=f"Deterministic Execution: {name}",
                details="Running deterministic triage pipeline analysis",
            )

    def _complete_tool_span(
        self,
        span_info: Mapping[str, Any],
        *,
        outputs: Any = None,
        error: Any = None,
        incomplete: bool = False,
    ) -> None:
        """Update a tool's original console row with its terminal outcome."""
        duration = round(max(0.0, time.perf_counter() - span_info["started_at"]), 3)
        if incomplete:
            status = "incomplete"
            outcome = "Tool span ended without a completion event."
        elif error is not None:
            status = "error"
            error_text = str(error).strip() or type(error).__name__
            outcome = _compact_trace_text(error_text, max_chars=320)
        elif _tool_output_failed(outputs):
            status = "error"
            outcome = _concise_tool_outcome(outputs)
        else:
            status = "success"
            outcome = _concise_tool_outcome(outputs)
        arguments = str(span_info["arguments"])
        task_label = span_info.get("task_label")
        self.session.emit_console(
            category="TOOL_CALL",
            title=f"Tool: {span_info['name']}",
            details=f"Args: {arguments}",
            event_id=str(span_info["event_id"]),
            status=status,
            duration_seconds=duration,
            outcome=outcome,
            task_label=task_label,
            data={
                "tool_name": span_info["name"],
                "arguments": arguments,
                "status": status,
                "outcome": outcome,
                "duration_seconds": duration,
                "task_label": task_label,
            },
        )

    def _handle_span_end(self, run_id: str, outputs: Any, error: Any) -> None:
        span_info = self._active_spans.pop(run_id, None)
        if span_info is None:
            return

        kind = span_info["kind"]
        name = span_info["name"]
        if kind == "tool":
            self._complete_tool_span(span_info, outputs=outputs, error=error)
            return

        if error is not None:
            safe_error = _compact_trace_text(str(error), max_chars=400)
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Failed: {name}",
                details=f"Error: {safe_error}",
            )
            return

        if kind == "node":
            if isinstance(outputs, Mapping):
                patch = dict(outputs)
                self.session.update_task_context(patch)
                dashboard_updates = self.session.tracker.process_node_state_update(name, patch)
                for update in dashboard_updates:
                    self.session.emit_dashboard_update(update)
                self._emit_node_key_results(name, patch)
        elif kind == "det":
            self.session.emit_console(
                category="DETERMINISTIC_FN",
                title=f"Completed: {name}",
                details="Deterministic computation finished",
            )

    def finish_incomplete_tool_spans(self) -> None:
        """Mark any started tool events without a completion callback as incomplete."""
        for run_id, span_info in list(self._active_spans.items()):
            if span_info.get("kind") == "tool":
                self._active_spans.pop(run_id, None)
                self._complete_tool_span(span_info, incomplete=True)

    def _emit_node_key_results(self, node_name: str, patch: dict[str, Any]) -> None:
        """Format and emit key milestone summaries."""
        if node_name == "initial_triage":
            groups = patch.get("valid_groups") or []
            status = patch.get("initial_triage_status", "completed")
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Triage Result: {len(groups)} Actionable Groups",
                details=f"Status: {status}. Formed {len(groups)} vulnerability groups for task queue.",
            )
        elif node_name == "workspace_builder":
            vol = patch.get("workspace_volume")
            status = patch.get("status")
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Workspace Built: {vol or 'failed'}",
                details=f"Status: {status}. Dependencies installed in isolated Docker volume.",
            )
        elif node_name == "workspace_setup_scan":
            evidence = patch.get("setup_scan_evidence")
            covered = len(getattr(evidence, "closure_package_names", []) or [])
            self.session.emit_console(
                category="KEY_RESULT",
                title="Workspace Setup Scan Complete",
                details=f"Fingerprinted baseline package-lock closures ({covered} packages covered).",
            )
        elif node_name == "supervisor":
            step = patch.get("next_routing_step")
            raw_targets = patch.get("active_target_task_ids") or []
            target_ids = (
                [str(task_id) for task_id in raw_targets]
                if isinstance(raw_targets, (list, tuple))
                else []
            )
            labels = self.session.task_labels_for_ids(target_ids)
            audit = patch.get("decision_code")
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Supervisor Route: {step}",
                details=f"Decision: {audit or 'ROUTED'} -> Target tasks: {', '.join(labels) or 'None'}",
            )
        elif node_name == "qa_critic":
            eval_status = patch.get("eval_status")
            evals = patch.get("qa_evaluations") or {}
            passed_cnt = sum(1 for e in evals.values() if getattr(e, "passed", False))
            total_cnt = len(evals)
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"QA Evaluation: {eval_status}",
                details=f"Verdicts: {passed_cnt}/{total_cnt} tasks passed QA gates.",
            )
        elif node_name == "teardown":
            files = patch.get("changed_files") or []
            self.session.emit_console(
                category="KEY_RESULT",
                title="Teardown & Cleanup Finished",
                details=f"Workspace volume cleaned up. Changed files: {len(files)}.",
            )


class ScopedEngineLogHandler(logging.Handler):
    """Intercepts important logging.INFO from remediation_engine and emits to the session console."""

    KEY_PHRASES = (
        "installing npm dependencies",
        "running unit tests",
        "eval_status=",
        "routing to '",
        "re-triaging",
        "marked UNFIXABLE",
        "closed task",
        "scan requested_scope=",
    )

    def __init__(self, session: RunSession) -> None:
        """Initialize the scoped log handler.

        Args:
            session: Run session receiving selected engine log messages.
        """
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
        """Emit selected remediation-engine milestones to the run console.

        Args:
            record: Logging record produced by the remediation engine.
        """
        try:
            msg = self.format(record)
            if any(phrase in msg for phrase in self.KEY_PHRASES):
                self.session.emit_console(
                    category="DETERMINISTIC_FN",
                    title="Engine Milestone",
                    details=msg,
                )
        except Exception:
            pass


def _execute_run_thread(session: RunSession) -> None:
    """Target function executed in background thread for run_remediation."""
    original_recorder_cls = graph_module.TrajectoryRecorder
    scoped_handler = ScopedEngineLogHandler(session)
    root_logger = logging.getLogger("remediation_engine")
    recorders: list[DemoTelemetryRecorder] = []
    try:
        session.emit_run_status("running", current_node="initial_triage")
        session.emit_console(
            category="NODE_TRANSITION",
            title="Remediation Engine Started",
            details=(
                f"Initiating run on repository: {session.repo_root}. Scope: "
                f"targeted development ({len(session.target_packages)} packages)."
            ),
        )

        # Non-intrusively bind DemoTelemetryRecorder to graph_module.TrajectoryRecorder
        class BoundRecorder(DemoTelemetryRecorder):
            def __init__(self) -> None:
                super().__init__(session)
                recorders.append(self)

        graph_module.TrajectoryRecorder = BoundRecorder  # type: ignore[misc]
        root_logger.addHandler(scoped_handler)

        request = RemediationRequest(
            repo_root=session.repo_root,  # type: ignore[arg-type]
            issues=session.issues,
            valid_groups=[],
            target_packages=session.target_packages,
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="development",
                primary_language="javascript/nodejs",
            ),
        )

        result = run_remediation(request)
        session.result = result

        # Reconcile final findings with raw state
        raw_state = getattr(result, "raw_state", {}) or {}
        final_update = session.tracker.reconcile_final_state(raw_state)
        session.emit_dashboard_update(final_update)

        # Update session state with report and diff
        session.state.report_available = bool(
            result.report_path or raw_state.get("report_markdown")
        )
        session.state.report_path = result.report_path
        session.state.diff = result.diff
        session.state.errors = list(result.errors or [])

        final_status = (
            "completed" if result.status == "completed" and not result.errors else result.status
        )
        session.emit_run_status(final_status, current_node="report")

        session.emit_console(
            category="KEY_RESULT",
            title=f"Remediation Finished ({final_status})",
            details=(
                f"Status: {final_status}. Remediated: {session.tracker.remediated_count} findings. "
                f"Changed files: {len(result.changed_files)}. Final report ready."
            ),
        )

    except Exception as exc:
        logger.exception("Remediation execution thread encountered an error: %s", exc)
        session.state.errors.append(str(exc))
        session.emit_console(
            category="KEY_RESULT",
            title="Remediation Engine Error",
            details=f"Run terminated with error: {exc}",
        )
        session.emit_run_status("failed", current_node="error")
    finally:
        for recorder in recorders:
            try:
                recorder.finish_incomplete_tool_spans()
            except Exception:
                logger.debug("Failed to close incomplete demo tool spans.", exc_info=True)
        # Restore original recorder class
        graph_module.TrajectoryRecorder = original_recorder_cls  # type: ignore[misc]
        root_logger.removeHandler(scoped_handler)
        # Sentinel to signal end of stream to subscribers
        session.event_queue.put(None)


def start_remediation_run(
    workspace_id: str,
    repo_root: str,
    issues: list[VulnerabilityIssue],
    target_packages: list[str] | None = None,
) -> RunSession:
    """Create and launch one remediation run.

    Args:
        workspace_id: Identifier of the attached demo workspace.
        repo_root: Absolute path to the workspace repository.
        issues: Parsed scanner findings for the workspace.
        target_packages: Optional package allowlist. When omitted, package
            targets are detected from the supplied findings.

    Returns:
        The registered run session with its execution thread started.

    Side effects:
        Registers the session and starts a daemon thread that calls the
        remediation engine.

    Raises:
        ValueError: If no package findings exist or the requested package names
            are not present in the supplied findings.
    """
    available_packages: dict[str, str] = {}
    for issue in issues:
        if issue.package_name and issue.package_name.strip():
            available_packages.setdefault(
                issue.package_name.strip().casefold(),
                issue.package_name.strip(),
            )

    if target_packages is None:
        requested_packages = list(available_packages.values())
    elif not target_packages:
        raise ValueError("At least one detected package is required for targeted remediation.")
    else:
        requested_packages = target_packages

    resolved_packages: dict[str, str] = {}
    unknown_packages: list[str] = []
    for requested in requested_packages:
        package_name = requested.strip()
        key = package_name.casefold()
        if not package_name or key not in available_packages:
            unknown_packages.append(requested)
        else:
            resolved_packages.setdefault(key, available_packages[key])
    if unknown_packages:
        raise ValueError(
            "Target packages must have findings in this workspace: " + ", ".join(unknown_packages)
        )
    if not resolved_packages:
        raise ValueError("No package findings were detected; targeted remediation cannot start.")

    resolved_target_packages = sorted(resolved_packages.values(), key=str.casefold)
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    session = RunSession(run_id, workspace_id, repo_root, issues, resolved_target_packages)
    _RUN_SESSIONS[run_id] = session

    thread = threading.Thread(target=_execute_run_thread, args=(session,), daemon=True)
    session.thread = thread
    thread.start()

    return session


def get_run_session(run_id: str) -> RunSession | None:
    """Retrieve run session by run_id."""
    return _RUN_SESSIONS.get(run_id)
