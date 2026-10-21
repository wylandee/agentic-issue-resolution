"""Bridge connecting remediation_engine execution to the demo SSE stream and console log view."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from remediation_engine.api import RemediationRequest, RemediationResult, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue
from remediation_engine.orchestration.trajectory_exporter import TrajectoryRecorder
import remediation_engine.orchestration.graph as graph_module

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

    def emit_console(self, category: str, title: str, details: str | None = None, data: dict[str, Any] | None = None) -> None:
        """Append to run logs and broadcast console event."""
        event = ConsoleEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=_now_iso(),
            category=category,  # type: ignore[arg-type]
            title=title,
            details=details,
            data=data,
        )
        self.state.logs.append(event)
        self.event_queue.put({"event": "console_event", "data": event.model_dump(mode="json")})

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
        super().__init__()
        self.session = session
        self._active_spans: dict[str, dict[str, Any]] = {}

    def _start(self, *, run_id: Any, parent_run_id: Any, name: str, run_type: str, inputs: Any = None, serialized: Any = None, tags: Any = None, metadata: Any = None) -> None:
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

    def _handle_span_start(self, run_id: str, name: str, run_type: str, inputs: Any, metadata: Any) -> None:
        node_name = None
        if isinstance(metadata, Mapping) and "langgraph_node" in metadata:
            node_name = metadata["langgraph_node"]
        elif name in GRAPH_NODES:
            node_name = name

        if node_name:
            self._active_spans[run_id] = {"kind": "node", "name": node_name}
            self.session.emit_run_status("running", current_node=node_name)
            self.session.emit_console(
                category="NODE_TRANSITION",
                title=f"Node Started: {node_name}",
                details=f"Entering {node_name} execution step",
            )
            return

        # Check tool calls
        if name.startswith("tool.") or run_type == "tool":
            tool_name = name[5:] if name.startswith("tool.") else name
            self._active_spans[run_id] = {"kind": "tool", "name": tool_name}
            
            # Extract key arguments for a concise log
            details = None
            if isinstance(inputs, Mapping):
                # Filter out verbose file contents
                clean_args = {k: v for k, v in inputs.items() if k not in {"file_contents", "diff", "content"}}
                details = f"args: {json.dumps(clean_args, default=str)}"
            elif isinstance(inputs, str):
                details = inputs[:200]
                
            self.session.emit_console(
                category="TOOL_CALL",
                title=f"Tool Invocation: {tool_name}",
                details=details,
            )
            return

        # Check key deterministic stages
        if name in {"triage.pipeline", "triage.grouping", "triage.reachability"}:
            self._active_spans[run_id] = {"kind": "det", "name": name}
            self.session.emit_console(
                category="DETERMINISTIC_FN",
                title=f"Deterministic Execution: {name}",
                details="Running deterministic triage pipeline analysis",
            )

    def _handle_span_end(self, run_id: str, outputs: Any, error: Any) -> None:
        span_info = self._active_spans.pop(run_id, None)
        if not span_info:
            return

        kind = span_info["kind"]
        name = span_info["name"]

        if error:
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Failed: {name}",
                details=f"Error: {error}",
            )
            return

        if kind == "node":
            # Process node output for findings dashboard updates
            if isinstance(outputs, Mapping):
                dashboard_updates = self.session.tracker.process_node_state_update(name, dict(outputs))
                for upd in dashboard_updates:
                    self.session.emit_dashboard_update(upd)

                # Emit key results for nodes
                self._emit_node_key_results(name, dict(outputs))

            self.session.emit_console(
                category="NODE_TRANSITION",
                title=f"Node Finished: {name}",
                details=f"Completed {name} execution",
            )

        elif kind == "tool":
            summary = "Tool executed successfully"
            if isinstance(outputs, str) and len(outputs) < 250:
                summary = outputs
            elif isinstance(outputs, Mapping):
                summary = json.dumps({k: v for k, v in outputs.items() if k not in {"raw_content"}}, default=str)[:200]
            self.session.emit_console(
                category="TOOL_CALL",
                title=f"Tool Completed: {name}",
                details=summary,
            )

        elif kind == "det":
            self.session.emit_console(
                category="DETERMINISTIC_FN",
                title=f"Completed: {name}",
                details="Deterministic computation finished",
            )

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
            targets = patch.get("active_target_task_ids") or []
            audit = patch.get("decision_code")
            self.session.emit_console(
                category="KEY_RESULT",
                title=f"Supervisor Route: {step}",
                details=f"Decision: {audit or 'ROUTED'} -> Target tasks: {targets or 'None'}",
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
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
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
        session.state.report_available = bool(result.report_path or raw_state.get("report_markdown"))
        session.state.report_path = result.report_path
        session.state.diff = result.diff
        session.state.errors = list(result.errors or [])

        final_status = "completed" if result.status == "completed" and not result.errors else result.status
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
            "Target packages must have findings in this workspace: "
            + ", ".join(unknown_packages)
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
