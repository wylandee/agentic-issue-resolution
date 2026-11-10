"""Focused tests for telemetry."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from demo.backend import workspace_manager
from demo.backend.engine_bridge import DemoTelemetryRecorder, RunSession
from demo.backend.findings_tracker import FindingsTracker
from langchain_core.messages import AIMessage

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.runtime.telemetry import (
    TrajectoryRecorder,
    _render_markdown,
    default_trajectory_dir,
    export_phase5_trajectory,
    fetch_langsmith_spans,
    invoke_with_trajectory,
    json_text,
    use_trajectory_recorder,
)


def test_json_text_redacts_secrets_but_preserves_prompt_context():
    payload = {
        "api_key": "sk-test-secret-value-123456789",
        "prompt": "Explain why the supervisor selected an already attempted version.",
    }

    rendered = json_text(payload)

    assert "sk-test-secret-value-123456789" not in rendered
    assert "[REDACTED]" in rendered
    assert "already attempted version" in rendered


def test_default_trajectory_directory_is_project_data_directory(monkeypatch):
    monkeypatch.delenv("REMEDIATION_TRAJECTORY_DIR", raising=False)

    assert default_trajectory_dir().as_posix().endswith("data/trajectories")


def test_fetch_langsmith_spans_queries_by_trace_id(monkeypatch):
    run = SimpleNamespace(
        id="run-1",
        name="phase5_orchestrator",
        run_type="chain",
        parent_run_id=None,
        start_time=None,
        end_time=None,
        inputs={"status": "pending"},
        outputs={"status": "completed"},
        error=None,
        extra={},
        tags=[],
        serialized={},
        status="completed",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    class FakeClient:
        def __init__(self):
            self.trace_id = None

        def list_runs(self, *, trace_id, limit):
            self.trace_id = trace_id
            assert limit is None
            return iter([run])

        def read_run(self, *args, **kwargs):
            raise AssertionError("read_run should not be needed when list_runs returns spans")

    fake_client = FakeClient()
    monkeypatch.setattr("remediation_engine.runtime.telemetry.Client", lambda: fake_client)
    monkeypatch.setattr("remediation_engine.runtime.telemetry.wait_for_all_tracers", lambda: None)

    spans = fetch_langsmith_spans("trace-123")

    assert fake_client.trace_id == "trace-123"
    assert spans[0]["name"] == "phase5_orchestrator"


def test_fetch_langsmith_spans_waits_for_pending_child_runs(monkeypatch):
    """Remote export settles asynchronous child spans before rendering them."""
    pending = SimpleNamespace(
        id="run-1",
        name="phase5_orchestrator",
        run_type="chain",
        parent_run_id=None,
        start_time=datetime(2026, 8, 21, tzinfo=UTC),
        end_time=None,
        inputs={},
        outputs={},
        error=None,
        extra={},
        tags=[],
        serialized={},
        status="pending",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )
    completed = SimpleNamespace(
        **{
            **vars(pending),
            "end_time": datetime(2026, 8, 21, 0, 0, 1, tzinfo=UTC),
            "status": "completed",
        }
    )

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def list_runs(self, *, trace_id, limit):
            self.calls += 1
            return iter([pending] if self.calls == 1 else [completed])

        def read_run(self, *args, **kwargs):
            raise AssertionError("read_run should not be needed")

    fake_client = FakeClient()
    monkeypatch.setattr("remediation_engine.runtime.telemetry.Client", lambda: fake_client)
    monkeypatch.setattr("remediation_engine.runtime.telemetry.wait_for_all_tracers", lambda: None)
    monkeypatch.setattr("remediation_engine.runtime.telemetry.time.sleep", lambda _seconds: None)

    spans = fetch_langsmith_spans("trace-123")

    assert fake_client.calls == 2
    assert spans[0]["status"] == "completed"
    assert spans[0]["ended_at"] is not None


def test_local_recorder_captures_llm_and_tool_fallback_events():
    recorder = TrajectoryRecorder()

    with use_trajectory_recorder(recorder):
        assert invoke_with_trajectory(
            "test.llm",
            lambda: {"answer": "use 8.5.1"},
            {"messages": ["select the next version"]},
        ) == {"answer": "use 8.5.1"}
        assert (
            invoke_with_trajectory(
                "test.tool",
                lambda: "SUCCESS: registry result",
                {"package_name": "test-pkg"},
                run_type="tool",
            )
            == "SUCCESS: registry result"
        )

    spans = recorder.spans()
    assert [span["name"] for span in spans] == ["test.llm", "test.tool"]
    assert spans[0]["run_type"] == "llm"
    assert spans[1]["run_type"] == "tool"


def test_local_recorder_extracts_message_token_usage_and_provider_cost():
    recorder = TrajectoryRecorder()
    response = AIMessage(
        content="done",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 95},
            },
            "token_cost_usd": 0.00042,
        },
    )

    with use_trajectory_recorder(recorder):
        assert (
            invoke_with_trajectory("test.llm", lambda: response, {"messages": ["prompt"]})
            is response
        )

    assert recorder.token_data_available is True
    assert recorder.total_prompt_tokens == 120
    assert recorder.total_completion_tokens == 30
    assert recorder.total_tokens == 150
    assert recorder.token_cost == 0.00042
    assert recorder.cached_input_tokens == 95


def test_attempt_snapshot_summary_renders_correlated_worker_and_qa_state():
    markdown = _render_markdown(
        trace_id="trace-attempts",
        repo_root="repo",
        initial_state={"state_revision": 1},
        final_state={
            "state_revision": 4,
            "task_queue": {
                "task-1": {
                    "task_id": "task-1",
                    "task_revision": 2,
                    "status": "needs_retry",
                }
            },
            "attempt_snapshots_by_id": {
                "attempt-2": {
                    "attempt_id": "attempt-2",
                    "task_id": "task-1",
                    "task_revision": 2,
                    "attempt_number": 2,
                    "strategy_stage": "npm_latest",
                    "qa_policy": "version_bump",
                    "selected_version": "8.5.1",
                    "dispatch_node": "update_subagent",
                    "instruction": "Update package.json to 8.5.1.",
                    "instruction_digest": "digest-2",
                }
            },
            "worker_results_by_attempt": {
                "attempt-2": {
                    "status": "surrender",
                    "executed_versions": ["8.5.1"],
                }
            },
            "qa_results_by_attempt": {
                "attempt-2": {
                    "qa_policy": "version_bump",
                    "qa_policy_source": "attempt_snapshot",
                    "evaluation": {"passed": False},
                }
            },
            "consistency_events": [],
        },
        spans=[],
        source="local-fallback",
        langsmith_url=None,
        warnings=[],
        run_error=None,
        trajectory_path=Path("data/trajectories/trace-attempts.md"),
    )

    assert "## Attempt Snapshot Summary" in markdown
    assert "attempt-2" in markdown
    assert "8.5.1" in markdown
    assert "version_bump" in markdown
    assert "attempt_snapshot" in markdown
    assert "Update package.json to 8.5.1." in markdown
    assert '"passed": false' in markdown


def test_export_writes_one_markdown_file_with_root_state_and_span_details(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()
    recorder.record_manual(
        name="supervisor.router",
        run_type="llm",
        inputs={"prompt": "route task-1"},
        outputs={"next_node": "update_subagent"},
    )
    trace_id = uuid4()

    path = export_phase5_trajectory(
        trace_id=trace_id,
        repo_root="D:/repos/juice-shop",
        initial_state={"status": "pending", "valid_groups": []},
        final_state={"status": "completed", "errors": []},
        recorder=recorder,
        langsmith_enabled=False,
    )

    assert path.parent == tmp_path
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Phase 5 Remediation Trajectory" in content
    assert "## Root Input State" in content
    assert '"status": "pending"' in content
    assert "supervisor.router" in content
    assert "route task-1" in content
    assert "local-fallback" in content
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_local_trajectory_exists_before_langsmith_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()
    recorder.record_manual(
        name="phase5.root_input",
        run_type="state",
        inputs={"status": "pending"},
    )
    observed: dict[str, str] = {}

    def fetch_during_export(_trace_id):
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        observed["content"] = files[0].read_text(encoding="utf-8")
        raise RuntimeError("trace unavailable")

    monkeypatch.setattr(
        "remediation_engine.runtime.telemetry.fetch_langsmith_spans",
        fetch_during_export,
    )

    path = export_phase5_trajectory(
        trace_id="trace-local-first",
        repo_root="repo",
        initial_state={"status": "pending"},
        final_state={"status": "completed_with_errors"},
        recorder=recorder,
        langsmith_enabled=True,
    )

    assert path.exists()
    assert '"status": "pending"' in observed["content"]
    assert "Export source: `local-fallback`" in path.read_text(encoding="utf-8")


def test_local_trajectory_survives_keyboard_interrupt_during_langsmith_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()

    def interrupt_during_fetch(_trace_id):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "remediation_engine.runtime.telemetry.fetch_langsmith_spans",
        interrupt_during_fetch,
    )

    with pytest.raises(KeyboardInterrupt):
        export_phase5_trajectory(
            trace_id="trace-interrupted",
            repo_root="repo",
            initial_state={"status": "pending"},
            final_state={"status": "completed_with_errors"},
            recorder=recorder,
            langsmith_enabled=True,
        )

    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert '"status": "pending"' in files[0].read_text(encoding="utf-8")


def test_langsmith_spans_are_preferred_and_rendered(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()
    spans = [
        {
            "run_id": "child-1",
            "name": "supervisor.router",
            "run_type": "llm",
            "parent_run_id": "root",
            "started_at": "2026-07-21T00:00:02+00:00",
            "ended_at": "2026-07-21T00:00:03+00:00",
            "inputs": {"messages": ["route"]},
            "outputs": {"next_node": "qa_critic"},
            "metadata": {},
            "tags": [],
            "sequence": 2,
        },
        {
            "run_id": "root",
            "name": "phase5_orchestrator",
            "run_type": "chain",
            "parent_run_id": None,
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:04+00:00",
            "inputs": {"status": "pending"},
            "outputs": {"status": "completed"},
            "metadata": {},
            "tags": [],
            "sequence": 1,
        },
    ]
    monkeypatch.setattr(
        "remediation_engine.runtime.telemetry.fetch_langsmith_spans",
        lambda _trace_id: spans,
    )

    path = export_phase5_trajectory(
        trace_id="root",
        repo_root="D:/repos/juice-shop",
        initial_state={"status": "pending"},
        final_state={"status": "completed"},
        recorder=recorder,
        langsmith_enabled=True,
        langsmith_url="https://smith.langchain.com/runs/root",
    )

    content = path.read_text(encoding="utf-8")
    assert "Export source: `langsmith`" in content
    assert "https://smith.langchain.com/runs/root" in content
    assert "supervisor.router" in content
    assert '"next_node": "qa_critic"' in content


def test_langsmith_retrieval_failure_writes_local_fallback_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()
    recorder.record_manual(
        name="phase5.root_input",
        run_type="state",
        inputs={"status": "pending"},
    )
    monkeypatch.setattr(
        "remediation_engine.runtime.telemetry.fetch_langsmith_spans",
        lambda _trace_id: (_ for _ in ()).throw(RuntimeError("trace unavailable")),
    )

    path = export_phase5_trajectory(
        trace_id="trace-fallback",
        repo_root="D:/repos/juice-shop",
        initial_state={"status": "pending"},
        final_state={"status": "failed", "errors": ["router failed"]},
        recorder=recorder,
        langsmith_enabled=True,
    )

    content = path.read_text(encoding="utf-8")
    assert "Export source: `local-fallback`" in content
    assert "LangSmith trace retrieval failed: trace unavailable" in content
    assert "router failed" in content


def test_repeated_runs_create_separate_files(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(tmp_path))
    recorder = TrajectoryRecorder()
    first = export_phase5_trajectory(
        trace_id="trace-one",
        repo_root="repo",
        initial_state={},
        final_state={},
        recorder=recorder,
        langsmith_enabled=False,
    )
    second = export_phase5_trajectory(
        trace_id="trace-two",
        repo_root="repo",
        initial_state={},
        final_state={},
        recorder=recorder,
        langsmith_enabled=False,
    )

    assert first != second
    assert first.exists() and second.exists()
    assert len(list(tmp_path.glob("*.md"))) == 2


_CVE = "CVE-2025-12345"
_GHSA = "GHSA-ABCD-EFGH-IJKL"


@pytest.fixture
def clean_workspaces():
    existing_ids = set(workspace_manager._WORKSPACES)
    yield
    for workspace_id in set(workspace_manager._WORKSPACES) - existing_ids:
        workspace_manager._WORKSPACES.pop(workspace_id, None)


def _issue(
    *,
    cve_id: str | None = _CVE,
    ghsa_id: str | None = None,
    package_name: str = "express-jwt",
    package_version: str | None = "1.0.0",
    file_path: str | None = "package-lock.json",
    severity: Severity = Severity.HIGH,
) -> VulnerabilityIssue:
    identifier = cve_id or ghsa_id or "scanner-rule"
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=severity,
        cve_id=cve_id,
        ghsa_id=ghsa_id,
        package_name=package_name,
        package_version=package_version,
        file_path=file_path,
        message=f"Details for {identifier}",
    )


def _write_issue_report(repo_path: Path, issues: list[VulnerabilityIssue]) -> None:
    report_path = repo_path / "baseline-issues.jsonl"
    report_path.write_text(
        "\n".join(issue.model_dump_json() for issue in issues) + "\n",
        encoding="utf-8",
    )


def test_local_and_uploaded_workspaces_share_canonical_findings(
    tmp_path: Path, clean_workspaces
) -> None:
    issues = [
        _issue(ghsa_id=_GHSA, severity=Severity.HIGH),
        _issue(cve_id=None, ghsa_id=_GHSA, severity=Severity.MEDIUM),
        _issue(ghsa_id=_GHSA, file_path="apps/other/package-lock.json", severity=Severity.LOW),
        _issue(ghsa_id=_GHSA, package_version="2.0.0", severity=Severity.MEDIUM),
    ]
    repo_path = tmp_path / "repository"
    repo_path.mkdir()
    _write_issue_report(repo_path, issues)

    local_info = workspace_manager.create_workspace_from_local_path(str(repo_path))
    uploaded_info = workspace_manager.finalize_uploaded_workspace("ws_uploaded", repo_path)

    assert local_info.total_findings == uploaded_info.total_findings == 3
    assert len(local_info.findings) == len(uploaded_info.findings) == 3
    assert (
        local_info.severity_counts
        == uploaded_info.severity_counts
        == {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 1,
            "LOW": 1,
            "INFO": 0,
            "UNKNOWN": 0,
        }
    )
    assert [finding.model_dump() for finding in local_info.findings] == [
        finding.model_dump() for finding in uploaded_info.findings
    ]

    linked = next(finding for finding in local_info.findings if finding.package_version == "1.0.0")
    assert linked.vuln_id == _CVE
    assert linked.aliases == [_GHSA]
    assert linked.severity == "HIGH"

    stored_issues = workspace_manager.get_workspace_data(local_info.workspace_id)["issues"]
    original_snapshots = [issue.model_dump(mode="json") for issue in stored_issues]
    tracker = FindingsTracker(stored_issues)
    summary = tracker.get_dashboard_summary()
    assert len(tracker.active_findings) == local_info.total_findings
    assert summary["severity_counts"] == local_info.severity_counts
    assert [issue.model_dump(mode="json") for issue in stored_issues] == original_snapshots


def test_alias_scan_update_merges_into_existing_row_without_inflating_counts() -> None:
    baseline = _issue()
    original_snapshot = baseline.model_dump(mode="json")
    tracker = FindingsTracker([baseline])

    updates = tracker.add_new_scan_findings([_issue(ghsa_id=_GHSA)])

    assert len(updates) == 1
    assert updates[0].type == "finding_updated"
    assert len(tracker.active_findings) == 1
    assert tracker.new_count == 0
    assert tracker.active_findings[0].vuln_id == _CVE
    assert tracker.active_findings[0].aliases == [_GHSA]
    assert baseline.model_dump(mode="json") == original_snapshot


@pytest.mark.parametrize(
    "occurrence_change",
    [
        {"package_name": "another-package"},
        {"package_version": "2.0.0"},
        {"file_path": "apps/other/package-lock.json"},
    ],
)
def test_same_identifier_in_distinct_package_occurrence_stays_separate(
    occurrence_change: dict[str, Any],
) -> None:
    tracker = FindingsTracker([_issue()])

    updates = tracker.add_new_scan_findings([_issue(ghsa_id=_GHSA, **occurrence_change)])

    assert len(updates) == 1
    assert updates[0].type == "finding_added"
    assert len(tracker.active_findings) == 2
    assert tracker.new_count == 1


def _task_context_patch() -> dict[str, Any]:
    return {
        "next_routing_step": "update_subagent",
        "valid_groups": [
            {"group_id": "group-1", "vulnerable_component": "fallback-package"},
            {"group_id": "group-2", "vulnerable_component": "qs"},
        ],
        "task_queue": {
            "task-4": {
                "task_id": "task-4",
                "parent_group_id": "group-1",
                "target_package_name": "express-jwt",
            },
            "task-5": {
                "task_id": "task-5",
                "parent_group_id": "group-2",
                "target_package_name": None,
            },
            "task-6": {
                "task_id": "task-6",
                "parent_group_id": "missing-group",
                "target_package_name": None,
            },
        },
        "active_target_task_ids": ["task-4", "task-5", "task-6"],
    }


def _session() -> RunSession:
    return RunSession("run-test", "ws-test", "/repo", [], ["express-jwt"])


def test_tool_span_with_inherited_node_metadata_updates_one_console_row() -> None:
    session = _session()
    session.update_task_context(_task_context_patch())
    recorder = DemoTelemetryRecorder(session)

    recorder._handle_span_start(
        "tool-span",
        "modify_and_validate_npm_dependency",
        "tool",
        {
            "package_name": "express-jwt",
            "version": "1.2.3",
            "api_key": "do-not-log-this",
            "content": "verbose source text",
        },
        {"langgraph_node": "update_subagent"},
    )

    assert len(session.state.logs) == 1
    running = session.state.logs[0]
    assert running.category == "TOOL_CALL"
    assert running.title == "Tool: modify_and_validate_npm_dependency"
    assert running.status == "running"
    assert running.task_label == "task-4 — express-jwt"
    assert "do-not-log-this" not in (running.details or "")
    assert "verbose source text" not in (running.details or "")

    recorder._handle_span_end("tool-span", "dependency updated", None)

    assert len(session.state.logs) == 1
    completed = session.state.logs[0]
    assert completed.id == running.id
    assert completed.status == "success"
    assert completed.outcome == "dependency updated"
    assert completed.duration_seconds is not None
    queued_console_events = []
    while not session.event_queue.empty():
        item = session.event_queue.get_nowait()
        if item["event"] == "console_event":
            queued_console_events.append(item["data"])
    assert len(queued_console_events) == 2
    assert {event["id"] for event in queued_console_events} == {running.id}
    assert queued_console_events[0]["status"] == "running"
    assert queued_console_events[1]["status"] == "success"


def test_tool_error_and_missing_completion_update_the_started_row() -> None:
    error_session = _session()
    error_recorder = DemoTelemetryRecorder(error_session)
    error_recorder._handle_span_start(
        "error-span",
        "read_workspace_file",
        "tool",
        {"file_path": "src/app.js"},
        {"langgraph_node": "qa_critic"},
    )
    error_recorder._handle_span_end("error-span", None, RuntimeError("api_key=secret-value"))

    assert len(error_session.state.logs) == 1
    error_event = error_session.state.logs[0]
    assert error_event.status == "error"
    assert error_event.duration_seconds is not None
    assert "secret-value" not in (error_event.outcome or "")

    incomplete_session = _session()
    incomplete_recorder = DemoTelemetryRecorder(incomplete_session)
    incomplete_recorder._handle_span_start(
        "incomplete-span",
        "search_codebase_pattern",
        "tool",
        {"search_pattern": "password"},
        {"langgraph_node": "update_subagent"},
    )
    incomplete_recorder.finish_incomplete_tool_spans()
    incomplete_recorder._handle_span_end("unstarted-span", None, None)

    assert len(incomplete_session.state.logs) == 1
    incomplete_event = incomplete_session.state.logs[0]
    assert incomplete_event.status == "incomplete"
    assert incomplete_event.duration_seconds is not None
    assert "without a completion event" in (incomplete_event.outcome or "")


def test_returned_tool_error_text_is_marked_as_error() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)
    recorder._handle_span_start(
        "returned-error-span",
        "validate_workaround",
        "tool",
        {"file_path": "routes/b2bOrder.ts"},
        {},
    )
    recorder._handle_span_end(
        "returned-error-span",
        "ERROR: [INVALID_RUNTIME_SMOKE] invalid validation input",
        None,
    )

    event = session.state.logs[0]
    assert event.status == "error"
    assert event.outcome == "ERROR: [INVALID_RUNTIME_SMOKE] invalid validation input"


def test_supervisor_and_associated_tool_events_include_task_package_labels() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)
    recorder._handle_span_start(
        "supervisor-span",
        "chain",
        "chain",
        {},
        {"langgraph_node": "supervisor"},
    )

    assert session.state.status == "running"
    assert session.state.current_node == "supervisor"
    assert session.state.logs == []

    recorder._handle_span_start(
        "inherited-child-span",
        "chain",
        "chain",
        {},
        {"langgraph_node": "supervisor"},
    )
    recorder._handle_span_end(
        "inherited-child-span",
        {"next_routing_step": "child"},
        None,
    )
    assert session.state.logs == []

    recorder._handle_span_end("supervisor-span", _task_context_patch(), None)

    route_event = session.state.logs[0]
    assert route_event.category == "KEY_RESULT"
    assert "task-4 — express-jwt" in (route_event.details or "")
    assert "task-5 — qs" in (route_event.details or "")
    assert "task-6" in (route_event.details or "")
    assert "task-6 —" not in (route_event.details or "")

    recorder._handle_span_start(
        "labeled-tool-span",
        "read_workspace_file",
        "tool",
        {"package_name": "express-jwt", "file_path": "src/index.js"},
        {"langgraph_node": "update_subagent"},
    )
    tool_event = next(event for event in session.state.logs if event.category == "TOOL_CALL")
    assert tool_event.task_label == "task-4 — express-jwt"
    recorder._handle_span_end("labeled-tool-span", "file read", None)

    assert not any(event.category == "NODE_TRANSITION" for event in session.state.logs)
    assert any(event.title == "Supervisor Route: update_subagent" for event in session.state.logs)


def test_tool_label_resolves_queued_package_without_active_targets() -> None:
    session = _session()
    task_context = _task_context_patch()
    task_context["active_target_task_ids"] = []
    session.update_task_context(task_context)
    recorder = DemoTelemetryRecorder(session)

    recorder._handle_span_start(
        "queued-package-span",
        "supervisor.fetch_registry_candidates",
        "tool",
        {"package_name": "express-jwt"},
        {"langgraph_node": "supervisor"},
    )

    assert session.state.logs[0].task_label == "task-4 — express-jwt"
    recorder._handle_span_end("queued-package-span", [], None)
