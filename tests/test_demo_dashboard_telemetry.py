from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from demo.backend import workspace_manager
from demo.backend.engine_bridge import DemoTelemetryRecorder, RunSession
from demo.backend.findings_tracker import FindingsTracker

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityIssue,
)

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


def test_failed_workaround_restores_prematurely_remediated_finding_as_unresolved() -> None:
    from remediation_engine.contracts.schemas import VulnerabilityGroup

    issue = _issue(cve_id=_CVE, package_name="express-jwt")
    tracker = FindingsTracker([issue])

    group = VulnerabilityGroup(
        group_id="group-1",
        issue_type=IssueType.SCA,
        representative_issue_id=issue.id,
        vulnerable_component="express-jwt",
        issues=[issue],
        cve_ids=[_CVE],
    )
    tracker.map_triaged_groups([group])

    # Simulate step 1: version bump passes QA temporarily
    task_passed = {
        "task_id": "task-1",
        "parent_group_id": "group-1",
        "target_package_name": "express-jwt",
        "current_attempt_id": "attempt-1",
        "task_revision": 0,
        "status": "qa_passed",
    }
    qa_result = {
        "attempt_id": "attempt-1",
        "task_id": "task-1",
        "task_revision": 0,
        "evaluation": {
            "passed": True,
            "scan_evidence": {
                "complete": True,
                "remaining_target_identifiers": [],
                "found_issues": [],
            },
        },
    }
    tracker.process_node_state_update(
        "supervisor",
        {
            "task_queue": {"task-1": task_passed},
            "qa_results_by_attempt": {"attempt-1": qa_result},
        },
    )
    assert tracker.remediated_count == 1
    assert len(tracker.active_findings) == 0

    # Simulate final state: task pivoted to workaround which failed/surrendered
    task_pivoted = {**task_passed, "status": "pivoted"}
    task_workaround_failed = {
        "task_id": "task-2",
        "parent_group_id": "group-1",
        "target_package_name": "express-jwt",
        "status": "unfixable",
    }
    tracker.reconcile_final_state(
        {
            "task_queue": {"task-1": task_pivoted, "task-2": task_workaround_failed},
            "qa_results_by_attempt": {"attempt-1": qa_result},
            "remaining_target_identifiers": [_CVE],
        }
    )

    assert tracker.remediated_count == 0
    assert len(tracker.active_findings) == 1
    assert tracker.active_findings[0].status == "unresolved"


def test_nested_and_sandbox_tool_spans_do_not_duplicate_console_rows() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)

    recorder._start(
        run_id="top-tool",
        parent_run_id="workaround-agent-chain",
        name="read_workspace_file",
        run_type="tool",
        inputs={"file_path": "package.json"},
        metadata={"langgraph_node": "workaround_subagent"},
    )
    assert len(session.state.logs) == 1
    assert session.state.logs[0].title == "Tool: read_workspace_file"
    assert session.state.logs[0].status == "running"

    recorder._start(
        run_id="sandbox-child",
        parent_run_id="top-tool",
        name="docker_sandbox.read_file",
        run_type="tool",
        inputs={"relative_path": "package.json"},
        metadata={},
    )
    assert len(session.state.logs) == 1

    recorder._end("sandbox-child", outputs="file contents", error=None)
    assert len(session.state.logs) == 1

    recorder._end("top-tool", outputs="file contents", error=None)
    assert len(session.state.logs) == 1
    assert session.state.logs[0].status == "success"
    assert session.state.logs[0].outcome == "file contents"


def test_composite_tool_with_multiple_nested_actions_shows_single_tool_row() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)

    recorder._start(
        run_id="composite-tool",
        parent_run_id="subagent-node",
        name="modify_and_validate_npm_dependency",
        run_type="tool",
        inputs={"package_name": "express-jwt", "version": "6.0.0"},
        metadata={"langgraph_node": "update_subagent"},
    )
    assert len(session.state.logs) == 1
    assert session.state.logs[0].title == "Tool: modify_and_validate_npm_dependency"

    recorder._start(
        run_id="child-1",
        parent_run_id="composite-tool",
        name="docker_sandbox.read_file",
        run_type="tool",
        inputs={"file_path": "package.json"},
    )
    recorder._end("child-1", outputs="package content")

    recorder._start(
        run_id="child-2",
        parent_run_id="composite-tool",
        name="validate_code_syntax",
        run_type="tool",
        inputs={"file_path": "package.json"},
    )
    recorder._end("child-2", outputs={"valid": True})

    recorder._start(
        run_id="child-3",
        parent_run_id="composite-tool",
        name="docker_sandbox.run",
        run_type="tool",
        inputs={"command": "npm install"},
    )
    recorder._end("child-3", outputs={"exit_code": 0})

    assert len(session.state.logs) == 1

    recorder._end("composite-tool", outputs="Dependency express-jwt updated successfully.")
    assert len(session.state.logs) == 1
    assert session.state.logs[0].status == "success"


def test_standalone_docker_sandbox_spans_are_not_emitted_as_tool_calls() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)

    recorder._start(
        run_id="sandbox-setup-1",
        parent_run_id="workspace_builder_node",
        name="docker_sandbox.start",
        run_type="tool",
        inputs={},
    )
    recorder._end("sandbox-setup-1", outputs=None)

    recorder._start(
        run_id="sandbox-setup-2",
        parent_run_id="workspace_builder_node",
        name="docker_sandbox.run",
        run_type="tool",
        inputs={"command": "git checkout ..."},
    )
    recorder._end("sandbox-setup-2", outputs={"exit_code": 0})

    assert len(session.state.logs) == 0


def test_sequential_independent_tool_calls_each_show_once() -> None:
    session = _session()
    recorder = DemoTelemetryRecorder(session)

    recorder._start(
        run_id="tool-round-1",
        parent_run_id="agent-node",
        name="read_workspace_file",
        run_type="tool",
        inputs={"file_path": "server.js"},
    )
    recorder._end("tool-round-1", outputs="server.js text")
    assert len(session.state.logs) == 1

    recorder._start(
        run_id="tool-round-2",
        parent_run_id="agent-node",
        name="search_codebase_pattern",
        run_type="tool",
        inputs={"search_pattern": "jwt"},
    )
    recorder._end("tool-round-2", outputs="matches found")
    assert len(session.state.logs) == 2
    assert [log.title for log in session.state.logs] == [
        "Tool: read_workspace_file",
        "Tool: search_codebase_pattern",
    ]


def test_concurrent_demo_runs_keep_telemetry_bound_to_each_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from demo.backend import engine_bridge

    from remediation_engine.api import RemediationResult
    from remediation_engine.runtime.telemetry import get_active_trajectory_recorder

    fixture_names = ("fixture-alpha", "fixture-beta")
    sessions = []
    for fixture_name in fixture_names:
        repo_root = tmp_path / fixture_name
        repo_root.mkdir()
        sessions.append(
            RunSession(
                run_id=f"run-{fixture_name}",
                workspace_id=f"ws-{fixture_name}",
                repo_root=str(repo_root),
                issues=[],
                target_packages=[fixture_name],
            )
        )

    overlap = Barrier(len(sessions))

    def fake_run_remediation(request):
        recorder = get_active_trajectory_recorder()
        assert isinstance(recorder, DemoTelemetryRecorder)
        fixture_name = request.repo_root.name
        overlap.wait(timeout=5)
        span_id = f"tool-{fixture_name}"
        recorder._start(
            run_id=span_id,
            parent_run_id="agent-node",
            name=span_id,
            run_type="tool",
            inputs={"fixture": fixture_name},
        )
        recorder._end(span_id, outputs=f"finished {fixture_name}")
        logging.getLogger("remediation_engine").info(
            "running unit tests for %s",
            fixture_name,
        )
        return RemediationResult(status="completed")

    monkeypatch.setattr(engine_bridge, "run_remediation", fake_run_remediation)
    monkeypatch.setattr(logging.getLogger("remediation_engine"), "level", logging.INFO)

    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        list(executor.map(engine_bridge._execute_run_thread, sessions))

    for session, fixture_name in zip(sessions, fixture_names, strict=True):
        tool_events = [event for event in session.state.logs if event.category == "TOOL_CALL"]
        assert len(tool_events) == 1
        assert tool_events[0].title == f"Tool: tool-{fixture_name}"

        milestone_details = [
            event.details for event in session.state.logs if event.category == "DETERMINISTIC_FN"
        ]
        assert any(
            f"running unit tests for {fixture_name}" in (detail or "")
            for detail in milestone_details
        )
        other_fixture = next(name for name in fixture_names if name != fixture_name)
        assert not any(
            f"running unit tests for {other_fixture}" in (detail or "")
            for detail in milestone_details
        )
