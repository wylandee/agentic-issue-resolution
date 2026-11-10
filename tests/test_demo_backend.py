"""Focused regressions for demo state, upload, and telemetry boundaries."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from demo.backend import engine_bridge, main
from demo.backend.engine_bridge import DemoTelemetryRecorder, RunSession
from demo.backend.findings_tracker import FindingsTracker
from fastapi import HTTPException
from pydantic import ValidationError

import remediation_engine.orchestration.graph as graph_module
from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    ODCScanEvidence,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.runtime.telemetry import get_active_trajectory_recorder


class _UploadRequest:
    def __init__(self, files: list[bytes], paths: list[str]) -> None:
        self.form_data = {"files": files, "paths": paths}

    async def form(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(getlist=lambda name: self.form_data[name])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/repo/package.json",
        "../repo/package.json",
        "repo/../../outside.txt",
        r"repo\..\..\outside.txt",
    ],
)
async def test_upload_rejects_escaping_paths_before_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setattr(main, "create_workspace_from_upload", lambda _name: pytest.fail())
    with pytest.raises(HTTPException) as raised:
        await main.upload_folder(_UploadRequest([b"content"], [path]))
    assert raised.value.status_code == 400


def test_run_status_is_validated_against_the_wire_contract(tmp_path) -> None:
    session = RunSession("run-1", "workspace-1", str(tmp_path), [], [])

    session.emit_run_status("triage_completed_no_work")
    assert session.state.status == "triage_completed_no_work"

    with pytest.raises(ValidationError):
        session.emit_run_status("no_work")


@pytest.mark.asyncio
async def test_run_package_validation_returns_http_400_before_starting_a_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    issue = _issue("target-package", "app/package-lock.json")
    monkeypatch.setattr(
        main,
        "get_workspace_data",
        lambda _workspace_id: {"repo_root": tmp_path, "issues": [issue]},
    )
    monkeypatch.setattr(engine_bridge, "_RUN_SESSIONS", {})

    with pytest.raises(HTTPException) as raised:
        await main.start_run(
            main.StartRunRequest(workspace_id="workspace-1", target_packages=["missing"])
        )

    assert raised.value.status_code == 400
    assert "missing" in raised.value.detail
    assert engine_bridge._RUN_SESSIONS == {}


def _issue(package: str, path: str) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name=package,
        package_version="1.0.0",
        file_path=path,
        cve_id="CVE-2026-12345",
        severity=Severity.HIGH,
    )


def test_tracker_uses_attempt_scan_occurrence_and_reopens_final_reappearance() -> None:
    target_issue = _issue("target-package", "app/package-lock.json")
    same_cve_elsewhere = _issue("other-package", "tools/package-lock.json")
    group = VulnerabilityGroup(
        group_id="group-target",
        issue_type=IssueType.SCA,
        vulnerable_component="target-package",
        cve_ids=["CVE-2026-12345"],
        representative_issue_id=target_issue.id,
        issues=[target_issue],
    )
    tracker = FindingsTracker([target_issue])
    tracker.map_triaged_groups([group])
    task = {
        "task_id": "task-target",
        "parent_group_id": group.group_id,
        "current_attempt_id": "attempt-target",
        "task_revision": 3,
        "status": "qa_passed",
    }
    task_evidence = ODCScanEvidence(
        requested_scope="targeted",
        effective_scope="targeted",
        covered_task_ids=[task["task_id"]],
        remaining_target_identifiers=["CVE-2026-12345"],
        found_issues=[same_cve_elsewhere],
        complete=True,
    )
    state_patch = {
        "task_queue": {task["task_id"]: task},
        "qa_results_by_attempt": {
            task["current_attempt_id"]: {
                "attempt_id": task["current_attempt_id"],
                "task_id": task["task_id"],
                "task_revision": task["task_revision"],
                "evaluation": {
                    "task_id": task["task_id"],
                    "passed": True,
                    "scan_evidence": task_evidence,
                },
            }
        },
    }

    tracker.process_node_state_update("supervisor", state_patch)
    assert tracker.remediated_count == 1
    assert tracker.active_findings == []

    tracker.reconcile_final_state(
        {
            **state_patch,
            "final_full_scan_result": {
                "completed": True,
                "authoritative": True,
                "coverage_scope": "full_workspace",
                "remaining_target_identifiers": ["CVE-2026-12345"],
                "found_issues": [target_issue],
            },
        }
    )
    assert tracker.remediated_count == 0
    assert len(tracker.active_findings) == 1
    assert tracker.active_findings[0].status == "unresolved"


def test_demo_trajectory_recorders_are_isolated_between_concurrent_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    barrier = threading.Barrier(2)
    observed: dict[str, int] = {}

    def run_remediation(_request: Any) -> Any:
        recorder = get_active_trajectory_recorder()
        assert isinstance(recorder, DemoTelemetryRecorder)
        observed[recorder.session.workspace_id] = id(recorder)
        barrier.wait(timeout=5)
        logging.getLogger("remediation_engine").warning(
            "running unit tests for %s", recorder.session.workspace_id
        )
        return SimpleNamespace(
            raw_state={},
            report_path=None,
            diff="",
            errors=[],
            status="completed",
            changed_files=[],
        )

    monkeypatch.setattr(engine_bridge, "run_remediation", run_remediation)
    original_recorder = graph_module.TrajectoryRecorder
    sessions = [
        RunSession(f"run-{index}", f"workspace-{index}", str(tmp_path), [], [])
        for index in range(2)
    ]
    threads = [
        threading.Thread(target=engine_bridge._execute_run_thread, args=(session,))
        for session in sessions
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert set(observed) == {"workspace-0", "workspace-1"}
    assert len(set(observed.values())) == 2
    assert graph_module.TrajectoryRecorder is original_recorder
    for session in sessions:
        milestones = [
            event.details for event in session.state.logs if event.title == "Engine Milestone"
        ]
        assert milestones == [f"running unit tests for {session.workspace_id}"]
