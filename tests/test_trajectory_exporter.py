from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from remediation_engine.orchestration.trajectory_exporter import (
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
    monkeypatch.setattr(
        "remediation_engine.orchestration.trajectory_exporter.Client", lambda: fake_client
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.trajectory_exporter.wait_for_all_tracers", lambda: None
    )

    spans = fetch_langsmith_spans("trace-123")

    assert fake_client.trace_id == "trace-123"
    assert spans[0]["name"] == "phase5_orchestrator"


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
    assert "Update package.json to 8.5.1." in markdown
    assert '"passed": false' in markdown


def test_attempt_snapshot_summary_handles_optional_qa_sections_set_to_none():
    markdown = _render_markdown(
        trace_id="trace-optional-qa",
        repo_root="repo",
        initial_state={},
        final_state={
            "task_queue": {
                "task-1": {
                    "task_id": "task-1",
                    "task_revision": 1,
                    "status": "qa_passed",
                }
            },
            "attempt_snapshots_by_id": {
                "attempt-1": {
                    "attempt_id": "attempt-1",
                    "task_id": "task-1",
                    "task_revision": 1,
                    "attempt_number": 1,
                    "strategy_stage": "package_removal",
                    "dispatch_node": "workaround_subagent",
                    "instruction": "Keep the package installed and remove vulnerable code.",
                }
            },
            "worker_results_by_attempt": {
                "attempt-1": {"status": "success", "executed_versions": []}
            },
            "qa_results_by_attempt": {
                "attempt-1": {
                    "evaluation": {
                        "passed": True,
                        "deterministic_gates": None,
                        "semantic_security_review": None,
                    }
                }
            },
        },
        spans=[],
        source="local-fallback",
        langsmith_url=None,
        warnings=[],
        run_error=None,
        trajectory_path=Path("data/trajectories/trace-optional-qa.md"),
    )

    assert "| True | pending | optional | qa_passed |" in markdown


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
        "remediation_engine.orchestration.trajectory_exporter.fetch_langsmith_spans",
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
        "remediation_engine.orchestration.trajectory_exporter.fetch_langsmith_spans",
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
