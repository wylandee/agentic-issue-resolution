"""Focused tests for deterministic report rendering and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from remediation_engine.orchestration.graph import build_orchestrator_graph
from remediation_engine.orchestration.report_node import (
    finalize_report,
    generate_report,
    run_report_node,
)


def _state() -> dict:
    """Build a small state fixture without Docker, scanners, or an LLM."""
    return {
        "run_id": "trace-report-1",
        "repo_root": "repo",
        "run_started_at": "2026-08-18T00:00:00+00:00",
        "status": "completed",
        "issues": [{"issue_type": "sca", "severity": "high"}],
        "initial_valid_groups": [
            {
                "group_id": "group-1",
                "vulnerable_component": "lodash|legacy",
                "issue_type": "sca",
                "sources": ["odc"],
                "file_path": "package.json",
                "issues": [{"severity": "high", "source": "odc"}],
                "fix_plan": {"status": "version_found"},
            }
        ],
        "valid_groups": [],
        "task_queue": {
            "task-1": {
                "task_id": "task-1",
                "parent_group_id": "group-1",
                "parent_task_id": None,
                "strategy": "VERSION_BUMP",
                "instruction": "Update package.json.\nRun tests.",
                "status": "qa_passed",
            }
        },
        "qa_evaluations": {"task-1": {"task_id": "task-1", "passed": True}},
        "triage_reconciliation": {
            "new_group_ids": ["group-new"],
            "reappeared_group_ids": [],
        },
        "diff": (
            "--- a/package.json\n"
            "+++ b/package.json\n"
            '-  "lodash": "4.17.15"\n'
            '+  "lodash": "4.17.21"\n'
        ),
        "changed_files": ["package.json"],
        "errors": [],
    }


def test_generate_report_contains_overview_metrics_findings_and_diff_evidence():
    """The canonical report exposes the run overview and deterministic evidence."""
    report = generate_report(
        _state(),
        token_summary={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        run_ended_at=datetime(2026, 8, 18, 0, 0, 2, tzinfo=UTC),
        trajectory_path="data/trajectories/trace.md",
    )

    assert "Total time taken" in report
    assert "2.00 seconds" in report
    assert "Total tokens" in report
    assert "15" in report
    assert "lodash" in report
    assert "4.17.15" in report
    assert "4.17.21" in report
    assert "QA passed" in report or "qa_passed" in report
    assert "\\|" in report
    assert "data/trajectories/trace.md" in report


def test_preliminary_report_marks_final_metrics_pending():
    """The graph node does not invent final timing or token values."""
    report = run_report_node(_state())["report_markdown"]

    assert "Pending finalization" in report
    assert "Unavailable" in report


def test_finalize_report_writes_canonical_report_atomically(tmp_path: Path, monkeypatch):
    """Finalization persists a report and includes recorder token totals."""
    monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(tmp_path))
    recorder = SimpleNamespace(
        token_data_available=True,
        total_prompt_tokens=20,
        total_completion_tokens=7,
    )

    markdown, path = finalize_report(
        _state(),
        recorder=recorder,
        trajectory_path="trajectory.md",
        trace_url="https://trace.example/run",
    )

    assert path == tmp_path / "remediation_trace-report-1.md"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == markdown
    assert "27" in markdown
    assert "https://trace.example/run" in markdown
    assert not list(tmp_path.glob("*.tmp"))


def test_report_node_converts_rendering_errors_to_new_error(monkeypatch):
    """A report failure is observable without masking the graph run."""
    monkeypatch.setattr(
        "remediation_engine.orchestration.report_node.generate_report",
        lambda state: (_ for _ in ()).throw(RuntimeError("render boom")),
    )

    result = run_report_node(_state())

    assert result["report_markdown"] == ""
    assert result["errors"] == ["report_node: failed to render report: render boom"]


def test_graph_executes_report_after_mocked_teardown(tmp_path: Path):
    """The report node is reachable as the graph terminal without live services."""
    state = {
        "repo_root": str(tmp_path),
        "valid_groups": [],
        "issues": [],
        "status": "pending",
        "errors": [],
        "changed_files": [],
        "group_strategies": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "constraints_ledger": [],
        "retry_counts": {},
        "workspace_volume": None,
    }
    with (
        patch(
            "remediation_engine.orchestration.graph.run_workspace_builder_node",
            return_value={"status": "workspace_ready"},
        ),
        patch(
            "remediation_engine.orchestration.graph.run_supervisor_node",
            return_value={"status": "supervisor_routed", "next_routing_step": "teardown"},
        ),
        patch(
            "remediation_engine.orchestration.graph.run_teardown_node",
            return_value={"status": "completed"},
        ),
    ):
        result = build_orchestrator_graph().invoke(state)

    assert "report" in build_orchestrator_graph().get_graph().nodes
    assert result["report_markdown"].startswith("# Remediation Run Report")
