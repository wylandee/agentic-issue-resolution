"""Focused tests for deterministic report rendering and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from remediation_engine.orchestration.graph import build_orchestrator_graph
from remediation_engine.orchestration.report_node import (
    _full_text,
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
            return_value={
                "status": "completed_with_errors",
                "errors": ["workspace volume cleanup failed"],
            },
        ),
    ):
        result = build_orchestrator_graph().invoke(state)

    assert "report" in build_orchestrator_graph().get_graph().nodes
    assert "workspace volume cleanup failed" in result["report_markdown"]
    assert result["report_markdown"].startswith("# Remediation Run Report")
    assert result["report_status"] == "rendered"
    assert result["report_path"] is None
    assert result["report_error"] is None


def test_report_separates_targeted_success_from_new_scan_findings_and_qa_records():
    """The overview distinguishes target QA coverage from post-scan security status."""
    state = _state()
    state["initial_valid_groups"] = [
        state["initial_valid_groups"][0],
        {
            "group_id": "group-2",
            "vulnerable_component": "got",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "medium", "source": "odc"}],
            "fix_plan": {"status": "version_found"},
        },
    ]
    state["task_queue"]["task-2"] = {
        "task_id": "task-2",
        "parent_group_id": "group-2",
        "parent_task_id": None,
        "strategy": "VERSION_BUMP",
        "instruction": "Update got.",
        "status": "qa_passed",
    }
    state["qa_evaluations"] = {"task-1": {"task_id": "task-1", "passed": True}}
    state["post_remediation_scan_issues"] = [
        {
            "cve_id": "CVE-2026-0001",
            "ghsa_id": "GHSA-AAAA-BBBB-CCCC",
            "finding_id": "ghsa-aaaa-bbbb-cccc",
            "package_name": "new-package",
            "source": "odc",
            "file_path": "package-lock.json",
            "severity": "high",
        },
        {
            "cve_id": "CVE-2026-0002",
            "package_name": "another-package",
            "source": "odc",
            "file_path": "package-lock.json",
            "severity": "low",
        },
    ]
    state["post_remediation_scan_identifiers"] = [
        "CVE-2026-0001",
        "GHSA-AAAA-BBBB-CCCC",
        "CVE-2026-0002",
    ]
    state["new_vulnerability_identifiers"] = list(state["post_remediation_scan_identifiers"])
    state["new_vulnerability_status"] = "detected"

    report = generate_report(state)

    assert "Completed with new findings (completed)" in report
    assert "Overall outcome: **Completed with new findings**" in report
    assert "2 passed / 2 targeted groups" in report
    assert "1 records retained" in report
    assert "2 findings" in report
    assert "3 unique identifiers" in report
    assert "### Newly Detected Findings" in report
    assert "CVE-2026-0001, GHSA-AAAA-BBBB-CCCC" in report
    assert "GHSA-AAAA-BBBB-CCCC, ghsa-aaaa-bbbb-cccc" not in report
    assert "Overall outcome: **Successful**" not in report


def test_report_summarizes_transitive_lockfile_changes_and_ignores_metadata():
    """Package output lists direct changes and summarizes transitive lockfile churn."""
    state = _state()
    state["diff"] = (
        "--- a/package.json\n"
        "+++ b/package.json\n"
        "@@\n"
        '  "dependencies": {\n'
        '-    "lodash": "4.17.15"\n'
        '+    "lodash": "4.17.21"\n'
        "  },\n"
        "--- a/package-lock.json\n"
        "+++ b/package-lock.json\n"
        "@@\n"
        '  "node_modules/lodash": {\n'
        '-    "version": "4.17.15",\n'
        '+    "version": "4.17.21",\n'
        '     "engines": {\n'
        '-      "node": ">=4"\n'
        '+      "node": ">=10"\n'
        "     },\n"
        '  "node_modules/transitive-package": {\n'
        '-    "version": "1.0.0",\n'
        '+    "version": "2.0.0",\n'
        '     "deprecated": "old package"\n'
    )

    report = generate_report(state)

    assert "#### Direct Manifest Changes" in report
    assert "lodash" in report
    assert "4.17.15" in report
    assert "4.17.21" in report
    assert "1 transitive lockfile package entries changed" in report
    assert "engines" not in report
    assert "deprecated" not in report
    assert "transitive-package" not in report


def test_failed_authoritative_scan_is_reported_as_unassessed_and_errors_are_grouped():
    """A scan timeout must not be rendered as a clean zero-finding result."""
    state = _state()
    state["status"] = "completed_with_errors"
    state["final_full_scan_result"] = {
        "completed": False,
        "authoritative": True,
        "status": "scan_failed",
        "triage_required": False,
        "error": "FAILURE: Dependency-Check timed out after 300s.",
    }
    state["new_vulnerability_status"] = "scan_failed"
    state["triage_reconciliation"] = {}
    state["errors"] = [
        "FAILURE: Dependency-Check timed out after 300s.",
        "supervisor: invalid planner commit rejected",
        "supervisor: invalid planner commit rejected",
    ]

    report = generate_report(state)

    assert "Unknown — authoritative scan failed" in report
    assert "Post-remediation scanner findings | 0 findings" not in report
    assert "No newly detected findings" not in report
    assert "| run, final_full_scan | ODC_TIMEOUT | 2 |" in report
    assert report.count("| supervisor | INVALID_PLANNER_COMMIT | 2 |") == 1


def test_pivot_child_group_status_uses_child_tasks_and_remaining_work_is_original_only():
    """Pivot groups must inherit their unfixable child status, not pending."""
    state = _state()
    state["initial_valid_groups"] = [
        {
            "group_id": "group-root",
            "vulnerable_component": "express-jwt",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "high", "source": "odc"}],
        }
    ]
    state["valid_groups"] = [
        state["initial_valid_groups"][0],
        {
            "group_id": "group-child",
            "vulnerable_component": "express-jwt",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "high", "source": "odc"}],
        },
    ]
    state["task_queue"] = {
        "task-root": {
            "task_id": "task-root",
            "parent_group_id": "group-root",
            "parent_task_id": None,
            "strategy": "UPDATE_VERSION",
            "status": "unfixable",
        },
        "task-child": {
            "task_id": "task-child",
            "parent_group_id": "group-child",
            "parent_task_id": "task-root",
            "strategy": "CODE_WORKAROUND",
            "status": "unfixable",
        },
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "unresolved",
        "triage_required": True,
    }
    state["new_vulnerability_status"] = "unresolved"
    state["triage_required"] = True
    state["triage_reconciliation"] = {"new_group_ids": ["group-child"]}

    report = generate_report(state)

    assert "| group-child | express-jwt |" in report
    assert "| group-child | express-jwt | odc | package.json | high | unfixable |" in report
    assert "### Remaining or Inconclusive Groups" in report
    assert "| group-root | unfixable |" in report
    assert "| group-child | express-jwt | unfixable |" in report
    assert "| group-child | pending |" not in report


def test_failed_pivot_child_overrides_historical_parent_qa_success():
    """A failed pivot descendant cannot be hidden by a passed parent task."""
    state = _state()
    state["initial_valid_groups"] = [
        {
            "group_id": "group-root",
            "vulnerable_component": "express-jwt",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "high", "source": "odc"}],
        }
    ]
    state["valid_groups"] = [state["initial_valid_groups"][0]]
    state["task_queue"] = {
        "task-root": {
            "task_id": "task-root",
            "parent_group_id": "group-root",
            "parent_task_id": None,
            "strategy": "VERSION_BUMP",
            "status": "qa_passed",
        },
        "task-child": {
            "task_id": "task-child",
            "parent_group_id": "group-child",
            "parent_task_id": "task-root",
            "strategy": "CODE_WORKAROUND",
            "status": "unfixable",
        },
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "unresolved",
        "remaining_target_identifiers": ["CVE-2020-15084"],
        "triage_required": True,
    }
    state["new_vulnerability_status"] = "unresolved"
    state["status"] = "completed"

    report = generate_report(state)

    assert "1/1 actionable groups fixed" not in report
    assert "0/1 actionable groups fixed" in report
    assert "Completed with errors (completed_with_errors)" in report
    assert "1 target identifiers remain" in report


def test_report_renders_historical_retry_activity_and_latest_action_per_task():
    """Historical diagnostics remain visible even when active plans are empty."""
    state = _state()
    state["retry_plans_by_task"] = {}
    state["retry_diagnostics_by_task"] = {
        "task-1": {
            "task_id": "task-1",
            "strategy_stage": "osv_minimum",
            "committed_attempt_id": "attempt-2",
            "target_package_name": "lodash",
            "attempted_versions": ["4.17.20", "4.17.21"],
            "executed_versions": ["4.17.21"],
            "reasoning_summary": "The first attempt was superseded by a later validated version.",
        }
    }
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "task_revision": 0,
            "status": "success",
            "summary": "Earlier attempt summary.",
        },
        {
            "task_id": "task-1",
            "attempt_id": "attempt-2",
            "task_revision": 1,
            "status": "surrender",
            "summary": "Latest attempt summary with the final worker conclusion.",
        },
    ]

    report = generate_report(state)

    assert "### Historical Retry/Pivot Activity" in report
    assert "No retry plans recorded" not in report
    assert "| task-1 | lodash | osv_minimum | attempt-2 |" in report
    assert "| task-1 | surrender | attempt-2 | 1 | 2 summaries |" in report
    assert report.count("Earlier attempt summary.") == 0


def test_report_text_preserves_full_text_without_truncation():
    """Report prose retains line breaks and the complete source text."""
    text = "A completed remediation attempt.\nThe full conclusion remains available."

    assert _full_text(text) == text


def test_report_renders_full_prose_and_failed_qa_attempt_history():
    """Full retry, QA, action, and repair prose is rendered without ellipses."""
    state = _state()
    retry_instruction = "retry instruction " + " ".join(f"detail-{index}" for index in range(120))
    retry_reason = "retry reason " + " ".join(f"reason-{index}" for index in range(120))
    qa_feedback = "QA feedback " + " ".join(f"feedback-{index}" for index in range(120))
    action_summary = "worker summary " + " ".join(f"summary-{index}" for index in range(120))
    repair_reason = "repair reason " + " ".join(f"repair-{index}" for index in range(120))
    state["retry_plans_by_task"] = {
        "task-1": {
            "action": "retry_update",
            "instructions": retry_instruction,
        }
    }
    state["retry_diagnostics_by_task"] = {
        "task-1": {
            "task_id": "task-1",
            "strategy_stage": "npm_latest",
            "reasoning_summary": retry_reason,
        }
    }
    state["qa_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "evaluation": {
                "task_id": "task-1",
                "passed": False,
                "failure_category": "security_flag",
                "retry_feedback": qa_feedback,
            },
        }
    }
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "task_revision": 1,
            "status": "surrender",
            "summary": action_summary,
        }
    ]
    state["consistency_events"] = [
        {"event_type": "repair", "reason": repair_reason},
    ]

    report = generate_report(state)

    assert retry_instruction in report
    assert retry_reason in report
    assert qa_feedback in report
    assert action_summary in report
    assert repair_reason in report
    assert "### Failed QA Gates (Attempt History)" in report
    assert "| task-1 | attempt-1 | security_flag |" in report
    assert "…" not in report


def test_worker_package_diagnostics_fall_back_to_retry_and_group_metadata():
    """Package diagnostics remain useful when an attempt snapshot omits the package."""
    state = _state()
    state["initial_valid_groups"].append(
        {
            "group_id": "group-express",
            "vulnerable_component": "express-jwt",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "high", "source": "odc"}],
        }
    )
    state["task_queue"]["task-2"] = {
        "task_id": "task-2",
        "parent_group_id": "group-express",
        "parent_task_id": None,
        "strategy": "UPDATE_VERSION",
        "status": "unfixable",
    }
    state["attempt_snapshots_by_id"] = {
        "attempt-2": {
            "attempt_id": "attempt-2",
            "task_id": "task-2",
            "target_package_name": None,
        }
    }
    state["worker_results_by_attempt"] = {
        "attempt-2": {
            "attempt_id": "attempt-2",
            "task_id": "task-2",
            "task_revision": 1,
            "execution_diagnostics": {
                "attempted_versions": ["4.0.0"],
                "executed_versions": [],
            },
        }
    }
    state["retry_diagnostics_by_task"] = {
        "task-2": {
            "task_id": "task-2",
            "attempted_versions_by_target": {"express-jwt": ["4.0.0"]},
        }
    }

    report = generate_report(state)

    assert "| task-2 | express-jwt |" in report
    assert "unknown — package missing from trace" not in report
