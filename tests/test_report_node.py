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


def test_generate_report_contains_four_sections_and_final_change_evidence():
    """The canonical report exposes only the user-facing sections and final change."""
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

    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == [
        "## 1. Summary",
        "## 2. Follow up Actions",
        "## 3. Findings Overview",
        "## 4. References",
    ]
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
    assert "Final change" in report
    assert "### Critical Errors Encountered" not in report
    assert "Targeted remediation" not in report
    assert "Post-remediation security status" not in report
    assert "Re-triage groups discovered" not in report
    assert "Targeted QA coverage" not in report
    assert "Patch present" not in report
    assert "## 2. Run Overview" not in report
    assert "## 3. Key Decisions" not in report
    assert "Validation and Remaining Issues" not in report


def test_preliminary_report_marks_final_metrics_pending():
    """The graph node does not invent final timing or token values."""
    report = run_report_node(_state())["report_markdown"]

    assert "Pending finalization" in report
    assert "Unavailable" in report
    assert "| New groups discovered | Not assessed — no authoritative scan |" in report


def test_non_authoritative_scan_keeps_new_group_metrics_unassessed():
    """Only an authoritative final scan can produce new-group counts."""
    state = _state()
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": False,
        "status": "none",
    }
    state["new_vulnerability_status"] = "none"

    report = generate_report(state)

    assert "| New groups discovered | Not assessed — no authoritative scan |" in report


def test_findings_tables_share_columns_and_successful_rows_show_only_final_change():
    """Attempt detail is reserved for follow-up rows, not successful findings."""
    state = _state()
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "status": "success",
            "summary": "Detailed successful attempt that should not be repeated.",
        }
    ]

    report = generate_report(state)
    header = (
        "| Finding | Source | Location | Package/component | Severity | Remediation | "
        "Final change | Final status | Validation |"
    )

    assert report.count(header) == 2
    assert "4.17.15 → 4.17.21" in report
    assert "Detailed successful attempt that should not be repeated." not in report


def test_follow_up_actions_include_only_outstanding_groups():
    """Successful groups stay out of follow-up actions while open groups remain."""
    state = _state()
    pending_group = {
        "group_id": "group-pending",
        "vulnerable_component": "express",
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": "package.json",
        "issues": [{"severity": "high", "source": "odc"}],
    }
    state["initial_valid_groups"].append(pending_group)
    state["valid_groups"] = list(state["initial_valid_groups"])
    state["task_queue"]["task-pending"] = {
        "task_id": "task-pending",
        "parent_group_id": "group-pending",
        "parent_task_id": None,
        "status": "pending",
        "instruction": "Update express and rerun QA.",
    }
    state["action_summaries"] = [
        {
            "task_id": "task-pending",
            "status": "surrender",
            "summary": "Complete pending attempt evidence.",
        }
    ]

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split("## 3. Findings Overview", 1)[0]

    assert "| group-pending | express | Pending |" in follow_up
    assert "Complete pending attempt evidence." in follow_up
    assert "group-1" not in follow_up


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
    assert "workspace volume cleanup failed" not in result["report_markdown"]
    assert result["report_markdown"].startswith("# Remediation Run Report")
    assert result["report_status"] == "rendered"
    assert result["report_path"] is None
    assert result["report_error"] is None


def test_report_summary_counts_new_and_reappeared_group_statuses():
    """New Summary metrics cover only new and reappeared groups."""
    state = _state()
    new_groups = [
        {
            "group_id": "group-new-unresolved",
            "vulnerable_component": "express",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "high", "source": "odc"}],
        },
        {
            "group_id": "group-new-inconclusive",
            "vulnerable_component": "lodash",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "medium", "source": "odc"}],
        },
        {
            "group_id": "group-reappeared-pending",
            "vulnerable_component": "got",
            "issue_type": "sca",
            "sources": ["odc"],
            "file_path": "package.json",
            "issues": [{"severity": "low", "source": "odc"}],
        },
    ]
    state["valid_groups"] = new_groups
    state["task_queue"].update(
        {
            "task-new-unresolved": {
                "task_id": "task-new-unresolved",
                "parent_group_id": "group-new-unresolved",
                "parent_task_id": None,
                "strategy": "VERSION_BUMP",
                "status": "unfixable",
            },
            "task-new-inconclusive": {
                "task_id": "task-new-inconclusive",
                "parent_group_id": "group-new-inconclusive",
                "parent_task_id": None,
                "strategy": "VERSION_BUMP",
                "status": "inconclusive",
            },
            "task-reappeared-pending": {
                "task_id": "task-reappeared-pending",
                "parent_group_id": "group-reappeared-pending",
                "parent_task_id": None,
                "strategy": "VERSION_BUMP",
                "status": "pending",
            },
        }
    )
    state["triage_reconciliation"] = {
        "new_group_ids": ["group-new-unresolved", "group-new-inconclusive"],
        "reappeared_group_ids": ["group-reappeared-pending"],
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "detected",
    }
    state["new_vulnerability_status"] = "detected"

    report = generate_report(state)

    assert "| New groups discovered | 3 |" in report
    assert "| New unresolved groups | 1 |" in report
    assert "| New inconclusive groups | 1 |" in report
    assert "| New pending groups | 1 |" in report
    assert "### Newly Discovered Groups" in report
    assert "group-new-unresolved" in report
    assert "group-new-inconclusive" in report
    assert "group-reappeared-pending" in report


def test_findings_show_final_package_change_without_package_detail_sections():
    """Successful findings show the final package change only."""
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

    assert "Final change" in report
    assert "lodash" in report
    assert "4.17.15" in report
    assert "4.17.21" in report
    assert "Packages Added or Changed" not in report
    assert "Worker Package Execution Evidence" not in report
    assert "transitive lockfile package entries changed" not in report
    assert "engines" not in report
    assert "deprecated" not in report
    assert "transitive-package" not in report


def test_failed_authoritative_scan_is_reported_as_unassessed_without_error_details():
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
    assert "Post-remediation scanner findings" not in report
    assert "ODC_TIMEOUT" not in report
    assert "INVALID_PLANNER_COMMIT" not in report
    assert "Critical Errors Encountered" not in report


def test_pivot_child_group_status_uses_child_tasks_in_findings_and_follow_up():
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
    assert "No validated change" in report
    assert "| group-root | express-jwt | Unresolved |" in report
    assert "| group-child | express-jwt | Unresolved |" in report
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
    assert "Completed with errors (completed_with_errors)" in report
    assert "1 target identifiers remain" not in report


def test_report_omits_retry_history_for_successful_findings():
    """Successful findings do not expose retry histories in the compact report."""
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

    assert "Historical Retry/Pivot Activity" not in report
    assert "Earlier attempt summary." not in report
    assert "Latest attempt summary with the final worker conclusion." not in report
    assert "attempt-2" not in report


def test_report_text_preserves_full_text_without_truncation():
    """Report prose retains line breaks and the complete source text."""
    text = "A completed remediation attempt.\nThe full conclusion remains available."

    assert _full_text(text) == text


def test_follow_up_actions_include_only_remediation_attempt_prose():
    """Follow-up rows exclude QA feedback and retry diagnostics from attempt text."""
    state = _state()
    state["task_queue"]["task-1"]["status"] = "needs_retry"
    retry_instruction = "retry instruction " + " ".join(f"detail-{index}" for index in range(120))
    retry_reason = "retry reason " + " ".join(f"reason-{index}" for index in range(120))
    qa_feedback = "QA feedback " + " ".join(f"feedback-{index}" for index in range(120))
    action_summary = "worker summary " + " ".join(f"summary-{index}" for index in range(120))
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
    report = generate_report(state)

    assert action_summary in report
    assert "Attempted fixes" in report
    attempted_fixes = report.split("| Remediation attempt (surrender):", 1)[1]
    assert retry_instruction not in attempted_fixes
    assert retry_reason not in report
    assert qa_feedback not in report
    assert "QA feedback:" not in report
    assert "Retry details:" not in report
    assert "Attempted versions:" not in report
    assert "Historical Retry/Pivot Activity" not in report
    assert "Failed QA Gates" not in report
    assert "attempt-1" not in report
    assert "…" not in report


def test_attempted_fixes_strip_agent_detail_sections():
    """Attempted fixes keep the action and files but omit verbose agent sections."""
    state = _state()
    state["task_queue"]["task-1"]["status"] = "unfixable"
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "task_revision": 1,
            "status": "success",
            "summary": (
                "Completed validated code workaround edits; changed files: src/guard.ts. "
                "Final note: Added a startup guard for the vulnerable dependency.\n\n"
                "What changed:\n- Added a guard around the vulnerable call.\n\n"
                "Validation:\n- Runtime smoke passed.\n\n"
                "Note:\n- This is an agent detail that should not be rendered."
            ),
        }
    ]

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split("## 3. Findings Overview", 1)[0]

    assert (
        "Remediation attempt (success): Added a startup guard for the vulnerable dependency; "
        "changed files: src/guard.ts."
    ) in follow_up
    assert "What changed:" not in follow_up
    assert "Validation:" not in follow_up
    assert "This is an agent detail" not in follow_up


def test_new_group_metrics_are_unassessed_until_post_scan_triage_runs():
    """A scan requiring triage cannot be reported as zero newly discovered groups."""
    state = _state()
    state["triage_reconciliation"] = {}
    state["triage_required"] = True
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "detected",
        "new_identifiers": ["CVE-2026-0001"],
        "triage_required": True,
    }
    state["new_vulnerability_status"] = "detected"

    report = generate_report(state)

    assert report.count("Not assessed — post-scan triage required") == 5
    assert "| Not assessed | — | — | — | — | — | No validated change | pending |" in report


def test_authoritative_remaining_finding_reopens_targeted_success():
    """A repository-wide finding overrides a narrower targeted QA success."""
    state = _state()
    state["initial_valid_groups"][0]["cve_ids"] = ["CVE-2024-0001"]
    state["initial_valid_groups"][0]["issues"][0]["cve_id"] = "CVE-2024-0001"
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "detected",
        "remaining_target_identifiers": ["CVE-2024-0001"],
    }
    state["new_vulnerability_status"] = "detected"
    state["triage_reconciliation"] = {}

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split("## 3. Findings Overview", 1)[0]

    assert "| Groups fixed | 0 |" in report
    assert "| Groups unresolved | 1 |" in report
    assert "CVE-2024-0001" in follow_up
    assert "Unresolved — retry required" in follow_up
    assert "No validated change" in report


def test_follow_up_actions_collapse_undiscovered_pivot_child_into_parent():
    """A pivot child is not a second follow-up issue without new-group evidence."""
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
            "status": "pivoted",
        },
        "task-child": {
            "task_id": "task-child",
            "parent_group_id": "group-child",
            "parent_task_id": "task-root",
            "status": "unfixable",
        },
    }
    state["triage_reconciliation"] = {}

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split("## 3. Findings Overview", 1)[0]

    assert follow_up.count("| group-root | express-jwt |") == 1
    assert "| group-child | express-jwt |" not in follow_up


def test_worker_package_diagnostics_are_excluded_without_an_attempt_summary():
    """Worker summaries populate Attempted fixes without leaking retry diagnostics."""
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
            "action_summary": {
                "status": "surrender",
                "summary": "Worker remediation attempt summary.",
            },
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

    assert "Worker remediation attempt summary." in report
    assert "No remediation attempt recorded." not in report
    assert "4.0.0" not in report
    assert "unknown — package missing from trace" not in report
