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
from remediation_engine.settings import AppSettings


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
        "## 3. Successful Remediations",
        "## 4. References",
    ]
    assert "Run duration" in report
    assert "2.00s" in report
    assert "Total findings scanned" in report
    assert "Successfully remediated" in report
    assert "Require follow-up" in report
    assert "Total tokens" in report
    assert "15 (Input: 10, Output: 5)" in report
    assert "Patch status" in report
    assert "lodash" in report
    assert "4.17.15" in report
    assert "4.17.21" in report
    assert "data/trajectories/trace.md" in report
    assert "Remediation Change" in report
    assert "### Critical Errors Encountered" not in report
    assert "Targeted remediation" not in report
    assert "Post-remediation security status" not in report
    assert "Re-triage groups discovered" not in report
    assert "Targeted QA coverage" not in report
    assert "Patch present" not in report
    assert "## 2. Run Overview" not in report
    assert "## 3. Key Decisions" not in report
    assert "Validation and Remaining Issues" not in report
    assert "qa_passed" not in report
    assert "unfixable" not in report
    assert "optimistically_fixed" not in report


def test_preliminary_report_marks_final_metrics_pending():
    """The graph node does not invent final timing or token values."""
    report = run_report_node(_state())["report_markdown"]

    assert "Pending finalization" in report
    assert "Unavailable" in report
    assert "| Patch status | Available (1 file changed) |" in report


def test_non_authoritative_scan_does_not_change_the_four_section_contract():
    """Internal post-scan state does not add diagnostic sections to the report."""
    state = _state()
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": False,
        "status": "none",
    }
    state["new_vulnerability_status"] = "none"

    report = generate_report(state)

    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## 1. Summary",
        "## 2. Follow up Actions",
        "## 3. Successful Remediations",
        "## 4. References",
    ]
    assert "New groups discovered" not in report


def test_successful_remediation_table_shows_only_final_change():
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
    header = "| Finding | Package / Target | Severity | Remediation Change | Files Changed |"

    assert report.count(header) == 1
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
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "### group-pending — express (HIGH)" in follow_up
    assert "**Status:** Pending" in follow_up
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


def test_follow_up_consolidates_new_and_reappeared_group_statuses():
    """New and reappeared groups are shown with the same follow-up format."""
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

    assert "| Successfully remediated | 1 |" in report
    assert "| Require follow-up | 3 |" in report
    assert "### Newly Discovered Groups" not in report
    assert "group-new-unresolved" in report
    assert "group-new-inconclusive" in report
    assert "group-reappeared-pending" in report
    assert "**Status:** Unresolved" in report
    assert "**Status:** Inconclusive" in report
    assert "**Status:** Pending" in report


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

    assert "Remediation Change" in report
    assert "lodash" in report
    assert "4.17.15" in report
    assert "4.17.21" in report
    assert "Packages Added or Changed" not in report
    assert "Worker Package Execution Evidence" not in report
    assert "transitive lockfile package entries changed" not in report
    assert "engines" not in report
    assert "deprecated" not in report
    assert "transitive-package" not in report


def test_failed_authoritative_scan_does_not_leak_error_details():
    """Internal scan and planner diagnostics are omitted from the user report."""
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

    assert "Dependency-Check timed out" not in report
    assert "authoritative scan failed" not in report
    assert "ODC_TIMEOUT" not in report
    assert "INVALID_PLANNER_COMMIT" not in report
    assert "Critical Errors Encountered" not in report
    assert "scan_failed" not in report


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

    assert "### group-child — express-jwt (HIGH)" in report
    assert "### group-root — express-jwt (HIGH)" in report
    assert report.count("**Status:** Unresolved") == 2
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
    assert "| Successfully remediated | 0 |" in report
    assert "| Require follow-up | 2 |" in report
    assert "completed_with_errors" not in report
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

    assert "Updated lodash 4.17.15 → 4.17.21 via dependencies." in report
    assert action_summary not in report
    assert "Attempted remediations" in report
    attempted_fixes = report.split("**Attempted remediations:**", 1)[1]
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
    """Attempted fixes expose code, file, and outcome evidence without raw agent noise."""
    state = _state()
    state["task_queue"]["task-1"]["status"] = "unfixable"
    state["task_queue"]["task-1"]["strategy"] = "CODE_WORKAROUND"
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
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "Attempted a code workaround in src/guard.ts." in follow_up
    assert "Outcome: Added a startup guard for the vulnerable dependency." in follow_up
    assert "What changed:" not in follow_up
    assert "Validation:" not in follow_up
    assert "This is an agent detail" not in follow_up


def test_final_change_describes_manifest_entry_and_natural_package_summary():
    """Final changes include the exact override edit, files, and a readable summary."""
    state = _state()
    state["initial_valid_groups"][0].update(
        {
            "vulnerable_component": "undici",
            "issues": [
                {
                    "package_name": "undici",
                    "package_version": "5.0",
                    "severity": "high",
                    "source": "odc",
                }
            ],
        }
    )
    state["task_queue"]["task-1"].update(
        {
            "target_package_name": "undici",
            "target_dependency_type": "overrides",
            "selected_version": "5.1",
        }
    )
    state["diff"] = (
        "--- a/package.json\n"
        "+++ b/package.json\n"
        "@@\n"
        '  "overrides": {\n'
        '-    "undici": "5.0"\n'
        '+    "undici": "5.1"\n'
        "  }\n"
    )

    report = generate_report(state)

    assert "| group-1 | undici | HIGH | 5.0 → 5.1 via overrides | package.json |" in report


def test_final_change_pairs_lockfile_version_with_package_from_resolved_url():
    """A replacement block must not attribute the new version to a removed nested package."""
    state = _state()
    state["initial_valid_groups"][0].update(
        {
            "vulnerable_component": "got",
            "issues": [{"package_name": "got", "package_version": "8.3.2"}],
        }
    )
    state["task_queue"]["task-1"].update(
        {
            "target_package_name": "got",
            "target_dependency_type": "overrides",
            "selected_version": "11.8.5",
        }
    )
    state["diff"] = (
        "--- a/package-lock.json\n"
        "+++ b/package-lock.json\n"
        "@@\n"
        '     "node_modules/got": {\n'
        '-      "version": "8.3.2",\n'
        '-      "resolved": "https://registry.npmjs.org/got/-/got-8.3.2.tgz",\n'
        '-    "node_modules/got/node_modules/pify": {\n'
        '-      "version": "3.0.0",\n'
        '-      "resolved": "https://registry.npmjs.org/pify/-/pify-3.0.0.tgz",\n'
        '-      "license": "MIT",\n'
        '-        "node": ">=4"\n'
        '+      "version": "11.8.5",\n'
        '+      "resolved": "https://registry.npmjs.org/got/-/got-11.8.5.tgz",\n'
        "       }\n"
        "     },\n"
    )

    report = generate_report(state)

    assert "| group-1 | got | UNKNOWN | 8.3.2 → 11.8.5 via lockfile | package-lock.json |" in report
    assert "got: 8.3.2 → removed via lockfile" not in report


def test_finalize_report_is_deterministic_when_report_llm_is_enabled(tmp_path: Path):
    """Report settings cannot cause model calls or model-written report prose."""
    state = _state()
    settings = AppSettings(
        remediation_report_dir=tmp_path,
        report_llm_enabled=True,
        report_llm_model="test-report-model",
    )
    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        report, _ = finalize_report(
            state,
            recorder=None,
            trajectory_path="trajectory.md",
            trace_url="https://trace.example/run",
            settings=settings,
        )

    chat_openai.assert_not_called()
    assert "4.17.15 → 4.17.21 via dependencies" in report
    assert "executive narrative" not in report.casefold()


def test_finalize_report_uses_deterministic_fallback_when_tokens_are_unavailable(tmp_path: Path):
    """Finalization still produces the canonical report without recorder usage."""
    settings = AppSettings(remediation_report_dir=tmp_path)
    report, path = finalize_report(
        _state(),
        recorder=None,
        trajectory_path="trajectory.md",
        trace_url=None,
        settings=settings,
    )

    assert path == tmp_path / "remediation_trace-report-1.md"
    assert "| Total tokens | Unavailable |" in report
    assert "| Patch status | Available (1 file changed) |" in report


def test_workaround_final_change_includes_replacement_evidence():
    """Successful source workarounds expose replay replacements, files, and the final note."""
    state = _state()
    state["initial_valid_groups"][0]["vulnerable_component"] = "express-jwt"
    state["task_queue"]["task-1"]["strategy"] = "CODE_WORKAROUND"
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "status": "success",
            "summary": (
                "Completed validated code workaround edits; changed files: src/guard.ts. "
                "Final note: Guarded the vulnerable call before rendering."
            ),
        }
    ]
    state["worker_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "status": "success",
            "changed_files": ["src/guard.ts"],
            "execution_diagnostics": {"validation_passed": True},
            "replay_plan": {
                "successful_edit_sets": [
                    {
                        "affected_files": ["src/guard.ts"],
                        "replacements": [
                            {
                                "file_path": "src/guard.ts",
                                "old_text": "render(input)",
                                "new_text": "render(escape(input))",
                            }
                        ],
                    }
                ]
            },
        }
    }

    report = generate_report(state)

    assert "Code workaround: Guarded the vulnerable call before rendering." in report
    assert "Files Changed |" in report
    assert "src/guard.ts" in report
    assert "```diff\n--- a/src/guard.ts" in report
    assert "-render(input)" in report
    assert "+render(escape(input))" in report


def test_workaround_replay_projection_supplies_attempt_diff_when_worker_result_is_sparse():
    """Task-keyed replay evidence remains visible when the attempt envelope is sparse."""
    state = _state()
    state["initial_valid_groups"][0]["vulnerable_component"] = "express-jwt"
    state["task_queue"]["task-1"]["strategy"] = "CODE_WORKAROUND"
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "status": "success",
            "summary": "Applied a guarded source change.",
        }
    ]
    state["workaround_replay_plans_by_task"] = {
        "task-1": {
            "successful_edit_sets": [
                {
                    "affected_files": ["src/guard.ts"],
                    "replacements": [
                        {
                            "file_path": "src/guard.ts",
                            "old_text": "render(input)",
                            "new_text": "render(escape(input))",
                        }
                    ],
                }
            ]
        }
    }

    report = generate_report(state)

    assert "```diff\n--- a/src/guard.ts" in report
    assert "-render(input)" in report
    assert "+render(escape(input))" in report


def test_attempt_without_files_does_not_claim_metadata_version_change():
    """A surrendered attempt with no edits must not report its planned version as applied."""
    state = _state()
    state["initial_valid_groups"][0].update(
        {
            "vulnerable_component": "undici",
            "issues": [{"package_name": "undici", "package_version": "5.0"}],
        }
    )
    state["task_queue"]["task-1"].update(
        {
            "status": "unfixable",
            "target_package_name": "undici",
            "target_dependency_type": "overrides",
            "selected_version": "5.1",
            "parent_package_version": "5.0",
        }
    )
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "status": "surrender",
            "summary": "Stopped without a validated code workaround; no files changed.",
        }
    ]
    state["attempt_snapshots_by_id"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "target_package_name": "undici",
            "selected_version": "5.1",
            "instruction": "Update package.json using overrides.",
        }
    }

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "No validated package change was applied." in follow_up
    assert "undici: 5.0 → 5.1" not in follow_up


def test_untriaged_authoritative_findings_require_follow_up():
    """A newly detected scanner identifier is presented as an open finding."""
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

    assert "| Successfully remediated | 1 |" in report
    assert "| Require follow-up | 1 |" in report
    assert "### CVE-2026-0001 — Untriaged finding (UNKNOWN)" in report
    assert "### Newly Discovered Groups" not in report


def test_multiple_untriaged_authoritative_findings_are_consolidated_as_follow_up():
    """Raw final-scan identifiers remain visible without a discovery subsection."""
    state = _state()
    state["triage_required"] = False
    state["triage_reconciliation"] = {}
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "detected",
        "found_identifiers": ["CVE-2026-0001", "GHSA-ABCD-EFGH-IJKL"],
        "new_identifiers": ["CVE-2026-0001", "GHSA-ABCD-EFGH-IJKL"],
        "remaining_target_identifiers": [],
        "triage_required": True,
        "found_issues": [
            {
                "cve_id": "CVE-2026-0001",
                "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
                "package_name": "new-package",
                "file_path": "package-lock.json",
                "severity": "HIGH",
                "source": "odc",
            }
        ],
    }

    report = generate_report(state)

    assert "| Successfully remediated | 1 |" in report
    assert "| Require follow-up | 2 |" in report
    assert "### CVE-2026-0001 — new-package (HIGH)" in report
    assert "### GHSA-ABCD-EFGH-IJKL — new-package (HIGH)" in report
    assert "New findings detected" not in report


def test_final_scan_reopened_groups_are_follow_up_actions():
    """A reopened group is rendered with the same follow-up block as any open group."""
    state = _state()
    reopened = {
        "group_id": "group-reopened",
        "vulnerable_component": "extract-zip",
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": "package-lock.json",
        "issues": [{"cve_id": "CVE-2026-0002", "severity": "high", "source": "odc"}],
    }
    state["valid_groups"] = [reopened]
    state["task_queue"]["task-reopened"] = {
        "task_id": "task-reopened",
        "parent_group_id": "group-reopened",
        "parent_task_id": None,
        "status": "unfixable",
    }
    state["triage_reconciliation"] = {
        "final_scan_reopened_group_ids": ["group-reopened"],
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "detected",
        "new_identifiers": ["CVE-2026-0002"],
        "found_identifiers": ["CVE-2026-0002"],
        "triage_required": True,
        "found_issues": [
            {
                "cve_id": "CVE-2026-0002",
                "file_path": "scan/package-lock.json?/extract-zip:2.0.1",
                "package_name": "extract-zip",
                "severity": "HIGH",
                "source": "odc",
            }
        ],
    }

    report = generate_report(state)

    assert "| Require follow-up | 1 |" in report
    assert "### CVE-2026-0002 — extract-zip (HIGH)" in report
    assert "**Status:** Unresolved" in report
    assert "CVE-2026-0002" in report


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
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "| Successfully remediated | 0 |" in report
    assert "| Require follow-up | 1 |" in report
    assert "CVE-2024-0001" in follow_up
    assert "**Status:** Retry needed" in follow_up
    assert "No successful remediations were produced during this run." in report


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
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert follow_up.count("### group-root — express-jwt (HIGH)") == 1
    assert "### group-child — express-jwt" not in follow_up


def test_finding_label_uses_vulnerable_component_not_transitive_edit_target():
    """A transitive finding stays labelled by its vulnerable package."""
    state = _state()
    group = state["initial_valid_groups"][0]
    group.update(
        {
            "vulnerable_component": "lodash",
            "issues": [
                {
                    "package_name": "lodash",
                    "package_version": "2.4.2",
                    "severity": "high",
                    "source": "odc",
                }
            ],
        }
    )
    state["task_queue"]["task-1"].update(
        {
            "target_package_name": "sanitize-html",
            "parent_package_name": "sanitize-html",
            "parent_package_version": "1.4.2",
            "status": "unfixable",
        }
    )

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "### group-1 — lodash (HIGH)" in follow_up
    assert "### group-1 — sanitize-html (HIGH)" not in follow_up
    assert "alternative remediation for sanitize-html" in follow_up


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

    assert "Remediation attempt summary." in report
    assert "Worker remediation attempt summary." not in report
    assert "No remediation attempt recorded." not in report
    assert "4.0.0" not in report
    assert "unknown — package missing from trace" not in report
