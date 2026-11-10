"""Focused tests for reporting."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import remediation_engine.agents.report as report_node
import remediation_engine.agents.report as report_summary_module
from remediation_engine.agents.report import (
    ExecutiveSummaryDraft,
    ExecutiveSummaryEvidence,
    SpecialRemediation,
    SummaryCount,
    _package_attempt_text,
    fallback_executive_summary,
    finalize_report,
    generate_executive_summary,
    generate_report,
    run_report_node,
)
from remediation_engine.agents.report_evidence import PackageChange, _package_changes
from remediation_engine.contracts.schemas import (
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    TaskStatus,
)
from remediation_engine.orchestration.graph import build_orchestrator_graph
from remediation_engine.orchestration.state import reconcile_phase5_state_before_teardown
from remediation_engine.runtime.telemetry import write_report_atomic
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


def _executive_summary_evidence(state: dict):
    """Build the final-state summary evidence used by report projections."""
    return report_node._build_executive_summary_evidence(report_node._build_context(state))


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
        "## 1. Executive Summary",
        "## 2. Follow up Actions",
        "## 3. Successful Remediations",
        "## 4. References",
    ]
    assert "Run duration" in report
    assert "2.00s" in report
    assert "Total findings (CVEs and GHSAs)" in report
    assert "Successfully remediated vulnerability groups" in report
    assert "Vulnerability groups requiring follow-up" in report
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


def test_package_diffs_preserve_same_package_across_manifests_and_localize_groups():
    diff = (
        "--- a/apps/one/package.json\n"
        "+++ b/apps/one/package.json\n"
        "@@ -1 +1 @@\n"
        '-  "shared-package": "1.0.0"\n'
        '+  "shared-package": "1.1.0"\n'
        "--- a/tools/two/package.json\n"
        "+++ b/tools/two/package.json\n"
        "@@ -1 +1 @@\n"
        '-  "shared-package": "2.0.0"\n'
        '+  "shared-package": "2.1.0"\n'
    )

    changes = _package_changes(diff)
    first_group = {
        "vulnerable_component": "shared-package",
        "file_path": "apps/one/package-lock.json",
    }
    second_group = {
        "vulnerable_component": "shared-package",
        "file_path": "tools/two/package-lock.json",
    }

    assert [(change.file, change.old, change.new) for change in changes] == [
        ("apps/one/package.json", "1.0.0", "1.1.0"),
        ("tools/two/package.json", "2.0.0", "2.1.0"),
    ]
    assert report_node._package_change_matches_group(changes[0], first_group)
    assert not report_node._package_change_matches_group(changes[0], second_group)
    assert not report_node._package_change_matches_group(changes[1], first_group)
    assert report_node._package_change_matches_group(changes[1], second_group)


def test_sections_two_through_four_use_the_existing_section_renderers():
    """The Executive Summary change leaves the detailed report sections intact."""
    state = _state()
    context = report_node._build_context(state)
    report = generate_report(state)
    section_start = report.index("## 2. Follow up Actions")
    expected_sections = (
        "\n\n".join(
            (
                report_node._render_follow_up_actions(context),
                report_node._render_successful_remediations(context),
                report_node._render_references(context),
            )
        )
        + "\n"
    )

    assert report[section_start:] == expected_sections
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


def test_report_total_uses_canonical_occurrence_rows():
    state = _state()
    ghsa = "GHSA-22JQ-VG5J-6VGG"
    state["issues"] = [
        {
            "id": "issue-cve",
            "cve_id": "CVE-2026-54272",
            "ghsa_id": ghsa,
            "package_name": "ip-address",
            "package_version": "10.2.0",
            "file_path": "package-lock.json",
        },
        {
            "id": "issue-ghsa",
            "ghsa_id": ghsa,
            "package_name": "ip-address",
            "package_version": "10.2.0",
            "file_path": "package-lock.json",
        },
        {
            "id": "distinct-occurrence",
            "cve_id": "CVE-2026-54272",
            "package_name": "another-package",
            "package_version": "10.2.0",
            "file_path": "package-lock.json",
        },
    ]

    report = generate_report(state)

    assert "| Total findings (CVEs and GHSAs) | 2 |" in report


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
        "## 1. Executive Summary",
        "## 2. Follow up Actions",
        "## 3. Successful Remediations",
        "## 4. References",
    ]
    assert "New groups discovered" not in report


def test_explicit_empty_baseline_stays_empty_when_final_groups_exist():
    state = _state()
    state["initial_valid_groups"] = []
    state["issues"] = []
    state["valid_groups"] = [
        {
            "group_id": "group-new",
            "issues": [{"cve_id": "CVE-NEW", "package_name": "new-package"}],
        }
    ]

    context = report_node._build_context(state)
    evidence = report_node._build_executive_summary_evidence(context)

    assert context.initial_valid_groups == []
    assert evidence.original_groups.value == 0
    assert evidence.original_findings.value == 0


def test_completed_no_work_keeps_known_zero_original_groups_with_issues():
    state = _state()
    state.update(
        {
            "initial_valid_groups": [],
            "issues": [
                {
                    "cve_id": "CVE-INITIAL",
                    "package_name": "untriaged-package",
                    "package_version": "1.0",
                    "file_path": "package-lock.json",
                }
            ],
            "valid_groups": [],
            "initial_triage_status": "completed_no_work",
        }
    )

    evidence = _executive_summary_evidence(state)

    assert evidence.original_findings.value == 1
    assert evidence.original_groups.value == 0
    assert evidence.original_groups.complete


def test_historical_new_group_survives_later_reappeared_triage_projection():
    state = _state()
    new_issue = {
        "cve_id": "CVE-NEW-HISTORICAL",
        "package_name": "new-package",
        "package_version": "1.0",
        "file_path": "package-lock.json",
    }
    state["valid_groups"] = [{"group_id": "group-new", "issues": [new_issue]}]
    state["task_queue"]["task-new"] = {
        "task_id": "task-new",
        "parent_group_id": "group-new",
        "parent_task_id": None,
        "status": "qa_passed",
    }
    state["triage_reconciliation"] = {
        "new_group_ids": [],
        "reappeared_group_ids": ["group-new"],
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "completed",
        "coverage_scope": "full_workspace",
        "coverage_complete": True,
        "found_issues": [new_issue],
        "new_identifiers": [],
    }

    evidence = _executive_summary_evidence(state)

    assert evidence.new_groups.value == 1
    assert evidence.new_findings.value == 1


def test_synthetic_coordination_root_is_not_counted_as_a_new_group():
    state = _state()
    state.update(
        {
            "initial_valid_groups": [],
            "valid_groups": [],
            "issues": [],
            "task_queue": {
                "task-coordination": {
                    "task_id": "task-coordination",
                    "parent_group_id": "synthetic-coordination-root",
                    "parent_task_id": None,
                    "is_synthetic": True,
                    "status": "qa_passed",
                }
            },
            "initial_triage_status": "completed_no_work",
            "triage_required": False,
            "triage_reconciliation": {},
            "final_full_scan_result": {
                "completed": True,
                "authoritative": True,
                "status": "completed",
                "coverage_scope": "full_workspace",
                "coverage_complete": True,
                "found_issues": [],
                "new_identifiers": [],
            },
        }
    )

    evidence = _executive_summary_evidence(state)

    assert evidence.new_groups.value == 0
    assert evidence.new_groups.complete
    assert evidence.new_findings.value == 0
    assert evidence.new_findings.complete


def test_partial_new_identifier_coverage_returns_partial_finding_count():
    state = _state()
    first = {
        "cve_id": "CVE-NEW-1",
        "package_name": "new-package",
        "package_version": "1.0",
        "file_path": "package-lock.json",
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "completed",
        "coverage_scope": "full_workspace",
        "coverage_complete": True,
        "found_issues": [first],
        "new_identifiers": ["CVE-NEW-1", "CVE-NEW-2"],
    }

    evidence = _executive_summary_evidence(state)

    assert evidence.new_findings.value == 1
    assert not evidence.new_findings.complete


def test_new_fixed_group_without_finding_identity_is_not_reported_as_zero_findings():
    state = _state()
    state["task_queue"]["task-new"] = {
        "task_id": "task-new",
        "parent_group_id": "group-new",
        "parent_task_id": None,
        "status": "qa_passed",
    }
    state["triage_reconciliation"] = {"new_group_ids": []}
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "completed",
        "coverage_scope": "full_workspace",
        "coverage_complete": True,
        "found_issues": [],
        "new_identifiers": [],
    }

    evidence = _executive_summary_evidence(state)

    assert evidence.new_groups.value == 1
    assert evidence.new_findings.value is None
    assert not evidence.new_findings.complete


def test_synthetic_follow_up_group_without_issue_identity_is_unknown():
    state = _state()
    state["task_queue"]["task-1"]["status"] = "qa_passed"
    state["task_queue"]["task-open"] = {
        "task_id": "task-open",
        "parent_group_id": "group-open",
        "parent_task_id": None,
        "status": "needs_retry",
    }

    evidence = _executive_summary_evidence(state)

    assert evidence.follow_up_groups.value == 1
    assert evidence.follow_up_findings.value is None
    assert not evidence.follow_up_findings.complete


def test_qa_pass_without_retained_special_patch_is_not_described_as_qa_failure():
    state = _state()
    state["task_queue"]["task-1"].update(
        {
            "strategy": "CODE_WORKAROUND",
            "strategy_stage": "code_workaround",
            "status": "qa_passed",
            "current_attempt_id": "attempt-1",
        }
    )
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "status": "success",
            "summary": "Completed validated code workaround edits; changed files: src/guard.ts.",
        }
    ]
    state["worker_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "status": "success",
            "changed_files": ["src/guard.ts"],
        }
    }
    state["qa_results_by_attempt"] = {"attempt-1": {"evaluation": {"passed": True}}}
    state["diff"] = ""

    evidence = _executive_summary_evidence(state)
    special = evidence.special_remediations[0]

    assert special.outcome == (
        "QA passed for this attempt; a retained current remediation is not established"
    )
    assert "not accepted by final QA" not in special.outcome


def test_terminal_success_uses_matching_latest_attempt_after_current_id_is_cleared():
    """A terminal QA-passed task can retain its exact immutable attempt lineage."""
    state = _state()
    state["task_queue"]["task-1"].update(
        {
            "strategy": "CODE_WORKAROUND",
            "strategy_stage": "migration_code_workaround",
            "status": "qa_passed",
            "task_revision": 2,
            "current_attempt_id": None,
        }
    )
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "status": "success",
            "summary": "Completed validated API migration; changed files: src/guard.ts.",
        }
    ]
    state["attempt_snapshots_by_id"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "task_revision": 1,
            "state_revision": 5,
            "attempt_number": 1,
            "dispatch_node": "workaround_subagent",
            "strategy_stage": "migration_code_workaround",
            "instruction": "Perform the committed API migration in src/guard.ts.",
        }
    }
    state["worker_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "task_revision": 1,
            "status": "success",
            "changed_files": ["src/guard.ts"],
        }
    }
    state["qa_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "task_revision": 1,
            "evaluation": {"passed": True},
        }
    }
    state["diff"] = (
        "--- a/src/guard.ts\n+++ b/src/guard.ts\n"
        "-const verify = insecure(input)\n+const verify = secure(input)\n"
    )

    evidence = _executive_summary_evidence(state)

    assert evidence.special_remediations[0].outcome == "successfully accepted by final QA"


def test_terminal_success_does_not_accept_an_older_attempt_if_a_newer_commit_exists():
    """A prior passing attempt is stale when the task has a newer commit."""
    state = _state()
    state["task_queue"]["task-1"].update(
        {
            "strategy": "CODE_WORKAROUND",
            "strategy_stage": "migration_code_workaround",
            "status": "qa_passed",
            "task_revision": 3,
            "current_attempt_id": None,
        }
    )
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": "attempt-old",
            "status": "success",
            "summary": "Completed validated API migration; changed files: src/guard.ts.",
        }
    ]
    state["attempt_snapshots_by_id"] = {
        "attempt-old": {
            "attempt_id": "attempt-old",
            "task_id": "task-1",
            "task_revision": 1,
            "state_revision": 5,
            "attempt_number": 1,
            "dispatch_node": "workaround_subagent",
            "strategy_stage": "migration_code_workaround",
        },
        "attempt-new": {
            "attempt_id": "attempt-new",
            "task_id": "task-1",
            "task_revision": 2,
            "state_revision": 8,
            "attempt_number": 2,
            "dispatch_node": "workaround_subagent",
            "strategy_stage": "migration_code_workaround",
        },
    }
    state["worker_results_by_attempt"] = {
        "attempt-old": {
            "attempt_id": "attempt-old",
            "task_id": "task-1",
            "task_revision": 1,
            "status": "success",
            "changed_files": ["src/guard.ts"],
        }
    }
    state["qa_results_by_attempt"] = {
        "attempt-old": {
            "attempt_id": "attempt-old",
            "task_id": "task-1",
            "task_revision": 1,
            "evaluation": {"passed": True},
        }
    }
    state["diff"] = (
        "--- a/src/guard.ts\n+++ b/src/guard.ts\n"
        "-const verify = insecure(input)\n+const verify = secure(input)\n"
    )

    evidence = _executive_summary_evidence(state)

    assert evidence.special_remediations[0].outcome == (
        "QA passed for this attempt; a retained current remediation is not established"
    )


def test_historical_attempt_without_qa_envelope_does_not_inherit_latest_task_verdict():
    state = _state()
    context = report_node._build_context(state)

    assert (
        report_node._attempt_qa_verdict(
            context,
            {"task_id": "task-1", "attempt_id": "attempt-old"},
            None,
        )
        is None
    )
    assert report_node._attempt_qa_verdict(context, {"task_id": "task-1"}, None) is True


def test_incremental_incomplete_report_keeps_known_new_identifiers_visible():
    state = _state()
    identifier = "CVE-2026-0042"
    state.update(
        {
            "triage_required": True,
            "new_vulnerability_identifiers": [identifier],
            "new_vulnerability_status": "detected",
            "post_remediation_scan_issues": [
                {
                    "cve_id": identifier,
                    "package_name": "new-package",
                    "file_path": "package-lock.json",
                    "severity": "HIGH",
                    "source": "odc",
                }
            ],
            "final_full_scan_result": {
                "completed": False,
                "authoritative": False,
                "coverage_scope": "incremental_dependency_inputs",
                "coverage_complete": False,
                "status": "detected",
                "found_identifiers": [identifier],
                "new_identifiers": [identifier],
                "triage_required": True,
            },
        }
    )

    report = generate_report(state)

    assert "| Dependency scan coverage | Incremental dependency inputs (incomplete) |" in report
    assert (
        f"| Known new vulnerability identifiers | {identifier} "
        "(observed; terminal scan evidence is incomplete) |"
    ) in report
    assert identifier in report
    assert "Incremental scan coverage is incomplete." in report
    assert "| Successfully remediated vulnerability groups | 1 |" in report


def test_unavailable_scan_does_not_render_no_identifiers_as_none():
    state = _state()
    state.pop("final_full_scan_result", None)
    state["new_vulnerability_identifiers"] = []
    state["new_vulnerability_status"] = "not_scanned"

    report = generate_report(state)

    assert (
        "| Known new vulnerability identifiers | Unknown (terminal scan evidence is unavailable) |"
    ) in report


def test_complete_scans_render_scoped_no_new_identifier_result():
    state = _state()
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "status": "completed",
        "coverage_scope": "full_workspace",
        "coverage_complete": True,
        "new_identifiers": [],
        "found_issues": [],
    }

    report = generate_report(state)

    assert (
        "| Known new vulnerability identifiers | None detected in the full-workspace scan |"
    ) in report


def test_incremental_complete_report_names_its_limited_scope():
    state = _state()
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": False,
        "coverage_scope": "incremental_dependency_inputs",
        "coverage_complete": True,
        "status": "none",
        "triage_required": False,
    }
    state["new_vulnerability_status"] = "none"

    report = generate_report(state)

    assert "| Dependency scan coverage | Incremental dependency inputs (complete) |" in report
    assert "Incremental scan coverage is incomplete." not in report


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
    summary_metadata = {}

    markdown, path = finalize_report(
        _state(),
        recorder=recorder,
        trajectory_path="trajectory.md",
        trace_url="https://trace.example/run",
        settings=AppSettings(openai_api_key="", remediation_report_dir=tmp_path),
        summary_metadata=summary_metadata,
    )

    assert path == tmp_path / "remediation_trace-report-1.md"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == markdown
    assert "27" in markdown
    assert "https://trace.example/run" in markdown
    assert not list(tmp_path.glob("*.tmp"))
    assert summary_metadata["status"] == "fallback_no_credentials"
    assert len(summary_metadata["source_fingerprint"]) == 64


def test_report_node_converts_rendering_errors_to_new_error(monkeypatch):
    """A report failure is observable without masking the graph run."""
    monkeypatch.setattr(
        "remediation_engine.agents.report._render_report_context",
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
    assert "| Run outcome | Completed with errors |" in result["report_markdown"]
    assert "Review the run errors before relying on this result." in result["report_markdown"]
    assert result["report_status"] == "rendered"
    assert result["report_path"] is None
    assert result["report_error"] is None


def test_follow_up_consolidates_new_and_reappeared_status_entries():
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

    assert "| Successfully remediated vulnerability groups | 1 |" in report
    assert "| Vulnerability groups requiring follow-up | 3 |" in report
    assert "| Total vulnerability groups | 4 |" in report
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
    state["triage_reconciliation"] = {}
    state["status"] = "completed"

    report = generate_report(state)

    assert "1/1 actionable groups fixed" not in report
    assert "| Successfully remediated vulnerability groups | 0 |" in report
    assert "| Vulnerability groups requiring follow-up | 2 |" in report
    assert "group-child" in report
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


def test_finalize_report_is_deterministic(tmp_path: Path):
    """Report generation is deterministic and never invokes ChatOpenAI."""
    state = _state()
    settings = AppSettings(
        remediation_report_dir=tmp_path,
    )
    with (
        patch(
            "remediation_engine.agents.report._build_context",
            wraps=report_node._build_context,
        ) as build_context,
        patch("langchain_openai.ChatOpenAI") as chat_openai,
    ):
        report, _ = finalize_report(
            state,
            recorder=None,
            trajectory_path="trajectory.md",
            trace_url="https://trace.example/run",
            settings=settings,
        )

    chat_openai.assert_not_called()
    build_context.assert_called_once()
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
    state["qa_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "task_revision": 1,
            "evaluation": {"passed": True},
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
    state["qa_results_by_attempt"] = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "task_id": "task-1",
            "task_revision": 1,
            "evaluation": {"passed": True},
        }
    }

    report = generate_report(state)

    assert "```diff\n--- a/src/guard.ts" in report
    assert "-render(input)" in report
    assert "+render(escape(input))" in report


def test_workaround_diffs_are_isolated_when_attempts_share_a_source_file():
    """Each workaround report block contains only its attempt's replacements."""
    state = _state()
    express_group = {
        "group_id": "group-express-jwt",
        "vulnerable_component": "express-jwt",
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": "package.json",
        "issues": [{"cve_id": "CVE-EXPRESS", "severity": "high", "source": "odc"}],
    }
    sanitize_group = {
        "group_id": "group-sanitize-html",
        "vulnerable_component": "sanitize-html",
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": "package.json",
        "issues": [{"cve_id": "CVE-SANITIZE", "severity": "high", "source": "odc"}],
    }
    state["initial_valid_groups"] = [express_group, sanitize_group]
    state["valid_groups"] = [express_group, sanitize_group]
    state["task_queue"] = {
        "task-express": {
            "task_id": "task-express",
            "parent_group_id": express_group["group_id"],
            "parent_task_id": None,
            "strategy": "CODE_WORKAROUND",
            "status": "qa_passed",
        },
        "task-sanitize": {
            "task_id": "task-sanitize",
            "parent_group_id": sanitize_group["group_id"],
            "parent_task_id": None,
            "strategy": "CODE_WORKAROUND",
            "status": "qa_passed",
        },
    }
    state["action_summaries"] = [
        {
            "task_id": "task-express",
            "attempt_id": "attempt-express",
            "status": "success",
            "summary": "Updated express-jwt middleware.",
            "changed_files": ["lib/insecurity.ts"],
        },
        {
            "task_id": "task-sanitize",
            "attempt_id": "attempt-sanitize",
            "status": "success",
            "summary": "Updated sanitizeSecure.",
            "changed_files": ["lib/insecurity.ts"],
        },
    ]
    state["worker_results_by_attempt"] = {
        "attempt-express": {
            "attempt_id": "attempt-express",
            "task_id": "task-express",
            "status": "success",
            "changed_files": ["lib/insecurity.ts"],
            "execution_diagnostics": {"validation_passed": True},
            "replay_plan": {
                "successful_edit_sets": [
                    {
                        "affected_files": ["lib/insecurity.ts"],
                        "replacements": [
                            {
                                "file_path": "lib/insecurity.ts",
                                "old_text": "old express middleware",
                                "new_text": "new express middleware",
                            }
                        ],
                    }
                ]
            },
        },
        "attempt-sanitize": {
            "attempt_id": "attempt-sanitize",
            "task_id": "task-sanitize",
            "status": "success",
            "changed_files": ["lib/insecurity.ts"],
            "execution_diagnostics": {"validation_passed": True},
            "replay_plan": {
                "successful_edit_sets": [
                    {
                        "affected_files": ["lib/insecurity.ts"],
                        "replacements": [
                            {
                                "file_path": "lib/insecurity.ts",
                                "old_text": "old sanitize implementation",
                                "new_text": "new sanitize implementation",
                            }
                        ],
                    }
                ]
            },
        },
    }
    state["diff"] = (
        "--- a/lib/insecurity.ts\n"
        "+++ b/lib/insecurity.ts\n"
        "@@\n"
        "-old express middleware\n"
        "+new express middleware\n"
        "@@\n"
        "-old sanitize implementation\n"
        "+new sanitize implementation\n"
    )
    state["changed_files"] = ["lib/insecurity.ts"]

    report = generate_report(state)
    express_section = report.split("#### CVE-EXPRESS — express-jwt", 1)[1].split(
        "#### CVE-SANITIZE — sanitize-html", 1
    )[0]
    sanitize_section = report.split("#### CVE-SANITIZE — sanitize-html", 1)[1]

    assert "-old express middleware" in express_section
    assert "+new express middleware" in express_section
    assert "old sanitize implementation" not in express_section
    assert "new sanitize implementation" not in express_section
    assert "-old sanitize implementation" in sanitize_section
    assert "+new sanitize implementation" in sanitize_section
    assert "old express middleware" not in sanitize_section
    assert "new express middleware" not in sanitize_section


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

    assert "| Successfully remediated vulnerability groups | 1 |" in report
    assert "| Vulnerability groups requiring follow-up | 1 |" in report
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

    assert "| Successfully remediated vulnerability groups | 1 |" in report
    assert "| Vulnerability groups requiring follow-up | 2 |" in report
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

    assert "| Vulnerability groups requiring follow-up | 1 |" in report
    assert "### CVE-2026-0002 — extract-zip (HIGH)" in report
    assert "**Status:** Unresolved" in report
    assert "CVE-2026-0002" in report


def test_authoritative_remaining_finding_reopens_targeted_success():
    """The teardown state barrier reopens a QA success by matching occurrence evidence."""
    state = _state()
    state["initial_valid_groups"][0]["cve_ids"] = ["CVE-2024-0001"]
    state["initial_valid_groups"][0]["issues"][0].update(
        {
            "cve_id": "CVE-2024-0001",
            "package_name": "lodash|legacy",
            "package_version": "4.17.15",
            "file_path": "package-lock.json",
        }
    )
    state["task_queue"]["task-1"] = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy=RoutingStrategy.VERSION_BUMP,
        status=TaskStatus.QA_PASSED,
    )
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": True,
        "coverage_scope": "full_workspace",
        "status": "detected",
        "remaining_target_identifiers": ["CVE-2024-0001"],
        "found_issues": [
            {
                "cve_id": "CVE-2024-0001",
                "package_name": "lodash|legacy",
                "package_version": "4.17.15",
                "file_path": "package-lock.json",
            }
        ],
    }
    state["new_vulnerability_status"] = "detected"
    state["triage_reconciliation"] = {}
    state.update(reconcile_phase5_state_before_teardown(state))
    state["status"] = "completed_with_errors"

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "| Successfully remediated vulnerability groups | 0 |" in report
    assert "| Vulnerability groups requiring follow-up | 1 |" in report
    assert "CVE-2024-0001" in follow_up
    assert "**Status:** Retry needed" in follow_up
    assert "No successful remediations were produced during this run." in report


def test_complete_incremental_scan_reopens_the_matching_qa_success():
    """A complete targeted scan reopens its matching occurrence after QA passed."""
    state = _state()
    state["task_queue"]["task-1"] = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy=RoutingStrategy.VERSION_BUMP,
        status=TaskStatus.QA_PASSED,
    )
    state["triage_reconciliation"] = {}
    state["initial_valid_groups"][0]["cve_ids"] = ["CVE-2024-0001"]
    state["initial_valid_groups"][0]["ghsa_ids"] = ["GHSA-ABCD-EFGH-IJKL"]
    state["initial_valid_groups"][0]["issues"][0].update(
        {
            "cve_id": "CVE-2024-0001",
            "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
            "package_name": "lodash|legacy",
            "package_version": "4.17.15",
            "file_path": "scan/package-lock.json?/lodash|legacy:4.17.15",
        }
    )
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": False,
        "coverage_scope": "incremental_dependency_inputs",
        "coverage_complete": True,
        "status": "detected",
        "remaining_target_identifiers": ["CVE-2024-0001", "GHSA-ABCD-EFGH-IJKL"],
        "found_issues": [
            {
                "cve_id": "CVE-2024-0001",
                "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
                "package_name": "lodash|legacy",
                "package_version": "4.17.15",
                "file_path": "scan/.odc-targeted/000/package-lock.json?/lodash|legacy:4.17.15",
            }
        ],
    }
    state.update(reconcile_phase5_state_before_teardown(state))
    state["status"] = "completed_with_errors"
    state["errors"] = ["teardown_node: incremental scan still contains CVE-2024-0001"]

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert state["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY
    assert "| Successfully remediated vulnerability groups | 0 |" in report
    assert "| Vulnerability groups requiring follow-up | 1 |" in report
    assert "CVE-2024-0001" in follow_up
    assert "No follow-up actions are required." not in follow_up
    assert "Review the run errors" in report


def test_incremental_scan_reopens_only_the_matching_same_cve_occurrence():
    state = _state()
    first_group = state["initial_valid_groups"][0]
    first_group.update(
        {
            "vulnerable_component": "shared-package",
            "cve_ids": ["CVE-2024-0001"],
            "issues": [
                {
                    "cve_id": "CVE-2024-0001",
                    "issue_type": "sca",
                    "package_name": "shared-package",
                    "package_version": "1.0.0",
                    "file_path": "scan/app/package-lock.json?/shared-package:1.0.0",
                    "source": "odc",
                }
            ],
        }
    )
    second_group = {
        **first_group,
        "group_id": "group-2",
        "file_path": "tools/package-lock.json",
        "file_paths": ["tools/package-lock.json"],
        "issues": [
            {
                "cve_id": "CVE-2024-0001",
                "issue_type": "sca",
                "package_name": "shared-package",
                "package_version": "1.0.0",
                "file_path": "scan/tools/package-lock.json?/shared-package:1.0.0",
                "source": "odc",
            }
        ],
    }
    state["initial_valid_groups"].append(second_group)
    state["task_queue"] = {
        "task-1": RemediationTask(
            task_id="task-1",
            parent_group_id="group-1",
            qa_policy=QAPolicy.VERSION_BUMP,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.QA_PASSED,
        ),
        "task-2": RemediationTask(
            task_id="task-2",
            parent_group_id="group-2",
            qa_policy=QAPolicy.VERSION_BUMP,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.QA_PASSED,
        ),
    }
    state["final_full_scan_result"] = {
        "completed": True,
        "authoritative": False,
        "coverage_scope": "incremental_dependency_inputs",
        "coverage_complete": True,
        "status": "detected",
        "remaining_target_identifiers": ["CVE-2024-0001"],
        "found_issues": [
            {
                "cve_id": "CVE-2024-0001",
                "issue_type": "sca",
                "package_name": "shared-package",
                "package_version": "1.0.0",
                "file_path": "scan/.odc-targeted/000/package-lock.json?/shared-package:1.0.0",
                "source": "odc",
            }
        ],
    }
    state["latest_incremental_scan_by_input"] = {
        "tools/package-lock.json": {
            "source_input_path": "tools/package-lock.json",
            "complete": True,
            "found_issues": state["final_full_scan_result"]["found_issues"],
        }
    }

    barrier = reconcile_phase5_state_before_teardown(state)

    assert barrier["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
    assert barrier["task_queue"]["task-2"].status == TaskStatus.NEEDS_RETRY


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


def test_follow_up_actions_render_each_attempt_with_the_attempted_version():
    """Every recorded version attempt is shown with its selected package version."""
    state = _state()
    state["initial_valid_groups"][0].update(
        {
            "vulnerable_component": "express-jwt",
            "issues": [
                {
                    "package_name": "express-jwt",
                    "package_version": "5.0.0",
                    "severity": "high",
                    "source": "odc",
                }
            ],
        }
    )
    state["task_queue"]["task-1"].update({"status": "unfixable", "strategy": "VERSION_BUMP"})
    state["action_summaries"] = [
        {
            "task_id": "task-1",
            "attempt_id": f"attempt-{number}",
            "status": "surrender",
            "summary": "Attempted a package version update.",
        }
        for number in range(1, 5)
    ]
    state["attempt_snapshots_by_id"] = {
        f"attempt-{number}": {
            "attempt_id": f"attempt-{number}",
            "task_id": "task-1",
            "selected_version": version,
            "instruction": f'Update "express-jwt" in package.json to version "{version}".',
            "strategy": "CODE_WORKAROUND" if number == 4 else "VERSION_BUMP",
        }
        for number, version in enumerate(("6.0.0", "6.1.2", "8.5.1", "8.6.0"), start=1)
    }
    state["action_summaries"][-1]["summary"] = "Attempted a code workaround."
    state["diff"] = "--- a/src/guard.ts\n+++ b/src/guard.ts\n@@\n-unsafe(input)\n+safe(input)\n"

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    for number in range(1, 5):
        assert f"Attempt {number}" in follow_up
    assert "Updated express-jwt 5.0.0 → 6.0.0 via dependencies." in follow_up
    assert "Updated express-jwt 5.0.0 → 6.1.2 via dependencies." in follow_up
    assert "Updated express-jwt 5.0.0 → 8.5.1 via dependencies." in follow_up
    assert "```diff\n--- a/src/guard.ts" in follow_up
    assert "\n   ```diff" not in follow_up


def test_successful_pivot_lineage_is_rendered_once():
    """A successful pivot child does not duplicate its original finding row."""
    state = _state()
    state["valid_groups"] = [
        state["initial_valid_groups"][0],
        {
            **state["initial_valid_groups"][0],
            "group_id": "group-child",
        },
    ]
    state["task_queue"]["task-child"] = {
        "task_id": "task-child",
        "parent_group_id": "group-child",
        "parent_task_id": "task-1",
        "strategy": "CODE_WORKAROUND",
        "status": "qa_passed",
    }

    report = generate_report(state)
    successful = report.split("## 3. Successful Remediations", 1)[1].split("## 4. References", 1)[0]

    assert "| Total vulnerability groups | 1 |" in report
    assert "| Successfully remediated vulnerability groups | 1 |" in report
    assert successful.count("| group-1 |") == 1
    assert "group-child" not in successful


def test_successful_task_only_group_uses_attempt_metadata_for_exact_change():
    """Successful groups absent from the final group projection remain visible."""
    state = _state()
    state["task_queue"]["task-ip"] = {
        "task_id": "task-ip",
        "parent_group_id": "group-ip-address",
        "parent_task_id": None,
        "strategy": "VERSION_BUMP",
        "strategy_stage": "package_override",
        "status": "qa_passed",
        "target_package_name": "ip-address",
        "target_dependency_type": "overrides",
        "parent_package_version": "10.2.0",
    }
    state["action_summaries"] = [
        {
            "task_id": "task-ip",
            "attempt_id": "attempt-ip",
            "status": "success",
            "summary": (
                "Completed validated manifest updates for ip-address in package.json; "
                "changed files: package.json."
            ),
        }
    ]
    state["worker_results_by_attempt"] = {
        "attempt-ip": {
            "attempt_id": "attempt-ip",
            "task_id": "task-ip",
            "status": "success",
            "changed_files": [],
            "execution_diagnostics": {"validation_passed": True},
        }
    }
    state["attempt_snapshots_by_id"] = {
        "attempt-ip": {
            "attempt_id": "attempt-ip",
            "task_id": "task-ip",
            "target_package_name": "ip-address",
            "target_dependency_type": "overrides",
            "selected_version": "10.3.1",
            "instruction": ('Add or update "overrides": {"ip-address": "10.3.1"} in package.json.'),
            "strategy": "VERSION_BUMP",
        }
    }
    state["diff"] = ""

    report = generate_report(state)
    successful = report.split("## 3. Successful Remediations", 1)[1].split("## 4. References", 1)[0]

    assert (
        "| group-ip-address | ip-address | UNKNOWN | 10.2.0 → 10.3.1 via overrides | package.json |"
        in successful
    )
    assert "| Successfully remediated vulnerability groups | 2 |" in report


def test_workerless_remediation_action_is_reported_without_claimed_files():
    state = _state()
    state["diff"] = ""
    state["changed_files"] = ["package-lock.json"]
    state["task_queue"]["task-1"]["remediation_action"] = (
        "Removed due to another package upgrade; target absent from every covered package-lock input."
    )

    report = generate_report(state)
    successful = report.split("## 3. Successful Remediations", 1)[1].split("## 4. References", 1)[0]

    assert (
        "| Removed due to another package upgrade; target absent from every covered "
        "package-lock input. | Not recorded |"
    ) in successful
    assert "| package-lock.json |" not in successful
    assert "Attempt 1" not in successful


def test_package_removal_attempt_includes_manifest_and_source_changes():
    """Package-removal attempts identify both manifest and source changes."""
    state = _state()
    notevil_group = {
        "group_id": "group-notevil",
        "vulnerable_component": "notevil",
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": "package.json",
        "issues": [{"cve_id": "CVE-NOTEVIL", "severity": "medium", "source": "odc"}],
    }
    state["initial_valid_groups"] = [notevil_group]
    state["valid_groups"] = [notevil_group]
    state["task_queue"] = {
        "task-notevil": {
            "task_id": "task-notevil",
            "parent_group_id": "group-notevil",
            "parent_task_id": None,
            "strategy": "CODE_WORKAROUND",
            "strategy_stage": "code_workaround",
            "no_fix_stage": "package_removal",
            "qa_policy": "no_fix_package_removal",
            "status": "unfixable",
        }
    }
    state["qa_evaluations"] = {}
    state["action_summaries"] = [
        {
            "task_id": "task-notevil",
            "attempt_id": "attempt-notevil",
            "status": "success",
            "summary": (
                "Completed validated code workaround edits; changed files: "
                "package-lock.json, package.json, routes/b2bOrder.ts. "
                "Final note: Removed notevil from package.json and synchronized package-lock.json."
            ),
        }
    ]
    state["worker_results_by_attempt"] = {
        "attempt-notevil": {
            "attempt_id": "attempt-notevil",
            "task_id": "task-notevil",
            "status": "success",
            "changed_files": ["package-lock.json", "package.json", "routes/b2bOrder.ts"],
            "execution_diagnostics": {"validation_passed": True},
            "replay_plan": {
                "pre_attempt_snapshots": {
                    "package.json": (
                        '{\n  "dependencies": {\n    "notevil": "^1.3.3",\n'
                        '    "express": "^4.0.0"\n  }\n}\n'
                    ),
                    "package-lock.json": (
                        '{\n  "packages": {\n    "": {\n      "dependencies": '
                        '{\n        "notevil": "^1.3.3"\n      }\n    },\n'
                        '    "node_modules/notevil": {\n      "version": "1.3.3"\n'
                        "    }\n  }\n}\n"
                    ),
                },
                "successful_edit_sets": [
                    {
                        "affected_files": ["routes/b2bOrder.ts"],
                        "replacements": [
                            {
                                "file_path": "routes/b2bOrder.ts",
                                "old_text": "unsafe()",
                                "new_text": "safe()",
                            }
                        ],
                    }
                ],
            },
        }
    }
    state["diff"] = ""

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "Removed notevil from package.json and synchronized package-lock.json." in follow_up
    assert "Attempted a code workaround in routes/b2bOrder.ts." in follow_up
    assert "Attempted a code workaround in package-lock.json, package.json" not in follow_up
    assert "```diff\n--- a/routes/b2bOrder.ts" in follow_up
    assert "--- a/package.json" in follow_up
    assert "--- a/package-lock.json" in follow_up
    assert '-    "notevil": "^1.3.3",' in follow_up


def _maven_report_state(
    *,
    no_fix: bool = False,
    diff: str = "",
    include_pom_baseline: bool = False,
) -> dict:
    """Build a committed Maven attempt fixture without external services."""
    coordinate = "org.example:widget"
    pom_path = "modules/widget/pom.xml"
    issue = {
        "package_name": coordinate,
        "package_version": "1.0.0",
        "severity": "high",
        "source": "odc",
        "ecosystem": "maven",
        "purl": "pkg:maven/org.example/widget@1.0.0",
    }
    group = {
        "group_id": "group-maven",
        "vulnerable_component": coordinate,
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": None,
        "file_paths": [],
        "issues": [issue],
        "localized_issues": [
            {
                "issue": issue,
                "package_manager": "maven",
                "manifest_file": None,
                "declaration_type": "dependencyManagement",
                "version_property_name": "widget.version",
                "version_property_file": None,
            }
        ],
    }
    task = {
        "task_id": "task-maven",
        "parent_group_id": "group-maven",
        "parent_task_id": None,
        "strategy": "CODE_WORKAROUND" if no_fix else "VERSION_BUMP",
        "strategy_stage": "code_workaround" if no_fix else "maven_latest",
        "status": "unfixable",
        "target_package_name": coordinate,
        "target_dependency_type": "dependencies" if no_fix else "dependencyManagement",
        "no_fix_stage": "package_removal" if no_fix else None,
        "qa_policy": "no_fix_package_removal" if no_fix else None,
        "selected_version": None if no_fix else "1.2.0",
    }
    attempt = {
        "attempt_id": "attempt-maven",
        "task_id": "task-maven",
        "target_package_name": coordinate,
        "target_dependency_type": task["target_dependency_type"],
        "selected_version": task["selected_version"],
        "strategy": task["strategy"],
        "no_fix_stage": task["no_fix_stage"],
        "target_manifest_paths": [pom_path],
        "instruction": (
            f"Remove exact Maven coordinate {coordinate} from the authorized direct dependency;"
            if no_fix
            else f"Update {coordinate} through dependencyManagement."
        ),
    }
    summary = {
        "attempt_id": "attempt-maven",
        "task_id": "task-maven",
        "status": "surrender",
        "summary": "Attempted a package removal."
        if no_fix
        else "Attempted a package version update.",
        "changed_files": [pom_path],
    }
    worker_result = {
        "attempt_id": "attempt-maven",
        "task_id": "task-maven",
        "status": "surrender",
        "changed_files": [pom_path],
    }
    if include_pom_baseline:
        worker_result["replay_plan"] = {
            "pre_attempt_snapshots": {
                pom_path: (
                    "<project><dependencies><dependency>"
                    "<groupId>org.example</groupId><artifactId>widget</artifactId>"
                    "<version>1.0.0</version></dependency></dependencies></project>"
                )
            }
        }

    state = _state()
    state.update(
        {
            "initial_valid_groups": [group],
            "valid_groups": [group],
            "task_queue": {"task-maven": task},
            "action_summaries": [summary],
            "attempt_snapshots_by_id": {"attempt-maven": attempt},
            "worker_results_by_attempt": {"attempt-maven": worker_result},
            "diff": diff,
            "changed_files": [pom_path] if diff else [],
        }
    )
    return state


def test_maven_attempt_text_includes_every_changed_pom_path():
    change = PackageChange(
        name="org.example:widget",
        old="1.0.0",
        new="1.2.0",
        file="pom.xml; modules/widget/pom.xml",
        scope="direct",
        section="dependencyManagement",
    )

    assert _package_attempt_text(change, "dependencyManagement") == (
        "Updated org.example:widget 1.0.0 → 1.2.0 via dependencyManagement "
        "in pom.xml, modules/widget/pom.xml."
    )


def test_maven_attempt_reports_committed_gav_target_and_pom_path():
    """Maven attempt evidence uses the exact GAV, dependency target, and POM."""
    pom_diff = (
        "--- a/modules/widget/pom.xml\n"
        "+++ b/modules/widget/pom.xml\n"
        "@@ -1,3 +1,3 @@\n"
        " <project><dependencies>\n"
        "-<version>1.0.0</version>\n"
        "+<version>1.2.0</version>\n"
        " </dependencies></project>\n"
    )
    report = generate_report(_maven_report_state(diff=pom_diff))
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert (
        "Updated org.example:widget 1.0.0 → 1.2.0 via dependencyManagement "
        "in modules/widget/pom.xml."
    ) in follow_up
    assert "modules/widget/pom.xml" in follow_up
    assert "widget.version" not in follow_up
    assert (
        _package_changes(
            "--- a/pom.xml\n"
            "+++ b/pom.xml\n"
            "@@ -1,3 +1,3 @@\n"
            " <description><![CDATA[\n"
            '-  "widget": "1.0.0"\n'
            '+  "widget": "1.2.0"\n'
            " ]]></description>\n"
        )
        == []
    )


def test_maven_package_removal_uses_real_pom_diff_when_available():
    """A committed Maven package removal renders its actual POM unified diff."""
    pom_diff = (
        "--- a/modules/widget/pom.xml\n"
        "+++ b/modules/widget/pom.xml\n"
        "@@ -1,7 +1,2 @@\n"
        " <project><dependencies>\n"
        "-  <dependency>\n"
        "-    <groupId>org.example</groupId>\n"
        "-    <artifactId>widget</artifactId>\n"
        "-    <version>1.0.0</version>\n"
        "-  </dependency>\n"
        " </dependencies></project>\n"
    )
    state = _maven_report_state(no_fix=True, diff=pom_diff)
    state["task_queue"]["task-maven"]["target_dependency_type"] = "dependencyManagement"
    state["attempt_snapshots_by_id"]["attempt-maven"]["target_dependency_type"] = (
        "dependencyManagement"
    )
    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "Removed org.example:widget from the dependency manifest." in follow_up
    assert "```diff\n--- a/modules/widget/pom.xml" in follow_up
    assert "-    <artifactId>widget</artifactId>" in follow_up
    assert "@@ package removal:" not in follow_up


def test_maven_package_removal_without_real_pom_diff_uses_generic_description():
    """POM XML is not synthesized from a snapshot when no real diff is present."""
    report = generate_report(_maven_report_state(no_fix=True, include_pom_baseline=True))
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "Removed org.example:widget from the dependency manifest." in follow_up
    assert "--- a/modules/widget/pom.xml" not in follow_up
    assert "@@ package removal:" not in follow_up


def test_maven_package_removal_requires_committed_stage():
    """Maven prose cannot authorize removal outside the committed removal stage."""
    state = _maven_report_state(no_fix=True)
    state["task_queue"]["task-maven"]["no_fix_stage"] = None
    state["attempt_snapshots_by_id"]["attempt-maven"]["no_fix_stage"] = None

    report = generate_report(state)
    follow_up = report.split("## 2. Follow up Actions", 1)[1].split(
        "## 3. Successful Remediations", 1
    )[0]

    assert "Removed org.example:widget from the dependency manifest." not in follow_up
    assert "--- a/modules/widget/pom.xml" not in follow_up


def test_successful_maven_update_uses_committed_metadata_with_pom_diff():
    """Maven POM changes retain exact GAV reporting despite XML parser exclusion."""
    pom_diff = (
        "--- a/modules/widget/pom.xml\n"
        "+++ b/modules/widget/pom.xml\n"
        "@@ -1,4 +1,4 @@\n"
        "<project><dependencies>\n"
        "-<version>1.0.0</version>\n"
        "+<version>1.2.0</version>\n"
        "</dependencies></project>\n"
    )
    state = _maven_report_state(diff=pom_diff)
    state["task_queue"]["task-maven"]["status"] = "qa_passed"
    state["action_summaries"][0]["status"] = "success"
    state["worker_results_by_attempt"]["attempt-maven"]["status"] = "success"

    report = generate_report(state)
    successful = report.split("## 3. Successful Remediations", 1)[1].split("## 4. References", 1)[0]

    assert "1.0.0 → 1.2.0 via dependencyManagement" in successful
    assert "modules/widget/pom.xml" in successful


_FACTS = (
    "original_counts",
    "special_remediations",
    "remediated_counts",
    "follow_up_counts",
    "new_counts",
    "evidence_limits",
)


def _draft_response(*, duplicate: str | None = None) -> dict[str, Any]:
    """Return natural sentence lead-ins with each typed fact reference."""
    lead_ins = {
        "original_counts": "The run began with",
        "special_remediations": "",
        "remediated_counts": "Final QA accepted remediation for",
        "follow_up_counts": "That leaves",
        "new_counts": "We could not confirm",
        "evidence_limits": "The report also notes",
    }
    return {
        "clauses": [
            {
                "fact": duplicate if fact == "new_counts" and duplicate else fact,
                "lead_in": lead_ins[fact],
            }
            for fact in _FACTS
        ]
    }


def _evidence() -> ExecutiveSummaryEvidence:
    """Return fixture evidence with mixed outcomes and incomplete scan scope."""
    return ExecutiveSummaryEvidence(
        original_findings=SummaryCount(5, source_refs=("issues",)),
        original_groups=SummaryCount(4, source_refs=("initial_valid_groups",)),
        remediated_findings=SummaryCount(0, source_refs=("qa_results_by_attempt",)),
        remediated_groups=SummaryCount(0, source_refs=("task_queue",)),
        follow_up_findings=SummaryCount(5, source_refs=("initial_valid_groups.issues",)),
        follow_up_groups=SummaryCount(3, source_refs=("task_queue",)),
        new_findings=SummaryCount(None, complete=False, scope="incremental dependency inputs"),
        new_groups=SummaryCount(None, complete=False, scope="incremental dependency inputs"),
        special_remediations=(
            SpecialRemediation(
                kind="code workaround",
                package="express-jwt",
                attempt_id="attempt-1",
                outcome="attempted but failed QA",
                details="Source edit was not accepted.",
                source_refs=("group:g1", "attempt:attempt-1"),
            ),
        ),
        scan_scope="Incremental dependency inputs (incomplete)",
        scan_complete=False,
        evidence_limits=("post-scan grouping is incomplete",),
    )


class _StructuredModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.schema: type[Any] | None = None
        self.messages: Any = None
        self.calls = 0

    def with_structured_output(self, schema: type[Any]) -> _StructuredModel:
        self.schema = schema
        return self

    def invoke(self, messages: Any) -> Any:
        self.messages = messages
        self.calls += 1
        if isinstance(self.response, list):
            return self.response[min(self.calls - 1, len(self.response) - 1)]
        return self.response


def test_summary_uses_configured_model_and_resolves_factual_slots() -> None:
    """Model-authored framing receives renderer-owned factual values."""
    model = _StructuredModel(_draft_response())
    settings = AppSettings(openai_api_key="test-key", remedy_llm_model="test-model")

    result = generate_executive_summary(
        _evidence(),
        settings=settings,
        model_factory=lambda actual: model if actual is settings else None,
    )

    assert model.schema is ExecutiveSummaryDraft
    assert result.status == "generated"
    assert result.used_llm
    assert result.model == "test-model"
    assert (
        "The run began with **5 original findings** across **4 vulnerability groups**."
        in result.text
    )
    assert "The work included Code workaround for express-jwt" in result.text
    assert (
        "Final QA accepted remediation for **0 findings** across **0 vulnerability groups**."
        in result.text
    )
    assert "at least" not in result.text
    assert "unknown number" not in result.text
    assert "We could not confirm the count of newly detected findings" in result.text
    assert "attempted but failed QA" in result.text
    assert "Evidence limits:" not in result.text
    assert "**5 original findings**" in result.text
    assert "terminal scan coverage is incomplete for incremental dependency inputs" in result.text
    assert "findings outside that scope remain unknown" in result.text
    assert result.text != fallback_executive_summary(_evidence()).text


def test_summary_builds_luna_client_with_bounded_reasoning_budget(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    model = _StructuredModel(_draft_response())

    def create_model(model_name: str, **kwargs: Any) -> _StructuredModel:
        captured.update(model_name=model_name, **kwargs)
        return model

    monkeypatch.setattr(report_summary_module, "create_chat_model", create_model)
    settings = AppSettings(
        openai_api_key="test-key",
        remedy_llm_model="gpt-6-luna",
        remedy_llm_reasoning_effort="medium",
    )

    result = generate_executive_summary(_evidence(), settings=settings)

    assert result.status == "generated"
    assert captured["model_name"] == "gpt-6-luna"
    assert captured["settings"] is settings
    assert captured["reasoning_effort"] == "medium"
    assert captured["request_timeout"] == 60
    assert captured["max_retries"] == 0
    assert captured["max_tokens"] == 2048


def test_complete_incremental_new_counts_are_scoped_in_summary() -> None:
    """A complete dependency-input scan is not presented as a full-workspace scan."""
    evidence = replace(
        _evidence(),
        new_findings=SummaryCount(0, scope="incremental_dependency_inputs"),
        new_groups=SummaryCount(0, scope="incremental_dependency_inputs"),
        scan_scope="Incremental dependency inputs (complete)",
        scan_complete=True,
    )
    response = _draft_response()
    response["clauses"][4]["lead_in"] = "The incremental dependency scan recorded"
    model = _StructuredModel(response)

    result = generate_executive_summary(
        evidence,
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert (
        "The incremental dependency scan recorded **0 newly detected findings** across "
        "**0 new vulnerability groups**."
    ) in result.text
    assert "terminal scan coverage is complete for incremental dependency inputs" in result.text


def test_summary_without_credentials_uses_deterministic_fallback() -> None:
    """Missing credentials select the local evidence renderer."""
    called = False

    def model_factory(_settings: AppSettings) -> _StructuredModel:
        nonlocal called
        called = True
        return _StructuredModel({})

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key=""),
        model_factory=model_factory,
    )

    assert result.status == "fallback_no_credentials"
    assert not result.used_llm
    assert not called
    assert result.text == fallback_executive_summary(_evidence()).text


def test_invalid_model_response_falls_back_and_redacts_credentials() -> None:
    """Invalid retries preserve factual fallback and redact secrets."""
    incomplete = _draft_response(duplicate="special_remediations")
    model = _StructuredModel(incomplete)
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-secret-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "fallback_error"
    assert not result.used_llm
    assert "fact-reference mismatch" in (result.error or "")
    assert "missing: new_counts" in (result.error or "")
    assert "repeated: special_remediations" in (result.error or "")
    assert model.calls == 2
    assert "test-secret-key" not in str(result.metadata())
    assert result.text == fallback_executive_summary(_evidence()).text


def test_model_slot_repair_is_bounded_and_keeps_validated_generation() -> None:
    """One repaired schema response may satisfy the required fact references."""
    incomplete = _draft_response(duplicate="special_remediations")
    complete = _draft_response()
    model = _StructuredModel([incomplete, complete])
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert result.used_llm
    assert model.calls == 2
    assert result.text.startswith("The run began with")
    assert "Final QA accepted remediation for" in result.text
    assert result.text != fallback_executive_summary(_evidence()).text


def test_model_repair_also_fixes_category_labels() -> None:
    """A category label and unsupported count in a lead-in get one repair call."""
    invalid = _draft_response()
    invalid["clauses"][0]["lead_in"] = "Baseline: The run began with 5 findings."
    model = _StructuredModel([invalid, _draft_response()])

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.used_llm
    assert model.calls == 2
    assert "Baseline:" not in result.text


def test_model_repair_rejects_unapproved_factual_lead_ins() -> None:
    """The schema rejects factual claims even when they fit the plain-text shape."""
    invalid = _draft_response()
    invalid["clauses"][0]["lead_in"] = "All original findings were present at start"
    model = _StructuredModel([invalid, _draft_response()])

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert model.calls == 2
    assert result.text.startswith("The run began with")


def test_summary_prompt_and_rendering_use_natural_bolded_prose() -> None:
    """The generated lead-ins avoid labels and renderer facts bold numeric figures."""
    model = _StructuredModel(_draft_response())
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    messages = model.messages
    assert "category followed by a colon" in messages[0].content
    assert '"fact": "original_counts"' in messages[1].content
    assert '"new_counts": ["We could not confirm"' in messages[1].content
    assert "The run began with **5 original findings**" in result.text
    assert not any(
        f"{label}:" in result.text.casefold()
        for label in ("baseline", "accepted outcomes", "remaining follow-up", "evidence limits")
    )


def test_summary_fingerprint_changes_when_special_evidence_changes() -> None:
    """Source fingerprints change when resolved remediation evidence changes."""
    original = _evidence()
    changed_remediation = replace(
        original.special_remediations[0],
        package="ws",
        details="A different accepted source edit.",
    )
    changed = replace(original, special_remediations=(changed_remediation,))

    first = fallback_executive_summary(original)
    second = fallback_executive_summary(changed)

    assert first.source_fingerprint != second.source_fingerprint


def test_write_report_atomic_writes_exact_content_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    """A successful write creates the report and leaves no sibling temporary file."""
    report_path = tmp_path / "reports" / "remediation_run-42.md"
    markdown = "# Remediation report\n\nPackage: lodash\nStatus: complete\n"

    write_report_atomic(report_path, markdown)

    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8") == markdown
    assert not list(report_path.parent.glob(f".{report_path.name}.*.tmp"))


def test_concurrent_report_writers_keep_independent_atomic_temporaries(
    tmp_path: Path, monkeypatch
) -> None:
    """Interleaved writers do not consume each other's temporary file."""
    report_path = tmp_path / "report.md"
    first_writer_at_replace = threading.Event()
    second_writer_replaced = threading.Event()
    original_replace = Path.replace
    failures: list[BaseException] = []

    def controlled_replace(source: Path, target: Path) -> Path:
        if threading.current_thread().name == "report-writer-a":
            first_writer_at_replace.set()
            assert second_writer_replaced.wait(timeout=5)
        result = original_replace(source, target)
        if threading.current_thread().name == "report-writer-b":
            second_writer_replaced.set()
        return result

    monkeypatch.setattr(Path, "replace", controlled_replace)

    def write(markdown: str) -> None:
        try:
            write_report_atomic(report_path, markdown)
        except BaseException as exc:  # capture thread failures for the assertion
            failures.append(exc)

    first = threading.Thread(target=write, args=("writer a\n",), name="report-writer-a")
    second = threading.Thread(target=write, args=("writer b\n",), name="report-writer-b")
    first.start()
    assert first_writer_at_replace.wait(timeout=5)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert report_path.read_text(encoding="utf-8") in {"writer a\n", "writer b\n"}
    assert not list(tmp_path.glob(".report.md.*.tmp"))
