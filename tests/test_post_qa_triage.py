"""Regression tests for the Supervisor-owned post-QA triage handoff."""

from __future__ import annotations

from unittest.mock import patch

from remediation_engine.contracts.schemas import (
    DependencyParentContext,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QAEvaluation,
    Severity,
    SystemContext,
    TaskStatus,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.graph import (
    _post_triage_issue_input,
    build_orchestrator_graph,
    post_qa_triage_node,
    run_qa_critic_from_orchestrator,
)
from remediation_engine.orchestration.state import initial_orchestrator_state
from remediation_engine.orchestration.supervisor_node import (
    _deterministic_routing,
    supervisor_router,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task


def _issue(cve: str, *, source: IssueSource = IssueSource.ODC) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id=cve,
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )


def _group(group_id: str, issue: VulnerabilityIssue) -> VulnerabilityGroup:
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component=issue.package_name,
        file_path=issue.file_path,
        cve_ids=[issue.cve_id] if issue.cve_id else [],
        versions=[issue.package_version] if issue.package_version else [],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="4.17.21",
            instruction="Upgrade lodash to 4.17.21.",
            strategy_used="test",
        ),
    )


def _triage_result(group: VulnerabilityGroup) -> TriageResult:
    return TriageResult(
        group_id=group.group_id,
        is_valid=True,
        revised_priority=Severity.HIGH,
        priority_reasoning="Test finding is actionable.",
        validity_confidence_score=1.0,
        priority_confidence_score=1.0,
        recommended_issue_id=group.issues[0].id,
        triage_method="deterministic",
    )


def test_post_triage_reuses_unchanged_groups_and_reopens_changed_tasks():
    unchanged_issue = _issue("CVE-2021-0001")
    changed_issue = _issue("CVE-2021-0002")
    new_issue = _issue("CVE-2026-0003", source=IssueSource.ODC)
    unchanged = _group("sca:package.json:unchanged:UPDATE_VERSION", unchanged_issue)
    changed = _group("sca:package.json:changed:UPDATE_VERSION", changed_issue)
    changed_candidate = _group(
        changed.group_id,
        _issue("CVE-2026-0004", source=IssueSource.ODC),
    )
    new_group = _group("sca:package.json:new:UPDATE_VERSION", new_issue)
    candidate_unchanged = unchanged.model_copy()

    changed_task = build_initial_remediation_task(changed, "task-changed").model_copy(
        update={
            "task_revision": 2,
            "retry_count": 3,
            "status": TaskStatus.QA_PASSED,
        }
    )
    unchanged_task = build_initial_remediation_task(unchanged, "task-unchanged")
    state = initial_orchestrator_state(
        "repo",
        [unchanged, changed],
        issues=[unchanged_issue, changed_issue],
        system_context=SystemContext(),
    )
    state.update(
        {
            "task_queue": {
                "task-changed": changed_task,
                "task-unchanged": unchanged_task,
            },
            "qa_evaluations": {
                changed.group_id: QAEvaluation(task_id=changed.group_id, passed=True),
                unchanged.group_id: QAEvaluation(task_id=unchanged.group_id, passed=True),
            },
            "triage_required": True,
            "new_vulnerability_status": "detected",
            "post_remediation_scan_issues": [new_issue],
            "active_target_task_ids": ["task-changed"],
        }
    )

    with patch(
        "remediation_engine.orchestration.graph.run_triage_pipeline",
        return_value=[
            (candidate_unchanged, _triage_result(candidate_unchanged)),
            (changed_candidate, _triage_result(changed_candidate)),
            (new_group, _triage_result(new_group)),
        ],
    ):
        result = post_qa_triage_node(state)

    groups_by_id = {group.group_id: group for group in result["valid_groups"]}
    assert groups_by_id[unchanged.group_id] is unchanged
    assert groups_by_id[changed.group_id] is changed_candidate
    assert result["triage_reconciliation"] == {
        "reused_group_ids": [unchanged.group_id],
        "changed_group_ids": [changed.group_id],
        "new_group_ids": [new_group.group_id],
        "reappeared_group_ids": [],
        "retained_removed_group_ids": [],
        "removed_group_ids": [],
    }

    reopened = result["task_queue"]["task-changed"]
    assert reopened.status == TaskStatus.PENDING
    assert reopened.task_revision == 3
    assert reopened.retry_count == 0
    assert reopened.current_attempt_id is None
    assert reopened.qa_policy == changed_task.qa_policy
    assert changed.group_id not in result["qa_evaluations"]
    assert result["task_queue"]["task-unchanged"] is unchanged_task
    assert result["active_target_task_ids"] == []


def test_post_triage_reopens_group_when_parent_context_becomes_available():
    """Parent evidence changes group content without changing its canonical ID."""
    issue = _issue("CVE-2026-0005")
    previous = _group("sca:package.json:lodash:UPDATE_VERSION", issue)
    candidate = previous.model_copy(
        update={
            "dependency_ancestry": ["sanitize-html", "lodash"],
            "dependency_versions": {"sanitize-html": "1.4.2", "lodash": "2.4.2"},
            "parent_package_name": "sanitize-html",
            "parent_package_version": "1.4.2",
            "parent_declaration_type": "dependencies",
            "parent_contexts": [
                DependencyParentContext(
                    package_name="sanitize-html",
                    package_version="1.4.2",
                    declaration_type="dependencies",
                    manifest_file="package.json",
                    dependency_ancestry=["sanitize-html", "lodash"],
                    dependency_versions={"sanitize-html": "1.4.2", "lodash": "2.4.2"},
                )
            ],
        }
    )
    task = build_initial_remediation_task(previous, "task-lodash")
    state = initial_orchestrator_state(
        "repo",
        [previous],
        issues=[issue],
        system_context=SystemContext(),
    )
    state.update(
        {
            "task_queue": {"task-lodash": task},
            "triage_required": True,
            "new_vulnerability_status": "detected",
            "post_remediation_scan_issues": [issue],
            "active_target_task_ids": ["task-lodash"],
        }
    )

    with patch(
        "remediation_engine.orchestration.graph.run_triage_pipeline",
        return_value=[(candidate, _triage_result(candidate))],
    ):
        result = post_qa_triage_node(state)

    assert result["triage_reconciliation"]["changed_group_ids"] == [previous.group_id]
    assert result["valid_groups"] == [candidate]
    assert result["task_queue"]["task-lodash"].status == TaskStatus.PENDING
    assert result["task_queue"]["task-lodash"].task_revision == 1


def test_post_triage_uses_post_scan_odc_issues_and_retains_non_odc_baseline():
    sast_issue = VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        rule_id="javascript.xss",
        file_path="src/app.js",
    )
    baseline_odc_issue = _issue("CVE-2021-0001")
    post_odc_issue = _issue("CVE-2026-0002")
    state = {
        "issues": [sast_issue, baseline_odc_issue],
        "post_remediation_scan_issues": [post_odc_issue],
        "valid_groups": [],
    }

    issue_input = _post_triage_issue_input(state)

    assert issue_input == [sast_issue, post_odc_issue]


def test_qa_wrapper_propagates_scan_snapshot_and_marks_triage_required():
    issue = _issue("CVE-2026-0001")
    group = _group("group-1", issue)
    state = initial_orchestrator_state("repo", [group], issues=[issue])
    post_issue = _issue("CVE-2026-0002")

    with patch(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        return_value={
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "post_remediation_scan_issues": [post_issue],
            "post_remediation_scan_identifiers": [post_issue.cve_id],
            "new_vulnerability_identifiers": [post_issue.cve_id],
            "new_vulnerability_status": "detected",
        },
    ):
        result = run_qa_critic_from_orchestrator(state)

    assert result["post_remediation_scan_issues"] == [post_issue]
    assert result["new_vulnerability_identifiers"] == [post_issue.cve_id]
    assert result["triage_required"] is True


def test_supervisor_routes_parseable_qa_results_to_triage_before_teardown():
    decision = _deterministic_routing(
        task_queue={},
        group_by_id={},
        qa_evaluations={},
        retry_diagnostics_by_task={},
        current_status="qa_failed",
        triage_required=True,
    )

    assert decision.next_node == "triage"
    assert supervisor_router({"next_routing_step": "triage"}) == "triage"


def test_graph_contains_separate_initial_and_post_qa_triage_nodes():
    node_names = set(build_orchestrator_graph().get_graph().nodes)

    assert "initial_triage" in node_names
    assert "triage" in node_names


def test_graph_routes_qa_through_supervisor_to_triage_and_back():
    initial_issue = _issue("CVE-2021-0001")
    post_issue = _issue("CVE-2026-0005")
    initial_group = _group("group-initial", initial_issue)
    post_group = _group("group-post", post_issue)
    state = initial_orchestrator_state(
        "repo",
        [initial_group],
        issues=[initial_issue],
        system_context=SystemContext(),
    )

    supervisor_calls = {"count": 0}

    def supervisor_side_effect(_state):
        supervisor_calls["count"] += 1
        if supervisor_calls["count"] == 1:
            return {"next_routing_step": "qa_critic", "status": "supervisor_routed"}
        if supervisor_calls["count"] == 2:
            return {"next_routing_step": "triage", "status": "supervisor_routed"}
        return {"next_routing_step": "teardown", "status": "supervisor_routed"}

    with (
        patch(
            "remediation_engine.orchestration.graph.run_workspace_builder_node",
            return_value={"status": "workspace_ready", "workspace_volume": "vol"},
        ),
        patch(
            "remediation_engine.orchestration.graph.run_supervisor_node",
            side_effect=supervisor_side_effect,
        ),
        patch(
            "remediation_engine.orchestration.graph.run_qa_critic_from_orchestrator",
            return_value={
                "status": "qa_completed",
                "triage_required": True,
                "post_remediation_scan_issues": [post_issue],
                "post_remediation_scan_identifiers": [post_issue.cve_id],
                "new_vulnerability_identifiers": [post_issue.cve_id],
                "new_vulnerability_status": "detected",
            },
        ),
        patch(
            "remediation_engine.orchestration.graph.run_triage_pipeline",
            return_value=[(post_group, _triage_result(post_group))],
        ) as triage_pipeline,
        patch(
            "remediation_engine.orchestration.graph.run_teardown_node",
            return_value={"status": "completed", "workspace_volume": None},
        ),
    ):
        result = build_orchestrator_graph().invoke(state)

    assert supervisor_calls["count"] == 3
    assert triage_pipeline.call_count == 1
    assert result["status"] == "completed"
