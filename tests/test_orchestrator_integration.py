"""
Integration tests for the Phase 5 orchestrator graph, specifically testing the newly added triage pipeline connection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.graph import (
    build_orchestrator_graph,
    route_after_triage,
    run_orchestrator,
    triage_node,
)
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    use_trajectory_recorder,
)


def _issue(issue_id: str | None = None) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        id=issue_id or str(uuid4()),
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )


def _group(group_id: str | None = None) -> VulnerabilityGroup:
    issue = _issue()
    return VulnerabilityGroup(
        group_id=group_id or str(uuid4()),
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path="package.json",
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _initial_state(tmp_path: Path, issues: list[VulnerabilityIssue], system_context: SystemContext):
    return {
        "repo_root": str(tmp_path),
        "valid_groups": [],
        "issues": issues,
        "system_context": system_context,
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "changed_files": [],
        "workspace_volume": None,
        "status": "pending",
        "errors": [],
    }


class TestTriageNodeIntegration:
    def test_triage_records_a_named_observability_span(self, tmp_path: Path):
        """Initial triage is visible in the local/LangSmith callback trace."""
        issues = [_issue()]
        state = _initial_state(tmp_path, issues, SystemContext(scan_id="trace-test"))
        recorder = TrajectoryRecorder()
        with (
            patch(
                "remediation_engine.orchestration.graph.run_triage_pipeline",
                return_value=[],
            ),
            use_trajectory_recorder(recorder),
        ):
            triage_node(state)

        assert any(span["name"] == "triage.pipeline" for span in recorder.spans())

    def test_route_after_triage(self):
        assert route_after_triage({"status": "triage_completed"}) == "workspace_builder"
        assert route_after_triage({"status": "triage_skipped"}) == "workspace_builder"
        assert route_after_triage({"status": "triage_completed_no_work"}) == "teardown"
        assert route_after_triage({"status": "failed"}) == "teardown"

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_skipped_when_no_issues(self, mock_pipeline, tmp_path: Path):
        # No issues provided
        state = {
            "repo_root": str(tmp_path),
            "valid_groups": [],
            "constraints_ledger": [],
            "retry_counts": {},
            "group_strategies": {},
            "qa_evaluations": {},
            "action_summaries": [],
            "changed_files": [],
            "workspace_volume": None,
            "status": "pending",
            "errors": [],
        }

        # We need to mock the rest of the nodes so we don't actually run them
        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_ws.return_value = {"status": "workspace_ready"}
            mock_teardown.return_value = {"status": "completed"}

            result = graph.invoke(state)

            mock_pipeline.assert_not_called()
            mock_ws.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_completed_with_valid_groups(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        group1 = _group()
        triage_res = TriageResult(
            group_id=group1.group_id,
            is_valid=True,
            false_positive_reason=None,
            revised_priority=Severity.HIGH,
            recommended_issue_id=issues[0].id,
            priority_reasoning="test reasoning",
            validity_confidence_score=0.9,
            priority_confidence_score=0.9,
            triage_method="static",
        )

        # Mock run_triage_pipeline to return one valid group
        mock_pipeline.return_value = [(group1, triage_res)]

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_supervisor_node") as mock_supervisor,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_ws.return_value = {"status": "workspace_ready"}
            mock_supervisor.return_value = {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "active_target_group_ids": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }
            mock_teardown.return_value = {"status": "phase5_refactor_blocked"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once_with(issues, system_context, str(tmp_path))
            assert group1.group_id in [g.group_id for g in result.get("valid_groups", [])]
            mock_ws.assert_called_once()
            mock_supervisor.assert_called_once()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_completed_no_work(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        group1 = _group()
        triage_res = TriageResult(
            group_id=group1.group_id,
            is_valid=False,
            false_positive_reason="test code",
            revised_priority=Severity.INFO,
            recommended_issue_id=issues[0].id,
            priority_reasoning="test reasoning",
            validity_confidence_score=0.9,
            priority_confidence_score=0.9,
            triage_method="static",
        )

        # Mock run_triage_pipeline to return only invalid groups
        mock_pipeline.return_value = [(group1, triage_res)]

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_teardown.return_value = {"status": "completed"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once_with(issues, system_context, str(tmp_path))
            assert len(result.get("valid_groups", [])) == 0
            # Should skip workspace builder and go directly to teardown
            mock_ws.assert_not_called()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_failure_routes_to_teardown(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        mock_pipeline.side_effect = RuntimeError("Something went wrong")

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_teardown.return_value = {"status": "completed"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once()
            assert len(result.get("valid_groups", [])) == 0
            assert len(result.get("errors", [])) > 0
            mock_ws.assert_not_called()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.orchestrator_engine")
    def test_run_orchestrator_with_triage_params(self, mock_engine, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        mock_engine.invoke.return_value = {"status": "completed", "valid_groups": [_group()]}

        with patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ):
            result = run_orchestrator(
                str(tmp_path),
                valid_groups=[],
                issues=issues,
                system_context=system_context,
            )

        invoked_state = mock_engine.invoke.call_args[0][0]
        assert invoked_state["issues"] == issues
        assert invoked_state["system_context"] == system_context
        assert invoked_state["repo_root"] == str(tmp_path)
        assert result["status"] == "completed"
