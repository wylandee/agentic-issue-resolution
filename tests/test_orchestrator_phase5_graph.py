"""
Tests for the Phase 5 LangGraph orchestrator wiring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator import (
    build_orchestrator_graph,
    build_remediation_graph,
    orchestrator_engine,
    remediation_engine,
    run_orchestrator,
    run_remediation,
)
from src.orchestrator.graph import (
    route_after_remedy_agent,
    route_after_scanner,
    route_after_tester,
    route_after_workspace_builder,
    route_after_workspace_sync,
)


def _issue(issue_type: IssueType, file_path: str | None = None) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC if issue_type == IssueType.SCA else IssueSource.SEMGREP,
        issue_type=issue_type,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228" if issue_type == IssueType.SCA else None,
        package_name="lodash" if issue_type == IssueType.SCA else None,
        package_version="4.17.15" if issue_type == IssueType.SCA else None,
        rule_id="javascript.xss" if issue_type == IssueType.SAST else None,
        file_path=file_path,
    )


def _fix_plan(status: FixPlanStatus) -> FixPlan:
    if status == FixPlanStatus.VERSION_FOUND:
        return FixPlan(
            status=status,
            fixed_version="4.17.21",
            instruction="Upgrade to 4.17.21",
            strategy_used="osv_api",
        )
    if status == FixPlanStatus.WORKAROUND_FOUND:
        return FixPlan(
            status=status,
            workaround_snippets=["Disable vulnerable code path"],
            instruction="Apply workaround",
            strategy_used="serper",
        )
    return FixPlan(
        status=status,
        fixed_version=None,
        workaround_snippets=None,
        instruction="No fix available",
        strategy_used="none",
    )


def _group(
    issue_type: IssueType,
    *,
    fix_plan: FixPlan | None = None,
    file_path: str | None = None,
) -> VulnerabilityGroup:
    issue = _issue(issue_type, file_path=file_path)
    return VulnerabilityGroup(
        group_id=f"{issue_type.value}:{uuid4()}",
        issue_type=issue_type,
        vulnerable_component="lodash" if issue_type == IssueType.SCA else "javascript.xss",
        file_path=file_path,
        cve_ids=["CVE-2021-44228"] if issue_type == IssueType.SCA else [],
        versions=["4.17.15"] if issue_type == IssueType.SCA else [],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=fix_plan,
    )


def _initial_state(tmp_path, groups):
    return {
        "repo_root": str(tmp_path),
        "valid_groups": groups,
        "retry_count": 0,
        "max_retries": 3,
        "messages": [],
        "changed_files": [],
        "workspace_volume": None,
        "install_failures": None,
        "test_failures": None,
        "scan_failures": None,
        "status": "pending",
        "errors": [],
    }


class TestPhase5Routing:
    def test_route_after_workspace_builder(self):
        assert route_after_workspace_builder({"status": "workspace_ready"}) == "remedy_agent"
        assert route_after_workspace_builder({"status": "workspace_build_failed"}) == "teardown"

    def test_route_after_remedy_agent(self):
        assert route_after_remedy_agent({"status": "edits_completed"}) == "workspace_sync"
        for status in ("no_changes_made", "remedy_failed", "max_retries_exceeded"):
            assert route_after_remedy_agent({"status": status}) == "teardown"

    def test_route_after_workspace_sync_for_sca_version_fix(self):
        state = {
            "status": "dependencies_ready",
            "valid_groups": [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))],
        }
        assert route_after_workspace_sync(state) == "scanner"

    def test_route_after_workspace_sync_for_sast_or_workaround(self):
        state = {
            "status": "dependencies_ready",
            "valid_groups": [_group(IssueType.SAST, file_path="routes/login.ts")],
        }
        assert route_after_workspace_sync(state) == "tester"

        for fix_status in (FixPlanStatus.WORKAROUND_FOUND, FixPlanStatus.NO_FIX):
            state = {
                "status": "dependencies_ready",
                "valid_groups": [_group(IssueType.SCA, fix_plan=_fix_plan(fix_status))],
            }
            assert route_after_workspace_sync(state) == "tester"

    def test_route_after_workspace_sync_failure_loops_to_remedy(self):
        assert route_after_workspace_sync({"status": "dependency_sync_failed"}) == "remedy_agent"

    def test_route_after_scanner(self):
        assert route_after_scanner({"status": "scanned"}) == "tester"
        assert route_after_scanner({"status": "scan_failed"}) == "remedy_agent"
        assert route_after_scanner({"status": "unknown"}) == "teardown"

    def test_route_after_tester(self):
        assert route_after_tester({"status": "tested"}) == "teardown"
        assert route_after_tester({"status": "test_failed"}) == "remedy_agent"
        assert route_after_tester({"status": "unknown"}) == "teardown"


class TestPhase5RunOrchestrator:
    def test_run_orchestrator_builds_initial_state_and_invokes_graph(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}

        with patch("src.orchestrator.graph.orchestrator_engine", mock_engine):
            result = run_orchestrator(str(tmp_path), groups, max_retries=5)

        invoked_state = mock_engine.invoke.call_args[0][0]
        assert invoked_state["repo_root"] == str(tmp_path)
        assert invoked_state["valid_groups"] == groups
        assert invoked_state["max_retries"] == 5
        assert invoked_state["messages"] == []
        assert result["status"] == "completed"


class TestPhase5GraphIntegration:
    def test_success_path_routes_through_scanner_then_tester(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]

        workspace_builder = MagicMock(return_value={"status": "workspace_ready", "workspace_volume": "agent_workspace_deadbeef"})
        remedy = MagicMock(return_value={"status": "edits_completed", "changed_files": ["package.json"], "messages": []})
        workspace_sync = MagicMock(return_value={"status": "dependencies_ready"})
        scanner = MagicMock(return_value={"status": "scanned"})
        tester = MagicMock(return_value={"status": "tested"})
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), patch(
            "src.orchestrator.graph.run_remedy_agent",
            remedy,
        ), patch(
            "src.orchestrator.graph.run_workspace_sync_node",
            workspace_sync,
        ), patch(
            "src.orchestrator.graph.run_scanner_node",
            scanner,
        ), patch(
            "src.orchestrator.graph.run_tester_node",
            tester,
        ), patch(
            "src.orchestrator.graph.run_teardown_node",
            teardown,
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert workspace_builder.call_count == 1
        assert remedy.call_count == 1
        assert workspace_sync.call_count == 1
        assert scanner.call_count == 1
        assert tester.call_count == 1
        assert teardown.call_count == 1
        assert result["status"] == "completed"

    def test_sast_path_skips_scanner_and_goes_to_tester(self, tmp_path):
        groups = [_group(IssueType.SAST, file_path="routes/login.ts")]

        workspace_builder = MagicMock(return_value={"status": "workspace_ready", "workspace_volume": "agent_workspace_deadbeef"})
        remedy = MagicMock(return_value={"status": "edits_completed", "changed_files": ["routes/login.ts"], "messages": []})
        workspace_sync = MagicMock(return_value={"status": "dependencies_ready"})
        scanner = MagicMock(return_value={"status": "scanned"})
        tester = MagicMock(return_value={"status": "tested"})
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), patch(
            "src.orchestrator.graph.run_remedy_agent",
            remedy,
        ), patch(
            "src.orchestrator.graph.run_workspace_sync_node",
            workspace_sync,
        ), patch(
            "src.orchestrator.graph.run_scanner_node",
            scanner,
        ), patch(
            "src.orchestrator.graph.run_tester_node",
            tester,
        ), patch(
            "src.orchestrator.graph.run_teardown_node",
            teardown,
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert scanner.call_count == 0
        assert tester.call_count == 1
        assert result["status"] == "completed"

    def test_workspace_sync_failure_loops_back_to_remedy(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]

        workspace_builder = MagicMock(return_value={"status": "workspace_ready", "workspace_volume": "agent_workspace_deadbeef"})
        remedy = MagicMock(
            side_effect=[
                {"status": "edits_completed", "changed_files": ["package.json"], "messages": []},
                {"status": "max_retries_exceeded"},
            ]
        )
        workspace_sync = MagicMock(return_value={"status": "dependency_sync_failed", "install_failures": "npm failed"})
        scanner = MagicMock(return_value={"status": "scanned"})
        tester = MagicMock(return_value={"status": "tested"})
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), patch(
            "src.orchestrator.graph.run_remedy_agent",
            remedy,
        ), patch(
            "src.orchestrator.graph.run_workspace_sync_node",
            workspace_sync,
        ), patch(
            "src.orchestrator.graph.run_scanner_node",
            scanner,
        ), patch(
            "src.orchestrator.graph.run_tester_node",
            tester,
        ), patch(
            "src.orchestrator.graph.run_teardown_node",
            teardown,
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert remedy.call_count == 2
        assert workspace_sync.call_count == 1
        assert scanner.call_count == 0
        assert tester.call_count == 0
        assert result["status"] == "completed"


class TestPhase5Exports:
    def test_phase4_and_phase5_exports_are_available(self):
        assert callable(build_remediation_graph)
        assert remediation_engine is not None
        assert callable(run_remediation)
        assert callable(build_orchestrator_graph)
        assert orchestrator_engine is not None
        assert callable(run_orchestrator)
