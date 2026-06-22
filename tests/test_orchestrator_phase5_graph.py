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
    route_after_workspace_builder,
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
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "group_statuses": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "changed_files": [],
        "workspace_volume": None,
        "status": "pending",
        "next_routing_step": "",
        "active_target_group_ids": [],
        "feedback_by_group": {},
        "supervisor_instructions": "",
        "eval_status": "",
        "errors": [],
    }


class TestPhase5Routing:
    def test_route_after_workspace_builder_routes_to_supervisor(self):
        assert route_after_workspace_builder({"status": "workspace_ready"}) == "supervisor"

    def test_route_after_workspace_builder_failure_routes_to_teardown(self):
        assert route_after_workspace_builder({"status": "workspace_build_failed"}) == "teardown"

    def test_route_after_workspace_builder_unknown_status_routes_to_teardown(self):
        assert route_after_workspace_builder({"status": "something_else"}) == "teardown"


class TestPhase5RunOrchestrator:
    def test_run_orchestrator_builds_initial_state_and_invokes_graph(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}

        with patch("src.orchestrator.graph.orchestrator_engine", mock_engine), patch(
            "src.orchestrator.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        invoked_state = mock_engine.invoke.call_args[0][0]
        assert invoked_state["repo_root"] == str(tmp_path)
        assert invoked_state["valid_groups"] == groups
        assert invoked_state["constraints_ledger"] == []
        assert invoked_state["changed_files"] == []
        assert invoked_state["group_statuses"] == {}
        assert invoked_state["next_routing_step"] == ""
        assert result["status"] == "completed"

    def test_run_orchestrator_passes_config_and_surfaces_trace_metadata(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}
        run_id = uuid4()
        config = {
            "run_id": run_id,
            "run_name": "phase5_orchestrator",
            "tags": ["phase-5", "orchestrator", "langgraph"],
            "metadata": {"repo_name": tmp_path.name},
        }

        with patch("src.orchestrator.graph.orchestrator_engine", mock_engine), patch(
            "src.orchestrator.graph.build_phase5_runnable_config",
            return_value=(config, run_id),
        ), patch(
            "src.orchestrator.graph.resolve_phase5_trace_url",
            return_value="https://smith.langchain.com/o/test/projects/p/runs/r",
        ):
            result = run_orchestrator(str(tmp_path), groups)

        invoked_state, invoked_config = mock_engine.invoke.call_args[0]
        assert invoked_state["repo_root"] == str(tmp_path)
        assert invoked_config == config
        assert result["langsmith_run_id"] == str(run_id)
        assert result["langsmith_trace_url"] == "https://smith.langchain.com/o/test/projects/p/runs/r"

    def test_run_orchestrator_succeeds_when_trace_url_lookup_fails(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}
        run_id = uuid4()
        config = {"run_id": run_id}

        with patch("src.orchestrator.graph.orchestrator_engine", mock_engine), patch(
            "src.orchestrator.graph.build_phase5_runnable_config",
            return_value=(config, run_id),
        ), patch(
            "src.orchestrator.graph.resolve_phase5_trace_url",
            return_value=None,
        ):
            result = run_orchestrator(str(tmp_path), groups)

        assert result["status"] == "completed"
        assert result["langsmith_run_id"] == str(run_id)
        assert "langsmith_trace_url" not in result


class TestPhase5GraphIntegration:
    def test_workspace_builder_success_routes_through_supervisor_to_teardown(self, tmp_path):
        """After workspace_builder succeeds, supervisor routes, then teardown runs."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]

        workspace_builder = MagicMock(
            return_value={
                "status": "workspace_ready",
                "workspace_volume": "agent_workspace_deadbeef",
            }
        )
        # Supervisor routes directly to teardown (simulates no workable groups)
        supervisor = MagicMock(
            return_value={
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "active_target_group_ids": [],
                "group_statuses": {},
                "group_strategies": {},
                "retry_counts": {},
                "constraints_ledger": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }
        )
        teardown = MagicMock(
            return_value={
                "status": "completed",
                "workspace_volume": None,
            }
        )

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), \
             patch("src.orchestrator.graph.run_supervisor_node", supervisor), \
             patch("src.orchestrator.graph.run_teardown_node", teardown):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert workspace_builder.call_count == 1
        assert supervisor.call_count == 1
        assert teardown.call_count == 1
        assert result["status"] == "completed"

    def test_workspace_builder_failure_still_tears_down(self, tmp_path):
        groups = [_group(IssueType.SAST, file_path="routes/login.ts")]

        workspace_builder = MagicMock(
            return_value={
                "status": "workspace_build_failed",
                "workspace_volume": "agent_workspace_deadbeef",
                "errors": ["copy failed"],
            }
        )
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), \
             patch("src.orchestrator.graph.run_teardown_node", teardown):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert workspace_builder.call_count == 1
        assert teardown.call_count == 1
        assert result["status"] == "completed"

    def test_supervisor_routes_to_update_subagent_then_back(self, tmp_path):
        """Supervisor routes to update_subagent once, then supervisor routes to teardown."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        gid = groups[0].group_id

        workspace_builder = MagicMock(return_value={
            "status": "workspace_ready",
            "workspace_volume": "vol123",
        })

        call_count = {"n": 0}

        def supervisor_side_effect(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "status": "supervisor_routed",
                    "next_routing_step": "update_subagent",
                    "active_target_group_ids": [gid],
                    "group_statuses": {},
                    "group_strategies": {},
                    "retry_counts": {},
                    "constraints_ledger": [],
                    "feedback_by_group": {},
                    "supervisor_instructions": "bump it",
                }
            return {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "active_target_group_ids": [],
                "group_statuses": {},
                "group_strategies": {},
                "retry_counts": {},
                "constraints_ledger": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }

        supervisor = MagicMock(side_effect=supervisor_side_effect)
        update_subagent = MagicMock(return_value={"errors": [], "group_statuses": {}})
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), \
             patch("src.orchestrator.graph.run_supervisor_node", supervisor), \
             patch("src.orchestrator.graph.run_update_subagent_node", update_subagent), \
             patch("src.orchestrator.graph.run_teardown_node", teardown):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert supervisor.call_count == 2
        assert teardown.call_count == 1

    def test_supervisor_routes_to_qa_critic_then_back(self, tmp_path):
        """Supervisor routes to qa_critic once, then to teardown."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]

        workspace_builder = MagicMock(return_value={
            "status": "workspace_ready",
            "workspace_volume": "vol123",
        })

        call_count = {"n": 0}

        def supervisor_side_effect(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "status": "supervisor_routed",
                    "next_routing_step": "qa_critic",
                    "active_target_group_ids": [],
                    "group_statuses": {},
                    "group_strategies": {},
                    "retry_counts": {},
                    "constraints_ledger": [],
                    "feedback_by_group": {},
                    "supervisor_instructions": "run qa",
                }
            return {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "active_target_group_ids": [],
                "group_statuses": {},
                "group_strategies": {},
                "retry_counts": {},
                "constraints_ledger": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }

        supervisor = MagicMock(side_effect=supervisor_side_effect)
        qa_critic = MagicMock(return_value={
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "errors": [],
        })
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("src.orchestrator.graph.run_workspace_builder_node", workspace_builder), \
             patch("src.orchestrator.graph.run_supervisor_node", supervisor), \
             patch("src.orchestrator.graph.run_qa_critic_node", qa_critic), \
             patch("src.orchestrator.graph.run_teardown_node", teardown):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert supervisor.call_count == 2
        assert qa_critic.call_count == 1
        assert teardown.call_count == 1


class TestPhase5Exports:
    def test_phase4_and_phase5_exports_are_available(self):
        assert callable(build_remediation_graph)
        assert remediation_engine is not None
        assert callable(run_remediation)
        assert callable(build_orchestrator_graph)
        assert orchestrator_engine is not None
        assert callable(run_orchestrator)

    def test_graph_compiles_without_error(self):
        graph = build_orchestrator_graph()
        assert graph is not None
