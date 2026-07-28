"""
Tests for the Phase 5 LangGraph orchestrator wiring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    SCARemediationStage,
    Severity,
    TaskAttemptSnapshot,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration import (
    build_orchestrator_graph,
    orchestrator_engine,
    run_orchestrator,
)
from remediation_engine.orchestration.graph import (
    route_after_workspace_builder,
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
    run_qa_critic_from_orchestrator,
)
from remediation_engine.orchestration.supervisor_node import _instruction_digest
from remediation_engine.orchestration.task_utils import build_initial_remediation_task


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

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
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

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(config, run_id),
        ), patch(
            "remediation_engine.orchestration.graph.resolve_phase5_trace_url",
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

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(config, run_id),
        ), patch(
            "remediation_engine.orchestration.graph.resolve_phase5_trace_url",
            return_value=None,
        ):
            result = run_orchestrator(str(tmp_path), groups)

        assert result["status"] == "completed"
        assert result["langsmith_run_id"] == str(run_id)
        assert "langsmith_trace_url" not in result

    def test_run_orchestrator_surfaces_local_trajectory_path(self, tmp_path, monkeypatch):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        trajectory_path = result["trajectory_path"]
        assert trajectory_path.startswith(str(trajectory_dir))
        assert trajectory_path.endswith(".md")
        assert trajectory_dir.joinpath(trajectory_path.split("\\")[-1]).exists()

    def test_failed_orchestration_still_writes_trajectory_and_preserves_error(
        self, tmp_path, monkeypatch
    ):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        mock_engine = MagicMock()
        mock_engine.invoke.side_effect = RuntimeError("graph exploded")

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ), pytest.raises(RuntimeError, match="graph exploded"):
            run_orchestrator(str(tmp_path), groups)

        files = list(trajectory_dir.glob("*.md"))
        assert len(files) == 1
        assert "graph exploded" in files[0].read_text(encoding="utf-8")

    def test_trajectory_export_failure_does_not_mask_orchestration_error(
        self, tmp_path
    ):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.side_effect = RuntimeError("graph exploded")

        with patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine), patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ), patch(
            "remediation_engine.orchestration.graph.export_phase5_trajectory",
            side_effect=OSError("disk full"),
        ), pytest.raises(RuntimeError, match="graph exploded"):
            run_orchestrator(str(tmp_path), groups)


class TestPhase5GraphIntegration:
    def test_update_wrapper_preserves_exact_task_instruction(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        task.instruction = 'Update "lodash" in package.json to version "4.17.22".'
        state = _initial_state(tmp_path, groups)
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["task_queue"] = {"task-1": task}
        state["active_target_task_ids"] = ["task-1"]
        state["supervisor_instructions"] = "Use registry lookup to find a safe compatible remediation."
        state["action_summaries"] = [
            AgentActionSummary(
                task_id="task-1",
                status=AgentActionStatus.SURRENDER,
                summary="Previous version bump failed manifest validation.",
            )
        ]

        update_subagent = MagicMock(return_value={"errors": [], "action_summaries": []})

        with patch("remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent):
            run_update_subagent_from_orchestrator(state)

        subagent_state = update_subagent.call_args[0][0]
        assert subagent_state["target_tasks"][0].instruction == 'Update "lodash" in package.json to version "4.17.22".'
        assert subagent_state["previous_action_summaries_by_task"]["task-1"] == "Previous version bump failed manifest validation."
        assert "supervisor_instruction" not in subagent_state

    def test_update_wrapper_surfaces_retry_diagnostics(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        state = _initial_state(tmp_path, groups)
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["task_queue"] = {"task-1": task}
        state["active_target_task_ids"] = ["task-1"]

        diagnostics = UpdateRetryDiagnostics(
            task_id="task-1",
            registry_query_performed=True,
            attempted_versions=["4.17.22"],
            candidate_versions_considered=["4.17.22", "4.17.21"],
            latest_version_seen="4.17.22",
            exhausted_update_path=False,
        )
        update_subagent = MagicMock(
            return_value={
                "errors": [],
                "action_summaries": [],
                "retry_diagnostics_by_task": {"task-1": diagnostics},
            }
        )

        with patch("remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent):
            result = run_update_subagent_from_orchestrator(state)

        assert result["retry_diagnostics_by_task"]["task-1"] == diagnostics

    def test_workaround_wrapper_tags_compatibility_summary_from_snapshot(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.WORKAROUND_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        task.instruction = "Apply the source workaround."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-workaround",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            selected_version=None,
            instruction=task.instruction,
            instruction_digest=_instruction_digest(task.instruction),
            dispatch_node="workaround_subagent",
        )
        task = task.model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": snapshot.attempt_id,
            }
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            }
        )
        untagged = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="workaround bypassed",
        )
        with patch(
            "remediation_engine.orchestration.graph.run_workaround_subagent_node",
            return_value={
                "action_summaries": [untagged],
                "action_summary": untagged,
                "worker_results_by_attempt": {},
                "errors": [],
            },
        ):
            result = run_workaround_subagent_from_orchestrator(state)

        summary = result["action_summaries"][0]
        assert summary.attempt_id == snapshot.attempt_id
        assert summary.task_revision == snapshot.task_revision
        assert summary.instruction_digest == snapshot.instruction_digest

    def test_update_wrapper_rejects_contradictory_snapshot_before_worker(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": "attempt-update",
                "selected_version": "4.17.22",
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-update",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            selected_version="4.17.21",
            instruction=task.instruction,
            instruction_digest=_instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            }
        )
        worker = MagicMock()
        with patch("remediation_engine.orchestration.graph.run_update_subagent_node", worker):
            result = run_update_subagent_from_orchestrator(state)

        worker.assert_not_called()
        assert result["active_target_task_ids"] == []
        assert any(
            event.error_code == "DISPATCH_SNAPSHOT_CONTRADICTION"
            for event in result["consistency_events"]
        )

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

        with patch("remediation_engine.orchestration.graph.run_workspace_builder_node", workspace_builder), \
             patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor), \
             patch("remediation_engine.orchestration.graph.run_teardown_node", teardown):
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

        with patch("remediation_engine.orchestration.graph.run_workspace_builder_node", workspace_builder), \
             patch("remediation_engine.orchestration.graph.run_teardown_node", teardown):
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

        with patch("remediation_engine.orchestration.graph.run_workspace_builder_node", workspace_builder), \
             patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor), \
             patch("remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent), \
             patch("remediation_engine.orchestration.graph.run_teardown_node", teardown):
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
            "qa_investigation_report": "# INVESTIGATIVE REPORT\n## Install Analysis",
            "baseline_scan_identifiers": ["CVE-2021-44228"],
            "post_remediation_scan_identifiers": ["CVE-2025-10001"],
            "new_vulnerability_identifiers": ["CVE-2025-10001"],
            "new_vulnerability_status": "detected",
            "status": "qa_completed",
            "errors": [],
        })
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with patch("remediation_engine.orchestration.graph.run_workspace_builder_node", workspace_builder), \
             patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor), \
             patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic), \
             patch("remediation_engine.orchestration.graph.run_teardown_node", teardown):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert supervisor.call_count == 2
        assert qa_critic.call_count == 1
        assert teardown.call_count == 1
        assert result["qa_investigation_report"].startswith("# INVESTIGATIVE REPORT")
        assert result["new_vulnerability_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_status"] == "detected"

    def test_qa_wrapper_scopes_valid_groups_to_active_batch(self, tmp_path):
        groups = [
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
        ]
        state = _initial_state(tmp_path, groups)
        state["active_target_group_ids"] = [groups[1].group_id]

        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {},
                "eval_status": "all_passed",
                "qa_investigation_report": "# INVESTIGATIVE REPORT\n## Install Analysis",
                "status": "qa_completed",
                "errors": [],
            }
        )

        with patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic):
            result = run_qa_critic_from_orchestrator(state)

        scoped_state = qa_critic.call_args[0][0]
        assert [group.group_id for group in scoped_state["valid_groups"]] == [groups[1].group_id]
        assert result["status"] == "qa_completed"


class TestPhase5Exports:
    def test_phase5_exports_are_available(self):
        assert callable(build_orchestrator_graph)
        assert orchestrator_engine is not None
        assert callable(run_orchestrator)

    def test_graph_compiles_without_error(self):
        graph = build_orchestrator_graph()
        assert graph is not None


