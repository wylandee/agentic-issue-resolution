"""
Tests for the Phase 5 LangGraph orchestrator wiring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    MultiPackageAction,
    PackageMutation,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    QATestAttribution,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TaskAttemptSnapshot,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.contracts.solver_models import PortfolioReplanRequest
from remediation_engine.orchestration import (
    build_orchestrator_graph,
    orchestrator_engine,
    run_orchestrator,
)
from remediation_engine.orchestration import graph_wrappers as _graph_wrappers
from remediation_engine.orchestration.graph import (
    MAX_PORTFOLIO_REPLAN_ATTEMPTS,
    _finish_workspace_attempt_snapshot,
    _record_portfolio_replan_attempt,
    route_after_portfolio,
    route_after_workspace_builder,
    run_portfolio_node,
    run_qa_critic_from_orchestrator,
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.supervisor_node import instruction_digest, run_supervisor_node
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from remediation_engine.triage.grouper import group_issues


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
        "qa_evaluations": {},
        "action_summaries": [],
        "changed_files": [],
        "workspace_volume": None,
        "status": "pending",
        "next_routing_step": "",
        "feedback_by_group": {},
        "supervisor_instructions": "",
        "eval_status": "",
        "errors": [],
    }


def _committed_dispatch(task, *, dispatch_node: str):
    """Return a task and immutable snapshot suitable for a graph dispatch."""
    committed_task = task.model_copy(
        update={
            "task_revision": max(1, task.task_revision),
            "current_attempt_id": f"attempt-{task.task_id}",
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=committed_task.current_attempt_id,
        task_id=committed_task.task_id,
        state_revision=1,
        task_revision=committed_task.task_revision,
        strategy_stage=committed_task.strategy_stage,
        qa_policy=committed_task.qa_policy,
        selected_version=committed_task.selected_version,
        instruction=committed_task.instruction,
        instruction_digest=instruction_digest(committed_task.instruction),
        dispatch_node=dispatch_node,
    )
    return committed_task, snapshot


class TestPhase5Routing:
    def test_route_after_workspace_builder_routes_to_portfolio(self):
        assert route_after_workspace_builder({"status": "workspace_ready"}) == "portfolio"

    def test_route_after_workspace_builder_failure_routes_to_teardown(self):
        assert route_after_workspace_builder({"status": "workspace_build_failed"}) == "teardown"

    def test_route_after_workspace_builder_unknown_status_routes_to_teardown(self):
        assert route_after_workspace_builder({"status": "something_else"}) == "teardown"


def test_supervisor_recovers_worker_result_after_portfolio_cleared_active_targets(
    tmp_path,
    monkeypatch,
):
    """A stale portfolio pointer must not strand a valid current attempt."""
    group = _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))
    task = build_initial_remediation_task(group, "task-1").model_copy(
        update={
            "task_revision": 1,
            "selected_version": "2.0.0",
            "allowed_target_versions": ["2.0.0"],
            "portfolio_plan_id": "portfolio-old",
            "instruction": "Update the committed dependency candidate.",
        }
    )
    committed_task, snapshot = _committed_dispatch(task, dispatch_node="update_subagent")
    committed_task = committed_task.model_copy(update={"portfolio_plan_id": "portfolio-old"})
    snapshot = snapshot.model_copy(
        update={
            "portfolio_plan_id": "portfolio-old",
            "selected_version": "2.0.0",
            "allowed_target_versions": ["2.0.0"],
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id=committed_task.task_id,
        task_revision=committed_task.task_revision,
        status=AgentActionStatus.SUCCESS,
        instruction_digest=snapshot.instruction_digest,
        execution_diagnostics=WorkerExecutionDiagnostics(
            executed_versions=["2.0.0"],
            effective_target_version="2.0.0",
        ),
    )
    state = _initial_state(tmp_path, [group])
    state.update(
        {
            "task_queue": {committed_task.task_id: committed_task},
            "active_target_task_ids": [],
            "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            "worker_results_by_attempt": {snapshot.attempt_id: worker_result},
            "portfolio_plan": SimpleNamespace(
                portfolio_plan_id="portfolio-new",
                clusters=[],
                cluster_order=[],
                task_order=[],
                task_to_cluster={},
                diagnostics=[],
            ),
            "status": "supervisor_routed",
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
        lambda *args, **kwargs: ["stale committed portfolio plan"],
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
        lambda *args, **kwargs: False,
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "qa_critic"
    assert result["active_target_task_ids"] == ["task-1"]
    assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
    assert snapshot.attempt_id in result["processed_worker_attempt_ids"]


def test_portfolio_replan_guard_routes_repeated_reason_to_teardown(tmp_path):
    reason = "committed portfolio plan has stale task revision"
    state = {
        "repo_root": str(tmp_path),
        "portfolio_iteration": 3,
        "portfolio_replan_request": PortfolioReplanRequest(reason=reason),
        "portfolio_replan_history": {reason: MAX_PORTFOLIO_REPLAN_ATTEMPTS},
        "portfolio_plan": object(),
        "portfolio_solver_plan": object(),
        "valid_groups": [],
        "task_queue": {},
    }

    with patch("remediation_engine.orchestration.graph.prepare_portfolio_inputs") as prepare:
        result = run_portfolio_node(state)

    prepare.assert_not_called()
    assert result["status"] == "portfolio_replan_guarded"
    assert result["next_routing_step"] == "teardown"
    assert result["portfolio_replan_request"] is None
    assert result["portfolio_replan_history"] == {reason: MAX_PORTFOLIO_REPLAN_ATTEMPTS}
    assert reason in result["errors"][0]
    assert route_after_portfolio(result) == "teardown"


def test_portfolio_replan_guard_normalizes_changing_plan_ids_and_revisions():
    reason_one = (
        "task task-64 portfolio plan 'portfolio-old' differs from committed 'portfolio-a'; "
        "task task-64 is older than committed planned revision 4 (current=3)"
    )
    reason_two = (
        "task task-64 portfolio plan 'portfolio-new' differs from committed 'portfolio-b'; "
        "task task-64 is older than committed planned revision 5 (current=4)"
    )
    history, error = _record_portfolio_replan_attempt(
        {
            "portfolio_replan_history": {reason_one: MAX_PORTFOLIO_REPLAN_ATTEMPTS},
            "portfolio_iteration": 7,
        },
        PortfolioReplanRequest(reason=reason_two),
    )

    assert error is not None
    assert len(history) == 1
    assert next(iter(history.values())) == MAX_PORTFOLIO_REPLAN_ATTEMPTS


def test_portfolio_reconciliation_status_routes_back_to_supervisor():
    assert route_after_portfolio({"status": "portfolio_reconciliation_required"}) == "supervisor"


def test_portfolio_node_blocks_missing_certificate_before_commit(tmp_path, monkeypatch):
    group = _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))
    task = build_initial_remediation_task(group, "task-1")
    candidate_plan = SimpleNamespace(
        plan_id="portfolio-new",
        portfolio_plan_id="portfolio-new",
        diagnostics=[],
        clusters=[],
        solver_plan=SimpleNamespace(
            status="OPTIMAL",
            candidate_catalog_complete=True,
            diagnostics=[],
            selected_plan=SimpleNamespace(
                task_decisions=[],
                selected_candidate_versions={},
                batches=[],
            ),
        ),
    )
    state = _initial_state(tmp_path, [group])
    state["task_queue"] = {"task-1": task}
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.prepare_portfolio_inputs",
        lambda *args, **kwargs: ([group], {"task-1": task}, []),
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.build_certified_portfolio_plan",
        lambda *args, **kwargs: candidate_plan,
    )
    apply_plan = MagicMock()
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.apply_portfolio_plan",
        apply_plan,
    )

    result = run_portfolio_node(state)

    assert result["status"] == "portfolio_unknown"
    assert result["next_routing_step"] == "teardown"
    assert any("missing its package-resolution certificate" in error for error in result["errors"])
    apply_plan.assert_not_called()


def test_portfolio_does_not_commit_partial_plan_around_active_attempt(tmp_path, monkeypatch):
    group = _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))
    task = build_initial_remediation_task(group, "task-1").model_copy(
        update={"current_attempt_id": "attempt-1"}
    )
    previous_plan = SimpleNamespace(portfolio_plan_id="portfolio-old")
    candidate_plan = SimpleNamespace(
        plan_id="portfolio-new",
        portfolio_plan_id="portfolio-new",
        diagnostics=[],
        clusters=[],
        solver_plan=SimpleNamespace(
            status="OPTIMAL",
            diagnostics=[],
            selected_plan=SimpleNamespace(task_decisions=[]),
        ),
    )
    state = _initial_state(tmp_path, [group])
    state.update(
        {
            "task_queue": {"task-1": task},
            "portfolio_plan": previous_plan,
            "portfolio_solver_plan": SimpleNamespace(),
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.prepare_portfolio_inputs",
        lambda *args, **kwargs: ([group], {"task-1": task}, []),
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.build_certified_portfolio_plan",
        lambda *args, **kwargs: candidate_plan,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph._portfolio_certificate_violations",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.apply_portfolio_plan",
        lambda *args, **kwargs: (
            [group],
            {"task-1": task.model_copy(update={"task_revision": 2})},
            ["task 'task-1' has an active attempt; plan decision not applied"],
        ),
    )

    result = run_portfolio_node(state)

    assert result["status"] == "portfolio_reconciliation_required"
    assert result["next_routing_step"] == "supervisor"
    assert result["portfolio_plan"] is previous_plan
    assert result["task_queue"]["task-1"].task_revision == task.task_revision
    assert result["active_target_task_ids"] == ["task-1"]


class TestPhase5RunOrchestrator:
    def test_run_orchestrator_builds_initial_state_and_invokes_graph(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=(None, None),
            ),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        invoked_state = mock_engine.invoke.call_args[0][0]
        assert invoked_state["repo_root"] == str(tmp_path)
        assert invoked_state["valid_groups"] == groups
        assert invoked_state["constraints_ledger"] == []
        assert invoked_state["changed_files"] == []
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

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=(config, run_id),
            ),
            patch(
                "remediation_engine.orchestration.graph.resolve_phase5_trace_url",
                return_value="https://smith.langchain.com/o/test/projects/p/runs/r",
            ),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        invoked_state, invoked_config = mock_engine.invoke.call_args[0]
        assert invoked_state["repo_root"] == str(tmp_path)
        assert invoked_config == config
        assert result["langsmith_run_id"] == str(run_id)
        assert (
            result["langsmith_trace_url"] == "https://smith.langchain.com/o/test/projects/p/runs/r"
        )

    def test_run_orchestrator_succeeds_when_trace_url_lookup_fails(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}
        run_id = uuid4()
        config = {"run_id": run_id}

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=(config, run_id),
            ),
            patch(
                "remediation_engine.orchestration.graph.resolve_phase5_trace_url",
                return_value=None,
            ),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        assert result["status"] == "completed"
        assert result["langsmith_run_id"] == str(run_id)
        assert "langsmith_trace_url" not in result

    def test_run_orchestrator_surfaces_local_trajectory_path(self, tmp_path, monkeypatch):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        report_dir = tmp_path / "reports"
        monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(report_dir))
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = {"status": "completed", "workspace_volume": None}

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=(None, None),
            ),
        ):
            result = run_orchestrator(str(tmp_path), groups)

        trajectory_path = result["trajectory_path"]
        assert trajectory_path.startswith(str(trajectory_dir))
        assert trajectory_path.endswith(".md")
        assert trajectory_dir.joinpath(trajectory_path.split("\\")[-1]).exists()
        assert result["report_status"] == "persisted"
        assert result["report_path"]
        assert Path(result["report_path"]).is_file()
        trajectory_text = Path(trajectory_path).read_text(encoding="utf-8")
        assert '"report_status": "persisted"' in trajectory_text
        assert '"report_path":' in trajectory_text
        assert Path(result["report_path"]).name in trajectory_text

    def test_failed_orchestration_still_writes_trajectory_and_preserves_error(
        self, tmp_path, monkeypatch
    ):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        report_dir = tmp_path / "reports"
        monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(report_dir))
        mock_engine = MagicMock()
        mock_engine.invoke.side_effect = RuntimeError("graph exploded")

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=(None, None),
            ),
            pytest.raises(RuntimeError, match="graph exploded"),
        ):
            run_orchestrator(str(tmp_path), groups)

        files = list(trajectory_dir.glob("*.md"))
        assert len(files) == 1
        assert "graph exploded" in files[0].read_text(encoding="utf-8")

        report_files = list(report_dir.glob("*.md"))
        assert len(report_files) == 1
        assert report_files[0].read_text(encoding="utf-8")

    def test_failed_langsmith_run_is_closed_without_waiting_for_remote_spans(
        self, tmp_path, monkeypatch
    ):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        run_id = uuid4()
        mock_engine = MagicMock()
        mock_engine.invoke.side_effect = RuntimeError("graph exploded")

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=({"run_id": run_id}, run_id),
            ),
            patch("remediation_engine.orchestration.graph.mark_phase5_trace_failed") as mark_trace,
            patch(
                "remediation_engine.orchestration.graph.export_phase5_trajectory",
                return_value=trajectory_dir / "trace.md",
            ) as export_trace,
            pytest.raises(RuntimeError, match="graph exploded"),
        ):
            run_orchestrator(str(tmp_path), groups)

        mark_trace.assert_called_once()
        assert mark_trace.call_args.args[0] == run_id
        assert export_trace.call_args.kwargs["langsmith_enabled"] is False

    def test_keyboard_interrupt_persists_local_snapshot_before_closing_trace(
        self, tmp_path, monkeypatch
    ):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        trajectory_dir = tmp_path / "trajectories"
        monkeypatch.setenv("REMEDIATION_TRAJECTORY_DIR", str(trajectory_dir))
        run_id = uuid4()
        mock_engine = MagicMock()
        mock_engine.invoke.side_effect = KeyboardInterrupt()
        events: list[str] = []
        exported_states: list[dict[str, object]] = []

        def fake_export(**kwargs):
            events.append("export")
            exported_states.append(dict(kwargs["final_state"]))
            path = Path(kwargs["output_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("interrupted trajectory", encoding="utf-8")
            return path

        def fake_mark(*args, **kwargs):
            events.append("mark")

        with (
            patch("remediation_engine.orchestration.graph.orchestrator_engine", mock_engine),
            patch(
                "remediation_engine.orchestration.graph.build_phase5_runnable_config",
                return_value=({"run_id": run_id}, run_id),
            ),
            patch(
                "remediation_engine.orchestration.graph.export_phase5_trajectory",
                side_effect=fake_export,
            ),
            patch(
                "remediation_engine.orchestration.graph.mark_phase5_trace_failed",
                side_effect=fake_mark,
            ),
            patch(
                "remediation_engine.orchestration.graph.run_report_node",
                return_value={"errors": []},
            ),
            patch(
                "remediation_engine.orchestration.graph.finalize_report",
                return_value=("", None),
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            run_orchestrator(str(tmp_path), groups)

        assert events[0] == "export"
        assert events.index("export") < events.index("mark")
        early_state = exported_states[0]
        assert early_state["status"] == "completed_with_errors"
        assert "KeyboardInterrupt" in early_state["errors"][-1]
        assert list(trajectory_dir.glob("*.md"))


class TestPhase5GraphIntegration:
    def test_update_wrapper_preserves_exact_task_instruction(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        task.instruction = 'Update "lodash" in package.json to version "4.17.22".'
        task, snapshot = _committed_dispatch(task, dispatch_node="update_subagent")
        state = _initial_state(tmp_path, groups)
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["task_queue"] = {"task-1": task}
        state["active_target_task_ids"] = ["task-1"]
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}
        state["supervisor_instructions"] = (
            "Use registry lookup to find a safe compatible remediation."
        )
        state["action_summaries"] = [
            AgentActionSummary(
                task_id="task-1",
                status=AgentActionStatus.SURRENDER,
                summary="Previous version bump failed manifest validation.",
                attempt_id=snapshot.attempt_id,
                task_revision=snapshot.task_revision,
                instruction_digest=snapshot.instruction_digest,
            )
        ]

        update_subagent = MagicMock(return_value={"errors": [], "action_summaries": []})

        with patch(
            "remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent
        ):
            run_update_subagent_from_orchestrator(state)

        subagent_state = update_subagent.call_args[0][0]
        assert (
            subagent_state["target_tasks"][0].instruction
            == 'Update "lodash" in package.json to version "4.17.22".'
        )
        assert (
            subagent_state["previous_action_summaries_by_task"]["task-1"]
            == "Previous version bump failed manifest validation."
        )
        assert "supervisor_instruction" not in subagent_state

    def test_update_wrapper_surfaces_retry_diagnostics(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        task, snapshot = _committed_dispatch(task, dispatch_node="update_subagent")
        state = _initial_state(tmp_path, groups)
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["task_queue"] = {"task-1": task}
        state["active_target_task_ids"] = ["task-1"]
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}
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

        with patch(
            "remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent
        ):
            result = run_update_subagent_from_orchestrator(state)

        assert result["retry_diagnostics_by_task"]["task-1"] == diagnostics

    def test_workaround_wrapper_preserves_typed_attempt_result(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.WORKAROUND_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1")
        task.instruction = "Apply the source workaround."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-workaround",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            qa_policy=task.qa_policy,
            selected_version=None,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="workaround_subagent",
        )
        task = task.model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": snapshot.attempt_id,
            }
        )
        summary = AgentActionSummary(
            task_id="task-1",
            attempt_id=snapshot.attempt_id,
            task_revision=snapshot.task_revision,
            instruction_digest=snapshot.instruction_digest,
            status=AgentActionStatus.SURRENDER,
            summary="workaround bypassed",
        )
        worker_result = WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=task.task_id,
            task_revision=snapshot.task_revision,
            status=summary.status,
            action_summary=summary,
            instruction_digest=snapshot.instruction_digest,
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            }
        )
        with patch(
            "remediation_engine.orchestration.graph.run_workaround_subagent_node",
            return_value={
                "action_summaries": [summary],
                "action_summary": summary,
                "worker_results_by_attempt": {snapshot.attempt_id: worker_result},
                "errors": [],
            },
        ):
            result = run_workaround_subagent_from_orchestrator(state)

        returned_summary = result["action_summaries"][0]
        assert returned_summary == summary
        assert result["worker_results_by_attempt"][snapshot.attempt_id] == worker_result

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
            qa_policy=task.qa_policy,
            selected_version="4.17.21",
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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

    def test_dispatch_rejects_missing_qa_policy_provenance(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": "attempt-update",
                "qa_policy": None,
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-update",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=None,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        assert any(
            event.error_code == "MISSING_QA_POLICY_PROVENANCE"
            for event in result["consistency_events"]
        )

    def test_failed_worker_restores_attempt_workspace_snapshot(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-update"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-update",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="The dependency update failed validation.",
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch(
                "remediation_engine.orchestration.graph.run_update_subagent_node",
                return_value={"action_summaries": [summary], "errors": ["validation failed"]},
            ),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_update_subagent_from_orchestrator(state)

        sandbox.create_workspace_snapshot.assert_called_once_with("attempt-attempt-update")
        sandbox.restore_workspace_snapshot.assert_called_once_with("attempt-attempt-update")
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-update")
        assert "validation failed" in result["errors"]

    def test_failed_restore_retains_attempt_snapshot_for_teardown(self, tmp_path):
        state = _initial_state(tmp_path, [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox
        sandbox.restore_workspace_snapshot.side_effect = RuntimeError("archive missing")

        with patch(
            "remediation_engine.orchestration.graph.DockerSandbox",
            return_value=sandbox,
        ):
            errors = _finish_workspace_attempt_snapshot(
                state,
                "attempt-missing",
                restore=True,
            )

        sandbox.remove_workspace_snapshot.assert_not_called()
        assert errors == [
            "graph: could not restore workspace snapshot attempt-missing: archive missing"
        ]

    def test_successful_worker_keeps_snapshot_until_qa_accepts_candidate(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-handoff"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-handoff",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        success_summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="The dependency update passed worker validation.",
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch(
                "remediation_engine.orchestration.graph.run_update_subagent_node",
                return_value={"action_summaries": [success_summary], "errors": []},
            ),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            run_update_subagent_from_orchestrator(state)

        sandbox.create_workspace_snapshot.assert_called_once_with("attempt-attempt-handoff")
        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_not_called()

        evaluation = QAEvaluation(task_id=task.task_id, passed=True)
        with (
            patch(
                "remediation_engine.orchestration.graph.run_qa_critic_node",
                return_value={
                    "qa_evaluations": {task.task_id: evaluation},
                    "status": "qa_completed",
                    "errors": [],
                },
            ),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            run_qa_critic_from_orchestrator(state)

        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-handoff")

    def test_workspace_builder_success_routes_through_supervisor_to_teardown(self, tmp_path):
        """After workspace prep, portfolio routes to Supervisor, then teardown runs."""
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
        portfolio = MagicMock(return_value={"status": "portfolio_ready"})

        with (
            patch(
                "remediation_engine.orchestration.graph.run_workspace_builder_node",
                workspace_builder,
            ),
            patch("remediation_engine.orchestration.graph.run_portfolio_node", portfolio),
            patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor),
            patch("remediation_engine.orchestration.graph.run_teardown_node", teardown),
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert workspace_builder.call_count == 1
        assert portfolio.call_count == 1
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
        supervisor = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.graph.run_workspace_builder_node",
                workspace_builder,
            ),
            patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor),
            patch("remediation_engine.orchestration.graph.run_teardown_node", teardown),
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert workspace_builder.call_count == 1
        supervisor.assert_not_called()
        assert teardown.call_count == 1
        assert result["status"] == "completed"

    def test_supervisor_routes_to_update_subagent_then_back(self, tmp_path):
        """Supervisor routes to update_subagent once, then supervisor routes to teardown."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        gid = groups[0].group_id

        workspace_builder = MagicMock(
            return_value={
                "status": "workspace_ready",
                "workspace_volume": "vol123",
            }
        )
        portfolio = MagicMock(return_value={"status": "portfolio_ready"})

        call_count = {"n": 0}

        def supervisor_side_effect(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "status": "supervisor_routed",
                    "next_routing_step": "update_subagent",
                    "active_target_task_ids": [gid],
                    "group_strategies": {},
                    "retry_counts": {},
                    "constraints_ledger": [],
                    "feedback_by_group": {},
                    "supervisor_instructions": "bump it",
                }
            return {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "group_strategies": {},
                "retry_counts": {},
                "constraints_ledger": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }

        supervisor = MagicMock(side_effect=supervisor_side_effect)
        update_subagent = MagicMock(return_value={"errors": []})
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with (
            patch(
                "remediation_engine.orchestration.graph.run_workspace_builder_node",
                workspace_builder,
            ),
            patch("remediation_engine.orchestration.graph.run_portfolio_node", portfolio),
            patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor),
            patch(
                "remediation_engine.orchestration.graph.run_update_subagent_node", update_subagent
            ),
            patch("remediation_engine.orchestration.graph.run_teardown_node", teardown),
        ):
            graph = build_orchestrator_graph()
            graph.invoke(_initial_state(tmp_path, groups))

        assert supervisor.call_count == 2
        assert portfolio.call_count == 1
        assert teardown.call_count == 1

    def test_supervisor_routes_to_qa_critic_then_back(self, tmp_path):
        """Supervisor routes to qa_critic once, then to teardown."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]

        workspace_builder = MagicMock(
            return_value={
                "status": "workspace_ready",
                "workspace_volume": "vol123",
            }
        )

        portfolio = MagicMock(return_value={"status": "portfolio_ready"})
        call_count = {"n": 0}

        def supervisor_side_effect(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "status": "supervisor_routed",
                    "next_routing_step": "qa_critic",
                    "group_strategies": {},
                    "retry_counts": {},
                    "constraints_ledger": [],
                    "feedback_by_group": {},
                    "supervisor_instructions": "run qa",
                }
            return {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "group_strategies": {},
                "retry_counts": {},
                "constraints_ledger": [],
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }

        supervisor = MagicMock(side_effect=supervisor_side_effect)
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {},
                "eval_status": "all_passed",
                "qa_investigation_report": "# INVESTIGATIVE REPORT\n## Install Analysis",
                "baseline_scan_identifiers": ["CVE-2021-44228"],
                "post_remediation_scan_identifiers": ["CVE-2025-10001"],
                "new_vulnerability_identifiers": ["CVE-2025-10001"],
                "new_vulnerability_status": "detected",
                "status": "qa_completed",
                "errors": [],
            }
        )
        teardown = MagicMock(return_value={"status": "completed", "workspace_volume": None})

        with (
            patch(
                "remediation_engine.orchestration.graph.run_workspace_builder_node",
                workspace_builder,
            ),
            patch("remediation_engine.orchestration.graph.run_portfolio_node", portfolio),
            patch("remediation_engine.orchestration.graph.run_supervisor_node", supervisor),
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch("remediation_engine.orchestration.graph.run_teardown_node", teardown),
        ):
            graph = build_orchestrator_graph()
            result = graph.invoke(_initial_state(tmp_path, groups))

        assert supervisor.call_count == 2
        assert qa_critic.call_count == 1
        assert teardown.call_count == 1
        assert portfolio.call_count == 1
        assert result["qa_investigation_report"].startswith("# INVESTIGATIVE REPORT")
        assert result["new_vulnerability_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_status"] == "detected"

    def test_qa_wrapper_scopes_valid_groups_to_active_batch(self, tmp_path):
        groups = [
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
        ]
        tasks = []
        snapshots = {}
        for index, group in enumerate(groups, start=1):
            task, snapshot = _committed_dispatch(
                build_initial_remediation_task(group, f"task-{index}"),
                dispatch_node="qa_critic",
            )
            tasks.append(task)
            snapshots[snapshot.attempt_id] = snapshot
        state = _initial_state(tmp_path, groups)
        state["task_queue"] = {task.task_id: task for task in tasks}
        state["active_target_task_ids"] = [tasks[1].task_id]
        state["attempt_snapshots_by_id"] = snapshots
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {
                    tasks[1].task_id: QAEvaluation(task_id=tasks[1].task_id, passed=True)
                },
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

    def test_qa_wrapper_rejects_policyless_task(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"qa_policy": None}
        )
        task, snapshot = _committed_dispatch(task, dispatch_node="qa_critic")
        state = _initial_state(tmp_path, groups)
        state["task_queue"] = {"task-1": task}
        state["active_target_task_ids"] = ["task-1"]
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}

        qa_critic = MagicMock()
        with patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic):
            result = run_qa_critic_from_orchestrator(state)

        qa_critic.assert_not_called()
        assert result["eval_status"] == "state_inconsistent"
        assert any(
            event.error_code == "MISSING_QA_POLICY_PROVENANCE"
            for event in result["consistency_events"]
        )

    def test_qa_wrapper_scopes_one_active_task_to_one_parent_group(self, tmp_path):
        groups = [
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
            _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
        ]
        task = build_initial_remediation_task(groups[1], "task-2").model_copy(
            update={"status": TaskStatus.OPTIMISTICALLY_FIXED}
        )
        task, snapshot = _committed_dispatch(task, dispatch_node="qa_critic")
        state = _initial_state(tmp_path, groups)
        state["task_queue"] = {"task-2": task}
        state["active_target_task_ids"] = ["task-2"]
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}

        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: QAEvaluation(task_id=task.task_id, passed=True)},
                "eval_status": "all_passed",
                "qa_investigation_report": "# INVESTIGATIVE REPORT",
                "status": "qa_completed",
                "errors": [],
            }
        )

        with patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic):
            result = run_qa_critic_from_orchestrator(state)

        scoped_state = qa_critic.call_args[0][0]
        assert [group.group_id for group in scoped_state["valid_groups"]] == [groups[1].group_id]
        assert result["status"] == "qa_completed"

    def test_breaking_change_qa_retains_pre_update_snapshot_for_workaround(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-qa"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        evaluation = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The candidate breaks the test suite.",
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "some_failed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_not_called()
        assert result["workspace_rollback_anchors_by_task"] == {"task-1": "attempt-attempt-qa"}
        assert result["qa_results_by_attempt"]["attempt-qa"].evaluation.passed is False

    def test_inconclusive_qa_retains_candidate_workspace_for_rerun(self, tmp_path):
        """Evidence-only QA reruns must not restore the pre-worker baseline."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-qa-inconclusive"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa-inconclusive",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        evaluation = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Dependency evidence was unavailable.",
            evidence_inconclusive=True,
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "failures_detected",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-qa-inconclusive")
        assert result["qa_results_by_attempt"]["attempt-qa-inconclusive"].evaluation == evaluation

    def test_qa_failed_contract_error_retains_candidate_workspace(self, tmp_path):
        """Infrastructure QA failures retain the candidate for the same attempt."""
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-qa-contract"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa-contract",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        evaluation = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            contract_error=True,
            contract_error_reason="QA execution did not complete.",
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Rerun QA.",
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "failures_detected",
                "status": "qa_failed",
                "errors": ["QA infrastructure failure"],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-qa-contract")
        assert result["qa_results_by_attempt"]["attempt-qa-contract"].evaluation.contract_error

    def test_security_failure_restores_and_removes_attempt_workspace_snapshot(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-security"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-security",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        evaluation = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="The candidate still contains the target vulnerability.",
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "some_failed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_called_once_with("attempt-attempt-security")
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-security")
        assert result["qa_results_by_attempt"]["attempt-security"].evaluation.passed is False

    def test_workaround_regression_restores_to_original_pre_update_workspace(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": "attempt-workaround",
                "parent_task_id": "task-parent",
                "strategy": RoutingStrategy.CODE_WORKAROUND,
                "qa_policy": QAPolicy.MITIGATION_CODE_WORKAROUND,
                "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
                "selected_version": None,
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-workaround",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            qa_policy=task.qa_policy,
            selected_version=None,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="workaround_subagent",
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
                "workspace_rollback_anchors_by_task": {
                    "task-parent": "attempt-parent",
                },
            }
        )
        evaluation = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The workaround still breaks a regression test.",
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "some_failed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        assert [call.args for call in sandbox.restore_workspace_snapshot.call_args_list] == [
            ("attempt-attempt-workaround",),
            ("attempt-parent",),
        ]
        assert [call.args for call in sandbox.remove_workspace_snapshot.call_args_list] == [
            ("attempt-attempt-workaround",),
            ("attempt-parent",),
        ]
        assert result["qa_results_by_attempt"]["attempt-workaround"].evaluation.passed is False

    def test_passed_qa_keeps_candidate_and_only_removes_attempt_snapshot(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-pass"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-pass",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
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
        evaluation = QAEvaluation(task_id=task.task_id, passed=True)
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: evaluation},
                "eval_status": "all_passed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-pass")
        qa_result = result["qa_results_by_attempt"]["attempt-pass"]
        assert qa_result.qa_policy == task.qa_policy
        assert qa_result.qa_policy_source == "attempt_snapshot"

    def test_regression_retries_keep_first_pre_task_baseline(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        first_task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 1, "current_attempt_id": "attempt-first"}
        )
        second_task = first_task.model_copy(
            update={"task_revision": 2, "current_attempt_id": "attempt-second"}
        )

        def snapshot(task, attempt_id, revision):
            return TaskAttemptSnapshot(
                attempt_id=attempt_id,
                task_id=task.task_id,
                state_revision=revision,
                task_revision=revision,
                strategy_stage=task.strategy_stage,
                qa_policy=task.qa_policy,
                selected_version=task.selected_version,
                instruction=task.instruction,
                instruction_digest=instruction_digest(task.instruction),
                dispatch_node="update_subagent",
            )

        first_snapshot = snapshot(first_task, "attempt-first", 1)
        second_snapshot = snapshot(second_task, "attempt-second", 2)
        evaluation = QAEvaluation(
            task_id=first_task.task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The candidate breaks the test suite.",
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {first_task.task_id: evaluation},
                "eval_status": "some_failed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {"task-1": first_task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {"attempt-first": first_snapshot},
            }
        )
        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch("remediation_engine.orchestration.graph.DockerSandbox", return_value=sandbox),
        ):
            first_result = run_qa_critic_from_orchestrator(state)
        retry_state = dict(state)
        retry_state.update(
            {
                "task_queue": {"task-1": second_task},
                "attempt_snapshots_by_id": {
                    "attempt-first": first_snapshot,
                    "attempt-second": second_snapshot,
                },
                "workspace_rollback_anchors_by_task": first_result[
                    "workspace_rollback_anchors_by_task"
                ],
            }
        )
        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch("remediation_engine.orchestration.graph.DockerSandbox", return_value=sandbox),
        ):
            second_result = run_qa_critic_from_orchestrator(retry_state)
        assert first_result["workspace_rollback_anchors_by_task"] == {
            "task-1": "attempt-attempt-first"
        }
        assert second_result["workspace_rollback_anchors_by_task"] == {
            "task-1": "attempt-attempt-first"
        }
        sandbox.restore_workspace_snapshot.assert_not_called()
        sandbox.remove_workspace_snapshot.assert_not_called()

    def test_failed_retry_worker_restores_baseline_not_rejected_candidate(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 2, "current_attempt_id": "attempt-second"}
        )
        first_snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-first",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        second_snapshot = first_snapshot.model_copy(
            update={"attempt_id": "attempt-second", "state_revision": 2, "task_revision": 2}
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {
                    "attempt-first": first_snapshot,
                    "attempt-second": second_snapshot,
                },
                "workspace_rollback_anchors_by_task": {"task-1": "attempt-attempt-first"},
            }
        )
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="The retry failed worker validation.",
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox
        with (
            patch(
                "remediation_engine.orchestration.graph.run_update_subagent_node",
                return_value={"action_summaries": [summary], "errors": ["validation failed"]},
            ),
            patch("remediation_engine.orchestration.graph.DockerSandbox", return_value=sandbox),
        ):
            run_update_subagent_from_orchestrator(state)
        sandbox.create_workspace_snapshot.assert_called_once_with("attempt-attempt-second")
        sandbox.restore_workspace_snapshot.assert_called_once_with("attempt-attempt-first")
        sandbox.remove_workspace_snapshot.assert_called_once_with("attempt-attempt-second")

    def test_successful_retry_discards_baseline_anchor(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-1").model_copy(
            update={"task_revision": 2, "current_attempt_id": "attempt-pass"}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-pass",
            task_id="task-1",
            state_revision=2,
            task_revision=2,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {"task-1": task},
                "active_target_task_ids": ["task-1"],
                "attempt_snapshots_by_id": {"attempt-pass": snapshot},
                "workspace_rollback_anchors_by_task": {"task-1": "attempt-attempt-first"},
            }
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: QAEvaluation(task_id=task.task_id, passed=True)},
                "eval_status": "all_passed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox
        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch("remediation_engine.orchestration.graph.DockerSandbox", return_value=sandbox),
        ):
            result = run_qa_critic_from_orchestrator(state)
        assert result["workspace_rollback_anchors_by_task"] == {}
        assert [call.args for call in sandbox.remove_workspace_snapshot.call_args_list] == [
            ("attempt-attempt-pass",),
            ("attempt-attempt-first",),
        ]


class TestPhase5Exports:
    def test_phase5_exports_are_available(self):
        assert callable(build_orchestrator_graph)
        assert orchestrator_engine is not None
        assert callable(run_orchestrator)

    def test_graph_compiles_without_error(self):
        graph = build_orchestrator_graph()
        assert graph is not None

    def test_successful_workaround_promotes_code_and_dependency_candidate(self, tmp_path):
        groups = [_group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND))]
        task = build_initial_remediation_task(groups[0], "task-workaround").model_copy(
            update={
                "task_revision": 2,
                "current_attempt_id": "attempt-workaround-pass",
                "parent_task_id": "task-parent",
                "strategy": RoutingStrategy.CODE_WORKAROUND,
                "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
                "selected_version": None,
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-workaround-pass",
            task_id=task.task_id,
            state_revision=2,
            task_revision=2,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            qa_policy=task.qa_policy,
            selected_version=None,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="workaround_subagent",
        )
        state = _initial_state(tmp_path, groups)
        state.update(
            {
                "workspace_volume": "agent_workspace_deadbeef",
                "task_queue": {task.task_id: task},
                "active_target_task_ids": [task.task_id],
                "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
                "workspace_rollback_anchors_by_task": {
                    "task-parent": "attempt-parent-baseline",
                    "task-sibling": "attempt-sibling-baseline",
                },
            }
        )
        qa_critic = MagicMock(
            return_value={
                "qa_evaluations": {task.task_id: QAEvaluation(task_id=task.task_id, passed=True)},
                "eval_status": "all_passed",
                "status": "qa_completed",
                "errors": [],
            }
        )
        sandbox = MagicMock()
        sandbox.__enter__.return_value = sandbox

        with (
            patch("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic),
            patch(
                "remediation_engine.orchestration.graph.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_qa_critic_from_orchestrator(state)

        sandbox.restore_workspace_snapshot.assert_not_called()
        assert [call.args for call in sandbox.remove_workspace_snapshot.call_args_list] == [
            ("attempt-attempt-workaround-pass",),
            ("attempt-parent-baseline",),
            ("attempt-sibling-baseline",),
        ]
        assert result["workspace_rollback_anchors_by_task"] == {}


class _DeltaSandbox:
    """In-memory archive adapter for delta-isolation workspace tests."""

    def __init__(
        self,
        workspace: dict[str, str],
        archives: dict[str, dict[str, str]],
        *,
        fail_candidate_restore: bool = False,
        fail_candidate_remove: bool = False,
        fail_snapshot_create: bool = False,
        fail_preworker_restore_once: bool = False,
    ) -> None:
        self.workspace = workspace
        self.archives = archives
        self.fail_candidate_restore = fail_candidate_restore
        self.fail_candidate_remove = fail_candidate_remove
        self.fail_preworker_restore_once = fail_preworker_restore_once
        self.preworker_restore_failed = False
        self.fail_snapshot_create = fail_snapshot_create
        self.events: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def create_workspace_snapshot(self, snapshot_id: str) -> None:
        self.events.append(("create", snapshot_id, dict(self.workspace)))
        if self.fail_snapshot_create and snapshot_id.startswith("delta-candidate-"):
            raise RuntimeError("candidate archive unavailable")
        self.archives[snapshot_id] = dict(self.workspace)

    def restore_workspace_snapshot(self, snapshot_id: str) -> None:
        self.events.append(("restore_attempt", snapshot_id))
        if self.fail_candidate_restore and snapshot_id.startswith("delta-candidate-"):
            raise RuntimeError("candidate archive cannot be restored")
        if (
            self.fail_preworker_restore_once
            and snapshot_id.startswith("batch-")
            and not self.preworker_restore_failed
        ):
            self.preworker_restore_failed = True
            raise RuntimeError("pre-worker archive temporarily unavailable")
        if snapshot_id not in self.archives:
            raise RuntimeError(f"missing archive {snapshot_id}")
        self.workspace.clear()
        self.workspace.update(self.archives[snapshot_id])
        self.events.append(("restored", snapshot_id, dict(self.workspace)))

    def remove_workspace_snapshot(self, snapshot_id: str) -> None:
        self.events.append(("remove_attempt", snapshot_id))
        if self.fail_candidate_remove and snapshot_id.startswith("delta-candidate-"):
            raise RuntimeError("candidate archive cannot be removed")
        self.archives.pop(snapshot_id, None)
        self.events.append(("removed", snapshot_id))


def _delta_test_setup(tmp_path, *, batch_id="batch-delta", attempt_prefix="attempt-delta"):
    groups = [
        _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
        _group(IssueType.SCA, fix_plan=_fix_plan(FixPlanStatus.VERSION_FOUND)),
    ]
    cluster_id = "cluster-delta"
    portfolio_plan_id = "portfolio-delta"
    tasks = []
    for suffix, group in zip(("a", "b"), groups, strict=True):
        task_id = f"task-{suffix}"
        task = build_initial_remediation_task(group, task_id).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": f"{attempt_prefix}-{task_id}",
                "selected_version": "2.0.0",
                "target_package_name": f"pkg-{suffix}",
                "target_dependency_type": "dependencies",
                "instruction": f"Update pkg-{suffix} to 2.0.0.",
            }
        )
        tasks.append(task)
    action = MultiPackageAction(
        cluster_id=cluster_id,
        dispatch_batch_id=batch_id,
        selected_strategy="version_bump",
        package_mutations=[
            PackageMutation(
                task_id=task.task_id,
                package_name=task.target_package_name,
                target_version="2.0.0",
                dependency_type="dependencies",
            )
            for task in tasks
        ],
        rationale="test atomic delta isolation",
    )
    action_digest = instruction_digest(action.model_dump_json())
    snapshots = {}
    for task in tasks:
        snapshot = TaskAttemptSnapshot(
            attempt_id=task.current_attempt_id,
            task_id=task.task_id,
            state_revision=1,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=batch_id,
            action_digest=action_digest,
            portfolio_plan_id=portfolio_plan_id,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            target_package_name=task.target_package_name,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        snapshots[snapshot.attempt_id] = snapshot
    decisions = [
        SimpleNamespace(
            task_id=task.task_id,
            target_package_name=task.target_package_name,
            installed_version="1.0.0",
        )
        for task in tasks
    ]
    state = _initial_state(tmp_path, groups)
    state.update(
        {
            "workspace_volume": "workspace-delta",
            "task_queue": {task.task_id: task for task in tasks},
            "active_target_task_ids": ["task-b", "task-a"],
            "active_cluster_id": cluster_id,
            "active_dispatch_batch_id": batch_id,
            "active_multi_package_action": action,
            "attempt_snapshots_by_id": snapshots,
            "portfolio_plan": SimpleNamespace(
                portfolio_plan_id=portfolio_plan_id,
                solver_plan=SimpleNamespace(
                    selected_plan=SimpleNamespace(task_decisions=decisions)
                ),
            ),
            "delta_isolation_by_cluster": {},
        }
    )
    return state, tasks, action, groups, snapshots


def _failed_delta_qa_result(tasks):
    evaluations = {}
    errors_by_task = {}
    for task in tasks:
        suffix = task.task_id[-1]
        package_name = task.target_package_name
        evaluations[task.task_id] = QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback=f"{suffix} retry feedback",
            failure_evidence=QAFailureEvidence(
                raw_excerpt=f"{package_name} raw excerpt",
                exact_diagnostics=[f"{suffix} exact diagnostic 1", f"{suffix} exact diagnostic 2"],
                failed_tests=[f"{suffix} failed test"],
            ),
            deterministic_gates=QADeterministicGates(
                status="fail",
                install_passed=True,
                scanner_execution_status="success",
                tests_passed=False,
                diagnostics=[f"{suffix} gate diagnostic"],
            ),
            test_attribution=QATestAttribution(
                verdict="inconclusive",
                failed_tests=[f"{suffix} attributed failed test"],
            ),
        )
        errors_by_task[task.task_id] = [f"{suffix} QA error"]
    return {
        "qa_evaluations": evaluations,
        "qa_errors_by_task": errors_by_task,
        "qa_investigation_report": "excluded investigation report",
        "eval_status": "some_failed",
        "status": "qa_completed",
        "errors": [],
    }


def _apply_fake_delta_action(sandbox, action, touched_files):
    package_names = []
    for mutation in action.package_mutations:
        sandbox.workspace[mutation.package_name] = mutation.target_version
        package_names.append(mutation.package_name)
        touched_files.add("package.json")
    sandbox.events.append(("applied", tuple(package_names), dict(sandbox.workspace)))
    return True, None


def test_delta_isolation_canaries_probe_baseline_singletons_and_restore_full_candidate(
    tmp_path, monkeypatch
):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives)
    observations = []
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)

    def qa_probe(active_sandbox, subset_action):
        package_names = tuple(mutation.package_name for mutation in subset_action.package_mutations)
        observations.append((package_names, dict(active_sandbox.workspace)))
        return "FAIL" if package_names == ("pkg-b",) else "PASS"

    result = _graph_wrappers.run_delta_isolation_canaries(
        state,
        tasks,
        action,
        qa_probe,
        ranked_tasks=tasks,
        max_canary_probes=2,
    )

    candidate_id = (
        "delta-candidate-" + hashlib.sha256(b"batch-delta:task-a,task-b").hexdigest()[:24]
    )
    assert result["status"] == "IDENTIFIED"
    assert result["responsible_task_ids"] == ["task-b"]
    assert result["candidate_restored"] is True
    assert result["executions"] == 2
    assert observations == [
        (("pkg-a",), {"pkg-a": "2.0.0", "pkg-b": "1.0.0"}),
        (("pkg-b",), {"pkg-a": "1.0.0", "pkg-b": "2.0.0"}),
    ]
    candidate_restores = [
        event for event in sandbox.events if event[0] == "restored" and event[1] == candidate_id
    ]
    assert len(candidate_restores) == 3
    assert all(event[2] == candidate for event in candidate_restores)
    assert sandbox.workspace == candidate
    assert archives["batch-delta"] == baseline
    assert candidate_id not in archives
    assert sandbox.events.index(("removed", candidate_id)) > max(
        index
        for index, event in enumerate(sandbox.events)
        if event[0] == "restored" and event[1] == candidate_id
    )


def test_delta_isolation_candidate_restore_failure_rolls_back_original_qa_failure(
    tmp_path, monkeypatch
):
    state, tasks, _action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives, fail_candidate_restore=True)
    original_qa = _failed_delta_qa_result(tasks)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        lambda _state: original_qa,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=1,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        lambda _sandbox: SimpleNamespace(ok=False),
    )

    result = run_qa_critic_from_orchestrator(state)

    isolation = result["delta_isolation_by_cluster"]["cluster-delta"]
    assert isolation["status"] == "INCONCLUSIVE"
    assert isolation["candidate_restored"] is False
    assert isolation["responsible_task_ids"] == []
    assert isolation["executions"] == 1
    assert (
        result["qa_results_by_attempt"]["attempt-delta-task-a"].evaluation.evidence_inconclusive
        is False
    )
    assert (
        result["qa_results_by_attempt"]["attempt-delta-task-a"].evaluation.retry_feedback
        == "a retry feedback"
    )
    assert sandbox.workspace == baseline
    assert "batch-delta" not in archives
    assert any(snapshot_id.startswith("delta-candidate-") for snapshot_id in archives)
    assert not any(
        event[0] == "removed" and event[1].startswith("delta-candidate-")
        for event in sandbox.events
    )


def test_delta_isolation_archive_removal_failure_clears_attribution(tmp_path, monkeypatch):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives, fail_candidate_remove=True)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)

    result = _graph_wrappers.run_delta_isolation_canaries(
        state,
        tasks,
        action,
        lambda _sandbox, _action: "FAIL",
        ranked_tasks=tasks,
        max_canary_probes=1,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["candidate_restored"] is True
    assert result["responsible_task_ids"] == []
    assert result["executions"] == 1
    assert sandbox.workspace == candidate
    assert "batch-delta" in archives
    assert any(snapshot_id.startswith("delta-candidate-") for snapshot_id in archives)


def test_delta_isolation_snapshot_creation_failure_never_probes(tmp_path, monkeypatch):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    archives = {"batch-delta": {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    sandbox = _DeltaSandbox(candidate.copy(), archives, fail_snapshot_create=True)
    qa_calls = []
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)

    result = _graph_wrappers.run_delta_isolation_canaries(
        state,
        tasks,
        action,
        lambda *_args: qa_calls.append(True) or "FAIL",
        ranked_tasks=tasks,
        max_canary_probes=2,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["candidate_restored"] is True
    assert result["executions"] == 0
    assert qa_calls == []
    assert sandbox.workspace == candidate
    assert archives == {"batch-delta": {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}}


def test_delta_isolation_error_aggregation_settings_and_dispatch_provenance(tmp_path, monkeypatch):
    state, tasks, action, groups, _snapshots = _delta_test_setup(tmp_path)
    result = _failed_delta_qa_result(tasks)
    result.update(
        {
            "errors": "top-level error",
            "diagnostics": ("top diagnostic", None, " ", 99, "second diagnostic"),
            "qa_errors_by_task": {
                "task-b": ("b QA error", 12, "b second QA error"),
                "task-a": ["a QA error", None, "a second QA error"],
            },
            "qa_investigation_report": "do-not-include this report transcript",
        }
    )
    expected_entries = [
        "top-level error",
        "top diagnostic",
        "second diagnostic",
        "a QA error",
        "a second QA error",
        "b QA error",
        "b second QA error",
        "pkg-a raw excerpt",
        "a exact diagnostic 1",
        "a exact diagnostic 2",
        "a failed test",
        "a gate diagnostic",
        "a attributed failed test",
        "a retry feedback",
        "pkg-b raw excerpt",
        "b exact diagnostic 1",
        "b exact diagnostic 2",
        "b failed test",
        "b gate diagnostic",
        "b attributed failed test",
        "b retry feedback",
    ]
    expected_error_text = "\n".join(expected_entries)
    ranked_task_ids = []
    captured = {}

    def ranker(active_tasks, active_action, error_logs, groups_by_id, installed_by_task):
        captured["error_logs"] = error_logs
        captured["groups"] = groups_by_id
        captured["installed_versions"] = installed_by_task
        return [(tasks[1], 13), (tasks[0], 3)]

    def canaries(
        active_state,
        active_tasks,
        active_action,
        qa_probe,
        *,
        ranked_tasks,
        max_canary_probes,
    ):
        captured["ranked_task_ids"] = [task.task_id for task in ranked_tasks]
        captured["max_canary_probes"] = max_canary_probes
        ranked_task_ids.extend(captured["ranked_task_ids"])
        return {
            "status": "INCONCLUSIVE",
            "responsible_task_ids": [],
            "tested_subsets": [["task-b"]],
            "executions": 1,
            "diagnostic": "budget exhausted",
            "candidate_restored": True,
        }

    monkeypatch.setattr(_graph_wrappers, "rank_suspect_tasks_by_suspicion", ranker)
    monkeypatch.setattr(_graph_wrappers, "run_delta_isolation_canaries", canaries)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(max_delta_canary_probes=1),
    )

    isolation = _graph_wrappers._maybe_run_delta_isolation(state, tasks, result)

    assert captured["error_logs"] == expected_error_text
    assert "do-not-include" not in captured["error_logs"]
    assert captured["groups"] == {group.group_id: group for group in groups}
    assert captured["installed_versions"] == {
        "task-a": "1.0.0",
        "task-b": "1.0.0",
    }
    assert ranked_task_ids == ["task-b", "task-a"]
    assert captured["max_canary_probes"] == 1
    assert isolation["status"] == "INCONCLUSIVE"
    assert isolation["candidate_restored"] is True
    assert isolation["cluster_id"] == "cluster-delta"
    assert isolation["dispatch_batch_id"] == "batch-delta"
    assert isolation["action_digest"] == instruction_digest(action.model_dump_json())
    assert isolation["source_portfolio_plan_id"] == "portfolio-delta"
    assert isolation["attempt_ids_by_task"] == {
        "task-a": "attempt-delta-task-a",
        "task-b": "attempt-delta-task-b",
    }


def test_delta_isolation_solver_baselines_require_one_matching_current_portfolio_plan(
    tmp_path,
):
    state, tasks, action, _groups, snapshots = _delta_test_setup(tmp_path)
    mutations_by_task = _graph_wrappers._action_mutations_by_task(action, tasks)
    provenance = _graph_wrappers._delta_isolation_provenance(state, tasks, action)

    state["portfolio_plan"].solver_plan.selected_plan.task_decisions[0].installed_version = None
    state["portfolio_plan"].solver_plan.selected_plan.task_decisions[
        1
    ].installed_version = "unknown"
    matching_versions = _graph_wrappers._solver_installed_versions_by_task(
        state, tasks, mutations_by_task, provenance
    )
    assert matching_versions == {"task-a": None, "task-b": "unknown"}

    stale_snapshot = snapshots[tasks[0].current_attempt_id].model_copy(
        update={"portfolio_plan_id": "portfolio-stale"}
    )
    state["attempt_snapshots_by_id"][stale_snapshot.attempt_id] = stale_snapshot
    stale_provenance = _graph_wrappers._delta_isolation_provenance(state, tasks, action)
    assert (
        _graph_wrappers._solver_installed_versions_by_task(
            state, tasks, mutations_by_task, stale_provenance
        )
        == {}
    )


@pytest.mark.parametrize("malformation", ["missing", "duplicate"])
def test_delta_isolation_action_task_id_mismatch_is_rejected_before_any_probe(
    tmp_path, monkeypatch, malformation
):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    if malformation == "missing":
        mutations = action.package_mutations[:1]
    else:
        mutations = [
            action.package_mutations[0],
            PackageMutation(
                task_id=tasks[0].task_id,
                package_name="pkg-extra",
                target_version="2.0.0",
                dependency_type="dependencies",
            ),
        ]
    malformed_action = MultiPackageAction(
        cluster_id=action.cluster_id,
        dispatch_batch_id=action.dispatch_batch_id,
        selected_strategy=action.selected_strategy,
        package_mutations=mutations,
        rationale=action.rationale,
    )
    state["active_multi_package_action"] = malformed_action
    ranker = MagicMock()
    canaries = MagicMock()
    apply_action = MagicMock()
    sandbox = MagicMock()
    monkeypatch.setattr(_graph_wrappers, "rank_suspect_tasks_by_suspicion", ranker)
    monkeypatch.setattr(_graph_wrappers, "run_delta_isolation_canaries", canaries)
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", apply_action)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )

    isolation = _graph_wrappers._maybe_run_delta_isolation(
        state, tasks, _failed_delta_qa_result(tasks)
    )

    assert isolation["status"] == "INCONCLUSIVE"
    assert isolation["candidate_restored"] is True
    assert isolation["executions"] == 0
    assert isolation["tested_subsets"] == []
    ranker.assert_not_called()
    canaries.assert_not_called()
    apply_action.assert_not_called()
    sandbox.create_workspace_snapshot.assert_not_called()


def test_qa_delta_isolation_attribution_projects_provenance_and_escalates_only_after_restore(
    tmp_path, monkeypatch
):
    state, tasks, _action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives)
    qa_result = _failed_delta_qa_result(tasks)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        lambda _state: qa_result,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=2,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )

    def run_unit_tests(active_sandbox):
        blame_b = (
            active_sandbox.workspace["pkg-b"] == "2.0.0"
            and active_sandbox.workspace["pkg-a"] == "1.0.0"
        )
        return SimpleNamespace(ok=not blame_b)

    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        run_unit_tests,
    )

    result = run_qa_critic_from_orchestrator(state)

    isolation = result["delta_isolation_by_cluster"]["cluster-delta"]
    assert isolation["status"] == "IDENTIFIED"
    assert isolation["responsible_task_ids"] == ["task-b"]
    assert isolation["candidate_restored"] is True
    assert isolation["executions"] == 2
    assert result["portfolio_escalation"] == {
        "reason": "DELTA_ISOLATION_ATTRIBUTION",
        "forced_singleton_task_ids": ["task-b"],
        "triggering_attempt_id": "attempt-delta-task-b",
        "source_portfolio_plan_id": "portfolio-delta",
    }
    assert result["portfolio_dirty"] is True
    assert result["portfolio_plan"] is None
    assert all(
        not result["qa_results_by_attempt"][
            f"attempt-delta-{task.task_id}"
        ].evaluation.evidence_inconclusive
        for task in tasks
    )
    assert sandbox.workspace == baseline
    assert "batch-delta" not in archives


def test_same_dispatch_delta_isolation_qa_rerun_reuses_isolation_and_new_dispatch_gets_budget(
    tmp_path, monkeypatch
):
    state, tasks, _action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives)
    apply_calls = []
    qa_critic_calls = []

    def qa_critic(active_state):
        qa_critic_calls.append(True)
        return _failed_delta_qa_result(
            [
                active_state["task_queue"][task_id]
                for task_id in active_state["active_target_task_ids"]
            ]
        )

    def apply_action(active_sandbox, action, touched_files):
        apply_calls.append(tuple(mutation.task_id for mutation in action.package_mutations))
        return _apply_fake_delta_action(active_sandbox, action, touched_files)

    monkeypatch.setattr("remediation_engine.orchestration.graph.run_qa_critic_node", qa_critic)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=1,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", apply_action)
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        lambda _sandbox: SimpleNamespace(ok=True),
    )

    first = run_qa_critic_from_orchestrator(state)
    rerun_state = {
        **state,
        "delta_isolation_by_cluster": first["delta_isolation_by_cluster"],
    }
    second = run_qa_critic_from_orchestrator(rerun_state)

    assert first["delta_isolation_by_cluster"]["cluster-delta"]["executions"] == 1
    assert (
        second["delta_isolation_by_cluster"]["cluster-delta"]
        == first["delta_isolation_by_cluster"]["cluster-delta"]
    )
    assert len(apply_calls) == 1
    assert (
        len(
            [
                event
                for event in sandbox.events
                if event[0] == "create" and event[1].startswith("delta-candidate-")
            ]
        )
        == 1
    )

    fresh_state, fresh_tasks, _fresh_action, _fresh_groups, _fresh_snapshots = _delta_test_setup(
        tmp_path,
        batch_id="batch-fresh",
        attempt_prefix="attempt-fresh",
    )
    fresh_state["delta_isolation_by_cluster"] = first["delta_isolation_by_cluster"]
    archives["batch-fresh"] = baseline.copy()
    sandbox.workspace.clear()
    sandbox.workspace.update(candidate)
    fresh = run_qa_critic_from_orchestrator(fresh_state)

    assert fresh["delta_isolation_by_cluster"]["cluster-delta"]["executions"] == 1
    assert len(apply_calls) == 2
    assert len(qa_critic_calls) == 3
    assert (
        len(
            [
                event
                for event in sandbox.events
                if event[0] == "create" and event[1].startswith("delta-candidate-")
            ]
        )
        == 2
    )
    assert fresh["delta_isolation_by_cluster"]["cluster-delta"]["attempt_ids_by_task"] == {
        "task-a": "attempt-fresh-task-a",
        "task-b": "attempt-fresh-task-b",
    }


def test_delta_isolation_preworker_restore_failure_consumes_slot_and_continues(
    tmp_path, monkeypatch
):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(
        candidate.copy(),
        archives,
        fail_preworker_restore_once=True,
    )
    qa_subsets = []
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)

    def qa_probe(active_sandbox, subset_action):
        task_ids = tuple(mutation.task_id for mutation in subset_action.package_mutations)
        qa_subsets.append((task_ids, dict(active_sandbox.workspace)))
        return "FAIL"

    result = _graph_wrappers.run_delta_isolation_canaries(
        state,
        tasks,
        action,
        qa_probe,
        ranked_tasks=tasks,
        max_canary_probes=2,
    )

    assert result["status"] == "IDENTIFIED"
    assert result["responsible_task_ids"] == ["task-b"]
    assert result["tested_subsets"] == [["task-a"], ["task-b"]]
    assert result["executions"] == 2
    assert qa_subsets == [(("task-b",), {"pkg-a": "1.0.0", "pkg-b": "2.0.0"})]
    assert sandbox.workspace == candidate
    assert archives["batch-delta"] == baseline
    assert not any(snapshot_id.startswith("delta-candidate-") for snapshot_id in archives)


def test_delta_isolation_probe_exception_consumes_slot_before_later_failure(tmp_path, monkeypatch):
    state, tasks, action, _groups, _snapshots = _delta_test_setup(tmp_path)
    baseline = {"pkg-a": "1.0.0", "pkg-b": "1.0.0"}
    candidate = {"pkg-a": "2.0.0", "pkg-b": "2.0.0"}
    archives = {"batch-delta": baseline.copy()}
    sandbox = _DeltaSandbox(candidate.copy(), archives)
    qa_subsets = []
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", _apply_fake_delta_action)

    def qa_probe(active_sandbox, subset_action):
        task_ids = tuple(mutation.task_id for mutation in subset_action.package_mutations)
        qa_subsets.append(task_ids)
        if task_ids == ("task-a",):
            raise RuntimeError("unit-test runner failed")
        return "FAIL"

    result = _graph_wrappers.run_delta_isolation_canaries(
        state,
        tasks,
        action,
        qa_probe,
        ranked_tasks=tasks,
        max_canary_probes=2,
    )

    assert result["status"] == "IDENTIFIED"
    assert result["responsible_task_ids"] == ["task-b"]
    assert result["tested_subsets"] == [["task-a"], ["task-b"]]
    assert result["executions"] == 2
    assert qa_subsets == [("task-a",), ("task-b",)]
    assert sandbox.workspace == candidate
    assert archives["batch-delta"] == baseline


@pytest.fixture
def socket_stack_delta_case(tmp_path):
    """Build one atomic three-package case from the canonical baseline issues."""
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "juice_shop"
        / "fixtures"
        / "baseline_issues.jsonl"
    )
    target_packages = {"socket.io", "engine.io", "socket.io-parser"}
    baseline_issues = []
    for line in fixture_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        issue = VulnerabilityIssue.model_validate_json(line)
        if issue.package_name in target_packages:
            baseline_issues.append(issue)
    groups = group_issues(baseline_issues)
    groups_by_package = {group.vulnerable_component: group for group in groups}
    assert set(groups_by_package) == target_packages
    assert {
        package_name: len(group.issues) for package_name, group in groups_by_package.items()
    } == {"socket.io": 1, "engine.io": 3, "socket.io-parser": 4}

    package_plan = [
        ("task-a", "socket.io", "3.1.2", "4.8.1"),
        ("task-b", "engine.io", "4.1.2", "6.6.7"),
        ("task-c", "socket.io-parser", "4.0.5", "4.2.6"),
    ]
    cluster_id = "cluster-socket-stack"
    dispatch_batch_id = "batch-socket-stack"
    portfolio_plan_id = "portfolio-socket-stack"
    tasks = []
    for task_id, package_name, _installed_version, target_version in package_plan:
        task = build_initial_remediation_task(
            groups_by_package[package_name],
            task_id,
        ).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": f"attempt-{task_id}",
                "portfolio_plan_id": portfolio_plan_id,
                "strategy": RoutingStrategy.VERSION_BUMP,
                "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                "no_fix_stage": None,
                "qa_policy": QAPolicy.VERSION_BUMP,
                "status": TaskStatus.OPTIMISTICALLY_FIXED,
                "retry_count": 0,
                "selected_version": target_version,
                "allowed_target_versions": [target_version],
                "allowed_dependency_types": ["overrides"],
                "target_package_name": package_name,
                "target_dependency_type": "overrides",
                "instruction": (
                    f"Apply the committed package override for {package_name} "
                    f"at exact version {target_version}."
                ),
            }
        )
        tasks.append(task)

    action = MultiPackageAction(
        cluster_id=cluster_id,
        dispatch_batch_id=dispatch_batch_id,
        selected_strategy="package_override",
        package_mutations=[
            PackageMutation(
                task_id=task.task_id,
                package_name=task.target_package_name,
                target_version=task.selected_version,
                dependency_type="overrides",
            )
            for task in tasks
        ],
        rationale="Exercise bounded delta isolation on the Socket.IO dependency stack.",
    )
    action_digest = instruction_digest(action.model_dump_json())
    snapshots = {}
    worker_results = {}
    for task in tasks:
        snapshot = TaskAttemptSnapshot(
            attempt_id=task.current_attempt_id,
            task_id=task.task_id,
            state_revision=1,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=dispatch_batch_id,
            action_digest=action_digest,
            portfolio_plan_id=portfolio_plan_id,
            qa_policy=task.qa_policy,
            strategy_stage=task.strategy_stage,
            selected_version=task.selected_version,
            target_package_name=task.target_package_name,
            target_dependency_type=task.target_dependency_type,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        snapshots[snapshot.attempt_id] = snapshot
        worker_results[snapshot.attempt_id] = WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=task.task_id,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=dispatch_batch_id,
            action_digest=action_digest,
            status=AgentActionStatus.SUCCESS,
            execution_diagnostics=WorkerExecutionDiagnostics(validation_passed=True),
            instruction_digest=snapshot.instruction_digest,
        )

    solver_decisions = [
        SimpleNamespace(
            task_id=task_id,
            target_package_name=package_name,
            installed_version=installed_version,
        )
        for task_id, package_name, installed_version, _target_version in package_plan
    ]
    state = _initial_state(tmp_path, groups)
    state.update(
        {
            "workspace_volume": "workspace-socket-stack",
            "task_queue": {task.task_id: task for task in tasks},
            "active_target_task_ids": [task.task_id for task in tasks],
            "active_cluster_id": cluster_id,
            "active_dispatch_batch_id": dispatch_batch_id,
            "active_multi_package_action": action,
            "attempt_snapshots_by_id": snapshots,
            "worker_results_by_attempt": worker_results,
            "portfolio_plan": SimpleNamespace(
                portfolio_plan_id=portfolio_plan_id,
                solver_plan=SimpleNamespace(
                    selected_plan=SimpleNamespace(task_decisions=solver_decisions)
                ),
            ),
            "delta_isolation_by_cluster": {},
        }
    )
    baseline = {
        package_name: installed_version
        for _task_id, package_name, installed_version, _target_version in package_plan
    }
    candidate = {
        package_name: target_version
        for _task_id, package_name, _installed_version, target_version in package_plan
    }
    return SimpleNamespace(
        state=state,
        groups=groups,
        tasks=tasks,
        action=action,
        snapshots=snapshots,
        baseline=baseline,
        candidate=candidate,
        cluster_id=cluster_id,
        dispatch_batch_id=dispatch_batch_id,
        portfolio_plan_id=portfolio_plan_id,
    )


def _socket_stack_failed_qa_result(tasks):
    """Return unattributed failing QA evidence for the baseline-backed cluster."""
    result = _failed_delta_qa_result(tasks)
    for task in tasks:
        package_name = task.target_package_name
        raw_excerpt = (
            f"{package_name} parser/transport integration failure"
            if package_name in {"socket.io", "socket.io-parser"}
            else "shared transport integration failure"
        )
        result["qa_evaluations"][task.task_id] = result["qa_evaluations"][task.task_id].model_copy(
            update={
                "failure_evidence": QAFailureEvidence(
                    raw_excerpt=raw_excerpt,
                    exact_diagnostics=["atomic test batch failed"],
                    failed_tests=["shared Socket.IO integration test"],
                    attempt_id=task.current_attempt_id,
                    task_revision=task.task_revision,
                ),
                "deterministic_gates": QADeterministicGates(
                    status="fail",
                    install_passed=True,
                    scanner_execution_status="success",
                    tests_passed=False,
                    diagnostics=["unit-test gate failed"],
                ),
                "test_attribution": QATestAttribution(
                    verdict="inconclusive",
                    failed_tests=["shared Socket.IO integration test"],
                ),
            }
        )
    return result


def test_socket_stack_delta_fixture_exercises_three_task_blame_and_replan(
    socket_stack_delta_case, monkeypatch
):
    case = socket_stack_delta_case
    sandbox = _DeltaSandbox(
        case.candidate.copy(),
        {case.dispatch_batch_id: case.baseline.copy()},
    )
    qa_observations = []
    qa_result = _socket_stack_failed_qa_result(case.tasks)

    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        lambda _state: qa_result,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=2,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(
        _graph_wrappers,
        "apply_multi_package_action",
        _apply_fake_delta_action,
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )

    def run_unit_tests(active_sandbox):
        changed_packages = [
            package_name
            for package_name, target_version in case.candidate.items()
            if active_sandbox.workspace[package_name] == target_version
        ]
        qa_observations.append((tuple(changed_packages), dict(active_sandbox.workspace)))
        return SimpleNamespace(ok=changed_packages == ["socket.io"])

    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        run_unit_tests,
    )

    qa_output = run_qa_critic_from_orchestrator(case.state)

    isolation = qa_output["delta_isolation_by_cluster"][case.cluster_id]
    assert isolation["status"] == "IDENTIFIED"
    assert isolation["responsible_task_ids"] == ["task-c"]
    assert isolation["tested_subsets"] == [["task-a"], ["task-c"]]
    assert isolation["executions"] == 2
    assert isolation["candidate_restored"] is True
    assert isolation["dispatch_batch_id"] == case.dispatch_batch_id
    assert isolation["source_portfolio_plan_id"] == case.portfolio_plan_id
    assert isolation["attempt_ids_by_task"] == {
        task.task_id: task.current_attempt_id for task in case.tasks
    }
    assert qa_observations == [
        (
            ("socket.io",),
            {
                "socket.io": "4.8.1",
                "engine.io": "4.1.2",
                "socket.io-parser": "4.0.5",
            },
        ),
        (
            ("socket.io-parser",),
            {
                "socket.io": "3.1.2",
                "engine.io": "4.1.2",
                "socket.io-parser": "4.2.6",
            },
        ),
    ]
    assert sandbox.workspace == case.baseline
    assert case.dispatch_batch_id not in sandbox.archives
    assert not any(snapshot_id.startswith("delta-candidate-") for snapshot_id in sandbox.archives)

    supervisor_state = {
        **case.state,
        **qa_output,
        "portfolio_plan": None,
    }
    supervisor_output = run_supervisor_node(supervisor_state)

    assert supervisor_output["task_queue"]["task-a"].status == TaskStatus.PENDING
    assert supervisor_output["task_queue"]["task-a"].retry_count == 0
    assert supervisor_output["task_queue"]["task-b"].status == TaskStatus.PENDING
    assert supervisor_output["task_queue"]["task-b"].retry_count == 0
    assert supervisor_output["task_queue"]["task-c"].status == TaskStatus.NEEDS_RETRY
    assert supervisor_output["task_queue"]["task-c"].retry_count == 1
    assert all(
        supervisor_output["task_queue"][task.task_id].current_attempt_id is None
        for task in case.tasks
    )
    request = supervisor_output["portfolio_replan_request"]
    assert isinstance(request, PortfolioReplanRequest)
    assert request.forced_singleton_task_ids == ["task-c"]
    assert request.triggering_attempt_id == case.tasks[2].current_attempt_id
    assert request.source_portfolio_plan_id == case.portfolio_plan_id
    assert supervisor_output["next_routing_step"] == "portfolio"

    stale_record = {
        **isolation,
        "attempt_ids_by_task": {
            **isolation["attempt_ids_by_task"],
            "task-c": "attempt-stale",
        },
    }
    stale_supervisor_output = run_supervisor_node(
        {
            **supervisor_state,
            "delta_isolation_by_cluster": {case.cluster_id: stale_record},
        }
    )
    for task in case.tasks:
        committed = stale_supervisor_output["task_queue"][task.task_id]
        assert committed.status == TaskStatus.NEEDS_RETRY
        assert committed.retry_count == task.retry_count + 1
        assert committed.current_attempt_id is None
    stale_request = stale_supervisor_output["portfolio_replan_request"]
    assert stale_request.forced_singleton_task_ids == []
    assert stale_request.triggering_attempt_id is None
    assert stale_request.source_portfolio_plan_id is None


def _socket_stack_new_dispatch(case, *, batch_id, attempt_prefix):
    """Copy the Socket.IO fixture into a distinct committed dispatch."""
    tasks = [
        task.model_copy(
            update={
                "task_revision": task.task_revision + 1,
                "current_attempt_id": f"{attempt_prefix}-{task.task_id}",
            }
        )
        for task in case.tasks
    ]
    action = MultiPackageAction(
        cluster_id=case.cluster_id,
        dispatch_batch_id=batch_id,
        selected_strategy=case.action.selected_strategy,
        package_mutations=list(case.action.package_mutations),
        rationale=case.action.rationale,
    )
    action_digest = instruction_digest(action.model_dump_json())
    snapshots = {}
    decisions = []
    for task in tasks:
        snapshot = TaskAttemptSnapshot(
            attempt_id=task.current_attempt_id,
            task_id=task.task_id,
            state_revision=2,
            task_revision=task.task_revision,
            attempt_number=2,
            cluster_id=case.cluster_id,
            dispatch_batch_id=batch_id,
            action_digest=action_digest,
            portfolio_plan_id=case.portfolio_plan_id,
            qa_policy=task.qa_policy,
            strategy_stage=task.strategy_stage,
            selected_version=task.selected_version,
            target_package_name=task.target_package_name,
            target_dependency_type=task.target_dependency_type,
            instruction=task.instruction,
            instruction_digest=instruction_digest(task.instruction),
            dispatch_node="update_subagent",
        )
        snapshots[snapshot.attempt_id] = snapshot
        decisions.append(
            SimpleNamespace(
                task_id=task.task_id,
                target_package_name=task.target_package_name,
                installed_version=case.baseline[task.target_package_name],
            )
        )
    state = {
        **case.state,
        "task_queue": {task.task_id: task for task in tasks},
        "active_target_task_ids": [task.task_id for task in tasks],
        "active_dispatch_batch_id": batch_id,
        "active_multi_package_action": action,
        "attempt_snapshots_by_id": {**case.snapshots, **snapshots},
        "portfolio_plan": SimpleNamespace(
            portfolio_plan_id=case.portfolio_plan_id,
            solver_plan=SimpleNamespace(selected_plan=SimpleNamespace(task_decisions=decisions)),
        ),
    }
    return SimpleNamespace(
        state=state,
        tasks=tasks,
        action=action,
        snapshots=snapshots,
        batch_id=batch_id,
    )


def test_socket_stack_fixture_reuses_same_attempt_budget_and_resets_for_new_dispatch(
    socket_stack_delta_case, monkeypatch
):
    case = socket_stack_delta_case
    sandbox = _DeltaSandbox(
        case.candidate.copy(),
        {case.dispatch_batch_id: case.baseline.copy()},
    )
    qa_calls = []
    apply_calls = []

    def qa_critic(scoped_state):
        qa_calls.append(True)
        active_tasks = [
            scoped_state["task_queue"][task_id]
            for task_id in scoped_state["active_target_task_ids"]
        ]
        return _socket_stack_failed_qa_result(active_tasks)

    def apply_action(active_sandbox, action, touched_files):
        apply_calls.append(tuple(mutation.task_id for mutation in action.package_mutations))
        return _apply_fake_delta_action(active_sandbox, action, touched_files)

    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        qa_critic,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=2,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(_graph_wrappers, "apply_multi_package_action", apply_action)
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        lambda _sandbox: SimpleNamespace(ok=True),
    )

    first = run_qa_critic_from_orchestrator(case.state)
    same_attempt = run_qa_critic_from_orchestrator(
        {
            **case.state,
            "delta_isolation_by_cluster": first["delta_isolation_by_cluster"],
        }
    )
    first_record = first["delta_isolation_by_cluster"][case.cluster_id]

    assert first_record["status"] == "INCONCLUSIVE"
    assert first_record["candidate_restored"] is True
    assert first_record["executions"] == 2
    assert same_attempt["delta_isolation_by_cluster"][case.cluster_id] == first_record
    assert len(apply_calls) == 2
    assert (
        len(
            [
                event
                for event in sandbox.events
                if event[0] == "create" and event[1].startswith("delta-candidate-")
            ]
        )
        == 1
    )

    fresh_case = _socket_stack_new_dispatch(
        case,
        batch_id="batch-socket-stack-fresh",
        attempt_prefix="attempt-fresh",
    )
    fresh_case.state["delta_isolation_by_cluster"] = first["delta_isolation_by_cluster"]
    sandbox.archives[fresh_case.batch_id] = case.baseline.copy()
    sandbox.workspace.clear()
    sandbox.workspace.update(case.candidate)

    fresh = run_qa_critic_from_orchestrator(fresh_case.state)
    fresh_record = fresh["delta_isolation_by_cluster"][case.cluster_id]

    assert fresh_record["status"] == "INCONCLUSIVE"
    assert fresh_record["candidate_restored"] is True
    assert fresh_record["executions"] == 2
    assert fresh_record["dispatch_batch_id"] == fresh_case.batch_id
    assert fresh_record["action_digest"] != first_record["action_digest"]
    assert fresh_record["attempt_ids_by_task"] == {
        task.task_id: task.current_attempt_id for task in fresh_case.tasks
    }
    assert len(apply_calls) == 4
    assert len(qa_calls) == 3
    assert (
        len(
            [
                event
                for event in sandbox.events
                if event[0] == "create" and event[1].startswith("delta-candidate-")
            ]
        )
        == 2
    )


def test_socket_stack_candidate_restore_failure_clears_blame(socket_stack_delta_case, monkeypatch):
    case = socket_stack_delta_case
    sandbox = _DeltaSandbox(
        case.candidate.copy(),
        {case.dispatch_batch_id: case.baseline.copy()},
        fail_candidate_restore=True,
    )
    qa_result = _socket_stack_failed_qa_result(case.tasks)
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.run_qa_critic_node",
        lambda _state: qa_result,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.get_runtime_settings",
        lambda: AppSettings(
            max_delta_canary_probes=2,
            remedy_disable_post_qa_triage=True,
        ),
    )
    monkeypatch.setattr(
        _graph_wrappers,
        "apply_multi_package_action",
        _apply_fake_delta_action,
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_install",
        lambda _sandbox: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        _graph_wrappers._qa_test_parsing,
        "_run_unit_tests",
        lambda _sandbox: SimpleNamespace(ok=False),
    )

    result = run_qa_critic_from_orchestrator(case.state)
    isolation = result["delta_isolation_by_cluster"][case.cluster_id]

    assert isolation["status"] == "INCONCLUSIVE"
    assert isolation["candidate_restored"] is False
    assert isolation["responsible_task_ids"] == []
    assert (
        result["qa_results_by_attempt"]["attempt-task-a"].evaluation.evidence_inconclusive is False
    )
    assert result["qa_results_by_attempt"]["attempt-task-a"].evaluation.retry_feedback == (
        "a retry feedback"
    )
    assert sandbox.workspace == case.baseline
    assert case.dispatch_batch_id not in sandbox.archives
    assert any(snapshot_id.startswith("delta-candidate-") for snapshot_id in sandbox.archives)


@pytest.mark.parametrize("malformation", ["missing", "duplicate"])
def test_socket_stack_fixture_rejects_mutation_id_mismatch_before_snapshot(
    socket_stack_delta_case, monkeypatch, malformation
):
    case = socket_stack_delta_case
    if malformation == "missing":
        mutations = case.action.package_mutations[:2]
    else:
        duplicate = PackageMutation(
            task_id="task-a",
            package_name="socket.io-shadow",
            target_version="4.8.1",
            dependency_type="overrides",
        )
        mutations = [
            case.action.package_mutations[0],
            duplicate,
            case.action.package_mutations[1],
        ]
    malformed_action = MultiPackageAction(
        cluster_id=case.action.cluster_id,
        dispatch_batch_id=case.action.dispatch_batch_id,
        selected_strategy=case.action.selected_strategy,
        package_mutations=mutations,
        rationale=case.action.rationale,
    )
    malformed_state = {
        **case.state,
        "active_multi_package_action": malformed_action,
    }
    ranker = MagicMock()
    canaries = MagicMock()
    apply_action = MagicMock()
    sandbox = MagicMock()
    monkeypatch.setattr(
        _graph_wrappers,
        "rank_suspect_tasks_by_suspicion",
        ranker,
    )
    monkeypatch.setattr(
        _graph_wrappers,
        "run_delta_isolation_canaries",
        canaries,
    )
    monkeypatch.setattr(
        _graph_wrappers,
        "apply_multi_package_action",
        apply_action,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.graph.DockerSandbox",
        lambda **_kwargs: sandbox,
    )

    result = _graph_wrappers._maybe_run_delta_isolation(
        malformed_state,
        case.tasks,
        _socket_stack_failed_qa_result(case.tasks),
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["candidate_restored"] is True
    assert result["executions"] == 0
    assert result["tested_subsets"] == []
    ranker.assert_not_called()
    canaries.assert_not_called()
    apply_action.assert_not_called()
    sandbox.create_workspace_snapshot.assert_not_called()
