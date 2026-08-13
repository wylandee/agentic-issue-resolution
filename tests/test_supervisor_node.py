"""
Tests for the Phase 5 Supervisor Node.

Tests have been updated to use the task-centric architecture:
- task_queue (Dict[str, RemediationTask]) replaces group_statuses/group_strategies/retry_counts
- target_task_ids replaces target_group_ids in SupervisorDecision
- AgentActionSummary.task_id replaces .group_id
- QAEvaluation.task_id replaces .group_id
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    SupervisorDecision,
    SupervisorRetryPlan,
    TaskAttemptSnapshot,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundPhase,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.subagent_runtime import ToolEvent
from remediation_engine.orchestration.supervisor_node import (
    MAX_RETRIES,
    _deterministic_routing,
    _instruction_digest,
    _materialize_spawn_requests,
    _normalize_target_task_ids_for_node,
    _parse_planner_retry_plans,
    _planner_plan_violations,
    _reconcile_registry_plan_evidence,
    build_supervisor_prompt,
    reconcile_phase5_state_before_teardown,
    run_supervisor_node,
    supervisor_router,
)
from remediation_engine.orchestration.task_utils import derive_initial_strategy


def _issue() -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        message="Test issue",
        id=str(uuid4()),
    )


def _sca_group(group_id="g1", fix_status=FixPlanStatus.VERSION_FOUND) -> VulnerabilityGroup:
    fix_plan = FixPlan(
        status=fix_status,
        fixed_version="1.2.3" if fix_status == FixPlanStatus.VERSION_FOUND else None,
        workaround_snippets=["test"] if fix_status == FixPlanStatus.WORKAROUND_FOUND else None,
        instruction="test",
        strategy_used="test",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="test-pkg",
        file_path="package.json",
        cve_ids=[],
        versions=[],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=str(uuid4()),
        issues=[_issue()],
        fix_plan=fix_plan,
    )


def _sast_group(group_id="g2") -> VulnerabilityGroup:
    fix_plan = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND,
        fixed_version=None,
        workaround_snippets=["disable it"],
        instruction="test",
        strategy_used="test",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SAST,
        vulnerable_component="test-func",
        file_path="src/index.js",
        cve_ids=[],
        versions=[],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=str(uuid4()),
        issues=[_issue()],
        fix_plan=fix_plan,
    )


def _make_task(
    task_id: str,
    group_id: str,
    strategy: RoutingStrategy = RoutingStrategy.VERSION_BUMP,
    status: TaskStatus = TaskStatus.PENDING,
    retry_count: int = 0,
) -> RemediationTask:
    return RemediationTask(
        task_id=task_id,
        parent_group_id=group_id,
        strategy=strategy,
        status=status,
        retry_count=retry_count,
    )


def _base_state(groups, **overrides) -> dict:
    state = {
        "repo_root": "/tmp/repo",
        "valid_groups": groups,
        "task_queue": {},
        "active_target_task_ids": [],
        "active_target_group_ids": [],
        "qa_evaluations": {},
        "action_summaries": [],
        "constraints_ledger": [],
        "feedback_by_group": {},
        "feedback_by_task": {},
        "eval_status": "",
        "status": "supervisor_entered",
    }
    state.update(overrides)
    return state


# ===========================================================================
# Schema tests
# ===========================================================================


class TestSupervisorDecisionSchema:
    def test_valid_routes_accepted(self):
        decision = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["t1"],
            instructions="test",
            decision_reason="test",
        )
        assert decision.next_node == "update_subagent"

    def test_workaround_subagent_rejects_zero_or_two_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_task_ids=["t1", "t2"],
                instructions="test",
                decision_reason="test",
            )

    def test_update_subagent_rejects_zero_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )

    def test_update_subagent_retains_reusable_batch_shape(self):
        decision = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["t1", "t2"],
            instructions="test",
            decision_reason="direct batch compatibility",
        )
        assert decision.target_task_ids == ["t1", "t2"]

    def test_update_subagent_rejects_more_than_ten_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=[f"t{i}" for i in range(11)],
                instructions="test",
                decision_reason="test",
            )

    def test_qa_critic_requires_non_empty_targets_and_teardown_rejects_them(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )
        decision = SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=["t1"],
            instructions="test",
            decision_reason="test",
        )
        assert decision.target_task_ids == ["t1"]
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="teardown",
                target_task_ids=["t1"],
                instructions="test",
                decision_reason="test",
            )

    def test_overlapping_unfixable_and_targets_rejected(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=["t1"],
                unfixable_task_ids=["t1"],
                instructions="test",
                decision_reason="test",
            )


# ===========================================================================
# derive_initial_strategy (now in task_utils)
# ===========================================================================


class TestDeriveInitialStrategy:
    def test_version_found_plan_yields_version_bump(self):
        g = _sca_group(fix_status=FixPlanStatus.VERSION_FOUND)
        assert derive_initial_strategy(g) == RoutingStrategy.VERSION_BUMP

    def test_workaround_found_plan_yields_code_workaround(self):
        g = _sca_group(fix_status=FixPlanStatus.WORKAROUND_FOUND)
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND

    def test_no_fix_plan_yields_code_workaround(self):
        g = _sca_group(fix_status=FixPlanStatus.NO_FIX)
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND

    def test_missing_fix_plan_yields_code_workaround(self):
        g = _sast_group()
        g.fix_plan = None
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND


# ===========================================================================
# supervisor_router
# ===========================================================================


class TestSupervisorRouterFunction:
    def test_valid_next_routing_steps_route_correctly(self):
        assert supervisor_router({"next_routing_step": "update_subagent"}) == "update_subagent"
        assert (
            supervisor_router({"next_routing_step": "workaround_subagent"}) == "workaround_subagent"
        )
        assert supervisor_router({"next_routing_step": "qa_critic"}) == "qa_critic"
        assert supervisor_router({"next_routing_step": "teardown"}) == "teardown"

    def test_missing_or_unknown_step_defaults_to_teardown(self):
        assert supervisor_router({}) == "teardown"
        assert supervisor_router({"next_routing_step": "invalid"}) == "teardown"


# ===========================================================================
# run_supervisor_node â€” task initialization
# ===========================================================================


class TestRunSupervisorNodeNormalization:
    def test_creates_tasks_for_all_groups(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])
        result = run_supervisor_node(state)
        tasks = result["task_queue"]
        assert len(tasks) == 1
        task = next(iter(tasks.values()))
        assert task.parent_group_id == "g1"
        assert task.strategy == RoutingStrategy.VERSION_BUMP
        assert task.status == TaskStatus.PENDING

    def test_does_not_create_duplicate_tasks_for_existing_groups(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        existing_task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP)
        state = _base_state([g1], task_queue={"task-1": existing_task})
        result = run_supervisor_node(state)
        # Should still have exactly 1 task
        assert len(result["task_queue"]) == 1


# ===========================================================================
# run_supervisor_node â€” routing decisions
# ===========================================================================


class TestRunSupervisorNodeVersionBump:
    @patch("langchain_openai.ChatOpenAI")
    def test_version_bump_tasks_dispatch_one_at_a_time(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1, g2])

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm

        # Make LLM fail so we rely on deterministic routing
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_deterministic_routing_caps_update_dispatch_at_one(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        state = _base_state(groups)

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_deterministic_retry_routing_dispatches_one_task(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            )
            for i in range(3)
        }
        decision = _deterministic_routing(
            {task.task_id: task for task in tasks.values()},
            {group.group_id: group for group in groups},
            {},
            {},
        )

        assert decision.next_node == "update_subagent"
        assert decision.target_task_ids == ["task-1"]

    def test_repeated_routing_advances_to_the_next_task_after_completion(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {f"task-{i + 1}": _make_task(f"task-{i + 1}", f"g{i}") for i in range(3)}
        task_queue = {task.task_id: task for task in tasks.values()}
        groups_by_id = {group.group_id: group for group in groups}

        first = _deterministic_routing(task_queue, groups_by_id, {}, {})
        assert first.target_task_ids == ["task-1"]

        task_queue["task-1"] = task_queue["task-1"].model_copy(
            update={"status": TaskStatus.QA_PASSED}
        )
        second = _deterministic_routing(task_queue, groups_by_id, {}, {})
        assert second.target_task_ids == ["task-2"]

    @patch("langchain_openai.ChatOpenAI")
    def test_version_bump_routes_to_update_subagent_via_llm(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1, g2])

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm

        def make_decision(prompt_text):
            # Extract task IDs from the state's task_queue (they're created during node)
            # We can't know them here, but we can check the result
            return SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=["task-1", "task-2"],
                instructions="test",
                decision_reason="test",
            )

        mock_llm.with_structured_output.return_value.invoke.side_effect = make_decision

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]


class TestRunSupervisorNodeWorkaround:
    def test_code_workaround_deterministic_routes_one_task(self):
        g1 = _sast_group("g1")
        g2 = _sast_group("g2")
        state = _base_state([g1, g2])

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "workaround_subagent"
        assert len(result["active_target_task_ids"]) == 1


class TestRunSupervisorNodeToQA:
    def test_optimistically_fixed_task_routes_to_qa(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert "task-1" in result["active_target_task_ids"]

    def test_all_optimistically_fixed_tasks_route_to_qa_one_at_a_time(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            # No active_target_task_ids â€” all tasks are already optimistically fixed
            active_target_task_ids=[],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_current_task_routes_to_qa_before_more_updates(self):
        # 12 groups: first 10 are optimistically fixed, 2 are pending.
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.OPTIMISTICALLY_FIXED if i < 10 else TaskStatus.PENDING,
            )
            for i in range(12)
        }
        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=[f"task-{i + 1}" for i in range(10)],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]

    @patch("langchain_openai.ChatOpenAI")
    def test_mixed_update_batch_routes_successful_subset_to_qa(self, mock_chat):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(5)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            )
            for i in range(5)
        }
        summaries = [
            AgentActionSummary(
                task_id="task-1", status=AgentActionStatus.SUCCESS, summary="updated jsonwebtoken"
            ),
            AgentActionSummary(
                task_id="task-2", status=AgentActionStatus.SUCCESS, summary="updated express-jwt"
            ),
            AgentActionSummary(
                task_id="task-3",
                status=AgentActionStatus.SUCCESS,
                summary="updated @tootallnate/once",
            ),
            AgentActionSummary(
                task_id="task-4",
                status=AgentActionStatus.SURRENDER,
                summary="ws reverted due to dependency conflict",
            ),
            AgentActionSummary(
                task_id="task-5",
                status=AgentActionStatus.SURRENDER,
                summary="elliptic update path exhausted",
            ),
        ]
        state = _base_state(
            groups,
            task_queue=tasks,
            action_summaries=summaries,
            active_target_task_ids=[f"task-{i + 1}" for i in range(5)],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-2"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-3"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-4"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-5"].status == TaskStatus.NEEDS_RETRY
        mock_chat.assert_not_called()

    def test_terminal_tasks_in_active_batch_are_omitted_from_qa_targets(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            "task-1": _make_task("task-1", "g0", status=TaskStatus.OPTIMISTICALLY_FIXED),
            "task-2": _make_task("task-2", "g1", status=TaskStatus.QA_PASSED),
            "task-3": _make_task("task-3", "g2", status=TaskStatus.UNFIXABLE),
        }
        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=["task-1", "task-2", "task-3"],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]

    @patch("remediation_engine.orchestration.supervisor_node._run_planner_phase")
    @patch("langchain_openai.ChatOpenAI")
    def test_optimistically_fixed_task_routes_to_qa_before_retry_planning(
        self, mock_chat, mock_planner
    ):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.NEEDS_RETRY, retry_count=1)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            active_target_task_ids=["task-1", "task-2"],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]
        mock_planner.assert_not_called()
        mock_chat.assert_not_called()


class TestRunSupervisorNodeToTeardown:
    def test_qa_passed_routes_to_teardown(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )
        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"

    def test_unfixable_routes_to_teardown(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )
        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"


# ===========================================================================
# run_supervisor_node â€” QA evaluation updates
# ===========================================================================


class TestRunSupervisorNodeQAUpdates:
    @patch("remediation_engine.orchestration.supervisor_node._run_planner_phase")
    @patch("langchain_openai.ChatOpenAI")
    def test_qa_failed_retry_invokes_planner_before_router(self, mock_chat, mock_planner):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.NEEDS_RETRY)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="retry with better update evidence",
                )
            },
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1"],
            revised_instructions={
                "task-1": "Investigate patched releases or override paths for test-pkg."
            },
            instructions="route retry task",
            decision_reason="planner kept update remediation active",
        )
        mock_planner.return_value = "Strategy Scratchpad\nretry update path still open"

        run_supervisor_node(state)

        mock_planner.assert_called_once()
        prompt_text = structured.invoke.call_args[0][0]
        assert "Strategy Scratchpad" in prompt_text
        assert "retry update path still open" in prompt_text

    def test_qa_completed_passed_marks_task_qa_passed(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_qa_completed_passed_adds_constraint(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["constraints_ledger"] == ["test-pkg: keep resolved version at 1.2.3"]

    def test_qa_completed_passed_workaround_adds_constraint(self):
        g1 = _sast_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
        )
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["constraints_ledger"] == ["test-func: preserve validated security workaround"]

    def test_qa_completed_passed_does_not_duplicate_existing_constraint(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
            constraints_ledger=["test-pkg: keep resolved version at 1.2.3"],
        )
        result = run_supervisor_node(state)
        assert result["constraints_ledger"] == []

    @patch("langchain_openai.ChatOpenAI")
    def test_qa_completed_failed_marks_task_needs_retry(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="try again",
                )
            },
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-1"].retry_count == 1

    def test_not_qa_completed_does_not_update_statuses_from_qa_evals(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="supervisor_entered",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        # Status should NOT become QA_PASSED because status != "qa_completed"
        # It will route to qa_critic since task is OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED

    @patch("langchain_openai.ChatOpenAI")
    def test_failed_group_keyed_qa_eval_replans_active_child_task(self, mock_chat):
        g1 = _sca_group("g1")
        parent = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        child = _make_task("task-2", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={
                "task-2": child,
                "task-1": parent,
            },
            active_target_task_ids=["task-2"],
            qa_evaluations={
                "g1": QAEvaluation(
                    task_id="g1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="retry the active child task",
                )
            },
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-2"],
            revised_instructions={
                "task-2": (
                    "Apply strategy stage npm_same_major: update package.json "
                    "to exact version 1.2.4."
                )
            },
            instructions="retry child",
            decision_reason="replan the active retry task",
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-2"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-2"].retry_count == 1
        assert result["task_queue"]["task-2"].instruction == (
            "Apply strategy stage npm_same_major: update package.json to exact version 1.2.4."
        )
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]

    @patch("langchain_openai.ChatOpenAI")
    def test_retry_update_without_revised_instruction_falls_back_to_high_level_retry(
        self, mock_chat
    ):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.NEEDS_RETRY,
        )
        task.instruction = 'Update "test-pkg" in package.json to version "1.2.3".'
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1"],
            instructions="Retry with current strategy.",
            decision_reason="retry the failed version bump",
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-1"].instruction.startswith("Apply strategy stage")
        assert "exact OSV minimum fixed version 1.2.3" in result["task_queue"]["task-1"].instruction
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert any("rejected update_subagent retry dispatch" in err for err in result["errors"])

    @patch("langchain_openai.ChatOpenAI")
    def test_pending_update_task_can_seed_instruction_from_revised_instructions(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        task.instruction = ""
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1"],
            revised_instructions={
                "task-1": 'Update "test-pkg" in package.json to version "1.2.4".'
            },
            instructions="seed the initial task instruction",
            decision_reason="planner provided an exact version",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert (
            result["task_queue"]["task-1"].instruction
            == 'Update "test-pkg" in package.json to version "1.2.4".'
        )


def test_planner_none_clears_stale_selection_and_marks_latest_path_exhausted():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=3,
    ).model_copy(
        update={
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "instruction": "Update package.json to exact version 1.2.3.",
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="1.2.3",
        attempted_versions=["1.2.3"],
        latest_version_seen="1.2.3",
    )

    updated, plans = _parse_planner_retry_plans(
        """
TASK: task-1
SELECTED_VERSION: NONE
ACTION: pivot_workaround
The only candidate has already been attempted; pivot to a workaround child.
""",
        {"task-1": task},
        {"task-1": diagnostics},
        {"g1": group},
    )

    assert updated["task-1"].selected_version is None
    assert updated["task-1"].exhausted_update_path is True
    assert plans["task-1"].action == "pivot_workaround"
    assert "1.2.3" not in plans["task-1"].exact_instruction


def test_planner_rejects_a_retry_for_an_already_attempted_version():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        strategy_stage=SCARemediationStage.NPM_LATEST,
        attempted_versions=["6.0.0", "6.1.2"],
        latest_version_seen="8.5.1",
    )

    updated, plans = _parse_planner_retry_plans(
        """
TASK: task-1, SELECTED_VERSION: 6.1.2, EFFECTIVE_STAGE: npm_latest, ACTION: retry_update
The same-major latest version 6.1.2 is already attempted, but retry it.
""",
        {"task-1": task},
        {"task-1": diagnostics},
        {"g1": group},
    )

    violations = _planner_plan_violations(plans, {"task-1": task}, updated)
    assert any(
        "6.1.2" in violation and "already attempted" in violation for violation in violations
    )


def test_planner_rejects_stage_regression_from_code_workaround_to_update():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
    ).model_copy(
        update={
            "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
            "selected_version": "6.1.1",
            "instruction": "stale update instruction",
        }
    )

    updated, plans = _parse_planner_retry_plans(
        """
TASK: task-1, SELECTED_VERSION: 6.1.1, EFFECTIVE_STAGE: same_major, ACTION: retry_update
The same-major candidate has not been attempted yet; retry it.
""",
        {"task-1": task},
        {
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.CODE_WORKAROUND,
                attempted_versions=["6.0.0", "6.1.2", "8.5.1"],
                latest_version_seen="8.5.1",
            )
        },
        {"g1": group},
    )

    violations = _planner_plan_violations(plans, {"task-1": task}, updated)

    assert plans["task-1"].strategy_stage == SCARemediationStage.NPM_SAME_MAJOR
    assert any("stage npm_same_major regresses" in violation for violation in violations)


def test_unknown_planner_stage_is_fail_closed():
    group = _sca_group("g1")
    task = _make_task("task-1", "g1", status=TaskStatus.NEEDS_RETRY)
    updated, plans = _parse_planner_retry_plans(
        "TASK: task-1, SELECTED_VERSION: 1.2.4, EFFECTIVE_STAGE: invented_stage, ACTION: retry_update",
        {"task-1": task},
        {"task-1": UpdateRetryDiagnostics(task_id="task-1")},
        {"g1": group},
    )

    violations = _planner_plan_violations(plans, {"task-1": task}, updated)

    assert plans["task-1"].strategy_stage == SCARemediationStage.CODE_WORKAROUND
    assert any(
        "retry_update cannot use code_workaround stage" in violation for violation in violations
    )


def test_same_major_latest_equal_latest_forces_workaround_pivot_even_if_llm_says_retry():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    updated, plans = _parse_planner_retry_plans(
        """
TASK: task-1, SELECTED_VERSION: NONE, EFFECTIVE_STAGE: npm_latest, ACTION: retry_update
The same-major latest version (8.21.1) is equal to the latest stable version.
The same-major latest version (8.21.1) is already attempted, so retry it.
""",
        {"task-1": task},
        {
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_LATEST,
                attempted_versions=["8.21.1"],
            )
        },
        {"g1": group},
    )

    assert updated["task-1"].exhausted_update_path is True
    assert plans["task-1"].action == "pivot_workaround"


def test_registry_tool_result_overrides_free_form_selected_version():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        strategy_stage=SCARemediationStage.NPM_LATEST,
        attempted_versions=["6.1.2"],
    )
    parsed_diagnostics, parsed_plans = _parse_planner_retry_plans(
        "TASK: task-1, SELECTED_VERSION: 6.1.2, EFFECTIVE_STAGE: npm_latest, ACTION: retry_update",
        {"task-1": task},
        {"task-1": diagnostics},
        {"g1": group},
    )

    tool_event = ToolEvent(
        name="plan_npm_version",
        args={
            "package_name": "test-pkg",
            "selection": "latest",
            "security_floor": "1.2.3",
            "attempted_versions": "6.1.2",
        },
        content=(
            "# NPM Version Plan: test-pkg\n"
            "- Selection: latest\n"
            "- Selected Version: 8.5.1\n"
            "- Same-Major Latest: 6.1.2\n"
            "- Latest Stable: 8.5.1\n"
            "- Same-Major Stage: APPLICABLE\n"
            "- Eligible Candidates: 8.5.1, 6.1.2"
        ),
    )
    updated, plans = _reconcile_registry_plan_evidence(
        parsed_plans,
        parsed_diagnostics,
        {"task-1": task},
        {"g1": group},
        [tool_event],
    )

    assert updated["task-1"].selected_version == "8.5.1"
    assert plans["task-1"].selected_version == "8.5.1"
    assert not _planner_plan_violations(plans, {"task-1": task}, updated)


def test_planner_pivot_is_committed_before_router_and_routes_workaround_child(monkeypatch):
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        qa_evaluations={
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Repair the express-jwt compatibility regression.",
                failure_evidence=QAFailureEvidence(
                    attempt_id="update-attempt-1",
                    task_revision=1,
                    exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
                    source_locations=["/workspace/lib/insecurity.ts:54:35"],
                ),
            )
        },
        qa_results_by_attempt={
            "update-attempt-1": QAAttemptResult(
                attempt_id="update-attempt-1",
                task_id="task-1",
                task_revision=1,
                evaluation=QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="Repair the express-jwt compatibility regression.",
                    failure_evidence=QAFailureEvidence(
                        attempt_id="update-attempt-1",
                        task_revision=1,
                        exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
                        source_locations=["/workspace/lib/insecurity.ts:54:35"],
                    ),
                ),
            )
        },
        retry_diagnostics_by_task={
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_LATEST,
                selected_version="1.2.3",
                attempted_versions=["1.2.3"],
                latest_version_seen="1.2.3",
            )
        },
        status="qa_completed",
    )

    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._run_planner_phase",
        lambda *args, **kwargs: (
            "TASK: task-1\n"
            "SELECTED_VERSION: NONE\n"
            "ACTION: pivot_workaround\n"
            "The update path is exhausted; pivot to a workaround child."
        ),
    )
    mock_chat = MagicMock()
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat)

    result = run_supervisor_node(state)

    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert result["next_routing_step"] == "workaround_subagent"
    assert len(result["active_target_task_ids"]) == 1
    child_id = result["active_target_task_ids"][0]
    assert result["task_queue"][child_id].strategy == RoutingStrategy.CODE_WORKAROUND
    assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
    child_attempt_id = result["task_queue"][child_id].current_attempt_id
    child_snapshot = result["attempt_snapshots_by_id"][child_attempt_id]
    assert child_snapshot.workaround_context is not None
    assert child_snapshot.workaround_context.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
    assert child_snapshot.workaround_context.qa_evidence is not None
    assert child_snapshot.workaround_context.qa_evidence.source_locations == [
        "/workspace/lib/insecurity.ts:54:35"
    ]


def test_invalid_planner_selection_is_corrected_before_router_and_worker_dispatch(monkeypatch):
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        status="qa_completed",
        qa_evaluations={
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="the attempted version still fails QA",
            )
        },
        retry_diagnostics_by_task={
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_LATEST,
                attempted_versions=["6.0.0", "6.1.2"],
                latest_version_seen="8.5.1",
            )
        },
    )

    planner_outputs = iter(
        [
            (
                "TASK: task-1, SELECTED_VERSION: 6.1.2, "
                "EFFECTIVE_STAGE: npm_latest, ACTION: retry_update\n"
                "The same-major latest is already attempted, but retry it."
            ),
            (
                "TASK: task-1, SELECTED_VERSION: 8.5.1, "
                "EFFECTIVE_STAGE: npm_latest, ACTION: retry_update\n"
                "Use the unattempted latest stable release."
            ),
        ]
    )
    planner_mock = MagicMock(side_effect=lambda *args, **kwargs: next(planner_outputs))
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._run_planner_phase", planner_mock
    )

    router_llm = MagicMock()
    structured = MagicMock()
    router_llm.with_structured_output.return_value = structured
    structured.invoke.return_value = SupervisorDecision(
        next_node="update_subagent",
        target_task_ids=["task-1"],
        instructions="route the corrected retry",
        decision_reason="planner supplied a validated exact version",
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", MagicMock(return_value=router_llm))

    result = run_supervisor_node(state)

    assert planner_mock.call_count == 2
    correction = planner_mock.call_args_list[1].kwargs["correction"]
    assert "already attempted" in correction
    # QA had already advanced the committed task to the terminal update
    # stage before the planner ran.  A corrective proposal cannot reopen that
    # path, even when it names an otherwise unattempted version.
    assert result["retry_diagnostics_by_task"]["task-1"].selected_version is None
    assert result["retry_diagnostics_by_task"]["task-1"].strategy_stage == (
        SCARemediationStage.NPM_LATEST
    )
    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert "8.5.1" not in result["task_queue"]["task-1"].instruction
    assert "exact version" not in result["task_queue"]["task-1"].instruction.lower()
    assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
    assert result["next_routing_step"] == "workaround_subagent"
    assert len(result["active_target_task_ids"]) == 1
    assert result["task_queue"][result["active_target_task_ids"][0]].strategy == (
        RoutingStrategy.CODE_WORKAROUND
    )
    assert any("planner semantic validation" in error for error in result["errors"])


def test_stale_worker_result_is_ignored_when_new_attempt_is_committed():
    group = _sca_group("g1")
    old_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-old",
        task_id="task-1",
        state_revision=1,
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        selected_version="1.0.0",
        instruction="Update test-pkg to 1.0.0.",
        instruction_digest=_instruction_digest("Update test-pkg to 1.0.0."),
        dispatch_node="update_subagent",
    )
    new_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-new",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction="Update test-pkg to 2.0.0.",
        instruction_digest=_instruction_digest("Update test-pkg to 2.0.0."),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": "attempt-new",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    stale_result = WorkerAttemptResult(
        attempt_id="attempt-old",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["1.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.0.0"],
            executed_versions=["1.0.0"],
            validation_passed=True,
        ),
        instruction_digest=old_snapshot.instruction_digest,
    )
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={
            old_snapshot.attempt_id: old_snapshot,
            new_snapshot.attempt_id: new_snapshot,
        },
        worker_results_by_attempt={old_snapshot.attempt_id: stale_result},
    )

    result = run_supervisor_node(state)

    committed = result["task_queue"]["task-1"]
    assert committed.current_attempt_id == "attempt-new"
    assert committed.selected_version == "2.0.0"
    assert committed.status == TaskStatus.OPTIMISTICALLY_FIXED
    assert any(event.error_code == "STALE_WORKER_RESULT" for event in result["consistency_events"])


def test_failed_update_attempt_is_closed_before_retry_planner_commit(monkeypatch):
    group = _sca_group("g1")
    old_instruction = "Update test-pkg to 1.0.0."
    old_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-failed",
        task_id="task-1",
        state_revision=1,
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        selected_version="1.0.0",
        instruction=old_instruction,
        instruction_digest=_instruction_digest(old_instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": old_snapshot.attempt_id,
            "strategy_stage": old_snapshot.strategy_stage,
            "selected_version": old_snapshot.selected_version,
            "instruction": old_snapshot.instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=old_snapshot.attempt_id,
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SURRENDER,
        executed_versions=["1.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.0.0"],
            executed_versions=["1.0.0"],
            validation_passed=False,
        ),
        instruction_digest=old_snapshot.instruction_digest,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._run_planner_phase",
        lambda *args, **kwargs: (
            "TASK: task-1, SELECTED_VERSION: 2.0.0, "
            "EFFECTIVE_STAGE: npm_same_major, ACTION: retry_update\n"
            "Use the unattempted same-major version."
        ),
    )
    router_llm = MagicMock()
    structured = MagicMock()
    router_llm.with_structured_output.return_value = structured
    structured.invoke.return_value = SupervisorDecision(
        next_node="update_subagent",
        target_task_ids=["task-1"],
        instructions="route the new retry",
        decision_reason="use the committed planner retry",
    )
    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        MagicMock(return_value=router_llm),
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={old_snapshot.attempt_id: old_snapshot},
            worker_results_by_attempt={old_snapshot.attempt_id: worker_result},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
                    attempted_versions=["1.0.0"],
                    candidate_versions_considered=["2.0.0"],
                )
            },
        )
    )

    committed = result["task_queue"]["task-1"]
    assert committed.current_attempt_id != old_snapshot.attempt_id
    assert committed.selected_version == "2.0.0"
    assert committed.task_revision > old_snapshot.task_revision
    assert (
        result["attempt_snapshots_by_id"][committed.current_attempt_id].selected_version == "2.0.0"
    )
    assert not any(
        event.error_code == "TASK_SNAPSHOT_REPAIRED" for event in result["consistency_events"]
    )


def test_stale_qa_result_does_not_increment_retry_count():
    group = _sca_group("g1")
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction="Update test-pkg to 2.0.0.",
        instruction_digest=_instruction_digest("Update test-pkg to 2.0.0."),
        dispatch_node="update_subagent",
    )
    stale_qa = QAAttemptResult(
        attempt_id="attempt-old",
        task_id="task-1",
        task_revision=1,
        evaluation=QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="stale result",
        ),
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        retry_count=1,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": "attempt-current",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    state = _base_state(
        [group],
        status="qa_completed",
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        qa_results_by_attempt={stale_qa.attempt_id: stale_qa},
    )

    result = run_supervisor_node(state)

    assert result["task_queue"]["task-1"].retry_count == 1
    assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
    assert any(event.error_code == "STALE_QA_RESULT" for event in result["consistency_events"])


def test_terminal_task_cannot_retain_retry_plan():
    group = _sca_group("g1")
    task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE).model_copy(
        update={"instruction": "Do not retry this task."}
    )
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        retry_plans_by_task={
            "task-1": SupervisorRetryPlan(
                task_id="task-1",
                selected_version="8.5.1",
                exact_instruction="Retry with 8.5.1.",
            )
        },
    )

    result = run_supervisor_node(state)

    assert "task-1" not in result["retry_plans_by_task"]
    assert any(
        event.error_code == "TERMINAL_TASK_PLAN_CLEARED" for event in result["consistency_events"]
    )


def test_teardown_barrier_detaches_terminal_attempt_and_is_idempotent():
    group = _sca_group("g1")
    instruction = "Apply the workaround for test-pkg."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-terminal",
        task_id="task-1",
        state_revision=4,
        task_revision=4,
        strategy_stage=SCARemediationStage.CODE_WORKAROUND,
        selected_version=None,
        instruction=instruction,
        instruction_digest=_instruction_digest(instruction),
        dispatch_node="workaround_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.UNFIXABLE,
    ).model_copy(
        update={
            "task_revision": 4,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
            "instruction": instruction,
        }
    )
    state = _base_state(
        [group],
        state_revision=4,
        task_queue={"task-1": task},
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        retry_plans_by_task={
            "task-1": SupervisorRetryPlan(
                task_id="task-1",
                strategy_stage=SCARemediationStage.CODE_WORKAROUND,
                selected_version=None,
                action="pivot_workaround",
                exact_instruction=instruction,
            )
        },
    )

    first = reconcile_phase5_state_before_teardown(state)

    committed = first["task_queue"]["task-1"]
    assert committed.status == TaskStatus.UNFIXABLE
    assert committed.current_attempt_id is None
    assert committed.selected_version is None
    assert first["retry_plans_by_task"] == {}
    assert any(
        event.error_code == "TERMINAL_TASK_FIELDS_NORMALIZED"
        for event in first["consistency_events"]
    )

    second = reconcile_phase5_state_before_teardown({**state, **first})
    assert second["task_queue"]["task-1"].current_attempt_id is None
    assert second["retry_plans_by_task"] == {}
    assert second["consistency_events"] == []


def test_terminal_pivot_parent_is_not_reseeded_from_initial_osv_plan():
    group = _sca_group("g1")
    pivot_instruction = "Implement a code workaround for the exhausted update path."
    task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE).model_copy(
        update={
            "task_revision": 3,
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": None,
            "exhausted_update_path": True,
            "instruction": pivot_instruction,
            "current_attempt_id": None,
        }
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_LATEST,
                    selected_version=None,
                    exhausted_update_path=True,
                )
            },
        )
    )

    committed = result["task_queue"]["task-1"]
    assert committed.selected_version is None
    assert committed.instruction == pivot_instruction
    assert committed.current_attempt_id is None
    assert "task-1" not in result["retry_plans_by_task"]


def test_pivot_detaches_previous_update_attempt_before_child_dispatch(monkeypatch):
    group = _sca_group("g1")
    instruction = "Apply the selected dependency version 8.5.1."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-update",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="8.5.1",
        instruction=instruction,
        instruction_digest=_instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "instruction": snapshot.instruction,
            "exhausted_update_path": False,
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._run_planner_phase",
        lambda *args, **kwargs: (
            "TASK: task-1\n"
            "SELECTED_VERSION: NONE\n"
            "EFFECTIVE_STAGE: npm_latest\n"
            "ACTION: pivot_workaround\n"
            "The latest update failed; pivot to a workaround child."
        ),
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=[],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_LATEST,
                    selected_version="8.5.1",
                    attempted_versions=["6.0.0", "6.1.2", "8.5.1"],
                    latest_version_seen="8.5.1",
                    exhausted_update_path=True,
                )
            },
        )
    )

    parent = result["task_queue"]["task-1"]
    child_id = result["active_target_task_ids"][0]
    assert result["next_routing_step"] == "workaround_subagent"
    assert parent.status == TaskStatus.UNFIXABLE
    assert parent.current_attempt_id is None
    assert parent.selected_version is None
    assert parent.strategy_stage == SCARemediationStage.NPM_LATEST
    assert parent.exhausted_update_path is True
    assert "exact version" not in parent.instruction.lower()
    assert "child" in parent.instruction.lower()
    assert result["retry_diagnostics_by_task"]["task-1"].strategy_stage == (
        SCARemediationStage.NPM_LATEST
    )
    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert result["task_queue"][child_id].current_attempt_id is not None
    assert result["task_queue"][child_id].current_attempt_id in result["attempt_snapshots_by_id"]


# ===========================================================================
# run_supervisor_node â€” action summary updates
# ===========================================================================


class TestRunSupervisorNodeActionSummary:
    @patch("remediation_engine.orchestration.supervisor_node._run_planner_phase")
    @patch("langchain_openai.ChatOpenAI")
    def test_qa_passed_subset_is_removed_before_retry_routing(self, mock_chat, mock_planner):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.NEEDS_RETRY, retry_count=1)
        state = _base_state(
            [g1, g2],
            status="qa_completed",
            task_queue={"task-1": task1, "task-2": task2},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            revised_instructions={
                "task-2": "Investigate patched releases or override paths for test-pkg."
            },
            instructions="retry remaining task",
            decision_reason="task-1 passed QA; task-2 still needs retry",
        )
        mock_planner.return_value = "Strategy Scratchpad\nretry task-2"

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        mock_planner.assert_called_once()


class TestRunSupervisorNodeActionSummaryUpdates:
    def test_updates_task_to_optimistically_fixed_on_success(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED

    def test_updates_task_to_needs_retry_on_surrender(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="failed",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY

    def test_updates_workaround_task_to_unfixable_on_surrender(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.CODE_WORKAROUND, status=TaskStatus.PENDING
        )
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="Workaround subagent bypassed",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE

    def test_handles_multiple_action_summaries(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        summary1 = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed task-1",
        )
        summary2 = AgentActionSummary(
            task_id="task-2",
            status=AgentActionStatus.SUCCESS,
            summary="fixed task-2",
        )
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            action_summaries=[summary1, summary2],
            active_target_task_ids=["task-1", "task-2"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-2"].status == TaskStatus.OPTIMISTICALLY_FIXED

    def test_does_not_overwrite_terminal_statuses_from_action_summary(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        # QA_PASSED is terminal â€” must not be overwritten
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_does_not_update_non_active_targets_from_action_summary(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.NEEDS_RETRY)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        summary_g1 = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        # task-1 has a SUCCESS summary but is NOT in active_target_task_ids.
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            action_summaries=[summary_g1],
            active_target_task_ids=["task-2"],
        )
        result = run_supervisor_node(state)
        # task-1 must NOT be overwritten by its historical summary
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY


# ===========================================================================
# run_supervisor_node â€” max retries / unfixable marking
# ===========================================================================


class TestRunSupervisorNodeTargetGuardrails:
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_update_targets_drop_terminal_tasks(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            revised_instructions={
                "task-1": "Do not dispatch terminal tasks.",
                "task-2": 'Update "test-pkg" in package.json to version "1.2.4".',
            },
            instructions="route update batch",
            decision_reason="router included one stale terminal task",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_worker_target_normalization_drops_optimistically_fixed_tasks(self):
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)

        assert _normalize_target_task_ids_for_node(
            "update_subagent",
            ["task-1", "task-2"],
            {"task-1": task1, "task-2": task2},
        ) == ["task-2"]

        assert _normalize_target_task_ids_for_node(
            "qa_critic",
            ["task-1", "task-2"],
            {"task-1": task1, "task-2": task2},
        ) == ["task-1"]


class TestRunSupervisorMaxRetries:
    def test_max_retries_marks_task_unfixable_and_removes_from_targets(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES,
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert "task-1" not in result["active_target_task_ids"]

    @patch("langchain_openai.ChatOpenAI")
    def test_all_terminal_after_retry_cap_skips_router_and_tears_down(self, mock_chat):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        g3 = _sca_group("g3")
        task1 = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        task2 = _make_task(
            "task-2",
            "g2",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES - 1,
        )
        task3 = _make_task(
            "task-3",
            "g3",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES - 1,
        )
        summaries = [
            AgentActionSummary(
                task_id="task-2",
                status=AgentActionStatus.SURRENDER,
                summary="No viable manifest update remains.",
            ),
            AgentActionSummary(
                task_id="task-3",
                status=AgentActionStatus.SURRENDER,
                summary="No viable manifest update remains.",
            ),
        ]
        state = _base_state(
            [g1, g2, g3],
            task_queue={
                "task-1": task1,
                "task-2": task2,
                "task-3": task3,
            },
            action_summaries=summaries,
            active_target_task_ids=["task-2", "task-3"],
            status="supervisor_entered",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "teardown"
        assert result["active_target_task_ids"] == []
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["task_queue"]["task-2"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-3"].status == TaskStatus.UNFIXABLE
        mock_chat.assert_not_called()


# ===========================================================================
# run_supervisor_node â€” strategy pivots
# ===========================================================================


class TestRunSupervisorNodePeerConflict:
    @patch("langchain_openai.ChatOpenAI")
    def test_workaround_spawn_with_missing_target_is_normalized_to_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
        )
        state = _base_state([g1], task_queue={"task-1": task})
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            SupervisorDecision.model_construct(
                next_node="workaround_subagent",
                target_task_ids=[],
                spawn_requests=[
                    TaskSpawnRequest(
                        parent_task_id="task-1",
                        strategy=RoutingStrategy.CODE_WORKAROUND,
                        instruction="Implement a code workaround for the failed update.",
                        reason="the update path is exhausted",
                    )
                ],
                instructions="pivot to workaround",
                decision_reason="pivot after exhausted update path",
            )
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert len(result["active_target_task_ids"]) == 1
        child_id = result["active_target_task_ids"][0]
        assert result["task_queue"][child_id].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_llm_strategy_pivot_spawns_child_and_routes_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Implement a code workaround for the unresolved peer conflict.",
                    reason="peer conflict requires a workaround child task",
                )
            ],
            instructions="test",
            decision_reason="test",
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].strategy == RoutingStrategy.VERSION_BUMP
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND
        assert (
            result["task_queue"]["task-2"].instruction
            == "Implement a code workaround for the unresolved peer conflict."
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_breaking_change_pivot_marks_parent_passed_and_routes_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="jsonwebtoken v9 broke runtime expectations",
                )
            },
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Implement a compatibility workaround for the new jsonwebtoken API.",
                    reason="breaking change requires validated upgrade plus workaround child",
                )
            ],
            instructions="test",
            decision_reason="spawn a workaround child after the validated version bump caused regressions",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_strategy_pivot_without_child_instruction_fails_closed(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            updated_task_strategies={"task-1": RoutingStrategy.CODE_WORKAROUND},
            instructions="retry with a workaround",
            decision_reason="pivot the task after the retry failed",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "teardown"
        assert result["active_target_task_ids"] == []
        assert "task-2" not in result["task_queue"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert any(
            "rejected strategy pivot without task-specific child instructions" in err
            for err in result["errors"]
        )

    @patch("langchain_openai.ChatOpenAI")
    def legacy_breaking_change_fallback_spawns_child_instead_of_retrying_parent(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="jsonwebtoken v9 broke runtime expectations",
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_exhausted_update_path_fallback_spawns_child_and_terminalizes_parent(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    selected_version=None,
                    latest_version_seen="9.0.3",
                    used_overrides=False,
                    package_abandoned=False,
                    exhausted_update_path=True,
                    failure_reason="latest version already attempted",
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_exhausted_update_at_retry_cap_pivots_before_terminal_cap(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES,
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    latest_version_seen="9.0.3",
                    exhausted_update_path=True,
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"

    @patch("remediation_engine.orchestration.supervisor_node._run_planner_phase")
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_cannot_route_exhausted_update_back_to_update_subagent(
        self, mock_chat, mock_planner
    ):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    latest_version_seen="9.0.3",
                    exhausted_update_path=True,
                )
            },
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1"],
            revised_instructions={"task-1": "Try update again even though it is exhausted."},
            instructions="incorrectly retry exhausted update",
            decision_reason="bad router decision",
        )
        mock_planner.return_value = "Strategy Scratchpad\nincorrect retry"

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"

    @patch("langchain_openai.ChatOpenAI")
    def test_multi_spawn_materializes_all_children_but_routes_first_child(self, mock_chat):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                strategy=RoutingStrategy.VERSION_BUMP,
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            )
            for i in range(3)
        }
        state = _base_state(
            groups,
            task_queue=tasks,
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 1.",
                    reason="pivot 1",
                ),
                TaskSpawnRequest(
                    parent_task_id="task-2",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 2.",
                    reason="pivot 2",
                ),
                TaskSpawnRequest(
                    parent_task_id="task-3",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 3.",
                    reason="pivot 3",
                ),
            ],
            instructions="spawn three workaround children",
            decision_reason="batch pivot to workaround",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-4"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-3"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-4"].parent_task_id == "task-1"
        assert result["task_queue"]["task-5"].parent_task_id == "task-2"
        assert result["task_queue"]["task-6"].parent_task_id == "task-3"
        assert result["task_queue"]["task-4"].status == TaskStatus.PENDING
        assert result["task_queue"]["task-5"].status == TaskStatus.PENDING
        assert result["task_queue"]["task-6"].status == TaskStatus.PENDING


# ===========================================================================
# run_supervisor_node â€” LLM fallback
# ===========================================================================


class TestRunSupervisorLLMFallback:
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_exception_uses_deterministic_fallback(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)
        # Deterministic fallback routes VERSION_FOUND to update_subagent
        assert result["next_routing_step"] == "update_subagent"
        assert len(result["active_target_task_ids"]) == 1


# ===========================================================================
# run_supervisor_node â€” LLM structured output call
# ===========================================================================


class TestSupervisorLLMStructuredOutput:
    @patch("langchain_openai.ChatOpenAI")
    def test_uses_with_structured_output(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state([g1])

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = SupervisorDecision(
            next_node="teardown",
            target_task_ids=[],
            instructions="test",
            decision_reason="test",
        )

        run_supervisor_node(state)

        mock_llm.with_structured_output.assert_called_once_with(
            SupervisorDecision, method="function_calling"
        )
        mock_structured.invoke.assert_called_once()


# ===========================================================================
# Supervisor prompt
# ===========================================================================


class TestSupervisorPrompt:
    def test_prompt_instructs_llm_to_route_one_update_and_qa_task(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        prompt = build_supervisor_prompt(_base_state(groups))

        assert "Send exactly one pending VERSION_BUMP task to update_subagent." in prompt
        assert "Send exactly one retry VERSION_BUMP task to update_subagent." in prompt
        assert "update_subagent MUST have exactly one target_task_id." in prompt
        assert "qa_critic MUST have exactly one target_task_id." in prompt

    def test_prompt_includes_task_ids(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP)
        state = _base_state([g1], task_queue={"task-1": task})
        prompt = build_supervisor_prompt(state)
        assert "task-1" in prompt
        assert "g1" in prompt

    def test_prompt_carries_new_global_scanner_findings_without_task_attribution(self):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            baseline_scan_identifiers=["CVE-2021-23337"],
            post_remediation_scan_identifiers=["CVE-2025-10001"],
            new_vulnerability_identifiers=["CVE-2025-10001"],
            new_vulnerability_status="detected",
        )

        prompt = build_supervisor_prompt(state)

        assert "CVE-2025-10001" in prompt
        assert "report-only until the later triage phase" in prompt

    def test_prompt_enforces_security_flag_update_and_breaking_change_workaround(self):
        g1 = _sca_group("g1")
        prompt = build_supervisor_prompt(_base_state([g1]))

        assert "SECURITY_FLAG and PEER_CONFLICT remain update remediation first" in prompt
        assert "BREAKING_CHANGE also advances through the ordered update stages" in prompt
        assert (
            "Only an exhausted NPM_LATEST stage may pivot to a CODE_WORKAROUND child task" in prompt
        )


class TestBugFixes:
    def test_bug1_deterministic_pivot_populates_feedback_by_task(self):
        group_id = "sca:package.json:express:UPDATE_VERSION"
        g = _sca_group(group_id)
        task = RemediationTask(
            task_id="task-1",
            parent_group_id=group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        task_queue = {"task-1": task}
        group_by_id = {group_id: g}
        qa_evaluations = {
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.PEER_CONFLICT,
                retry_feedback="Peer dependency conflict with body-parser",
            )
        }
        retry_diagnostics_by_task = {
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                exhausted_update_path=True,
            )
        }

        decision = _deterministic_routing(
            task_queue=task_queue,
            group_by_id=group_by_id,
            qa_evaluations=qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
        )

        assert decision.next_node == "workaround_subagent"
        assert len(decision.spawn_requests) == 1
        assert (
            decision.feedback_by_task.get("task-1") == "Peer dependency conflict with body-parser"
        )

    def test_bug2_materialize_spawn_requests_replaces_triage_strategy_bucket(self):
        group_id = "sca:package.json:express:UPDATE_VERSION"
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id=group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        task_queue = {"task-1": parent_task}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Pivot to workaround",
            reason="Exhausted version bump",
        )
        errors: list[str] = []

        new_tasks, _ = _materialize_spawn_requests(
            spawn_requests=[spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
        )

        assert len(new_tasks) == 1
        child_task = next(iter(new_tasks.values()))
        assert child_task.parent_group_id == "sca:package.json:express:CODE_WORKAROUND"
        assert child_task.strategy == RoutingStrategy.CODE_WORKAROUND
        assert len(errors) == 0

    def test_bug3_reject_workaround_to_workaround_spawn(self):
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id="g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Workaround",
        )
        task_queue = {"task-1": parent_task}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Refactor workaround",
            reason="Workaround failed",
        )
        errors: list[str] = []

        new_tasks, _ = _materialize_spawn_requests(
            spawn_requests=[spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
        )

        assert len(new_tasks) == 0
        assert len(errors) == 1
        assert (
            "CODE_WORKAROUND parent 'task-1' cannot spawn another CODE_WORKAROUND child"
            in errors[0]
        )

    def test_bug3_deterministic_routing_filters_exhausted_workaround_tasks(self):
        task = RemediationTask(
            task_id="task-1",
            parent_group_id="g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=3,  # MAX_RETRIES
            instruction="Workaround",
        )
        task_queue = {"task-1": task}

        decision = _deterministic_routing(
            task_queue=task_queue,
            group_by_id={},
            qa_evaluations={},
            retry_diagnostics_by_task={},
        )

        assert decision.next_node == "teardown"
        assert decision.target_task_ids == []

    @patch("langchain_openai.ChatOpenAI")
    def test_bug4_programmatic_feedback_injection_for_workaround_retries(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            target_task_ids=["task-2"],
            next_node="workaround_subagent",
            spawn_requests=[],
            status_updates={},
            unfixable_task_ids=[],
            new_constraints=[],
            feedback_by_task={},
            instructions="Routing workaround",
            decision_reason="Testing",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        group_id = "sca:package.json:express:CODE_WORKAROUND"
        g = _sca_group(group_id)
        task = RemediationTask(
            task_id="task-2",
            parent_group_id=group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=1,
            instruction="Workaround",
        )
        task_queue = {"task-2": task}
        group_by_id = {group_id: g}
        qa_evaluations = {
            "task-2": QAEvaluation(
                task_id="task-2",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Real feedback from QA",
            )
        }

        result = run_supervisor_node(
            {
                "repo_root": "/tmp",
                "issues": [],
                "system_context": MagicMock(),
                "constraints_ledger": [],
                "retry_counts": {},
                "group_statuses": {},
                "group_strategies": {},
                "action_summaries": [],
                "retry_plans_by_task": {},
                "task_queue": task_queue,
                "valid_groups": [g],
                "qa_evaluations": qa_evaluations,
                "retry_diagnostics_by_task": {},
            }
        )

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["feedback_by_task"]["task-2"] == "Real feedback from QA"
