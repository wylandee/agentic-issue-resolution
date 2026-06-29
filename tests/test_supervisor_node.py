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

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    Severity,
    SupervisorDecision,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.supervisor_node import (
    MAX_RETRIES,
    build_supervisor_prompt,
    run_supervisor_node,
    supervisor_router,
)
from src.orchestrator.task_utils import build_initial_remediation_task, derive_initial_strategy


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
        assert supervisor_router({"next_routing_step": "workaround_subagent"}) == "workaround_subagent"
        assert supervisor_router({"next_routing_step": "qa_critic"}) == "qa_critic"
        assert supervisor_router({"next_routing_step": "teardown"}) == "teardown"

    def test_missing_or_unknown_step_defaults_to_teardown(self):
        assert supervisor_router({}) == "teardown"
        assert supervisor_router({"next_routing_step": "invalid"}) == "teardown"


# ===========================================================================
# run_supervisor_node — task initialization
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
# run_supervisor_node — routing decisions
# ===========================================================================


class TestRunSupervisorNodeVersionBump:
    @patch("langchain_openai.ChatOpenAI")
    def test_version_bump_tasks_batch_to_update_subagent(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1, g2])

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm

        # Make LLM fail so we rely on deterministic routing
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert len(result["active_target_task_ids"]) == 2

    def test_deterministic_routing_caps_update_batch_at_ten(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        state = _base_state(groups)

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert len(result["active_target_task_ids"]) == 10

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

    def test_all_optimistically_fixed_tasks_route_to_qa(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            # No active_target_task_ids — all tasks are already optimistically fixed
            active_target_task_ids=[],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert set(result["active_target_task_ids"]) == {"task-1", "task-2"}

    def test_current_batch_routes_to_qa_before_more_updates(self):
        # 12 groups: first 10 are optimistically fixed (active batch), 2 are pending
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        tasks = {
            f"task-{i+1}": _make_task(
                f"task-{i+1}",
                f"g{i}",
                status=TaskStatus.OPTIMISTICALLY_FIXED if i < 10 else TaskStatus.PENDING,
            )
            for i in range(12)
        }
        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=[f"task-{i+1}" for i in range(10)],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert set(result["active_target_task_ids"]) == {f"task-{i+1}" for i in range(10)}


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
# run_supervisor_node — QA evaluation updates
# ===========================================================================


class TestRunSupervisorNodeQAUpdates:
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
            "task-1", "g1",
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

    def test_qa_completed_failed_marks_task_needs_retry(self):
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


# ===========================================================================
# run_supervisor_node — action summary updates
# ===========================================================================


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

    def test_handles_batch_prefix_in_action_summary(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        summary = AgentActionSummary(
            task_id="batch:task-1, task-2",
            status=AgentActionStatus.SUCCESS,
            summary="fixed batch",
        )
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            action_summaries=[summary],
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
        # QA_PASSED is terminal — must not be overwritten
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
# run_supervisor_node — max retries / unfixable marking
# ===========================================================================


class TestRunSupervisorMaxRetries:
    def test_max_retries_marks_task_unfixable_and_removes_from_targets(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1",
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


# ===========================================================================
# run_supervisor_node — strategy pivots
# ===========================================================================


class TestRunSupervisorNodePeerConflict:
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_updated_strategy_pivot_applied(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY)
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
            instructions="test",
            decision_reason="test",
        )

        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].strategy == RoutingStrategy.CODE_WORKAROUND


# ===========================================================================
# run_supervisor_node — LLM fallback
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
# run_supervisor_node — LLM structured output call
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

        mock_llm.with_structured_output.assert_called_once_with(SupervisorDecision, method="function_calling")
        mock_structured.invoke.assert_called_once()


# ===========================================================================
# Supervisor prompt
# ===========================================================================


class TestSupervisorPrompt:
    def test_prompt_instructs_llm_to_cap_update_batches_at_ten(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        prompt = build_supervisor_prompt(_base_state(groups))

        assert "batches of at most 10" in prompt
        assert "Never send more than 10 target_task_ids to update_subagent" in prompt

    def test_prompt_includes_task_ids(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP)
        state = _base_state([g1], task_queue={"task-1": task})
        prompt = build_supervisor_prompt(state)
        assert "task-1" in prompt
        assert "g1" in prompt
