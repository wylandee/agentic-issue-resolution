"""
Tests for the Phase 5 Supervisor Node.
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
    GroupRemediationStatus,
    IssueSource,
    IssueType,
    QAEvaluation,
    RoutingStrategy,
    Severity,
    SupervisorDecision,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.supervisor_node import (
    MAX_RETRIES,
    build_supervisor_prompt,
    derive_initial_strategy,
    run_supervisor_node,
    supervisor_router,
)


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


def _base_state(groups, **overrides) -> dict:
    return {
        "repo_root": "/tmp/repo",
        "valid_groups": groups,
        "group_strategies": {},
        "group_statuses": {},
        "retry_counts": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "constraints_ledger": [],
        "eval_status": "",
        "status": "supervisor_entered",
        **overrides,
    }


class TestSupervisorDecisionSchema:
    def test_valid_routes_accepted(self):
        decision = SupervisorDecision(
            next_node="update_subagent",
            target_group_ids=["g1"],
            instructions="test",
            decision_reason="test",
        )
        assert decision.next_node == "update_subagent"

    def test_workaround_subagent_rejects_zero_or_two_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_group_ids=[],
                instructions="test",
                decision_reason="test",
            )
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_group_ids=["g1", "g2"],
                instructions="test",
                decision_reason="test",
            )

    def test_update_subagent_rejects_zero_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_group_ids=[],
                instructions="test",
                decision_reason="test",
            )

    def test_qa_critic_and_teardown_reject_non_empty_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="qa_critic",
                target_group_ids=["g1"],
                instructions="test",
                decision_reason="test",
            )
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="teardown",
                target_group_ids=["g1"],
                instructions="test",
                decision_reason="test",
            )

    def test_overlapping_unfixable_and_targets_rejected(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_group_ids=["g1"],
                unfixable_group_ids=["g1"],
                instructions="test",
                decision_reason="test",
            )


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


class TestSupervisorRouterFunction:
    def test_valid_next_routing_steps_route_correctly(self):
        assert supervisor_router({"next_routing_step": "update_subagent"}) == "update_subagent"
        assert supervisor_router({"next_routing_step": "workaround_subagent"}) == "workaround_subagent"
        assert supervisor_router({"next_routing_step": "qa_critic"}) == "qa_critic"
        assert supervisor_router({"next_routing_step": "teardown"}) == "teardown"

    def test_missing_or_unknown_step_defaults_to_teardown(self):
        assert supervisor_router({}) == "teardown"
        assert supervisor_router({"next_routing_step": "invalid"}) == "teardown"


class TestRunSupervisorNodeNormalization:
    def test_initializes_missing_strategies(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])
        result = run_supervisor_node(state)
        assert result["group_strategies"]["g1"] == RoutingStrategy.VERSION_BUMP

    def test_initializes_missing_statuses_to_pending(self):
        g1 = _sca_group("g1")
        state = _base_state([g1])
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.PENDING


class TestRunSupervisorNodeVersionBump:
    @patch("langchain_openai.ChatOpenAI")
    def test_version_bump_groups_batch_to_update_subagent(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1, g2])

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_group_ids=["g1", "g2"],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert set(result["active_target_group_ids"]) == {"g1", "g2"}


class TestRunSupervisorNodeWorkaround:
    @patch("langchain_openai.ChatOpenAI")
    def test_code_workaround_routes_one_group(self, mock_chat):
        g1 = _sast_group("g1")
        g2 = _sast_group("g2")
        state = _base_state([g1, g2])

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_group_ids=["g1"],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_group_ids"] == ["g1"]


class TestRunSupervisorNodeToQA:
    @patch("langchain_openai.ChatOpenAI")
    def test_optimistically_fixed_routes_to_qa(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="qa_critic",
            target_group_ids=[],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_group_ids"] == []


class TestRunSupervisorNodeToTeardown:
    @patch("langchain_openai.ChatOpenAI")
    def test_qa_passed_routes_to_teardown(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            group_statuses={"g1": GroupRemediationStatus.QA_PASSED},
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="teardown",
            target_group_ids=[],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"
        assert result["active_target_group_ids"] == []

    @patch("langchain_openai.ChatOpenAI")
    def test_unfixable_routes_to_teardown(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            group_statuses={"g1": GroupRemediationStatus.UNFIXABLE},
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="teardown",
            target_group_ids=[],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"
        assert result["active_target_group_ids"] == []


class TestRunSupervisorNodeQAUpdates:
    def test_qa_completed_passed_updates_status(self):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            status="qa_completed",
            qa_evaluations={"g1": QAEvaluation(group_id="g1", passed=True)},
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.QA_PASSED
        assert result["constraints_ledger"] == ["test-pkg: keep resolved version at 1.2.3"]

    def test_qa_completed_passed_workaround_adds_constraint(self):
        g1 = _sast_group("g1")
        state = _base_state(
            [g1],
            status="qa_completed",
            qa_evaluations={"g1": QAEvaluation(group_id="g1", passed=True)},
            group_strategies={"g1": RoutingStrategy.CODE_WORKAROUND},
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.QA_PASSED
        assert result["constraints_ledger"] == ["test-func: preserve validated security workaround"]

    def test_qa_completed_passed_does_not_duplicate_existing_constraint(self):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            status="qa_completed",
            qa_evaluations={"g1": QAEvaluation(group_id="g1", passed=True)},
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
            constraints_ledger=["test-pkg: keep resolved version at 1.2.3"],
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.QA_PASSED
        assert result["constraints_ledger"] == []

    def test_qa_completed_failed_updates_status_and_retry_count(self):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            status="qa_completed",
            qa_evaluations={
                "g1": QAEvaluation(
                    group_id="g1", 
                    passed=False, 
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="try again"
                )
            },
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
            retry_counts={"g1": 0},
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.NEEDS_RETRY
        assert result["retry_counts"]["g1"] == 1

    def test_not_qa_completed_does_not_update_statuses(self):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            status="supervisor_entered",
            qa_evaluations={"g1": QAEvaluation(group_id="g1", passed=True)},
            group_statuses={"g1": GroupRemediationStatus.OPTIMISTICALLY_FIXED},
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.OPTIMISTICALLY_FIXED


class TestRunSupervisorNodePeerConflict:
    @patch("langchain_openai.ChatOpenAI")
    def test_peer_conflict_pivots_strategy(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state([g1])

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_group_ids=["g1"],
            updated_strategies={"g1": RoutingStrategy.CODE_WORKAROUND},
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["group_strategies"]["g1"] == RoutingStrategy.CODE_WORKAROUND


class TestRunSupervisorMaxRetries:
    @patch("langchain_openai.ChatOpenAI")
    def test_max_retries_marks_unfixable_and_removes_from_targets(self, mock_chat):
        g1 = _sca_group("g1")
        state = _base_state(
            [g1],
            group_statuses={"g1": GroupRemediationStatus.NEEDS_RETRY},
            retry_counts={"g1": MAX_RETRIES},
        )

        # Mock deterministic routing to avoid LLM error
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="teardown",
            target_group_ids=[],
            instructions="test",
            decision_reason="test",
        )
        mock_chat.return_value = mock_llm

        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.UNFIXABLE
        assert "g1" not in result["active_target_group_ids"]


class TestRunSupervisorLLMFallback:
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_exception_uses_deterministic_fallback(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])

        # Mock ChatOpenAI to raise ImportError (simulating missing langchain-openai)
        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)
        # Deterministic fallback routes VERSION_FOUND to update_subagent
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_group_ids"] == ["g1"]


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
            target_group_ids=[],
            instructions="test",
            decision_reason="test",
        )

        run_supervisor_node(state)

        mock_llm.with_structured_output.assert_called_once_with(SupervisorDecision, method="function_calling")
        mock_structured.invoke.assert_called_once()


class TestRunSupervisorNodeActionSummaryUpdates:
    def test_updates_status_to_optimistically_fixed_on_success(self):
        g1 = _sca_group("g1")
        summary = AgentActionSummary(
            group_id="g1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it"
        )
        state = _base_state(
            [g1],
            action_summaries=[summary],
            group_statuses={"g1": GroupRemediationStatus.PENDING},
            active_target_group_ids=["g1"]
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.OPTIMISTICALLY_FIXED

    def test_updates_status_to_needs_retry_on_surrender(self):
        g1 = _sca_group("g1")
        summary = AgentActionSummary(
            group_id="g1",
            status=AgentActionStatus.SURRENDER,
            summary="failed"
        )
        state = _base_state(
            [g1],
            action_summaries=[summary],
            group_statuses={"g1": GroupRemediationStatus.PENDING},
            active_target_group_ids=["g1"]
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.NEEDS_RETRY

    def test_handles_batch_prefix_in_action_summary(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        summary = AgentActionSummary(
            group_id="batch:g1, g2",
            status=AgentActionStatus.SUCCESS,
            summary="fixed batch"
        )
        state = _base_state(
            [g1, g2],
            action_summaries=[summary],
            group_statuses={
                "g1": GroupRemediationStatus.PENDING,
                "g2": GroupRemediationStatus.PENDING
            },
            active_target_group_ids=["g1", "g2"]
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.OPTIMISTICALLY_FIXED
        assert result["group_statuses"]["g2"] == GroupRemediationStatus.OPTIMISTICALLY_FIXED

    def test_does_not_overwrite_terminal_statuses(self):
        g1 = _sca_group("g1")
        summary = AgentActionSummary(
            group_id="g1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it"
        )
        state = _base_state(
            [g1],
            action_summaries=[summary],
            group_statuses={"g1": GroupRemediationStatus.QA_PASSED},
            active_target_group_ids=["g1"]
        )
        result = run_supervisor_node(state)
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.QA_PASSED

    def test_does_not_overwrite_status_for_non_active_targets(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        summary_g1 = AgentActionSummary(
            group_id="g1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it"
        )
        # G1 is NEEDS_RETRY and is NOT in active_target_group_ids.
        # G2 is the active target.
        state = _base_state(
            [g1, g2],
            action_summaries=[summary_g1],
            group_statuses={
                "g1": GroupRemediationStatus.NEEDS_RETRY,
                "g2": GroupRemediationStatus.PENDING
            },
            active_target_group_ids=["g2"]
        )
        result = run_supervisor_node(state)
        # G1's status MUST NOT be overwritten back to OPTIMISTICALLY_FIXED by its historical summary.
        assert result["group_statuses"]["g1"] == GroupRemediationStatus.NEEDS_RETRY
