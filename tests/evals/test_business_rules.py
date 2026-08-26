"""Phase 5: Business rules evaluation — Token budgets, Latency SLAs, and Tool Call budgets."""

from __future__ import annotations

import logging

import pytest

from tests.evals.adapters import (
    LLMTestCase,
    ToolCall,
    TrajectoryDocument,
    extract_token_usage,
    extract_tool_calls,
)
from tests.evals.conftest import TrajectoryLoader
from tests.evals.custom_metrics import (
    LatencySLAMetric,
    TokenBudgetMetric,
    ToolCallBudgetMetric,
)

try:
    from deepeval import assert_test
except ImportError:
    assert_test = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit Tests for Metric Classes
# ---------------------------------------------------------------------------


class TestTokenBudgetMetricUnit:
    """Unit tests for the TokenBudgetMetric BaseMetric implementation."""

    def test_within_budget_passes(self) -> None:
        """Metric succeeds with score 1.0 when prompt and completion are within ceiling."""
        metric = TokenBudgetMetric(
            agent_type="triage", max_prompt_tokens=5000, max_completion_tokens=1000
        )
        tc = LLMTestCase(
            name="TokenBudget [triage - within budget]",
            input="triage request",
            actual_output="valid triage",
            additional_metadata={"prompt_tokens": 3000, "completion_tokens": 400},
        )
        if assert_test:
            assert_test(tc, [metric], run_async=False)
        else:
            score = metric.measure(tc)
            assert score == 1.0
            assert metric.is_successful() is True
            assert "within token budget" in (metric.reason or "")

    def test_prompt_budget_exceeded(self) -> None:
        """Metric fails with score 0.0 when prompt tokens exceed ceiling."""
        metric = TokenBudgetMetric(
            agent_type="triage", max_prompt_tokens=5000, max_completion_tokens=1000
        )
        tc = LLMTestCase(
            input="triage request",
            actual_output="valid triage",
            additional_metadata={"prompt_tokens": 8000, "completion_tokens": 400},
        )
        score = metric.measure(tc)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "Prompt tokens (8,000) exceeded budget" in (metric.reason or "")

    def test_completion_budget_exceeded(self) -> None:
        """Metric fails with score 0.0 when completion tokens exceed ceiling."""
        metric = TokenBudgetMetric(
            agent_type="update_subagent", max_prompt_tokens=8000, max_completion_tokens=500
        )
        tc = LLMTestCase(
            input="update request",
            actual_output="update code",
            additional_metadata={"prompt_tokens": 4000, "completion_tokens": 1200},
        )
        score = metric.measure(tc)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "Completion tokens (1,200) exceeded budget" in (metric.reason or "")

    @pytest.mark.asyncio
    async def test_async_measure(self) -> None:
        """Async measure delegates correctly to synchronous measurement."""
        metric = TokenBudgetMetric(agent_type="report")
        tc = LLMTestCase(
            input="report input",
            actual_output="report text",
            additional_metadata={"prompt_tokens": 2000, "completion_tokens": 500},
        )
        score = await metric.a_measure(tc)
        assert score == 1.0


class TestLatencySLAMetricUnit:
    """Unit tests for the LatencySLAMetric BaseMetric implementation."""

    def test_within_sla_passes(self) -> None:
        """Metric succeeds with score 1.0 when completion time is within SLA ceiling."""
        metric = LatencySLAMetric(agent_type="update_subagent", max_latency_seconds=60.0)
        tc = LLMTestCase(
            name="LatencySLA [update_subagent - within SLA]",
            input="update",
            actual_output="done",
            completion_time=25.4,
        )
        if assert_test:
            assert_test(tc, [metric], run_async=False)
        else:
            score = metric.measure(tc)
            assert score == 1.0
            assert metric.is_successful() is True
            assert "satisfied latency SLA" in (metric.reason or "")

    def test_sla_breached_fails(self) -> None:
        """Metric fails with score 0.0 when completion time breaches ceiling."""
        metric = LatencySLAMetric(agent_type="triage", max_latency_seconds=30.0)
        tc = LLMTestCase(
            input="triage",
            actual_output="done",
            completion_time=45.2,
        )
        score = metric.measure(tc)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "Latency SLA breached" in (metric.reason or "")

    @pytest.mark.asyncio
    async def test_async_measure(self) -> None:
        """Async measure delegates correctly."""
        metric = LatencySLAMetric(agent_type="report", max_latency_seconds=15.0)
        tc = LLMTestCase(input="report", actual_output="done", completion_time=5.0)
        score = await metric.a_measure(tc)
        assert score == 1.0


class TestToolCallBudgetMetricUnit:
    """Unit tests for the ToolCallBudgetMetric implementation."""

    def test_under_warning_threshold_full_score(self) -> None:
        """<= 75% budget used yields a clean pass (score = 1.0)."""
        metric = ToolCallBudgetMetric(max_tool_rounds=24, warning_percentage=0.75)
        tools = [ToolCall(name=f"tool_{i}") for i in range(12)]
        tc = LLMTestCase(
            name="ToolCallBudget [clean pass under 75%]",
            input="run",
            actual_output="done",
            tools_called=tools,
        )
        if assert_test:
            assert_test(tc, [metric], run_async=False)
        else:
            score = metric.measure(tc)
            assert score == 1.0
            assert metric.is_successful() is True
            assert "satisfied" in (metric.reason or "")

    def test_warning_threshold_triggers_score_penalty(self) -> None:
        """75% - 100% budget used yields a score penalty of 0.75 (warning)."""
        metric = ToolCallBudgetMetric(max_tool_rounds=24, warning_percentage=0.75, threshold=0.70)
        tools = [ToolCall(name=f"tool_{i}") for i in range(20)]
        tc = LLMTestCase(
            name="ToolCallBudget [penalty warning at 75-100%]",
            input="run",
            actual_output="done",
            tools_called=tools,
        )
        if assert_test:
            assert_test(tc, [metric], run_async=False)
        else:
            score = metric.measure(tc)
            assert score == 0.75
            assert metric.is_successful() is True
            assert "Score penalty applied" in (metric.reason or "")

    def test_exceeding_max_budget_fails(self) -> None:
        """> 100% budget used yields score 0.0 (hard budget failure)."""
        metric = ToolCallBudgetMetric(max_tool_rounds=24, warning_percentage=0.75)
        tools = [ToolCall(name=f"tool_{i}") for i in range(26)]
        tc = LLMTestCase(input="run", actual_output="done", tools_called=tools)
        score = metric.measure(tc)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "Tool call budget exceeded" in (metric.reason or "")


# ---------------------------------------------------------------------------
# Dynamic Trajectory-Driven Business Rules Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestBusinessRules:
    """Phase 5 business rules and SLA verification across historical trajectories."""

    @classmethod
    def _get_dynamic_trajectories(
        cls, loader: TrajectoryLoader, max_count: int = 25
    ) -> list[TrajectoryDocument]:
        """Discover trajectories dynamically while skipping abnormally large debug dumps."""
        all_paths = loader.get_trajectory_paths()
        # Filter for standard trajectory files (< 10MB) to keep test execution swift and deterministic
        suitable_paths = [p for p in all_paths if p.stat().st_size < 10 * 1024 * 1024]
        target_paths = suitable_paths[:max_count]
        return [loader.load_by_path(p) for p in target_paths]

    @pytest.mark.parametrize(
        "agent_type",
        ["triage", "update_subagent", "workaround_subagent", "qa_critic", "report"],
    )
    def test_token_budgets(
        self,
        trajectory_loader: TrajectoryLoader,
        agent_type: str,
    ) -> None:
        """Verify that agent LLM spans across trajectories stay within empirical token budgets."""
        trajectories = self._get_dynamic_trajectories(trajectory_loader)
        if not trajectories:
            pytest.skip("No trajectory files discovered in data/trajectories.")

        metric = TokenBudgetMetric(agent_type=agent_type)
        evaluated_spans = 0
        passed_spans = 0
        violations: list[str] = []

        for doc in trajectories:
            agent_spans = doc.spans_for_agent(agent_type)
            for span in agent_spans:
                p_tok, c_tok = extract_token_usage(span)
                if p_tok == 0 and c_tok == 0 and not span.is_llm:
                    continue

                evaluated_spans += 1
                tc = LLMTestCase(
                    input=f"Trajectory span: {span.name}",
                    actual_output=str(span.outputs or ""),
                    additional_metadata={
                        "agent_type": agent_type,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "trace_id": doc.trace_id,
                        "run_id": span.run_id,
                    },
                )
                score = metric.measure(tc)
                if score >= metric.threshold:
                    passed_spans += 1
                else:
                    violations.append(
                        f"[{doc.trace_id[:8]}/{span.name}] Prompt={p_tok:,}, Completion={c_tok:,} -> {metric.reason}"
                    )

        if evaluated_spans == 0:
            pytest.skip(f"No evaluated token spans found for agent '{agent_type}'.")

        pass_rate = passed_spans / evaluated_spans
        logger.info(
            "Token Budget Evaluation [%s]: %d/%d passed (%.1f%%)",
            agent_type,
            passed_spans,
            evaluated_spans,
            pass_rate * 100,
        )

        assert pass_rate >= 0.80, (
            f"Token budget pass rate for '{agent_type}' was {pass_rate:.1%} (< 80%). Violations: {violations[:5]}"
        )

    @pytest.mark.parametrize(
        "agent_type",
        ["triage", "update_subagent", "workaround_subagent", "qa_critic", "report"],
    )
    def test_latency_sla(
        self,
        trajectory_loader: TrajectoryLoader,
        agent_type: str,
    ) -> None:
        """Verify that agent spans across trajectories stay within empirical latency SLA ceilings."""
        trajectories = self._get_dynamic_trajectories(trajectory_loader)
        if not trajectories:
            pytest.skip("No trajectory files discovered in data/trajectories.")

        metric = LatencySLAMetric(agent_type=agent_type)
        evaluated_spans = 0
        passed_spans = 0
        violations: list[str] = []

        for doc in trajectories:
            agent_spans = doc.spans_for_agent(agent_type)
            for span in agent_spans:
                if span.duration_seconds is None or span.duration_seconds <= 0:
                    continue

                evaluated_spans += 1
                tc = LLMTestCase(
                    input=f"Trajectory span: {span.name}",
                    actual_output=str(span.outputs or ""),
                    completion_time=span.duration_seconds,
                    additional_metadata={
                        "agent_type": agent_type,
                        "duration_seconds": span.duration_seconds,
                        "trace_id": doc.trace_id,
                        "run_id": span.run_id,
                    },
                )
                score = metric.measure(tc)
                if score >= metric.threshold:
                    passed_spans += 1
                else:
                    violations.append(
                        f"[{doc.trace_id[:8]}/{span.name}] Duration={span.duration_seconds:.2f}s -> {metric.reason}"
                    )

        if evaluated_spans == 0:
            pytest.skip(f"No evaluated latency spans found for agent '{agent_type}'.")

        pass_rate = passed_spans / evaluated_spans
        logger.info(
            "Latency SLA Evaluation [%s]: %d/%d passed (%.1f%%)",
            agent_type,
            passed_spans,
            evaluated_spans,
            pass_rate * 100,
        )

        assert pass_rate >= 0.80, (
            f"Latency SLA pass rate for '{agent_type}' was {pass_rate:.1%} (< 80%). Violations: {violations[:5]}"
        )

    def test_tool_call_budget(
        self,
        trajectory_loader: TrajectoryLoader,
    ) -> None:
        """Verify that subagent workers do not exhaust the configured tool call budget."""
        trajectories = self._get_dynamic_trajectories(trajectory_loader)
        if not trajectories:
            pytest.skip("No trajectory files discovered in data/trajectories.")

        metric = ToolCallBudgetMetric(max_tool_rounds=24, warning_percentage=0.75)
        evaluated_workers = 0
        passed_workers = 0
        penalized_workers = 0
        violations: list[str] = []

        for doc in trajectories:
            worker_spans = [
                s
                for s in doc.spans
                if ("update_subagent" in s.name.lower() or "workaround_subagent" in s.name.lower())
                and s.run_type == "chain"
            ]

            for span in worker_spans:
                child_tools = extract_tool_calls(doc.spans, parent_span_id=span.run_id)
                if not child_tools:
                    continue

                evaluated_workers += 1
                tc = LLMTestCase(
                    input=f"Worker run: {span.name}",
                    actual_output=str(span.outputs or ""),
                    tools_called=child_tools,
                    additional_metadata={"trace_id": doc.trace_id, "run_id": span.run_id},
                )
                score = metric.measure(tc)
                if score >= metric.threshold:
                    passed_workers += 1
                    if score < 1.0:
                        penalized_workers += 1
                else:
                    violations.append(
                        f"[{doc.trace_id[:8]}/{span.name}] Rounds={len(child_tools)} -> {metric.reason}"
                    )

        if evaluated_workers == 0:
            pytest.skip("No worker execution chains with child tools discovered.")

        pass_rate = passed_workers / evaluated_workers
        logger.info(
            "Tool Call Budget Evaluation: %d/%d passed (%.1f%%), %d penalized with warnings",
            passed_workers,
            evaluated_workers,
            pass_rate * 100,
            penalized_workers,
        )

        assert pass_rate >= 0.85, (
            f"Tool call budget pass rate was {pass_rate:.1%} (< 85%). Violations: {violations[:5]}"
        )
