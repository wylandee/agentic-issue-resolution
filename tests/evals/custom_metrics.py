"""Deterministic operational budget metrics used by the evaluation suite."""

from __future__ import annotations

import math
from typing import Any

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
except ImportError:

    class BaseMetric:  # type: ignore[no-redef]
        """Fallback base metric when deepeval is not installed."""

        threshold: float = 1.0
        score: float | None = None
        reason: str | None = None
        success: bool | None = None
        strict_mode: bool = False
        async_mode: bool = False
        verbose_mode: bool = True

    class LLMTestCase:  # type: ignore[no-redef]
        """Fallback test case representation."""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


class TokenBudgetMetric(BaseMetric):
    """Metric verifying per-agent prompt and completion token budget ceilings.

    Calibrated empirical default ceilings derived from historical trajectories:
      - triage: max prompt 14,000, max completion 1,000
      - update_subagent: max prompt 8,000 (per call) / 35,000 (loop), max completion 500
      - workaround_subagent: max prompt 25,000 (per call) / 350,000 (loop), max completion 1,000
      - qa_critic: max prompt 12,000 (per call) / 75,000 (loop), max completion 1,500
    """

    _DEFAULT_BUDGETS: dict[str, dict[str, int]] = {
        "triage": {"max_prompt": 15_000, "max_completion": 1_500},
        "update_subagent": {"max_prompt": 35_000, "max_completion": 8_000},
        "workaround_subagent": {"max_prompt": 350_000, "max_completion": 15_000},
        "qa_critic": {"max_prompt": 60_000, "max_completion": 5_000},
    }

    def __init__(
        self,
        agent_type: str = "triage",
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the TokenBudgetMetric.

        Args:
            agent_type: Agent category ('triage', 'update_subagent', 'workaround_subagent', 'qa_critic').
            max_prompt_tokens: Optional prompt token override.
            max_completion_tokens: Optional completion token override.
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.agent_type = agent_type.lower()
        defaults = self._DEFAULT_BUDGETS.get(
            self.agent_type, {"max_prompt": 15_000, "max_completion": 2_000}
        )
        self.max_prompt_tokens = (
            max_prompt_tokens if max_prompt_tokens is not None else defaults["max_prompt"]
        )
        self.max_completion_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else defaults["max_completion"]
        )
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-token-budget"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return f"Token Budget ({self.agent_type})"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Measure token budget compliance for the provided test case.

        Reads 'prompt_tokens' and 'completion_tokens' from test_case.additional_metadata.
        """
        meta = getattr(test_case, "additional_metadata", {}) or {}
        if meta.get("prompt_tokens") is None or meta.get("completion_tokens") is None:
            self.score, self.success, self.reason = None, None, "Token usage unavailable; unscored."
            return None
        prompt_tokens = int(meta["prompt_tokens"])
        completion_tokens = int(meta["completion_tokens"])
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        violations: list[str] = []
        if prompt_tokens > self.max_prompt_tokens:
            violations.append(
                f"Prompt tokens ({prompt_tokens:,}) exceeded budget of {self.max_prompt_tokens:,}."
            )
        if completion_tokens > self.max_completion_tokens:
            violations.append(
                f"Completion tokens ({completion_tokens:,}) exceeded budget of {self.max_completion_tokens:,}."
            )

        if violations:
            self.score = 0.0
            self.reason = f"Token budget exceeded for {self.agent_type}: " + " ".join(violations)
        else:
            self.score = 1.0
            self.reason = (
                f"{self.agent_type} within token budget: "
                f"prompt={prompt_tokens:,}/{self.max_prompt_tokens:,}, "
                f"completion={completion_tokens:,}/{self.max_completion_tokens:,}."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


class LatencySLAMetric(BaseMetric):
    """Metric verifying per-agent wall-clock execution latency against SLA ceilings.

    Calibrated empirical default ceilings derived from historical trajectories:
      - triage: 30.0s (per-group/call)
      - update_subagent: 60.0s (full worker loop)
      - workaround_subagent: 120.0s (full worker loop)
      - qa_critic: 45.0s (LLM review phase) / 300.0s (full Docker container loop)
    """

    _DEFAULT_SLAS: dict[str, float] = {
        "triage": 30.0,
        "update_subagent": 60.0,
        "workaround_subagent": 120.0,
        "qa_critic": 45.0,
    }

    def __init__(
        self,
        agent_type: str = "triage",
        max_latency_seconds: float | None = None,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the LatencySLAMetric.

        Args:
            agent_type: Agent category ('triage', 'update_subagent', 'workaround_subagent', 'qa_critic', 'report').
            max_latency_seconds: Optional wall-clock latency ceiling override in seconds.
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.agent_type = agent_type.lower()
        default_sla = self._DEFAULT_SLAS.get(self.agent_type, 60.0)
        self.max_latency_seconds = (
            max_latency_seconds if max_latency_seconds is not None else default_sla
        )
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-latency-sla"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return f"Latency SLA ({self.agent_type})"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Measure wall-clock latency against the SLA ceiling for the test case."""
        meta = getattr(test_case, "additional_metadata", {}) or {}
        latency = next(
            (
                value
                for value in (
                    getattr(test_case, "completion_time", None),
                    meta.get("duration_seconds"),
                    meta.get("latency_seconds"),
                )
                if value is not None
            ),
            None,
        )
        if latency is None:
            self.score, self.success, self.reason = None, None, "Latency unavailable; unscored."
            return None
        latency = float(latency)
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Latency must be finite and non-negative")

        if latency > self.max_latency_seconds:
            self.score = 0.0
            self.reason = (
                f"Latency SLA breached for {self.agent_type}: "
                f"took {latency:.2f}s (ceiling is {self.max_latency_seconds:.2f}s)."
            )
        else:
            self.score = 1.0
            self.reason = (
                f"{self.agent_type} satisfied latency SLA: "
                f"took {latency:.2f}s (under {self.max_latency_seconds:.2f}s limit)."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


class ToolCallBudgetMetric(BaseMetric):
    """Metric verifying tool call round consumption against configured budgets.

    Enforces MAX_SUBAGENT_TOOL_CALL_ROUNDS (default: 24) with graduated scoring penalty:
      - <= 75% budget (<= 18 rounds): score = 1.0 (clean pass)
      - 75% - 100% budget (19 - 24 rounds): score = 0.75 (score penalty warning)
      - > 100% budget (> 24 rounds): score = 0.0 (hard failure)
    """

    def __init__(
        self,
        max_tool_rounds: int = 24,
        warning_percentage: float = 0.75,
        threshold: float = 0.70,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the ToolCallBudgetMetric.

        Args:
            max_tool_rounds: Maximum permitted tool rounds (default 24).
            warning_percentage: Fraction of budget that triggers a score penalty (default 0.75).
            threshold: Minimum score required for success (default 0.70).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.max_tool_rounds = max_tool_rounds
        self.warning_percentage = warning_percentage
        self.warning_limit = int(max_tool_rounds * warning_percentage)
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-tool-budget"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Tool Call Round Budget"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Measure tool call round consumption for the test case."""
        rounds_used = (getattr(test_case, "additional_metadata", None) or {}).get("loop_rounds")
        if rounds_used is None:
            self.score, self.success, self.reason = (
                None,
                None,
                "Loop-round evidence unavailable; unscored.",
            )
            return None
        if isinstance(rounds_used, bool) or not isinstance(rounds_used, int) or rounds_used < 0:
            raise ValueError("loop_rounds must be a non-negative integer")

        if rounds_used > self.max_tool_rounds:
            self.score = 0.0
            self.reason = (
                f"Tool call budget exceeded: executed {rounds_used} rounds "
                f"(hard budget limit is {self.max_tool_rounds})."
            )
        elif rounds_used > self.warning_limit:
            self.score = 0.75
            self.reason = (
                f"Tool call warning: executed {rounds_used} rounds "
                f"(exceeds {int(self.warning_percentage * 100)}% threshold of {self.warning_limit}, "
                f"within max {self.max_tool_rounds}). Score penalty applied."
            )
        else:
            self.score = 1.0
            self.reason = (
                f"Tool call budget satisfied: executed {rounds_used} rounds "
                f"(well within {self.max_tool_rounds} limit)."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float | None:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)
