"""Data models for evaluation runs, test cases, and metric results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """Evaluation metric result record."""

    metric_name: str
    score: float
    threshold: float = 0.70
    success: bool = True
    reason: str | None = None
    evaluation_model: str | None = None
    verbose_logs: str | None = None


class EvalTestCaseRecord(BaseModel):
    """Evaluation test case record representing an LLMTestCase outcome."""

    case_id: str | None = None
    test_name: str
    suite: str = "general"
    status: str = "PASSED"  # PASSED, FAILED, SKIPPED, ERROR
    input_text: str = ""
    actual_output: str = ""
    expected_output: str | None = None
    context_text: str | None = None
    retrieval_context: str | None = None
    latency_seconds: float = 0.0
    cost: float = 0.0
    error_message: str | None = None
    additional_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricRecord] = Field(default_factory=list)


class EvalRunRecord(BaseModel):
    """Evaluation run record encapsulating a full evaluation session."""

    run_id: str
    timestamp: str  # ISO format string
    suite_name: str = "eval_suite"
    judge_model: str = "gpt-4o"
    is_live: bool = False
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    duration_seconds: float = 0.0
    total_cost: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    test_cases: list[EvalTestCaseRecord] = Field(default_factory=list)
