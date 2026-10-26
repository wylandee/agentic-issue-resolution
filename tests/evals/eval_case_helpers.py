"""Shared case and DeepEval conversion helpers for the evaluation suites.

The helpers deliberately keep fixture observations and live observations on
separate paths.  A builder receives a ``ReplayCapture`` when a production
node was executed; without one it reads only ``offline_fixture`` so a golden
case can never accidentally turn its historical observation into a live
result.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

from remediation_engine.evals.trajectories import ToolCall
from tests.evals.golden_schema import offline_fixture
from tests.evals.replay_harness import ReplayCapture, format_tool_trace


def as_output_text(value: Any) -> str:
    """Serialize an evaluator value into stable text for an LLM test case."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def observations(
    case: Mapping[str, Any], capture: ReplayCapture | None = None
) -> tuple[str, list[dict[str, Any]], str]:
    """Return actual output, actual tools, and their provenance label.

    Args:
        case: Canonical golden case.
        capture: Optional result from a production live replay.

    Returns:
        A tuple ``(actual_output, actual_tools, replay_source)``.  The
        fixture branch is intentionally the only offline observation source.
    """
    if capture is not None:
        return capture.actual_output, list(capture.actual_tools), "production_live"
    fixture = offline_fixture(case)
    output = fixture.get("actual_output", "")
    tools = fixture.get("actual_tools", [])
    return as_output_text(output), list(tools) if isinstance(tools, list) else [], "offline_fixture"


def make_tool_calls(raw_calls: Any, *, component: str) -> list[ToolCall]:
    """Convert canonical tool-call records into the local DeepEval adapter."""
    if not isinstance(raw_calls, list):
        raise TypeError(f"{component} tool calls must be a list")
    converted: list[ToolCall] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, Mapping):
            raise TypeError(f"{component} tool call {index} must be an object")
        args = call.get("args", {})
        if not isinstance(args, Mapping):
            raise TypeError(f"{component} tool call {index} args must be an object")
        converted.append(
            ToolCall(
                name=str(call.get("name", "")),
                input_parameters=dict(args),
                output=call.get("output", ""),
            )
        )
    return converted


def expected_tools(case: Mapping[str, Any], *, component: str) -> list[ToolCall]:
    """Convert the independent canonical expected-tool trace."""
    return make_tool_calls(case.get("expected_tools", []), component=component)


def case_metadata(
    case: Mapping[str, Any],
    *,
    component: str,
    replay_source: str,
    actual_tools: list[dict[str, Any]],
    capture: ReplayCapture | None = None,
) -> dict[str, Any]:
    """Build common metadata without mixing expected and observed traces."""
    metadata: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "eval_type": case.get("eval_type", component),
        "component": component,
        "replay_source": replay_source,
        "expected_tools": case.get("expected_tools", []),
        "observed_tool_count": len(actual_tools),
    }
    for key in (
        "golden_kind",
        "language",
        "ecosystem",
        "provenance",
        "provenance_status",
        "evidence_status",
        "evaluation_note",
        "expected_completion_pass",
        "task_completion_applicable",
        "expected_tool_correctness_pass",
        "tool_correctness_applicable",
        "expected_qa_verdict",
        "attempt_id",
        "task_revision",
        "target_package_name",
        "selected_version",
        "dependency_type",
        "action_status",
        "changed_files",
        "expected_strategy",
        "expected_fixed_version",
    ):
        if key in case:
            metadata[key] = case[key]
    if capture is not None:
        metadata.update(
            {
                "attempt_id": capture.attempt_id,
                "task_revision": capture.task_revision,
                "replay_errors": list(capture.errors),
                "external_call_count": len(capture.external_calls),
                "input_tokens": capture.input_tokens,
                "output_tokens": capture.output_tokens,
                "total_tokens": capture.total_tokens,
                "cached_input_tokens": capture.cached_input_tokens,
                "token_cost": capture.token_cost,
                "token_usage_available": any(
                    value is not None
                    for value in (
                        capture.input_tokens,
                        capture.output_tokens,
                        capture.total_tokens,
                    )
                ),
                "cache_usage_available": capture.cached_input_tokens is not None,
            }
        )
    return metadata


def context_strings(case: Mapping[str, Any], *extra: str) -> list[str]:
    """Return canonical context strings plus non-mutating evaluator context."""
    context = case.get("context", [])
    values = [str(item) for item in context] if isinstance(context, list) else []
    return values + [item for item in extra if item]


def actual_output_with_tools(output: Any, tools: list[dict[str, Any]]) -> str:
    """Append a captured trace to a typed production result for judging."""
    rendered = as_output_text(output)
    return f"{rendered}\n\n{format_tool_trace(tools)}"


def task_completion_input(case: Mapping[str, Any]) -> str:
    """Put the complete task and its acceptance criteria in DeepEval's input.

    DeepEval 2.x TaskCompletionMetric reads ``input`` and ``actual_output``;
    expected output and context fields do not reliably reach its task extractor.
    Historical provenance is deliberately omitted from the judge prompt.
    """
    sections = [str(case.get("input", "")).strip()]
    objective = str(case.get("completion_task", "")).strip()
    expected = str(case.get("expected_output", "")).strip()
    if objective and objective not in sections[0]:
        sections.append(f"Task objective: {objective}")
    if expected:
        sections.append(f"Required completion criteria: {expected}")
    return "\n\n".join(section for section in sections if section)


_METRIC_FAILURES: ContextVar[list[str] | None] = ContextVar("eval_metric_failures", default=None)


def set_pytest_item_identity(test_case: Any) -> None:
    """Attach the active pytest node ID when the case accepts metadata."""
    nodeid = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
    if not nodeid:
        return
    metadata = dict(getattr(test_case, "additional_metadata", None) or {})
    metadata["pytest_nodeid"] = nodeid
    with suppress(AttributeError, TypeError):
        test_case.additional_metadata = metadata


@contextmanager
def collect_metric_assertions() -> Iterator[None]:
    """Run every applicable metric before raising accumulated threshold failures."""
    failures: list[str] = []
    token = _METRIC_FAILURES.set(failures)
    try:
        yield
    finally:
        _METRIC_FAILURES.reset(token)
    assert not failures, "\n\n".join(failures)


def measure_expected(
    metric: Any, test_case: Any, *, expected_pass: bool, case_id: str, label: str, register: Any
) -> None:
    """Register positive and negative measurements, deferring only assertion failures.

    Provider and metric execution errors always propagate. A negative expectation
    is satisfied only by a measured threshold failure, never by an execution error.
    """
    set_pytest_item_identity(test_case)
    try:
        if register is None:
            metric.measure(test_case)
        else:
            try:
                register(test_case, [metric], run_async=False)
            except AssertionError:
                if expected_pass or metric.is_successful() or getattr(metric, "error", None):
                    raise
        if getattr(metric, "error", None):
            raise RuntimeError(str(metric.error))
        assert bool(metric.is_successful()) is expected_pass, (
            f"Case {case_id!r} expected {label} to {'pass' if expected_pass else 'fail'}."
        )
    except AssertionError as exc:
        failures = _METRIC_FAILURES.get()
        if failures is None:
            raise
        failures.append(str(exc))


def measure_diagnostic(metric: Any, test_case: Any, *, register: Any) -> None:
    """Measure and register a metric whose threshold is diagnostic only."""
    set_pytest_item_identity(test_case)
    try:
        if register is None:
            metric.measure(test_case)
        else:
            try:
                register(test_case, [metric], run_async=False)
            except AssertionError:
                if getattr(metric, "error", None) or getattr(metric, "score", None) is None:
                    raise
        if getattr(metric, "error", None):
            raise RuntimeError(str(metric.error))
        if getattr(metric, "score", None) is None:
            raise RuntimeError("Diagnostic metric produced no score")
    except AssertionError:
        # DeepEval's assert_test raises after persisting a below-threshold
        # result. The numeric score remains useful as a diagnostic observation.
        if getattr(metric, "error", None) or getattr(metric, "score", None) is None:
            raise


def measure_and_record_metric(
    metric: Any,
    test_case: Any,
    *,
    case_id: str,
    suite: str,
    judge_model: str,
) -> bool:
    """Measure synchronously and persist one DeepEval metric with pytest.

    Args:
        metric: DeepEval metric already configured with a bounded judge.
        test_case: Metric input and independently prepared evidence.
        case_id: Stable golden case identifier.
        suite: Dashboard suite name.
        judge_model: Human-readable judge model identifier.

    Returns:
        Whether the measured score meets its threshold.

    Raises:
        RuntimeError: The provider failed or returned an invalid score.
    """
    from remediation_engine.evals.models import EvalTestCaseRecord, MetricRecord
    from tests.evals.conftest import record_eval_result

    started = time.monotonic()
    try:
        metric.measure(test_case)
        if getattr(metric, "error", None):
            raise RuntimeError(str(metric.error))
        score = float(metric.score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Judge returned a non-finite or out-of-range score")
    except Exception as exc:
        record_eval_result(
            EvalTestCaseRecord(
                case_id=case_id,
                test_name=str(getattr(test_case, "name", case_id)),
                suite=suite,
                status="ERROR",
                input_text=str(getattr(test_case, "input", "") or ""),
                actual_output=str(getattr(test_case, "actual_output", "") or ""),
                expected_output=getattr(test_case, "expected_output", None),
                retrieval_context="\n---\n".join(
                    str(value) for value in (getattr(test_case, "retrieval_context", None) or [])
                )
                or None,
                latency_seconds=time.monotonic() - started,
                error_message=str(exc),
                additional_metadata=dict(getattr(test_case, "additional_metadata", {}) or {}),
            )
        )
        raise RuntimeError(f"{getattr(metric, '__name__', type(metric).__name__)}: {exc}") from exc

    successful = bool(metric.is_successful())
    metadata = dict(getattr(test_case, "additional_metadata", {}) or {})
    record_eval_result(
        EvalTestCaseRecord(
            case_id=case_id,
            test_name=str(getattr(test_case, "name", case_id)),
            suite=suite,
            status="PASSED" if successful else "FAILED",
            input_text=str(getattr(test_case, "input", "") or ""),
            actual_output=str(getattr(test_case, "actual_output", "") or ""),
            expected_output=getattr(test_case, "expected_output", None),
            retrieval_context="\n---\n".join(
                str(value) for value in (getattr(test_case, "retrieval_context", None) or [])
            )
            or None,
            latency_seconds=time.monotonic() - started,
            cost=float(getattr(metric, "evaluation_cost", None) or 0),
            additional_metadata=metadata,
            metrics=[
                MetricRecord(
                    metric_name=str(getattr(metric, "__name__", type(metric).__name__)),
                    score=score,
                    threshold=float(metric.threshold),
                    success=successful,
                    reason=getattr(metric, "reason", None),
                    evaluation_model=getattr(metric, "evaluation_model", None) or judge_model,
                    verbose_logs=getattr(metric, "verbose_logs", None),
                )
            ],
        )
    )
    return successful
