"""Shared case and DeepEval conversion helpers for the evaluation suites.

The helpers deliberately keep fixture observations and live observations on
separate paths.  A builder receives a ``ReplayCapture`` when a production
node was executed; without one it reads only ``offline_fixture`` so a golden
case can never accidentally turn its historical observation into a live
result.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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


def case_metadata_dict(test_case: Any) -> dict[str, Any]:
    """Return a copy of metadata across supported DeepEval case shapes.

    Args:
        test_case: A classic ``LLMTestCase`` or a serialized API case. Older
            cases expose ``additional_metadata``; current API cases expose
            ``metadata``.

    Returns:
        A merged metadata dictionary, with legacy case values taking
        precedence if both attributes are present. The input case is not
        mutated.
    """
    metadata: dict[str, Any] = {}
    for attribute in ("metadata", "additional_metadata"):
        value = getattr(test_case, attribute, None)
        if isinstance(value, Mapping):
            metadata.update(value)
    return metadata


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

    TaskCompletionMetric reads ``input`` and ``actual_output``; expected output
    and context fields do not reliably reach its task extractor.
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
    metadata = case_metadata_dict(test_case)
    metadata["pytest_nodeid"] = nodeid
    for attribute in ("additional_metadata", "metadata"):
        if not hasattr(test_case, attribute):
            continue
        try:
            setattr(test_case, attribute, metadata)
            break
        except (AttributeError, TypeError):
            continue


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
