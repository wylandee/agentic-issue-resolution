"""Shared case and DeepEval conversion helpers for the evaluation suites.

The helpers deliberately keep fixture observations and live observations on
separate paths.  A builder receives a ``ReplayCapture`` when a production
node was executed; without one it reads only ``offline_fixture`` so a golden
case can never accidentally turn its historical observation into a live
result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tests.evals.adapters import ToolCall
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
        "provenance",
        "provenance_status",
        "evidence_status",
        "evaluation_note",
        "expected_completion_pass",
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
