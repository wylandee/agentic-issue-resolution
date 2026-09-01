"""DeepEval evaluation suite for the update subagent.

The update worker is evaluated only on the two requested dimensions:

* ``ToolCorrectnessMetric`` checks the exact ordered combined-tool calls.
* ``TaskCompletionMetric`` checks whether the worker actually completed the
  supervisor instruction (or correctly failed to complete it after surrender).

The cases are replay fixtures, not live remediation runs.  The live flag is
therefore required for the LLM-judged task-completion metric, while tool
correctness remains deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.evals.adapters import ToolCall
from tests.evals.conftest import EvalSettings

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        TaskCompletionMetric as DeepEvalTaskCompletionMetric,
    )
    from deepeval.metrics import (
        ToolCorrectnessMetric as DeepEvalToolCorrectnessMetric,
    )
    from deepeval.test_case import LLMTestCase, ToolCallParams

    HAS_DEEPEVAL = True
except ImportError:
    from tests.evals.adapters import DeepEvalLLMTestCase as LLMTestCase  # type: ignore[assignment]

    HAS_DEEPEVAL = False
    assert_test = None  # type: ignore[assignment]
    DeepEvalTaskCompletionMetric = None  # type: ignore[assignment,misc]
    DeepEvalToolCorrectnessMetric = None  # type: ignore[assignment,misc]
    ToolCallParams = None  # type: ignore[assignment,misc]


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "update_subagent_cases.json"


def _load_update_cases() -> list[dict[str, Any]]:
    """Load the dedicated update-subagent golden dataset.

    Returns:
        A list of update-subagent case dictionaries.  Malformed or missing
        datasets produce an empty list so pytest can report the unavailable
        optional evaluation data without importing unrelated suites.
    """
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [case for case in data if case.get("eval_type") == "update_subagent"]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if case.get("eval_type") == "update_subagent"]
    return []


_UPDATE_CASES = _load_update_cases()
_UPDATE_CASE_IDS = [
    case.get("case_id", f"case_{index}") for index, case in enumerate(_UPDATE_CASES)
]


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert serialized tool-call records to DeepEval tool calls."""
    return [
        ToolCall(
            name=str(tool_call.get("name", "")),
            input_parameters=tool_call.get("args", {}) or {},
            output=tool_call.get("output", ""),
        )
        for tool_call in raw_calls
    ]


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render the replayed tool trace in the worker's observable outcome."""
    if not raw_calls:
        return "Observed tool trace: none"
    lines = ["Observed tool trace:"]
    for index, tool_call in enumerate(raw_calls, start=1):
        args = json.dumps(tool_call.get("args", {}) or {}, sort_keys=True)
        output = str(tool_call.get("output", ""))
        lines.append(f"{index}. {tool_call.get('name', '')}({args}) -> {output}")
    return "\n".join(lines)


def build_update_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Construct a DeepEval test case from one update golden.

    Args:
        case: Serialized update golden containing the supervisor instruction,
            observed tool calls, expected tool calls, and outcome metadata.

    Returns:
        An ``LLMTestCase`` suitable for both requested DeepEval metrics.

    Raises:
        TypeError: If the golden's tool-call fields are not lists.
    """
    tool_calls_raw = case.get("tool_calls", [])
    expected_tool_calls_raw = case.get("expected_tool_calls", tool_calls_raw)
    if not isinstance(tool_calls_raw, list) or not isinstance(expected_tool_calls_raw, list):
        raise TypeError("Update golden tool_calls and expected_tool_calls must be lists.")

    tools_called = _make_tool_calls(tool_calls_raw)
    expected_tools = _make_tool_calls(expected_tool_calls_raw)
    instruction = str(case.get("supervisor_instruction", ""))
    target_package = str(case.get("target_package_name", ""))
    action_status = str(case.get("action_status", "APPLIED"))
    changed_files = case.get("changed_files", []) or []
    final_output = str(case.get("final_output", ""))
    if not final_output:
        final_output = (
            f"Status: {action_status}; package={target_package}; "
            f"changed_files={', '.join(changed_files) if changed_files else 'none'}."
        )

    actual_output = f"{final_output}\n\n{_format_tool_trace(tool_calls_raw)}"
    expected_output = str(
        case.get(
            "expected_output",
            "Report whether the requested npm dependency transaction was actually completed.",
        )
    )
    provenance = str(case.get("provenance", ""))
    evaluation_note = str(case.get("evaluation_note", ""))
    repository_map = str(case.get("repository_map", "(workspace map unavailable)"))
    completion_task = str(
        case.get(
            "completion_task",
            "Execute the supervisor instruction and do not claim success unless the dependency transaction completed.",
        )
    )

    metadata = {
        "case_id": case.get("case_id"),
        "eval_type": "update_subagent",
        "golden_kind": case.get("golden_kind"),
        "proposal": case.get("proposal"),
        "provenance": provenance,
        "is_retry": bool(case.get("is_retry", False)),
        "target_package_name": target_package,
        "selected_version": case.get("selected_version"),
        "dependency_type": case.get("dependency_type"),
        "manifest_path": case.get("manifest_path", "package.json"),
        "changed_files": changed_files,
        "action_status": action_status,
        "expected_completion_pass": bool(case.get("expected_completion_pass", True)),
    }

    return LLMTestCase(
        name=f"{case.get('case_id')} [Update Subagent]",
        input=instruction,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[
            f"Completion task:\n{completion_task}",
            f"Golden provenance:\n{provenance}",
            f"Evaluation note:\n{evaluation_note}",
            f"Deterministic repository map:\n{repository_map}",
        ],
        tools_called=tools_called,
        expected_tools=expected_tools,
        additional_metadata=metadata,
    )


def _measure_expected_completion(
    metric: Any,
    test_case: LLMTestCase,
    *,
    expected_pass: bool,
    case_id: str,
) -> None:
    """Run TaskCompletionMetric and support intentional surrender goldens."""
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} did not complete the task."
        return

    # A surrender is intentionally not a completed dependency update.  Use
    # the metric directly so pytest can assert the expected negative outcome
    # without making an expected failure look like a broken test.
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Case {case_id!r} was expected to remain incomplete after surrender, "
        f"but scored {getattr(metric, 'score', None)}."
    )


def _measure_expected_tool_correctness(
    metric: Any,
    test_case: LLMTestCase,
    *,
    expected_pass: bool,
    case_id: str,
) -> None:
    """Run ToolCorrectnessMetric, including intentional retry-trace failures."""
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} used an incorrect tool trace."
        return

    # Invalid/repeated calls remain in tools_called so the trace tests the
    # recovery path, but they are omitted from expected_tools.  The metric
    # should therefore flag the tool trace while TaskCompletionMetric can
    # independently verify whether the eventual outcome was recovered.
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Case {case_id!r} was expected to expose an incorrect tool call, "
        f"but scored {getattr(metric, 'score', None)}."
    )


@pytest.mark.eval
class TestUpdateSubagentEval:
    """Evaluate update-subagent goldens with exactly two DeepEval metrics."""

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_tool_correctness_deepeval(self, case: dict[str, Any]) -> None:
        """Check exact combined-tool names, arguments, and order."""
        if not HAS_DEEPEVAL or DeepEvalToolCorrectnessMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_update_test_case(case)
        metric = DeepEvalToolCorrectnessMetric(
            threshold=1.0,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            should_consider_ordering=True,
            should_exact_match=True,
        )
        _measure_expected_tool_correctness(
            metric,
            test_case,
            expected_pass=bool(case.get("expected_tool_correctness_pass", True)),
            case_id=str(case.get("case_id", "unknown")),
        )

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_task_completion_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Judge whether the requested update was completed, live only."""
        if not eval_settings.is_live:
            pytest.skip("DeepEval TaskCompletionMetric requires --run-eval-live.")
        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_update_test_case(case)
        metric = DeepEvalTaskCompletionMetric(
            threshold=0.70,
            model=eval_settings.judge_model,
        )
        _measure_expected_completion(
            metric,
            test_case,
            expected_pass=bool(case.get("expected_completion_pass", True)),
            case_id=str(case.get("case_id", "unknown")),
        )
