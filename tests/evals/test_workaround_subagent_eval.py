"""DeepEval evaluation suite for the workaround subagent.

The workaround worker is evaluated on exactly two dimensions:

* ``ToolCorrectnessMetric`` checks the exact ordered tool calls and arguments.
* ``TaskCompletionMetric`` checks whether the supervisor instruction was
  completed, or intentionally remained incomplete after a bounded surrender.

The cases are replay fixtures. Tool failures are kept in ``tools_called`` so
that recovery behavior remains observable, while ``expected_tool_calls``
describes the correct trace for the case.
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


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "workaround_subagent_cases.json"


def _load_workaround_cases() -> list[dict[str, Any]]:
    """Load the dedicated workaround-subagent golden dataset.

    Returns:
        Workaround golden case dictionaries. A missing or malformed optional
        dataset produces an empty list so collection remains safe in minimal
        environments.
    """
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [case for case in data if isinstance(case, dict)]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if isinstance(case, dict)]
    return []


_WORKAROUND_CASES = _load_workaround_cases()
_WORKAROUND_CASE_IDS = [
    case.get("case_id", f"case_{index}") for index, case in enumerate(_WORKAROUND_CASES)
]


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert serialized golden calls into DeepEval tool calls.

    Args:
        raw_calls: Serialized calls with ``name``, ``args``, and ``output``.

    Returns:
        DeepEval-compatible tool-call objects.

    Raises:
        TypeError: If the serialized call collection is not a list.
    """
    if not isinstance(raw_calls, list):
        raise TypeError("Workaround golden tool calls must be lists.")
    return [
        ToolCall(
            name=str(tool_call.get("name", "")),
            input_parameters=tool_call.get("args", {}) or {},
            output=tool_call.get("output", ""),
        )
        for tool_call in raw_calls
    ]


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render the complete observed tool trace for the task-completion judge."""
    if not raw_calls:
        return "Observed tool trace: none"
    lines = ["Observed tool trace:"]
    for index, tool_call in enumerate(raw_calls, start=1):
        args = json.dumps(tool_call.get("args", {}) or {}, sort_keys=True)
        output = str(tool_call.get("output", ""))
        lines.append(f"{index}. {tool_call.get('name', '')}({args}) -> {output}")
    return "\n".join(lines)


def build_workaround_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Construct one DeepEval test case from a workaround golden.

    Args:
        case: Golden containing the supervisor instruction, observed and
            expected tool traces, completion task, and final outcome.

    Returns:
        An ``LLMTestCase`` suitable for both requested DeepEval metrics.

    Raises:
        TypeError: If ``tool_calls`` or ``expected_tool_calls`` is not a list.
    """
    tool_calls_raw = case.get("tool_calls", [])
    expected_tool_calls_raw = case.get("expected_tool_calls", tool_calls_raw)
    if not isinstance(tool_calls_raw, list) or not isinstance(expected_tool_calls_raw, list):
        raise TypeError("Workaround golden tool_calls and expected_tool_calls must be lists.")

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
            "Complete and validate the supervisor's workaround instruction, or report a bounded surrender without claiming success.",
        )
    )
    completion_task = str(
        case.get(
            "completion_task",
            "Apply the requested workaround and validate it against the selected smoke module and targeted test.",
        )
    )
    provenance = str(case.get("provenance", ""))
    evaluation_note = str(case.get("evaluation_note", ""))
    attempt_id = case.get("attempt_id")
    task_revision = case.get("task_revision")

    metadata = {
        "case_id": case.get("case_id"),
        "eval_type": "workaround_subagent",
        "golden_kind": case.get("golden_kind"),
        "provenance": provenance,
        "evaluation_note": evaluation_note,
        "is_retry": bool(case.get("is_retry", False)),
        "attempt_id": attempt_id,
        "task_revision": task_revision,
        "target_package_name": target_package,
        "changed_files": changed_files,
        "action_status": action_status,
        "terminal_error_code": case.get("terminal_error_code"),
        "observed_round_count": case.get("observed_round_count"),
        "observed_tool_count": case.get("observed_tool_count", len(tool_calls_raw)),
        "expected_completion_pass": bool(case.get("expected_completion_pass", True)),
        "expected_tool_correctness_pass": bool(
            case.get("expected_tool_correctness_pass", True)
        ),
    }

    return LLMTestCase(
        name=f"{case.get('case_id')} [Workaround Subagent]",
        input=instruction,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[
            f"Completion task:\n{completion_task}",
            f"Golden provenance:\n{provenance}",
            f"Evaluation note:\n{evaluation_note}",
            f"Attempt identity: attempt_id={attempt_id}; task_revision={task_revision}",
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
    """Run TaskCompletionMetric while supporting intentional surrender cases."""
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} did not complete the task."
        return

    # A bounded surrender is intentionally not a completed remediation. The
    # metric is measured directly so the expected negative result is asserted
    # without making the pytest case itself look like an evaluation failure.
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
    """Run ToolCorrectnessMetric, including intentional erroneous-call traces."""
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} used an incorrect tool trace."
        return

    # The erroneous call remains in tools_called and is intentionally omitted
    # from expected_tools. This lets the tool metric flag the model's bad call
    # while the task metric independently evaluates recovery.
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Case {case_id!r} was expected to expose an incorrect tool call, "
        f"but scored {getattr(metric, 'score', None)}."
    )


@pytest.mark.eval
class TestWorkaroundSubagentEval:
    """Evaluate workaround goldens with exactly two DeepEval metrics."""

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_tool_correctness_deepeval(self, case: dict[str, Any]) -> None:
        """Check exact workaround tool names, arguments, and order."""
        if not HAS_DEEPEVAL or DeepEvalToolCorrectnessMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_workaround_test_case(case)
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

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_task_completion_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Judge whether the requested workaround was completed, live only."""
        if not eval_settings.is_live:
            pytest.skip("DeepEval TaskCompletionMetric requires --run-eval-live.")
        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_workaround_test_case(case)
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
