"""DeepEval evaluation suite for the task-scoped QA Critic.

The QA Critic is evaluated on exactly two dimensions:

* ``ToolCorrectnessMetric`` checks the expected read-only investigation path,
  input parameters, and terminal ``emit_qa_evaluation`` call.
* ``TaskCompletionMetric`` checks whether the Critic completed the assigned QA
  decision, including correctly reporting intentional QA failures.

The cases are replay fixtures. Historical trajectory evidence is retained in
the golden data as provenance, while ``tool_calls`` and ``expected_tool_calls``
are separate so ToolCorrectness cannot be made tautological by copying the
observed trace into the expected trace.
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


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "qa_cases.json"


def _load_qa_cases() -> list[dict[str, Any]]:
    """Load the dedicated QA Critic golden cases.

    Returns:
        QA Critic replay cases. Missing, malformed, or differently typed
        golden data produces an empty collection so the optional eval suite
        remains safe in minimal environments.
    """
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [
            case for case in data if isinstance(case, dict) and case.get("eval_type") == "qa_critic"
        ]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [
            case
            for case in data["cases"]
            if isinstance(case, dict) and case.get("eval_type") == "qa_critic"
        ]
    return []


_QA_CASES = _load_qa_cases()
_QA_CASE_IDS = [case.get("case_id", f"case_{index}") for index, case in enumerate(_QA_CASES)]


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert serialized QA tool calls into DeepEval-compatible calls.

    Args:
        raw_calls: Serialized calls containing ``name``, ``args``, and an
            optional result under ``output``.

    Returns:
        DeepEval-compatible tool-call objects.

    Raises:
        TypeError: If the serialized call collection or call arguments are not
            dictionaries.
    """
    if not isinstance(raw_calls, list):
        raise TypeError("QA golden tool calls must be lists.")

    converted: list[ToolCall] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            raise TypeError("Each QA golden tool call must be a dictionary.")
        args = call.get("args", {}) or {}
        if not isinstance(args, dict):
            raise TypeError("QA golden tool-call args must be dictionaries.")
        converted.append(
            ToolCall(
                name=str(call.get("name", "")),
                input_parameters=args,
                output=call.get("output", ""),
            )
        )
    return converted


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render the observed tool trace for the task-completion judge.

    Args:
        raw_calls: Serialized observed calls in execution order.

    Returns:
        A bounded human-readable trace string.
    """
    if not raw_calls:
        return "Observed QA tool trace: none"

    lines = ["Observed QA tool trace:"]
    for index, call in enumerate(raw_calls, start=1):
        args = json.dumps(call.get("args", {}) or {}, sort_keys=True)
        output = str(call.get("output", ""))
        lines.append(f"{index}. {call.get('name', '')}({args}) -> {output}")
    return "\n".join(lines)


def _format_provenance(case: dict[str, Any]) -> str:
    """Render compact provenance and evidence metadata for a test case."""
    provenance = case.get("provenance", "")
    evidence = case.get("historical_evidence", [])
    if isinstance(evidence, list) and evidence:
        evidence_text = "\n".join(f"- {item}" for item in evidence)
    else:
        evidence_text = "- no historical evidence recorded"
    return (
        f"Evidence status: {case.get('evidence_status', 'unknown')}\n{provenance}\n{evidence_text}"
    )


def build_qa_production_prompt(case: dict[str, Any]) -> str:
    """Build the task and deterministic context for one QA replay.

    Args:
        case: QA Critic golden containing policy, gate evidence, and expected
            outcome metadata.

    Returns:
        A structured prompt suitable for the TaskCompletionMetric.
    """
    vulnerability = case.get("vulnerability_context", {})
    execution = case.get("execution_context", {})
    logs = case.get("execution_logs", {})
    changed_files = case.get("changed_files", []) or []
    expected_output = case.get("expected_output", "")

    cves = vulnerability.get("cve_ids", []) or []
    ghsas = vulnerability.get("ghsa_ids", []) or []
    task = case.get(
        "completion_task",
        "Evaluate the assigned vulnerability group from deterministic evidence and emit one QA evaluation.",
    )

    return (
        "=== QA CRITIC TASK ===\n"
        f"{task}\n\n"
        "=== ASSIGNED GROUP ===\n"
        f"- Group ID: {vulnerability.get('group_id')}\n"
        f"- Component: {vulnerability.get('vulnerable_component')}\n"
        f"- Target file: {vulnerability.get('file_path')}\n"
        f"- CVE IDs: {', '.join(cves) if cves else 'none'}\n"
        f"- GHSA IDs: {', '.join(ghsas) if ghsas else 'none'}\n"
        f"- QA policy: {case.get('qa_policy')}\n"
        f"- Changed files: {', '.join(changed_files) if changed_files else 'none'}\n"
        f"- Worker action: {case.get('worker_action_summary', 'none')}\n\n"
        "=== DETERMINISTIC QA EVIDENCE ===\n"
        f"- Install passed: {execution.get('install_passed')}\n"
        f"Install log:\n{logs.get('install_log', 'none')}\n\n"
        f"- Scanner execution status: {execution.get('scanner_execution_status')}\n"
        f"- Target scanner cleared: {execution.get('target_scanner_cleared')}\n"
        f"- Remaining target identifiers: {execution.get('target_remaining_identifiers', [])}\n"
        f"Scan log:\n{logs.get('scan_summary', 'none')}\n\n"
        f"- Tests passed: {execution.get('tests_passed')}\n"
        f"Test log:\n{logs.get('test_output', 'none')}\n\n"
        "=== EXPECTED COMPLETION CONTRACT ===\n"
        f"{expected_output}\n\n"
        "Use only the authorized read-only QA review tools and finish with one "
        "emit_qa_evaluation call when the task is completable."
    )


def build_qa_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Construct one DeepEval test case from a QA golden.

    Args:
        case: Golden containing separate observed and expected tool traces,
            completion context, and the replayed QA outcome.

    Returns:
        An ``LLMTestCase`` suitable for both requested DeepEval metrics.

    Raises:
        TypeError: If observed or expected tool traces are not lists.
        ValueError: If the golden omits the independent expected tool trace.
    """
    tool_calls_raw = case.get("tool_calls", [])
    expected_tool_calls_raw = case.get("expected_tool_calls")
    if not isinstance(tool_calls_raw, list):
        raise TypeError("QA golden tool_calls must be a list.")
    if not isinstance(expected_tool_calls_raw, list):
        raise ValueError("QA golden must define expected_tool_calls independently.")

    tools_called = _make_tool_calls(tool_calls_raw)
    expected_tools = _make_tool_calls(expected_tool_calls_raw)
    prompt = build_qa_production_prompt(case)

    llm_output = case.get("llm_qa_output", case.get("qa_output"))
    if isinstance(llm_output, dict):
        final_output = json.dumps(llm_output, indent=2, sort_keys=True)
    else:
        final_output = str(case.get("final_output", ""))

    actual_output = f"{final_output}\n\n{_format_tool_trace(tool_calls_raw)}"
    expected_output = str(
        case.get(
            "expected_output",
            "Complete the assigned QA evaluation and report the evidence-backed outcome.",
        )
    )

    metadata = {
        "case_id": case.get("case_id"),
        "eval_type": "qa_critic",
        "golden_kind": case.get("golden_kind"),
        "evidence_status": case.get("evidence_status"),
        "provenance": case.get("provenance"),
        "historical_evidence": case.get("historical_evidence", []),
        "qa_policy": case.get("qa_policy"),
        "task_id": llm_output.get("task_id") if isinstance(llm_output, dict) else None,
        "qa_verdict": llm_output.get("passed") if isinstance(llm_output, dict) else None,
        "failure_category": (
            llm_output.get("failure_category") if isinstance(llm_output, dict) else None
        ),
        "expected_qa_verdict": case.get("expected_qa_verdict", {}),
        "execution_context": case.get("execution_context", {}),
        "expected_task_completion_pass": bool(case.get("expected_task_completion_pass", True)),
        "expected_tool_correctness_pass": case.get("expected_tool_correctness_pass", True),
        "tool_correctness_applicable": bool(case.get("tool_correctness_applicable", True)),
        "observed_round_count": case.get("observed_round_count"),
        "observed_tool_count": case.get("observed_tool_count", len(tool_calls_raw)),
    }

    return LLMTestCase(
        name=f"{case.get('case_id')} [QA Critic]",
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[
            f"Completion task:\n{case.get('completion_task', '')}",
            f"Golden provenance:\n{_format_provenance(case)}",
            f"Expected QA output:\n{json.dumps(case.get('expected_qa_verdict', {}), indent=2)}",
        ],
        tools_called=tools_called,
        expected_tools=expected_tools if expected_tools else None,
        additional_metadata=metadata,
    )


def _measure_expected_completion(
    metric: Any,
    test_case: LLMTestCase,
    *,
    expected_pass: bool,
    case_id: str,
) -> None:
    """Measure TaskCompletionMetric and support an intentional surrender.

    Args:
        metric: Configured DeepEval TaskCompletionMetric.
        test_case: QA replay case.
        expected_pass: Whether the replay should complete its assigned task.
        case_id: Golden identifier used in assertion messages.

    Raises:
        AssertionError: If the metric direction does not match the golden.
    """
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} did not complete the QA task."
        return

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
    """Measure ToolCorrectnessMetric against the independent expected trace.

    Args:
        metric: Configured DeepEval ToolCorrectnessMetric.
        test_case: QA replay case.
        expected_pass: Whether the observed trace should be correct.
        case_id: Golden identifier used in assertion messages.

    Raises:
        AssertionError: If the metric direction does not match the golden.
    """
    if expected_pass:
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} used an incorrect QA tool trace."
        return

    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Case {case_id!r} was expected to expose an incorrect QA tool trace, "
        f"but scored {getattr(metric, 'score', None)}."
    )


@pytest.mark.eval
class TestQACriticEval:
    """Evaluate QA Critic goldens with exactly two DeepEval metrics."""

    @pytest.mark.parametrize("case", _QA_CASES, ids=_QA_CASE_IDS)
    def test_tool_correctness_deepeval(self, case: dict[str, Any]) -> None:
        """Check expected QA tool names, input parameters, and ordering."""
        if not HAS_DEEPEVAL or DeepEvalToolCorrectnessMetric is None:
            pytest.skip("DeepEval is not installed.")
        if not case.get("tool_correctness_applicable", True):
            pytest.skip(
                "Case is a runtime-boundary TaskCompletion case without a complete tool trace."
            )

        test_case = build_qa_test_case(case)
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

    @pytest.mark.parametrize("case", _QA_CASES, ids=_QA_CASE_IDS)
    def test_task_completion_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Judge whether the QA Critic completed its assigned task."""
        if not eval_settings.is_live:
            pytest.skip("DeepEval TaskCompletionMetric requires --run-eval-live.")
        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_qa_test_case(case)
        metric = DeepEvalTaskCompletionMetric(
            threshold=0.70,
            model=eval_settings.judge_model,
        )
        _measure_expected_completion(
            metric,
            test_case,
            expected_pass=bool(case.get("expected_task_completion_pass", True)),
            case_id=str(case.get("case_id", "unknown")),
        )
