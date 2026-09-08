"""DeepEval TaskCompletionMetric evaluation for the triage subagent.

The fixtures are replayed triage outcomes. The live suite intentionally uses
one judge metric: TaskCompletionMetric. Scenario-specific expectations are
part of each case's task input because DeepEval's task-completion metric does
not consume expected_output or arbitrary metadata when judging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.evals.conftest import EvalSettings

try:
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric as DeepEvalTaskCompletionMetric
    from deepeval.test_case import LLMTestCase

    HAS_DEEPEVAL = True
except ImportError:
    from tests.evals.adapters import DeepEvalLLMTestCase as LLMTestCase  # type: ignore[assignment]

    HAS_DEEPEVAL = False
    assert_test = None  # type: ignore[assignment]
    DeepEvalTaskCompletionMetric = None  # type: ignore[assignment,misc]


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "triage_cases.json"


def _load_triage_cases() -> list[dict[str, Any]]:
    """Load the task-completion-only triage golden dataset.

    Returns:
        Triage golden dictionaries from the dedicated JSON file. Missing or
        malformed data is represented by an empty list so the optional live
        eval can be skipped cleanly.
    """
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [
            case
            for case in data
            if isinstance(case, dict) and case.get("eval_type", "triage") == "triage"
        ]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [
            case
            for case in data["cases"]
            if isinstance(case, dict) and case.get("eval_type", "triage") == "triage"
        ]
    return []


_TRIAGE_CASES = _load_triage_cases()
_TRIAGE_CASE_IDS = [
    case.get("case_id", f"case_{index}") for index, case in enumerate(_TRIAGE_CASES)
]


def _as_output_text(value: Any) -> str:
    """Serialize a replayed outcome for DeepEval."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def build_triage_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Build a DeepEval test case from a triage golden.

    Args:
        case: Golden dictionary containing scenario, completion_task,
            actual_output, and optional provenance metadata.

    Returns:
        An LLMTestCase with no tool calls and the replayed final outcome.

    Raises:
        ValueError: If the case does not contain a task or observable outcome.
    """
    case_id = str(case.get("case_id", "unknown"))
    scenario = str(case.get("scenario", "")).strip()
    completion_task = str(case.get("completion_task", "")).strip()
    actual_output = _as_output_text(case.get("actual_output", "")).strip()
    if not scenario or not completion_task or not actual_output:
        raise ValueError(
            f"Golden case {case_id!r} must contain scenario, completion_task, and actual_output."
        )

    task_input = f"Triage task:\n{completion_task}\n\nFinding scenario:\n{scenario}"
    expected_output = _as_output_text(case.get("expected_output", "")).strip() or None
    evidence = case.get("historical_evidence", [])
    evidence_text = json.dumps(evidence, indent=2, sort_keys=True)
    metadata = {
        "case_id": case_id,
        "eval_type": "triage",
        "golden_kind": case.get("golden_kind"),
        "provenance_status": case.get("provenance_status"),
        "expected_completion_pass": bool(case.get("expected_completion_pass", True)),
    }

    return LLMTestCase(
        name=f"{case_id} [Triage Task Completion]",
        input=task_input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[
            f"Historical trajectory mapping:\n{evidence_text}",
            f"Evaluation note:\n{case.get('evaluation_note', '')}",
        ],
        tools_called=[],
        additional_metadata=metadata,
    )


@pytest.mark.eval
class TestTriageEval:
    """Evaluate triage replay outcomes with exactly one DeepEval metric."""

    @pytest.mark.parametrize(
        "case",
        _TRIAGE_CASES or [{}],
        ids=_TRIAGE_CASE_IDS or ["no_cases"],
    )
    def test_task_completion_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Judge whether the final triage task was completed."""
        if not case:
            pytest.skip("No golden triage cases available")
        if not eval_settings.is_live:
            pytest.skip("DeepEval TaskCompletionMetric requires --run-eval-live.")
        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

        test_case = build_triage_test_case(case)
        metric = DeepEvalTaskCompletionMetric(
            threshold=0.70,
            model=eval_settings.judge_model,
            async_mode=False,
        )
        if assert_test is not None:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), (
                f"Case {case.get('case_id', 'unknown')!r} did not complete the task."
            )
