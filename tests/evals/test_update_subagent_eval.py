"""Phase 3: DeepEval and structural evaluation for Update Subagent dependency workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.evals.adapters import ToolCall
from tests.evals.conftest import EvalSettings
from tests.evals.custom_metrics import (
    ArchitectureBoundaryMetric,
    DeterministicTaskCompletionMetric,
    ToolEfficiencyMetric,
)

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        TaskCompletionMetric as DeepEvalTaskCompletionMetric,
    )
    from deepeval.metrics import (
        ToolCorrectnessMetric as DeepEvalToolCorrectnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    HAS_DEEPEVAL = True
except ImportError:
    from tests.evals.adapters import DeepEvalLLMTestCase as LLMTestCase  # type: ignore[assignment]

    HAS_DEEPEVAL = False
    assert_test = None  # type: ignore[assignment]
    DeepEvalTaskCompletionMetric = None  # type: ignore[assignment,misc]
    DeepEvalToolCorrectnessMetric = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Golden Dataset Loader for Pytest Parametrization
# ---------------------------------------------------------------------------

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "subagent_cases.json"


def _load_update_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if c.get("eval_type") == "update_subagent"]
        return []
    except Exception:
        return []


_UPDATE_CASES = _load_update_cases()
_UPDATE_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_UPDATE_CASES)]


# ---------------------------------------------------------------------------
# Test Case Construction Helper
# ---------------------------------------------------------------------------


def build_update_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Construct an LLMTestCase from a golden update subagent case dictionary."""
    tool_calls_raw = case.get("tool_calls", [])
    tools_called = [
        ToolCall(
            name=tc.get("name", ""),
            input_parameters=tc.get("args", {}) or {},
            output=tc.get("output", ""),
        )
        for tc in tool_calls_raw
    ]

    instruction = case.get("supervisor_instruction", "")
    target_pkg = case.get("target_package_name", "")
    changed_files = case.get("changed_files", [])
    status = case.get("action_status", "APPLIED")

    actual_output = (
        f"Status: {status}\n"
        f"Modified package: {target_pkg}\n"
        f"Changed files: {', '.join(changed_files) if changed_files else 'none'}\n"
        f"Tool execution rounds: {len(tools_called)}"
    )

    expected_tools: list[ToolCall] = []
    if case.get("expected_pass", True) and status == "APPLIED":
        expected_tools.append(
            ToolCall(name="modify_and_validate_npm_dependency", input_parameters={})
        )

    metadata = {
        "case_id": case.get("case_id"),
        "eval_type": "update_subagent",
        "provenance": case.get("provenance"),
        "is_retry": case.get("is_retry", False),
        "target_package_name": target_pkg,
        "selected_version": case.get("selected_version"),
        "changed_files": changed_files,
        "action_status": status,
        "expected_pass": case.get("expected_pass", True),
        "expected_boundary_pass": case.get("expected_boundary_pass", True),
        "expected_efficiency_pass": case.get("expected_efficiency_pass", True),
        "expected_completion_pass": case.get("expected_completion_pass", True),
    }

    return LLMTestCase(
        name=f"{case.get('case_id')} [Update Subagent]",
        input=instruction,
        actual_output=actual_output,
        expected_output=f"Update {target_pkg} in manifest files with the combined edit-and-sync transaction.",
        context=[
            instruction,
            case.get("provenance", ""),
            f"Deterministic repository map:\n{case.get('repository_map', '(workspace is empty)')}",
        ],
        tools_called=tools_called,
        expected_tools=expected_tools if expected_tools else None,
        additional_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Phase 3 Update Subagent Evaluation Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestUpdateSubagentEval:
    """Evaluation suite for Update Subagent tool usage, boundary adherence, and efficiency."""

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_tool_sequence_correctness(self, case: dict[str, Any]) -> None:
        """Active update trajectories expose only the combined manifest transaction."""
        tool_calls = case.get("tool_calls", [])
        tool_names = [tc.get("name", "") for tc in tool_calls]

        assert "modify_npm_dependency" not in tool_names
        assert "validate_manifest_sync" not in tool_names
        assert "read_repository_map" not in tool_names
        assert "revert_workspace_file" not in tool_names
        if case.get("action_status") == "APPLIED" and case.get("expected_pass", True):
            assert "modify_and_validate_npm_dependency" in tool_names

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_tool_correctness_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """DeepEval built-in ToolCorrectnessMetric evaluates ordered tool execution."""
        if not HAS_DEEPEVAL or DeepEvalToolCorrectnessMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_update_test_case(case)
        if not getattr(test_case, "expected_tools", None):
            pytest.skip("No expected tools defined for negative/surrender case.")

        metric = DeepEvalToolCorrectnessMetric(threshold=0.5, should_consider_ordering=True)
        if assert_test:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful()

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_no_version_selection_boundary_violation(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Worker does not attempt to discover or select dependency versions during first-pass."""
        test_case = build_update_test_case(case)
        metric = ArchitectureBoundaryMetric()
        expected_boundary_pass = case.get("expected_boundary_pass", True)

        if expected_boundary_pass:
            if HAS_DEEPEVAL and assert_test:
                assert_test(test_case, [metric], run_async=False)
            else:
                score = metric.measure(test_case)
                assert metric.is_successful(), (
                    f"Case '{case['case_id']}' was expected to pass boundary check but scored {score}."
                )
        else:
            score = metric.measure(test_case)
            assert not metric.is_successful(), (
                f"Case '{case['case_id']}' was expected to violate boundary check but passed."
            )
            assert score == 0.0

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_tool_call_efficiency(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Worker completes task within reasonable tool call budget without redundant reads."""
        test_case = build_update_test_case(case)
        metric = ToolEfficiencyMetric(threshold=0.70)
        expected_efficiency_pass = case.get("expected_efficiency_pass", True)

        if expected_efficiency_pass:
            if HAS_DEEPEVAL and assert_test:
                assert_test(test_case, [metric], run_async=False)
            else:
                score = metric.measure(test_case)
                assert metric.is_successful(), (
                    f"Case '{case['case_id']}' was expected to pass efficiency metric but scored {score}."
                )
        else:
            score = metric.measure(test_case)
            assert not metric.is_successful(), (
                f"Case '{case['case_id']}' was expected to fail efficiency metric but scored {score}."
            )
            assert score < 0.70

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_task_completion(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Worker produces changed files and validates matching supervisor instruction."""
        test_case = build_update_test_case(case)
        metric = DeterministicTaskCompletionMetric(threshold=0.70)
        expected_completion_pass = case.get("expected_completion_pass", True)

        if expected_completion_pass:
            if HAS_DEEPEVAL and assert_test:
                assert_test(test_case, [metric], run_async=False)
            else:
                score = metric.measure(test_case)
                assert metric.is_successful(), (
                    f"Case '{case['case_id']}' was expected to pass task completion but scored {score}."
                )
        else:
            score = metric.measure(test_case)
            assert not metric.is_successful(), (
                f"Case '{case['case_id']}' was expected to fail task completion (e.g. surrender) but passed."
            )
            assert score == 0.0

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_live_task_completion_deepeval(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """DeepEval built-in TaskCompletionMetric evaluates task completion with LLM judge (requires --run-eval-live)."""
        if not eval_settings.is_live:
            pytest.skip("DeepEval TaskCompletionMetric requires live LLM judge (--run-eval-live)")

        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")

        test_case = build_update_test_case(case)
        metric = DeepEvalTaskCompletionMetric(
            threshold=0.70,
            model=eval_settings.judge_model,
        )

        if case.get("expected_completion_pass", True) and case.get("action_status") == "APPLIED":
            if assert_test:
                assert_test(test_case, [metric], run_async=False)
            else:
                metric.measure(test_case)
                assert metric.is_successful()
