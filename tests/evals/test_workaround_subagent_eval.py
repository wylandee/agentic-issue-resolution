"""Phase 3: DeepEval and structural evaluation for Workaround Subagent code-patching specialists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.evals.adapters import ToolCall
from tests.evals.conftest import EvalSettings
from tests.evals.custom_metrics import (
    ArchitectureBoundaryMetric,
    ToolEfficiencyMetric,
    WorkaroundLifecycleMetric,
)

try:
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    from tests.evals.adapters import DeepEvalLLMTestCase as LLMTestCase  # type: ignore[assignment]

    HAS_DEEPEVAL = False
    LLMTestCaseParams = None  # type: ignore[assignment,misc]
    GEval = None  # type: ignore[assignment,misc]
    assert_test = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Golden Dataset Loader for Pytest Parametrization
# ---------------------------------------------------------------------------

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "subagent_cases.json"


def _load_workaround_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if c.get("eval_type") == "workaround_subagent"]
        return []
    except Exception:
        return []


_WORKAROUND_CASES = _load_workaround_cases()
_WORKAROUND_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_WORKAROUND_CASES)]


# ---------------------------------------------------------------------------
# Test Case Construction Helper
# ---------------------------------------------------------------------------


def build_workaround_test_case(case: dict[str, Any]) -> LLMTestCase:
    """Construct an LLMTestCase from a golden workaround subagent case dictionary."""
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
        f"Workaround target component: {target_pkg}\n"
        f"Modified source/manifest files: {', '.join(changed_files) if changed_files else 'none'}\n"
        f"Tool execution rounds: {len(tools_called)}"
    )

    metadata = {
        "case_id": case.get("case_id"),
        "eval_type": "workaround_subagent",
        "provenance": case.get("provenance"),
        "is_retry": case.get("is_retry", False),
        "target_package_name": target_pkg,
        "changed_files": changed_files,
        "action_status": status,
        "expected_pass": case.get("expected_pass", True),
        "expected_boundary_pass": case.get("expected_boundary_pass", True),
        "expected_efficiency_pass": case.get("expected_efficiency_pass", True),
        "expected_lifecycle_pass": case.get("expected_lifecycle_pass", True),
        "expected_completion_pass": case.get("expected_completion_pass", True),
    }

    return LLMTestCase(
        input=instruction,
        actual_output=actual_output,
        expected_output=f"Investigate, plan, and apply minimal code workaround for {target_pkg} without modifying unrelated files.",
        context=[instruction, case.get("provenance", "")],
        tools_called=tools_called,
        additional_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Phase 3 Workaround Subagent Evaluation Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestWorkaroundSubagentEval:
    """Evaluation suite for Workaround Subagent lifecycle ordering, boundaries, and minimality."""

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_workaround_lifecycle_enforcement(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Workaround worker follows investigate -> plan -> execute -> validate lifecycle."""
        test_case = build_workaround_test_case(case)
        metric = WorkaroundLifecycleMetric()
        score = metric.measure(test_case)

        expected_lifecycle_pass = case.get("expected_lifecycle_pass", True)
        if expected_lifecycle_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed lifecycle check: {metric.reason}"
            )
            assert score == 1.0
        else:
            assert not metric.is_successful(), (
                f"Case '{case['case_id']}' was expected to violate lifecycle check but passed."
            )
            assert score == 0.0

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_workaround_tool_correctness(self, case: dict[str, Any]) -> None:
        """record_plan appears before any edit tools in compliant cases."""
        tool_calls = case.get("tool_calls", [])
        tool_names = [tc.get("name", "") for tc in tool_calls]

        edit_tools = {
            "deterministic_apply_edit_set",
            "remove_no_fix_dependency",
            "deterministic_search_replace",
            "deterministic_replace_ast_symbol",
        }

        has_edit = any(name in edit_tools for name in tool_names)
        expected_lifecycle_pass = case.get("expected_lifecycle_pass", True)

        if expected_lifecycle_pass and has_edit:
            assert "record_plan" in tool_names, (
                f"Case '{case['case_id']}': record_plan was missing despite code edits."
            )
            plan_idx = tool_names.index("record_plan")
            first_edit_idx = min(i for i, name in enumerate(tool_names) if name in edit_tools)
            assert plan_idx < first_edit_idx, (
                f"Case '{case['case_id']}': record_plan appeared at step {plan_idx + 1}, after first edit at {first_edit_idx + 1}."
            )

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_workaround_no_boundary_violation(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Workaround worker does not call update-only manifest modification tools."""
        test_case = build_workaround_test_case(case)
        metric = ArchitectureBoundaryMetric()
        score = metric.measure(test_case)

        expected_boundary_pass = case.get("expected_boundary_pass", True)
        if expected_boundary_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed boundary check: {metric.reason}"
            )
            assert score == 1.0
        else:
            assert not metric.is_successful()
            assert score == 0.0

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_workaround_efficiency(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Workaround worker operates efficiently within tool rounds budget."""
        test_case = build_workaround_test_case(case)
        metric = ToolEfficiencyMetric(threshold=0.70)
        score = metric.measure(test_case)

        expected_efficiency_pass = case.get("expected_efficiency_pass", True)
        if expected_efficiency_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed efficiency: {metric.reason} (score={score})"
            )
            assert score >= 0.70
        else:
            assert not metric.is_successful()
            assert score < 0.70

    @pytest.mark.parametrize("case", _WORKAROUND_CASES, ids=_WORKAROUND_CASE_IDS)
    def test_workaround_minimality(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Code change only modifies the vulnerable sink/call site (requires --run-eval-live)."""
        if not eval_settings.is_live:
            pytest.skip("Workaround Minimality GEval requires live LLM judge (--run-eval-live)")

        if not HAS_DEEPEVAL or GEval is None:
            pytest.skip("DeepEval is not installed in the current environment.")

        test_case = build_workaround_test_case(case)
        metric = GEval(
            name="Workaround Minimality",
            criteria=(
                "The code change should only modify the vulnerable sink or call site. "
                "It should not refactor unrelated code, rename variables unnecessarily, "
                "or restructure control flow beyond what is needed for the security fix."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=eval_settings.judge_model,
            threshold=0.70,
        )

        metric.measure(test_case)
        if case.get("expected_pass", True):
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' failed Workaround Minimality: {metric.reason}"
            )
