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
    TaskCompletionMetric,
    ToolEfficiencyMetric,
)

try:
    from deepeval.test_case import LLMTestCase

    HAS_DEEPEVAL = True
except ImportError:
    from tests.evals.adapters import DeepEvalLLMTestCase as LLMTestCase  # type: ignore[assignment]

    HAS_DEEPEVAL = False


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
        input=instruction,
        actual_output=actual_output,
        expected_output=f"Update {target_pkg} in manifest files and validate manifest synchronization.",
        context=[instruction, case.get("provenance", "")],
        tools_called=tools_called,
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
        """Update worker calls modify_npm_dependency then validate_manifest_sync."""
        tool_calls = case.get("tool_calls", [])
        tool_names = [tc.get("name", "") for tc in tool_calls]

        # In any run that attempts manifest edits, modify_npm_dependency must be followed
        # by validate_manifest_sync
        if "modify_npm_dependency" in tool_names:
            assert "validate_manifest_sync" in tool_names, (
                f"Case '{case['case_id']}': 'modify_npm_dependency' was invoked without subsequent 'validate_manifest_sync'."
            )
            modify_idx = tool_names.index("modify_npm_dependency")
            validate_idx = tool_names.index("validate_manifest_sync")
            assert validate_idx > modify_idx, (
                f"Case '{case['case_id']}': 'validate_manifest_sync' appeared before 'modify_npm_dependency'."
            )

    @pytest.mark.parametrize("case", _UPDATE_CASES, ids=_UPDATE_CASE_IDS)
    def test_no_version_selection_boundary_violation(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Worker does not attempt to discover or select dependency versions during first-pass."""
        test_case = build_update_test_case(case)
        metric = ArchitectureBoundaryMetric()
        score = metric.measure(test_case)

        expected_boundary_pass = case.get("expected_boundary_pass", True)
        if expected_boundary_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed boundary check: {metric.reason}"
            )
            assert score == 1.0
        else:
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
        score = metric.measure(test_case)

        expected_efficiency_pass = case.get("expected_efficiency_pass", True)
        if expected_efficiency_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed efficiency metric: {metric.reason} (score={score})"
            )
            assert score >= 0.70
        else:
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
        metric = TaskCompletionMetric(threshold=0.70)
        score = metric.measure(test_case)

        expected_completion_pass = case.get("expected_completion_pass", True)
        if expected_completion_pass:
            assert metric.is_successful(), (
                f"Case '{case['case_id']}' unexpectedly failed task completion: {metric.reason} (score={score})"
            )
            assert score == 1.0
        else:
            assert not metric.is_successful(), (
                f"Case '{case['case_id']}' was expected to fail task completion (e.g. surrender) but passed."
            )
            assert score == 0.0
