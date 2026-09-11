"""DeepEval evaluation suite for the update subagent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts.schemas import AgentActionStatus
from tests.evals.conftest import EvalSettings
from tests.evals.eval_case_helpers import (
    as_output_text,
    case_metadata,
    context_strings,
    expected_tools,
    make_tool_calls,
    observations,
)
from tests.evals.golden_schema import load_golden_dataset
from tests.evals.replay_adapters import replay_update_case
from tests.evals.replay_harness import (
    ReplayCapture,
    ScriptedReplayModel,
    cached_replay,
    format_tool_trace,
)

try:
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric as DeepEvalTaskCompletionMetric
    from deepeval.metrics import ToolCorrectnessMetric as DeepEvalToolCorrectnessMetric
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
_UPDATE_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="update_subagent_cases")
    if case.get("eval_type") == "update_subagent"
]
_UPDATE_CASE_IDS = [case["case_id"] for case in _UPDATE_CASES]
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[Any]:
    """Convert canonical update tool records."""
    return make_tool_calls(raw_calls, component="update")


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render captured update tools."""
    return format_tool_trace(raw_calls)


def build_update_test_case(
    case: dict[str, Any],
    observed_output: Any | None = None,
    observed_tools: list[dict[str, Any]] | None = None,
    *,
    replay_source: str | None = None,
    capture: ReplayCapture | None = None,
) -> LLMTestCase:
    """Construct an update test case from explicit observations."""
    if observed_output is None and observed_tools is None:
        final_output, tool_trace, source = observations(case)
    else:
        final_output = as_output_text(observed_output or "")
        tool_trace = list(observed_tools or [])
        source = replay_source or "production_live"
    target_package = str(case.get("target_package_name", ""))
    metadata = case_metadata(
        case,
        component="update_subagent",
        replay_source=source,
        actual_tools=tool_trace,
        capture=capture,
    )
    metadata.update(
        {
            "target_package_name": target_package,
            "selected_version": case.get("selected_version"),
            "dependency_type": case.get("dependency_type"),
            "action_status": case.get("action_status", "APPLIED"),
            "attempt_id": case.get("attempt_id"),
            "task_revision": case.get("task_revision"),
        }
    )
    return LLMTestCase(
        name=f"{case['case_id']} [Update Subagent]",
        input=str(case["input"]),
        actual_output=f"{final_output}\n\n{_format_tool_trace(tool_trace)}",
        expected_output=str(case["expected_output"]),
        context=context_strings(
            case,
            f"Supervisor instruction: {case.get('supervisor_instruction', '')}",
            f"Repository map: {case.get('repository_map', '(workspace map unavailable)')}",
        ),
        tools_called=_make_tool_calls(tool_trace),
        expected_tools=expected_tools(case, component="update"),
        token_cost=capture.token_cost if capture is not None else None,
        additional_metadata=metadata,
    )


def _measure_expected(
    metric: Any,
    test_case: LLMTestCase,
    *,
    expected_pass: bool,
    case_id: str,
    label: str,
) -> None:
    """Measure an expected-positive or expected-negative metric."""
    if expected_pass:
        if assert_test is not None:
            assert_test(test_case, [metric], run_async=False)
        else:
            metric.measure(test_case)
            assert metric.is_successful(), f"Case {case_id!r} failed {label}."
    else:
        metric.measure(test_case)
        assert not metric.is_successful(), f"Case {case_id!r} unexpectedly passed {label}."


def _live_capture(case: dict[str, Any], settings: EvalSettings) -> ReplayCapture:
    """Run one real update worker replay per process."""
    return cached_replay(
        _LIVE_CACHE,
        "update_subagent",
        str(case["case_id"]),
        lambda: replay_update_case(case, settings),
    )


@pytest.mark.eval
class TestUpdateSubagentEval:
    """Evaluate update output and worker tool traces."""

    @pytest.mark.parametrize("case", _UPDATE_CASES or [{}], ids=_UPDATE_CASE_IDS or ["no_cases"])
    def test_update_subagent_live_replay_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run the production worker and judge its captured result."""
        if not case:
            pytest.skip("No golden update cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if not HAS_DEEPEVAL:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        test_case = build_update_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            replay_source="production_live",
            capture=capture,
        )
        if DeepEvalToolCorrectnessMetric is not None:
            tool_metric = DeepEvalToolCorrectnessMetric(
                threshold=1.0,
                evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
                should_consider_ordering=True,
                should_exact_match=True,
            )
            _measure_expected(
                tool_metric,
                test_case,
                expected_pass=bool(case.get("expected_tool_correctness_pass", True)),
                case_id=str(case["case_id"]),
                label="update tool correctness",
            )
        if DeepEvalTaskCompletionMetric is not None:
            task_metric = DeepEvalTaskCompletionMetric(
                threshold=0.70,
                model=eval_settings.judge_model,
                async_mode=False,
            )
            _measure_expected(
                task_metric,
                test_case,
                expected_pass=bool(case.get("expected_completion_pass", True)),
                case_id=str(case["case_id"]),
                label="update task completion",
            )


def _tool_signature(tool_events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return tool names and arguments without fixture output text."""
    return [
        (str(event.get("name", "")), dict(event.get("args", {}) or {})) for event in tool_events
    ]


@pytest.mark.parametrize("case", _UPDATE_CASES or [{}], ids=_UPDATE_CASE_IDS or ["no_cases"])
def test_update_subagent_offline_production_replay(
    case: dict[str, Any],
    eval_settings: EvalSettings,
) -> None:
    """Exercise every update case through the real worker and bounded loop."""
    if not case:
        pytest.skip("No golden update cases available")

    fixture = case.get("offline_fixture", {})
    scripted_tools = fixture.get("actual_tools", []) if isinstance(fixture, dict) else []
    assert isinstance(scripted_tools, list)
    model = ScriptedReplayModel.from_tool_trace(scripted_tools, final_text="")
    capture = replay_update_case(case, eval_settings, llm=model)

    assert model.invocation_count == len(scripted_tools) + 1
    assert _tool_signature(capture.actual_tools) == _tool_signature(scripted_tools)
    assert capture.attempt_id
    assert capture.task_revision == int(case.get("task_revision") or 1)
    assert all(call.get("kind") == "sandbox_command" for call in capture.external_calls)

    case_id = str(case["case_id"])
    summary = capture.typed_result["action_summaries"][0]
    if case_id == "update-retry-limit-surrender":
        assert summary.status == AgentActionStatus.SURRENDER
        assert "RETRY_LIMIT_REACHED" in (capture.actual_output + "\n" + "\n".join(capture.errors))
        baseline = json.loads(str(case["replay"]["input"]["workspace_files"]["package.json"]))
        final_manifest = json.loads(capture.final_files["package.json"])
        assert final_manifest == baseline
        return

    assert summary.status == AgentActionStatus.SUCCESS
    assert capture.changed_files
    manifest = json.loads(capture.final_files["package.json"])
    dependency_section = str(case["dependency_type"])
    package_name = str(case["target_package_name"])
    assert manifest[dependency_section][package_name] == case["selected_version"]
