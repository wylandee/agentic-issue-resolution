"""DeepEval evaluation suite for the workaround subagent."""

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
from tests.evals.replay_adapters import replay_workaround_case
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


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "workaround_subagent_cases.json"
_WORKAROUND_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="workaround_subagent_cases")
    if case.get("eval_type") == "workaround_subagent"
]
_WORKAROUND_CASE_IDS = [case["case_id"] for case in _WORKAROUND_CASES]
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[Any]:
    """Convert canonical workaround tool records."""
    return make_tool_calls(raw_calls, component="workaround")


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render captured workaround tools."""
    return format_tool_trace(raw_calls)


def build_workaround_test_case(
    case: dict[str, Any],
    observed_output: Any | None = None,
    observed_tools: list[dict[str, Any]] | None = None,
    *,
    replay_source: str | None = None,
    capture: ReplayCapture | None = None,
) -> LLMTestCase:
    """Construct a workaround test case from explicit observations."""
    if observed_output is None and observed_tools is None:
        final_output, tool_trace, source = observations(case)
    else:
        final_output = as_output_text(observed_output or "")
        tool_trace = list(observed_tools or [])
        source = replay_source or "production_live"
    metadata = case_metadata(
        case,
        component="workaround_subagent",
        replay_source=source,
        actual_tools=tool_trace,
        capture=capture,
    )
    metadata.update(
        {
            "attempt_id": case.get("attempt_id"),
            "task_revision": case.get("task_revision"),
            "target_package_name": case.get("target_package_name"),
            "action_status": case.get("action_status", "APPLIED"),
        }
    )
    return LLMTestCase(
        name=f"{case['case_id']} [Workaround Subagent]",
        input=str(case["input"]),
        actual_output=f"{final_output}\n\n{_format_tool_trace(tool_trace)}",
        expected_output=str(case["expected_output"]),
        context=context_strings(
            case,
            f"Supervisor instruction: {case.get('supervisor_instruction', '')}",
            f"Completion task: {case.get('completion_task', '')}",
            f"Attempt identity: attempt_id={case.get('attempt_id')}; task_revision={case.get('task_revision')}",
        ),
        tools_called=_make_tool_calls(tool_trace),
        expected_tools=expected_tools(case, component="workaround"),
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
    """Run one real workaround worker replay per process."""
    return cached_replay(
        _LIVE_CACHE,
        "workaround_subagent",
        str(case["case_id"]),
        lambda: replay_workaround_case(case, settings),
    )


@pytest.mark.eval
class TestWorkaroundSubagentEval:
    """Evaluate workaround output and captured tool traces."""

    @pytest.mark.parametrize(
        "case",
        _WORKAROUND_CASES or [{}],
        ids=_WORKAROUND_CASE_IDS or ["no_cases"],
    )
    def test_workaround_subagent_live_replay_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run the production worker and judge its captured result."""
        if not case:
            pytest.skip("No golden workaround cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if not HAS_DEEPEVAL:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        test_case = build_workaround_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            replay_source="production_live",
            capture=capture,
        )
        if DeepEvalToolCorrectnessMetric is not None:
            tool_metric = DeepEvalToolCorrectnessMetric(
                threshold=0.50,
                evaluation_params=[],
                should_consider_ordering=True,
                should_exact_match=False,
            )
            if bool(case.get("expected_tool_correctness_pass", True)):
                _measure_expected(
                    tool_metric,
                    test_case,
                    expected_pass=True,
                    case_id=str(case["case_id"]),
                    label="workaround tool correctness",
                )
            else:
                tool_metric.measure(test_case)
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
                label="workaround task completion",
            )


def _tool_signature(tool_events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return tool names and arguments without fixture output text."""
    return [
        (str(event.get("name", "")), dict(event.get("args", {}) or {})) for event in tool_events
    ]


@pytest.mark.parametrize(
    "case",
    _WORKAROUND_CASES or [{}],
    ids=_WORKAROUND_CASE_IDS or ["no_cases"],
)
def test_workaround_subagent_offline_production_replay(
    case: dict[str, Any],
    eval_settings: EvalSettings,
) -> None:
    """Exercise every workaround lifecycle through the real worker and loop."""
    if not case:
        pytest.skip("No golden workaround cases available")

    case_id = str(case["case_id"])
    fixture = case.get("offline_fixture", {})
    historical_tools = fixture.get("actual_tools", []) if isinstance(fixture, dict) else []
    assert isinstance(historical_tools, list)
    if case_id == "workaround-surrender-after-max-tool-round-limit":
        scripted_tools = [{"name": "read_repository_map", "args": {}} for _ in range(24)]
        model = ScriptedReplayModel.from_tool_trace(
            scripted_tools,
            enforce_bound_tools=False,
        )
    else:
        scripted_tools = historical_tools
        model = ScriptedReplayModel.from_tool_trace(
            scripted_tools,
            final_text="",
            enforce_bound_tools=False,
        )

    capture = replay_workaround_case(case, eval_settings, llm=model)

    # Historical traces can contain one later recovery call that production
    # correctly omits after terminal validation. The executed trace must stay
    # an ordered prefix of the recorded tool plan.
    assert _tool_signature(capture.actual_tools) == _tool_signature(
        scripted_tools[: len(capture.actual_tools)]
    )
    assert model.invocation_count == len(capture.actual_tools)
    assert capture.attempt_id == case.get("attempt_id")
    assert capture.task_revision == int(case.get("task_revision") or 1)
    assert all(
        call.get("kind") == "sandbox_command" or call.get("method") in {"GET", "POST"}
        for call in capture.external_calls
    )

    surrender_cases = {
        "workaround-surrender-after-max-validation-input-limit",
        "workaround-surrender-after-max-validation-gate-limit",
        "workaround-surrender-after-max-tool-round-limit",
    }
    summary = capture.typed_result["action_summaries"][0]
    if case_id in surrender_cases:
        assert summary.status == AgentActionStatus.SURRENDER
        evidence = "\n".join(
            [*capture.errors, capture.actual_output]
            + [str(event.get("output", "")) for event in capture.actual_tools]
        )
        terminal_marker = str(case["terminal_error_code"])
        if terminal_marker == "MAX_SUBAGENT_TOOL_CALL_ROUNDS":
            assert "maximum tool-call rounds" in evidence
        else:
            assert terminal_marker in evidence
        source_path = next(
            path for path in case["changed_files"] if not str(path).endswith(".json")
        )
        baseline = case["replay"]["input"]["workspace_files"][source_path]
        assert capture.final_files[source_path] == baseline
        return
    assert summary.status == AgentActionStatus.SUCCESS
    expected_changed_files = set(case["changed_files"])
    if case_id == "workaround-no-fix-package-removal":
        expected_changed_files.discard("package-lock.json")
    assert set(capture.changed_files) == expected_changed_files
    assert any(
        event["name"] == "validate_workaround" and event["output"].startswith("SUCCESS:")
        for event in capture.actual_tools
    )

    if case_id == "workaround-code-change-initial-normal":
        content = capture.final_files["lib/insecurity.ts"]
        assert "expressjwt" in content
        assert "algorithms: ['RS256']" in content
    elif case_id == "workaround-code-change-pivot":
        assert "safeArchiveExtract(archive, input)" in capture.final_files["server.js"]
    elif case_id == "workaround-code-change-clean-first-attempt":
        content = capture.final_files["server.ts"]
        assert "cors" in content
        assert "allowedOrigins" in content
    elif case_id == "workaround-no-fix-package-removal":
        package_json = json.loads(capture.final_files["package.json"])
        assert all(
            "notevil" not in section
            for section in package_json.values()
            if isinstance(section, dict)
        )
        assert "require('notevil')" not in capture.final_files["routes/b2bOrder.ts"]
    elif case_id == "workaround-no-fix-pivot":
        actual_package = json.loads(capture.final_files["package.json"])
        baseline_package = json.loads(case["replay"]["input"]["workspace_files"]["package.json"])
        assert actual_package.get("dependencies") == baseline_package.get("dependencies")
        assert actual_package.get("overrides") == baseline_package.get("overrides")
        assert actual_package.get("scripts", {}).get("test") == "mocha"
        assert actual_package.get("devDependencies", {}).get("mocha") == "10.0.0"
        assert "notevil" not in capture.final_files["routes/b2bOrder.ts"]
    elif case_id == "workaround-retry-after-validation-failure":
        assert "allowInsecureKeySizes: true } as any" in capture.final_files["lib/insecurity.ts"]
    elif case_id == "workaround-retry-after-validation-infra-failure":
        assert any(
            event["name"] == "record_targeted_test_substitution" for event in capture.actual_tools
        )
