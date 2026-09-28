"""DeepEval evaluation suite for the tactical Supervisor boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts.schemas import TaskStatus
from remediation_engine.orchestration.tactical_supervisor import _safe_relative_file_hints
from tests.evals.conftest import EvalSettings
from tests.evals.eval_case_helpers import (
    as_output_text,
    case_metadata,
    context_strings,
    expected_tools,
    make_tool_calls,
)
from tests.evals.golden_schema import load_golden_dataset
from tests.evals.replay_adapters import replay_supervisor_case
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


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "supervisor_cases.json"
_SUPERVISOR_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="supervisor_cases")
    if case.get("eval_type") == "supervisor"
]
_SUPERVISOR_CASE_IDS = [case["case_id"] for case in _SUPERVISOR_CASES]
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}
_EXPECTED_BOUND_TOOL_NAMES = [
    "VersionBumpSupervisorAction",
    "PackageOverrideSupervisorAction",
    "CodeWorkaroundSupervisorAction",
    "PortfolioEscalationSupervisorAction",
]


def build_supervisor_test_case(
    case: dict[str, Any],
    observed_output: Any,
    observed_tools: list[dict[str, Any]],
    replay_source: str,
    capture: ReplayCapture,
) -> LLMTestCase:
    """Construct a live tactical test case from an explicit production capture."""
    tool_trace = list(observed_tools)
    metadata = case_metadata(
        case,
        component="supervisor",
        replay_source=replay_source,
        actual_tools=tool_trace,
        capture=capture,
    )
    return LLMTestCase(
        name=f"{case['case_id']} [Tactical Supervisor]",
        input=str(case["input"]),
        actual_output=f"{as_output_text(observed_output)}\n\n{format_tool_trace(tool_trace)}",
        expected_output=str(case["expected_output"]),
        context=context_strings(case),
        tools_called=make_tool_calls(tool_trace, component="supervisor"),
        expected_tools=expected_tools(case, component="supervisor"),
        token_cost=capture.token_cost,
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
    """Run and cache one live tactical Supervisor decision per case."""
    return cached_replay(
        _LIVE_CACHE,
        "supervisor",
        str(case["case_id"]),
        lambda: replay_supervisor_case(case, settings),
    )


def _assert_live_proposal_matches_expected(
    case: dict[str, Any], observed_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the single canonical proposal after deterministic target checks."""
    expected = case["expected_tools"]
    if not expected:
        assert case["tool_correctness_applicable"] is False
        assert case["expected_tool_correctness_pass"] is None
        assert observed_tools == []
        return []

    expected_call = expected[0]
    expected_args = expected_call["args"]
    matching = []
    for observed in observed_tools:
        arguments = observed.get("args", {})
        if not isinstance(arguments, dict):
            continue
        if observed.get("name") != expected_call["name"]:
            continue
        if arguments.get("selected_strategy") != expected_args["selected_strategy"]:
            continue
        if "target_version" in expected_args and (
            arguments.get("target_version") != expected_args["target_version"]
        ):
            continue
        if "target_files_hint" in expected_args:
            normalized_files = _safe_relative_file_hints(
                arguments.get("target_files_hint", []),
                limit=5,
            )
            expected_files = _safe_relative_file_hints(
                expected_args["target_files_hint"],
                limit=5,
            )
            if normalized_files != expected_files:
                continue
        matching.append(observed)

    assert len(matching) == 1, (
        f"Expected one live {expected_call['name']} proposal matching the canonical action; "
        f"observed {observed_tools!r}."
    )
    return matching


def _tool_signature(tool_calls: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return ordered canonical tool names and arguments without call metadata."""
    return [(str(call.get("name", "")), dict(call.get("args", {}) or {})) for call in tool_calls]


_STABLE_TOOL_ARGUMENT_KEYS = ("selected_strategy", "target_version", "target_files_hint")


def _project_supervisor_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only deterministic action fields for ToolCorrectness judging."""
    projected = []
    for call in tool_calls:
        arguments = call.get("args", {})
        if not isinstance(arguments, dict):
            raise TypeError("Supervisor proposal arguments must be an object.")
        projected.append(
            {
                "name": call.get("name", ""),
                "args": {
                    key: arguments[key] for key in _STABLE_TOOL_ARGUMENT_KEYS if key in arguments
                },
            }
        )
    return projected


def _tool_correctness_test_case(
    test_case: LLMTestCase,
    case: dict[str, Any],
    observed_tools: list[dict[str, Any]],
) -> LLMTestCase:
    """Build a metric view that leaves free-form evidence text to completion judging."""
    projected_case = {
        **case,
        "expected_tools": _project_supervisor_tool_calls(case["expected_tools"]),
    }
    return LLMTestCase(
        name=test_case.name,
        input=test_case.input,
        actual_output=test_case.actual_output,
        expected_output=test_case.expected_output,
        context=test_case.context,
        tools_called=make_tool_calls(
            _project_supervisor_tool_calls(observed_tools),
            component="supervisor",
        ),
        expected_tools=expected_tools(projected_case, component="supervisor"),
        token_cost=test_case.token_cost,
        additional_metadata=test_case.additional_metadata,
    )


def test_tool_correctness_view_ignores_flexible_action_text() -> None:
    """Preserve prose in observations while judging only deterministic arguments."""
    for case_id in (
        "supervisor-direct-version-bump-clean",
        "supervisor-workaround-pivot-breaking-change",
    ):
        case = next(item for item in _SUPERVISOR_CASES if item["case_id"] == case_id)
        expected_call = case["expected_tools"][0]
        observed_args = dict(expected_call["args"])
        observed_args["diagnostic_basis"] = "A different concise evidence summary."
        observed_args["rationale"] = "A different concise decision rationale."
        if "workaround_hypothesis" in observed_args:
            observed_args["workaround_hypothesis"] = "A different bounded workaround hypothesis."
        observed_tools = [{"name": expected_call["name"], "args": observed_args}]
        capture = ReplayCapture(
            case_id=case_id,
            component="supervisor",
            actual_output="ACCEPTED: metric projection fixture.",
            actual_tools=observed_tools,
        )

        full_case = build_supervisor_test_case(
            case,
            capture.actual_output,
            observed_tools,
            "metric_projection_test",
            capture,
        )
        metric_case = _tool_correctness_test_case(full_case, case, observed_tools)

        called_args = metric_case.tools_called[0].input_parameters
        expected_args = metric_case.expected_tools[0].input_parameters
        assert called_args == expected_args
        assert set(called_args) == set(_STABLE_TOOL_ARGUMENT_KEYS).intersection(
            expected_call["args"]
        )
        assert "diagnostic_basis" in full_case.tools_called[0].input_parameters
        assert "rationale" in full_case.tools_called[0].input_parameters
        assert "A different concise evidence summary." in full_case.actual_output


def test_live_proposal_matcher_keeps_one_canonical_action_from_a_repair_trace() -> None:
    """The metric view can score a repaired action without hiding the full trace."""
    case = next(
        item
        for item in _SUPERVISOR_CASES
        if item["case_id"] == "supervisor-direct-version-bump-clean"
    )
    expected = case["expected_tools"][0]
    invalid = {
        "name": expected["name"],
        "args": {**expected["args"], "target_version": "9.9.9"},
    }
    matching = _assert_live_proposal_matches_expected(case, [invalid, expected])

    assert matching == [expected]


def _assert_accepted_instruction(case: dict[str, Any], capture: ReplayCapture) -> None:
    """Check worker instructions remain bound to verified targets and policy."""
    verification = capture.typed_result["verification"]
    action_args = case["expected_tools"][0]["args"]
    strategy = action_args["selected_strategy"]
    if strategy == "escalate_to_portfolio":
        return

    instruction = verification.instruction
    assert instruction is not None
    group = case["replay"]["input"]["group"]
    target_package = group["vulnerable_component"]
    assert verification.target_package_name == target_package
    assert f"AUTHORIZED TARGET: {target_package}" in instruction
    assert "PROHIBITED OPERATIONS" in instruction

    if "target_version" in action_args:
        assert verification.selected_version == action_args["target_version"]
        assert f"EXACT VERSION OR HYPOTHESIS: {action_args['target_version']}" in instruction
    else:
        hypothesis = action_args["workaround_hypothesis"]
        assert f"EXACT VERSION OR HYPOTHESIS: {hypothesis}" in instruction
        for target_file in action_args["target_files_hint"]:
            assert target_file in instruction


@pytest.mark.eval
class TestSupervisorEval:
    """Evaluate live tactical proposals, verification, and decision quality."""

    @pytest.mark.parametrize(
        "case",
        _SUPERVISOR_CASES or [{}],
        ids=_SUPERVISOR_CASE_IDS or ["no_cases"],
    )
    def test_supervisor_live_replay_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run the production caller and judge its accepted tactical outcome."""
        if not case:
            pytest.skip("No golden Supervisor cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if not HAS_DEEPEVAL:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        matching_tools = _assert_live_proposal_matches_expected(case, capture.actual_tools)
        test_case = build_supervisor_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            "production_live",
            capture,
        )
        if case.get("tool_correctness_applicable", True):
            assert DeepEvalToolCorrectnessMetric is not None
            assert ToolCallParams is not None
            assert case.get("expected_tool_correctness_pass") is True
            tool_metric = DeepEvalToolCorrectnessMetric(
                threshold=1.0,
                evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
                should_consider_ordering=True,
                should_exact_match=False,
            )
            tool_metric_case = _tool_correctness_test_case(
                test_case,
                case,
                matching_tools,
            )
            _measure_expected(
                tool_metric,
                tool_metric_case,
                expected_pass=bool(case["expected_tool_correctness_pass"]),
                case_id=str(case["case_id"]),
                label="Supervisor tool correctness",
            )
        else:
            assert case["expected_tools"] == []
            assert case["expected_tool_correctness_pass"] is None

        if (
            case.get("task_completion_applicable", True)
            and DeepEvalTaskCompletionMetric is not None
        ):
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
                label="Supervisor task completion",
            )
        elif not case.get("task_completion_applicable", True):
            assert case.get("expected_completion_pass") is None


@pytest.mark.parametrize(
    "case",
    _SUPERVISOR_CASES or [{}],
    ids=_SUPERVISOR_CASE_IDS or ["no_cases"],
)
def test_supervisor_offline_production_replay(
    case: dict[str, Any],
    eval_settings: EvalSettings,
) -> None:
    """Exercise every tactical policy case without credentials or external services."""
    if not case:
        pytest.skip("No golden Supervisor cases available")

    scripted_tools = case["offline_fixture"]["actual_tools"]
    model = ScriptedReplayModel.from_tool_trace(scripted_tools)
    capture = replay_supervisor_case(case, eval_settings, llm=model)
    expected = case["expected_replay"]
    typed = capture.typed_result
    verification = typed["verification"]
    decision = typed["decision"]

    assert capture.component == "supervisor"
    assert capture.external_calls == []
    assert capture.actual_tools == scripted_tools
    assert _tool_signature(capture.actual_tools) == _tool_signature(scripted_tools)
    assert typed["model_invocations"] == model.invocation_count == expected["model_invocations"]
    assert capture.task_revision == case["replay"]["input"]["task"]["task_revision"]
    assert capture.attempt_id == case["replay"]["input"]["task"]["current_attempt_id"]
    if scripted_tools:
        assert model.bound_tool_names == _EXPECTED_BOUND_TOOL_NAMES

    assert (verification.accepted if verification is not None else None) == (
        expected["verification_accepted"]
    )
    assert (
        decision.decision_code.value
        if decision is not None and decision.decision_code is not None
        else None
    ) == expected["decision_code"]
    assert (len(decision.spawn_requests) if decision is not None else 0) == (
        expected["spawn_request_count"]
    )

    if expected["verification_accepted"] is True:
        assert verification is not None and verification.accepted is True
        assert decision is not None
        assert typed["errors"] == []
        _assert_accepted_instruction(case, capture)
    elif expected["verification_accepted"] is False:
        assert verification is not None and verification.accepted is False
        assert verification.instruction is None
        assert decision is None
        assert typed["staged_resolutions"] == []
    else:
        assert verification is None
        assert decision is None
        assert capture.actual_tools == []
        assert typed["model_factory_calls"] == 0
        assert typed["registry_resolution_calls"] == 0
        assert typed["staged_resolutions"] == []
        assert typed["task_before"] == typed["task_after"]
        assert typed["retry_diagnostics_before"] == typed["retry_diagnostics_after"]
        assert typed["retry_diagnostics_before"] is not None
        no_call_output = json.loads(capture.actual_output)
        assert no_call_output["qa_attribution_status"] == "inconclusive"
        assert no_call_output["registry_resolution_calls"] == 0
        assert no_call_output["model_factory_calls"] == 0
        assert no_call_output["task_state_unchanged"] is True
        assert no_call_output["retry_diagnostics_unchanged"] is True
        assert no_call_output["staged_resolution_count"] == 0
        return

    case_id = str(case["case_id"])
    if case_id == "supervisor-portfolio-escalation-peer-conflict":
        task_id = case["replay"]["input"]["task"]["task_id"]
        assert decision.next_node == "teardown"
        assert decision.unfixable_task_ids == [task_id]
        assert decision.task_status_updates[task_id] == TaskStatus.UNFIXABLE
        assert decision.new_constraints
        assert decision.spawn_requests == []
    elif case_id == "supervisor-workaround-pivot-breaking-change":
        assert decision.next_node == "workaround_subagent"
        assert len(decision.spawn_requests) == 1
        assert decision.spawn_requests[0].strategy.value == "code_workaround"
        assert decision.spawn_requests[0].instruction == verification.instruction
    elif decision is not None:
        assert decision.next_node == "update_subagent"

    if case_id == "supervisor-repair-loop-recovery":
        assert len(typed["model_invocation_messages"]) == 2
        second_prompt = "\n".join(
            str(message.get("content", "")) for message in typed["model_invocation_messages"][1]
        )
        assert (
            "Version 9.9.9 is not in the version_bump registry-verified candidate authorization."
            in second_prompt
        )
    if case_id == "supervisor-version-advance-after-test-fail":
        retry = typed["retry_diagnostics_by_task"][case["replay"]["input"]["task"]["task_id"]]
        assert retry.attempted_versions == ["6.0.0"]
        assert verification.selected_version == "6.1.2"


def test_supervisor_eval_case_labels_and_expected_action_contract() -> None:
    """Keep the ten stable cases and their live metric applicability explicit."""
    assert len(_SUPERVISOR_CASES) == 10
    assert len(set(_SUPERVISOR_CASE_IDS)) == 10
    no_call_cases = [case for case in _SUPERVISOR_CASES if not case["tool_correctness_applicable"]]
    assert [case["case_id"] for case in no_call_cases] == ["supervisor-inconclusive-qa-suppressed"]
    for case in _SUPERVISOR_CASES:
        if case["tool_correctness_applicable"]:
            assert case["expected_tool_correctness_pass"] is True
            assert case.get("task_completion_applicable", True) is True
            assert case["expected_completion_pass"] is True
            assert len(case["expected_tools"]) == 1
            assert case["expected_tools"][0]["name"] in _EXPECTED_BOUND_TOOL_NAMES
        else:
            assert case["expected_tools"] == []
            assert case["expected_tool_correctness_pass"] is None
            assert case["task_completion_applicable"] is False
            assert case["expected_completion_pass"] is None


def test_supervisor_test_case_builder_records_explicit_provenance() -> None:
    """Build a test record from explicit inputs without reading offline observations."""
    case = _SUPERVISOR_CASES[0]
    tools = [case["expected_tools"][0]]
    capture = ReplayCapture(
        case_id=case["case_id"],
        component="supervisor",
        actual_output="ACCEPTED: explicit builder-test capture.",
        actual_tools=tools,
        token_cost=0.001,
    )

    test_case = build_supervisor_test_case(
        case,
        capture.actual_output,
        capture.actual_tools,
        "builder_test",
        capture,
    )

    assert test_case.name == f"{case['case_id']} [Tactical Supervisor]"
    assert test_case.input == case["input"]
    assert test_case.expected_output == case["expected_output"]
    assert test_case.additional_metadata["component"] == "supervisor"
    assert test_case.additional_metadata["replay_source"] == "builder_test"
    assert test_case.token_cost == 0.001


def test_supervisor_suite_mapping_classifications() -> None:
    """Persist tactical pytest items and named cases under the supervisor suite."""
    from remediation_engine.evals.runner import SUITE_PATHS
    from tests.evals.conftest import _suite_from_nodeid, _suite_from_test_case

    case = _SUPERVISOR_CASES[0]
    capture = ReplayCapture(
        case_id=case["case_id"],
        component="supervisor",
        actual_output="ACCEPTED: mapping-test capture.",
        actual_tools=[],
    )
    test_case = build_supervisor_test_case(
        case,
        capture.actual_output,
        capture.actual_tools,
        "suite_mapping_test",
        capture,
    )
    node_id = (
        "tests/evals/test_supervisor_eval.py::"
        "test_supervisor_offline_production_replay[supervisor-direct-version-bump-clean]"
    )

    assert _suite_from_nodeid(node_id) == "supervisor"
    assert _suite_from_test_case(test_case) == "supervisor"
    assert SUITE_PATHS["supervisor"] == ["tests/evals/test_supervisor_eval.py"]


def test_supervisor_peer_conflict_rejection_falls_back_to_referral(
    eval_settings: EvalSettings,
) -> None:
    """Rejected source workarounds cannot bypass a required portfolio referral."""
    case = next(
        item
        for item in _SUPERVISOR_CASES
        if item["case_id"] == "supervisor-portfolio-escalation-peer-conflict"
    )
    invalid_workaround = {
        "name": "CodeWorkaroundSupervisorAction",
        "args": {
            "selected_strategy": "code_workaround",
            "diagnostic_basis": "The peer conflict needs a source workaround.",
            "workaround_hypothesis": "Edit source files even though QA did not authorize them.",
            "target_files_hint": ["src/index.js"],
            "rationale": "Try an unsupported source edit to avoid the peer conflict.",
        },
    }
    model = ScriptedReplayModel.from_tool_trace([invalid_workaround, invalid_workaround])

    capture = replay_supervisor_case(case, eval_settings, llm=model)
    verification = capture.typed_result["verification"]
    decision = capture.typed_result["decision"]
    task_id = case["replay"]["input"]["task"]["task_id"]

    assert model.invocation_count == 2
    assert verification is not None and verification.accepted is False
    assert verification.instruction is None
    assert decision is not None
    assert decision.decision_code.value == "PEER_CONFLICT_ESCALATION"
    assert decision.next_node == "teardown"
    assert decision.unfixable_task_ids == [task_id]
    assert decision.task_status_updates[task_id] == TaskStatus.UNFIXABLE
    assert decision.spawn_requests == []
