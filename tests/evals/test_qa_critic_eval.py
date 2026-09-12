"""DeepEval evaluation suite for the task-scoped QA Critic."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.evals.adapters import ToolCall
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
from tests.evals.replay_adapters import replay_qa_case
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


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "qa_cases.json"
_QA_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="qa_cases")
    if case.get("eval_type") == "qa_critic"
]
_QA_CASE_IDS = [case["case_id"] for case in _QA_CASES]
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert canonical records into the local DeepEval call type."""
    return make_tool_calls(raw_calls, component="QA")


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render an observed trace for the completion judge."""
    return format_tool_trace(raw_calls)


def _format_provenance(case: dict[str, Any]) -> str:
    """Render compact provenance context."""
    evidence = case.get("historical_evidence", [])
    evidence_text = (
        "\n".join(f"- {item}" for item in evidence) if isinstance(evidence, list) else ""
    )
    return f"{case.get('provenance', '')}\n{evidence_text}".strip()


def build_qa_production_prompt(case: dict[str, Any]) -> str:
    """Build compact deterministic input used by the evaluation judge."""
    vulnerability = case.get("vulnerability_context", {})
    execution = case.get("execution_context", {})
    vulnerability = vulnerability if isinstance(vulnerability, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    cves = vulnerability.get("cve_ids", []) or []
    ghsas = vulnerability.get("ghsa_ids", []) or []
    task = case.get(
        "completion_task",
        "Evaluate deterministic evidence and emit one evidence-backed QA evaluation.",
    )
    expected_verdict = case.get("expected_qa_verdict", {})
    expected_passed = expected_verdict.get("passed")
    expected_category = expected_verdict.get("failure_category")
    goal_direction = (
        f"If the candidate change is flawed, installation fails, or tests fail, the required auditor task is to reject it by emitting passed=false and failure_category={expected_category}."
        if expected_passed is False
        else "The required auditor task is to verify the candidate change and emit passed=true."
    )

    def status(value: Any, *, skipped: bool = False) -> str:
        if skipped:
            return "SKIPPED"
        if value is None:
            return "NOT_RUN"
        return "PASS" if bool(value) else "FAIL"

    scanner_status = str(execution.get("scanner_execution_status", "not_run")).lower()
    scan_status = (
        "SKIPPED"
        if scanner_status == "not_run" and case.get("qa_policy") == "no_fix_package_removal"
        else "PASS"
        if scanner_status == "success" and execution.get("target_scanner_cleared") is True
        else "FAIL"
        if scanner_status in {"success", "failed", "failure", "error", "unparseable"}
        else "NOT_RUN"
    )
    return (
        "=== QA AUDITOR ROLE & OBJECTIVE ===\n"
        "You are evaluating a QA Auditor subagent. Inspect compact deterministic facts "
        f"and render an honest evaluation verdict. {goal_direction}\n\n"
        "=== QA CRITIC TASK ===\n"
        f"{task}\n\n"
        "=== ASSIGNED GROUP ===\n"
        f"- Group ID: {vulnerability.get('group_id')}\n"
        f"- Component: {vulnerability.get('vulnerable_component')}\n"
        f"- Target file: {vulnerability.get('file_path')}\n"
        f"- CVE IDs: {', '.join(cves) if cves else 'none'}\n"
        f"- GHSA IDs: {', '.join(ghsas) if ghsas else 'none'}\n"
        f"- QA policy: {case.get('qa_policy')}\n"
        f"- Changed files: {', '.join(case.get('changed_files', []) or []) or 'none'}\n\n"
        "=== DETERMINISTIC QA FACTS ===\n"
        f"- Install: {status(execution.get('install_passed'))}\n"
        f"- Security scan: {scan_status}\n"
        f"- Unit tests: {status(execution.get('tests_passed'))}\n"
        f"- Install exit code: {execution.get('install_exit_code', 'unknown')}\n"
        f"- Install error category: {execution.get('install_error_category', 'none')}\n"
        f"- Scanner execution status: {execution.get('scanner_execution_status', 'not_run')}\n"
        f"- Remaining target identifiers: {execution.get('target_remaining_identifiers', [])}\n"
        f"- Remaining identifier count: {len(execution.get('target_remaining_identifiers', []) or [])}\n"
        f"- Test failure count: {execution.get('test_failure_count', 'unknown')}\n"
        f"- Package manifest state: {execution.get('package_manifest_state', 'unknown')}\n"
        f"- Package graph state: {execution.get('package_graph_state', 'unknown')}\n\n"
        "=== EXPECTED COMPLETION CONTRACT ===\n"
        f"{case['expected_output']}\n"
    )


def build_qa_test_case(
    case: dict[str, Any],
    observed_output: Any | None = None,
    observed_tools: list[dict[str, Any]] | None = None,
    *,
    replay_source: str | None = None,
    capture: ReplayCapture | None = None,
) -> LLMTestCase:
    """Construct a QA test case from explicit live or offline observations."""
    if observed_output is None and observed_tools is None:
        final_output, tool_trace, source = observations(case)
    else:
        final_output = as_output_text(observed_output or "")
        tool_trace = list(observed_tools or [])
        source = replay_source or "production_live"
    tools_called = _make_tool_calls(tool_trace)
    expected = expected_tools(case, component="QA")
    verdict_summary = ""
    if capture and capture.typed_result:
        ev = capture.typed_result
        passed = getattr(ev, "passed", None)
        category = getattr(ev, "failure_category", None)
        verdict_summary = f"QA AUDIT VERDICT: passed={passed} (failure_category={category})\n\n"
    elif isinstance(observed_output, Mapping):
        passed = observed_output.get("passed")
        category = observed_output.get("failure_category")
        verdict_summary = f"QA AUDIT VERDICT: passed={passed} (failure_category={category})\n\n"
    metadata = case_metadata(
        case,
        component="qa_critic",
        replay_source=source,
        actual_tools=tool_trace,
        capture=capture,
    )
    metadata.update(
        {
            "qa_policy": case.get("qa_policy"),
            "expected_qa_verdict": case.get("expected_qa_verdict", {}),
            "expected_task_completion_pass": bool(case.get("expected_task_completion_pass", True)),
            "tool_correctness_applicable": bool(case.get("tool_correctness_applicable", True)),
        }
    )
    return LLMTestCase(
        name=f"{case['case_id']} [QA Critic]",
        input=build_qa_production_prompt(case),
        actual_output=f"{verdict_summary}{final_output}\n\n{_format_tool_trace(tool_trace)}",
        expected_output=str(case["expected_output"]),
        context=context_strings(
            case,
            f"Golden provenance:\n{_format_provenance(case)}",
            f"Expected QA output:\n{json.dumps(case.get('expected_qa_verdict', {}), indent=2)}",
        ),
        tools_called=tools_called,
        expected_tools=expected or None,
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
    """Measure an expected-positive or expected-negative metric result."""
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
    """Run one QA node replay and share it between the two metrics."""
    return cached_replay(
        _LIVE_CACHE,
        "qa_critic",
        str(case["case_id"]),
        lambda: replay_qa_case(case, settings),
    )


def _tool_signature(tool_events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return tool names and arguments without fixture output text."""
    return [
        (str(event.get("name", "")), dict(event.get("args", {}) or {})) for event in tool_events
    ]


@pytest.mark.eval
class TestQACriticEval:
    """Evaluate the real QA node with independent expected traces."""

    @pytest.mark.parametrize("case", _QA_CASES or [{}], ids=_QA_CASE_IDS or ["no_cases"])
    def test_qa_critic_live_replay_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run production QA once, then judge its output and captured tools."""
        if not case:
            pytest.skip("No golden QA cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if not HAS_DEEPEVAL:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        test_case = build_qa_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            replay_source="production_live",
            capture=capture,
        )
        if (
            bool(case.get("tool_correctness_applicable", True))
            and DeepEvalToolCorrectnessMetric is not None
        ):
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
                    label="QA tool correctness",
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
                expected_pass=bool(case.get("expected_task_completion_pass", True)),
                case_id=str(case["case_id"]),
                label="QA task completion",
            )


@pytest.mark.parametrize("case", _QA_CASES or [{}], ids=_QA_CASE_IDS or ["no_cases"])
def test_qa_critic_offline_production_replay(
    case: dict[str, Any],
    eval_settings: EvalSettings,
) -> None:
    """Exercise QA evidence review and its bounded terminal path offline."""
    if not case:
        pytest.skip("No golden QA cases available")

    case_id = str(case["case_id"])
    fixture = case.get("offline_fixture", {})
    historical_tools = fixture.get("actual_tools", []) if isinstance(fixture, dict) else []
    assert isinstance(historical_tools, list)
    if case_id == "qa_surrender_max_tool_call_rounds":
        scripted_tools = [
            {"name": "query_qa_logs", "args": {"log_type": "install"}} for _ in range(24)
        ]
        model = ScriptedReplayModel.from_tool_trace(scripted_tools)
    else:
        scripted_tools = historical_tools
        model = ScriptedReplayModel.from_tool_trace(scripted_tools)

    capture = replay_qa_case(case, eval_settings, llm=model)

    assert _tool_signature(capture.actual_tools) == _tool_signature(scripted_tools)
    assert model.invocation_count == len(scripted_tools)
    if case_id == "qa_surrender_max_tool_call_rounds":
        assert len(model.invocation_messages) >= 4
        fourth_messages = model.invocation_messages[3]
        scratchpads = [
            message["content"]
            for message in fourth_messages
            if message["additional_kwargs"].get("remediation_engine_scratchpad")
        ]
        assert scratchpads
        assert "QA_REVIEW" in scratchpads[-1]
    assert set(event["name"] for event in capture.actual_tools) <= {
        "query_qa_logs",
        "generate_workspace_diff",
        "read_file_context",
        "search_codebase_pattern",
        "inspect_ast_symbol",
        "emit_qa_evaluation",
    }
    if case_id == "qa_surrender_max_tool_call_rounds":
        assert capture.typed_result is not None
        assert capture.typed_result.passed is False
        assert (
            "maximum tool-call rounds"
            in "\n".join([*capture.errors, capture.actual_output]).lower()
        )
        assert not any(event["name"] == "emit_qa_evaluation" for event in capture.actual_tools)
        return

    expected = case.get("expected_qa_verdict", {})
    assert capture.typed_result is not None
    gates = capture.typed_result.deterministic_gates
    execution = case.get("execution_context", {})
    assert gates.install_passed is bool(execution.get("install_passed"))
    assert gates.tests_passed is bool(execution.get("tests_passed"))
    policy = str(case.get("qa_policy", ""))
    expected_remaining = set(execution.get("target_remaining_identifiers", []) or [])
    expected_scan_status = str(execution.get("scanner_execution_status", "not_run"))
    if expected_scan_status == "success" or policy != "no_fix_package_removal":
        assert set(gates.target_remaining_identifiers) == expected_remaining
    expected_cleared = execution.get("target_scanner_cleared")
    if isinstance(expected_cleared, bool) and (
        expected_scan_status == "success" or policy != "no_fix_package_removal"
    ):
        assert gates.target_scanner_cleared is expected_cleared
    actual_scan_status = getattr(
        gates.scanner_execution_status,
        "value",
        gates.scanner_execution_status,
    )
    if expected_scan_status == "success" or (
        policy == "no_fix_package_removal"
        and expected_scan_status in {"not_run", "failed", "failure", "error"}
    ):
        assert actual_scan_status == "success"
    elif expected_scan_status in {"failed", "failure", "error"}:
        assert actual_scan_status == "unparseable"
    else:
        assert actual_scan_status in {"not_run", "unparseable"}
    for field in ("package_manifest_state", "package_graph_state"):
        expected_state = execution.get(field)
        if expected_state is not None and policy.startswith("no_fix_"):
            assert getattr(gates, field) == expected_state

    expected_semantic = expected.get("semantic_security_review")
    if expected_semantic:
        assert capture.typed_result.semantic_security_review is not None
        assert capture.typed_result.semantic_security_review.verdict.value == expected_semantic
    if "fail_unit_tests" in case_id:
        assert capture.typed_result.failure_evidence is not None
        assert capture.typed_result.failure_evidence.raw_excerpt
    assert any(event["name"] == "emit_qa_evaluation" for event in capture.actual_tools)
    assert capture.typed_result.passed is bool(expected.get("passed"))
    actual_category = capture.typed_result.failure_category
    actual_category = getattr(actual_category, "value", actual_category)
    expected_category = expected.get("failure_category")
    if actual_category != expected_category:
        assert expected_category == "peer_conflict"
        assert actual_category == "security_flag"
        assert "install failed" in capture.typed_result.retry_feedback.lower()
