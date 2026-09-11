"""DeepEval task-completion evaluation for the triage agent."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from remediation_engine.contracts.schemas import Severity, TriageResult
from remediation_engine.settings import AppSettings
from remediation_engine.triage.agent import run_triage
from remediation_engine.triage.pipeline import select_issues_for_remediation
from tests.evals.conftest import EvalSettings
from tests.evals.eval_case_helpers import (
    as_output_text,
    case_metadata,
    context_strings,
    observations,
)
from tests.evals.golden_schema import load_golden_dataset
from tests.evals.replay_adapters import replay_triage_case
from tests.evals.replay_harness import (
    ReplayCapture,
    ScriptedReplayModel,
    build_system_context,
    build_vulnerability_group,
    cached_replay,
)

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
_TRIAGE_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="triage_cases")
    if case.get("eval_type", "triage") == "triage"
]
_TRIAGE_CASE_IDS = [case["case_id"] for case in _TRIAGE_CASES]
_DETERMINISTIC_ONLY_CASE_IDS = frozenset(
    {
        "triage-fallback-llm-timeout-to-deterministic",
        "triage-pipeline-selection-hallucinated-issue-id",
    }
)
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}


def _as_output_text(value: Any) -> str:
    """Keep the historical helper name while using shared serialization."""
    return as_output_text(value)


def build_triage_test_case(
    case: dict[str, Any],
    observed_output: Any | None = None,
    observed_tools: list[dict[str, Any]] | None = None,
    *,
    replay_source: str | None = None,
    capture: ReplayCapture | None = None,
) -> LLMTestCase:
    """Build a task-completion case from explicit observations.

    Args:
        case: Canonical triage golden case.
        observed_output: Production output supplied by a live adapter.  When
            omitted, the explicit offline fixture branch is used.
        observed_tools: Production tool events, normally empty for triage.
        replay_source: Optional observation provenance label.
        capture: Optional capture used only to enrich metadata.

    Returns:
        A DeepEval-compatible task case.
    """
    case_id = str(case.get("case_id", "unknown"))
    if observed_output is None and observed_tools is None:
        actual_output, actual_tools, source = observations(case)
    else:
        actual_output = as_output_text(observed_output or "")
        actual_tools = list(observed_tools or [])
        source = replay_source or "production_live"
    if not str(case.get("input", "")).strip() or not str(actual_output).strip():
        raise ValueError(f"Golden case {case_id!r} must contain input and an observed output.")

    metadata = case_metadata(
        case,
        component="triage",
        replay_source=source,
        actual_tools=actual_tools,
        capture=capture,
    )
    metadata.update(
        {
            "golden_kind": case.get("golden_kind"),
            "triage_metadata": case.get("triage_metadata", {}),
            "expected_negative": not bool(case.get("expected_completion_pass", True)),
        }
    )
    context = context_strings(
        case,
        f"Triage scenario: {case.get('scenario', '')}",
        f"Historical evidence: {json.dumps(case.get('historical_evidence', []), sort_keys=True)}",
    )
    return LLMTestCase(
        name=f"{case_id} [Triage Task Completion]",
        input=str(case["input"]),
        actual_output=str(actual_output),
        expected_output=str(case["expected_output"]),
        context=context,
        tools_called=[],
        token_cost=capture.token_cost if capture is not None else None,
        additional_metadata=metadata,
    )


def _live_capture(case: dict[str, Any], settings: EvalSettings) -> ReplayCapture:
    """Run one triage production replay and reuse it across metrics."""
    return cached_replay(
        _LIVE_CACHE,
        "triage",
        str(case["case_id"]),
        lambda: replay_triage_case(case, settings),
    )


@pytest.mark.eval
class TestTriageEval:
    """Evaluate live triage output after production guardrails."""

    @pytest.mark.parametrize("case", _TRIAGE_CASES or [{}], ids=_TRIAGE_CASE_IDS or ["no_cases"])
    def test_triage_live_replay_task_completion(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Judge a production ``run_triage`` result, never a golden observation."""
        if not case:
            pytest.skip("No golden triage cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if str(case.get("case_id")) in _DETERMINISTIC_ONLY_CASE_IDS:
            pytest.skip("Fault-injection boundary is covered by deterministic triage tests.")
        if not HAS_DEEPEVAL or DeepEvalTaskCompletionMetric is None:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        test_case = build_triage_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            replay_source="production_live",
            capture=capture,
        )
        metric = DeepEvalTaskCompletionMetric(
            threshold=0.70,
            model=eval_settings.judge_model,
            async_mode=False,
        )
        expected_pass = bool(case.get("expected_completion_pass", True))
        if expected_pass:
            if assert_test is not None:
                assert_test(test_case, [metric], run_async=False)
            else:
                metric.measure(test_case)
                assert metric.is_successful(), f"Case {case['case_id']!r} did not complete triage."
        else:
            metric.measure(test_case)
            assert not metric.is_successful(), (
                f"Case {case['case_id']!r} was expected to remain incomplete after guardrails."
            )

    @pytest.mark.parametrize(
        "case",
        [case for case in _TRIAGE_CASES if case.get("case_id") in _DETERMINISTIC_ONLY_CASE_IDS],
        ids=[
            case["case_id"]
            for case in _TRIAGE_CASES
            if case.get("case_id") in _DETERMINISTIC_ONLY_CASE_IDS
        ],
    )
    def test_triage_deterministic_fault_injection_boundaries(
        self,
        case: dict[str, Any],
    ) -> None:
        """Keep timeout and hallucinated-selection behavior deterministic."""
        group = build_vulnerability_group(case)
        context = build_system_context(case)
        if case.get("case_id") == "triage-fallback-llm-timeout-to-deterministic":
            settings = replace(AppSettings.from_env(), triage_llm_enabled=True)
            with patch("remediation_engine.triage.agent._llm_triage", return_value=None):
                result = run_triage(group, context, settings=settings)
            assert result.triage_method == "deterministic"
            assert result.recommended_issue_id == group.representative_issue_id
            return

        result = run_triage(
            group,
            context,
            settings=replace(AppSettings.from_env(), triage_llm_enabled=False),
        )
        hallucinated = result.model_copy(update={"recommended_issue_id": uuid4()})
        selected = select_issues_for_remediation([(group, hallucinated)])
        assert [issue.id for issue in selected] == [group.representative_issue_id]

    def test_triage_offline_guardrail_replay(
        self,
        eval_settings: EvalSettings,
    ) -> None:
        """Replay an under-ranked KEV verdict through the real triage guardrail."""
        case = next(
            case
            for case in _TRIAGE_CASES
            if case.get("case_id") == "triage-guardrail-drop-everything-kev-override"
        )
        group = build_vulnerability_group(case)
        raw_result = TriageResult(
            chain_of_thought="The observed prototype-pollution path appears difficult to trigger.",
            group_id=group.group_id,
            is_valid=False,
            false_positive_reason="The vulnerable path appears unreachable in normal traffic.",
            original_severity=Severity.MEDIUM,
            revised_priority=Severity.MEDIUM,
            is_unreachable_code=True,
            priority_reasoning="The raw model under-ranked the finding.",
            validity_confidence_score=0.4,
            priority_confidence_score=0.4,
            recommended_issue_id=group.representative_issue_id,
            triage_method="llm",
        )
        model = ScriptedReplayModel.from_structured_result(raw_result)

        capture = replay_triage_case(case, eval_settings, llm=model)

        result = capture.typed_result
        assert capture.actual_tools == []
        assert capture.external_calls == []
        assert model.structured_invocation_count == 1
        assert result is not None
        assert result.is_valid is True
        assert result.false_positive_reason is None
        assert result.revised_priority == Severity.CRITICAL
        assert result.priority_confidence_score == 1.0
        assert result.triage_method == "llm"
        assert result.recommended_issue_id == group.representative_issue_id
        assert "CISA KEV" in result.priority_reasoning
