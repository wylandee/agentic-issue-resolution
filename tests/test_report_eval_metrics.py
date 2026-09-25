"""Focused tests for report-evaluation judge inputs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.evals import test_report_eval


def test_report_metric_cases_separate_facts_from_policy(monkeypatch) -> None:
    """Factual judges get source only; GEval gets separate policy and facts."""

    class FakeLLMTestCase:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(test_report_eval, "LLMTestCase", FakeLLMTestCase)
    case = test_report_eval._load_golden_cases()[0]
    report = test_report_eval.render_report(case)
    source_evidence, policy_context = test_report_eval.build_report_context(case)
    factual_case, policy_case = test_report_eval._build_report_metric_cases(
        case,
        report,
        [source_evidence, policy_context],
    )

    assert factual_case.input == source_evidence
    assert factual_case.actual_output == report
    assert getattr(factual_case, "expected_output", None) is None
    assert factual_case.context == [source_evidence]
    assert factual_case.retrieval_context == [source_evidence]
    assert policy_case.input == source_evidence
    assert policy_case.actual_output == report
    assert policy_case.context == [source_evidence, policy_context]
    policy_expected = json.loads(policy_case.expected_output)
    assert "code_detail_diff_paths" not in policy_expected
    assert "forbidden_code_detail_diff_paths" not in policy_expected
    assert "package.json" not in policy_context
    assert "expected_contract" not in source_evidence
    assert "report_format" not in source_evidence


def test_summarization_metric_receives_case_specific_questions(monkeypatch) -> None:
    """The judge receives factual yes/no questions answerable from both texts."""

    class FakeMetric:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(test_report_eval, "HAS_DEEPEVAL", True)
    monkeypatch.setattr(test_report_eval, "HallucinationMetric", FakeMetric)
    monkeypatch.setattr(test_report_eval, "FaithfulnessMetric", FakeMetric)
    monkeypatch.setattr(test_report_eval, "SummarizationMetric", FakeMetric)
    monkeypatch.setattr(test_report_eval, "GEval", FakeMetric)
    monkeypatch.setattr(
        test_report_eval,
        "LLMTestCaseParams",
        SimpleNamespace(
            INPUT="input",
            ACTUAL_OUTPUT="actual_output",
            EXPECTED_OUTPUT="expected_output",
            CONTEXT="context",
        ),
    )
    case = test_report_eval._load_golden_cases()[0]
    metrics = test_report_eval._build_metrics(
        SimpleNamespace(judge_model="test-model"),
        case,
    )

    questions = metrics[2].kwargs["assessment_questions"]
    assert questions == test_report_eval._report_summarization_questions(case)
    assert questions[0] == "Are there exactly 5 scanner findings across 4 vulnerability groups?"
    assert (
        questions[1]
        == "Are 4 vulnerability groups successfully remediated and 0 vulnerability groups "
        "requiring follow-up?"
    )
    assert any(
        "CVE-2026-30001 -> vulnerable package base64url" in question for question in questions
    )
    assert all("Does the report" not in question for question in questions)


def test_code_detail_files_are_source_only_for_package_removal() -> None:
    """Manifests stay in the main row, while code details list accepted source paths."""
    case = next(
        case
        for case in test_report_eval._load_golden_cases()
        if case["case_id"] == "no_fix_package_removal"
    )
    report = test_report_eval.render_report(case)
    parsed = test_report_eval.parse_report(report)
    table_files = parsed["successful_rows"][0]["Files Changed"]
    _, policy_context = test_report_eval.build_report_context(case)

    assert "package.json" in table_files
    assert "package-lock.json" in table_files
    assert parsed["code_detail_file_paths"] == ["routes/b2bOrder.ts"]
    assert "package.json" not in parsed["code_detail_text"]
    assert "package-lock.json" not in parsed["code_detail_text"]
    assert not test_report_eval.validate_report_contract(report, case["expected_contract"])
    assert "package.json" not in policy_context
    assert "forbidden_code_detail_diff_paths" not in policy_context

    tampered_report = report.replace(
        "- **Files changed:** routes/b2bOrder.ts",
        "- **Files changed:** package.json, routes/b2bOrder.ts",
        1,
    )
    violations = test_report_eval.validate_report_contract(
        tampered_report, case["expected_contract"]
    )
    assert any(
        "manifest leaked into Code workaround details: package.json" in item for item in violations
    )


def test_live_report_evaluates_geval_after_factual_threshold_failure(monkeypatch) -> None:
    """A factual threshold failure does not skip the separate GEval case."""

    class FakeLLMTestCase:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    factual_case = object()
    policy_case = object()
    metrics = [object(), object(), object(), object()]
    calls: list[tuple[Any, list[Any]]] = []

    def fake_assert_test(test_case: Any, selected_metrics: list[Any]) -> None:
        calls.append((test_case, selected_metrics))
        if test_case is factual_case:
            raise AssertionError("factual threshold failure")

    monkeypatch.setattr(test_report_eval, "HAS_DEEPEVAL", True)
    monkeypatch.setattr(test_report_eval, "LLMTestCase", FakeLLMTestCase)
    monkeypatch.setattr(test_report_eval, "render_report", lambda case: "report")
    monkeypatch.setattr(test_report_eval, "validate_report_contract", lambda report, contract: [])
    monkeypatch.setattr(
        test_report_eval,
        "build_report_context",
        lambda case: ["evidence", "policy"],
    )
    monkeypatch.setattr(
        test_report_eval,
        "_build_report_metric_cases",
        lambda case, report, context: (factual_case, policy_case),
    )
    monkeypatch.setattr(
        test_report_eval,
        "_build_metrics",
        lambda settings, case: metrics,
    )
    monkeypatch.setattr(test_report_eval, "assert_test", fake_assert_test)

    case = {"case_id": "fixture", "expected_contract": {}}
    settings = SimpleNamespace(is_live=True, openai_api_key="test", judge_model="test-model")
    with pytest.raises(pytest.fail.Exception, match="factual threshold failure"):
        test_report_eval.TestReportNodeEval().test_report_uses_only_the_four_requested_metrics(
            case,
            settings,
        )

    assert calls == [
        (factual_case, metrics[:3]),
        (policy_case, metrics[3:]),
    ]


def test_report_evidence_documents_build_for_all_golden_cases() -> None:
    """Every curated case produces source facts and policy guidance."""
    for case in test_report_eval._load_golden_cases():
        source_evidence, policy_context = test_report_eval.build_report_context(case)

        assert f"Run ID: {case['case_id']}." in source_evidence
        assert "There are exactly" in source_evidence
        assert "Constraint-adherence rules follow" in policy_context
        assert "File-path scope is checked deterministically" in policy_context
        for group in case["fixture"].get("groups", []):
            if group.get("package"):
                assert group["package"] in source_evidence
