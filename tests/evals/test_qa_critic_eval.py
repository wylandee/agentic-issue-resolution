"""Phase 4: DeepEval and structural evaluation for QA Critic diagnostic accuracy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts.schemas import QACriticLLMOutput
from tests.evals.conftest import EvalSettings
from tests.evals.custom_metrics import (
    QAGuardrailConsistencyMetric,
    QAStructuredOutputMetric,
)

try:
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    LLMTestCase = None  # type: ignore[assignment,misc]
    LLMTestCaseParams = None  # type: ignore[assignment,misc]
    GEval = None  # type: ignore[assignment,misc]
    assert_test = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Golden Dataset Loader for Pytest Parametrization
# ---------------------------------------------------------------------------

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "qa_cases.json"


def _load_qa_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if c.get("eval_type", "qa_critic") == "qa_critic"]
        return []
    except Exception:
        return []


_QA_CASES = _load_qa_cases()
_QA_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_QA_CASES)]


# ---------------------------------------------------------------------------
# Prompt & Context Formatting Helpers (Matching Production qa_critic.py)
# ---------------------------------------------------------------------------


def build_qa_production_prompt(case: dict[str, Any]) -> str:
    """Build a structured QA prompt mirroring the production QA Critic investigator system prompt."""
    vuln_ctx = case.get("vulnerability_context", {})
    exec_ctx = case.get("execution_context", {})
    logs = case.get("execution_logs", {})
    changed_files = case.get("changed_files", [])
    worker_summary = case.get("worker_action_summary", "none")
    qa_policy = case.get("qa_policy", "version_bump")

    cves = vuln_ctx.get("cve_ids", [])
    ghsas = vuln_ctx.get("ghsa_ids", [])

    return (
        "=== QA REVIEW CONTEXT ===\n"
        f"- Target Group ID: {vuln_ctx.get('group_id')}\n"
        f"- Vulnerable Component: {vuln_ctx.get('vulnerable_component')}\n"
        f"- Target File: {vuln_ctx.get('file_path')}\n"
        f"- CVE IDs: {', '.join(cves) if cves else 'none'}\n"
        f"- GHSA IDs: {', '.join(ghsas) if ghsas else 'none'}\n"
        f"- QA Policy: {qa_policy}\n"
        f"- Changed Files: {', '.join(changed_files) if changed_files else 'none'}\n"
        f"- Worker Action Summary: {worker_summary}\n\n"
        "=== STEP 0 DETERMINISTIC EXECUTION LOGS ===\n"
        f"- Install Succeeded: {exec_ctx.get('install_passed')}\n"
        f"Install Log Excerpt:\n{logs.get('install_log', 'none')}\n\n"
        f"- Scanner Execution Status: {exec_ctx.get('scanner_execution_status')}\n"
        f"- Target Scanner Cleared: {exec_ctx.get('target_scanner_cleared')}\n"
        f"- Target Remaining Identifiers: {exec_ctx.get('target_remaining_identifiers', [])}\n"
        f"Scan Summary:\n{logs.get('scan_summary', 'none')}\n\n"
        f"- Tests Succeeded: {exec_ctx.get('tests_passed')}\n"
        f"Test Output Excerpt:\n{logs.get('test_output', 'none')}\n"
    )


# ---------------------------------------------------------------------------
# Structural & Deterministic Validation Helpers
# ---------------------------------------------------------------------------


def validate_qa_schema(llm_output_dict: dict[str, Any]) -> list[str]:
    """Validate that the LLM output conforms to the Pydantic QACriticLLMOutput schema."""
    violations: list[str] = []
    try:
        QACriticLLMOutput(**llm_output_dict)
    except Exception as exc:
        violations.append(f"QACriticLLMOutput contract violation: {exc}")
    return violations


def validate_qa_category_accuracy(
    llm_output_dict: dict[str, Any],
    expected_case: dict[str, Any],
) -> list[str]:
    """Validate that the QA decision and failure category match ground truth."""
    violations: list[str] = []
    expected_passed = expected_case.get("expected_passed")
    actual_passed = llm_output_dict.get("passed")

    if actual_passed != expected_passed:
        violations.append(f"Expected passed={expected_passed}, got passed={actual_passed}")

    expected_category = expected_case.get("expected_failure_category")
    actual_category = llm_output_dict.get("failure_category")

    if expected_category != actual_category:
        violations.append(
            f"Expected failure_category='{expected_category}', got '{actual_category}'"
        )

    expected_sec_verdict = expected_case.get("expected_security_review_verdict")
    sec_review = llm_output_dict.get("semantic_security_review")
    if expected_sec_verdict is not None:
        if sec_review is None:
            violations.append(
                f"Expected semantic_security_review with verdict='{expected_sec_verdict}', but got None."
            )
        else:
            actual_verdict = sec_review.get("verdict")
            if actual_verdict != expected_sec_verdict:
                violations.append(
                    f"Expected security review verdict='{expected_sec_verdict}', got '{actual_verdict}'"
                )

    return violations


# ---------------------------------------------------------------------------
# Test Suite: TestQACriticEval
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestQACriticEval:
    """Evaluation test suite for QA Critic diagnostics."""

    @pytest.mark.parametrize(
        "case",
        _QA_CASES or [{}],
        ids=_QA_CASE_IDS or ["no_cases"],
    )
    def test_qa_structured_output_validity(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """QACriticLLMOutput conforms to strict Pydantic invariants."""
        if not case:
            pytest.skip("No golden QA cases available")

        case_id = case.get("case_id", "unknown")
        llm_output = case.get("llm_qa_output", {})

        # 1. Deterministic Pydantic schema validation
        schema_violations = validate_qa_schema(llm_output)
        assert not schema_violations, f"Case '{case_id}' failed schema: {schema_violations}"

        # 2. Metric evaluation with QAStructuredOutputMetric
        test_case = LLMTestCase(
            name=f"{case_id} [Structured Output Invariants]",
            input=build_qa_production_prompt(case),
            actual_output=json.dumps(llm_output, indent=2),
            additional_metadata={
                "task_id": llm_output.get("task_id"),
                "passed": llm_output.get("passed"),
                "failure_category": llm_output.get("failure_category"),
                "retry_feedback": llm_output.get("retry_feedback"),
                "semantic_security_review": llm_output.get("semantic_security_review"),
                "test_attribution": llm_output.get("test_attribution"),
            },
        )

        metric = QAStructuredOutputMetric(threshold=1.0)
        score = metric.measure(test_case)
        assert score >= 1.0, f"Case '{case_id}' failed QAStructuredOutputMetric: {metric.reason}"

    @pytest.mark.parametrize(
        "case",
        _QA_CASES or [{}],
        ids=_QA_CASE_IDS or ["no_cases"],
    )
    def test_qa_failure_category_accuracy(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """LLM assigns the correct failure category (SECURITY_FLAG, PEER_CONFLICT, BREAKING_CHANGE)."""
        if not case:
            pytest.skip("No golden QA cases available")

        case_id = case.get("case_id", "unknown")
        llm_output = case.get("llm_qa_output", {})

        # 1. Deterministic category & pass/fail validation
        accuracy_violations = validate_qa_category_accuracy(llm_output, case)
        assert not accuracy_violations, (
            f"Case '{case_id}' failed category accuracy: {accuracy_violations}"
        )

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_qa_production_prompt(case)
            actual_output = json.dumps(
                {
                    "passed": llm_output.get("passed"),
                    "failure_category": llm_output.get("failure_category"),
                    "retry_feedback": llm_output.get("retry_feedback"),
                },
                indent=2,
            )
            expected_output = json.dumps(
                {
                    "passed": case.get("expected_passed"),
                    "failure_category": case.get("expected_failure_category"),
                },
                indent=2,
            )

            test_case = LLMTestCase(
                name=f"{case_id} [Failure Category & Diagnostic Accuracy]",
                input=prompt,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            category_geval = GEval(
                name="Failure Category & Diagnostic Accuracy",
                criteria=(
                    "Evaluate whether the QA Critic correctly determines 'passed' and categorizes "
                    "failures into the exact FailureCategory (SECURITY_FLAG, PEER_CONFLICT, BREAKING_CHANGE). "
                    "Verify that the Actual Output aligns with the Expected Output."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [category_geval])

    @pytest.mark.parametrize(
        "case",
        _QA_CASES or [{}],
        ids=_QA_CASE_IDS or ["no_cases"],
    )
    def test_qa_guardrail_consistency(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """LLM QA verdict survives deterministic policy guardrails without override."""
        if not case:
            pytest.skip("No golden QA cases available")

        case_id = case.get("case_id", "unknown")
        llm_output = case.get("llm_qa_output", {})

        test_case = LLMTestCase(
            name=f"{case_id} [QA Guardrail Consistency]",
            input=build_qa_production_prompt(case),
            actual_output=json.dumps(llm_output, indent=2),
            additional_metadata={
                "qa_policy": case.get("qa_policy"),
                "execution_context": case.get("execution_context", {}),
                "llm_passed": llm_output.get("passed"),
                "llm_failure_category": llm_output.get("failure_category"),
                "llm_security_review": llm_output.get("semantic_security_review"),
            },
        )

        metric = QAGuardrailConsistencyMetric(threshold=1.0)
        score = metric.measure(test_case)
        assert score >= 1.0, (
            f"Case '{case_id}' failed QAGuardrailConsistencyMetric: {metric.reason}"
        )

    @pytest.mark.parametrize(
        "case",
        _QA_CASES or [{}],
        ids=_QA_CASE_IDS or ["no_cases"],
    )
    def test_qa_semantic_security_review(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Semantic security review verdicts and evidence references are sound."""
        if not case:
            pytest.skip("No golden QA cases available")

        expected_verdict = case.get("expected_security_review_verdict")
        if expected_verdict is None:
            pytest.skip("Case does not require semantic security review evaluation")

        case_id = case.get("case_id", "unknown")
        llm_output = case.get("llm_qa_output", {})
        sec_review = llm_output.get("semantic_security_review")

        assert sec_review is not None, f"Case '{case_id}' missing semantic_security_review"
        assert sec_review.get("verdict") == expected_verdict, (
            f"Case '{case_id}' expected security verdict '{expected_verdict}', got '{sec_review.get('verdict')}'"
        )
        if expected_verdict == "pass":
            refs = sec_review.get("evidence_refs", [])
            assert refs and len(refs) > 0, (
                f"Case '{case_id}' with verdict='pass' must provide evidence_refs"
            )

        # Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_qa_production_prompt(case)
            test_case = LLMTestCase(
                name=f"{case_id} [Semantic Security Review Quality]",
                input=prompt,
                actual_output=json.dumps(sec_review, indent=2),
                expected_output=json.dumps(
                    {"verdict": expected_verdict},
                    indent=2,
                ),
            )

            review_geval = GEval(
                name="Semantic Security Review Quality",
                criteria=(
                    "Evaluate whether the semantic security review correctly assesses whether "
                    "the code workaround addresses the vulnerability sink. The verdict must match "
                    "the Expected Output and the reasoning must cite specific evidence."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [review_geval])

    @pytest.mark.parametrize(
        "case",
        _QA_CASES or [{}],
        ids=_QA_CASE_IDS or ["no_cases"],
    )
    def test_qa_retry_feedback_actionability(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """When passed=False, retry feedback provides clear and actionable diagnostic guidance."""
        if not case:
            pytest.skip("No golden QA cases available")

        if case.get("expected_passed") is True:
            pytest.skip("Case passed; retry feedback not applicable")

        case_id = case.get("case_id", "unknown")
        llm_output = case.get("llm_qa_output", {})
        feedback = llm_output.get("retry_feedback")

        assert feedback and len(str(feedback).strip()) > 10, (
            f"Case '{case_id}' missing actionable retry feedback"
        )

        # Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_qa_production_prompt(case)
            test_case = LLMTestCase(
                name=f"{case_id} [QA Retry Feedback Actionability]",
                input=prompt,
                actual_output=str(feedback),
            )

            actionability_geval = GEval(
                name="QA Retry Feedback Actionability",
                criteria=(
                    "Evaluate whether the QA retry feedback provides specific, actionable guidance "
                    "referencing the exact failure mechanism (e.g. remaining CVE, peer conflict, "
                    "broken API export, or test regression) to direct the next remediation attempt."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [actionability_geval])


# ---------------------------------------------------------------------------
# Standalone Adversarial Unit Tests (Adversarial Validation)
# ---------------------------------------------------------------------------


def test_qa_evaluator_catches_schema_violations() -> None:
    """Ensure validation helpers catch illegal passed/category/feedback combinations."""
    # 1. passed=True with failure_category
    bad_pass = {
        "task_id": "test-task",
        "passed": True,
        "failure_category": "security_flag",
        "retry_feedback": None,
    }
    violations = validate_qa_schema(bad_pass)
    assert any("passed=True" in v or "failure_category" in v for v in violations)

    # 2. passed=False without retry_feedback
    bad_fail = {
        "task_id": "test-task",
        "passed": False,
        "failure_category": "breaking_change",
        "retry_feedback": "",
    }
    violations_fail = validate_qa_schema(bad_fail)
    assert any("passed=False" in v or "retry_feedback" in v for v in violations_fail)


def test_qa_evaluator_catches_category_mismatch() -> None:
    """Ensure validate_qa_category_accuracy flags misclassified failure categories."""
    llm_output = {
        "task_id": "test-task",
        "passed": False,
        "failure_category": "breaking_change",
        "retry_feedback": "some feedback",
    }
    expected_case = {
        "expected_passed": False,
        "expected_failure_category": "peer_conflict",
    }
    violations = validate_qa_category_accuracy(llm_output, expected_case)
    assert any("Expected failure_category='peer_conflict'" in v for v in violations)


def test_qa_structured_output_metric_catches_invalid_verdict() -> None:
    """Ensure QAStructuredOutputMetric returns 0.0 when review verdict is invalid."""
    test_case = LLMTestCase(
        name="adversarial_test",
        input="test input",
        actual_output="test output",
        additional_metadata={
            "task_id": "test-task",
            "passed": True,
            "failure_category": None,
            "retry_feedback": None,
            "semantic_security_review": {
                "verdict": "INVALID_VERDICT",
                "evidence_refs": [],
            },
        },
    )
    metric = QAStructuredOutputMetric(threshold=1.0)
    score = metric.measure(test_case)
    assert score == 0.0
    assert "Invalid semantic_security_review.verdict" in metric.reason


def test_qa_guardrail_metric_catches_overrides() -> None:
    """Ensure QAGuardrailConsistencyMetric flags cases where deterministic rules would override LLM."""
    test_case = LLMTestCase(
        name="override_test",
        input="test input",
        actual_output="test output",
        additional_metadata={
            "qa_policy": "version_bump",
            "execution_context": {
                "install_passed": True,
                "scanner_execution_status": "success",
                "target_remaining_identifiers": ["CVE-2022-37601"],  # Finding remains!
                "tests_passed": True,
            },
            "llm_passed": True,  # LLM erroneously passed!
            "llm_failure_category": None,
            "llm_security_review": None,
        },
    )
    metric = QAGuardrailConsistencyMetric(threshold=1.0)
    score = metric.measure(test_case)
    assert score == 0.0
    assert "Remaining scanner findings" in metric.reason
