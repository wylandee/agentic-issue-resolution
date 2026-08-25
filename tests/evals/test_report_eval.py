"""Phase 1: DeepEval and structural evaluation for Report Node executive narratives."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.evals.conftest import EvalSettings

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    LLMTestCase = None  # type: ignore[assignment,misc]
    assert_test = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Golden Dataset Loader for Pytest Parametrization
# ---------------------------------------------------------------------------

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "report_cases.json"


def _load_golden_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


_REPORT_CASES = _load_golden_cases()
_REPORT_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_REPORT_CASES)]


# ---------------------------------------------------------------------------
# Prompt & Context Formatting Helpers (Matching Production report_node.py)
# ---------------------------------------------------------------------------


def build_production_prompt(evidence_payload: dict[str, Any]) -> str:
    """Build the exact prompt template used by report_node._generate_executive_narrative."""
    evidence = json.dumps(evidence_payload, sort_keys=True, default=str)
    return (
        "Write a concise executive narrative for a human reader of a software security remediation run.\n"
        "Use only the deterministic evidence below. Do not add facts, calculate metrics, change statuses, "
        "or recommend actions. Do not use a heading. Write 3 to 6 short paragraphs.\n\n"
        f"Deterministic evidence:\n{evidence[:16000]}"
    )


def build_context_documents(evidence_payload: dict[str, Any]) -> list[str]:
    """Build ground-truth context documents including structured JSON and factual summary."""
    evidence_json = json.dumps(evidence_payload, indent=2, sort_keys=True)
    metrics = evidence_payload.get("metrics", {})
    changed_files = evidence_payload.get("changed_files", [])
    errors = evidence_payload.get("errors", [])
    strategies = evidence_payload.get("strategies", {})

    summary = (
        f"Remediation status: {evidence_payload.get('status')}. "
        f"Overall label: {evidence_payload.get('overall_label')}. "
        f"Actionable groups: {metrics.get('actionable_groups')}, "
        f"Fixed: {metrics.get('groups_fixed')}, "
        f"Unresolved: {metrics.get('groups_unresolved')}, "
        f"Inconclusive: {metrics.get('groups_inconclusive')}. "
        f"Strategies: {json.dumps(strategies)}. "
        f"Errors count: {len(errors)}. Errors list: {errors}. "
        f"Modified files count: {len(changed_files)}. Modified files list: {changed_files}."
    )
    return [summary, evidence_json]


# ---------------------------------------------------------------------------
# Structural Validation Helpers (Offline-first / Deterministic)
# ---------------------------------------------------------------------------

_PRESCRIPTIVE_PATTERNS = [
    re.compile(r"\b(?:we recommend|recommend(?:s|ing)? that|should immediately)\b", re.IGNORECASE),
    re.compile(r"\b(?:recommend manual review|we suggest reverting)\b", re.IGNORECASE),
    re.compile(r"\b(?:recommend rolling back|recommend checking)\b", re.IGNORECASE),
]

_CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


def validate_forbidden_claims(narrative: str, forbidden_claims: list[str]) -> list[str]:
    """Return a list of forbidden claim strings found in the narrative."""
    violations: list[str] = []
    narrative_lower = narrative.casefold()
    for claim in forbidden_claims:
        if claim.casefold() in narrative_lower:
            violations.append(claim)
    return violations


def validate_expected_coverage(narrative: str, expected_coverage: list[str]) -> list[str]:
    """Return a list of expected coverage terms missing from the narrative."""
    missing: list[str] = []
    narrative_lower = narrative.casefold()
    for item in expected_coverage:
        if item.casefold() not in narrative_lower:
            missing.append(item)
    return missing


def validate_negative_constraints(narrative: str, evidence_payload: dict[str, Any]) -> list[str]:
    """Check negative constraints: headings, ungrounded CVEs, and prescriptive language."""
    violations: list[str] = []

    # 1. No Markdown headings (# ...)
    for line_num, line in enumerate(narrative.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            violations.append(f"Line {line_num} contains a markdown heading: {line.strip()}")

    # 2. No ungrounded CVEs
    evidence_text = json.dumps(evidence_payload, default=str)
    evidence_cves = {cve.upper() for cve in _CVE_PATTERN.findall(evidence_text)}
    narrative_cves = {cve.upper() for cve in _CVE_PATTERN.findall(narrative)}
    ungrounded_cves = narrative_cves - evidence_cves
    if ungrounded_cves:
        violations.append(f"Narrative contains ungrounded CVEs: {sorted(ungrounded_cves)}")

    # 3. No prescriptive recommendation phrases
    for pattern in _PRESCRIPTIVE_PATTERNS:
        match = pattern.search(narrative)
        if match:
            violations.append(f"Narrative contains prescriptive language: '{match.group(0)}'")

    return violations


# ---------------------------------------------------------------------------
# Test Suite: TestReportNodeEval (Parametrized by Golden Case)
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestReportNodeEval:
    """Evaluation suite for Report Node executive narratives."""

    @pytest.mark.parametrize(
        "case",
        _REPORT_CASES or [{}],
        ids=_REPORT_CASE_IDS or ["no_cases"],
    )
    def test_narrative_no_hallucination(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Narrative only contains facts from _evidence_payload (no hallucinated claims)."""
        if not case:
            pytest.skip("No golden report cases available")

        case_id = case.get("case_id", "unknown")
        narrative = case.get("generated_narrative", "")
        expected_output = case.get("expected_output", "")
        forbidden = case.get("forbidden_claims", [])
        evidence = case.get("evidence_payload", {})

        # 1. Deterministic check: forbidden claims must not appear
        forbidden_found = validate_forbidden_claims(narrative, forbidden)
        assert not forbidden_found, f"Case '{case_id}' violated forbidden claims: {forbidden_found}"

        # 2. Live evaluation with DeepEval judge
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_production_prompt(evidence)
            context = build_context_documents(evidence)

            test_case = LLMTestCase(
                name=f"{case_id} [Hallucination & Faithfulness]",
                input=prompt,
                actual_output=narrative,
                expected_output=expected_output,
                context=context,
                retrieval_context=context,
            )

            # Hallucination metric: lower is better, threshold <= 0.3
            hallucination = HallucinationMetric(
                threshold=0.3,
                model=eval_settings.judge_model,
                include_reason=True,
                verbose_mode=True,
            )

            # Faithfulness metric: higher is better, threshold >= 0.70
            faithfulness = FaithfulnessMetric(
                threshold=0.70,
                model=eval_settings.judge_model,
                include_reason=True,
                verbose_mode=True,
            )

            assert_test(test_case, [hallucination, faithfulness])

    @pytest.mark.parametrize(
        "case",
        _REPORT_CASES or [{}],
        ids=_REPORT_CASE_IDS or ["no_cases"],
    )
    def test_narrative_covers_key_findings(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Narrative covers all key findings, metrics, and packages accurately."""
        if not case:
            pytest.skip("No golden report cases available")

        case_id = case.get("case_id", "unknown")
        narrative = case.get("generated_narrative", "")
        expected_output = case.get("expected_output", "")
        expected_coverage = case.get("expected_coverage", [])
        evidence = case.get("evidence_payload", {})

        # 1. Deterministic check: every expected coverage term must appear
        missing_terms = validate_expected_coverage(narrative, expected_coverage)
        assert not missing_terms, (
            f"Case '{case_id}' missed expected coverage items: {missing_terms}"
        )

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_production_prompt(evidence)

            test_case = LLMTestCase(
                name=f"{case_id} [Key Findings Coverage]",
                input=prompt,
                actual_output=narrative,
                expected_output=expected_output,
            )

            coverage_geval = GEval(
                name="Finding Coverage & Accuracy",
                criteria=(
                    "Evaluate whether the Actual Output accurately covers all key findings, "
                    "remediation statuses, metrics, and package modifications present in the "
                    "Expected Output and Input without omitting critical outcomes."
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

            assert_test(test_case, [coverage_geval])

    @pytest.mark.parametrize(
        "case",
        _REPORT_CASES or [{}],
        ids=_REPORT_CASE_IDS or ["no_cases"],
    )
    def test_narrative_respects_negative_constraints(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Narrative does not recommend actions, invent CVEs, or use markdown headings."""
        if not case:
            pytest.skip("No golden report cases available")

        case_id = case.get("case_id", "unknown")
        narrative = case.get("generated_narrative", "")
        evidence = case.get("evidence_payload", {})

        # 1. Deterministic check: negative constraints validation
        violations = validate_negative_constraints(narrative, evidence)
        assert not violations, f"Case '{case_id}' violated negative constraints: {violations}"

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_production_prompt(evidence)
            context = build_context_documents(evidence)

            test_case = LLMTestCase(
                name=f"{case_id} [Negative Constraints]",
                input=prompt,
                actual_output=narrative,
                context=context,
            )

            constraint_geval = GEval(
                name="Report Constraint Adherence",
                criteria=(
                    "The output must not invent CVE IDs not in the evidence, "
                    "change task statuses, calculate metrics not in the evidence, "
                    "or recommend actions. It must not use markdown headings."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [constraint_geval])


# ---------------------------------------------------------------------------
# Unit tests for the evaluation helpers themselves (adversarial validation)
# ---------------------------------------------------------------------------


def test_evaluator_catches_violations() -> None:
    """Ensure validation helpers correctly catch headings, bad CVEs, and forbidden text."""
    bad_narrative = (
        "# Executive Summary\n"
        "We recommend immediately deploying CVE-9999-99999.\n"
        "All 100 vulnerabilities are completely fixed."
    )
    evidence: dict[str, Any] = {"status": "completed", "errors": []}

    forbidden = ["completely fixed", "CVE-9999-99999"]
    forbidden_found = validate_forbidden_claims(bad_narrative, forbidden)
    assert "completely fixed" in forbidden_found
    assert "CVE-9999-99999" in forbidden_found

    negative_violations = validate_negative_constraints(bad_narrative, evidence)
    assert any("heading" in v for v in negative_violations)
    assert any("CVE-9999-99999" in v for v in negative_violations)
    assert any("prescriptive language" in v for v in negative_violations)

    missing = validate_expected_coverage(bad_narrative, ["missing_pkg", "Successful"])
    assert "missing_pkg" in missing
    assert "Successful" in missing
