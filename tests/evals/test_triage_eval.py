"""Phase 2: DeepEval and structural evaluation for Triage Agent classification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
    TriageResult,
)
from remediation_engine.triage.agent import _apply_guardrails
from tests.evals.conftest import EvalSettings
from tests.evals.custom_metrics import TriageConsistencyMetric

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

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "triage_cases.json"


def _load_triage_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if c.get("eval_type", "triage") == "triage"]
        return []
    except Exception:
        return []


_TRIAGE_CASES = _load_triage_cases()
_TRIAGE_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_TRIAGE_CASES)]


# ---------------------------------------------------------------------------
# Prompt & Context Formatting Helpers (Matching Production triage/agent.py)
# ---------------------------------------------------------------------------


def build_triage_production_prompt(
    vuln_context: dict[str, Any],
    system_context: dict[str, Any],
    enrichment: dict[str, Any] | None,
) -> str:
    """Build a structured triage prompt mirroring production ``_build_triage_prompt``."""
    cves = vuln_context.get("cve_ids", [])
    ghsas = vuln_context.get("ghsa_ids", [])
    epss_info = "EPSS score: unavailable"
    if enrichment and enrichment.get("epss") is not None:
        epss = enrichment.get("epss", 0.0)
        pct = enrichment.get("epss_percentile", 0.0)
        epss_info = f"EPSS score: {epss:.3f} (percentile: {pct:.3f})"

    kev_info = "CISA KEV: no"
    if enrichment and enrichment.get("in_kev"):
        date = enrichment.get("kev_date_added") or "date unknown"
        kev_info = f"CISA KEV: YES (added {date})"

    cve_details_lines = []
    for issue in vuln_context.get("issues", []):
        cid = issue.get("cve_id") or "UNKNOWN"
        sev = issue.get("severity") or "UNKNOWN"
        msg = issue.get("message") or "unavailable"
        cve_details_lines.append(f"  - CVE ID: {cid}\n    Severity: {sev}\n    Description: {msg}")

    cve_details = "\n".join(cve_details_lines) if cve_details_lines else "  (No details available)"

    return (
        "=== DATA ===\n"
        f"- Group ID: {vuln_context.get('group_id')}\n"
        f"- Issue Type: {vuln_context.get('issue_type')}\n"
        f"- Vulnerable Component: {vuln_context.get('vulnerable_component')}\n"
        f"- Target File: {vuln_context.get('file_path')}\n"
        f"- CVE IDs: {', '.join(cves) if cves else 'none'}\n"
        f"- GHSA IDs: {', '.join(ghsas) if ghsas else 'none'}\n"
        f"- Original Severity: {vuln_context.get('original_severity')}\n"
        f"- Reachability Analysis: {'Reachable' if vuln_context.get('is_reachable', True) else 'Not reachable'}\n"
        f"- {epss_info}\n"
        f"- {kev_info}\n\n"
        "=== CVE DETAILS ===\n"
        f"{cve_details}\n\n"
        "=== SYSTEM CONTEXT ===\n"
        f"- Environment: {system_context.get('environment')}\n"
        f"- Deployment OS: {system_context.get('deployment_os')}\n"
        f"- Public Facing: {'yes' if system_context.get('public_facing') else 'no'}\n"
        f"- Primary Language: {system_context.get('primary_language')}\n"
        f"- Deployment Architecture: {system_context.get('deployment_architecture')}\n"
        f"- Data Sensitivity: {system_context.get('data_sensitivity')}\n"
    )


# ---------------------------------------------------------------------------
# Structural & Deterministic Validation Helpers
# ---------------------------------------------------------------------------


def validate_triage_schema(result_dict: dict[str, Any]) -> list[str]:
    """Validate that the triage result conforms to the Pydantic TriageResult schema."""
    violations: list[str] = []
    try:
        TriageResult(**result_dict)
    except Exception as exc:
        violations.append(f"TriageResult contract violation: {exc}")

    is_valid = result_dict.get("is_valid")
    fp_reason = result_dict.get("false_positive_reason")
    if is_valid is False and (not fp_reason or not str(fp_reason).strip()):
        violations.append("false_positive_reason is required when is_valid is False")

    valid_conf = result_dict.get("validity_confidence_score")
    if valid_conf is None or not (0.0 <= float(valid_conf) <= 1.0):
        violations.append(f"validity_confidence_score must be in [0.0, 1.0], got {valid_conf}")

    prio_conf = result_dict.get("priority_confidence_score")
    if prio_conf is None or not (0.0 <= float(prio_conf) <= 1.0):
        violations.append(f"priority_confidence_score must be in [0.0, 1.0], got {prio_conf}")

    return violations


def validate_guardrail_alignment(
    result_dict: dict[str, Any],
    vuln_context: dict[str, Any],
    system_context: dict[str, Any],
    enrichment: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Re-run deterministic guardrails against the LLM triage verdict and return alignment."""
    raw_prio = result_dict.get("revised_priority", "UNKNOWN")
    raw_is_valid = result_dict.get("is_valid", True)
    raw_fp_reason = result_dict.get("false_positive_reason")
    orig_sev_raw = vuln_context.get("original_severity", "UNKNOWN")

    try:
        prio = Severity(raw_prio)
    except ValueError:
        prio = Severity.UNKNOWN

    try:
        orig_sev = Severity(orig_sev_raw)
    except ValueError:
        orig_sev = Severity.UNKNOWN

    context = SystemContext(**system_context) if system_context else SystemContext()
    enrich_obj = None
    if enrichment:
        enrich_dict = dict(enrichment)
        enrich_dict.setdefault("cve_id", "CVE-UNKNOWN")
        enrich_obj = CVEEnrichment(**enrich_dict)

    (
        final_prio,
        guardrail_note,
        final_is_valid,
        final_fp_reason,
        priority_overridden,
    ) = _apply_guardrails(
        priority=prio,
        is_valid=raw_is_valid,
        false_positive_reason=raw_fp_reason,
        context=context,
        enrichment=enrich_obj,
        original_severity=orig_sev,
    )

    validity_overridden = final_is_valid != raw_is_valid
    is_aligned = not (priority_overridden or validity_overridden)

    details = []
    if priority_overridden:
        details.append(
            f"Priority changed from {prio.value} to {final_prio.value} ({guardrail_note})"
        )
    if validity_overridden:
        details.append(f"Validity changed from {raw_is_valid} to {final_is_valid}")

    return is_aligned, "; ".join(details)


def validate_false_positive_rate(
    result_dict: dict[str, Any],
    expected_case: dict[str, Any],
) -> list[str]:
    """Ensure that actionable vulnerabilities are not incorrectly classified as false positives."""
    violations: list[str] = []
    expected_valid = expected_case.get("expected_is_valid", True)
    actual_valid = result_dict.get("is_valid", True)

    if expected_valid and not actual_valid:
        fp_reason = result_dict.get("false_positive_reason", "no reason provided")
        violations.append(
            f"Actionable finding was incorrectly marked as false positive with reason: '{fp_reason}'"
        )
    return violations


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestTriageEval:
    """Evaluation test suite for Triage Agent classifications."""

    @pytest.mark.parametrize(
        "case",
        _TRIAGE_CASES or [{}],
        ids=_TRIAGE_CASE_IDS or ["no_cases"],
    )
    def test_triage_accuracy_against_golden_set(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """LLM triage matches expected validity, revised priority, and reachability."""
        if not case:
            pytest.skip("No golden triage cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_triage_result", {})
        vuln_ctx = case.get("vulnerability_context", {})
        sys_ctx = case.get("system_context", {})
        enrichment = case.get("enrichment")
        expected_valid = case.get("expected_is_valid")
        expected_prio = case.get("expected_priority")
        expected_unreachable = case.get("expected_is_unreachable")

        # 1. Deterministic schema & contract assertions
        schema_violations = validate_triage_schema(result)
        assert not schema_violations, f"Case '{case_id}' failed schema: {schema_violations}"

        assert result.get("is_valid") == expected_valid, (
            f"Case '{case_id}' validity mismatch: expected {expected_valid}, got {result.get('is_valid')}"
        )
        assert result.get("revised_priority") == expected_prio, (
            f"Case '{case_id}' priority mismatch: expected {expected_prio}, got {result.get('revised_priority')}"
        )
        if expected_unreachable is not None:
            assert result.get("is_unreachable_code") == expected_unreachable, (
                f"Case '{case_id}' reachability mismatch: expected {expected_unreachable}, got {result.get('is_unreachable_code')}"
            )

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_triage_production_prompt(vuln_ctx, sys_ctx, enrichment)
            actual_output = json.dumps(
                {
                    "is_valid": result.get("is_valid"),
                    "revised_priority": result.get("revised_priority"),
                    "priority_reasoning": result.get("priority_reasoning"),
                },
                indent=2,
            )
            expected_output = json.dumps(
                {
                    "is_valid": expected_valid,
                    "revised_priority": expected_prio,
                    "priority_reasoning": result.get("priority_reasoning"),
                },
                indent=2,
            )

            test_case = LLMTestCase(
                name=f"{case_id} [Triage Verdict Accuracy]",
                input=prompt,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            accuracy_geval = GEval(
                name="Triage Verdict Accuracy",
                criteria=(
                    "Evaluate whether the triage classification in the Actual Output correctly matches "
                    "the Expected Output. Confirm that is_valid and revised_priority align with the Expected Output, "
                    "and that the reasoning in priority_reasoning is sound given the input context."
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

            assert_test(test_case, [accuracy_geval])

    @pytest.mark.parametrize(
        "case",
        _TRIAGE_CASES or [{}],
        ids=_TRIAGE_CASE_IDS or ["no_cases"],
    )
    def test_triage_guardrail_alignment(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """LLM triage output is in full alignment with deterministic RBVM guardrails."""
        if not case:
            pytest.skip("No golden triage cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_triage_result", {})
        vuln_ctx = case.get("vulnerability_context", {})
        sys_ctx = case.get("system_context", {})
        enrichment = case.get("enrichment")

        # 1. Deterministic guardrail consistency check
        is_aligned, details = validate_guardrail_alignment(result, vuln_ctx, sys_ctx, enrichment)
        assert is_aligned, f"Case '{case_id}' was overridden by deterministic guardrails: {details}"

        # 2. Live / Custom metric evaluation with TriageConsistencyMetric
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")

            prompt = build_triage_production_prompt(vuln_ctx, sys_ctx, enrichment)
            test_case = LLMTestCase(
                name=f"{case_id} [Guardrail Consistency]",
                input=prompt,
                actual_output=json.dumps(result, indent=2),
                additional_metadata={
                    "llm_priority": result.get("revised_priority"),
                    "llm_is_valid": result.get("is_valid"),
                    "llm_false_positive_reason": result.get("false_positive_reason"),
                    "original_severity": vuln_ctx.get("original_severity"),
                    "system_context": sys_ctx,
                    "enrichment": enrichment,
                },
            )

            metric = TriageConsistencyMetric(threshold=1.0)
            assert_test(test_case, [metric])

    @pytest.mark.parametrize(
        "case",
        _TRIAGE_CASES or [{}],
        ids=_TRIAGE_CASE_IDS or ["no_cases"],
    )
    def test_triage_false_positive_rate(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """LLM does not classify actionable findings as false positives."""
        if not case:
            pytest.skip("No golden triage cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_triage_result", {})

        # 1. Deterministic false positive check
        fp_violations = validate_false_positive_rate(result, case)
        assert not fp_violations, f"Case '{case_id}' had FP rate violations: {fp_violations}"

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            vuln_ctx = case.get("vulnerability_context", {})
            sys_ctx = case.get("system_context", {})
            enrichment = case.get("enrichment")
            prompt = build_triage_production_prompt(vuln_ctx, sys_ctx, enrichment)

            actual_output = json.dumps(
                {
                    "is_valid": result.get("is_valid"),
                    "false_positive_reason": result.get("false_positive_reason"),
                    "priority_reasoning": result.get("priority_reasoning"),
                },
                indent=2,
            )
            expected_output = json.dumps(
                {
                    "is_valid": case.get("expected_is_valid"),
                    "false_positive_reason": (
                        result.get("false_positive_reason")
                        if not case.get("expected_is_valid")
                        else None
                    ),
                    "priority_reasoning": result.get("priority_reasoning"),
                },
                indent=2,
            )

            test_case = LLMTestCase(
                name=f"{case_id} [False Positive Quality]",
                input=prompt,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            fp_geval = GEval(
                name="False Positive Detection Quality",
                criteria=(
                    "Evaluate whether the triage classification correctly determines is_valid and "
                    "distinguishes genuinely inapplicable vulnerabilities (wrong operating system, "
                    "mismatched runtime/technology, test-only paths) from actionable security findings. "
                    "When is_valid is False, verify that a clear false_positive_reason is provided."
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

            assert_test(test_case, [fp_geval])


# ---------------------------------------------------------------------------
# Unit tests for the evaluation helpers themselves (adversarial validation)
# ---------------------------------------------------------------------------


def test_triage_evaluator_catches_schema_violations() -> None:
    """Ensure validation helpers correctly catch invalid schemas and missing FP reasons."""
    bad_result: dict[str, Any] = {
        "group_id": "test_group",
        "is_valid": False,
        "false_positive_reason": None,  # Violation: required when is_valid=False
        "revised_priority": "CRITICAL",
        "validity_confidence_score": 1.5,  # Violation: > 1.0
        "priority_confidence_score": -0.1,  # Violation: < 0.0
        "priority_reasoning": "some reasoning",
        "recommended_issue_id": "b1a10001-0001-0001-0001-000000000001",
        "triage_method": "llm",
    }
    violations = validate_triage_schema(bad_result)
    assert any("false_positive_reason" in v for v in violations)
    assert any("validity_confidence_score" in v for v in violations)
    assert any("priority_confidence_score" in v for v in violations)


def test_triage_evaluator_catches_guardrail_override() -> None:
    """Ensure validate_guardrail_alignment flags cases where deterministic rules override LLM."""
    llm_result: dict[str, Any] = {
        "revised_priority": "LOW",  # LLM under-ranked
        "is_valid": True,
        "false_positive_reason": None,
    }
    vuln_ctx: dict[str, Any] = {"original_severity": "MEDIUM"}
    sys_ctx: dict[str, Any] = {"environment": "production"}
    enrichment: dict[str, Any] = {"cve_id": "CVE-2022-0001", "in_kev": True, "epss": 0.9}

    is_aligned, details = validate_guardrail_alignment(llm_result, vuln_ctx, sys_ctx, enrichment)
    assert not is_aligned
    assert "Priority changed from LOW to CRITICAL" in details
