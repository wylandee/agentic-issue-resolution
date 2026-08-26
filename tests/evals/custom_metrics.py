"""Phase 2 & Phase 3: Custom DeepEval BaseMetric implementations for remediation evaluation.

This module provides deterministic, domain-specific evaluation metrics that conform
to DeepEval's ``BaseMetric`` interface. These metrics can run both offline (CI-safe,
zero LLM cost) and within live evaluation test suites.
"""

from __future__ import annotations

import re
from typing import Any

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
)
from remediation_engine.triage.agent import _apply_guardrails

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False

    class BaseMetric:  # type: ignore[no-redef]
        """Fallback base metric when deepeval is not installed."""

        threshold: float = 1.0
        score: float | None = None
        reason: str | None = None
        success: bool | None = None
        strict_mode: bool = False
        async_mode: bool = False
        verbose_mode: bool = True

    class LLMTestCase:  # type: ignore[no-redef]
        """Fallback test case representation."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLMTestCaseParams:  # type: ignore[no-redef]
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"
        EXPECTED_OUTPUT = "expected_output"
        CONTEXT = "context"


# ---------------------------------------------------------------------------
# Semver regex for version validation
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^\bv?(\d+)\.(\d+)(?:\.(\d+))?(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b$"
)


# ---------------------------------------------------------------------------
# Triage Consistency Metric
# ---------------------------------------------------------------------------


class TriageConsistencyMetric(BaseMetric):
    """Metric evaluating whether deterministic RBVM guardrails override LLM triage decisions.

    A score of 1.0 indicates that the LLM verdict (is_valid, revised_priority) was in full
    harmony with the engine's deterministic policy guardrails. A score of 0.0 indicates that
    post-LLM guardrails had to override the verdict (e.g. CISA KEV in production forced to CRITICAL,
    or low-EPSS internal app downgraded to MEDIUM).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the triage consistency metric.

        Args:
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-guardrails"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Triage Consistency (Guardrail Alignment)"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure guardrail alignment for the provided test case.

        Expected metadata keys on test_case:
          - 'llm_priority': str (Severity name)
          - 'llm_is_valid': bool
          - 'llm_false_positive_reason': str | None
          - 'original_severity': str (Severity name)
          - 'system_context': dict (SystemContext serialized)
          - 'enrichment': dict | None (CVEEnrichment serialized)
        """
        meta = getattr(test_case, "additional_metadata", {}) or {}

        raw_priority = meta.get("llm_priority", "UNKNOWN")
        raw_is_valid = meta.get("llm_is_valid", True)
        raw_fp_reason = meta.get("llm_false_positive_reason")
        raw_orig_sev = meta.get("original_severity", "UNKNOWN")
        raw_context = meta.get("system_context", {})
        raw_enrichment = meta.get("enrichment")

        try:
            priority = Severity(raw_priority)
        except ValueError:
            priority = Severity.UNKNOWN

        try:
            orig_sev = Severity(raw_orig_sev)
        except ValueError:
            orig_sev = Severity.UNKNOWN

        context = SystemContext(**raw_context) if raw_context else SystemContext()
        enrichment = None
        if raw_enrichment:
            enrich_dict = dict(raw_enrichment)
            enrich_dict.setdefault("cve_id", "CVE-UNKNOWN")
            enrichment = CVEEnrichment(**enrich_dict)

        (
            final_priority,
            guardrail_note,
            final_is_valid,
            final_fp_reason,
            priority_overridden,
        ) = _apply_guardrails(
            priority=priority,
            is_valid=raw_is_valid,
            false_positive_reason=raw_fp_reason,
            context=context,
            enrichment=enrichment,
            original_severity=orig_sev,
        )

        validity_overridden = final_is_valid != raw_is_valid

        if priority_overridden or validity_overridden:
            self.score = 0.0
            reasons = []
            if priority_overridden:
                reasons.append(
                    f"Priority was overridden from {priority.value} to {final_priority.value}. "
                    f"Guardrail detail: {guardrail_note or 'deterministic override rule fired'}."
                )
            if validity_overridden:
                reasons.append(f"Validity was overridden from {raw_is_valid} to {final_is_valid}.")
            self.reason = " ".join(reasons)
        else:
            self.score = 1.0
            self.reason = (
                "LLM triage verdict aligns with deterministic RBVM guardrails without override."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Fix Planner Schema Metric
# ---------------------------------------------------------------------------


class FixPlannerSchemaMetric(BaseMetric):
    """Metric validating structured output invariants of SerperLLMResult.

    Structural invariants enforced:
      1. strategy must be one of 'VERSION_BUMP', 'CODE_WORKAROUND', 'NO_FIX'.
      2. If strategy == 'VERSION_BUMP', fixed_version must be non-empty and match semver pattern.
      3. If strategy == 'CODE_WORKAROUND', workaround_snippets must be a non-empty list of strings.
      4. If strategy == 'NO_FIX', fixed_version and workaround_snippets must both be null/empty.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the Fix Planner schema metric.

        Args:
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-schema-validation"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Fix Planner Schema Invariants"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Validate SerperLLMResult invariants on the provided test case.

        Expected metadata keys on test_case:
          - 'strategy': str
          - 'fixed_version': str | None
          - 'workaround_snippets': list[str] | None
        """
        meta = getattr(test_case, "additional_metadata", {}) or {}

        strategy = meta.get("strategy")
        fixed_version = meta.get("fixed_version")
        workaround_snippets = meta.get("workaround_snippets")

        violations: list[str] = []

        valid_strategies = {"VERSION_BUMP", "CODE_WORKAROUND", "NO_FIX"}
        if strategy not in valid_strategies:
            violations.append(
                f"Invalid strategy '{strategy}'. Must be one of {sorted(valid_strategies)}."
            )

        if strategy == "VERSION_BUMP":
            if not fixed_version or not str(fixed_version).strip():
                violations.append("VERSION_BUMP strategy requires a non-empty 'fixed_version'.")
            elif not _SEMVER_RE.match(str(fixed_version).strip()):
                violations.append(
                    f"fixed_version '{fixed_version}' does not match valid semver format."
                )

        elif strategy == "CODE_WORKAROUND":
            if not workaround_snippets or not isinstance(workaround_snippets, list):
                violations.append(
                    "CODE_WORKAROUND strategy requires a non-empty 'workaround_snippets' list."
                )
            elif not any(str(s).strip() for s in workaround_snippets):
                violations.append("workaround_snippets list contains only empty strings.")

        elif strategy == "NO_FIX":
            if fixed_version is not None and str(fixed_version).strip():
                violations.append(
                    f"NO_FIX strategy must have null fixed_version, got '{fixed_version}'."
                )
            if workaround_snippets:
                violations.append(
                    f"NO_FIX strategy must have null workaround_snippets, got {workaround_snippets}."
                )

        if violations:
            self.score = 0.0
            self.reason = "Schema violations detected: " + "; ".join(violations)
        else:
            self.score = 1.0
            self.reason = (
                f"SerperLLMResult satisfies all schema invariants for strategy '{strategy}'."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 4: QA Structured Output Metric
# ---------------------------------------------------------------------------


class QAStructuredOutputMetric(BaseMetric):
    """Metric validating structured output invariants of QACriticLLMOutput.

    Structural invariants enforced:
      1. task_id must be non-empty.
      2. If passed == True, failure_category and retry_feedback must both be None.
      3. If passed == False, failure_category must be non-null and valid, and retry_feedback non-empty.
      4. failure_category must be one of 'security_flag', 'peer_conflict', 'breaking_change'.
      5. If semantic_security_review is present, verdict must be 'pass', 'fail', or 'inconclusive';
         when verdict == 'pass', evidence_refs must be a non-empty list of non-empty strings.
      6. If test_attribution is present, verdict must be 'responsible', 'exonerated', or 'inconclusive'.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the QA structured output metric.

        Args:
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-schema-validation"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "QA Structured Output Invariants"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Validate QACriticLLMOutput invariants on the provided test case.

        Expected metadata keys on test_case:
          - 'task_id': str
          - 'passed': bool
          - 'failure_category': str | None
          - 'retry_feedback': str | None
          - 'semantic_security_review': dict | None
          - 'test_attribution': dict | None
        """
        meta = getattr(test_case, "additional_metadata", {}) or {}

        task_id = meta.get("task_id")
        passed = meta.get("passed")
        failure_category = meta.get("failure_category")
        retry_feedback = meta.get("retry_feedback")
        security_review = meta.get("semantic_security_review")
        test_attribution = meta.get("test_attribution")

        violations: list[str] = []

        if not task_id or not str(task_id).strip():
            violations.append("task_id must be a non-empty string.")

        if not isinstance(passed, bool):
            violations.append(f"'passed' must be a boolean, got {type(passed).__name__}.")

        valid_categories = {"security_flag", "peer_conflict", "breaking_change"}

        if passed is True:
            if failure_category is not None:
                violations.append(
                    f"passed=True requires failure_category=None, got '{failure_category}'."
                )
            if retry_feedback is not None and str(retry_feedback).strip():
                violations.append(
                    f"passed=True requires retry_feedback=None, got '{retry_feedback}'."
                )
        elif passed is False:
            if failure_category is None:
                violations.append("passed=False requires a non-null failure_category.")
            elif str(failure_category).lower() not in valid_categories:
                violations.append(
                    f"Invalid failure_category '{failure_category}'. Must be one of {sorted(valid_categories)}."
                )
            if not retry_feedback or not str(retry_feedback).strip():
                violations.append("passed=False requires a non-empty retry_feedback string.")

        if security_review is not None and isinstance(security_review, dict):
            verdict = str(security_review.get("verdict", "")).lower()
            valid_verdicts = {"pass", "fail", "inconclusive"}
            if verdict not in valid_verdicts:
                violations.append(
                    f"Invalid semantic_security_review.verdict '{verdict}'. Must be one of {sorted(valid_verdicts)}."
                )
            if verdict == "pass":
                refs = security_review.get("evidence_refs", [])
                if not refs or not isinstance(refs, list) or not any(str(r).strip() for r in refs):
                    violations.append(
                        "semantic_security_review with verdict='pass' requires non-empty evidence_refs."
                    )

        if test_attribution is not None and isinstance(test_attribution, dict):
            attr_verdict = str(test_attribution.get("verdict", "")).lower()
            valid_attr_verdicts = {"responsible", "exonerated", "inconclusive"}
            if attr_verdict not in valid_attr_verdicts:
                violations.append(
                    f"Invalid test_attribution.verdict '{attr_verdict}'. Must be one of {sorted(valid_attr_verdicts)}."
                )

        if violations:
            self.score = 0.0
            self.reason = "QACriticLLMOutput violations: " + "; ".join(violations)
        else:
            self.score = 1.0
            self.reason = (
                f"QACriticLLMOutput satisfies all structured output invariants (passed={passed})."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 4: QA Guardrail Consistency Metric
# ---------------------------------------------------------------------------


class QAGuardrailConsistencyMetric(BaseMetric):
    """Metric evaluating whether deterministic QA policy guardrails override LLM QA decisions.

    A score of 1.0 indicates that the LLM verdict (passed, failure_category, security review)
    is in full alignment with the engine's deterministic policy gates for the target QAPolicy.
    A score of 0.0 indicates that post-LLM guardrails would override the verdict.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the QA guardrail consistency metric.

        Args:
            threshold: Minimum score required for success (default 1.0).
            verbose_mode: Whether to generate detailed diagnostic reasons.
        """
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-qa-guardrails"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "QA Guardrail Consistency"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure guardrail alignment for the provided QA test case.

        Expected metadata keys on test_case:
          - 'qa_policy': str
          - 'execution_context': dict (install_passed, scanner_execution_status, target_remaining_identifiers, tests_passed, package_manifest_state, package_graph_state)
          - 'llm_passed': bool
          - 'llm_failure_category': str | None
          - 'llm_security_review': dict | None
        """
        meta = getattr(test_case, "additional_metadata", {}) or {}

        qa_policy_str = str(meta.get("qa_policy", "version_bump")).lower()
        ctx = meta.get("execution_context", {})
        llm_passed = bool(meta.get("llm_passed", False))
        llm_category = meta.get("llm_failure_category")
        llm_sec_review = meta.get("llm_security_review")

        install_passed = bool(ctx.get("install_passed", True))
        scanner_status = str(ctx.get("scanner_execution_status", "success")).lower()
        remaining = list(ctx.get("target_remaining_identifiers", []))
        tests_passed = ctx.get("tests_passed")
        manifest_state = str(ctx.get("package_manifest_state", "present")).lower()
        graph_state = str(ctx.get("package_graph_state", "present")).lower()

        overrides: list[str] = []

        # 1. Install failure rule
        if not install_passed:
            if llm_passed:
                overrides.append("Install failed but LLM marked passed=True.")
            elif llm_category not in ("peer_conflict", "security_flag", "breaking_change"):
                overrides.append(
                    f"Install failed but LLM gave unexpected category '{llm_category}'."
                )

        # 2. Strict scanner policies (version_bump, no_fix_package_removal)
        strict_scanner_policies = {"version_bump", "no_fix_package_removal"}
        if qa_policy_str in strict_scanner_policies and scanner_status == "success" and remaining:
            if llm_passed:
                overrides.append(
                    f"Remaining scanner findings {remaining} require passed=False (SECURITY_FLAG), but LLM marked passed=True."
                )
            elif llm_category != "security_flag":
                overrides.append(
                    f"Remaining scanner findings {remaining} require failure_category=SECURITY_FLAG, but LLM assigned '{llm_category}'."
                )

        # 3. Hard test policies
        if tests_passed is False:
            if llm_passed:
                overrides.append("Unit tests failed but LLM marked passed=True.")
            elif llm_category not in ("breaking_change", "security_flag"):
                overrides.append(
                    f"Unit test failure requires BREAKING_CHANGE (or SECURITY_FLAG), but LLM assigned '{llm_category}'."
                )

        # 4. Workaround policies require semantic security review PASS
        workaround_policies = {
            "initial_code_workaround",
            "mitigation_code_workaround",
            "migration_code_workaround",
            "no_fix_code_removal",
        }
        if qa_policy_str in workaround_policies:
            review_verdict = (
                str(llm_sec_review.get("verdict", "")).lower() if llm_sec_review else ""
            )
            review_refs = llm_sec_review.get("evidence_refs", []) if llm_sec_review else []
            if (review_verdict != "pass" or not review_refs) and llm_passed:
                overrides.append(
                    f"Workaround policy '{qa_policy_str}' requires semantic_security_review verdict='pass' with evidence_refs, but review verdict was '{review_verdict}'."
                )

        # 5. Package removal manifest/graph invariants
        if (
            qa_policy_str == "no_fix_package_removal"
            and (manifest_state != "absent" or graph_state != "absent")
            and llm_passed
        ):
            overrides.append(
                f"NO_FIX_PACKAGE_REMOVAL requires manifest & graph state 'absent', got manifest='{manifest_state}', graph='{graph_state}'."
            )

        # 6. Code removal manifest/graph invariants
        if (
            qa_policy_str == "no_fix_code_removal"
            and (manifest_state != "present" or graph_state != "present")
            and llm_passed
        ):
            overrides.append(
                f"NO_FIX_CODE_REMOVAL requires package present in manifest & graph, got manifest='{manifest_state}', graph='{graph_state}'."
            )

        if overrides:
            self.score = 0.0
            self.reason = "Deterministic guardrail overrides: " + "; ".join(overrides)
        else:
            self.score = 1.0
            self.reason = (
                f"LLM QA verdict aligns with deterministic policy gates for '{qa_policy_str}'."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)
