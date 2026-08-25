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
