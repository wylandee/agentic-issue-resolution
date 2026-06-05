"""
tests/test_triage_agent.py — Unit tests for src/triage/agent.py.

Covers:
- Deterministic path with no LLM → valid TriageResult, triage_method="deterministic"
- KEV group → revised_priority clamped to CRITICAL regardless of input
- EPSS ≥ 0.5 group → revised_priority at least HIGH
- Mock LLM returns LOW priority for KEV group → guardrail clamps to CRITICAL
- Dev-scope false positive → is_valid=False only with dev/test env + paths
- No evidence group → is_valid=True (optimistic default)
- Original HIGH severity clamps baseline to at least HIGH
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.contracts.schemas import (
    CVEEnrichment,
    IssueSource,
    IssueType,
    LineRange,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.triage.agent import _build_triage_prompt, run_triage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _issue(
    *,
    severity: Severity = Severity.MEDIUM,
    file_path: str | None = "src/app.js",
    issue_type: IssueType = IssueType.SCA,
    package_name: str = "lodash",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        severity=severity,
        file_path=file_path,
        package_name=package_name,
    )


def _group(
    issue: VulnerabilityIssue,
    *,
    enrichment: CVEEnrichment | None = None,
) -> VulnerabilityGroup:
    return VulnerabilityGroup(
        group_id="sca:src/app.js:lodash",
        issue_type=issue.issue_type,
        vulnerable_component="lodash",
        file_path=issue.file_path,
        cve_ids=["CVE-2021-23337"] if issue.issue_type == IssueType.SCA else [],
        representative_issue_id=issue.id,
        issues=[issue],
        enrichment=enrichment,
    )


def _kev_enrichment(cve: str = "CVE-2021-23337") -> CVEEnrichment:
    return CVEEnrichment(
        cve_id=cve,
        epss=0.9,
        epss_percentile=0.99,
        in_kev=True,
        kev_date_added="2022-01-01",
        enrichment_source="epss+kev",
    )


def _epss_enrichment(epss: float = 0.7, cve: str = "CVE-2021-23337") -> CVEEnrichment:
    return CVEEnrichment(
        cve_id=cve,
        epss=epss,
        epss_percentile=0.85,
        in_kev=False,
        enrichment_source="epss",
    )


def _context(
    environment: str = "production",
    *,
    public_facing: bool | None = None,
    data_sensitivity: str | None = None,
) -> SystemContext:
    return SystemContext(
        environment=environment,
        public_facing=public_facing,
        data_sensitivity=data_sensitivity,
    )


# ---------------------------------------------------------------------------
# Deterministic path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_llm_triage_by_default(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")


class TestDeterministicTriage:
    def test_basic_valid_result(self):
        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue)
        result = run_triage(group, _context())

        assert isinstance(result, TriageResult)
        assert result.is_valid is True
        assert result.triage_method == "deterministic"
        assert result.group_id == group.group_id
        assert result.recommended_issue_id == issue.id
        assert result.original_severity == Severity.MEDIUM
        assert result.is_unreachable_code is False
        assert result.validity_confidence_score == 1.0
        assert result.priority_confidence_score == 1.0

    def test_no_evidence_means_valid(self):
        """Unknown evidence must never automatically become a false positive."""
        issue = _issue(severity=Severity.UNKNOWN)
        group = _group(issue, enrichment=None)
        result = run_triage(group, _context())
        assert result.is_valid is True
        assert result.false_positive_reason is None

    def test_kev_clamps_to_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_kev_enrichment())
        result = run_triage(group, _context())
        assert result.revised_priority == Severity.CRITICAL
        assert result.is_valid is True  # KEV overrides any FP decision

    def test_kev_in_staging_falls_through_to_epss_rule(self):
        """In staging, KEV alone does not force CRITICAL; EPSS 0.9 ≥ 0.36 clamps to HIGH."""
        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue, enrichment=_kev_enrichment())  # epss=0.9, in_kev=True
        result = run_triage(group, _context(environment="staging"))
        # Not production → KEV rule skipped; EPSS 0.9 ≥ 0.36 rule fires → at least HIGH
        assert result.revised_priority in (Severity.HIGH, Severity.CRITICAL)

    def test_high_epss_clamps_to_at_least_high(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context())
        assert result.revised_priority in (Severity.HIGH, Severity.CRITICAL)

    def test_high_epss_public_facing_forces_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context(public_facing=True))
        assert result.revised_priority == Severity.CRITICAL

    def test_high_epss_high_sensitivity_forces_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context(data_sensitivity="HIGH"))
        assert result.revised_priority == Severity.CRITICAL

    def test_low_epss_does_not_clamp(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.1))
        result = run_triage(group, _context())
        # No guardrail should raise LOW above LOW in this case
        assert result.revised_priority in (Severity.LOW, Severity.MEDIUM)

    def test_internal_low_epss_high_severity_downgrades_to_medium(self):
        issue = _issue(severity=Severity.HIGH)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.005))
        result = run_triage(group, _context(public_facing=False))
        assert result.revised_priority == Severity.MEDIUM

    def test_original_critical_severity_preserved(self):
        issue = _issue(severity=Severity.CRITICAL)
        group = _group(issue)
        result = run_triage(group, _context())
        assert result.original_severity == Severity.CRITICAL
        assert result.revised_priority == Severity.CRITICAL

    def test_priority_reasoning_is_non_empty(self):
        group = _group(_issue())
        result = run_triage(group, _context())
        assert len(result.priority_reasoning) > 0

    def test_unreachable_code_is_flagged_but_not_marked_false_positive(self):
        issue = _issue(severity=Severity.HIGH)
        group = _group(issue)
        group.is_reachable = False

        result = run_triage(group, _context())

        assert result.is_valid is True
        assert result.false_positive_reason is None
        assert result.is_unreachable_code is True
        assert "Reachability analysis shows the package is not imported" in result.priority_reasoning


# ---------------------------------------------------------------------------
# Dev-scope false positive
# ---------------------------------------------------------------------------


class TestDevScopeFalsePositive:
    def test_dev_env_test_paths_marks_false_positive(self):
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue)
        result = run_triage(group, _context(environment="test"))
        assert result.is_valid is False
        assert result.false_positive_reason is not None

    def test_dev_env_but_prod_path_stays_valid(self):
        """Dev environment alone is not enough — the file paths must also be dev/test."""
        issue = _issue(file_path="src/utils.js")  # Not a dev path
        group = _group(issue)
        result = run_triage(group, _context(environment="dev"))
        assert result.is_valid is True

    def test_prod_env_test_path_stays_valid(self):
        """Test path in prod environment is not a false positive."""
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue)
        result = run_triage(group, _context(environment="production"))
        assert result.is_valid is True

    def test_kev_does_not_override_dev_false_positive_outside_production(self):
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue, enrichment=_kev_enrichment())
        result = run_triage(group, _context(environment="test"))
        assert result.is_valid is False
        assert result.false_positive_reason is not None


# ---------------------------------------------------------------------------
# LLM path guardrails
# ---------------------------------------------------------------------------


class TestLLMGuardrails:
    def _make_llm_result(
        self,
        group: VulnerabilityGroup,
        priority: Severity,
    ) -> TriageResult:
        return TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=True,
            original_severity=group.issues[0].severity if group.issues else Severity.UNKNOWN,
            revised_priority=priority,
            is_unreachable_code=False,
            priority_reasoning="LLM says LOW.",
            validity_confidence_score=0.6,
            priority_confidence_score=0.5,
            recommended_issue_id=group.representative_issue_id,
            triage_method="llm",
        )

    def test_llm_under_ranks_kev_guardrail_clamps_to_critical(self, monkeypatch):
        """If LLM returns LOW for a KEV issue, guardrail must clamp to CRITICAL."""
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_kev_enrichment())

        llm_result = self._make_llm_result(group, Severity.LOW)

        with patch("src.triage.agent._llm_triage", return_value=llm_result):
            result = run_triage(group, _context())

        assert result.revised_priority == Severity.CRITICAL
        assert "[Guardrail]" in result.priority_reasoning
        assert result.priority_confidence_score == 1.0

    def test_llm_no_priority_override_preserves_priority_confidence(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.1))
        llm_result = self._make_llm_result(group, Severity.MEDIUM)

        with patch("src.triage.agent._llm_triage", return_value=llm_result):
            result = run_triage(group, _context())

        assert result.revised_priority == Severity.MEDIUM
        assert result.original_severity == Severity.MEDIUM
        assert result.priority_confidence_score == 0.5

    def test_llm_failure_falls_back_to_deterministic(self, monkeypatch):
        """When _llm_triage returns None, deterministic path must be used."""
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.HIGH)
        group = _group(issue)

        with patch("src.triage.agent._llm_triage", return_value=None):
            result = run_triage(group, _context())

        assert result.triage_method == "deterministic"

    def test_triage_method_deterministic_when_llm_disabled(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")
        group = _group(_issue())
        result = run_triage(group, _context())
        assert result.triage_method == "deterministic"


class TestTriagePrompt:
    def test_prompt_includes_reachability_analysis_true(self):
        issue = _issue()
        group = _group(issue)
        group.is_reachable = True

        prompt = _build_triage_prompt(group, _context())

        assert "=== DATA ===" in prompt
        assert (
            "Reachability Analysis: TRUE (Package is explicitly imported in application code)"
            in prompt
        )
        assert "CONFIDENCE SCORING RUBRIC:" in prompt

    def test_prompt_rule_d_uses_explicit_false_reachability(self):
        issue = _issue()
        group = _group(issue)
        group.is_reachable = False

        prompt = _build_triage_prompt(group, _context())

        assert (
            "Reachability Analysis: FALSE (Package is a direct dependency but is NEVER imported in the application source code)"
            in prompt
        )
        assert (
            "Rule D: If Reachability Analysis is explicitly FALSE, the package is unreachable in this application. Set is_unreachable_code=True, but do not treat that fact alone as a false positive."
            in prompt
        )
        assert (
            "Always return original_severity, revised_priority, is_unreachable_code, validity_confidence_score, and priority_confidence_score."
            in prompt
        )
