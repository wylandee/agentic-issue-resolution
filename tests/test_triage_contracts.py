"""
tests/test_triage_contracts.py — Unit tests for the Phase 4.0 triage contracts.

Covers:
- TriageResult.false_positive_reason required when is_valid=False
- VulnerabilityGroup list fields default to []
- Severity enum coercion from strings
- SystemContext default fields
- CVEEnrichment safe defaults
- Stable JSON round-trip for all four new models
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.contracts.schemas import (
    CVEEnrichment,
    IssueSource,
    IssueType,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    *,
    issue_type: IssueType = IssueType.SCA,
    severity: Severity = Severity.HIGH,
    package_name: str = "lodash",
    cve_id: str | None = "CVE-2021-23337",
    file_path: str | None = "package.json",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        severity=severity,
        package_name=package_name,
        cve_id=cve_id,
        file_path=file_path,
    )


def _make_group(issue: VulnerabilityIssue | None = None) -> VulnerabilityGroup:
    if issue is None:
        issue = _make_issue()
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _make_triage_result(group: VulnerabilityGroup) -> TriageResult:
    return TriageResult(
        group_id=group.group_id,
        is_valid=True,
        revised_priority=Severity.HIGH,
        priority_reasoning="Original severity HIGH.",
        recommended_issue_id=group.representative_issue_id,
        triage_method="deterministic",
    )


# ---------------------------------------------------------------------------
# SystemContext
# ---------------------------------------------------------------------------


class TestSystemContext:
    def test_defaults(self):
        ctx = SystemContext()
        assert ctx.repo_url is None
        assert ctx.base_ref is None
        assert ctx.environment is None
        assert ctx.tags == {}
        assert isinstance(ctx.scanned_at, datetime)

    def test_with_fields(self):
        ctx = SystemContext(
            repo_url="https://github.com/org/repo",
            base_ref="main",
            environment="production",
            tags={"team": "appsec"},
        )
        assert ctx.environment == "production"
        assert ctx.tags["team"] == "appsec"

    def test_json_round_trip(self):
        ctx = SystemContext(environment="staging")
        data = json.loads(ctx.model_dump_json())
        restored = SystemContext.model_validate(data)
        assert restored.environment == "staging"


# ---------------------------------------------------------------------------
# CVEEnrichment
# ---------------------------------------------------------------------------


class TestCVEEnrichment:
    def test_safe_defaults(self):
        enrichment = CVEEnrichment(cve_id="CVE-2021-44228")
        assert enrichment.epss == 0.0
        assert enrichment.epss_percentile == 0.0
        assert enrichment.in_kev is False
        assert enrichment.kev_date_added is None
        assert enrichment.enrichment_source == "none"

    def test_full_enrichment(self):
        enrichment = CVEEnrichment(
            cve_id="CVE-2021-44228",
            epss=0.97,
            epss_percentile=0.999,
            in_kev=True,
            kev_date_added="2021-12-10",
            enrichment_source="epss+kev",
        )
        assert enrichment.in_kev is True
        assert enrichment.epss == pytest.approx(0.97)

    def test_epss_bounds(self):
        with pytest.raises(ValidationError):
            CVEEnrichment(cve_id="CVE-2021-44228", epss=1.5)
        with pytest.raises(ValidationError):
            CVEEnrichment(cve_id="CVE-2021-44228", epss=-0.1)

    def test_json_round_trip(self):
        enrichment = CVEEnrichment(
            cve_id="CVE-2021-44228",
            epss=0.5,
            in_kev=True,
        )
        data = json.loads(enrichment.model_dump_json())
        restored = CVEEnrichment.model_validate(data)
        assert restored.in_kev is True
        assert restored.epss == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# VulnerabilityGroup
# ---------------------------------------------------------------------------


class TestVulnerabilityGroup:
    def test_list_fields_default_to_empty(self):
        issue = _make_issue()
        group = VulnerabilityGroup(
            group_id="sca:pkg.json:lodash",
            issue_type=IssueType.SCA,
            representative_issue_id=issue.id,
        )
        assert group.cve_ids == []
        assert group.versions == []
        assert group.sources == []
        assert group.issues == []
        assert group.enrichment is None

    def test_with_cve_and_sources(self):
        issue = _make_issue()
        group = VulnerabilityGroup(
            group_id="sca:pkg.json:lodash",
            issue_type=IssueType.SCA,
            vulnerable_component="lodash",
            representative_issue_id=issue.id,
            issues=[issue],
            cve_ids=["CVE-2021-23337"],
            sources=[IssueSource.SEMGREP, IssueSource.ODC],
        )
        assert len(group.cve_ids) == 1
        assert len(group.sources) == 2

    def test_json_round_trip(self):
        issue = _make_issue()
        group = _make_group(issue)
        data = json.loads(group.model_dump_json())
        restored = VulnerabilityGroup.model_validate(data)
        assert restored.group_id == group.group_id
        assert restored.representative_issue_id == group.representative_issue_id

    def test_enrichment_can_be_attached(self):
        issue = _make_issue()
        group = _make_group(issue)
        enrichment = CVEEnrichment(cve_id="CVE-2021-23337", epss=0.7, in_kev=True)
        group.enrichment = enrichment
        assert group.enrichment.in_kev is True


# ---------------------------------------------------------------------------
# TriageResult
# ---------------------------------------------------------------------------


class TestTriageResult:
    def test_valid_result(self):
        group = _make_group()
        result = _make_triage_result(group)
        assert result.is_valid is True
        assert result.triage_method == "deterministic"

    def test_false_positive_requires_reason(self):
        group = _make_group()
        with pytest.raises(ValidationError, match="false_positive_reason is required"):
            TriageResult(
                group_id=group.group_id,
                is_valid=False,
                # missing false_positive_reason
                revised_priority=Severity.LOW,
                priority_reasoning="Dev only.",
                recommended_issue_id=group.representative_issue_id,
                triage_method="deterministic",
            )

    def test_false_positive_with_reason(self):
        group = _make_group()
        result = TriageResult(
            group_id=group.group_id,
            is_valid=False,
            false_positive_reason="All findings are in test/ paths; dev-only dependency.",
            revised_priority=Severity.LOW,
            priority_reasoning="Dev only.",
            recommended_issue_id=group.representative_issue_id,
            triage_method="deterministic",
        )
        assert result.is_valid is False
        assert result.false_positive_reason is not None

    def test_severity_coercion_from_string(self):
        group = _make_group()
        result = TriageResult(
            group_id=group.group_id,
            is_valid=True,
            revised_priority="HIGH",  # type: ignore[arg-type]
            priority_reasoning="Test.",
            recommended_issue_id=group.representative_issue_id,
            triage_method="deterministic",
        )
        assert result.revised_priority is Severity.HIGH

    def test_json_round_trip(self):
        group = _make_group()
        result = _make_triage_result(group)
        data = json.loads(result.model_dump_json())
        restored = TriageResult.model_validate(data)
        assert restored.group_id == result.group_id
        assert restored.revised_priority == result.revised_priority

    def test_all_severity_values_accepted(self):
        group = _make_group()
        for sev in Severity:
            r = TriageResult(
                group_id=group.group_id,
                is_valid=True,
                revised_priority=sev,
                priority_reasoning="Test.",
                recommended_issue_id=group.representative_issue_id,
                triage_method="deterministic",
            )
            assert r.revised_priority == sev


# ---------------------------------------------------------------------------
# Contract imports from __init__
# ---------------------------------------------------------------------------


def test_re_exported_from_contracts_init():
    """All four new models must be importable from src.contracts directly."""
    from src.contracts import (  # noqa: F401
        CVEEnrichment,
        SystemContext,
        TriageResult,
        VulnerabilityGroup,
    )
