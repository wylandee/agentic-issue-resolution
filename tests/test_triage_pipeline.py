"""
tests/test_triage_pipeline.py — Unit tests for src/triage/pipeline.py.

Covers:
- Mixed SAST+SCA issues → correct group count and types
- Invalid groups excluded from select_issues_for_remediation
- Selected issues are valid VulnerabilityIssue objects (passable to run_remediation)
- Priority sorting: CRITICAL before HIGH before MEDIUM
- Empty input → empty result
- run_triage_pipeline with mocked enrichment (no real HTTP)
"""

from __future__ import annotations

from typing import List
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
from src.triage.pipeline import run_triage_pipeline, select_issues_for_remediation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def disable_llm_triage_by_default(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")

def _ctx() -> SystemContext:
    return SystemContext(environment="production")


def _sca(
    *,
    package_name: str = "lodash",
    file_path: str = "package.json",
    cve_id: str | None = "CVE-2021-23337",
    severity: Severity = Severity.HIGH,
    source: IssueSource = IssueSource.SEMGREP,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SCA,
        severity=severity,
        package_name=package_name,
        cve_id=cve_id,
        file_path=file_path,
    )


def _sast(
    *,
    file_path: str = "src/app.js",
    rule_id: str = "javascript.xss",
    line_start: int = 10,
    line_end: int = 15,
    severity: Severity = Severity.MEDIUM,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=severity,
        file_path=file_path,
        rule_id=rule_id,
        line_range=LineRange(start=line_start, end=line_end),
    )


def _empty_enrichment_map(cve_ids: List[str]):
    """Return safe-default CVEEnrichment for all requested CVEs."""
    return {
        cve: CVEEnrichment(cve_id=cve, epss=0.0, in_kev=False, enrichment_source="none")
        for cve in cve_ids
    }


# ---------------------------------------------------------------------------
# run_triage_pipeline
# ---------------------------------------------------------------------------


class TestRunTriagePipeline:
    def test_empty_input_returns_empty(self):
        with patch("src.triage.pipeline.enrich_cves", return_value={}):
            result = run_triage_pipeline([], _ctx())
        assert result == []

    def test_sca_issue_produces_one_pair(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]
        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 1
        group, triage = result[0]
        assert isinstance(group, VulnerabilityGroup)
        assert isinstance(triage, TriageResult)

    def test_sast_issue_produces_one_pair(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sast()]
        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 1

    def test_mixed_input_correct_group_count(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [
            _sca(package_name="lodash", file_path="package.json"),
            _sca(package_name="express", file_path="backend/package.json"),
            _sast(file_path="src/login.js", rule_id="sqli", line_start=5, line_end=5),
        ]
        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 3

    def test_all_results_are_tuples_of_correct_types(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca(), _sast()]
        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map):
            result = run_triage_pipeline(issues, _ctx())

        for pair in result:
            assert len(pair) == 2
            group, triage = pair
            assert isinstance(group, VulnerabilityGroup)
            assert isinstance(triage, TriageResult)

    def test_cve_enrichment_attached_to_group(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        kev_enrichment = CVEEnrichment(
            cve_id="CVE-2021-23337",
            epss=0.8,
            in_kev=True,
            enrichment_source="epss+kev",
        )
        with patch("src.triage.pipeline.enrich_cves", return_value={"CVE-2021-23337": kev_enrichment}):
            result = run_triage_pipeline([_sca(cve_id="CVE-2021-23337")], _ctx())

        group, triage = result[0]
        assert group.enrichment is not None
        assert group.enrichment.in_kev is True
        assert triage.revised_priority == Severity.CRITICAL

    def test_reachability_runs_when_repo_root_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]

        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map), patch(
            "src.triage.pipeline.analyze_reachability"
        ) as mock_reachability:
            run_triage_pipeline(issues, _ctx(), repo_root=str(tmp_path))

        mock_reachability.assert_called_once()

    def test_reachability_skipped_when_repo_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]
        missing_root = tmp_path / "missing-repo"

        with patch("src.triage.pipeline.enrich_cves", side_effect=_empty_enrichment_map), patch(
            "src.triage.pipeline.analyze_reachability"
        ) as mock_reachability:
            run_triage_pipeline(issues, _ctx(), repo_root=str(missing_root))

        mock_reachability.assert_not_called()


# ---------------------------------------------------------------------------
# select_issues_for_remediation
# ---------------------------------------------------------------------------


class TestSelectIssuesForRemediation:
    def _make_pair(
        self,
        issue: VulnerabilityIssue,
        *,
        is_valid: bool = True,
        priority: Severity = Severity.HIGH,
        fp_reason: str | None = None,
    ):
        group = VulnerabilityGroup(
            group_id=f"sca:{issue.file_path}:{issue.package_name}",
            issue_type=issue.issue_type,
            representative_issue_id=issue.id,
            issues=[issue],
        )
        triage = TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=is_valid,
            false_positive_reason=fp_reason if not is_valid else None,
            original_severity=issue.severity,
            revised_priority=priority,
            is_unreachable_code=False,
            priority_reasoning="test",
            validity_confidence_score=1.0,
            priority_confidence_score=1.0,
            recommended_issue_id=issue.id,
            triage_method="deterministic",
        )
        return group, triage

    def test_invalid_groups_excluded(self):
        valid_issue = _sca(package_name="lodash", file_path="package.json")
        invalid_issue = _sca(package_name="mocha", file_path="package.json")

        pairs = [
            self._make_pair(valid_issue, is_valid=True, priority=Severity.HIGH),
            self._make_pair(
                invalid_issue,
                is_valid=False,
                fp_reason="Test-only dependency.",
                priority=Severity.LOW,
            ),
        ]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 1
        assert selected[0].package_name == "lodash"

    def test_selected_issues_are_vulnerability_issues(self):
        issue = _sca()
        pairs = [self._make_pair(issue)]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 1
        assert isinstance(selected[0], VulnerabilityIssue)

    def test_priority_sort_critical_before_high_before_medium(self):
        critical_issue = _sca(package_name="pkg_c", file_path="c/package.json", severity=Severity.CRITICAL)
        high_issue = _sca(package_name="pkg_h", file_path="h/package.json", severity=Severity.HIGH)
        medium_issue = _sca(package_name="pkg_m", file_path="m/package.json", severity=Severity.MEDIUM)

        # Provide in wrong order intentionally
        pairs = [
            self._make_pair(medium_issue, priority=Severity.MEDIUM),
            self._make_pair(critical_issue, priority=Severity.CRITICAL),
            self._make_pair(high_issue, priority=Severity.HIGH),
        ]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 3
        assert selected[0].package_name == "pkg_c"
        assert selected[1].package_name == "pkg_h"
        assert selected[2].package_name == "pkg_m"

    def test_empty_input_returns_empty(self):
        assert select_issues_for_remediation([]) == []

    def test_all_invalid_returns_empty(self):
        issues = [_sca(package_name=f"pkg{i}", file_path=f"p{i}/pkg.json") for i in range(3)]
        pairs = [
            self._make_pair(
                i,
                is_valid=False,
                fp_reason="dev-only",
                priority=Severity.LOW,
            )
            for i in issues
        ]
        selected = select_issues_for_remediation(pairs)
        assert selected == []

    def test_selected_issue_ids_match_recommended(self):
        issue = _sca()
        group = VulnerabilityGroup(
            group_id="sca:package.json:lodash",
            issue_type=IssueType.SCA,
            representative_issue_id=issue.id,
            issues=[issue],
        )
        triage = TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=True,
            original_severity=issue.severity,
            revised_priority=Severity.HIGH,
            is_unreachable_code=False,
            priority_reasoning="test",
            validity_confidence_score=1.0,
            priority_confidence_score=1.0,
            recommended_issue_id=issue.id,
            triage_method="deterministic",
        )
        selected = select_issues_for_remediation([(group, triage)])
        assert selected[0].id == issue.id
