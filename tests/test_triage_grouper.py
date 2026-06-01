"""
tests/test_triage_grouper.py — Unit tests for src/triage/grouper.py.

Covers:
- Same package/file with multiple distinct CVEs → one group, both CVE IDs
- Duplicate CVE on same component/file → deduplicated in group
- Semgrep SCA + ODC SCA for same CVE/component/file → merged, two sources
- SAST issues → singleton groups
- Mixed SAST+SCA list → correct counts and types
- group_sca_issues compatibility alias
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.contracts.schemas import (
    IssueSource,
    IssueType,
    LineRange,
    Severity,
    VulnerabilityIssue,
)
from src.triage.grouper import group_issues, group_sca_issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sca(
    *,
    package_name: str = "lodash",
    file_path: str = "package.json",
    cve_id: str | None = "CVE-2021-23337",
    source: IssueSource = IssueSource.SEMGREP,
    severity: Severity = Severity.HIGH,
    fixed_version: str | None = None,
    package_version: str | None = "4.17.20",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SCA,
        severity=severity,
        package_name=package_name,
        cve_id=cve_id,
        file_path=file_path,
        fixed_version=fixed_version,
        package_version=package_version,
    )


def _sast(
    *,
    file_path: str = "src/app.js",
    rule_id: str = "javascript.xss",
    line_start: int = 10,
    line_end: int = 15,
    source: IssueSource = IssueSource.SEMGREP,
    severity: Severity = Severity.MEDIUM,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SAST,
        severity=severity,
        file_path=file_path,
        rule_id=rule_id,
        line_range=LineRange(start=line_start, end=line_end),
    )


# ---------------------------------------------------------------------------
# SCA grouping
# ---------------------------------------------------------------------------


class TestSCAGrouping:
    def test_same_package_multiple_cves_become_one_group(self):
        issues = [
            _sca(cve_id="CVE-2021-23337"),
            _sca(cve_id="CVE-2020-28500"),  # Different CVE, same component/file
        ]
        groups = group_issues(issues)
        assert len(groups) == 1
        g = groups[0]
        assert set(g.cve_ids) == {"CVE-2021-23337", "CVE-2020-28500"}
        assert g.issue_type == IssueType.SCA

    def test_duplicate_cve_same_component_is_deduped(self):
        """Same CVE appearing twice (e.g. from the same scanner run) → one CVE ID."""
        issues = [
            _sca(cve_id="CVE-2021-23337"),
            _sca(cve_id="CVE-2021-23337"),  # Exact duplicate
        ]
        groups = group_issues(issues)
        assert len(groups) == 1
        assert groups[0].cve_ids == ["CVE-2021-23337"]

    def test_cross_tool_same_cve_component_file_merges(self):
        """Semgrep SCA + ODC finding the same CVE on the same package/file → merge."""
        semgrep_issue = _sca(source=IssueSource.SEMGREP, cve_id="CVE-2021-44228")
        odc_issue = _sca(source=IssueSource.ODC, cve_id="CVE-2021-44228")
        groups = group_issues([semgrep_issue, odc_issue])
        assert len(groups) == 1
        g = groups[0]
        assert IssueSource.SEMGREP in g.sources
        assert IssueSource.ODC in g.sources
        assert len(g.issues) == 2

    def test_different_packages_different_files_make_separate_groups(self):
        issues = [
            _sca(package_name="lodash", file_path="package.json"),
            _sca(package_name="express", file_path="backend/package.json"),
        ]
        groups = group_issues(issues)
        assert len(groups) == 2

    def test_representative_prefers_issue_with_fixed_version(self):
        no_fix = _sca(fixed_version=None)
        with_fix = _sca(fixed_version="4.17.21")
        groups = group_issues([no_fix, with_fix])
        assert len(groups) == 1
        assert groups[0].representative_issue_id == with_fix.id

    def test_versions_deduplicated(self):
        issues = [
            _sca(package_version="4.17.20", cve_id="CVE-2021-23337"),
            _sca(package_version="4.17.20", cve_id="CVE-2020-28500"),
        ]
        groups = group_issues(issues)
        assert groups[0].versions == ["4.17.20"]

    def test_no_cve_sca_still_groups(self):
        issues = [_sca(cve_id=None), _sca(cve_id=None)]
        groups = group_issues(issues)
        # Same component/file → one group; no CVE IDs expected
        assert len(groups) == 1
        assert groups[0].cve_ids == []


# ---------------------------------------------------------------------------
# SAST grouping
# ---------------------------------------------------------------------------


class TestSASTGrouping:
    def test_sast_issue_becomes_singleton_group(self):
        issues = [_sast()]
        groups = group_issues(issues)
        assert len(groups) == 1
        g = groups[0]
        assert g.issue_type == IssueType.SAST
        assert g.group_id.startswith("sast:")
        assert len(g.issues) == 1

    def test_different_sast_locations_make_separate_groups(self):
        issues = [
            _sast(file_path="src/a.js", rule_id="xss", line_start=10, line_end=10),
            _sast(file_path="src/a.js", rule_id="xss", line_start=50, line_end=55),
            _sast(file_path="src/b.js", rule_id="xss", line_start=10, line_end=10),
        ]
        groups = group_issues(issues)
        assert len(groups) == 3

    def test_sast_group_has_empty_cve_ids(self):
        groups = group_issues([_sast()])
        assert groups[0].cve_ids == []

    def test_sast_group_captures_rule_id_as_component(self):
        groups = group_issues([_sast(rule_id="javascript.sqli")])
        assert groups[0].vulnerable_component == "javascript.sqli"


# ---------------------------------------------------------------------------
# Mixed SAST + SCA
# ---------------------------------------------------------------------------


class TestMixedGrouping:
    def test_mixed_list_correct_group_count(self):
        issues = [
            _sca(package_name="lodash", file_path="package.json"),
            _sca(package_name="express", file_path="backend/package.json"),
            _sast(file_path="src/login.js", rule_id="xss", line_start=5, line_end=5),
        ]
        groups = group_issues(issues)
        assert len(groups) == 3

    def test_mixed_list_correct_types(self):
        issues = [
            _sca(),
            _sast(),
        ]
        groups = group_issues(issues)
        types = {g.issue_type for g in groups}
        assert IssueType.SCA in types
        assert IssueType.SAST in types

    def test_empty_input_returns_empty_list(self):
        assert group_issues([]) == []


# ---------------------------------------------------------------------------
# group_sca_issues alias
# ---------------------------------------------------------------------------


class TestGroupSCAAlias:
    def test_alias_excludes_sast(self):
        issues = [_sca(), _sast()]
        sca_groups = group_sca_issues(issues)
        assert all(g.issue_type == IssueType.SCA for g in sca_groups)

    def test_alias_matches_group_issues_on_sca_only_input(self):
        # Create one shared list so both calls operate on the same issue objects
        issues = [_sca(cve_id="CVE-2021-23337"), _sca(cve_id="CVE-2020-28500")]
        alias_result = group_sca_issues(issues)
        direct_result = group_issues(issues)
        assert len(alias_result) == len(direct_result)
        alias_ids = {g.group_id for g in alias_result}
        direct_ids = {g.group_id for g in direct_result}
        assert alias_ids == direct_ids
