"""
Tests for src/tools/semgrep_parser.py

All network calls are mocked — no live Semgrep API required.

Coverage
--------
_extract_findings_page
  - {"findings": [...]}
  - {"results": [...]}
  - {"data": [...]}
  - unknown shape returns []


normalize_finding
  - non-OPEN statuses return None
  - OPEN (and missing status) passes through
  - returns VulnerabilityIssue with source=SEMGREP
  - rule_id priority: rule_name > rule.name > rule_id
  - finding_id from id / finding_id / uuid
  - severity coercion, unknown severity → UNKNOWN
  - line_range built when start >= 1
  - zero / missing line numbers → line_range None
  - file_path from location or path fallback
  - message from rule_message > rule.message > message
  - finding_url from line_of_code_url > url > finding_url
  - repo_url and base_ref from repository dict
  - sast issue_type mapped correctly
  - sca issue_type mapped correctly
  - raw_payload preserved

fetch_findings
  - iterates over sast then sca issue_types
  - stops each type on empty page
  - aggregates findings from both types
  - increments page within each type
  - issue_type param sent correctly

export_to_jsonl
  - writes one line per issue
  - each line round-trips via VulnerabilityIssue.model_validate_json

export_to_csv
  - correct CSV_HEADERS written
  - row fields match issue values
  - empty list writes header-only file

compat shim (parse_semgrep)
  - normalize_finding importable and works
  - _extract_findings_page importable and works
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.contracts import CWEEntry, IssueSource, IssueType, Severity, VulnerabilityIssue
from src.tools.semgrep_parser import (
    CSV_HEADERS,
    _extract_findings_page,
    export_to_csv,
    export_to_jsonl,
    fetch_findings,
    normalize_finding,
)


# ===========================================================================
# Helpers / factories
# ===========================================================================


def _raw(
    status: str = "OPEN",
    rule_name: str = "js.sql-injection",
    severity: str = "HIGH",
    **overrides,
) -> Dict[str, Any]:
    """Minimal valid raw SAST Semgrep finding."""
    base: Dict[str, Any] = {
        "id": "find-001",
        "status": status,
        "rule_name": rule_name,
        "severity": severity,
        "location": {"file_path": "routes/user.ts", "line": 42, "end_line": 44},
        "rule_message": "Possible SQL injection.",
        "repository": {
            "name": "juice-shop",
            "url": "https://github.com/juice-shop/juice-shop",
            "ref": "main",
        },
        "line_of_code_url": "https://semgrep.dev/orgs/acme/findings/1",
    }
    base.update(overrides)
    return base


def _mock_response(body: Any, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _session_with_pages(pages: List[Any]) -> MagicMock:
    """Build a mock session whose .get() returns pages in sequence."""
    session = MagicMock()
    session.get.side_effect = [_mock_response(p) for p in pages]
    return session


# ===========================================================================
# _extract_findings_page
# ===========================================================================


class TestExtractFindingsPage:
    def test_findings_key(self):
        assert _extract_findings_page({"findings": [{"id": 1}]}) == [{"id": 1}]

    def test_results_key(self):
        assert _extract_findings_page({"results": [{"id": 2}]}) == [{"id": 2}]

    def test_data_key(self):
        assert _extract_findings_page({"data": [{"id": 3}]}) == [{"id": 3}]

    def test_unknown_shape_returns_empty(self):
        assert _extract_findings_page({"other": [{"id": 4}]}) == []

    def test_empty_dict_returns_empty(self):
        assert _extract_findings_page({}) == []




# ===========================================================================
# normalize_finding — status filtering
# ===========================================================================


class TestNormalizeFindingStatus:
    @pytest.mark.parametrize("status", ["fixed", "ignored", "removed", "IGNORED", "FIXED"])
    def test_non_open_returns_none(self, status):
        assert normalize_finding(_raw(status=status)) is None

    def test_open_passes(self):
        assert normalize_finding(_raw(status="OPEN")) is not None

    def test_missing_status_passes(self):
        finding = _raw()
        del finding["status"]
        assert normalize_finding(finding) is not None


# ===========================================================================
# normalize_finding — field extraction
# ===========================================================================


class TestNormalizeFindingFields:
    def test_source_is_semgrep(self):
        issue = normalize_finding(_raw())
        assert issue.source == IssueSource.SEMGREP

    def test_issue_type_sast(self):
        issue = normalize_finding(_raw())
        assert issue.issue_type == IssueType.SAST

    def test_issue_type_sca(self):
        finding = _raw()
        finding["issue_type"] = "sca"
        issue = normalize_finding(finding)
        assert issue.issue_type == IssueType.SCA

    def test_cwe_parsing(self):
        finding = _raw()
        finding["rule"] = {
            "cwe_names": [
                "CWE-94: Improper Control of Generation of Code ('Code Injection')",
                "CWE-79",
                "invalid",
            ]
        }
        issue = normalize_finding(finding)
        assert len(issue.cwe) == 2
        assert issue.cwe[0].id == "CWE-94"
        assert issue.cwe[0].name == "Improper Control of Generation of Code ('Code Injection')"
        assert issue.cwe[1].id == "CWE-79"
        assert issue.cwe[1].name is None

    def test_owasp_parsing(self):
        finding = _raw()
        finding["rule"] = {
            "owasp_names": [
                "A03:2021 - Injection",
                "A05:2025 - Injection",
                "A1:2017",
            ]
        }
        issue = normalize_finding(finding)
        assert issue.owasp == ["A03:2021", "A05:2025", "A1:2017"]

    def test_sca_fields_extraction_from_sca_info(self):
        finding = _raw(issue_type="sca")
        finding["sca_info"] = {
            "dependency_name": "lodash",
            "found_version": "4.17.20",
            "fix_versions": ["4.17.21"],
            "ecosystem": "npm",
            "vulnerability_id": "CVE-2020-8203",
        }
        issue = normalize_finding(finding)
        assert issue.package_name == "lodash"
        assert issue.package_version == "4.17.20"
        assert issue.fixed_version == "4.17.21"
        assert issue.ecosystem == "npm"
        assert issue.purl == "pkg:npm/lodash@4.17.20"
        assert issue.cve_id == "CVE-2020-8203"

    def test_sca_fields_extraction_from_dep_matches(self):
        finding = _raw(issue_type="sca")
        finding["extra"] = {
            "dependency_matches": [{
                "dependency": {
                    "package": {"name": "lodash", "ecosystem": "npm"},
                    "version": "4.17.20",
                },
                "fix_versions": ["4.17.21"],
            }]
        }
        issue = normalize_finding(finding)
        assert issue.package_name == "lodash"
        assert issue.package_version == "4.17.20"
        assert issue.fixed_version == "4.17.21"
        assert issue.ecosystem == "npm"
        assert issue.purl == "pkg:npm/lodash@4.17.20"

    def test_sca_fields_ecosystem_inference_and_cve_fallback(self):
        finding = _raw(issue_type="sca")
        finding["location"] = {"file_path": "yarn.lock"}
        finding["package_name"] = "lodash"
        finding["found_version"] = "4.17.20"
        finding["rule_message"] = "Vulnerable dependency. See CVE-2020-8203 for info."
        issue = normalize_finding(finding)
        assert issue.ecosystem == "npm"
        assert issue.purl == "pkg:npm/lodash@4.17.20"
        assert issue.cve_id == "CVE-2020-8203"

    def test_rule_id_from_rule_name(self):
        issue = normalize_finding(_raw(rule_name="js.sqli"))
        assert issue.rule_id == "js.sqli"

    def test_rule_id_from_nested_rule_name(self):
        finding = _raw()
        del finding["rule_name"]
        finding["rule"] = {"name": "rule.nested"}
        issue = normalize_finding(finding)
        assert issue.rule_id == "rule.nested"

    def test_rule_id_from_rule_id_field(self):
        finding = _raw()
        del finding["rule_name"]
        finding["rule_id"] = "direct.rule_id"
        issue = normalize_finding(finding)
        assert issue.rule_id == "direct.rule_id"

    def test_finding_id_from_id(self):
        issue = normalize_finding(_raw(id="abc-123"))
        assert issue.finding_id == "abc-123"

    def test_finding_id_from_uuid(self):
        finding = _raw()
        del finding["id"]
        finding["uuid"] = "uuid-456"
        issue = normalize_finding(finding)
        assert issue.finding_id == "uuid-456"

    def test_severity_high(self):
        assert normalize_finding(_raw(severity="HIGH")).severity == Severity.HIGH

    def test_severity_coercion_lowercase(self):
        assert normalize_finding(_raw(severity="critical")).severity == Severity.CRITICAL

    def test_severity_unknown_for_garbage(self):
        assert normalize_finding(_raw(severity="GARBAGE")).severity == Severity.UNKNOWN

    def test_line_range_populated(self):
        issue = normalize_finding(_raw())
        assert issue.line_range is not None
        assert issue.line_range.start == 42
        assert issue.line_range.end == 44

    def test_line_range_single_line_when_end_missing(self):
        finding = _raw()
        finding["location"] = {"file_path": "f.ts", "line": 7}
        issue = normalize_finding(finding)
        assert issue.line_range is not None
        assert issue.line_range.start == 7
        assert issue.line_range.end == 7

    def test_line_range_none_for_zero_line(self):
        finding = _raw()
        finding["location"] = {"file_path": "f.ts", "line": 0, "end_line": 0}
        issue = normalize_finding(finding)
        assert issue.line_range is None

    def test_file_path_from_location(self):
        assert normalize_finding(_raw()).file_path == "routes/user.ts"

    def test_file_path_from_path_key(self):
        finding = _raw()
        finding["location"] = {}
        finding["path"] = "src/app.ts"
        issue = normalize_finding(finding)
        assert issue.file_path == "src/app.ts"

    def test_message_from_rule_message(self):
        assert normalize_finding(_raw()).message == "Possible SQL injection."

    def test_message_from_nested_rule(self):
        finding = _raw()
        del finding["rule_message"]
        finding["rule"] = {"message": "Nested msg."}
        issue = normalize_finding(finding)
        assert issue.message == "Nested msg."

    def test_message_from_message_field(self):
        finding = _raw()
        del finding["rule_message"]
        finding["message"] = "Top-level msg."
        issue = normalize_finding(finding)
        assert issue.message == "Top-level msg."

    def test_finding_url_from_line_of_code_url(self):
        issue = normalize_finding(_raw())
        assert issue.finding_url == "https://semgrep.dev/orgs/acme/findings/1"

    def test_finding_url_fallback_to_url(self):
        finding = _raw()
        del finding["line_of_code_url"]
        finding["url"] = "https://alt.url/1"
        issue = normalize_finding(finding)
        assert issue.finding_url == "https://alt.url/1"

    def test_repo_url_and_base_ref(self):
        issue = normalize_finding(_raw())
        assert issue.repo_url == "https://github.com/juice-shop/juice-shop"
        assert issue.base_ref == "main"

    def test_raw_payload_preserved(self):
        finding = _raw()
        issue = normalize_finding(finding)
        assert issue.raw_payload is not None
        assert issue.raw_payload["id"] == "find-001"


# ===========================================================================
# fetch_findings — pagination and issue_type iteration
# ===========================================================================


class TestFetchFindings:
    def test_iterates_sast_then_sca(self):
        """
        fetch_findings makes requests for sast first, then sca.
        Each type gets an empty terminator to stop pagination.
        """
        sast_page = {"findings": [_raw(id="s1"), _raw(id="s2")]}
        sast_empty = {"findings": []}
        sca_page = {"findings": [_raw(id="c1", package_name="lodash")]}
        sca_empty = {"findings": []}

        session = _session_with_pages([sast_page, sast_empty, sca_page, sca_empty])
        result = fetch_findings(session, "acme")

        assert len(result) == 3
        # Four total calls: sast page0, sast page1 (empty), sca page0, sca page1 (empty)
        assert session.get.call_count == 4
        assert result[0]["issue_type"] == "sast"
        assert result[1]["issue_type"] == "sast"
        assert result[2]["issue_type"] == "sca"

    def test_issue_type_param_sent_for_sast(self):
        session = _session_with_pages([
            {"findings": []},  # sast empty
            {"findings": []},  # sca empty
        ])
        fetch_findings(session, "acme")
        first_call_params = session.get.call_args_list[0][1]["params"]
        assert first_call_params["issue_type"] == "sast"

    def test_issue_type_param_sent_for_sca(self):
        session = _session_with_pages([
            {"findings": []},  # sast empty
            {"findings": []},  # sca empty
        ])
        fetch_findings(session, "acme")
        second_call_params = session.get.call_args_list[1][1]["params"]
        assert second_call_params["issue_type"] == "sca"

    def test_page_increments_within_type(self):
        """Pages within a single issue_type are incremented from 0."""
        sast_p0 = {"findings": [_raw(id="0")]}
        sast_p1 = {"findings": [_raw(id="1")]}
        sast_empty = {"findings": []}
        sca_empty = {"findings": []}

        session = _session_with_pages([sast_p0, sast_p1, sast_empty, sca_empty])
        result = fetch_findings(session, "acme")
        assert len(result) == 2

        pages = [
            c[1]["params"]["page"]
            for c in session.get.call_args_list
            if c[1]["params"]["issue_type"] == "sast"
        ]
        assert pages == [0, 1, 2]

    def test_stops_on_empty_page(self):
        """If first page for a type is empty, no further pages are fetched for it."""
        session = _session_with_pages([
            {"findings": []},  # sast immediately empty
            {"findings": []},  # sca immediately empty
        ])
        result = fetch_findings(session, "acme")
        assert result == []
        assert session.get.call_count == 2

    def test_results_key_response_shape(self):
        session = _session_with_pages([
            {"results": [_raw(id="r1")]},
            {"results": []},
            {"results": []},
        ])
        result = fetch_findings(session, "acme")
        assert len(result) == 1

    def test_data_key_response_shape(self):
        session = _session_with_pages([
            {"data": [_raw(id="d1")]},
            {"data": []},
            {"data": []},
        ])
        result = fetch_findings(session, "acme")
        assert len(result) == 1


# ===========================================================================
# export_to_jsonl
# ===========================================================================


class TestExportToJsonl:
    def test_writes_one_line_per_issue(self, tmp_path):
        issues = [normalize_finding(_raw(id=str(i))) for i in range(3)]
        issues = [i for i in issues if i is not None]
        out = tmp_path / "out.jsonl"
        export_to_jsonl(issues, out)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_each_line_round_trips(self, tmp_path):
        issues = [normalize_finding(_raw(id=str(i))) for i in range(3)]
        issues = [i for i in issues if i is not None]
        out = tmp_path / "out.jsonl"
        export_to_jsonl(issues, out)
        for line in out.read_text(encoding="utf-8").strip().split("\n"):
            loaded = VulnerabilityIssue.model_validate_json(line)
            assert loaded.source == IssueSource.SEMGREP

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "out.jsonl"
        export_to_jsonl([], out)
        assert out.exists()

    def test_empty_list_writes_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        export_to_jsonl([], out)
        assert out.read_text(encoding="utf-8") == ""


# ===========================================================================
# export_to_csv
# ===========================================================================


class TestExportToCsv:
    def test_correct_headers(self, tmp_path):
        out = tmp_path / "out.csv"
        export_to_csv([], out)
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_HEADERS

    def test_row_content(self, tmp_path):
        issue = normalize_finding(_raw())
        assert issue is not None
        out = tmp_path / "out.csv"
        export_to_csv([issue], out)
        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["Repository"] == "juice-shop"
        assert row["Issue_Type"] == "sast"
        assert row["Rule_ID"] == "js.sql-injection"
        assert row["Severity"] == "HIGH"
        assert row["File_Path"] == "routes/user.ts"
        assert row["Line_Start"] == "42"
        assert row["Line_End"] == "44"
        assert row["Message"] == "Possible SQL injection."
        assert row["Finding_URL"] == "https://semgrep.dev/orgs/acme/findings/1"

    def test_empty_issues_header_only(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_to_csv([], out)
        with out.open(encoding="utf-8") as f:
            content = f.read()
        assert CSV_HEADERS[0] in content
        # Only one newline (header row)
        assert content.count("\n") == 1

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "out.csv"
        export_to_csv([], out)
        assert out.exists()


# ===========================================================================
# compat shim (parse_semgrep)
# ===========================================================================


class TestCompatShim:
    def test_normalize_finding_importable(self):
        from src.tools.parse_semgrep import normalize_finding as shim_nf
        issue = shim_nf(_raw())
        assert issue is not None
        assert issue.source == IssueSource.SEMGREP

    def test_extract_findings_page_importable(self):
        from src.tools.parse_semgrep import _extract_findings_page as shim_efp
        assert shim_efp({"findings": [{"id": 1}]}) == [{"id": 1}]

    def test_export_to_jsonl_importable(self, tmp_path):
        from src.tools.parse_semgrep import export_to_jsonl as shim_jsonl
        issue = normalize_finding(_raw())
        out = tmp_path / "shim.jsonl"
        shim_jsonl([issue], out)
        assert out.exists()

