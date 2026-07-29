"""
Tests for remediation_engine.tools.semgrep_parser.

All tests are local-file based - no live Semgrep API required.
"""

from __future__ import annotations

import csv
import json

from remediation_engine.contracts import IssueSource, IssueType, Severity, VulnerabilityIssue
from remediation_engine.tools import semgrep_parser
from remediation_engine.tools.semgrep_parser import (
    CSV_HEADERS,
    _extract_findings_page,
    export_to_csv,
    export_to_jsonl,
    fetch_findings,
    load_findings_from_json,
    main,
    normalize_finding,
)


def _cli_sast_raw(**overrides):
    finding = {
        "check_id": "javascript.crypto.weak-hash",
        "path": "Gruntfile.js",
        "start": {"line": 76, "col": 37, "offset": 2408},
        "end": {"line": 76, "col": 42, "offset": 2413},
        "extra": {
            "message": "Weak cryptographic algorithm detected.",
            "severity": "WARNING",
            "fingerprint": "fp-sast-001",
            "is_ignored": False,
            "dataflow_trace": {
                "taint_source": ["CliLoc", [{"path": "Gruntfile.js"}, "'md5'"]],
                "intermediate_vars": [],
                "taint_sink": ["CliLoc", [{"path": "Gruntfile.js"}, "'md5'"]],
            },
            "metadata": {
                "confidence": "HIGH",
                "cwe": ["CWE-327: Use of a Broken or Risky Cryptographic Algorithm"],
                "owasp": ["A02:2021 - Cryptographic Failures"],
                "semgrep.url": "https://semgrep.dev/r/javascript.crypto.weak-hash",
            },
        },
    }
    finding.update(overrides)
    return finding


def _cli_sca_raw(**overrides):
    finding = {
        "check_id": "ssc-parity-001",
        "path": "frontend/package-lock.json",
        "start": {"line": 7406, "col": 1, "offset": 0},
        "end": {"line": 7406, "col": 20, "offset": 0},
        "extra": {
            "message": "Vulnerable dependency detected.",
            "severity": "ERROR",
            "fingerprint": "fp-sca-001",
            "is_ignored": False,
            "metadata": {
                "confidence": "MEDIUM",
                "cwe": ["CWE-1104: Use of Unmaintained Third Party Components"],
                "owasp": ["A06:2021 - Vulnerable and Outdated Components"],
                "semgrep.url": "https://semgrep.dev/r/ssc-parity-001",
            },
            "sca_info": {
                "reachability_rule": False,
                "sca_finding_schema": 20220913,
                "dependency_match": {
                    "dependency_pattern": {
                        "ecosystem": "npm",
                        "package": "elliptic",
                        "semver_range": "<=6.6.1",
                    },
                    "found_dependency": {
                        "package": "elliptic",
                        "version": "6.6.1",
                        "ecosystem": "npm",
                        "manifest_path": "frontend/package.json",
                        "lockfile_path": "frontend/package-lock.json",
                        "line_number": 7406,
                    },
                    "lockfile": "frontend/package-lock.json",
                },
                "reachable": False,
            },
        },
    }
    finding.update(overrides)
    return finding


class TestExtractFindingsPage:
    def test_results_key(self):
        assert _extract_findings_page({"results": [{"check_id": "1"}]}) == [{"check_id": "1"}]

    def test_findings_key(self):
        assert _extract_findings_page({"findings": [{"id": 1}]}) == [{"id": 1}]

    def test_data_key(self):
        assert _extract_findings_page({"data": [{"id": 2}]}) == [{"id": 2}]

    def test_unknown_shape_returns_empty(self):
        assert _extract_findings_page({"other": []}) == []


class TestLoadFindingsFromJson:
    def test_loads_results_from_top_level_object(self, tmp_path):
        json_path = tmp_path / "semgrep.json"
        payload = {"version": "1", "results": [_cli_sast_raw(), _cli_sca_raw()]}
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        findings = load_findings_from_json(json_path)

        assert len(findings) == 2
        assert findings[0]["check_id"] == "javascript.crypto.weak-hash"

    def test_fetch_findings_compat_wrapper_uses_json_file(self, tmp_path):
        json_path = tmp_path / "semgrep.json"
        json_path.write_text(json.dumps({"results": [_cli_sast_raw()]}), encoding="utf-8")

        findings = fetch_findings(None, None, json_path=json_path)

        assert len(findings) == 1


class TestNormalizeFinding:
    def test_source_is_semgrep(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.source == IssueSource.SEMGREP

    def test_issue_type_is_sast_without_extra_sca_info(self):
        issue = normalize_finding(_cli_sast_raw(issue_type="sca"))
        assert issue is not None
        assert issue.issue_type == IssueType.SAST

    def test_issue_type_is_sca_only_when_extra_sca_info_exists(self):
        issue = normalize_finding(_cli_sca_raw())
        assert issue is not None
        assert issue.issue_type == IssueType.SCA

    def test_rule_id_comes_from_check_id(self):
        issue = normalize_finding(_cli_sast_raw(check_id="custom.rule"))
        assert issue is not None
        assert issue.rule_id == "custom.rule"

    def test_finding_id_prefers_extra_fingerprint(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.finding_id == "fp-sast-001"

    def test_file_path_and_line_range_come_from_cli_fields(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.file_path == "Gruntfile.js"
        assert issue.line_range is not None
        assert issue.line_range.start == 76
        assert issue.line_range.end == 76

    def test_message_and_finding_url_come_from_extra(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.message == "Weak cryptographic algorithm detected."
        assert issue.finding_url == "https://semgrep.dev/r/javascript.crypto.weak-hash"

    def test_confidence_comes_from_extra_metadata(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.confidence == "HIGH"

    def test_cwe_and_owasp_come_from_extra_metadata(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.cwe[0].id == "CWE-327"
        assert issue.owasp == ["A02:2021"]

    def test_sast_dataflow_trace_is_preserved(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.dataflow_trace is not None
        assert issue.dataflow_trace["intermediate_vars"] == []

    def test_sca_does_not_copy_dataflow_trace(self):
        finding = _cli_sca_raw()
        finding["extra"]["dataflow_trace"] = {"taint_source": []}
        issue = normalize_finding(finding)
        assert issue is not None
        assert issue.dataflow_trace is None

    def test_sca_fields_come_from_extra_sca_info(self):
        issue = normalize_finding(_cli_sca_raw())
        assert issue is not None
        assert issue.issue_type == IssueType.SCA
        assert issue.package_name == "elliptic"
        assert issue.package_version == "6.6.1"
        assert issue.ecosystem == "npm"
        assert issue.purl == "pkg:npm/elliptic@6.6.1"
        assert issue.fixed_version is None
        assert issue.cve_id is None

    def test_severity_mapping_error_to_high(self):
        issue = normalize_finding(_cli_sca_raw())
        assert issue is not None
        assert issue.severity == Severity.HIGH

    def test_severity_mapping_warning_to_medium(self):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        assert issue.severity == Severity.MEDIUM

    def test_severity_mapping_critical_to_critical(self):
        issue = normalize_finding(
            _cli_sast_raw(
                extra={"message": "x", "severity": "CRITICAL", "fingerprint": "f", "metadata": {}}
            )
        )
        assert issue is not None
        assert issue.severity == Severity.CRITICAL

    def test_unknown_severity_becomes_unknown(self):
        issue = normalize_finding(
            _cli_sast_raw(
                extra={"message": "x", "severity": "GARBAGE", "fingerprint": "f", "metadata": {}}
            )
        )
        assert issue is not None
        assert issue.severity == Severity.UNKNOWN

    def test_ignored_finding_is_skipped(self):
        issue = normalize_finding(
            _cli_sast_raw(
                extra={
                    "message": "x",
                    "severity": "WARNING",
                    "fingerprint": "f",
                    "metadata": {},
                    "is_ignored": True,
                }
            )
        )
        assert issue is None

    def test_raw_payload_is_preserved(self):
        finding = _cli_sast_raw()
        issue = normalize_finding(finding)
        assert issue is not None
        assert issue.raw_payload == finding


class TestExports:
    def test_export_to_jsonl_round_trips(self, tmp_path):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        out = tmp_path / "out.jsonl"

        export_to_jsonl([issue], out)

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        loaded = VulnerabilityIssue.model_validate_json(lines[0])
        assert loaded.rule_id == issue.rule_id

    def test_export_to_csv_writes_expected_row(self, tmp_path):
        issue = normalize_finding(_cli_sast_raw())
        assert issue is not None
        out = tmp_path / "out.csv"

        export_to_csv([issue], out)

        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0].keys()) == CSV_HEADERS
        assert rows[0]["Issue_Type"] == "sast"
        assert rows[0]["Rule_ID"] == "javascript.crypto.weak-hash"


class TestMain:
    def test_main_reads_local_json_and_writes_outputs(self, tmp_path, monkeypatch):
        json_path = tmp_path / "semgrep.json"
        json_path.write_text(json.dumps({"results": [_cli_sast_raw()]}), encoding="utf-8")

        captured = {}

        def _capture_jsonl(issues, output_path):
            captured["jsonl"] = (issues, output_path)

        def _capture_csv(issues, output_path):
            captured["csv"] = (issues, output_path)

        monkeypatch.setattr(semgrep_parser, "_default_input_json_path", lambda: json_path)
        monkeypatch.setattr(semgrep_parser, "export_to_jsonl", _capture_jsonl)
        monkeypatch.setattr(semgrep_parser, "export_to_csv", _capture_csv)

        main()

        assert "jsonl" in captured
        assert "csv" in captured
        assert len(captured["jsonl"][0]) == 1
        assert captured["jsonl"][0][0].rule_id == "javascript.crypto.weak-hash"
