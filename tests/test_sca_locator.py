"""
Tests for the upgraded SCA locator layer.

Coverage
--------
odc_parser
  - PURL parsing (npm scoped, plain, javascript ecosystem)
  - Severity fallback chain (highestSeverity / cvssv3 / cvssv2 / raw)
  - CWE normalisation (with/without "CWE-" prefix)
  - CVE-ID extraction vs GHSA advisory IDs
  - parse_vulnerabilities â†’ List[VulnerabilityIssue]
  - export_to_jsonl / export_to_csv round-trips

manifest_locator
  - normalize_package_name (tgz, jar, scoped, colon-version, PURL form)
  - parse_lockfile_path (root entry, transitive chain, node_modules path,
    scoped package, bench.js non-lockfile path)
  - detect_package_manager (npm / yarn / pnpm)
  - _locate_in_manifest (direct + line found, direct no line, transitive)
  - _find_nearest_manifest (nested manifest resolution)
  - locate_dependency (end-to-end, no OSV)
  - locate_from_issue (typed LocalizedIssue returned, no OSV)

All tests are fully offline â€” no live network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts import IssueSource, IssueType, Severity, VulnerabilityIssue
from remediation_engine.tools.manifest_locator import (
    PackageManagerKind,
    _find_nearest_manifest,
    _locate_in_manifest,
    detect_package_manager,
    locate_dependency,
    locate_from_issue,
    normalize_package_name,
    parse_dependency_ancestry,
    parse_lockfile_path,
)
from remediation_engine.tools.odc_parser import (
    _ecosystem_from_purl,
    _extract_cve_id,
    _extract_ghsa_id,
    _extract_severity,
    _package_name_from_purl,
    _parse_cwes,
    _parse_purl,
    _version_from_purl,
    export_to_csv,
    export_to_jsonl,
    parse_vulnerabilities,
)

# ===========================================================================
# Fixtures / helpers
# ===========================================================================


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo layout with package.json."""
    pkg = {
        "name": "my-app",
        "version": "1.0.0",
        "dependencies": {
            "lodash": "^4.17.20",
            "express": "^4.18.0",
        },
        "devDependencies": {
            "mocha": "^9.0.0",
        },
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))
    (tmp_path / "package-lock.json").touch()
    return tmp_path


@pytest.fixture()
def tmp_repo_yarn(tmp_path: Path) -> Path:
    """Repo using Yarn (has yarn.lock, no package-lock.json)."""
    pkg = {
        "name": "yarn-app",
        "version": "1.0.0",
        "dependencies": {"lodash": "^4.17.20"},
        "resolutions": {"lodash": "4.17.21"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))
    (tmp_path / "yarn.lock").touch()
    return tmp_path


@pytest.fixture()
def tmp_repo_pnpm(tmp_path: Path) -> Path:
    """Repo using pnpm."""
    pkg = {
        "name": "pnpm-app",
        "version": "1.0.0",
        "dependencies": {"lodash": "^4.17.20"},
        "pnpm": {"overrides": {}},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))
    (tmp_path / "pnpm-lock.yaml").touch()
    return tmp_path


@pytest.fixture()
def tmp_repo_nested(tmp_path: Path) -> Path:
    """Repo with a nested frontend sub-package."""
    root_pkg = {
        "name": "root-app",
        "version": "1.0.0",
        "dependencies": {"express": "^4.18.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(root_pkg, indent=2))
    (tmp_path / "package-lock.json").touch()

    frontend = tmp_path / "frontend"
    frontend.mkdir()
    fe_pkg = {
        "name": "frontend-app",
        "version": "1.0.0",
        "dependencies": {"elliptic": "^6.6.1"},
    }
    (frontend / "package.json").write_text(json.dumps(fe_pkg, indent=2))
    (frontend / "package-lock.json").touch()
    return tmp_path


def _minimal_dep(
    file_name: str = "lodash-4.17.20.tgz",
    file_path: str = "/src/package.json",
    purl: str = "pkg:npm/lodash@4.17.20",
    cve_name: str = "CVE-2021-23337",
    severity: str = "HIGH",
    cwes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal ODC dependency dict."""
    return {
        "fileName": file_name,
        "filePath": file_path,
        "packages": [{"id": purl, "confidence": "HIGHEST"}],
        "vulnerabilities": [
            {
                "source": "NVD",
                "name": cve_name,
                "severity": severity,
                "cvssv3": {"baseSeverity": severity},
                "cwes": cwes or ["CWE-77"],
                "description": f"Test vuln for {file_name}",
            }
        ],
    }


# ===========================================================================
# odc_parser â€” PURL helpers
# ===========================================================================


class TestPURLHelpers:
    """Tests for the private PURL-parsing functions in odc_parser."""

    @pytest.mark.parametrize(
        "purl,expected_name",
        [
            ("pkg:npm/lodash@4.17.20", "lodash"),
            ("pkg:npm/@scope/pkg@1.2.3", "@scope/pkg"),
            ("pkg:npm/%40tootallnate%2Fonce@1.1.2", "@tootallnate/once"),
            ("pkg:npm/base64url@0.0.6", "base64url"),
            ("pkg:javascript/underscore.js@1.7.0", "underscore.js"),
            # maven: namespace=commons-io, name=commons-io â†’ joined with ':'
            ("pkg:maven/commons-io/commons-io@2.4", "commons-io:commons-io"),
        ],
    )
    def test_package_name_from_purl(self, purl, expected_name):

        assert _package_name_from_purl(purl) == expected_name

    @pytest.mark.parametrize(
        "purl,expected_eco",
        [
            ("pkg:npm/lodash@4.17.20", "npm"),
            ("pkg:javascript/underscore.js@1.7.0", "javascript"),
            ("pkg:maven/commons-io/commons-io@2.4", "maven"),
        ],
    )
    def test_ecosystem_from_purl(self, purl, expected_eco):
        assert _ecosystem_from_purl(purl) == expected_eco

    @pytest.mark.parametrize(
        "purl,expected_version",
        [
            ("pkg:npm/lodash@4.17.20", "4.17.20"),
            ("pkg:npm/%40tootallnate%2Fonce@1.1.2", "1.1.2"),
            ("pkg:npm/elliptic@6.6.1", "6.6.1"),
        ],
    )
    def test_version_from_purl(self, purl, expected_version):
        assert _version_from_purl(purl) == expected_version

    def test_ecosystem_from_none_purl(self):
        assert _ecosystem_from_purl(None) is None

    def test_package_name_from_none_purl(self):

        assert _package_name_from_purl(None) is None

    def test_parse_purl_from_dep(self):
        dep = {"packages": [{"id": "pkg:npm/lodash@4.17.20", "confidence": "HIGHEST"}]}
        assert _parse_purl(dep) == "pkg:npm/lodash@4.17.20"

    def test_parse_purl_missing(self):
        assert _parse_purl({}) is None
        assert _parse_purl({"packages": []}) is None


# ===========================================================================
# odc_parser â€” severity extraction
# ===========================================================================


class TestSeverityExtraction:
    @pytest.mark.parametrize(
        "vuln,expected",
        [
            ({"highestSeverity": "CRITICAL"}, Severity.CRITICAL),
            ({"cvssv3": {"baseSeverity": "HIGH"}}, Severity.HIGH),
            ({"cvssv2": {"severity": "MEDIUM"}}, Severity.MEDIUM),
            ({"severity": "low"}, Severity.LOW),
            # highestSeverity takes precedence over cvssv3
            (
                {"highestSeverity": "CRITICAL", "cvssv3": {"baseSeverity": "HIGH"}},
                Severity.CRITICAL,
            ),
            # Unknown garbage â†’ UNKNOWN
            ({"severity": "moderate"}, Severity.UNKNOWN),  # ODC moderate is not a valid Severity
            ({}, Severity.UNKNOWN),
        ],
    )
    def test_severity_fallback_chain(self, vuln, expected):
        assert _extract_severity(vuln) == expected

    def test_severity_case_insensitive(self):
        assert _extract_severity({"severity": "HIGH"}) == Severity.HIGH
        assert _extract_severity({"highestSeverity": "critical"}) == Severity.CRITICAL


# ===========================================================================
# odc_parser â€” CWE parsing
# ===========================================================================


class TestCWEParsing:
    @pytest.mark.parametrize(
        "raw,expected_ids",
        [
            (["CWE-77"], ["CWE-77"]),
            (["77"], ["CWE-77"]),
            (["CWE79", "CWE-89"], ["CWE-79", "CWE-89"]),
            ([], []),
            (["notacwe"], []),  # no digits â†’ skipped
        ],
    )
    def test_parse_cwes(self, raw, expected_ids):
        vuln = {"cwes": raw}
        result = _parse_cwes(vuln)
        assert [c.id for c in result] == expected_ids

    def test_parse_cwes_missing_key(self):
        assert _parse_cwes({}) == []


# ===========================================================================
# odc_parser â€” CVE-ID extraction
# ===========================================================================


class TestCVEIDExtraction:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("CVE-2021-23337", "CVE-2021-23337"),
            ("cve-2021-23337", "CVE-2021-23337"),
            ("GHSA-vpq2-c234-7xj6", None),
            ("", None),
        ],
    )
    def test_extract_cve_id(self, name, expected):
        vuln = {"name": name}
        assert _extract_cve_id(vuln) == expected

    def test_extract_cve_id_from_references(self):
        vuln = {
            "name": "GHSA-vpq2-c234-7xj6",
            "references": [
                {"url": "https://nvd.nist.gov/vuln/detail/CVE-2020-15084"},
                {"name": "some random reference"},
            ],
        }
        assert _extract_cve_id(vuln) == "CVE-2020-15084"


class TestGHSAIDExtraction:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("GHSA-vpq2-c234-7xj6", "GHSA-VPQ2-C234-7XJ6"),
            ("ghsa-vpq2-c234-7xj6", "GHSA-VPQ2-C234-7XJ6"),
            ("CVE-2021-23337", None),
            ("", None),
        ],
    )
    def test_extract_ghsa_id(self, name, expected):
        vuln = {"name": name}
        assert _extract_ghsa_id(vuln) == expected

    def test_extract_ghsa_id_from_references(self):
        vuln = {
            "name": "random id",
            "references": [
                {"url": "https://github.com/advisories/GHSA-vpq2-c234-7xj6"},
            ],
        }
        assert _extract_ghsa_id(vuln) == "GHSA-VPQ2-C234-7XJ6"


# ===========================================================================
# odc_parser â€” parse_vulnerabilities
# ===========================================================================


class TestParseVulnerabilities:
    def test_minimal_report(self):
        report = {"dependencies": [_minimal_dep()]}
        issues = parse_vulnerabilities(report)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.source == IssueSource.ODC
        assert issue.issue_type == IssueType.SCA
        assert issue.severity == Severity.HIGH
        assert issue.cve_id == "CVE-2021-23337"
        assert issue.package_name == "lodash"
        assert issue.package_version == "4.17.20"
        assert issue.purl == "pkg:npm/lodash@4.17.20"
        assert issue.ecosystem == "npm"
        assert len(issue.cwe) == 1 and issue.cwe[0].id == "CWE-77"
        assert issue.raw_payload is not None

    def test_ghsa_advisory_no_cve(self):
        dep = _minimal_dep(
            file_name="@tootallnate/once:1.1.2",
            purl="pkg:npm/%40tootallnate%2Fonce@1.1.2",
            cve_name="GHSA-vpq2-c234-7xj6",
            cwes=["CWE-705"],
        )
        issues = parse_vulnerabilities({"dependencies": [dep]})
        assert len(issues) == 1
        issue = issues[0]
        assert issue.cve_id is None
        assert issue.ghsa_id == "GHSA-VPQ2-C234-7XJ6"
        assert issue.rule_id == "GHSA-vpq2-c234-7xj6"
        assert issue.package_name == "@tootallnate/once"

    def test_ghsa_and_cve_are_both_preserved(self):
        dep = _minimal_dep(
            file_name="once:1.1.2",
            purl="pkg:npm/once@1.1.2",
            cve_name="GHSA-vpq2-c234-7xj6",
        )
        dep["vulnerabilities"][0]["references"] = [
            {"url": "https://nvd.nist.gov/vuln/detail/CVE-2020-15084"}
        ]
        issues = parse_vulnerabilities({"dependencies": [dep]})
        assert len(issues) == 1
        issue = issues[0]
        assert issue.cve_id == "CVE-2020-15084"
        assert issue.ghsa_id == "GHSA-VPQ2-C234-7XJ6"

    def test_scoped_package_purl(self):
        dep = _minimal_dep(
            file_name="@tootallnate/once:1.1.2",
            purl="pkg:npm/%40tootallnate%2Fonce@1.1.2",
            cve_name="CVE-2021-99999",
        )
        issues = parse_vulnerabilities({"dependencies": [dep]})
        assert issues[0].package_name == "@tootallnate/once"
        assert issues[0].package_version == "1.1.2"

    def test_repo_metadata_propagated(self):
        report = {"dependencies": [_minimal_dep()]}
        issues = parse_vulnerabilities(
            report,
            repo_url="https://github.com/juice-shop/juice-shop",
            base_ref="v15.0.0",
        )
        assert issues[0].repo_url == "https://github.com/juice-shop/juice-shop"
        assert issues[0].base_ref == "v15.0.0"

    def test_multiple_vulns_per_dep(self):
        dep = {
            "fileName": "file-type:11.1.0",
            "filePath": "/src/package-lock.json?/file-type:11.1.0",
            "packages": [{"id": "pkg:npm/file-type@11.1.0"}],
            "vulnerabilities": [
                {"name": "CVE-2022-0001", "severity": "HIGH", "cwes": []},
                {"name": "CVE-2022-0002", "severity": "MEDIUM", "cwes": []},
            ],
        }
        issues = parse_vulnerabilities({"dependencies": [dep]})
        assert len(issues) == 2
        assert {i.cve_id for i in issues} == {"CVE-2022-0001", "CVE-2022-0002"}

    def test_empty_dependencies(self):
        assert parse_vulnerabilities({"dependencies": []}) == []

    def test_no_vulnerabilities_skipped(self):
        dep = {"fileName": "safe-pkg-1.0.0.tgz", "vulnerabilities": []}
        assert parse_vulnerabilities({"dependencies": [dep]}) == []

    def test_bench_js_javascript_ecosystem(self):
        """bench.js is flagged as pkg:javascript/underscore.js in real report."""
        dep = {
            "fileName": "bench.js",
            "filePath": "/src/node_modules/fast.js/dist/bench.js",
            "packages": [{"id": "pkg:javascript/underscore.js@1.7.0"}],
            "vulnerabilities": [{"name": "CVE-2026-27601", "severity": "HIGH", "cwes": []}],
        }
        issues = parse_vulnerabilities({"dependencies": [dep]})
        assert len(issues) == 1
        assert issues[0].ecosystem == "javascript"
        assert issues[0].package_name == "underscore.js"


# ===========================================================================
# odc_parser â€” export functions
# ===========================================================================


class TestExportFunctions:
    def test_jsonl_round_trip(self, tmp_path):
        report = {
            "dependencies": [
                _minimal_dep(),
                _minimal_dep(
                    file_name="express:4.18.0",
                    purl="pkg:npm/express@4.18.0",
                    cve_name="CVE-2022-9999",
                ),
            ]
        }
        issues = parse_vulnerabilities(report)
        out = tmp_path / "out.jsonl"
        export_to_jsonl(issues, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        restored = [VulnerabilityIssue.model_validate_json(line) for line in lines]
        assert restored[0].source == IssueSource.ODC
        assert {r.cve_id for r in restored} == {"CVE-2021-23337", "CVE-2022-9999"}

    def test_csv_written(self, tmp_path):
        report = {"dependencies": [_minimal_dep()]}
        issues = parse_vulnerabilities(report)
        out = tmp_path / "out.csv"
        export_to_csv(issues, out)
        content = out.read_text()
        assert "CVE-2021-23337" in content
        assert "CWE-77" in content
        assert "npm" in content

    def test_creates_parent_dirs(self, tmp_path):
        report = {"dependencies": [_minimal_dep()]}
        issues = parse_vulnerabilities(report)
        deep = tmp_path / "a" / "b" / "c" / "out.jsonl"
        export_to_jsonl(issues, deep)
        assert deep.exists()


# ===========================================================================
# manifest_locator â€” normalize_package_name
# ===========================================================================


class TestNormalizePackageName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("lodash-4.17.21.tgz", "lodash"),
            ("lodash-4.17.20.tgz", "lodash"),
            ("log4j-core-2.17.2.jar", "log4j-core"),
            ("@tootallnate/once:1.1.2", "@tootallnate/once"),
            ("@scope/pkg:2.0.0", "@scope/pkg"),
            ("cookie:0.4.2", "cookie"),
            ("base64url:0.0.6", "base64url"),
            ("bench.js", "bench"),  # .js suffix stripped
            ("pkg:npm/lodash@4.17.20", "lodash"),
            ("pkg:npm/%40tootallnate%2Fonce@1.1.2", "@tootallnate/once"),
            ("  ", ""),  # whitespace only
            ("", ""),
        ],
    )
    def test_normalise(self, raw, expected):
        assert normalize_package_name(raw) == expected


# ===========================================================================
# manifest_locator â€” parse_lockfile_path
# ===========================================================================


class TestParseLockfilePath:
    def test_dependency_ancestry_preserves_scoped_names_and_versions(self):
        ancestry = parse_dependency_ancestry(
            "/src/package-lock.json?sqlite3:5.1.7/http-proxy-agent:4.0.1/@tootallnate/once:1.1.2"
        )
        assert ancestry == [
            ("sqlite3", "5.1.7"),
            ("http-proxy-agent", "4.0.1"),
            ("@tootallnate/once", "1.1.2"),
        ]

    def test_root_npm_entry(self):
        lockfile, parent, leaf = parse_lockfile_path("/src/package-lock.json?/lodash:2.4.2")
        assert lockfile == "/src/package-lock.json"
        assert parent is None
        assert leaf == "lodash"

    def test_root_npm_entry_no_leading_slash_after_question(self):
        lockfile, parent, leaf = parse_lockfile_path("/src/package-lock.json?/cookie:0.4.2")
        assert lockfile == "/src/package-lock.json"
        assert leaf == "cookie"

    def test_transitive_chain(self):
        lockfile, parent, leaf = parse_lockfile_path(
            "/src/package-lock.json?jws:0.2.6/base64url:0.0.6"
        )
        assert lockfile == "/src/package-lock.json"
        assert parent == "jws"
        assert leaf == "base64url"

    def test_scoped_package_entry(self):
        lockfile, parent, leaf = parse_lockfile_path(
            "/src/package-lock.json?/@tootallnate/once:1.1.2"
        )
        assert lockfile == "/src/package-lock.json"
        assert leaf == "@tootallnate/once"

    def test_frontend_lockfile(self):
        lockfile, parent, leaf = parse_lockfile_path(
            "/src/frontend/package-lock.json?/elliptic:6.6.1"
        )
        assert "frontend" in lockfile
        assert leaf == "elliptic"

    def test_node_modules_file_path(self):
        lockfile, parent, leaf = parse_lockfile_path(
            "/src/node_modules/express-jwt/node_modules/moment/moment.js"
        )
        assert lockfile is None
        assert parent == "express-jwt"
        assert leaf == "moment"

    def test_non_lockfile_bench_js(self):
        """bench.js in node_modules should still extract the containing package."""
        lockfile, parent, leaf = parse_lockfile_path("/src/node_modules/fast.js/dist/bench.js")
        assert lockfile is None
        # leaf should resolve to fast.js (the node_module containing it)
        # parent is None as there's only one node_modules level
        assert leaf == "fast.js"

    def test_plain_path_no_match(self):
        lockfile, parent, leaf = parse_lockfile_path("/src/app.ts")
        assert lockfile is None
        assert leaf is None


# ===========================================================================
# manifest_locator â€” detect_package_manager
# ===========================================================================


class TestDetectPackageManager:
    def test_npm(self, tmp_repo):
        assert detect_package_manager(tmp_repo) == PackageManagerKind.NPM

    def test_yarn(self, tmp_repo_yarn):
        assert detect_package_manager(tmp_repo_yarn) == PackageManagerKind.YARN

    def test_pnpm(self, tmp_repo_pnpm):
        assert detect_package_manager(tmp_repo_pnpm) == PackageManagerKind.PNPM

    def test_pnpm_takes_priority_over_yarn(self, tmp_path):
        """pnpm-lock.yaml wins even if yarn.lock also present."""
        (tmp_path / "pnpm-lock.yaml").touch()
        (tmp_path / "yarn.lock").touch()
        (tmp_path / "package.json").write_text("{}")
        assert detect_package_manager(tmp_path) == PackageManagerKind.PNPM


# ===========================================================================
# manifest_locator â€” _locate_in_manifest
# ===========================================================================


class TestLocateInManifest:
    def test_direct_dep_line_found(self, tmp_repo):
        result = _locate_in_manifest(tmp_repo / "package.json", "lodash", "npm")
        assert result["is_direct"] is True
        assert result["line_number"] is not None
        assert "lodash" in result["snippet"]
        # No fix_instruction in result
        assert "fix_instruction" not in result

    def test_direct_dep_express(self, tmp_repo):
        result = _locate_in_manifest(tmp_repo / "package.json", "express", "npm")
        assert result["is_direct"] is True
        assert result["line_number"] is not None

    def test_dev_dep_mocha(self, tmp_repo):
        result = _locate_in_manifest(tmp_repo / "package.json", "mocha", "npm")
        assert result["is_direct"] is True

    def test_transitive_dep_no_line(self, tmp_repo):
        """Transitive deps are not in direct deps â€” no line number returned."""
        result = _locate_in_manifest(tmp_repo / "package.json", "cookie", "npm")
        assert result["is_direct"] is False
        assert result["line_number"] is None
        assert result["snippet"] is None
        # No fix_instruction in result
        assert "fix_instruction" not in result

    def test_package_manager_propagated(self, tmp_repo):
        result = _locate_in_manifest(tmp_repo / "package.json", "lodash", "npm")
        assert result["package_manager"] == "npm"

    def test_package_manager_yarn(self, tmp_repo_yarn):
        result = _locate_in_manifest(tmp_repo_yarn / "package.json", "lodash", "yarn")
        assert result["package_manager"] == "yarn"

    def test_package_manager_pnpm(self, tmp_repo_pnpm):
        result = _locate_in_manifest(tmp_repo_pnpm / "package.json", "cookie", "pnpm")
        assert result["package_manager"] == "pnpm"

    def test_keys_returned(self, tmp_repo):
        """Exactly the right keys â€” no OSV/fix artefacts."""
        result = _locate_in_manifest(tmp_repo / "package.json", "lodash", "npm")
        expected_keys = {
            "manifest_file",
            "package_name",
            "is_direct",
            "line_number",
            "snippet",
            "package_manager",
        }
        assert expected_keys.issubset(result.keys())
        assert "fix_instruction" not in result
        assert "fixed_version" not in result


# ===========================================================================
# manifest_locator â€” _find_nearest_manifest
# ===========================================================================


class TestFindNearestManifest:
    def test_root_lockfile_path(self, tmp_repo):
        manifest = _find_nearest_manifest(tmp_repo, "/src/package-lock.json?/lodash:4.17.20")
        assert manifest is not None
        assert manifest.name == "package.json"

    def test_node_modules_path_resolves_root(self, tmp_repo):
        manifest = _find_nearest_manifest(tmp_repo, "/src/node_modules/lodash/package.json")
        assert manifest is not None

    def test_nested_frontend_prefers_frontend_manifest(self, tmp_repo_nested):
        """ODC path inside frontend/ should resolve to frontend/package.json."""
        manifest = _find_nearest_manifest(
            tmp_repo_nested,
            "/src/frontend/package-lock.json?/elliptic:6.6.1",
        )
        assert manifest is not None
        assert manifest.parent.name == "frontend"
        assert manifest.name == "package.json"

    def test_empty_path_returns_none(self, tmp_repo):
        result = _find_nearest_manifest(tmp_repo, "")
        # Empty path â€” falls back gracefully
        assert result is None or result.name == "package.json"


# ===========================================================================
# manifest_locator â€” locate_dependency (end-to-end, no OSV)
# ===========================================================================


class TestLocateDependency:
    def test_direct_dep_root(self, tmp_repo):
        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="lodash-4.17.20.tgz",
            odc_file_path="/src/package-lock.json?/lodash:4.17.20",
        )
        assert result["status"] == "success"
        assert result["package_name"] == "lodash"
        assert result["is_direct"] is True
        assert result["line_number"] is not None
        assert "lockfile_ancestry" in result
        # No OSV/fix artefacts
        assert "fix_instruction" not in result
        assert "fixed_version" not in result

    def test_transitive_dep(self, tmp_repo):
        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="cookie:0.4.2",
            odc_file_path="/src/package-lock.json?/cookie:0.4.2",
        )
        assert result["status"] == "success"
        assert result["is_direct"] is False
        assert result["line_number"] is None
        assert "fix_instruction" not in result

    def test_lockfile_ancestry_propagated(self, tmp_repo):
        """jws â†’ base64url transitive chain ancestry is preserved."""
        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="base64url:0.0.6",
            odc_file_path="/src/package-lock.json?jws:0.2.6/base64url:0.0.6",
        )
        assert result["status"] == "success"
        assert result["package_name"] == "base64url"
        ancestry = result["lockfile_ancestry"]
        assert ancestry["ancestor_pkg"] == "jws"
        assert ancestry["leaf_pkg"] == "base64url"

    def test_transitive_parent_is_nearest_direct_manifest_dependency(self, tmp_repo):
        package_json = json.loads((tmp_repo / "package.json").read_text())
        package_json["dependencies"]["sqlite3"] = "^5.1.7"
        (tmp_repo / "package.json").write_text(json.dumps(package_json, indent=2))

        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="@tootallnate/once:1.1.2",
            odc_file_path=(
                "/src/package-lock.json?sqlite3:5.1.7/http-proxy-agent:4.0.1/"
                "@tootallnate/once:1.1.2"
            ),
        )

        assert result["status"] == "success"
        assert result["dependency_versions"]["sqlite3"] == "5.1.7"
        assert result["parent_package_name"] == "sqlite3"
        assert result["parent_package_version"] == "5.1.7"
        assert result["parent_declaration_type"] == "dependencies"
        assert result["parent_manifest_line"] is not None
        assert '"sqlite3": "^5.1.7"' in result["parent_manifest_snippet"]

    def test_expands_compressed_odc_ancestry_from_npm_lockfile(self, tmp_repo):
        package_json = json.loads((tmp_repo / "package.json").read_text())
        package_json["dependencies"]["sqlite3"] = "^5.1.7"
        (tmp_repo / "package.json").write_text(json.dumps(package_json, indent=2))
        (tmp_repo / "package-lock.json").write_text(
            json.dumps(
                {
                    "lockfileVersion": 3,
                    "packages": {
                        "": {"dependencies": {"sqlite3": "^5.1.7"}},
                        "node_modules/sqlite3": {
                            "version": "5.1.7",
                            "optionalDependencies": {"node-gyp": "8.x"},
                        },
                        "node_modules/sqlite3/node_modules/node-gyp": {
                            "version": "8.4.1",
                            "dependencies": {"make-fetch-happen": "^9.1.0"},
                        },
                        "node_modules/sqlite3/node_modules/make-fetch-happen": {
                            "version": "9.1.0",
                            "dependencies": {"http-proxy-agent": "^4.0.1"},
                        },
                        "node_modules/sqlite3/node_modules/http-proxy-agent": {
                            "version": "4.0.1",
                            "dependencies": {"@tootallnate/once": "1"},
                        },
                        "node_modules/@tootallnate/once": {"version": "1.1.2"},
                    },
                }
            )
        )

        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="@tootallnate/once:1.1.2",
            odc_file_path=(
                "/src/package-lock.json?sqlite3:5.1.7/http-proxy-agent:4.0.1/"
                "@tootallnate/once:1.1.2"
            ),
        )

        assert result["dependency_ancestry"] == [
            "sqlite3",
            "node-gyp",
            "make-fetch-happen",
            "http-proxy-agent",
            "@tootallnate/once",
        ]
        assert result["dependency_versions"] == {
            "sqlite3": "5.1.7",
            "node-gyp": "8.4.1",
            "make-fetch-happen": "9.1.0",
            "http-proxy-agent": "4.0.1",
            "@tootallnate/once": "1.1.2",
        }
        assert result["parent_package_name"] == "sqlite3"

    def test_package_manager_detected(self, tmp_repo_yarn):
        result = locate_dependency(
            repo_path=tmp_repo_yarn,
            raw_dependency_name="cookie:0.4.2",
            odc_file_path="/src/package-lock.json?/cookie:0.4.2",
        )
        assert result["status"] == "success"
        assert result["package_manager"] == PackageManagerKind.YARN

    def test_package_manager_detected_pnpm(self, tmp_repo_pnpm):
        result = locate_dependency(
            repo_path=tmp_repo_pnpm,
            raw_dependency_name="cookie:0.4.2",
            odc_file_path="/src/package-lock.json?/cookie:0.4.2",
        )
        assert result["status"] == "success"
        assert result["package_manager"] == PackageManagerKind.PNPM

    def test_empty_package_name(self, tmp_repo):
        result = locate_dependency(
            repo_path=tmp_repo,
            raw_dependency_name="  ",
        )
        assert result["status"] == "error"
        assert "empty" in result["message"].lower()

    def test_no_package_json(self, tmp_path):
        result = locate_dependency(
            repo_path=tmp_path,
            raw_dependency_name="lodash",
        )
        assert result["status"] == "error"

    def test_scoped_package_entry(self, tmp_path):
        """@tootallnate/once:1.1.2 root-level lockfile path."""
        pkg = {
            "name": "test",
            "version": "1.0.0",
            "dependencies": {},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))
        (tmp_path / "package-lock.json").touch()
        result = locate_dependency(
            repo_path=tmp_path,
            raw_dependency_name="@tootallnate/once:1.1.2",
            odc_file_path="/src/package-lock.json?/@tootallnate/once:1.1.2",
        )
        assert result["status"] == "success"
        assert result["package_name"] == "@tootallnate/once"
        assert result["is_direct"] is False  # not in deps â†’ transitive

    def test_no_extra_parameters_accepted(self):
        """Verify that old OSV parameters are gone from the signature."""
        import inspect

        sig = inspect.signature(locate_dependency)
        param_names = set(sig.parameters.keys())
        # These parameters must NOT exist any more
        assert "purl" not in param_names
        assert "package_version" not in param_names
        assert "known_fixed_version" not in param_names
        assert "enrich_osv" not in param_names
        # These must still be present
        assert "repo_path" in param_names
        assert "raw_dependency_name" in param_names
        assert "odc_file_path" in param_names


# ===========================================================================
# manifest_locator â€” locate_from_issue
# ===========================================================================


class TestLocateFromIssue:
    def _make_issue(self, **kwargs) -> VulnerabilityIssue:
        defaults = dict(
            source=IssueSource.ODC,
            issue_type=IssueType.SCA,
            severity=Severity.HIGH,
            package_name="lodash",
            package_version="4.17.20",
            purl="pkg:npm/lodash@4.17.20",
            cve_id="CVE-2021-23337",
            file_path="/src/package-lock.json?/lodash:4.17.20",
            raw_payload={"filePath": "/src/package-lock.json?/lodash:4.17.20"},
        )
        defaults.update(kwargs)
        return VulnerabilityIssue(**defaults)

    def test_returns_localized_issue(self, tmp_repo):
        from remediation_engine.contracts import LocalizedIssue

        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo)
        assert isinstance(loc, LocalizedIssue)
        assert loc.issue.id == issue.id
        assert loc.manifest_file is not None
        assert loc.is_direct_dependency is True
        assert loc.manifest_line is not None
        assert loc.localization_confidence > 0.5

    def test_high_confidence_for_direct_with_line(self, tmp_repo):
        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo)
        assert loc.localization_confidence >= 0.9

    def test_low_confidence_for_transitive(self, tmp_repo):
        issue = self._make_issue(
            package_name="cookie",
            purl="pkg:npm/cookie@0.4.2",
            file_path="/src/package-lock.json?/cookie:0.4.2",
            raw_payload={"filePath": "/src/package-lock.json?/cookie:0.4.2"},
        )
        loc = locate_from_issue(issue, tmp_repo)
        assert loc.is_direct_dependency is False
        assert loc.localization_confidence < 0.8

    def test_no_fix_instruction_field(self, tmp_repo):
        """LocalizedIssue must not have a fix_instruction field at all."""

        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo)
        assert not hasattr(loc, "fix_instruction")

    def test_no_osv_parameter_on_locate_from_issue(self):
        """enrich_osv parameter must be gone from locate_from_issue."""
        import inspect

        sig = inspect.signature(locate_from_issue)
        assert "enrich_osv" not in sig.parameters

    def test_issue_fixed_version_not_mutated(self, tmp_repo):
        """locate_from_issue must NOT mutate issue.fixed_version."""
        issue = self._make_issue()
        assert issue.fixed_version is None
        locate_from_issue(issue, tmp_repo)
        # fixed_version should remain None â€” no OSV enrichment happens
        assert issue.fixed_version is None

    def test_package_manager_in_localized_issue(self, tmp_repo):
        """package_manager field should be populated on LocalizedIssue."""
        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo)
        assert loc.package_manager == PackageManagerKind.NPM

    def test_package_manager_yarn(self, tmp_repo_yarn):
        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo_yarn)
        assert loc.package_manager == PackageManagerKind.YARN

    def test_json_round_trip(self, tmp_repo):
        from remediation_engine.contracts import LocalizedIssue

        issue = self._make_issue()
        loc = locate_from_issue(issue, tmp_repo)
        reloaded = LocalizedIssue.model_validate_json(loc.model_dump_json())
        assert reloaded.issue.id == issue.id
        assert reloaded.manifest_file == loc.manifest_file
        assert reloaded.package_manager == loc.package_manager


# ===========================================================================
# Integration: full pipeline with the sample ODC report
# ===========================================================================


class TestSampleODCReport:
    """Parse the bundled sample ODC report and spot-check results."""

    @pytest.fixture()
    def sample_report(self) -> dict:
        path = Path(__file__).resolve().parent.parent / "data" / "sample_odc_report.json"
        return json.loads(path.read_text())

    def test_parses_lodash(self, sample_report):
        issues = parse_vulnerabilities(sample_report)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.package_name == "lodash"
        assert issue.package_version == "4.17.20"
        assert issue.cve_id == "CVE-2021-23337"
        assert issue.severity == Severity.HIGH
        assert issue.ecosystem == "npm"
        assert issue.purl == "pkg:npm/lodash@4.17.20"
        assert any(c.id == "CWE-77" for c in issue.cwe)

    def test_jsonl_and_csv_both_written(self, sample_report, tmp_path):
        issues = parse_vulnerabilities(sample_report)
        export_to_jsonl(issues, tmp_path / "out.jsonl")
        export_to_csv(issues, tmp_path / "out.csv")
        assert (tmp_path / "out.jsonl").exists()
        assert (tmp_path / "out.csv").exists()
        # JSONL has exactly one line per issue
        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert len(lines) == len(issues)
