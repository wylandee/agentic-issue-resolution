"""
Tests for remediation_engine.tools.fix_planner.

All network calls are mocked â€” no live internet required.

Coverage
--------
plan_fix waterfall
  - Step 1: local regex extraction from ODC-style advisory messages
  - Step 2: OSV querybatch with embedded ranges (full detail in result)
  - Step 2b: OSV querybatch returns only vuln IDs â†’ GET /v1/vulns/{id} detail
  - Step 3: npm registry latest dist-tag (plain + scoped packages)
  - Step 4: Serper workaround snippets (key present / key absent)
  - Step 5: no_fix when all strategies fail

_build_instruction
  - Direct dependency (npm/yarn/pnpm)
  - Transitive dependency â€” npm overrides
  - Transitive dependency â€” Yarn resolutions
  - Transitive dependency â€” pnpm overrides
  - Missing manifest_file falls back to "package.json"

Helper unit tests
  - _extract_local_version: various message forms, None input
  - _is_npm_issue: npm / javascript ecosystem, purl matching
  - _package_name_from_issue: direct name, PURL fallback, scoped packages
  - _fetch_osv_vuln_detail: success and failure
  - _fetch_npm_latest: success, scoped packages (@scope/pkg), failure

FixPlan round-trip
  - Constructed from plan_fix dict validates as FixPlan
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.tools.fix_planner import (
    SerperLLMResult,
    _build_instruction,
    _extract_fixed_from_osv_vuln,
    _extract_local_version,
    _fetch_npm_latest,
    _fetch_osv_vuln_detail,
    _fetch_page_content,
    _is_npm_issue,
    _llm_extract_remediation,
    _minimum_fixed_version,
    _package_name_from_issue,
    _query_osv_fixed_version,
    _query_serper,
    plan_fix,
)

# ===========================================================================
# Shared factories
# ===========================================================================


def _make_vuln_issue(**kwargs) -> VulnerabilityIssue:
    defaults: dict[str, Any] = dict(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="lodash",
        package_version="4.17.20",
        purl="pkg:npm/lodash@4.17.20",
        cve_id="CVE-2021-23337",
        ecosystem="npm",
        message="Prototype pollution. Update to version 4.17.21 or later.",
        file_path="/src/package-lock.json?/lodash:4.17.20",
    )
    defaults.update(kwargs)
    return VulnerabilityIssue(**defaults)


def _make_localized(issue: VulnerabilityIssue | None = None, **kwargs) -> LocalizedIssue:
    if issue is None:
        issue = _make_vuln_issue()
    defaults: dict[str, Any] = dict(
        issue=issue,
        manifest_file="package.json",
        is_direct_dependency=True,
        manifest_line=42,
        package_manager="npm",
        localization_confidence=0.95,
    )
    defaults.update(kwargs)
    return LocalizedIssue(**defaults)


# ===========================================================================
# _extract_local_version
# ===========================================================================


class TestExtractLocalVersion:
    @pytest.mark.parametrize(
        "msg,expected",
        [
            ("Update to version 3.0.0 or later.", "3.0.0"),
            ("Upgrade to 1.2.3.", "1.2.3"),
            ("Fixed in v2.1.4.", "2.1.4"),
            ("Patched in 5.0.0-beta.1.", "5.0.0"),
            ("Use version 0.0.1.", "0.0.1"),
            ("Requires 2.0.0 or higher.", "2.0.0"),
            ("No version mentioned here.", None),
            ("", None),
        ],
    )
    def test_variants(self, msg, expected):
        assert _extract_local_version(msg) == expected

    def test_none_input(self):
        assert _extract_local_version(None) is None

    def test_case_insensitive(self):
        assert _extract_local_version("UPDATE TO VERSION 1.2.3") == "1.2.3"

    def test_takes_first_version(self):
        # Should return the first match only
        result = _extract_local_version("Update to version 4.17.21 or 4.18.0.")
        assert result == "4.17.21"


# ===========================================================================
# _is_npm_issue
# ===========================================================================


class TestIsNpmIssue:
    def test_npm_ecosystem(self):
        issue = _make_vuln_issue(ecosystem="npm")
        assert _is_npm_issue(issue) is True

    def test_javascript_ecosystem(self):
        issue = _make_vuln_issue(ecosystem="javascript")
        assert _is_npm_issue(issue) is True

    def test_purl_npm(self):
        issue = _make_vuln_issue(ecosystem=None, purl="pkg:npm/lodash@4.17.20")
        assert _is_npm_issue(issue) is True

    def test_purl_javascript(self):
        issue = _make_vuln_issue(ecosystem=None, purl="pkg:javascript/underscore.js@1.7.0")
        assert _is_npm_issue(issue) is True

    def test_maven_is_not_npm(self):
        issue = _make_vuln_issue(
            ecosystem="maven",
            purl="pkg:maven/commons-io/commons-io@2.4",
        )
        assert _is_npm_issue(issue) is False


# ===========================================================================
# _package_name_from_issue
# ===========================================================================


class TestPackageNameFromIssue:
    def test_uses_package_name(self):
        issue = _make_vuln_issue(package_name="lodash")
        assert _package_name_from_issue(issue) == "lodash"

    def test_falls_back_to_purl(self):
        issue = _make_vuln_issue(package_name=None, purl="pkg:npm/express@4.18.0")
        assert _package_name_from_issue(issue) == "express"

    def test_scoped_package_from_purl(self):
        issue = _make_vuln_issue(
            package_name=None,
            purl="pkg:npm/%40tootallnate%2Fonce@1.1.2",
        )
        name = _package_name_from_issue(issue)
        assert "@tootallnate/once" in name

    def test_empty_when_no_data(self):
        issue = _make_vuln_issue(package_name=None, purl=None)
        assert _package_name_from_issue(issue) == ""


# ===========================================================================
# _build_instruction
# ===========================================================================


class TestBuildInstruction:
    def test_direct_npm(self):
        instr = _build_instruction("lodash", "4.17.21", "npm", True, "package.json")
        assert "lodash" in instr
        assert "4.17.21" in instr
        assert "package.json" in instr
        assert "Update" in instr

    def test_transitive_npm_overrides(self):
        instr = _build_instruction("cookie", "0.7.0", "npm", False, "package.json")
        assert "overrides" in instr
        assert "cookie" in instr
        assert "0.7.0" in instr

    def test_transitive_yarn_resolutions(self):
        instr = _build_instruction("lodash", "4.17.21", "yarn", False, "package.json")
        assert "resolutions" in instr
        assert "lodash" in instr

    def test_transitive_pnpm_overrides(self):
        instr = _build_instruction("lodash", "4.17.21", "pnpm", False, "package.json")
        assert "pnpm" in instr
        assert "overrides" in instr

    def test_missing_manifest_falls_back(self):
        instr = _build_instruction("lodash", "4.17.21", "npm", True, None)
        assert "package.json" in instr

    def test_nested_manifest_basename_used(self):
        instr = _build_instruction("lodash", "4.17.21", "npm", True, "frontend/package.json")
        assert "package.json" in instr
        assert "frontend" not in instr  # only basename


# ===========================================================================
# _extract_fixed_from_osv_vuln
# ===========================================================================


class TestExtractFixedFromOsvVuln:
    def _make_vuln(self, rng_type: str, fixed: str, pkg_name: str = "lodash") -> dict:
        return {
            "id": "GHSA-xxxx",
            "affected": [
                {
                    "package": {"name": pkg_name, "ecosystem": "npm"},
                    "ranges": [
                        {
                            "type": rng_type,
                            "events": [
                                {"introduced": "0"},
                                {"fixed": fixed},
                            ],
                        }
                    ],
                }
            ],
        }

    def test_semver_range(self):
        vuln = self._make_vuln("SEMVER", "4.17.21")
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == ("4.17.21", None)

    def test_ecosystem_range(self):
        vuln = self._make_vuln("ECOSYSTEM", "4.17.21")
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == ("4.17.21", None)

    def test_git_range_ignored(self):
        vuln = self._make_vuln("GIT", "abc123")
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == (None, None)

    def test_semver_preferred_over_ecosystem(self):
        vuln = {
            "affected": [
                {
                    "package": {"name": "lodash"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"fixed": "4.17.0"}]},
                        {"type": "SEMVER", "events": [{"fixed": "4.17.21"}]},
                    ],
                }
            ]
        }
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == ("4.17.21", None)

    def test_package_name_mismatch_skipped(self):
        vuln = self._make_vuln("SEMVER", "4.17.21", pkg_name="other-pkg")
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == (None, None)

    def test_falls_back_to_workaround_text_case_insensitive(self):
        vuln = {
            "details": "Workaround: disable the vulnerable feature until a patch ships.",
            "affected": [],
        }
        assert _extract_fixed_from_osv_vuln(vuln, "lodash") == (
            None,
            ["Workaround: disable the vulnerable feature until a patch ships."],
        )

    def test_empty_vuln(self):
        assert _extract_fixed_from_osv_vuln({}, "lodash") == (None, None)


# ===========================================================================
# _fetch_osv_vuln_detail
# ===========================================================================


class TestFetchOsvVulnDetail:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "GHSA-xxxx", "affected": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_osv_vuln_detail("GHSA-xxxx")
        assert result is not None
        assert result["id"] == "GHSA-xxxx"

    def test_network_failure_returns_none(self):
        with patch(
            "remediation_engine.tools.fix_planner.requests.get", side_effect=Exception("timeout")
        ):
            result = _fetch_osv_vuln_detail("GHSA-xxxx")
        assert result is None

    def test_http_error_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_osv_vuln_detail("GHSA-xxxx")
        assert result is None


# ===========================================================================
# _fetch_npm_latest
# ===========================================================================


class TestFetchNpmLatest:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"version": "4.17.21", "name": "lodash"}
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_npm_latest("lodash")
        assert result == "4.17.21"

    def test_scoped_package_encoded_in_url(self):
        """@scope/pkg must be percent-encoded in the registry URL."""
        captured = {}

        def fake_get(url, timeout):
            captured["url"] = url
            mock = MagicMock()
            mock.json.return_value = {"version": "1.2.0"}
            mock.raise_for_status = MagicMock()
            return mock

        with patch("remediation_engine.tools.fix_planner.requests.get", side_effect=fake_get):
            result = _fetch_npm_latest("@tootallnate/once")
        assert result == "1.2.0"
        # @ and / must be encoded
        assert "%40" in captured["url"] or "%2F" in captured["url"]

    def test_network_failure_returns_none(self):
        with patch(
            "remediation_engine.tools.fix_planner.requests.get", side_effect=Exception("DNS")
        ):
            result = _fetch_npm_latest("lodash")
        assert result is None

    def test_missing_version_key_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "lodash"}  # no "version" key
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_npm_latest("lodash")
        assert result is None


# ===========================================================================
# _query_osv_fixed_version
# ===========================================================================


class TestQueryOsvFixedVersion:
    def _post_mock(self, body: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_embedded_ranges_extracted(self):
        """querybatch returns full vuln with ranges inline."""
        body = {
            "results": [
                {
                    "vulns": [
                        {
                            "id": "GHSA-xxxx",
                            "aliases": ["CVE-2021-23337"],
                            "affected": [
                                {
                                    "package": {"name": "lodash"},
                                    "ranges": [
                                        {
                                            "type": "SEMVER",
                                            "events": [
                                                {"introduced": "0"},
                                                {"fixed": "4.17.21"},
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        with patch(
            "remediation_engine.tools.fix_planner.requests.post", return_value=self._post_mock(body)
        ):
            issue = _make_vuln_issue()
            result = _query_osv_fixed_version(issue)
        assert result == ("4.17.21", None)

    def test_id_only_triggers_detail_fetch(self):
        """querybatch returns only {id: ...}, no 'affected' â†’ fall back to GET."""
        querybatch_body = {"results": [{"vulns": [{"id": "GHSA-xxxx"}]}]}
        detail_body = {
            "id": "GHSA-xxxx",
            "affected": [
                {
                    "package": {"name": "lodash"},
                    "ranges": [
                        {
                            "type": "SEMVER",
                            "events": [{"introduced": "0"}, {"fixed": "4.17.21"}],
                        }
                    ],
                }
            ],
        }
        post_mock = self._post_mock(querybatch_body)
        get_mock = MagicMock()
        get_mock.json.return_value = detail_body
        get_mock.raise_for_status = MagicMock()

        with (
            patch("remediation_engine.tools.fix_planner.requests.post", return_value=post_mock),
            patch("remediation_engine.tools.fix_planner.requests.get", return_value=get_mock),
        ):
            issue = _make_vuln_issue()
            result = _query_osv_fixed_version(issue)
        assert result == ("4.17.21", None)

    def test_workaround_text_returned_when_no_fixed_event(self):
        body = {
            "results": [
                {
                    "vulns": [
                        {
                            "id": "GHSA-xxxx",
                            "aliases": ["CVE-2021-23337"],
                            "details": "Mitigation: disable the parser while awaiting a fix.",
                            "affected": [
                                {
                                    "package": {"name": "lodash"},
                                    "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}]}],
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        with patch(
            "remediation_engine.tools.fix_planner.requests.post", return_value=self._post_mock(body)
        ):
            result = _query_osv_fixed_version(_make_vuln_issue())
        assert result == (
            None,
            ["Mitigation: disable the parser while awaiting a fix."],
        )

    def test_network_failure_returns_none(self):
        with patch(
            "remediation_engine.tools.fix_planner.requests.post", side_effect=Exception("timeout")
        ):
            result = _query_osv_fixed_version(_make_vuln_issue())
        assert result == (None, None)

    def test_empty_results_returns_none(self):
        body = {"results": [{}]}
        with patch(
            "remediation_engine.tools.fix_planner.requests.post", return_value=self._post_mock(body)
        ):
            result = _query_osv_fixed_version(_make_vuln_issue())
        assert result == (None, None)

    def test_no_package_name_returns_none(self):
        issue = _make_vuln_issue(package_name=None, purl=None)
        # No network calls should be made
        with patch("remediation_engine.tools.fix_planner.requests.post") as mock_post:
            result = _query_osv_fixed_version(issue)
        mock_post.assert_not_called()
        assert result == (None, None)


# ===========================================================================
# _query_serper
# ===========================================================================


class TestQuerySerper:
    def test_returns_results_dict(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        body = {
            "organic": [
                {
                    "link": "https://github.com/foo/bar/issues/1",
                    "snippet": "Workaround A",
                    "title": "Blog post",
                },
                {
                    "link": "https://github.com/foo/bar/issues/2",
                    "snippet": "Workaround B",
                    "title": "GitHub issue",
                },
                {
                    "link": "https://github.com/foo/bar/issues/3",
                    "snippet": "Workaround C",
                    "title": "StackOverflow",
                },
                {
                    "link": "https://github.com/foo/bar/issues/4",
                    "snippet": "Workaround D",
                    "title": "Extra",
                },  # capped at 3
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.post", return_value=mock_resp):
            result = _query_serper(_make_vuln_issue(), "lodash")
        assert result is not None
        assert len(result) == 3
        assert result[0]["url"] == "https://github.com/foo/bar/issues/1"
        assert result[0]["snippet"] == "Workaround A"
        assert result[0]["title"] == "Blog post"

    def test_returns_none_when_no_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        with patch("remediation_engine.tools.fix_planner.requests.post") as mock_post:
            result = _query_serper(_make_vuln_issue(), "lodash")
        mock_post.assert_not_called()
        assert result is None

    def test_network_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        with patch(
            "remediation_engine.tools.fix_planner.requests.post", side_effect=Exception("timeout")
        ):
            result = _query_serper(_make_vuln_issue(), "lodash")
        assert result is None

    def test_empty_organic_returns_none(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"organic": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.post", return_value=mock_resp):
            result = _query_serper(_make_vuln_issue(), "lodash")
        assert result is None


# ===========================================================================
# _fetch_page_content
# ===========================================================================


class TestFetchPageContent:
    def test_github_api_success(self):
        mock_resp = MagicMock()
        mock_resp.text = "GitHub advisory content"
        mock_resp.json.side_effect = Exception("Not JSON")
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_page_content("https://github.com/foo/bar/issues/1")
        assert result == "GitHub advisory content"

    def test_github_raw_fallback(self):
        def fake_get(url, **kwargs):
            mock = MagicMock()
            if "api.github.com" in url:
                mock.raise_for_status.side_effect = Exception("API error")
            else:
                mock.text = "Raw github file content"
                mock.raise_for_status = MagicMock()
            return mock

        with patch("remediation_engine.tools.fix_planner.requests.get", side_effect=fake_get):
            result = _fetch_page_content("https://github.com/foo/bar/blob/main/README.md")
        assert result == "Raw github file content"

    def test_jina_reader_fallback(self):
        def fake_get(url, **kwargs):
            mock = MagicMock()
            if "r.jina.ai" in url:
                mock.text = "Jina reader markdown"
                mock.raise_for_status = MagicMock()
            else:
                mock.raise_for_status.side_effect = Exception("Failed")
            return mock

        with patch("remediation_engine.tools.fix_planner.requests.get", side_effect=fake_get):
            result = _fetch_page_content("https://example.com/advisory")
        assert result == "Jina reader markdown"

    def test_direct_fetch_fallback(self):
        def fake_get(url, **kwargs):
            mock = MagicMock()
            if "r.jina.ai" in url:
                mock.raise_for_status.side_effect = Exception("Jina fail")
            else:
                mock.text = "Direct HTML content"
                mock.raise_for_status = MagicMock()
            return mock

        with patch("remediation_engine.tools.fix_planner.requests.get", side_effect=fake_get):
            result = _fetch_page_content("https://example.com/advisory")
        assert result == "Direct HTML content"

    def test_all_tiers_fail_returns_none(self):
        with patch(
            "remediation_engine.tools.fix_planner.requests.get",
            side_effect=Exception("network fail"),
        ):
            result = _fetch_page_content("https://example.com/advisory")
        assert result is None

    def test_truncates_long_content(self):
        mock_resp = MagicMock()
        mock_resp.text = "A" * 20_000
        mock_resp.raise_for_status = MagicMock()
        with patch("remediation_engine.tools.fix_planner.requests.get", return_value=mock_resp):
            result = _fetch_page_content("https://example.com/advisory")
        assert result is not None
        assert len(result) == 16_000


# ===========================================================================
# _llm_extract_remediation
# ===========================================================================


class TestLLMExtractRemediation:
    def test_version_bump_extraction(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = SerperLLMResult(
            strategy="VERSION_BUMP",
            fixed_version="4.17.21",
            reasoning="Upgrade recommended",
        )

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            pages = [{"url": "https://example.com", "content": "Update lodash to 4.17.21"}]
            result = _llm_extract_remediation(pages, "lodash", "CVE-2021-23337")

        assert result is not None
        assert result["strategy"] == "VERSION_BUMP"
        assert result["fixed_version"] == "4.17.21"

    def test_code_workaround_extraction(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = SerperLLMResult(
            strategy="CODE_WORKAROUND",
            workaround_snippets=["Disable feature X"],
            reasoning="Code mitigation",
        )

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            pages = [{"url": "https://example.com", "content": "Workaround: Disable feature X"}]
            result = _llm_extract_remediation(pages, "lodash", "CVE-2021-23337")

        assert result is not None
        assert result["strategy"] == "CODE_WORKAROUND"
        assert result["workaround_snippets"] == ["Disable feature X"]

    def test_no_fix_returns_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = SerperLLMResult(
            strategy="NO_FIX",
            reasoning="No patch available",
        )

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            pages = [{"url": "https://example.com", "content": "No fix yet."}]
            result = _llm_extract_remediation(pages, "lodash", "CVE-2021-23337")

        assert result is not None
        assert result["strategy"] == "NO_FIX"

    def test_llm_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("langchain_openai.ChatOpenAI", side_effect=Exception("API Error")):
            pages = [{"url": "https://example.com", "content": "some text"}]
            result = _llm_extract_remediation(pages, "lodash", "CVE-2021-23337")
        assert result is None

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        pages = [{"url": "https://example.com", "content": "some text"}]
        result = _llm_extract_remediation(pages, "lodash", "CVE-2021-23337")
        assert result is None


# ===========================================================================
# plan_fix — Serper LLM fallback integration
# ===========================================================================


class TestPlanFixSerperFallback:
    def test_osv_fails_serper_version_bump(self):
        loc = _make_localized()
        llm_dict = {"strategy": "VERSION_BUMP", "fixed_version": "4.17.21"}
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=llm_dict,
            ),
        ):
            result = plan_fix(loc)

        assert result["status"] == FixPlanStatus.VERSION_FOUND.value
        assert result["fixed_version"] == "4.17.21"
        assert result["strategy_used"] == "serper_llm"

    def test_osv_fails_serper_code_workaround(self):
        loc = _make_localized()
        llm_dict = {"strategy": "CODE_WORKAROUND", "workaround_snippets": ["Sanitize input"]}
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=llm_dict,
            ),
        ):
            result = plan_fix(loc)

        assert result["status"] == FixPlanStatus.WORKAROUND_FOUND.value
        assert result["workaround_snippets"] == ["Sanitize input"]
        assert result["strategy_used"] == "serper_llm"

    def test_osv_fails_serper_search_returns_none(self):
        loc = _make_localized()
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=None,
            ),
        ):
            result = plan_fix(loc)

        assert result["status"] == FixPlanStatus.NO_FIX.value
        assert result["strategy_used"] == "none"

    def test_osv_fails_serper_llm_returns_no_fix(self):
        loc = _make_localized()
        llm_dict = {"strategy": "NO_FIX"}
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=llm_dict,
            ),
        ):
            result = plan_fix(loc)

        assert result["status"] == FixPlanStatus.NO_FIX.value
        assert result["strategy_used"] == "none"


# ===========================================================================
# plan_fix — FixPlan round-trip
# ===========================================================================


class TestPlanFixRoundTrip:
    """Every plan_fix result must be constructable as a FixPlan."""

    def test_version_found_validates(self):
        loc = _make_localized()
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        assert fp.status == FixPlanStatus.VERSION_FOUND
        assert fp.fixed_version == "4.17.21"
        assert fp.strategy_used == "osv_api"

    def test_workaround_found_validates(self):
        issue = _make_vuln_issue(message="No hint.")
        loc = _make_localized(issue=issue)
        snippets = ["patch A"]
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=(None, snippets),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        assert fp.status == FixPlanStatus.WORKAROUND_FOUND
        assert fp.workaround_snippets == ["patch A"]
        assert fp.strategy_used == "osv_api"

    def test_serper_llm_version_round_trip(self):
        loc = _make_localized()
        llm_dict = {"strategy": "VERSION_BUMP", "fixed_version": "4.17.21"}
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=llm_dict,
            ),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        assert fp.status == FixPlanStatus.VERSION_FOUND
        assert fp.strategy_used == "serper_llm"
        assert fp.fixed_version == "4.17.21"

    def test_serper_llm_workaround_round_trip(self):
        loc = _make_localized()
        llm_dict = {"strategy": "CODE_WORKAROUND", "workaround_snippets": ["step 1"]}
        with (
            patch(
                "remediation_engine.tools.fix_planner._query_osv_fixed_version",
                return_value=(None, None),
            ),
            patch(
                "remediation_engine.tools.fix_planner._serper_search_and_extract",
                return_value=llm_dict,
            ),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        assert fp.status == FixPlanStatus.WORKAROUND_FOUND
        assert fp.strategy_used == "serper_llm"
        assert fp.workaround_snippets == ["step 1"]

    def test_no_fix_validates(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        issue = _make_vuln_issue(message="No hint.")
        loc = _make_localized(issue=issue)
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=(None, None),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        assert fp.status == FixPlanStatus.NO_FIX
        assert fp.strategy_used == "none"

    def test_json_round_trip(self):
        loc = _make_localized()
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        fp = FixPlan(**result)
        reloaded = FixPlan.model_validate_json(fp.model_dump_json())
        assert reloaded.status == fp.status
        assert reloaded.fixed_version == fp.fixed_version
        assert reloaded.instruction == fp.instruction


# ===========================================================================
# Instruction content correctness
# ===========================================================================


class TestPlanFixInstructionContent:
    def test_direct_dep_instruction_mentions_update(self):
        loc = _make_localized(is_direct_dependency=True, manifest_file="package.json")
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        assert "Update" in result["instruction"]
        assert "lodash" in result["instruction"]
        assert "4.17.21" in result["instruction"]

    def test_transitive_dep_npm_instruction_mentions_overrides(self):
        loc = _make_localized(
            is_direct_dependency=False,
            package_manager="npm",
            manifest_file="package.json",
        )
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        assert "overrides" in result["instruction"]

    def test_transitive_dep_yarn_instruction_mentions_resolutions(self):
        loc = _make_localized(
            is_direct_dependency=False,
            package_manager="yarn",
            manifest_file="package.json",
        )
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        assert "resolutions" in result["instruction"]

    def test_transitive_dep_pnpm_instruction_mentions_pnpm(self):
        loc = _make_localized(
            is_direct_dependency=False,
            package_manager="pnpm",
            manifest_file="package.json",
        )
        with patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ):
            result = plan_fix(loc)
        assert "pnpm" in result["instruction"]


# ===========================================================================
# _minimum_fixed_version & context-aware version selection
# ===========================================================================


class TestMinimumFixedVersion:
    def test_selects_lowest_when_no_current_version(self):
        assert _minimum_fixed_version(["6.0.6", "7.0.5"]) == "6.0.6"

    def test_selects_same_major_when_current_version_specified(self):
        assert _minimum_fixed_version(["6.0.6", "7.0.5"], current_version="7.0.0") == "7.0.5"
        assert _minimum_fixed_version(["6.0.6", "7.0.5"], current_version="6.0.0") == "6.0.6"

    def test_selects_higher_major_when_no_same_major_available(self):
        assert _minimum_fixed_version(["4.0.8"], current_version="3.1.10") == "4.0.8"

    def test_empty_versions_returns_none(self):
        assert _minimum_fixed_version([]) is None


# ===========================================================================
# OSV Alias and Database Specific Events Extraction
# ===========================================================================


class TestOsvAliasAndExtractedEvents:
    def test_extract_fixed_from_database_specific_events(self):
        vuln = {
            "id": "CVE-2024-4067",
            "affected": [
                {
                    "package": {"name": "micromatch"},
                    "ranges": [
                        {
                            "type": "GIT",
                            "database_specific": {
                                "extracted_events": [{"introduced": "0"}, {"fixed": "4.0.8"}]
                            },
                        }
                    ],
                }
            ],
        }
        fixed, snippets = _extract_fixed_from_osv_vuln(vuln, "micromatch")
        assert fixed == "4.0.8"
        assert snippets is None

    def test_query_osv_fixed_version_follows_aliases(self):
        issue = _make_vuln_issue(
            package_name="micromatch",
            package_version="3.1.10",
            cve_id="CVE-2024-4067",
            ghsa_id=None,
        )
        cve_detail = {
            "id": "CVE-2024-4067",
            "aliases": ["GHSA-952p-6rrq-rcjv"],
            "affected": [
                {
                    "package": {"name": "micromatch"},
                    "ranges": [{"type": "GIT"}],
                }
            ],
            "details": "This should be mitigated by using safe patterns.",
        }
        ghsa_detail = {
            "id": "GHSA-952p-6rrq-rcjv",
            "aliases": ["CVE-2024-4067"],
            "affected": [
                {
                    "package": {"name": "micromatch"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": "0"}, {"fixed": "4.0.8"}],
                        }
                    ],
                }
            ],
        }

        def mock_fetch(vuln_id):
            if vuln_id.upper() == "CVE-2024-4067":
                return cve_detail
            if vuln_id.upper() == "GHSA-952P-6RRQ-RCJV":
                return ghsa_detail
            return None

        with (
            patch("requests.post") as mock_post,
            patch(
                "remediation_engine.tools.fix_planner._fetch_osv_vuln_detail",
                side_effect=mock_fetch,
            ),
        ):
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "results": [{"vulns": [{"id": "GHSA-952p-6rrq-rcjv"}]}]
            }
            fixed, snippets = _query_osv_fixed_version(issue)
            assert fixed == "4.0.8"
            assert snippets is None
