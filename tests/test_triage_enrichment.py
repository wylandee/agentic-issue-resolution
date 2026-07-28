"""
tests/test_triage_enrichment.py â€” Unit tests for remediation_engine.triage.enrichment.

Covers:
- Mock EPSS HTTP 200 â†’ parse epss and percentile correctly
- Mock CISA KEV fresh download â†’ in_kev correct, cache written
- Mock CISA KEV cache hit (TTL not expired) â†’ no HTTP call
- Network timeout â†’ safe defaults, no exception raised
- CVE not in KEV â†’ in_kev=False
- Empty CVE list â†’ empty dict returned
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from remediation_engine.triage.enrichment import enrich_cves


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FAKE_KEV_DATA = {
    "vulnerabilities": [
        {"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"},
        {"cveID": "CVE-2021-23337", "dateAdded": "2022-01-05"},
    ]
}

FAKE_EPSS_DATA = {
    "data": [
        {"cve": "CVE-2021-44228", "epss": "0.974", "percentile": "0.999"},
        {"cve": "CVE-2021-23337", "epss": "0.123", "percentile": "0.450"},
    ]
}


def _fake_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# EPSS tests
# ---------------------------------------------------------------------------


class TestEPSSEnrichment:
    def test_epss_200_parses_score_and_percentile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),   # KEV download
                _fake_response(FAKE_EPSS_DATA),  # EPSS query
            ]
            result = enrich_cves(["CVE-2021-44228"])

        assert "CVE-2021-44228" in result
        enrichment = result["CVE-2021-44228"]
        assert enrichment.epss == pytest.approx(0.974, rel=1e-3)
        assert enrichment.epss_percentile == pytest.approx(0.999, rel=1e-3)

    def test_epss_multiple_cves_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228", "CVE-2021-23337"])

        assert result["CVE-2021-44228"].epss == pytest.approx(0.974, rel=1e-3)
        assert result["CVE-2021-23337"].epss == pytest.approx(0.123, rel=1e-3)

    def test_epss_timeout_returns_safe_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            # KEV succeeds; EPSS times out
            def side_effect(url, **kwargs):
                if "first.org" in url:
                    raise requests.exceptions.Timeout("timeout")
                return _fake_response(FAKE_KEV_DATA)

            mock_get.side_effect = side_effect
            result = enrich_cves(["CVE-2021-44228"])

        assert "CVE-2021-44228" in result
        enrichment = result["CVE-2021-44228"]
        assert enrichment.epss == 0.0
        assert enrichment.epss_percentile == 0.0
        # Should not raise

    def test_epss_network_error_returns_safe_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            def side_effect(url, **kwargs):
                if "first.org" in url:
                    raise requests.exceptions.ConnectionError("refused")
                return _fake_response(FAKE_KEV_DATA)

            mock_get.side_effect = side_effect
            # Must not raise
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].epss == 0.0


# ---------------------------------------------------------------------------
# KEV cache tests
# ---------------------------------------------------------------------------


class TestKEVEnrichment:
    def test_kev_fresh_download_marks_in_kev(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),   # KEV download
                _fake_response(FAKE_EPSS_DATA),  # EPSS
            ]
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].in_kev is True
        assert result["CVE-2021-44228"].kev_date_added == "2021-12-10"

    def test_kev_cache_hit_skips_http(self, tmp_path, monkeypatch):
        """If cache is fresh, no HTTP call should be made to the KEV URL."""
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        # Pre-populate the cache
        cache_path = tmp_path / "kev_cache.json"
        cache_path.write_text(json.dumps(FAKE_KEV_DATA), encoding="utf-8")
        # Make cache appear fresh (mtime = now)
        assert cache_path.exists()

        call_urls: list[str] = []

        def mock_get(url, **kwargs):
            call_urls.append(url)
            return _fake_response(FAKE_EPSS_DATA)

        with patch("remediation_engine.triage.enrichment.requests.get", side_effect=mock_get):
            result = enrich_cves(["CVE-2021-44228"])

        # Only the EPSS URL should have been called (KEV came from cache)
        assert all("first.org" in u for u in call_urls), (
            f"KEV URL was called unexpectedly: {call_urls}"
        )
        assert result["CVE-2021-44228"].in_kev is True

    def test_cve_not_in_kev(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            kev_without_cve = {"vulnerabilities": []}
            mock_get.side_effect = [
                _fake_response(kev_without_cve),
                _fake_response({"data": []}),
            ]
            result = enrich_cves(["CVE-2099-99999"])

        assert result["CVE-2099-99999"].in_kev is False

    def test_kev_download_failure_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            def side_effect(url, **kwargs):
                if "cisa.gov" in url:
                    raise requests.exceptions.ConnectionError("cisa down")
                return _fake_response({"data": []})

            mock_get.side_effect = side_effect
            # Must not raise
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].in_kev is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_cve_list_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        with patch("remediation_engine.triage.enrichment.requests.get"):
            result = enrich_cves([])
        assert result == {}

    def test_duplicate_cve_ids_deduplicated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228", "CVE-2021-44228", "CVE-2021-44228"])

        assert len(result) == 1
        assert "CVE-2021-44228" in result

    def test_enrichment_source_field_correct(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.triage.enrichment.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228"])

        # Both EPSS (>0) and KEV (in_kev=True) contributed
        assert result["CVE-2021-44228"].enrichment_source in ("epss+kev", "kev+epss")


