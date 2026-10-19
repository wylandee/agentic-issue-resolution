from __future__ import annotations

from typing import Any

import pytest
import requests

from remediation_engine.tools.maven_registry_tools import fetch_maven_registry_candidates


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _XmlResponse:
    def __init__(self, content: str) -> None:
        self.content = content.encode()

    def raise_for_status(self) -> None:
        return None


def test_metadata_candidate_lookup_prefers_canonical_metadata_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_url = "https://repo.maven.apache.org/maven2/org/safe/widget/maven-metadata.xml"
    calls: list[dict[str, Any]] = []

    def fake_get(
        url: str,
        *,
        timeout: int,
        params: dict[str, Any] | None = None,
    ) -> _XmlResponse:
        calls.append({"url": url, "params": params, "timeout": timeout})
        assert params is None
        return _XmlResponse(
            """<metadata>
  <versioning>
    <latest>100.0</latest>
    <release>99.0</release>
    <lastUpdated>20261020123456</lastUpdated>
    <versions>
      <version>3.0</version>
      <version>1.0</version>
      <version>2.0</version>
      <version>2.1-rc1</version>
      <version>4.0-SNAPSHOT</version>
    </versions>
  </versioning>
</metadata>"""
        )

    monkeypatch.setattr("remediation_engine.tools.maven_registry_tools.requests.get", fake_get)
    candidates = fetch_maven_registry_candidates("org.safe:widget", "2.0", {"3.0"})

    assert [candidate.version for candidate in candidates] == ["2.0", "3.0"]
    assert candidates[0].selection_roles == ("maven_minimum",)
    assert candidates[1].selection_roles == ("maven_latest",)
    assert [candidate.already_attempted for candidate in candidates] == [False, True]
    assert all(candidate.security_floor_met and candidate.is_stable for candidate in candidates)
    assert len(calls) == 1
    assert calls[0]["url"] == metadata_url
    assert calls[0]["timeout"] > 0


def test_solr_candidate_lookup_uses_params_paginates_and_returns_safe_extremes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_url = "https://repo.maven.apache.org/maven2/org/safe/widget/maven-metadata.xml"
    calls: list[dict[str, Any]] = []

    def fake_get(
        url: str,
        *,
        timeout: int,
        params: dict[str, Any] | None = None,
    ) -> _Response:
        calls.append({"url": url, "params": params, "timeout": timeout})
        if url == metadata_url:
            raise requests.Timeout("repository metadata timed out")
        assert params is not None
        start = params["start"]
        versions = ["1.8", "2.0-SNAPSHOT", "2.0-rc1", "2.0", "1.9"]
        docs = [
            {"v": version, "timestamp": 100000 - index} for index, version in enumerate(versions)
        ]
        if start == 0:
            docs.extend({"v": "0.1"} for _ in range(195))
            return _Response({"response": {"numFound": 201, "docs": docs}})
        return _Response({"response": {"numFound": 201, "docs": [{"v": "3.0"}]}})

    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.time.sleep",
        lambda *_args: None,
    )
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.9", {"3.0"})
    solr_calls = [call for call in calls if call["params"] is not None]

    assert [candidate.version for candidate in candidates] == ["1.9", "3.0"]
    assert candidates[0].selection_roles == ("maven_minimum",)
    assert candidates[1].selection_roles == ("maven_latest",)
    assert candidates[0].already_attempted is False
    assert candidates[1].already_attempted is True
    assert [call["url"] for call in calls[:3]] == [metadata_url] * 3
    assert all(call["url"] == "https://search.maven.org/solrsearch/select" for call in solr_calls)
    assert solr_calls[0]["params"]["q"] == 'g:"org.safe" AND a:"widget"'
    assert solr_calls[0]["params"]["core"] == "gav"
    assert solr_calls[0]["params"]["rows"] == 200
    assert solr_calls[0]["params"]["wt"] == "json"
    assert all(call["timeout"] > 0 for call in calls)


def test_solr_candidate_lookup_deduplicates_equal_roles_and_marks_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        _url: str,
        *,
        params: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> _Response:
        if params is None:
            raise requests.Timeout("repository metadata unavailable")
        return _Response({"response": {"numFound": 1, "docs": [{"v": "1.0"}]}})

    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.time.sleep",
        lambda *_args: None,
    )
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.0", {"1.0.0"})
    assert len(candidates) == 1
    assert candidates[0].selection_roles == ("maven_minimum", "maven_latest")
    assert candidates[0].already_attempted is True


def test_maven_registry_rejects_unsafe_coordinates_and_reports_both_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def failed_request(
        url: str,
        *,
        params: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        calls.append((url, params))
        if params is None:
            raise requests.Timeout("repository metadata timed out")
        raise requests.ConnectionError("Solr unavailable")

    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get",
        failed_request,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.time.sleep",
        lambda *_args: None,
    )
    with pytest.raises(ValueError):
        fetch_maven_registry_candidates("org.safe:widget?x=1", "1.0")
    assert calls == []

    with pytest.raises(ValueError, match="Maven Central") as error:
        fetch_maven_registry_candidates("org.safe:widget", "1.0")
    assert "repository metadata timed out" in str(error.value)
    assert "Solr unavailable" in str(error.value)
    assert sum(params is None for _url, params in calls) == 3
    assert sum(params is not None for _url, params in calls) == 3
