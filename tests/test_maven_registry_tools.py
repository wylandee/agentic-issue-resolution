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


def test_solr_candidate_lookup_uses_params_paginates_and_returns_safe_extremes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: dict[str, Any], timeout: int) -> _Response:
        calls.append({"url": url, "params": params, "timeout": timeout})
        start = params["start"]
        versions = ["1.8", "2.0-SNAPSHOT", "2.0-rc1", "2.0", "1.9"]
        docs = [
            {"v": version, "timestamp": 100000 - index} for index, version in enumerate(versions)
        ]
        if start == 0:
            docs.extend({"v": "0.1"} for _ in range(195))
            return _Response({"response": {"numFound": 201, "docs": docs}})
        return _Response({"response": {"numFound": 201, "docs": [{"v": "3.0"}]}})

    monkeypatch.setattr("remediation_engine.tools.maven_registry_tools.requests.get", fake_get)
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.9", {"3.0"})
    assert [candidate.version for candidate in candidates] == ["1.9", "3.0"]
    assert candidates[0].selection_roles == ("maven_minimum",)
    assert candidates[1].selection_roles == ("maven_latest",)
    assert candidates[0].already_attempted is False
    assert candidates[1].already_attempted is True
    assert [call["params"]["start"] for call in calls] == [0, 200]
    assert calls[0]["url"] == "https://search.maven.org/solrsearch/select"
    assert calls[0]["params"]["q"] == 'g:"org.safe" AND a:"widget"'
    assert calls[0]["params"]["core"] == "gav"
    assert calls[0]["params"]["rows"] == 200
    assert calls[0]["params"]["wt"] == "json"
    assert calls[0]["timeout"] > 0


def test_solr_candidate_lookup_deduplicates_equal_roles_and_marks_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get",
        lambda *_args, **_kwargs: _Response({"response": {"numFound": 1, "docs": [{"v": "1.0"}]}}),
    )
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.0", {"1.0.0"})
    assert len(candidates) == 1
    assert candidates[0].selection_roles == ("maven_minimum", "maven_latest")
    assert candidates[0].already_attempted is True


def test_maven_registry_rejects_unsafe_coordinates_and_propagates_http_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_request(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unsafe coordinate must not issue a request")

    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get", unexpected_request
    )
    with pytest.raises(ValueError):
        fetch_maven_registry_candidates("org.safe:widget?x=1", "1.0")

    def failed_request(*_args: Any, **_kwargs: Any) -> None:
        raise requests.Timeout("timed out")

    monkeypatch.setattr(
        "remediation_engine.tools.maven_registry_tools.requests.get", failed_request
    )
    with pytest.raises(ValueError, match="Maven Central"):
        fetch_maven_registry_candidates("org.safe:widget", "1.0")
