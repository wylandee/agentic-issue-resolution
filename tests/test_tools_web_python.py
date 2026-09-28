from __future__ import annotations

from unittest.mock import MagicMock, call

from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration._tool_support import (
    _is_authoritative_evidence_source,
)
from remediation_engine.orchestration.tools_web import _make_read_web_page_tool


def _response(*, text: str = "", payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.raise_for_status.return_value = None
    if payload is not None:
        response.json.return_value = payload
    return response


def test_python_package_sources_are_authoritative() -> None:
    assert _is_authoritative_evidence_source("https://pypi.org/project/Some_Package/")
    assert not _is_authoritative_evidence_source(
        "https://pypi.org.evil.example/project/Some_Package/"
    )
    assert _is_authoritative_evidence_source("https://packaging.python.org/en/latest/")
    assert _is_authoritative_evidence_source("https://docs.python.org/3/library/")


def test_python_bare_package_uses_normalized_bounded_pypi_json(monkeypatch) -> None:
    plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    reader = _make_read_web_page_tool(plan_state, language=ProjectLanguage.PYTHON)
    response = _response(
        payload={
            "info": {
                "name": "Some_Package",
                "version": "2.3.1",
                "summary": "A useful package",
                "requires_python": ">=3.10",
                "project_urls": {"Source": "https://github.com/example/project"},
            }
        }
    )
    get = MagicMock(return_value=response)
    monkeypatch.setattr("remediation_engine.orchestration.tools_web.requests.get", get)

    result = reader.invoke({"url": "Some.Package"})

    get.assert_called_once_with("https://pypi.org/pypi/some-package/json", timeout=15)
    assert "# PyPI project: Some_Package" in result
    assert "Latest version: 2.3.1" in result
    assert "A useful package" in result
    assert plan_state["has_authoritative_evidence"] is True
    assert plan_state["evidence_source"] == "https://pypi.org/project/some-package/"


def test_python_pypi_project_url_uses_json_then_bounded_page_fallback(monkeypatch) -> None:
    plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    reader = _make_read_web_page_tool(plan_state, language=ProjectLanguage.PYTHON)
    json_failure = _response()
    json_failure.raise_for_status.side_effect = RuntimeError("unavailable")
    page_response = _response(text="# Project page\nPackage details")
    get = MagicMock(side_effect=[json_failure, page_response])
    monkeypatch.setattr("remediation_engine.orchestration.tools_web.requests.get", get)

    result = reader.invoke({"url": "https://pypi.org/project/Some_Package/"})

    assert get.call_args_list == [
        call("https://pypi.org/pypi/some-package/json", timeout=15),
        call(
            "https://r.jina.ai/https://pypi.org/project/some-package/",
            headers={"Accept": "text/plain"},
            timeout=15,
        ),
    ]
    assert "Package details" in result
    assert plan_state["has_authoritative_evidence"] is True
    assert plan_state["evidence_source"] == "https://pypi.org/project/Some_Package/"


def test_node_page_reading_keeps_jina_fallback_unchanged(monkeypatch) -> None:
    reader = _make_read_web_page_tool()
    response = _response(text="# Existing guide")
    get = MagicMock(return_value=response)
    monkeypatch.setattr("remediation_engine.orchestration.tools_web.requests.get", get)

    result = reader.invoke({"url": "https://example.com/guide"})

    get.assert_called_once_with(
        "https://r.jina.ai/https://example.com/guide",
        headers={"Accept": "text/plain"},
        timeout=15,
    )
    assert "# Existing guide" in result
