"""Tests for environment-backed application settings."""

import pytest

from remediation_engine.settings import AppSettings


def test_node_models_can_be_configured_independently(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "remedy-default")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "triage-model")
    monkeypatch.setenv("UPDATE_LLM_MODEL", "update-model")
    monkeypatch.setenv("WORKAROUND_LLM_MODEL", "workaround-model")
    monkeypatch.setenv("QA_LLM_MODEL", "qa-model")

    settings = AppSettings.from_env()

    assert settings.triage_llm_model == "triage-model"
    assert settings.update_llm_model == "update-model"
    assert settings.workaround_llm_model == "workaround-model"
    assert settings.qa_llm_model == "qa-model"


def test_remediation_node_models_fall_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    for name in (
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings.from_env()

    assert settings.remedy_llm_model == "shared-remedy-model"
    assert settings.update_llm_model == "shared-remedy-model"
    assert settings.workaround_llm_model == "shared-remedy-model"
    assert settings.qa_llm_model == "shared-remedy-model"


def test_empty_triage_model_falls_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "")

    assert AppSettings.from_env().triage_llm_model == "shared-remedy-model"


def test_report_settings_are_explicit(monkeypatch, tmp_path):
    """Report persistence directory is configurable via environment."""
    monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(tmp_path))

    settings = AppSettings.from_env()

    assert settings.remediation_report_dir == tmp_path


def test_retriage_limit_is_disabled_by_default_and_toggleable(monkeypatch):
    monkeypatch.delenv("REMEDY_RETRIAGE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("REMEDY_RETRIAGE_LIMIT", raising=False)
    assert AppSettings.from_env().remedy_retriage_limit_enabled is False
    assert AppSettings.from_env().remedy_retriage_limit == 3

    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT", "7")
    assert AppSettings.from_env().remedy_retriage_limit_enabled is True
    assert AppSettings.from_env().remedy_retriage_limit == 7


def test_retriage_limit_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT", "not-a-number")

    with pytest.raises(ValueError, match="REMEDY_RETRIAGE_LIMIT must be an integer"):
        AppSettings.from_env()
