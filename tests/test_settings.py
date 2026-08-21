"""Tests for environment-backed application settings."""

from remediation_engine.settings import AppSettings


def test_node_models_can_be_configured_independently(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "remedy-default")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "triage-model")
    monkeypatch.setenv("SUPERVISOR_LLM_MODEL", "supervisor-model")
    monkeypatch.setenv("UPDATE_LLM_MODEL", "update-model")
    monkeypatch.setenv("WORKAROUND_LLM_MODEL", "workaround-model")
    monkeypatch.setenv("QA_LLM_MODEL", "qa-model")

    settings = AppSettings.from_env()

    assert settings.triage_llm_model == "triage-model"
    assert settings.supervisor_llm_model == "supervisor-model"
    assert settings.update_llm_model == "update-model"
    assert settings.workaround_llm_model == "workaround-model"
    assert settings.qa_llm_model == "qa-model"


def test_remediation_node_models_fall_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    for name in (
        "SUPERVISOR_LLM_MODEL",
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings.from_env()

    assert settings.remedy_llm_model == "shared-remedy-model"
    assert settings.supervisor_llm_model == "shared-remedy-model"
    assert settings.update_llm_model == "shared-remedy-model"
    assert settings.workaround_llm_model == "shared-remedy-model"
    assert settings.qa_llm_model == "shared-remedy-model"


def test_empty_triage_model_falls_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "")

    assert AppSettings.from_env().triage_llm_model == "shared-remedy-model"


def test_report_settings_are_explicit_and_default_to_deterministic(monkeypatch, tmp_path):
    """Report persistence and narrative generation have independent settings."""
    monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_LLM_ENABLED", "true")
    monkeypatch.setenv("REPORT_LLM_MODEL", "report-model")

    settings = AppSettings.from_env()

    assert settings.remediation_report_dir == tmp_path
    assert settings.report_llm_enabled is True
    assert settings.report_llm_model == "report-model"


def test_report_llm_model_falls_back_to_legacy_remedy_model(monkeypatch):
    """The optional report narrative uses the existing model fallback contract."""
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    monkeypatch.delenv("REPORT_LLM_MODEL", raising=False)

    assert AppSettings.from_env().report_llm_model == "shared-remedy-model"
