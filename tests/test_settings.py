"""Tests for environment-backed application settings."""

import pytest

from remediation_engine.settings import (
    DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET,
    AppSettings,
)

_LLM_REASONING_EFFORT_ENV_NAMES = (
    "REMEDY_LLM_REASONING_EFFORT",
    "TRIAGE_LLM_REASONING_EFFORT",
    "UPDATE_LLM_REASONING_EFFORT",
    "WORKAROUND_LLM_REASONING_EFFORT",
    "QA_LLM_REASONING_EFFORT",
)


def _clear_llm_reasoning_effort_environment(monkeypatch):
    for env_name in _LLM_REASONING_EFFORT_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)


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


@pytest.mark.parametrize(
    ("env_name", "field_name"),
    [
        ("REMEDY_LLM_REASONING_EFFORT", "remedy_llm_reasoning_effort"),
        ("TRIAGE_LLM_REASONING_EFFORT", "triage_llm_reasoning_effort"),
        ("UPDATE_LLM_REASONING_EFFORT", "update_llm_reasoning_effort"),
        ("WORKAROUND_LLM_REASONING_EFFORT", "workaround_llm_reasoning_effort"),
        ("QA_LLM_REASONING_EFFORT", "qa_llm_reasoning_effort"),
    ],
)
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_luna_reasoning_efforts_are_configurable_per_agent(
    monkeypatch, env_name, field_name, effort
):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    for model_env in (
        "TRIAGE_LLM_MODEL",
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
    ):
        monkeypatch.delenv(model_env, raising=False)
    monkeypatch.setenv(env_name, effort)

    assert getattr(AppSettings.from_env(), field_name) == effort


def test_luna_reasoning_effort_defaults_to_provider_default(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")

    assert AppSettings.from_env().remedy_llm_reasoning_effort is None


def test_luna_reasoning_effort_is_case_normalized(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", " HIGH ")

    assert AppSettings.from_env().remedy_llm_reasoning_effort == "high"


def test_luna_reasoning_effort_rejects_unknown_values(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", "minimal")

    with pytest.raises(ValueError, match="REMEDY_LLM_REASONING_EFFORT must be one of"):
        AppSettings.from_env()


def test_luna_reasoning_effort_requires_gpt_6_luna(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", "low")

    with pytest.raises(ValueError, match="only applies when its model is gpt-6-luna"):
        AppSettings.from_env()


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


def test_final_full_scan_is_disabled_by_default_and_toggleable(monkeypatch):
    variable = "REMEDY_FINAL_FULL_SCAN_ENABLED"
    monkeypatch.delenv(variable, raising=False)
    assert AppSettings().remedy_final_full_scan_enabled is False
    assert AppSettings.from_env().remedy_final_full_scan_enabled is False

    monkeypatch.setenv(variable, "true")
    assert AppSettings.from_env().remedy_final_full_scan_enabled is True


def test_retriage_limit_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT", "not-a-number")

    with pytest.raises(ValueError, match="REMEDY_RETRIAGE_LIMIT must be an integer"):
        AppSettings.from_env()


def test_solver_certification_timeout_has_positive_configurable_default(monkeypatch):
    monkeypatch.delenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", raising=False)
    assert AppSettings.from_env().solver_certification_timeout_seconds == 900

    monkeypatch.setenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", "37")
    assert AppSettings.from_env().solver_certification_timeout_seconds == 37

    monkeypatch.setenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be at least 1"):
        AppSettings.from_env()


def test_candidate_limit_default_and_override(monkeypatch):
    variable = "REMEDY_SOLVER_MAX_CANDIDATES_PER_TARGET"
    monkeypatch.delenv(variable, raising=False)

    assert (
        AppSettings().solver_max_candidates_per_target == DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET
    )
    assert (
        AppSettings.from_env().solver_max_candidates_per_target
        == DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET
    )

    monkeypatch.setenv(variable, "96")
    assert AppSettings.from_env().solver_max_candidates_per_target == 96


def test_delta_canary_probe_budget_default_and_override(monkeypatch):
    variable = "REMEDY_MAX_DELTA_CANARY_PROBES"
    monkeypatch.delenv(variable, raising=False)

    assert AppSettings().max_delta_canary_probes == 2
    assert AppSettings.from_env().max_delta_canary_probes == 2

    monkeypatch.setenv(variable, "5")
    assert AppSettings.from_env().max_delta_canary_probes == 5


@pytest.mark.parametrize(
    ("value", "message"),
    [("0", "must be at least 1"), ("not-an-integer", "must be an integer")],
)
def test_delta_canary_probe_budget_requires_positive_integer(monkeypatch, value, message):
    monkeypatch.setenv("REMEDY_MAX_DELTA_CANARY_PROBES", value)

    with pytest.raises(ValueError, match=message):
        AppSettings.from_env()
