"""Tests for per-run runtime settings context binding."""

import pytest

from remediation_engine.orchestration.runtime_context import (
    get_bound_runtime_settings,
    get_runtime_settings,
    use_runtime_settings,
)
from remediation_engine.settings import AppSettings


def test_use_runtime_settings_binds_settings_for_current_run() -> None:
    settings = AppSettings(openai_api_key="bound-key", remedy_llm_model="bound-model")

    assert get_bound_runtime_settings() is None

    with use_runtime_settings(settings):
        assert get_bound_runtime_settings() is settings
        assert get_runtime_settings() is settings

    assert get_bound_runtime_settings() is None


def test_use_runtime_settings_restores_outer_binding_for_nested_contexts() -> None:
    outer_settings = AppSettings(openai_api_key="outer-key")
    inner_settings = AppSettings(openai_api_key="inner-key")

    with use_runtime_settings(outer_settings):
        assert get_bound_runtime_settings() is outer_settings

        with use_runtime_settings(inner_settings):
            assert get_bound_runtime_settings() is inner_settings
            assert get_runtime_settings() is inner_settings

        assert get_bound_runtime_settings() is outer_settings

    assert get_bound_runtime_settings() is None


def test_use_runtime_settings_restores_binding_when_context_raises() -> None:
    outer_settings = AppSettings(openai_api_key="outer-key")
    inner_settings = AppSettings(openai_api_key="inner-key")

    def raise_from_inner_context() -> None:
        with use_runtime_settings(inner_settings):
            assert get_bound_runtime_settings() is inner_settings
            raise RuntimeError("context failure")

    with (
        use_runtime_settings(outer_settings),
        pytest.raises(RuntimeError, match="context failure"),
    ):
        raise_from_inner_context()
        assert get_bound_runtime_settings() is outer_settings

    assert get_bound_runtime_settings() is None
