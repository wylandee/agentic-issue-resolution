"""Credential redaction checks for persisted evaluation records."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evals import conftest
from tests.evals.conftest import EvalSettings, _redact_eval_secrets


def test_eval_error_text_redacts_the_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured credentials must not survive in text prepared for persistence."""
    secret = "sk-svcacct-" + "x" * 24
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    redacted = _redact_eval_secrets(f"failure had credential {secret}")
    if redacted != "failure had credential [REDACTED]":
        pytest.fail("Evaluation text did not redact the configured API key.")


def test_eval_settings_repr_omits_the_api_key() -> None:
    """Pytest traceback locals must not expose the credential through dataclass repr."""
    secret = "sk-svcacct-" + "y" * 24
    settings = EvalSettings(
        judge_model="gpt-4o-mini",
        is_live=True,
        openai_api_key=secret,
        trajectory_dir=Path("."),
        golden_dir=Path("."),
    )

    if secret in repr(settings):
        pytest.fail("EvalSettings repr exposed the API key.")


def test_collect_only_session_finish_does_not_persist_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection-only pytest commands must not create skipped eval runs."""
    monkeypatch.setattr(
        conftest,
        "_load_deep_eval_test_cases",
        lambda: pytest.fail("Collect-only session reached eval persistence."),
    )
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(collectonly=True)))

    conftest.pytest_sessionfinish(session, 0)
