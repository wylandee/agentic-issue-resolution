"""Per-run dependencies that must not be serialized into graph state."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from remediation_engine.settings import AppSettings

_settings_var: ContextVar[AppSettings | None] = ContextVar(
    "remediation_engine_settings",
    default=None,
)


def get_runtime_settings() -> AppSettings:
    """Return the current run settings, falling back at the app boundary."""
    settings = _settings_var.get()
    return settings if settings is not None else AppSettings.from_env()


def openai_api_key_kwargs(settings: AppSettings | None = None) -> dict[str, str]:
    """Return the explicit OpenAI credential argument when settings provide one.

    Leaving the argument out when the configured key is empty preserves the
    OpenAI client's normal environment lookup for callers that do not inject
    settings.
    """
    api_key = (settings or get_runtime_settings()).openai_api_key
    return {"api_key": api_key} if api_key else {}


def get_bound_runtime_settings() -> AppSettings | None:
    """Return settings explicitly bound by the current graph invocation."""
    return _settings_var.get()


@contextmanager
def use_runtime_settings(settings: AppSettings) -> Iterator[None]:
    """Bind validated settings for the current graph execution context."""
    token: Token[AppSettings | None] = _settings_var.set(settings)
    try:
        yield
    finally:
        _settings_var.reset(token)
