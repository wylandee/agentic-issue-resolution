"""Shared configuration for runtime LangChain/OpenAI model clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from remediation_engine.orchestration.runtime_context import openai_api_key_kwargs
from remediation_engine.settings import AppSettings, LunaReasoningEffort


def _request_parameters(
    model_name: str,
    reasoning_effort: LunaReasoningEffort | None,
) -> dict[str, Any]:
    """Build compatible Responses API parameters for a configured model.

    GPT-5.4 mini/nano explicitly use ``none`` reasoning to retain their
    low-latency behavior. GPT-6 Luna defaults to ``medium`` and does not accept
    temperature when reasoning effort is active. Other configured models keep
    the service's historical deterministic temperature setting.
    """
    normalized_model = model_name.strip().lower()
    parameters: dict[str, Any] = {"use_responses_api": True}

    if normalized_model == "gpt-6-luna":
        parameters["reasoning_effort"] = reasoning_effort or "medium"
    elif normalized_model.startswith(("gpt-5.4-mini", "gpt-5.4-nano")):
        parameters["reasoning_effort"] = "none"
        parameters["temperature"] = 0
    else:
        parameters["temperature"] = 0

    return parameters


def create_chat_model(
    model_name: str,
    *,
    settings: AppSettings | None = None,
    reasoning_effort: LunaReasoningEffort | None = None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    chat_openai_class: Callable[..., Any] | None = None,
) -> Any:
    """Create a ChatOpenAI client configured to use the Responses API.

    Args:
        model_name: OpenAI model ID selected for the agent.
        settings: Runtime settings supplying an explicit API key when present.
        reasoning_effort: Optional validated GPT-6 Luna reasoning effort.
        request_timeout: Optional bounded client request timeout in seconds.
        max_retries: Optional retry count; callers may set zero for one-shot
            provider requests.
        max_tokens: Optional output-token ceiling for bounded generations.
        chat_openai_class: Optional client constructor, primarily for tests and
            existing dependency-injection seams.

    Returns:
        A LangChain ChatOpenAI client using the Responses API.

    Raises:
        ImportError: If ``langchain-openai`` is unavailable and no constructor
            was provided.
        ValueError: If the selected model/effort combination or a supplied
            request bound is invalid.
    """
    if model_name.strip().lower() == "gpt-6-luna":
        if reasoning_effort is not None and reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("GPT-6 Luna reasoning effort is invalid")
    elif reasoning_effort is not None:
        raise ValueError("Reasoning effort only applies when the model is gpt-6-luna")
    if request_timeout is not None and request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    if max_retries is not None and max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    if chat_openai_class is None:
        from langchain_openai import ChatOpenAI

        chat_openai_class = ChatOpenAI

    client_kwargs: dict[str, Any] = {
        "model": model_name,
        **_request_parameters(model_name, reasoning_effort),
        **openai_api_key_kwargs(settings),
    }
    if request_timeout is not None:
        client_kwargs["request_timeout"] = request_timeout
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    if max_tokens is not None:
        client_kwargs["max_tokens"] = max_tokens
    return chat_openai_class(**client_kwargs)
