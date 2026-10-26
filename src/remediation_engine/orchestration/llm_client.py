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
    chat_openai_class: Callable[..., Any] | None = None,
) -> Any:
    """Create a ChatOpenAI client configured to use the Responses API.

    Args:
        model_name: OpenAI model ID selected for the agent.
        settings: Runtime settings supplying an explicit API key when present.
        reasoning_effort: Optional validated GPT-6 Luna reasoning effort.
        chat_openai_class: Optional client constructor, primarily for tests and
            existing dependency-injection seams.

    Returns:
        A LangChain ChatOpenAI client using the Responses API.

    Raises:
        ImportError: If ``langchain-openai`` is unavailable and no constructor
            was provided.
        ValueError: If the selected model/effort combination is invalid.
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

    if chat_openai_class is None:
        from langchain_openai import ChatOpenAI

        chat_openai_class = ChatOpenAI

    client_kwargs: dict[str, Any] = {
        "model": model_name,
        **_request_parameters(model_name, reasoning_effort),
        **openai_api_key_kwargs(settings),
    }
    return chat_openai_class(**client_kwargs)
