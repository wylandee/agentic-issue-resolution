"""Tests for model-specific Responses API client construction."""

import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from remediation_engine.orchestration.llm_client import create_chat_model
from remediation_engine.settings import AppSettings


class _StructuredResult(BaseModel):
    """Small response schema for testing LangChain's offline payload conversion."""

    answer: str


@pytest.mark.parametrize(
    ("model", "effort", "expected_effort", "expected_temperature"),
    [
        ("gpt-4o-mini", None, None, 0),
        ("gpt-5.4-mini", None, "none", 0),
        ("gpt-5.4-nano", None, "none", 0),
        ("gpt-6-luna", None, "medium", None),
        ("gpt-6-luna", "none", "none", None),
        ("gpt-6-luna", "low", "low", None),
        ("gpt-6-luna", "medium", "medium", None),
        ("gpt-6-luna", "high", "high", None),
        ("gpt-6-luna", "xhigh", "xhigh", None),
        ("gpt-6-luna", "max", "max", None),
    ],
)
def test_responses_payload_uses_model_compatible_parameters(
    model, effort, expected_effort, expected_temperature
):
    settings = AppSettings(openai_api_key="test-key")
    llm = create_chat_model(
        model,
        settings=settings,
        reasoning_effort=effort,
        chat_openai_class=ChatOpenAI,
    )

    payload = llm._get_request_payload("test prompt")

    assert payload["model"] == model
    assert payload["input"]
    assert llm.use_responses_api is True
    if expected_effort is None:
        assert "reasoning" not in payload
    else:
        assert payload["reasoning"] == {"effort": expected_effort}
    if expected_temperature is None:
        assert "temperature" not in payload
    else:
        assert payload["temperature"] == expected_temperature


def test_factory_rejects_effort_for_non_luna_model():
    with pytest.raises(ValueError, match="only applies when the model is gpt-6-luna"):
        create_chat_model("gpt-4o-mini", reasoning_effort="low", chat_openai_class=ChatOpenAI)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-6-luna"])
def test_responses_client_converts_function_tools(model):
    llm = create_chat_model(
        model,
        settings=AppSettings(openai_api_key="test-key"),
        chat_openai_class=ChatOpenAI,
    )
    bound_llm = llm.bind_tools(
        [
            {
                "name": "lookup_item",
                "description": "Look up an item.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        ],
        parallel_tool_calls=False,
    )

    payload = llm._get_request_payload("test prompt", **bound_llm.kwargs)

    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "lookup_item"
    assert payload["parallel_tool_calls"] is False


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-6-luna"])
def test_responses_client_converts_structured_output(model):
    llm = create_chat_model(
        model,
        settings=AppSettings(openai_api_key="test-key"),
        chat_openai_class=ChatOpenAI,
    )
    structured_llm = llm.with_structured_output(_StructuredResult)

    payload = llm._get_request_payload("test prompt", **structured_llm.first.kwargs)

    assert payload["text_format"] is _StructuredResult
