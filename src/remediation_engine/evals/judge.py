"""Bounded DeepEval judge model shared by CLI and pytest runners."""

from __future__ import annotations

import json
from typing import Any


def build_bounded_judge_model(
    model_name: str,
    api_key: str,
    *,
    timeout: int = 60,
) -> Any | None:
    """Build a DeepEval OpenAI model with one bounded request per judgment.

    Args:
        model_name: OpenAI model name used by DeepEval.
        api_key: OpenAI credential restored by the evaluation boundary.
        timeout: Per-request timeout in seconds.

    Returns:
        A DeepEval GPT model, or ``None`` when optional DeepEval/OpenAI
        dependencies are unavailable.

    Raises:
        RuntimeError: DeepEval changes its retry-decorated call interface.
    """
    try:
        from deepeval.models.llms.openai_model import GPTModel
        from openai import AsyncOpenAI, OpenAI
    except ImportError:
        return None

    def call_once(method: Any, *args: Any, **kwargs: Any) -> Any:
        raw_method = getattr(method, "__wrapped__", None)
        if raw_method is None:
            if hasattr(method, "retry"):
                raise RuntimeError(
                    "DeepEval GPTModel retry wrapper changed; configure a bounded judge call."
                )
            raw_method = method
        return raw_method(*args, **kwargs)

    def validate_json_completion(model: Any, completion: Any, schema: Any) -> Any:
        """Parse a JSON-mode completion into DeepEval's requested schema."""
        output = completion.choices[0].message.content
        if not output:
            raise ValueError("Judge returned an empty Verdicts response")
        return (
            schema.model_validate(json.loads(output)),
            model.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            ),
        )

    def uses_verdicts_json_mode(schema: Any) -> bool:
        """Avoid strict structured output for DeepEval's nested Verdicts schema."""
        return getattr(schema, "__name__", "") == "Verdicts"

    class BoundedGPTModel(GPTModel):
        """GPTModel adapter that disables unbounded retries."""

        def load_model(self, async_mode: bool = False) -> Any:
            client_type = AsyncOpenAI if async_mode else OpenAI
            return client_type(
                api_key=self._openai_api_key,
                timeout=timeout,
                max_retries=0,
            )

        def generate(self, prompt: str, schema: Any = None) -> Any:
            if uses_verdicts_json_mode(schema):
                # This nested schema repeatedly read-times out through strict
                # structured outputs. JSON mode returns the same shape promptly;
                # validate the typed payload locally before using it.
                completion = self.load_model(async_mode=False).chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                )
                return validate_json_completion(self, completion, schema)
            return call_once(GPTModel.generate, self, prompt, schema)

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            if uses_verdicts_json_mode(schema):
                completion = await self.load_model(async_mode=True).chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                )
                return validate_json_completion(self, completion, schema)
            return await call_once(GPTModel.a_generate, self, prompt, schema)

        def generate_raw_response(self, prompt: str, top_logprobs: int = 5) -> Any:
            return call_once(
                GPTModel.generate_raw_response, self, prompt, top_logprobs=top_logprobs
            )

        async def a_generate_raw_response(self, prompt: str, top_logprobs: int = 5) -> Any:
            return await call_once(
                GPTModel.a_generate_raw_response,
                self,
                prompt,
                top_logprobs=top_logprobs,
            )

    return BoundedGPTModel(model=model_name, _openai_api_key=api_key)
