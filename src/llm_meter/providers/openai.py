# src/llm_meter/providers/openai.py
from __future__ import annotations

from typing import Any, Tuple

from llm_meter.providers.base import BaseProvider
from llm_meter.pricing.table import calculate_cost


class OpenAIProvider(BaseProvider):
    """Handles responses from openai.types.chat.ChatCompletion.

    Also handles Ollama responses because Ollama exposes an
    OpenAI-compatible API — the response object shape is identical.

    Token counts come ONLY from response.usage — no tiktoken.
    """

    @property
    def name(self) -> str:
        return "openai"

    def detect(self, response: Any) -> bool:
        """Return True for any object that looks like a ChatCompletion.

        We check for the attributes we actually use rather than importing
        the openai package (which is an optional dependency).
        """
        try:
            return (
                hasattr(response, "usage")
                and hasattr(response.usage, "prompt_tokens")
                and hasattr(response.usage, "completion_tokens")
                and hasattr(response, "model")
                and hasattr(response, "choices")
            )
        except Exception:
            return False

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Return (input_tokens, output_tokens) from response.usage.

        Raises ValueError if usage fields are missing or non-integer.
        """
        try:
            input_tokens  = int(response.usage.prompt_tokens)
            output_tokens = int(response.usage.completion_tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"OpenAIProvider: could not extract token counts from response. "
                f"Expected response.usage.prompt_tokens and "
                f"response.usage.completion_tokens to be integers. "
                f"Got: {exc}"
            ) from exc

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"OpenAIProvider: token counts must be non-negative, "
                f"got input={input_tokens}, output={output_tokens}"
            )

        return input_tokens, output_tokens

    def extract_model(self, response: Any) -> str:
        """Return the model string from response.model.

        Raises ValueError if model is missing or empty.
        """
        try:
            model = str(response.model).strip()
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                f"OpenAIProvider: could not read response.model: {exc}"
            ) from exc

        if not model:
            raise ValueError("OpenAIProvider: response.model is empty")

        return model

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Convenience wrapper around the shared pricing table."""
        return calculate_cost(model, input_tokens, output_tokens)