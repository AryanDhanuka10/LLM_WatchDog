# src/infertrack/providers/anthropic.py
"""Provider adapter for Anthropic Claude API responses.

Handles ``anthropic.types.Message`` objects returned by
``anthropic.Anthropic().messages.create()``.

Token counts come ONLY from ``response.usage`` — no external library.
Cost is delegated to ``pricing.table.calculate_cost``.
"""
from __future__ import annotations

from typing import Any, Tuple

from infertrack.providers.base import BaseProvider
from infertrack.pricing.table import calculate_cost


class AnthropicProvider(BaseProvider):
    """Handles responses from anthropic.types.Message.

    Detection is duck-typed — we never import ``anthropic`` at module
    level so the package remains optional.

    Anthropic response shape (relevant fields)::

        response.model           → "claude-3-5-sonnet-20241022"
        response.usage.input_tokens   → int
        response.usage.output_tokens  → int
        response.content[0].text → str   (not used for counting)
        response.stop_reason     → "end_turn" | "max_tokens" | …
    """

    @property
    def name(self) -> str:
        return "anthropic"

    def detect(self, response: Any) -> bool:
        """Return True for objects that look like anthropic.types.Message.

        Key distinguishing features vs OpenAI ChatCompletion:
          - uses ``usage.input_tokens`` / ``usage.output_tokens``
            (OpenAI uses ``prompt_tokens`` / ``completion_tokens``)
          - has ``content`` as a list of typed blocks, not ``choices``
          - has ``stop_reason`` not ``finish_reason``
        """
        try:
            return (
                hasattr(response, "usage")
                and hasattr(response.usage, "input_tokens")
                and hasattr(response.usage, "output_tokens")
                and hasattr(response, "model")
                and hasattr(response, "content")
                and not hasattr(response, "choices")   # rules out OpenAI
            )
        except Exception:
            return False

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Return (input_tokens, output_tokens) from response.usage.

        Raises ValueError if usage fields are missing or non-integer.
        """
        try:
            input_tokens  = int(response.usage.input_tokens)
            output_tokens = int(response.usage.output_tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"AnthropicProvider: could not extract token counts. "
                f"Expected response.usage.input_tokens and "
                f"response.usage.output_tokens to be integers. "
                f"Got: {exc}"
            ) from exc

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"AnthropicProvider: token counts must be non-negative, "
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
                f"AnthropicProvider: could not read response.model: {exc}"
            ) from exc

        if not model:
            raise ValueError("AnthropicProvider: response.model is empty")

        return model

    def calculate_cost(self, model: str, input_tokens: int,
                       output_tokens: int) -> float:
        """Convenience wrapper around the shared pricing table."""
        return calculate_cost(model, input_tokens, output_tokens)