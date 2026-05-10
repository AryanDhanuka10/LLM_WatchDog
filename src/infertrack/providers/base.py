# src/infertrack/providers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseProvider(ABC):
    """Abstract interface every provider adapter must implement.

    A provider is responsible for three things:
      1. Deciding whether a given API response object belongs to it.
      2. Extracting (input_tokens, output_tokens) from that response.
      3. Reporting its canonical name string.

    Cost calculation is intentionally NOT on the provider — it delegates
    to pricing.table.calculate_cost so the pricing table is the single
    source of truth for all providers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name, e.g. 'openai' or 'anthropic'."""

    @abstractmethod
    def detect(self, response: Any) -> bool:
        """Return True if *response* was produced by this provider.

        Should never raise — return False on any unexpected input.
        """

    @abstractmethod
    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Return (input_tokens, output_tokens) from a response object.

        Both values must be non-negative integers.
        Raises ValueError if usage data cannot be found.
        """

    @abstractmethod
    def extract_model(self, response: Any) -> str:
        """Return the model identifier string from the response.

        Raises ValueError if model cannot be determined.
        """