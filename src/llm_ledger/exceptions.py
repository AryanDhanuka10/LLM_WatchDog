# src/llm_ledger/exceptions.py
"""All llm-meter exceptions in one place."""
from __future__ import annotations

from typing import Optional


class WatchdogError(Exception):
    """Base class for all llm-meter exceptions."""


class BudgetExceeded(WatchdogError):
    """Raised when accumulated spend crosses the configured limit.

    Attributes:
        spent:   Total USD spent so far in the Budget window.
        limit:   The max_usd threshold that was crossed.
        user_id: The user identifier associated with this budget (may be None).

    Example::

        try:
            with Budget(max_usd=0.10, user_id="alice"):
                call_llm()
        except BudgetExceeded as e:
            print(f"{e.user_id} spent ${e.spent:.4f} (limit ${e.limit:.4f})")
    """

    def __init__(
        self,
        spent: float,
        limit: float,
        user_id: Optional[str] = None,
    ) -> None:
        self.spent   = spent
        self.limit   = limit
        self.user_id = user_id

        who = f"user '{user_id}'" if user_id else "session"
        super().__init__(
            f"Budget exceeded for {who}: "
            f"spent ${spent:.6f}, limit ${limit:.6f}"
        )


class ProviderNotDetected(WatchdogError):
    """Raised when no provider can be matched to an API response object.

    The watchdog decorator logs these as ``provider='unknown'`` rather
    than raising — this exception is available for stricter usage if
    callers opt in.
    """

    def __init__(self, response_type: str = "unknown") -> None:
        self.response_type = response_type
        super().__init__(
            f"Could not detect LLM provider from response of type '{response_type}'. "
            f"Supported providers: openai (and Ollama), anthropic."
        )


class PricingModelNotFound(WatchdogError):
    """Raised when a model is not present in prices.json.

    The pricing table returns 0.0 cost for unknown models rather than
    raising — this exception is available for stricter usage.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Model '{model}' not found in the pricing table. "
            f"Cost will be reported as $0.00. "
            f"Add it to prices.json or call pricing.table.reload() after editing."
        )