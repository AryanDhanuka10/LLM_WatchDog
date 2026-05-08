# src/llm_ledger/__init__.py
"""llm-ledger: Zero-config LLM call interceptor.

Track token usage, cost, and latency for every LLM call — locally,
with no cloud account required.

Quick start::

    from llm_ledger import watchdog, watch, Budget, BudgetExceeded

    # Decorator
    @watchdog(tag="my-feature")
    def ask(prompt):
        return client.chat.completions.create(...)

    # Context manager
    with watch(user_id="alice") as w:
        resp = client.chat.completions.create(...)
        w.add_response(resp)
    print(w.cost_usd, w.tokens_used)

    # Budget enforcement
    with Budget(max_usd=0.10, user_id="alice") as b:
        b.add_response(client.chat.completions.create(...))
"""

__version__ = "0.1.0"

from llm_ledger.core.decorator import watchdog
from llm_ledger.core.context import watch, WatchContext
from llm_ledger.core.budget import Budget, BudgetContext
from llm_ledger.exceptions import (
    WatchdogError,
    BudgetExceeded,
    ProviderNotDetected,
    PricingModelNotFound,
)
from llm_ledger.core.interceptor import intercept, stop, is_active

__all__ = [
    # Core API
    "watchdog",
    "watch",
    "WatchContext",
    "Budget",
    "BudgetContext",
    # Exceptions
    "WatchdogError",
    "BudgetExceeded",
    "ProviderNotDetected",
    "PricingModelNotFound",
    # Meta
    "__version__",
]