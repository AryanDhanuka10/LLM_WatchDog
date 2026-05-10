# src/infertrack/__init__.py
"""infertrack: Zero-config LLM call interceptor.

Track token usage, cost, and latency for every LLM call — locally,
with no cloud account required.

Quick start::

    from infertrack import watchdog, watch, Budget, BudgetExceeded

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

    # Zero-code-change global intercept
    import infertrack
    infertrack.intercept(tag="my-app")
    # all subsequent client.chat.completions.create() calls are logged
"""

__version__ = "1.0.0"

from infertrack.core.decorator import watchdog
from infertrack.core.context import watch, WatchContext
from infertrack.core.budget import Budget, BudgetContext
from infertrack.core.interceptor import intercept, stop, is_active
from infertrack.exceptions import (
    WatchdogError,
    BudgetExceeded,
    ProviderNotDetected,
    PricingModelNotFound,
)

__all__ = [
    # Core API
    "watchdog",
    "watch",
    "WatchContext",
    "Budget",
    "BudgetContext",
    # Interceptor
    "intercept",
    "stop",
    "is_active",
    # Exceptions
    "WatchdogError",
    "BudgetExceeded",
    "ProviderNotDetected",
    "PricingModelNotFound",
    # Meta
    "__version__",
]