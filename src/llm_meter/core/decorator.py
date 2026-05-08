# src/llm_meter/core/decorator.py
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from llm_meter.providers.openai import OpenAIProvider
from llm_meter.pricing.table import calculate_cost
from llm_meter.storage.models import CallLog
from llm_meter.storage.db import insert_log, DEFAULT_DB_PATH

from pathlib import Path

F = TypeVar("F", bound=Callable[..., Any])

# Registry of provider detectors — Day 8 will add AnthropicProvider
_PROVIDERS = [
    OpenAIProvider(),
]


def _detect_provider(response: Any):
    """Return the first provider whose detect() returns True, or None."""
    for provider in _PROVIDERS:
        if provider.detect(response):
            return provider
    return None


def watchdog(
    *,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Callable[[F], F]:
    """Decorator that records every LLM call to the local SQLite log.

    Usage::

        @watchdog(tag="summarise", user_id="alice")
        def ask(prompt: str) -> str:
            response = client.chat.completions.create(...)
            return response.choices[0].message.content

    The decorated function must **return the raw API response object**
    so the decorator can extract token counts and model name.
    The original return value is passed through to the caller unchanged.

    Args:
        tag:        Arbitrary label for grouping calls in the CLI.
        user_id:    Per-user identifier (used by Budget enforcement).
        session_id: Optional session grouping label.
        db_path:    Override the default ``~/.llm-meter/logs.db`` path.
                    Primarily useful in tests.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_db = db_path or DEFAULT_DB_PATH

            t_start = time.perf_counter()
            exc_caught: Optional[Exception] = None
            response: Any = None

            try:
                response = func(*args, **kwargs)
            except Exception as exc:
                exc_caught = exc

            latency_ms = (time.perf_counter() - t_start) * 1000

            # --- build CallLog ---
            if exc_caught is not None:
                log = CallLog(
                    provider="unknown",
                    model="unknown",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                    success=False,
                    error_msg=str(exc_caught),
                    tag=tag,
                    user_id=user_id,
                    session_id=session_id,
                )
            else:
                provider = _detect_provider(response)

                if provider is not None:
                    try:
                        input_tokens, output_tokens = provider.extract_usage(response)
                        model = provider.extract_model(response)
                        cost_usd = calculate_cost(model, input_tokens, output_tokens)
                        provider_name = provider.name
                    except Exception as parse_exc:
                        # Parsing failure — still log the call, just without metrics
                        input_tokens = output_tokens = 0
                        cost_usd = 0.0
                        model = "unknown"
                        provider_name = "unknown"
                        exc_caught = parse_exc
                else:
                    # Response not recognised — log as unknown, don't crash
                    input_tokens = output_tokens = 0
                    cost_usd = 0.0
                    model = "unknown"
                    provider_name = "unknown"

                log = CallLog(
                    provider=provider_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    success=(exc_caught is None),
                    error_msg=str(exc_caught) if exc_caught else None,
                    tag=tag,
                    user_id=user_id,
                    session_id=session_id,
                )

            insert_log(log, db_path=resolved_db)

            # Re-raise the original exception AFTER logging
            if exc_caught is not None:
                raise exc_caught

            return response

        return wrapper  # type: ignore[return-value]

    return decorator