# src/infertrack/core/decorator.py
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from infertrack.providers.openai import OpenAIProvider
from infertrack.providers.anthropic import AnthropicProvider
from infertrack.pricing.table import calculate_cost
from infertrack.storage.models import CallLog
from infertrack.storage.db import insert_log, init_db, DEFAULT_DB_PATH
from infertrack.core.streaming import is_streaming_response, StreamingWrapper

from pathlib import Path

F = TypeVar("F", bound=Callable[..., Any])

_PROVIDERS = [
    OpenAIProvider(),
    AnthropicProvider(),
]


def _detect_provider(response: Any):
    """Return the first provider whose detect() returns True, or None."""
    for provider in _PROVIDERS:
        if provider.detect(response):
            return provider
    return None


def _build_log(
    *,
    response: Any,
    exc_caught: Optional[Exception],
    latency_ms: float,
    tag: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    retry_count: int = 0,
) -> CallLog:
    """Build a CallLog from a completed (or failed) non-streaming call."""
    if exc_caught is not None:
        return CallLog(
            provider      = "unknown",
            model         = "unknown",
            input_tokens  = 0,
            output_tokens = 0,
            cost_usd      = 0.0,
            latency_ms    = latency_ms,
            success       = False,
            error_msg     = str(exc_caught),
            tag           = tag,
            user_id       = user_id,
            session_id    = session_id,
        )

    provider = _detect_provider(response)

    if provider is not None:
        try:
            input_tokens, output_tokens = provider.extract_usage(response)
            model         = provider.extract_model(response)
            cost_usd      = calculate_cost(model, input_tokens, output_tokens)
            provider_name = provider.name
            parse_error   = None
        except Exception as parse_exc:
            input_tokens = output_tokens = 0
            cost_usd      = 0.0
            model         = "unknown"
            provider_name = "unknown"
            parse_error   = parse_exc
    else:
        input_tokens = output_tokens = 0
        cost_usd      = 0.0
        model         = "unknown"
        provider_name = "unknown"
        parse_error   = None

    return CallLog(
        provider      = provider_name,
        model         = model,
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        cost_usd      = cost_usd,
        latency_ms    = latency_ms,
        success       = (parse_error is None),
        error_msg     = str(parse_error) if parse_error else None,
        tag           = tag,
        user_id       = user_id,
        session_id    = session_id,
    )


def watchdog(
    *,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    retry: int = 0,
    backoff: str = "exponential",
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorator that records every LLM call to the local SQLite log.

    Works with both regular and streaming responses::

        # Non-streaming (existing behaviour)
        @watchdog(tag="summarise")
        def ask(prompt):
            return client.chat.completions.create(model=..., messages=...)

        # Streaming — usage logged after stream is exhausted
        @watchdog(tag="stream")
        def ask_stream(prompt):
            return client.chat.completions.create(
                model=..., messages=...,
                stream=True,
                stream_options={"include_usage": True},  # required for token counts
            )

        for chunk in ask_stream("hello"):
            print(chunk.choices[0].delta.content or "", end="")
        # CallLog written to DB after last chunk is consumed

    Args:
        tag:        Arbitrary label for grouping calls in the CLI.
        user_id:    Per-user identifier (used by Budget enforcement).
        session_id: Optional session grouping label.
        db_path:    Override the default DB path (useful in tests).
        retry:      Number of retry attempts on failure (default 0 = no retry).
        backoff:    Delay strategy: ``'exponential'``, ``'linear'``, or ``'fixed'``.
        base_delay: Initial retry delay in seconds (default 1.0).
        max_delay:  Maximum retry delay cap in seconds (default 60.0).
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_db = db_path or DEFAULT_DB_PATH
            init_db(resolved_db)

            if retry > 0:
                from infertrack.core.retry import with_retry

                retry_count: int = 0
                exc_caught: Optional[Exception] = None
                response: Any = None

                def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
                    nonlocal retry_count
                    retry_count = attempt

                t_start = time.perf_counter()
                try:
                    response = with_retry(
                        func,
                        args=args,
                        kwargs=kwargs,
                        retries=retry,
                        backoff=backoff,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        on_retry=_on_retry,
                    )
                except Exception as exc:
                    exc_caught = exc

                latency_ms = (time.perf_counter() - t_start) * 1000

            else:
                t_start     = time.perf_counter()
                exc_caught  = None
                response    = None
                retry_count = 0

                try:
                    response = func(*args, **kwargs)
                except Exception as exc:
                    exc_caught = exc

                latency_ms = (time.perf_counter() - t_start) * 1000

            # ---- Failed call (exception before any response) ----
            if exc_caught is not None:
                log = _build_log(
                    response    = None,
                    exc_caught  = exc_caught,
                    latency_ms  = latency_ms,
                    tag         = tag,
                    user_id     = user_id,
                    session_id  = session_id,
                    retry_count = retry_count,
                )
                insert_log(log, db_path=resolved_db)
                raise exc_caught

            # ---- Streaming response ----
            if is_streaming_response(response):
                def _on_stream_complete(
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    stream_latency_ms: float,
                    success: bool,
                    error_msg: Optional[str],
                ) -> None:
                    cost = calculate_cost(model, input_tokens, output_tokens)
                    log  = CallLog(
                        provider      = "openai",   # streams are OpenAI only for now
                        model         = model,
                        input_tokens  = input_tokens,
                        output_tokens = output_tokens,
                        cost_usd      = cost,
                        latency_ms    = stream_latency_ms,
                        success       = success,
                        error_msg     = error_msg,
                        tag           = tag,
                        user_id       = user_id,
                        session_id    = session_id,
                    )
                    insert_log(log, db_path=resolved_db)

                return StreamingWrapper(
                    response,
                    on_complete = _on_stream_complete,
                    t_start     = t_start,
                )

            # ---- Normal (non-streaming) response ----
            log = _build_log(
                response    = response,
                exc_caught  = None,
                latency_ms  = latency_ms,
                tag         = tag,
                user_id     = user_id,
                session_id  = session_id,
                retry_count = retry_count,
            )
            insert_log(log, db_path=resolved_db)
            return response

        return wrapper  # type: ignore[return-value]

    return decorator