# src/infertrack/core/retry.py
"""Retry logic for the @watchdog decorator.

Provides two backoff strategies:
  - exponential: 1s, 2s, 4s, 8s, … (capped at max_delay)
  - linear:      1s, 2s, 3s, 4s, … (capped at max_delay)

Usage via @watchdog::

    @watchdog(retry=3, backoff="exponential")
    def ask(prompt):
        return client.chat.completions.create(...)

Usage standalone::

    from infertrack.core.retry import with_retry

    result = with_retry(my_fn, args=(prompt,), retries=3, backoff="linear")
"""
from __future__ import annotations

import time
import logging
from typing import Any, Callable, Optional, Sequence, Type

logger = logging.getLogger(__name__)

# Exceptions that should NOT be retried — budget and config errors are
# programmer mistakes, not transient failures.
_NO_RETRY_EXCEPTIONS: tuple[Type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
)

try:
    from infertrack.exceptions import BudgetExceeded
    _NO_RETRY_EXCEPTIONS = _NO_RETRY_EXCEPTIONS + (BudgetExceeded,)
except ImportError:
    pass


def _compute_delay(
    attempt: int,          # 0-indexed (0 = first retry)
    backoff: str,
    base_delay: float,
    max_delay: float,
) -> float:
    """Return the sleep duration in seconds for a given attempt."""
    if backoff == "exponential":
        delay = base_delay * (2 ** attempt)
    elif backoff == "linear":
        delay = base_delay * (attempt + 1)
    elif backoff == "fixed":
        delay = base_delay
    else:
        raise ValueError(
            f"Unknown backoff strategy '{backoff}'. "
            f"Choose from: 'exponential', 'linear', 'fixed'."
        )
    return min(delay, max_delay)


def with_retry(
    fn: Callable[..., Any],
    *,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    retries: int = 3,
    backoff: str = "exponential",
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: Optional[Sequence[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """Call ``fn(*args, **kwargs)`` and retry on failure.

    Args:
        fn:          The callable to invoke.
        args:        Positional arguments for ``fn``.
        kwargs:      Keyword arguments for ``fn``.
        retries:     Maximum number of retry attempts (not counting the first call).
                     Total attempts = retries + 1.
        backoff:     Delay strategy: ``'exponential'``, ``'linear'``, or ``'fixed'``.
        base_delay:  Initial delay in seconds (default 1.0).
        max_delay:   Maximum delay cap in seconds (default 60.0).
        retry_on:    If provided, only retry on these exception types.
                     If None, retry on any Exception not in the no-retry list.
        on_retry:    Optional callback ``(attempt, exception, delay) -> None``
                     called before each sleep. Useful for logging/metrics.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        The last exception raised by ``fn`` if all attempts are exhausted.
        ValueError: If ``backoff`` is not a recognised strategy.
    """
    if kwargs is None:
        kwargs = {}

    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")

    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):  # attempt 0 = first try
        try:
            return fn(*args, **kwargs)
        except _NO_RETRY_EXCEPTIONS:
            raise
        except Exception as exc:
            last_exc = exc

            # Check retry_on whitelist
            if retry_on is not None:
                if not isinstance(exc, tuple(retry_on)):
                    raise

            if attempt == retries:
                # Exhausted all retries
                break

            delay = _compute_delay(attempt, backoff, base_delay, max_delay)

            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            else:
                logger.warning(
                    "infertrack retry %d/%d after %.1fs — %s: %s",
                    attempt + 1, retries, delay,
                    type(exc).__name__, exc,
                )

            time.sleep(delay)

    assert last_exc is not None
    raise last_exc