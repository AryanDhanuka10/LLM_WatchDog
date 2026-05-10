# src/infertrack/core/budget.py
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

from infertrack.exceptions import BudgetExceeded
from infertrack.providers.openai import OpenAIProvider
from infertrack.pricing.table import calculate_cost
from infertrack.storage.models import CallLog
from infertrack.storage.db import insert_log, get_total_cost, DEFAULT_DB_PATH

_PROVIDERS = [OpenAIProvider()]


def _detect_and_cost(response: Any) -> tuple[str, str, int, int, float]:
    """Extract (provider, model, input_tok, output_tok, cost) from a response.

    Returns zeros/unknowns on any failure — never raises.
    """
    for provider in _PROVIDERS:
        if provider.detect(response):
            try:
                inp, out = provider.extract_usage(response)
                model    = provider.extract_model(response)
                cost     = calculate_cost(model, inp, out)
                return provider.name, model, inp, out, cost
            except Exception:
                break
    return "unknown", "unknown", 0, 0, 0.0


# BudgetContext — yielded object                                    

class BudgetContext:
    """Live view of spend within a ``with Budget(...) as b:`` block.

    Attributes:
        spent_usd:  Accumulated cost of calls made inside this block
                    plus any pre-existing spend for the user/period.
        call_count: Number of ``add_response()`` calls made so far.
    """

    def __init__(
        self,
        max_usd: float,
        user_id: Optional[str],
        session_id: Optional[str],
        tag: Optional[str],
        db_path: Path,
        period_start: datetime,
        prior_spend: float,
    ) -> None:
        self._max_usd       = max_usd
        self._user_id       = user_id
        self._session_id    = session_id
        self._tag           = tag
        self._db_path       = db_path
        self._period_start  = period_start

        # Start from whatever was already spent in the period
        self.spent_usd: float = prior_spend
        self.call_count: int  = 0

    # Public                                                             

    def add_response(self, response: Any) -> None:
        """Register a raw API response, accumulate cost, enforce budget.

        Call this once per API response inside the ``with Budget() as b:``
        block::

            with Budget(max_usd=0.10, user_id="alice") as b:
                resp = client.chat.completions.create(...)
                b.add_response(resp)   # raises BudgetExceeded if over limit

        Raises:
            BudgetExceeded: if total spend (prior + this block) exceeds max_usd
                            *after* logging the call that tipped the limit.
        """
        prov_name, model, inp, out, cost = _detect_and_cost(response)

        log = CallLog(
            provider      = prov_name,
            model         = model,
            input_tokens  = inp,
            output_tokens = out,
            cost_usd      = cost,
            latency_ms    = 0.0,
            success       = True,
            tag           = self._tag,
            user_id       = self._user_id,
            session_id    = self._session_id,
        )
        insert_log(log, db_path=self._db_path)

        self.spent_usd  += cost
        self.call_count += 1

        if self.spent_usd > self._max_usd:
            raise BudgetExceeded(
                spent   = self.spent_usd,
                limit   = self._max_usd,
                user_id = self._user_id,
            )

    # Internal helpers                                                   

    @property
    def remaining_usd(self) -> float:
        """How many USD remain before the budget is exhausted."""
        return max(0.0, self._max_usd - self.spent_usd)

    @property
    def is_over_budget(self) -> bool:
        return self.spent_usd > self._max_usd


# Budget() — public context manager                                    

@contextmanager
def Budget(
    *,
    max_usd: float,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tag: Optional[str] = None,
    period: str = "today",
    db_path: Optional[Path] = None,
) -> Generator[BudgetContext, None, None]:
    """Context manager that enforces a USD spend limit.

    Usage::

        from infertrack import Budget, BudgetExceeded

        try:
            with Budget(max_usd=0.10, user_id="alice") as b:
                for chunk in docs:
                    resp = client.chat.completions.create(...)
                    b.add_response(resp)          # raises if over limit
        except BudgetExceeded as e:
            print(f"Stopped: spent ${e.spent:.4f} of ${e.limit:.4f}")

    Args:
        max_usd:    Maximum USD allowed for the period.
        user_id:    Scope budget to a specific user (reads prior spend
                    for that user from the DB).  If None, only spend
                    accumulated inside this block is counted.
        session_id: Optional label forwarded to every logged CallLog.
        tag:        Optional label forwarded to every logged CallLog.
        period:     ``"today"`` (default) — counts spend since midnight UTC.
                    ``"all"``  — counts all historical spend for the user.
                    ``"session"`` — counts only spend inside this block
                    (prior_spend starts at 0.0).
        db_path:    Override the default DB path (useful in tests).

    Raises:
        BudgetExceeded: propagated from ``b.add_response()`` when spend
                        exceeds ``max_usd``.
        ValueError:     If ``max_usd`` is not positive.
    """
    if max_usd <= 0:
        raise ValueError(f"max_usd must be positive, got {max_usd}")

    resolved_db  = db_path or DEFAULT_DB_PATH
    now          = datetime.now(timezone.utc)

    # Determine the look-back window for prior spend
    if period == "today":
        period_start: Optional[datetime] = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif period == "all":
        period_start = None
    elif period == "session":
        period_start = now   # nothing before this moment counts
    else:
        raise ValueError(
            f"period must be 'today', 'all', or 'session', got '{period}'"
        )

    # Load prior spend so existing usage counts against the budget
    prior_spend = get_total_cost(
        db_path = resolved_db,
        user_id = user_id,
        since   = period_start,
    ) if user_id is not None else 0.0

    ctx = BudgetContext(
        max_usd      = max_usd,
        user_id      = user_id,
        session_id   = session_id,
        tag          = tag,
        db_path      = resolved_db,
        period_start = period_start or now,
        prior_spend  = prior_spend,
    )

    # Pre-flight check: already over budget before we even start
    if prior_spend > max_usd:
        raise BudgetExceeded(
            spent   = prior_spend,
            limit   = max_usd,
            user_id = user_id,
        )

    yield ctx
    # BudgetExceeded (if raised inside) propagates naturally — no suppression