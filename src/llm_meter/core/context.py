# src/llm_meter/core/context.py
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from llm_meter.providers.openai import OpenAIProvider
from llm_meter.pricing.table import calculate_cost
from llm_meter.storage.models import CallLog
from llm_meter.storage.db import insert_log, DEFAULT_DB_PATH

# Same provider registry as decorator — kept in sync manually until Day 8
_PROVIDERS = [
    OpenAIProvider(),
]


def _detect_provider(response: Any):
    for p in _PROVIDERS:
        if p.detect(response):
            return p
    return None


# WatchContext — the object yielded by watch()                         

@dataclass
class WatchContext:
    """Holds live metrics accumulated inside a ``with watch() as w:`` block.

    Attributes are populated after the block exits (``__exit__``).
    Accessing them before the block closes returns sentinel zero values.
    """

    tag: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Populated on exit
    tokens_used: int = 0          # input + output
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error_msg: Optional[str] = None

    # Internal — not part of public API
    _calls: list[CallLog] = field(default_factory=list, repr=False)

    def record(self, call: CallLog) -> None:
        """Accumulate a completed call into the running totals."""
        self._calls.append(call)
        self.input_tokens  += call.input_tokens
        self.output_tokens += call.output_tokens
        self.tokens_used   = self.input_tokens + self.output_tokens
        self.cost_usd      += call.cost_usd

    @property
    def call_count(self) -> int:
        return len(self._calls)


# watch() — the public context manager                                 

@contextmanager
def watch(
    *,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Generator[WatchContext, None, None]:
    """Context manager that tracks LLM calls made inside the block.

    Usage::

        with watch(tag="summarise", user_id="alice") as w:
            response = client.chat.completions.create(...)
            w.add_response(response)          # <-- register the response

        print(w.tokens_used)   # 1234
        print(w.cost_usd)      # 0.0037
        print(w.latency_ms)    # 892.1

    The context manager measures **wall-clock latency** for the entire
    block (not individual calls).  Individual call logs are also written
    to SQLite so the CLI can show them later.

    Args:
        tag:        Grouping label visible in ``watchdog tail`` / ``top``.
        user_id:    Per-user identifier; used by Budget enforcement.
        session_id: Optional session label.
        db_path:    Override default DB path (useful in tests).
    """
    resolved_db = db_path or DEFAULT_DB_PATH
    ctx = WatchContext(tag=tag, user_id=user_id, session_id=session_id)

    t_start = time.perf_counter()
    exc_caught: Optional[BaseException] = None

    try:
        yield ctx
    except Exception as exc:
        exc_caught = exc
        ctx.success = False
        ctx.error_msg = str(exc)
    finally:
        ctx.latency_ms = (time.perf_counter() - t_start) * 1000

        # If no explicit add_response calls were made but the block raised,
        # still write a failure record so the event is in the log.
        if ctx.call_count == 0 and exc_caught is not None:
            failure_log = CallLog(
                provider="unknown",
                model="unknown",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=ctx.latency_ms,
                success=False,
                error_msg=ctx.error_msg,
                tag=tag,
                user_id=user_id,
                session_id=session_id,
            )
            insert_log(failure_log, db_path=resolved_db)

    if exc_caught is not None:
        raise exc_caught


# WatchContext.add_response  — must be defined after watch() because  
# it needs access to the same _PROVIDERS list and insert_log           

def _add_response(self: WatchContext, response: Any, *, db_path: Optional[Path] = None) -> None:
    """Register a raw API response inside the watch() block.

    Extracts tokens + cost, writes a CallLog to SQLite, and accumulates
    totals on this WatchContext so they're readable after the block.

    Call this once per API response inside the ``with watch() as w:`` block::

        with watch() as w:
            resp = client.chat.completions.create(...)
            w.add_response(resp)
    """
    resolved_db = db_path or DEFAULT_DB_PATH
    provider = _detect_provider(response)

    if provider is not None:
        try:
            input_tokens, output_tokens = provider.extract_usage(response)
            model = provider.extract_model(response)
            cost_usd = calculate_cost(model, input_tokens, output_tokens)
            provider_name = provider.name
            success = True
            error_msg = None
        except Exception as exc:
            input_tokens = output_tokens = 0
            cost_usd = 0.0
            model = "unknown"
            provider_name = "unknown"
            success = False
            error_msg = str(exc)
    else:
        input_tokens = output_tokens = 0
        cost_usd = 0.0
        model = "unknown"
        provider_name = "unknown"
        success = False
        error_msg = "Provider not detected"

    log = CallLog(
        provider=provider_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=0.0,           # per-call latency not tracked in context mode
        success=success,
        error_msg=error_msg,
        tag=self.tag,
        user_id=self.user_id,
        session_id=self.session_id,
    )

    insert_log(log, db_path=resolved_db)
    self.record(log)


# Attach as a method on WatchContext (avoids circular imports)
WatchContext.add_response = _add_response  # type: ignore[attr-defined]