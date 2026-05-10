# src/infertrack/core/interceptor.py
"""Global monkey-patch interceptor.

Usage::

    import infertrack
    infertrack.intercept()          # patch once at startup

    # All subsequent calls are logged automatically — zero other changes
    client = OpenAI(...)
    client.chat.completions.create(...)   # logged
    client.chat.completions.create(...)   # logged

    infertrack.stop()               # restore originals
"""
from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Optional

from infertrack.providers.openai import OpenAIProvider
from infertrack.providers.anthropic import AnthropicProvider
from infertrack.pricing.table import calculate_cost
from infertrack.storage.models import CallLog
from infertrack.storage.db import insert_log, init_db, DEFAULT_DB_PATH

# Module-level state                                                   

_PROVIDERS = [
    OpenAIProvider(),
    AnthropicProvider(),
]

# Stores original methods so stop() can restore them
_originals: dict[str, Any] = {}

# Runtime config — set by intercept()
_intercept_config: dict[str, Any] = {
    "db_path":    None,
    "tag":        None,
    "user_id":    None,
    "session_id": None,
    "active":     False,
}


# Wrapper factory                                                      

def _make_wrapper(original_fn, db_path: Path):
    """Return a drop-in replacement that logs every call then delegates."""

    @functools.wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t_start = time.perf_counter()
        exc_caught: Optional[Exception] = None
        response: Any = None

        try:
            response = original_fn(*args, **kwargs)
        except Exception as exc:
            exc_caught = exc

        latency_ms = (time.perf_counter() - t_start) * 1000

        cfg = _intercept_config

        if exc_caught is not None:
            log = CallLog(
                provider   = "unknown",
                model      = "unknown",
                input_tokens  = 0,
                output_tokens = 0,
                cost_usd   = 0.0,
                latency_ms = latency_ms,
                success    = False,
                error_msg  = str(exc_caught),
                tag        = cfg["tag"],
                user_id    = cfg["user_id"],
                session_id = cfg["session_id"],
            )
        else:
            _matched = next((p for p in _PROVIDERS if p.detect(response)), None)
            if _matched is not None:
                try:
                    inp, out  = _matched.extract_usage(response)
                    model     = _matched.extract_model(response)
                    cost      = calculate_cost(model, inp, out)
                    prov_name = _matched.name
                    success   = True
                    error_msg = None
                except Exception as parse_exc:
                    inp = out = 0
                    cost = 0.0
                    model = "unknown"
                    prov_name = "unknown"
                    success = False
                    error_msg = str(parse_exc)
            else:
                inp = out = 0
                cost = 0.0
                model = "unknown"
                prov_name = "unknown"
                success = True
                error_msg = None

            log = CallLog(
                provider      = prov_name,
                model         = model,
                input_tokens  = inp,
                output_tokens = out,
                cost_usd      = cost,
                latency_ms    = latency_ms,
                success       = success,
                error_msg     = error_msg,
                tag           = cfg["tag"],
                user_id       = cfg["user_id"],
                session_id    = cfg["session_id"],
            )

        insert_log(log, db_path=db_path)

        if exc_caught is not None:
            raise exc_caught

        return response

    return wrapper


# Public API                                                           

def intercept(
    *,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Monkey-patch the OpenAI client so every call is logged automatically.

    Call once at application startup — before any ``OpenAI()`` client is
    used.  All subsequent ``client.chat.completions.create()`` calls
    (including Ollama via the OpenAI-compatible endpoint) will be
    intercepted and logged to the local SQLite database.

    Args:
        tag:        Label attached to every intercepted log entry.
        user_id:    User identifier attached to every log entry.
        session_id: Session identifier attached to every log entry.
        db_path:    Override default DB path.

    Raises:
        ImportError:  If ``openai`` is not installed.
        RuntimeError: If ``intercept()`` has already been called without
                      a matching ``stop()``.

    Example::

        import infertrack
        infertrack.intercept(tag="my-app")

        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.chat.completions.create(...)   # automatically logged
    """
    if _intercept_config["active"]:
        raise RuntimeError(
            "intercept() already active. Call infertrack.stop() first."
        )

    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "openai package is required for intercept(). "
            "Install it with: pip install openai"
        ) from exc

    resolved_db = db_path or DEFAULT_DB_PATH
    init_db(resolved_db)  # ensure dir + schema exist before any write

    # Update runtime config
    _intercept_config.update({
        "db_path":    resolved_db,
        "tag":        tag,
        "user_id":    user_id,
        "session_id": session_id,
        "active":     True,
    })

    # ---- Patch openai.chat.completions.create ----
    # The method lives on the Completions resource class, not the instance,
    # so patching the class affects all existing and future client instances.
    try:
        from openai.resources.chat.completions import Completions
        original = Completions.create
        _originals["openai.Completions.create"] = original
        Completions.create = _make_wrapper(original, resolved_db)
    except (ImportError, AttributeError) as exc:
        _intercept_config["active"] = False
        raise RuntimeError(
            f"Could not patch openai.resources.chat.completions.Completions.create. "
            f"Your openai version may have a different internal structure. "
            f"Error: {exc}"
        ) from exc

    # ---- Patch anthropic.messages.create (optional — skip if not installed) ----
    try:
        from anthropic.resources.messages import Messages  # pragma: no cover
        original_anthropic = Messages.create  # pragma: no cover
        _originals["anthropic.Messages.create"] = original_anthropic  # pragma: no cover
        Messages.create = _make_wrapper(original_anthropic, resolved_db)  # pragma: no cover
    except ImportError:
        pass   # anthropic not installed — that's fine
    except AttributeError as exc:  # pragma: no cover
        # anthropic installed but internal structure changed — warn, don't crash
        import warnings  # pragma: no cover
        warnings.warn(  # pragma: no cover
            f"infertrack: could not patch anthropic.Messages.create: {exc}. "
            f"Anthropic calls will not be intercepted.",
            stacklevel=2,
        )


def stop() -> None:
    """Restore all original methods patched by ``intercept()``.

    Safe to call even if ``intercept()`` was never called.

    Example::

        infertrack.intercept()
        # ... do work ...
        infertrack.stop()
    """
    if not _intercept_config["active"]:
        return

    try:
        from openai.resources.chat.completions import Completions
        if "openai.Completions.create" in _originals:
            Completions.create = _originals.pop("openai.Completions.create")
    except (ImportError, AttributeError):
        pass

    try:  # pragma: no cover
        from anthropic.resources.messages import Messages  # pragma: no cover
        if "anthropic.Messages.create" in _originals:  # pragma: no cover
            Messages.create = _originals.pop("anthropic.Messages.create")  # pragma: no cover
    except (ImportError, AttributeError):  # pragma: no cover
        pass  # pragma: no cover

    _originals.clear()
    _intercept_config.update({
        "db_path":    None,
        "tag":        None,
        "user_id":    None,
        "session_id": None,
        "active":     False,
    })


def is_active() -> bool:
    """Return True if intercept() is currently active."""
    return bool(_intercept_config["active"])