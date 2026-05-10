# src/infertrack/pricing/table.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# prices.json lives next to this file
_PRICES_PATH = Path(__file__).parent / "prices.json"

# Module-level cache so we only parse the file once per process
_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with open(_PRICES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        # Strip meta keys (start with "_") so only model entries remain
        _cache = {k: v for k, v in data.items() if not k.startswith("_")}
    return _cache


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for a completed call.

    Falls back to 0.0 for unknown / Ollama models — never raises.
    Uses prices.json embedded in the package.

    Args:
        model:         Model identifier as returned by the API (e.g. "gpt-4o").
        input_tokens:  Prompt / input token count from response.usage.
        output_tokens: Completion / output token count from response.usage.

    Returns:
        Cost in USD as a float (may be 0.0 for free/local models).
    """
    table = _load()
    entry = table.get(model)

    if entry is None:
        # Unknown model — free / custom / ollama variant not in table
        return 0.0

    input_cost  = entry["input_per_1k"]  * input_tokens  / 1000
    output_cost = entry["output_per_1k"] * output_tokens / 1000
    return input_cost + output_cost


def get_price_entry(model: str) -> Optional[dict]:
    """Return the raw price entry for a model, or None if not found."""
    return _load().get(model)


def known_models() -> list[str]:
    """Return all model names present in prices.json."""
    return list(_load().keys())


def reload() -> None:
    """Force a reload of prices.json (useful in tests or after manual edits)."""
    global _cache
    _cache = None
    _load()