# src/infertrack/storage/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CallLog:
    """Represents a single recorded LLM API call.

    All fields have defaults so ``CallLog()`` is valid (useful in tests
    and as a blank slate before population).  ``id`` is intentionally
    ``None`` by default — the DB layer assigns it only when persisting,
    and the placeholder test asserts ``log.id is None``.

    ``total_tokens`` is computed by ``__post_init__`` and stored as a
    plain int field so it round-trips through SQLite without a property.
    """

    # Core fields — all have safe defaults
    provider: str      = "unknown"
    model: str         = "unknown"
    input_tokens: int  = 0
    output_tokens: int = 0
    cost_usd: float    = 0.0
    latency_ms: float  = 0.0
    success: bool      = True

    # Optional metadata
    tag: Optional[str]        = None
    user_id: Optional[str]    = None
    session_id: Optional[str] = None
    error_msg: Optional[str]  = None

    # id is None until explicitly set or inserted into the DB
    id: Optional[str] = None

    # Auto-set to now when not provided
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Computed in __post_init__ — do NOT set manually
    total_tokens: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.total_tokens = self.input_tokens + self.output_tokens

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()