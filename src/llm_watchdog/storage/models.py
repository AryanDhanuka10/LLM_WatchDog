# src/llm_watchdog/storage/models.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CallLog:
    """Represents a single recorded LLM API call."""

    provider: str                          # "openai" | "anthropic" | "unknown"
    model: str                             # e.g. "qwen2.5:0.5b", "gpt-4o"
    input_tokens: int                      # from response.usage only
    output_tokens: int                     # from response.usage only
    cost_usd: float                        # calculated from prices.json
    latency_ms: float                      # wall-clock milliseconds
    success: bool                          # False if an exception was raised

    # Optional metadata
    tag: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error_msg: Optional[str] = None

    # Auto-populated fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Convenience properties                                               

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()