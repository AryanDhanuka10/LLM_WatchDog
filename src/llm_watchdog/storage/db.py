# src/llm_watchdog/storage/db.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_watchdog.storage.models import CallLog

# DB location                                                          

DEFAULT_DB_DIR = Path.home() / ".llm-watchdog"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "logs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS call_logs (
    id          TEXT    PRIMARY KEY,
    timestamp   TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    latency_ms  REAL    NOT NULL DEFAULT 0.0,
    success     INTEGER NOT NULL DEFAULT 1,
    tag         TEXT,
    user_id     TEXT,
    session_id  TEXT,
    error_msg   TEXT
);
"""


# Public API                                                           

def init_db(db_path: Optional[Path] = None) -> Path:
    """Create the database file and schema if they don't exist.

    Returns the resolved path used so callers always know where the DB is.
    """
    resolved = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with _connect(resolved) as conn:
        conn.executescript(_SCHEMA)

    return resolved


def insert_log(call_log: CallLog, db_path: Optional[Path] = None) -> None:
    """Persist a CallLog entry to the database."""
    resolved = _resolve_path(db_path)

    row = (
        call_log.id,
        call_log.timestamp_iso,
        call_log.provider,
        call_log.model,
        call_log.input_tokens,
        call_log.output_tokens,
        call_log.cost_usd,
        call_log.latency_ms,
        1 if call_log.success else 0,
        call_log.tag,
        call_log.user_id,
        call_log.session_id,
        call_log.error_msg,
    )

    sql = """
        INSERT INTO call_logs
            (id, timestamp, provider, model, input_tokens, output_tokens,
             cost_usd, latency_ms, success, tag, user_id, session_id, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with _connect(resolved) as conn:
        conn.execute(sql, row)


def query_logs(
    *,
    db_path: Optional[Path] = None,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: Optional[int] = None,
    success_only: bool = False,
) -> list[CallLog]:
    """Query call logs with optional filters.

    All keyword arguments are optional; omitting them returns everything.
    Results are ordered newest-first.
    """
    resolved = _resolve_path(db_path)

    conditions: list[str] = []
    params: list = []

    if tag is not None:
        conditions.append("tag = ?")
        params.append(tag)
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
    if model is not None:
        conditions.append("model = ?")
        params.append(model)
    if since is not None:
        conditions.append("timestamp >= ?")
        params.append(since.isoformat())
    if success_only:
        conditions.append("success = 1")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""
        SELECT id, timestamp, provider, model, input_tokens, output_tokens,
               cost_usd, latency_ms, success, tag, user_id, session_id, error_msg
        FROM call_logs
        {where}
        ORDER BY timestamp DESC
        {limit_clause}
    """

    with _connect(resolved) as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

    return [_row_to_calllog(row) for row in rows]


def get_total_cost(
    *,
    db_path: Optional[Path] = None,
    user_id: Optional[str] = None,
    since: Optional[datetime] = None,
) -> float:
    """Return the sum of cost_usd matching the given filters."""
    resolved = _resolve_path(db_path)

    conditions: list[str] = []
    params: list = []

    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if since is not None:
        conditions.append("timestamp >= ?")
        params.append(since.isoformat())

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT COALESCE(SUM(cost_usd), 0.0) FROM call_logs {where}"

    with _connect(resolved) as conn:
        row = conn.execute(sql, params).fetchone()

    return float(row[0])


# Internal helpers                                                     

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent writers
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_path(db_path: Optional[Path]) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _row_to_calllog(row: sqlite3.Row) -> CallLog:
    return CallLog(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        provider=row["provider"],
        model=row["model"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        success=bool(row["success"]),
        tag=row["tag"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        error_msg=row["error_msg"],
    )