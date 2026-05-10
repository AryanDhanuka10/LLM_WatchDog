# src/infertrack/cli/commands.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import click

from infertrack.storage.db import (
    DEFAULT_DB_PATH,
    init_db,
    query_logs,
    get_total_cost,
    _connect,
)
from infertrack.storage.models import CallLog


# Helpers                                                              

def _resolve_db(db_path: Optional[str]) -> Path:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        init_db(path)
    return path


def _since_datetime(last: str) -> Optional[datetime]:
    """Convert a --last flag value to a UTC datetime cutoff."""
    now = datetime.now(timezone.utc)
    mapping = {
        "1h":  now - timedelta(hours=1),
        "24h": now - timedelta(hours=24),
        "7d":  now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "all": None,
    }
    if last not in mapping:
        raise click.BadParameter(
            f"'{last}' is not valid. Choose from: {', '.join(mapping)}"
        )
    return mapping[last]


def _fmt_cost(usd: float) -> str:
    if usd == 0.0:
        return "$0.00"
    if usd < 0.0001:
        return f"${usd:.8f}"
    if usd < 0.01:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_latency(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def _fmt_ts(ts: datetime) -> str:
    """Format timestamp as HH:MM:SS in local time."""
    local = ts.astimezone()
    return local.strftime("%H:%M:%S")


def _divider(width: int = 72) -> str:
    return "─" * width


# summary                                                              

@click.command("summary")
@click.option(
    "--last", default="24h",
    type=click.Choice(["1h", "24h", "7d", "30d", "all"], case_sensitive=False),
    show_default=True,
    help="Time window to summarise.",
)
@click.option("--db", default=None, help="Path to logs.db (default: ~/.llm-meter/logs.db)")
def summary_cmd(last: str, db: Optional[str]) -> None:
    """Show aggregate stats for a time window."""
    db_path = _resolve_db(db)
    since   = _since_datetime(last)
    logs    = query_logs(db_path=db_path, since=since)

    if not logs:
        click.echo(f"No calls recorded in the last {last}.")
        return

    total_calls    = len(logs)
    total_input    = sum(l.input_tokens  for l in logs)
    total_output   = sum(l.output_tokens for l in logs)
    total_tokens   = total_input + total_output
    total_cost     = sum(l.cost_usd    for l in logs)
    avg_latency    = sum(l.latency_ms  for l in logs) / total_calls
    failed_calls   = sum(1 for l in logs if not l.success)
    success_rate   = (total_calls - failed_calls) / total_calls * 100

    # Model breakdown
    model_counts: dict[str, int] = {}
    for l in logs:
        model_counts[l.model] = model_counts.get(l.model, 0) + 1

    label = f"Last {last}" if last != "all" else "All time"

    click.echo()
    click.echo(f"  llm-meter summary  ·  {label}")
    click.echo(_divider())
    click.echo(f"  Calls          {total_calls:>10,}")
    click.echo(f"  Input tokens   {_fmt_tokens(total_input):>10}")
    click.echo(f"  Output tokens  {_fmt_tokens(total_output):>10}")
    click.echo(f"  Total tokens   {_fmt_tokens(total_tokens):>10}")
    click.echo(f"  Total cost     {_fmt_cost(total_cost):>10}")
    click.echo(f"  Avg latency    {_fmt_latency(avg_latency):>10}")
    click.echo(f"  Success rate   {success_rate:>9.1f}%")
    if failed_calls:
        click.echo(f"  Failed calls   {failed_calls:>10,}")
    click.echo(_divider())

    # Top models
    if model_counts:
        click.echo("  Models:")
        for model, count in sorted(model_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
            click.echo(f"    {model:<38} {count:>4} calls")

    click.echo()


# tail                                                                 

@click.command("tail")
@click.option("-n", "--number", default=20, show_default=True,
              help="Number of recent records to show.")
@click.option("--tag",     default=None, help="Filter by tag.")
@click.option("--user",    default=None, help="Filter by user_id.")
@click.option("--model",   default=None, help="Filter by model name.")
@click.option("--db",      default=None, help="Path to logs.db.")
def tail_cmd(number: int, tag: Optional[str], user: Optional[str],
             model: Optional[str], db: Optional[str]) -> None:
    """Show the N most recent LLM calls."""
    db_path = _resolve_db(db)
    logs    = query_logs(
        db_path      = db_path,
        tag          = tag,
        user_id      = user,
        model        = model,
        limit        = number,
    )

    if not logs:
        click.echo("No records found.")
        return

    click.echo()
    click.echo(f"  {'TIME':<10} {'MODEL':<28} {'TOKENS':>7} {'COST':>10} {'LATENCY':>8}  STATUS")
    click.echo(_divider(78))

    for log in reversed(logs):   # oldest first so it reads top-down
        status = "✓" if log.success else "✗"
        tokens = _fmt_tokens(log.total_tokens)
        parts  = [
            f"  {_fmt_ts(log.timestamp):<10}",
            f" {log.model[:27]:<28}",
            f" {tokens:>7}",
            f" {_fmt_cost(log.cost_usd):>10}",
            f" {_fmt_latency(log.latency_ms):>8}",
            f"  {status}",
        ]
        if log.tag:
            parts.append(f"  [{log.tag}]")
        if log.user_id:
            parts.append(f"  user:{log.user_id}")
        click.echo("".join(parts))

    click.echo()


# top                                                                  

@click.command("top")
@click.option(
    "--by", default="cost",
    type=click.Choice(["cost", "tokens", "calls"], case_sensitive=False),
    show_default=True,
    help="Metric to rank by.",
)
@click.option("--limit",  default=10, show_default=True, help="Number of rows to show.")
@click.option(
    "--group", default="tag",
    type=click.Choice(["tag", "user", "model"], case_sensitive=False),
    show_default=True,
    help="Dimension to group by.",
)
@click.option(
    "--last", default="all",
    type=click.Choice(["1h", "24h", "7d", "30d", "all"], case_sensitive=False),
    show_default=True,
    help="Time window.",
)
@click.option("--db", default=None, help="Path to logs.db.")
def top_cmd(by: str, limit: int, group: str, last: str, db: Optional[str]) -> None:
    """Show top callers ranked by cost, tokens, or call count."""
    db_path = _resolve_db(db)
    since   = _since_datetime(last)
    logs    = query_logs(db_path=db_path, since=since)

    if not logs:
        click.echo("No records found.")
        return

    # Group logs by chosen dimension
    groups: dict[str, list[CallLog]] = {}
    for log in logs:
        if group == "tag":
            key = log.tag or "(no tag)"
        elif group == "user":
            key = log.user_id or "(no user)"
        else:  # model
            key = log.model

        groups.setdefault(key, []).append(log)

    # Build rows
    rows = []
    for key, group_logs in groups.items():
        total_cost   = sum(l.cost_usd    for l in group_logs)
        total_tokens = sum(l.total_tokens for l in group_logs)
        call_count   = len(group_logs)
        rows.append((key, total_cost, total_tokens, call_count))

    # Sort
    sort_index = {"cost": 1, "tokens": 2, "calls": 3}[by]
    rows.sort(key=lambda r: r[sort_index], reverse=True)
    rows = rows[:limit]

    label = f"Last {last}" if last != "all" else "All time"
    click.echo()
    click.echo(f"  llm-meter top  ·  by {by}  ·  grouped by {group}  ·  {label}")
    click.echo(_divider(72))
    click.echo(f"  {'NAME':<30} {'CALLS':>6} {'TOKENS':>8} {'COST':>12}")
    click.echo(_divider(72))

    for key, cost, tokens, calls in rows:
        click.echo(
            f"  {key[:29]:<30} {calls:>6,} {_fmt_tokens(tokens):>8} {_fmt_cost(cost):>12}"
        )

    click.echo()