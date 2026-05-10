# src/infertrack/cli/export.py
"""watchdog export subcommand — CSV and JSON export."""
from __future__ import annotations

import csv
import io
import json
import sys
from datetime import timezone
from pathlib import Path
from typing import Optional

import click

from infertrack.storage.db import DEFAULT_DB_PATH, init_db, query_logs
from infertrack.storage.models import CallLog


def _log_to_dict(log: CallLog) -> dict:
    """Serialise a CallLog to a plain dict safe for CSV and JSON."""
    return {
        "id":            log.id,
        "timestamp":     log.timestamp.astimezone(timezone.utc).isoformat(),
        "provider":      log.provider,
        "model":         log.model,
        "input_tokens":  log.input_tokens,
        "output_tokens": log.output_tokens,
        "total_tokens":  log.total_tokens,
        "cost_usd":      log.cost_usd,
        "latency_ms":    log.latency_ms,
        "success":       log.success,
        "tag":           log.tag,
        "user_id":       log.user_id,
        "session_id":    log.session_id,
        "error_msg":     log.error_msg,
    }


_CSV_FIELDS = [
    "id", "timestamp", "provider", "model",
    "input_tokens", "output_tokens", "total_tokens",
    "cost_usd", "latency_ms", "success",
    "tag", "user_id", "session_id", "error_msg",
]


def _export_csv(logs: list[CallLog], output) -> None:
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for log in logs:
        writer.writerow(_log_to_dict(log))


def _export_json(logs: list[CallLog], output) -> None:
    rows = [_log_to_dict(log) for log in logs]
    json.dump(rows, output, indent=2, ensure_ascii=False)
    output.write("\n")


@click.command("export")
@click.option(
    "--format", "fmt",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output", "-o", "output_path",
    default=None,
    help="File to write to. Defaults to stdout.",
)
@click.option(
    "--last",
    type=click.Choice(["1h", "24h", "7d", "30d", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Time window to export.",
)
@click.option("--tag",   default=None, help="Filter by tag.")
@click.option("--user",  default=None, help="Filter by user_id.")
@click.option("--model", default=None, help="Filter by model name.")
@click.option("--db",    default=None, help="Path to logs.db.")
def export_cmd(
    fmt: str,
    output_path: Optional[str],
    last: str,
    tag: Optional[str],
    user: Optional[str],
    model: Optional[str],
    db: Optional[str],
) -> None:
    """Export call logs to CSV or JSON.

    \b
    Examples:
        watchdog export                          # CSV to stdout
        watchdog export --format json            # JSON to stdout
        watchdog export -o calls.csv             # CSV to file
        watchdog export --format json -o out.json
        watchdog export --last 7d --tag search   # filtered
    """
    # Resolve DB
    db_path = Path(db) if db else DEFAULT_DB_PATH
    if not db_path.exists():
        init_db(db_path)

    # Resolve time filter
    from infertrack.cli.commands import _since_datetime
    since = _since_datetime(last)

    logs = query_logs(
        db_path     = db_path,
        tag         = tag,
        user_id     = user,
        model       = model,
        since       = since,
    )
    # Export newest-first from query_logs; reverse for chronological order
    logs = list(reversed(logs))

    if not logs:
        click.echo("No records to export.", err=True)
        return

    if output_path:
        # Write to file
        out_file = Path(output_path)
        with open(out_file, "w", newline="" if fmt == "csv" else None,
                  encoding="utf-8") as fh:
            if fmt == "csv":
                _export_csv(logs, fh)
            else:
                _export_json(logs, fh)
        click.echo(
            f"Exported {len(logs)} record(s) to {out_file} ({fmt.upper()})",
            err=True,
        )
    else:
        # Write to stdout
        if fmt == "csv":
            output = io.StringIO()
            _export_csv(logs, output)
            click.echo(output.getvalue(), nl=False)
        else:
            output = io.StringIO()
            _export_json(logs, output)
            click.echo(output.getvalue(), nl=False)