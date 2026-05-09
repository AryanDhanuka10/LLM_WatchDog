# src/llm_ledger/cli/__main__.py
"""Entry point for the watchdog CLI.

Installed as ``watchdog`` via pyproject.toml::

    [project.scripts]
    watchdog = "llm_ledger.cli.__main__:cli"

Can also be invoked as::

    python -m llm_ledger summary
    python -m llm_ledger tail -n 5
"""
from __future__ import annotations

import click

from llm_ledger.cli.commands import summary_cmd, tail_cmd, top_cmd
from llm_ledger.cli.export import export_cmd


@click.group()
@click.version_option(package_name="llm-watchdog")
def cli() -> None:
    """llm-watchdog: zero-config LLM call interceptor.

    Track token usage, cost, and latency for every LLM call — locally,
    with no cloud account required.

    \b
    Quick start:
        watchdog summary          # last 24 h at a glance
        watchdog tail -n 10       # last 10 calls
        watchdog top --by cost    # biggest spenders
    """


cli.add_command(summary_cmd, name="summary")
cli.add_command(tail_cmd,    name="tail")
cli.add_command(top_cmd,     name="top")
cli.add_command(export_cmd,  name="export")


if __name__ == "__main__":
    cli()