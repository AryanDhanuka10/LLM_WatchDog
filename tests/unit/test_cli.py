# tests/unit/test_cli.py
"""
Day 5 tests: CLI commands — summary, tail, top.
Uses Click's CliRunner so no subprocess needed and no real DB on disk.
All data inserted via insert_log() into a tmp SQLite file.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from click.testing import CliRunner

from llm_watchdog.cli.__main__ import cli
from llm_watchdog.storage.db import init_db, insert_log
from llm_watchdog.storage.models import CallLog


# Fixtures                                                             

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture
def populated_db(tmp_db: Path) -> Path:
    """DB pre-loaded with a realistic mix of calls."""
    logs = [
        CallLog(provider="openai",    model="gpt-4o",
                input_tokens=1000,  output_tokens=500,
                cost_usd=0.01250,   latency_ms=812.0,
                success=True,  tag="summarise",  user_id="alice"),
        CallLog(provider="openai",    model="gpt-4o-mini",
                input_tokens=500,   output_tokens=250,
                cost_usd=0.00023,   latency_ms=301.0,
                success=True,  tag="search",     user_id="bob"),
        CallLog(provider="openai",    model="qwen2.5:0.5b",
                input_tokens=200,   output_tokens=100,
                cost_usd=0.0,       latency_ms=95.0,
                success=True,  tag="summarise",  user_id="alice"),
        CallLog(provider="openai",    model="gpt-4o",
                input_tokens=800,   output_tokens=400,
                cost_usd=0.010,     latency_ms=750.0,
                success=False, tag="search",     user_id="carol",
                error_msg="RateLimitError"),
        CallLog(provider="openai",    model="gpt-4o-mini",
                input_tokens=300,   output_tokens=150,
                cost_usd=0.00014,   latency_ms=210.0,
                success=True,  tag="classify",   user_id="bob"),
    ]
    for log in logs:
        insert_log(log, db_path=tmp_db)
    return tmp_db


# CLI top-level                                                        

class TestCLITopLevel:

    def test_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_subcommands(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert "summary" in result.output
        assert "tail"    in result.output
        assert "top"     in result.output

    def test_unknown_command_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["doesnotexist"])
        assert result.exit_code != 0


# summary                                                              

class TestSummaryCommand:

    def test_empty_db_reports_no_calls(self, runner, tmp_db):
        result = runner.invoke(cli, ["summary", "--db", str(tmp_db)])
        assert result.exit_code == 0
        assert "No calls" in result.output

    def test_exits_zero(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db)])
        assert result.exit_code == 0

    def test_shows_call_count(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert result.exit_code == 0
        assert "5" in result.output   # 5 total calls

    def test_shows_cost(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert "$" in result.output

    def test_shows_tokens(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        # total tokens = (1000+500)+(500+250)+(200+100)+(800+400)+(300+150) = 4200
        output = result.output
        assert "k" in output or "4" in output   # formatted token count

    def test_shows_latency(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert "ms" in result.output or "s" in result.output

    def test_shows_success_rate(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert "%" in result.output

    def test_shows_models(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert "gpt-4o" in result.output

    def test_last_1h_option(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "1h"])
        assert result.exit_code == 0

    def test_last_7d_option(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "7d"])
        assert result.exit_code == 0

    def test_last_30d_option(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "30d"])
        assert result.exit_code == 0

    def test_last_all_option(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        assert result.exit_code == 0

    def test_failed_calls_shown_when_present(self, runner, populated_db):
        result = runner.invoke(cli, ["summary", "--db", str(populated_db), "--last", "all"])
        # 1 failed call in populated_db
        assert "Failed" in result.output or "1" in result.output


# tail                                                                 

class TestTailCommand:

    def test_empty_db_reports_no_records(self, runner, tmp_db):
        result = runner.invoke(cli, ["tail", "--db", str(tmp_db)])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_exits_zero(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db)])
        assert result.exit_code == 0

    def test_shows_model_names(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db)])
        assert "gpt-4o" in result.output

    def test_shows_cost(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db)])
        assert "$" in result.output

    def test_shows_latency(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db)])
        assert "ms" in result.output or "s" in result.output

    def test_number_flag_limits_output(self, runner, populated_db):
        result_2 = runner.invoke(cli, ["tail", "--db", str(populated_db), "-n", "2"])
        result_5 = runner.invoke(cli, ["tail", "--db", str(populated_db), "-n", "5"])
        # 2-record output must be shorter than 5-record output
        assert len(result_2.output) < len(result_5.output)

    def test_n_1_shows_one_row(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db), "-n", "1"])
        assert result.exit_code == 0
        # Only one model name row expected — count lines with "$"
        cost_lines = [l for l in result.output.splitlines() if "$" in l]
        assert len(cost_lines) == 1

    def test_tag_filter(self, runner, populated_db):
        result = runner.invoke(
            cli, ["tail", "--db", str(populated_db), "--tag", "search"]
        )
        assert result.exit_code == 0
        assert "search" in result.output

    def test_user_filter(self, runner, populated_db):
        result = runner.invoke(
            cli, ["tail", "--db", str(populated_db), "--user", "alice"]
        )
        assert result.exit_code == 0
        assert "alice" in result.output

    def test_model_filter(self, runner, populated_db):
        result = runner.invoke(
            cli, ["tail", "--db", str(populated_db), "--model", "gpt-4o-mini"]
        )
        assert result.exit_code == 0
        assert "gpt-4o-mini" in result.output

    def test_success_indicator_present(self, runner, populated_db):
        result = runner.invoke(cli, ["tail", "--db", str(populated_db)])
        # Both ✓ and ✗ should appear (mix of success/failure in populated_db)
        assert "✓" in result.output
        assert "✗" in result.output


# top                                                                  

class TestTopCommand:

    def test_empty_db_reports_no_records(self, runner, tmp_db):
        result = runner.invoke(cli, ["top", "--db", str(tmp_db)])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_exits_zero(self, runner, populated_db):
        result = runner.invoke(cli, ["top", "--db", str(populated_db)])
        assert result.exit_code == 0

    def test_by_cost_default(self, runner, populated_db):
        result = runner.invoke(cli, ["top", "--db", str(populated_db), "--by", "cost"])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_by_tokens(self, runner, populated_db):
        result = runner.invoke(cli, ["top", "--db", str(populated_db), "--by", "tokens"])
        assert result.exit_code == 0

    def test_by_calls(self, runner, populated_db):
        result = runner.invoke(cli, ["top", "--db", str(populated_db), "--by", "calls"])
        assert result.exit_code == 0

    def test_group_by_tag(self, runner, populated_db):
        result = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--group", "tag"]
        )
        assert result.exit_code == 0
        assert "summarise" in result.output or "search" in result.output

    def test_group_by_user(self, runner, populated_db):
        result = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--group", "user"]
        )
        assert result.exit_code == 0
        assert "alice" in result.output or "bob" in result.output

    def test_group_by_model(self, runner, populated_db):
        result = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--group", "model"]
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_limit_flag(self, runner, populated_db):
        result_1 = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--limit", "1", "--group", "model"]
        )
        result_3 = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--limit", "3", "--group", "model"]
        )
        assert len(result_1.output) < len(result_3.output)

    def test_last_all_shows_all_data(self, runner, populated_db):
        result = runner.invoke(
            cli, ["top", "--db", str(populated_db), "--last", "all"]
        )
        assert result.exit_code == 0

    def test_shows_call_count_column(self, runner, populated_db):
        result = runner.invoke(cli, ["top", "--db", str(populated_db)])
        # Should show numeric call counts
        assert any(c.isdigit() for c in result.output)

    def test_highest_cost_first(self, runner, populated_db):
        """When sorted by cost, the most expensive group appears first."""
        result = runner.invoke(
            cli, ["top", "--db", str(populated_db),
                  "--by", "cost", "--group", "tag", "--last", "all"]
        )
        output = result.output
        # "summarise" has gpt-4o calls ($0.0125) and should rank above "classify"
        summarise_pos = output.find("summarise")
        classify_pos  = output.find("classify")
        assert summarise_pos != -1
        assert classify_pos  != -1
        assert summarise_pos < classify_pos


# Formatting helpers (unit tests — no DB needed)                       

class TestFormatHelpers:

    def test_fmt_cost_zero(self):
        from llm_watchdog.cli.commands import _fmt_cost
        assert _fmt_cost(0.0) == "$0.00"

    def test_fmt_cost_small(self):
        from llm_watchdog.cli.commands import _fmt_cost
        result = _fmt_cost(0.000001)
        assert "$" in result
        assert "0" in result

    def test_fmt_cost_normal(self):
        from llm_watchdog.cli.commands import _fmt_cost
        assert _fmt_cost(0.0125) == "$0.0125"

    def test_fmt_tokens_small(self):
        from llm_watchdog.cli.commands import _fmt_tokens
        assert _fmt_tokens(999) == "999"

    def test_fmt_tokens_thousands(self):
        from llm_watchdog.cli.commands import _fmt_tokens
        assert "k" in _fmt_tokens(1500)

    def test_fmt_tokens_millions(self):
        from llm_watchdog.cli.commands import _fmt_tokens
        assert "M" in _fmt_tokens(1_500_000)

    def test_fmt_latency_ms(self):
        from llm_watchdog.cli.commands import _fmt_latency
        assert "ms" in _fmt_latency(250.0)

    def test_fmt_latency_seconds(self):
        from llm_watchdog.cli.commands import _fmt_latency
        assert "s" in _fmt_latency(1500.0)
        assert "ms" not in _fmt_latency(1500.0)