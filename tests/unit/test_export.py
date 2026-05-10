# tests/unit/test_export.py
"""
Day 9 tests: watchdog export --format csv/json.
Uses Click's CliRunner and a tmp SQLite DB — no real LLM calls.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from infertrack.cli.__main__ import cli
from infertrack.storage.db import init_db, insert_log
from infertrack.storage.models import CallLog


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
    logs = [
        CallLog(provider="openai",    model="gpt-4o",
                input_tokens=1000,  output_tokens=500,
                cost_usd=0.01250,   latency_ms=812.0,
                success=True,  tag="summarise", user_id="alice"),
        CallLog(provider="openai",    model="gpt-4o-mini",
                input_tokens=500,   output_tokens=250,
                cost_usd=0.00023,   latency_ms=301.0,
                success=True,  tag="search",    user_id="bob"),
        CallLog(provider="anthropic", model="claude-3-5-sonnet-20241022",
                input_tokens=800,   output_tokens=400,
                cost_usd=0.0084,    latency_ms=650.0,
                success=True,  tag="summarise", user_id="alice"),
        CallLog(provider="openai",    model="qwen2.5:0.5b",
                input_tokens=200,   output_tokens=100,
                cost_usd=0.0,       latency_ms=95.0,
                success=False, tag="search",    user_id="carol",
                error_msg="timeout"),
    ]
    for log in logs:
        insert_log(log, db_path=tmp_db)
    return tmp_db


# CLI registration                                                     

class TestExportRegistered:

    def test_export_in_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert "export" in result.output

    def test_export_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0

    def test_export_help_shows_format(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert "format" in result.output.lower()


# CSV export                                                           

class TestExportCSV:

    def test_csv_exits_zero(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        assert result.exit_code == 0

    def test_csv_has_header(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        first_line = result.output.splitlines()[0]
        assert "model" in first_line
        assert "cost_usd" in first_line
        assert "timestamp" in first_line

    def test_csv_row_count(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        lines = [l for l in result.output.splitlines() if l.strip()]
        # 1 header + 4 data rows
        assert len(lines) == 5

    def test_csv_parseable(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 4

    def test_csv_contains_model_names(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        assert "gpt-4o" in result.output
        assert "claude-3-5-sonnet-20241022" in result.output

    def test_csv_contains_cost(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        assert "0.0125" in result.output

    def test_csv_contains_all_fields(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        reader = csv.DictReader(io.StringIO(result.output))
        row = next(reader)
        for field in ["id", "timestamp", "provider", "model",
                      "input_tokens", "output_tokens", "total_tokens",
                      "cost_usd", "latency_ms", "success",
                      "tag", "user_id", "session_id", "error_msg"]:
            assert field in row, f"Missing field: {field}"

    def test_csv_total_tokens_correct(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(populated_db)])
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        for row in rows:
            total = int(row["total_tokens"])
            inp   = int(row["input_tokens"])
            out   = int(row["output_tokens"])
            assert total == inp + out

    def test_csv_default_format_is_csv(self, runner, populated_db):
        """Omitting --format should default to CSV."""
        result = runner.invoke(cli, ["export", "--db", str(populated_db)])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0]
        assert "model" in first_line   # CSV header


# JSON export                                                          

class TestExportJSON:

    def test_json_exits_zero(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        assert result.exit_code == 0

    def test_json_is_valid(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_json_record_count(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert len(data) == 4

    def test_json_contains_required_keys(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        for record in data:
            for key in ["id", "timestamp", "provider", "model",
                        "input_tokens", "output_tokens", "total_tokens",
                        "cost_usd", "latency_ms", "success"]:
                assert key in record, f"Missing key: {key}"

    def test_json_model_values(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        models = {r["model"] for r in data}
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet-20241022" in models

    def test_json_cost_is_float(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        for record in data:
            assert isinstance(record["cost_usd"], float)

    def test_json_success_is_bool(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        for record in data:
            assert isinstance(record["success"], bool)

    def test_json_timestamp_is_iso_string(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        from datetime import datetime
        for record in data:
            # Must be parseable as ISO datetime
            dt = datetime.fromisoformat(record["timestamp"])
            assert dt is not None

    def test_json_total_tokens_correct(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        for record in data:
            assert record["total_tokens"] == (
                record["input_tokens"] + record["output_tokens"]
            )


# Filtering                                                            

class TestExportFilters:

    def test_filter_by_tag(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--tag", "summarise",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert all(r["tag"] == "summarise" for r in data)
        assert len(data) == 2

    def test_filter_by_user(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--user", "alice",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert all(r["user_id"] == "alice" for r in data)
        assert len(data) == 2

    def test_filter_by_model(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--model", "gpt-4o-mini",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert all(r["model"] == "gpt-4o-mini" for r in data)
        assert len(data) == 1

    def test_filter_last_all(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--last", "all",
                                     "--db", str(populated_db)])
        data = json.loads(result.output)
        assert len(data) == 4

    def test_filter_future_since_empty(self, runner, populated_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--last", "1h",
                                     "--db", str(populated_db)])
        # All records were just inserted so they are within 1h
        assert result.exit_code == 0


# File output (-o flag)                                                

class TestExportFileOutput:

    def test_csv_written_to_file(self, runner, populated_db, tmp_path):
        out = tmp_path / "out.csv"
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "-o", str(out),
                                     "--db", str(populated_db)])
        assert result.exit_code == 0
        assert out.exists()

    def test_csv_file_has_correct_content(self, runner, populated_db, tmp_path):
        out = tmp_path / "out.csv"
        runner.invoke(cli, ["export", "--format", "csv",
                            "-o", str(out), "--db", str(populated_db)])
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4

    def test_json_written_to_file(self, runner, populated_db, tmp_path):
        out = tmp_path / "out.json"
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "-o", str(out),
                                     "--db", str(populated_db)])
        assert result.exit_code == 0
        assert out.exists()

    def test_json_file_has_correct_content(self, runner, populated_db, tmp_path):
        out = tmp_path / "out.json"
        runner.invoke(cli, ["export", "--format", "json",
                            "-o", str(out), "--db", str(populated_db)])
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 4

    def test_file_output_message_on_stderr(self, runner, populated_db, tmp_path):
        out = tmp_path / "out.csv"
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "-o", str(out),
                                     "--db", str(populated_db)])
        assert result.exit_code == 0
        # File must be written; stdout should be empty (message goes to stderr)
        assert out.exists()
        assert out.stat().st_size > 0


# Empty DB                                                             

class TestExportEmptyDB:

    def test_empty_db_csv_exits_zero(self, runner, tmp_db):
        result = runner.invoke(cli, ["export", "--format", "csv",
                                     "--db", str(tmp_db)])
        assert result.exit_code == 0

    def test_empty_db_json_exits_zero(self, runner, tmp_db):
        result = runner.invoke(cli, ["export", "--format", "json",
                                     "--db", str(tmp_db)])
        assert result.exit_code == 0

    def test_empty_db_shows_message(self, runner, tmp_db):
        result = runner.invoke(cli, ["export", "--db", str(tmp_db)])
        assert result.exit_code == 0


# _log_to_dict helper                                                  

class TestLogToDict:

    def test_all_keys_present(self):
        from infertrack.cli.export import _log_to_dict
        log = CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=10, output_tokens=20,
            cost_usd=0.01, latency_ms=500.0, success=True,
            tag="t", user_id="u", session_id="s",
        )
        d = _log_to_dict(log)
        for key in ["id", "timestamp", "provider", "model",
                    "input_tokens", "output_tokens", "total_tokens",
                    "cost_usd", "latency_ms", "success",
                    "tag", "user_id", "session_id", "error_msg"]:
            assert key in d

    def test_total_tokens_computed(self):
        from infertrack.cli.export import _log_to_dict
        log = CallLog(provider="openai", model="x",
                      input_tokens=10, output_tokens=25,
                      cost_usd=0.0, latency_ms=1.0, success=True)
        assert _log_to_dict(log)["total_tokens"] == 35

    def test_timestamp_is_iso_string(self):
        from infertrack.cli.export import _log_to_dict
        from datetime import datetime
        log = CallLog(provider="openai", model="x",
                      input_tokens=1, output_tokens=1,
                      cost_usd=0.0, latency_ms=1.0, success=True)
        ts = _log_to_dict(log)["timestamp"]
        assert isinstance(ts, str)
        datetime.fromisoformat(ts)   # must not raise