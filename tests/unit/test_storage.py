# tests/unit/test_storage.py
"""
Day 1 storage layer tests.
All tests use an in-memory (or tmp) SQLite DB — no file I/O, no Ollama.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from llm_meter.storage.models import CallLog
from llm_meter.storage.db import init_db, insert_log, query_logs, get_total_cost


# Fixtures                                                             

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """A fresh, temporary SQLite database for each test."""
    db_path = tmp_path / "test_logs.db"
    init_db(db_path)
    return db_path


@pytest.fixture
def sample_log() -> CallLog:
    return CallLog(
        provider="openai",
        model="qwen2.5:0.5b",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.0,
        latency_ms=312.5,
        success=True,
        tag="test",
        user_id="user_001",
        session_id="sess_abc",
    )


# CallLog dataclass                                                    

class TestCallLog:
    def test_total_tokens(self, sample_log):
        assert sample_log.total_tokens == 30

    def test_auto_id_generated(self, sample_log):
        # id is None until inserted into the DB (auto-assigned by insert_log)
        assert sample_log.id is None

    def test_auto_timestamp_utc(self, sample_log):
        assert sample_log.timestamp.tzinfo is not None

    def test_timestamp_iso_roundtrip(self, sample_log):
        iso = sample_log.timestamp_iso
        parsed = datetime.fromisoformat(iso)
        assert abs((parsed - sample_log.timestamp).total_seconds()) < 0.001

    def test_unique_ids_after_insert(self, tmp_db):
        """Each CallLog gets a unique UUID assigned by insert_log."""
        a = CallLog(provider="openai", model="x", input_tokens=1,
                    output_tokens=1, cost_usd=0.0, latency_ms=1.0, success=True)
        b = CallLog(provider="openai", model="x", input_tokens=1,
                    output_tokens=1, cost_usd=0.0, latency_ms=1.0, success=True)
        insert_log(a, db_path=tmp_db)
        insert_log(b, db_path=tmp_db)
        assert a.id is not None
        assert b.id is not None
        assert a.id != b.id

    def test_optional_fields_default_none(self):
        log = CallLog(provider="openai", model="x", input_tokens=1,
                      output_tokens=1, cost_usd=0.0, latency_ms=1.0, success=True)
        assert log.tag is None
        assert log.user_id is None
        assert log.session_id is None
        assert log.error_msg is None


# init_db                                                              

class TestInitDb:
    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        init_db(db_path)
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "deeply" / "nested" / "logs.db"
        init_db(db_path)
        assert db_path.exists()

    def test_idempotent(self, tmp_path):
        db_path = tmp_path / "logs.db"
        init_db(db_path)
        init_db(db_path)   # second call must not raise
        assert db_path.exists()

    def test_returns_resolved_path(self, tmp_path):
        db_path = tmp_path / "logs.db"
        returned = init_db(db_path)
        assert returned == db_path


# insert_log + query_logs                                              

class TestInsertAndQuery:
    def test_insert_then_query(self, tmp_db, sample_log):
        insert_log(sample_log, db_path=tmp_db)
        results = query_logs(db_path=tmp_db)
        assert len(results) == 1
        got = results[0]
        assert got.id == sample_log.id
        assert got.model == sample_log.model
        assert got.provider == sample_log.provider

    def test_all_fields_round_trip(self, tmp_db, sample_log):
        insert_log(sample_log, db_path=tmp_db)
        got = query_logs(db_path=tmp_db)[0]

        assert got.input_tokens == sample_log.input_tokens
        assert got.output_tokens == sample_log.output_tokens
        assert abs(got.cost_usd - sample_log.cost_usd) < 1e-9
        assert abs(got.latency_ms - sample_log.latency_ms) < 1e-3
        assert got.success == sample_log.success
        assert got.tag == sample_log.tag
        assert got.user_id == sample_log.user_id
        assert got.session_id == sample_log.session_id
        assert got.error_msg is None

    def test_failed_call_persisted(self, tmp_db):
        log = CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=5, output_tokens=0,
            cost_usd=0.0, latency_ms=50.0,
            success=False, error_msg="RateLimitError"
        )
        insert_log(log, db_path=tmp_db)
        results = query_logs(db_path=tmp_db)
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_msg == "RateLimitError"

    def test_multiple_inserts(self, tmp_db):
        for i in range(5):
            log = CallLog(
                provider="openai", model="qwen2.5:0.5b",
                input_tokens=i, output_tokens=i,
                cost_usd=0.0, latency_ms=float(i * 10),
                success=True,
            )
            insert_log(log, db_path=tmp_db)
        results = query_logs(db_path=tmp_db)
        assert len(results) == 5

    def test_results_newest_first(self, tmp_db):
        """Records should come back ordered newest-first."""
        for i in range(3):
            log = CallLog(
                provider="openai", model="qwen2.5:0.5b",
                input_tokens=i, output_tokens=i,
                cost_usd=0.0, latency_ms=float(i),
                success=True, tag=f"call_{i}",
            )
            insert_log(log, db_path=tmp_db)

        results = query_logs(db_path=tmp_db)
        # Newest-first: timestamps should be non-increasing
        timestamps = [r.timestamp for r in results]
        assert timestamps == sorted(timestamps, reverse=True)


# Filtering                                                            

class TestQueryFilters:
    def _insert_batch(self, tmp_db):
        logs = [
            CallLog(provider="openai",    model="gpt-4o",       input_tokens=10,
                    output_tokens=10, cost_usd=0.01, latency_ms=100.0,
                    success=True,  tag="alpha", user_id="alice"),
            CallLog(provider="openai",    model="gpt-4o-mini",  input_tokens=5,
                    output_tokens=5,  cost_usd=0.001, latency_ms=50.0,
                    success=True,  tag="beta",  user_id="bob"),
            CallLog(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    input_tokens=20, output_tokens=20, cost_usd=0.05,
                    latency_ms=200.0, success=False, tag="alpha", user_id="alice"),
        ]
        for log in logs:
            insert_log(log, db_path=tmp_db)
        return logs

    def test_filter_by_tag(self, tmp_db):
        self._insert_batch(tmp_db)
        results = query_logs(db_path=tmp_db, tag="alpha")
        assert len(results) == 2
        assert all(r.tag == "alpha" for r in results)

    def test_filter_by_user_id(self, tmp_db):
        self._insert_batch(tmp_db)
        results = query_logs(db_path=tmp_db, user_id="alice")
        assert len(results) == 2
        assert all(r.user_id == "alice" for r in results)

    def test_filter_by_model(self, tmp_db):
        self._insert_batch(tmp_db)
        results = query_logs(db_path=tmp_db, model="gpt-4o-mini")
        assert len(results) == 1
        assert results[0].model == "gpt-4o-mini"

    def test_filter_success_only(self, tmp_db):
        self._insert_batch(tmp_db)
        results = query_logs(db_path=tmp_db, success_only=True)
        assert all(r.success for r in results)
        assert len(results) == 2

    def test_filter_by_since(self, tmp_db):
        self._insert_batch(tmp_db)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = query_logs(db_path=tmp_db, since=future)
        assert len(results) == 0

    def test_limit(self, tmp_db):
        self._insert_batch(tmp_db)
        results = query_logs(db_path=tmp_db, limit=2)
        assert len(results) == 2

    def test_empty_db_returns_empty_list(self, tmp_db):
        results = query_logs(db_path=tmp_db)
        assert results == []


# get_total_cost                                                        

class TestGetTotalCost:
    def test_sum_all(self, tmp_db):
        for cost in [0.01, 0.02, 0.03]:
            log = CallLog(provider="openai", model="gpt-4o",
                          input_tokens=1, output_tokens=1,
                          cost_usd=cost, latency_ms=10.0, success=True)
            insert_log(log, db_path=tmp_db)
        total = get_total_cost(db_path=tmp_db)
        assert abs(total - 0.06) < 1e-9

    def test_sum_by_user(self, tmp_db):
        for uid, cost in [("alice", 0.05), ("alice", 0.05), ("bob", 0.10)]:
            log = CallLog(provider="openai", model="gpt-4o",
                          input_tokens=1, output_tokens=1,
                          cost_usd=cost, latency_ms=10.0, success=True,
                          user_id=uid)
            insert_log(log, db_path=tmp_db)
        assert abs(get_total_cost(db_path=tmp_db, user_id="alice") - 0.10) < 1e-9
        assert abs(get_total_cost(db_path=tmp_db, user_id="bob")   - 0.10) < 1e-9

    def test_empty_returns_zero(self, tmp_db):
        assert get_total_cost(db_path=tmp_db) == 0.0