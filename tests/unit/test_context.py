# tests/unit/test_context.py
"""
Day 3 tests: watch() context manager and WatchContext.
All tests use mocked response objects and a temp SQLite DB — no real LLM calls.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from llm_ledger.core.context import watch, WatchContext
from llm_ledger.storage.db import init_db, query_logs


# Fixtures                                                             

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def make_response(model="qwen2.5:0.5b", prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = "Hello"
    return resp


# WatchContext dataclass                                               

class TestWatchContext:

    def test_initial_zero_state(self):
        ctx = WatchContext()
        assert ctx.tokens_used == 0
        assert ctx.input_tokens == 0
        assert ctx.output_tokens == 0
        assert ctx.cost_usd == 0.0
        assert ctx.latency_ms == 0.0
        assert ctx.success is True
        assert ctx.call_count == 0

    def test_metadata_stored(self):
        ctx = WatchContext(tag="t", user_id="u", session_id="s")
        assert ctx.tag == "t"
        assert ctx.user_id == "u"
        assert ctx.session_id == "s"


# Basic watch() usage                                                  

class TestWatchBasic:

    def test_yields_watch_context(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            assert isinstance(w, WatchContext)

    def test_tokens_populated_after_add_response(self, tmp_db):
        resp = make_response(prompt_tokens=10, completion_tokens=20)
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        assert w.input_tokens == 10
        assert w.output_tokens == 20
        assert w.tokens_used == 30

    def test_cost_populated_for_paid_model(self, tmp_db):
        resp = make_response(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        assert w.cost_usd > 0.0

    def test_cost_zero_for_ollama(self, tmp_db):
        resp = make_response(model="qwen2.5:0.5b", prompt_tokens=100, completion_tokens=200)
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        assert w.cost_usd == 0.0

    def test_latency_positive_after_block(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            _ = make_response()   # some work

        assert w.latency_ms > 0

    def test_success_true_on_clean_block(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            pass

        assert w.success is True
        assert w.error_msg is None


# DB persistence                                                       

class TestWatchDBPersistence:

    def test_response_written_to_db(self, tmp_db):
        resp = make_response()
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1

    def test_db_record_matches_context_tokens(self, tmp_db):
        resp = make_response(prompt_tokens=15, completion_tokens=25)
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  == w.input_tokens  == 15
        assert log.output_tokens == w.output_tokens == 25

    def test_tag_propagated_to_db(self, tmp_db):
        resp = make_response()
        with watch(tag="my-tag", db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        assert query_logs(db_path=tmp_db)[0].tag == "my-tag"

    def test_user_id_propagated_to_db(self, tmp_db):
        resp = make_response()
        with watch(user_id="bob", db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        assert query_logs(db_path=tmp_db)[0].user_id == "bob"


# Multiple responses in one block                                      

class TestWatchMultipleResponses:

    def test_multiple_add_response_accumulates(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            w.add_response(make_response(prompt_tokens=10, completion_tokens=10), db_path=tmp_db)
            w.add_response(make_response(prompt_tokens=20, completion_tokens=20), db_path=tmp_db)

        assert w.input_tokens  == 30
        assert w.output_tokens == 30
        assert w.tokens_used   == 60
        assert w.call_count    == 2

    def test_multiple_responses_all_written_to_db(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            for _ in range(3):
                w.add_response(make_response(), db_path=tmp_db)

        assert len(query_logs(db_path=tmp_db)) == 3

    def test_cost_accumulates_across_responses(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            w.add_response(
                make_response(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
                db_path=tmp_db
            )
            w.add_response(
                make_response(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
                db_path=tmp_db
            )

        single_cost = w.cost_usd / 2
        assert w.cost_usd > single_cost   # accumulated > one call


# Exception handling                                                   

class TestWatchExceptionHandling:

    def test_exception_propagates(self, tmp_db):
        with pytest.raises(RuntimeError, match="exploded"):
            with watch(db_path=tmp_db) as w:
                raise RuntimeError("exploded")

    def test_success_false_on_exception(self, tmp_db):
        with pytest.raises(Exception):
            with watch(db_path=tmp_db) as w:
                raise Exception("fail")

        assert w.success is False

    def test_error_msg_stored_on_exception(self, tmp_db):
        with pytest.raises(Exception):
            with watch(db_path=tmp_db) as w:
                raise ValueError("bad input")

        assert "bad input" in w.error_msg

    def test_failure_record_written_to_db_when_no_responses(self, tmp_db):
        with pytest.raises(Exception):
            with watch(db_path=tmp_db):
                raise RuntimeError("crash")

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].success is False

    def test_latency_recorded_even_on_exception(self, tmp_db):
        with pytest.raises(Exception):
            with watch(db_path=tmp_db) as w:
                raise Exception("fail")

        assert w.latency_ms > 0

    def test_no_spurious_db_entry_on_clean_empty_block(self, tmp_db):
        """An empty block with no exception must NOT write a record."""
        with watch(db_path=tmp_db):
            pass

        assert len(query_logs(db_path=tmp_db)) == 0


# Metadata                                                             

class TestWatchMetadata:

    def test_all_metadata_forwarded(self, tmp_db):
        resp = make_response()
        with watch(tag="t", user_id="u", session_id="s", db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        log = query_logs(db_path=tmp_db)[0]
        assert log.tag        == "t"
        assert log.user_id    == "u"
        assert log.session_id == "s"

    def test_no_metadata_defaults_none(self, tmp_db):
        resp = make_response()
        with watch(db_path=tmp_db) as w:
            w.add_response(resp, db_path=tmp_db)

        log = query_logs(db_path=tmp_db)[0]
        assert log.tag is None
        assert log.user_id is None
        assert log.session_id is None


# Unknown provider                                                     

class TestWatchUnknownProvider:

    def test_unknown_response_logged_gracefully(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            w.add_response({"not": "a response"}, db_path=tmp_db)

        assert w.call_count == 1
        assert w.cost_usd == 0.0

    def test_unknown_response_in_db(self, tmp_db):
        with watch(db_path=tmp_db) as w:
            w.add_response("raw string", db_path=tmp_db)

        log = query_logs(db_path=tmp_db)[0]
        assert log.provider == "unknown"