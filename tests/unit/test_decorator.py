# tests/unit/test_decorator.py
"""
Day 3 tests: @watchdog decorator.
All tests use mocked response objects and a temp SQLite DB — no real LLM calls.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_ledger.core.decorator import watchdog
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


# Basic logging                                                        

class TestWatchdogBasicLogging:

    def test_call_logged_to_db(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1

    def test_tokens_recorded_correctly(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response(prompt_tokens=10, completion_tokens=20)

        ask()
        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens == 10
        assert log.output_tokens == 20

    def test_model_recorded(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response(model="gpt-4o")

        ask()
        assert query_logs(db_path=tmp_db)[0].model == "gpt-4o"

    def test_provider_recorded_as_openai(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].provider == "openai"

    def test_success_true_on_clean_call(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].success is True

    def test_latency_recorded_and_positive(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].latency_ms > 0

    def test_ollama_cost_is_zero(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response(model="qwen2.5:0.5b")

        ask()
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0

    def test_gpt4o_cost_is_positive(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)

        ask()
        assert query_logs(db_path=tmp_db)[0].cost_usd > 0.0


# Metadata tags                                                        

class TestWatchdogMetadata:

    def test_tag_stored(self, tmp_db):
        @watchdog(tag="my-feature", db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].tag == "my-feature"

    def test_user_id_stored(self, tmp_db):
        @watchdog(user_id="alice", db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].user_id == "alice"

    def test_session_id_stored(self, tmp_db):
        @watchdog(session_id="sess_123", db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        assert query_logs(db_path=tmp_db)[0].session_id == "sess_123"

    def test_no_metadata_defaults_to_none(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        log = query_logs(db_path=tmp_db)[0]
        assert log.tag is None
        assert log.user_id is None
        assert log.session_id is None


# Return value passthrough                                             

class TestWatchdogReturnValue:

    def test_returns_original_response(self, tmp_db):
        resp = make_response()

        @watchdog(db_path=tmp_db)
        def ask():
            return resp

        result = ask()
        assert result is resp

    def test_passes_args_through(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt: str, *, model: str = "qwen2.5:0.5b"):
            return make_response(model=model)

        result = ask("hello", model="gpt-4o")
        assert result.model == "gpt-4o"

    def test_preserves_function_name(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def my_special_function():
            return make_response()

        assert my_special_function.__name__ == "my_special_function"

    def test_preserves_docstring(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            """My docstring."""
            return make_response()

        assert ask.__doc__ == "My docstring."


# Exception handling                                                   

class TestWatchdogExceptionHandling:

    def test_exception_is_reraised(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            ask()

    def test_failed_call_logged(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            raise RuntimeError("api down")

        with pytest.raises(RuntimeError):
            ask()

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].success is False

    def test_error_message_stored(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            raise ConnectionError("timeout")

        with pytest.raises(ConnectionError):
            ask()

        assert "timeout" in query_logs(db_path=tmp_db)[0].error_msg

    def test_tokens_zero_on_failure(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            raise Exception("fail")

        with pytest.raises(Exception):
            ask()

        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens == 0
        assert log.output_tokens == 0
        assert log.cost_usd == 0.0

    def test_latency_recorded_even_on_failure(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            raise Exception("fail")

        with pytest.raises(Exception):
            ask()

        assert query_logs(db_path=tmp_db)[0].latency_ms > 0


# Unknown / unrecognised response                                      

class TestWatchdogUnknownProvider:

    def test_unknown_response_still_logged(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return {"just": "a dict"}   # not a real API response

        ask()
        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].provider == "unknown"
        assert logs[0].model == "unknown"

    def test_unknown_response_zero_cost(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return "raw string"

        ask()
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0


# Multiple calls                                                       

class TestWatchdogMultipleCalls:

    def test_each_call_creates_separate_log(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask():
            return make_response()

        ask()
        ask()
        ask()
        assert len(query_logs(db_path=tmp_db)) == 3

    def test_different_models_logged_separately(self, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(model):
            return make_response(model=model)

        ask("gpt-4o")
        ask("qwen2.5:0.5b")

        logs = query_logs(db_path=tmp_db)
        models = {log.model for log in logs}
        assert models == {"gpt-4o", "qwen2.5:0.5b"}