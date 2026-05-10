# tests/unit/test_interceptor.py
"""
Day 7 tests: global interceptor via infertrack.intercept() / stop().
All tests mock the openai module entirely — no real openai install needed.
"""
from __future__ import annotations

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from infertrack.storage.db import init_db, query_logs


# Build a minimal fake openai module tree before any test runs         #

def _build_fake_openai():
    """Construct a fake openai package in sys.modules.

    Structure mirrored from real openai>=1.0:
      openai
      openai.resources
      openai.resources.chat
      openai.resources.chat.completions   → Completions class with .create
    """
    # Root
    openai_mod = types.ModuleType("openai")

    # openai.resources
    resources_mod = types.ModuleType("openai.resources")
    openai_mod.resources = resources_mod

    # openai.resources.chat
    chat_mod = types.ModuleType("openai.resources.chat")
    resources_mod.chat = chat_mod

    # openai.resources.chat.completions
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError("real create — should be mocked in tests")

    completions_mod.Completions = Completions
    chat_mod.completions = completions_mod

    sys.modules.setdefault("openai", openai_mod)
    sys.modules.setdefault("openai.resources", resources_mod)
    sys.modules.setdefault("openai.resources.chat", chat_mod)
    sys.modules.setdefault("openai.resources.chat.completions", completions_mod)

    return completions_mod.Completions


# Install fake openai BEFORE importing interceptor so it resolves cleanly
FakeCompletions = _build_fake_openai()

# Now safe to import
from infertrack.core.interceptor import intercept, stop, is_active, _make_wrapper
import infertrack.core.interceptor as imod


# Fixtures                                                             

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture(autouse=True)
def always_stop():
    """Guarantee interceptor is fully reset after every test."""
    yield
    stop()
    # Reset _intercept_config manually in case stop() failed
    imod._intercept_config.update({
        "db_path": None, "tag": None,
        "user_id": None, "session_id": None, "active": False,
    })
    imod._originals.clear()
    # Restore Completions.create to original if it was patched
    FakeCompletions.create = _original_create


# Capture the original method once at module load
_original_create = FakeCompletions.create


def make_response(model="qwen2.5:0.5b", prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = "Hello"
    return resp


def _do_intercept(tmp_db, **kwargs):
    """Call intercept() and ensure FakeCompletions is targeted."""
    intercept(db_path=tmp_db, **kwargs)


def _invoke_wrapper(tmp_db, response=None, side_effect=None, **intercept_kwargs):
    """
    Full round-trip helper:
      1. intercept()
      2. Replace stored original with a mock returning `response`
      3. Re-wrap Completions.create with the mock original
      4. Invoke the wrapper
      5. Return the result (or raise if side_effect set)
    """
    _do_intercept(tmp_db, **intercept_kwargs)

    mock_original = MagicMock()
    if side_effect is not None:
        mock_original.side_effect = side_effect
    elif response is not None:
        mock_original.return_value = response
    mock_original.__name__ = "create"
    mock_original.__doc__ = None
    mock_original.__module__ = "openai.resources.chat.completions"
    mock_original.__qualname__ = "Completions.create"
    mock_original.__wrapped__ = None

    imod._originals["openai.Completions.create"] = mock_original
    FakeCompletions.create = _make_wrapper(mock_original, tmp_db)

    fake_self = MagicMock()
    return FakeCompletions.create(fake_self, model="qwen2.5:0.5b", messages=[])


# is_active / basic state                                              

class TestInterceptorState:

    def test_not_active_by_default(self):
        assert is_active() is False

    def test_active_after_intercept(self, tmp_db):
        _do_intercept(tmp_db)
        assert is_active() is True

    def test_inactive_after_stop(self, tmp_db):
        _do_intercept(tmp_db)
        stop()
        assert is_active() is False

    def test_stop_is_idempotent(self):
        stop()
        stop()
        assert is_active() is False

    def test_double_intercept_raises(self, tmp_db):
        _do_intercept(tmp_db)
        with pytest.raises(RuntimeError, match="already active"):
            _do_intercept(tmp_db)


# Patching behaviour                                                   

class TestInterceptorPatching:

    def _live_completions(self):
        """Always read Completions fresh from sys.modules to avoid
        cross-module class identity issues when multiple test files
        each register a fake openai module."""
        import sys
        return sys.modules["openai.resources.chat.completions"].Completions

    def test_intercept_replaces_completions_create(self, tmp_db):
        """intercept() must store the original and replace it."""
        _do_intercept(tmp_db)
        assert "openai.Completions.create" in imod._originals
        live = self._live_completions()
        # Current method on the live class must differ from the stored original
        assert live.create is not imod._originals["openai.Completions.create"]

    def test_stop_restores_completions_create(self, tmp_db):
        """stop() must restore Completions.create to exactly the original."""
        _do_intercept(tmp_db)
        stored_original = imod._originals["openai.Completions.create"]
        stop()
        live = self._live_completions()
        assert live.create is stored_original

    def test_wrapper_preserves_function_name(self, tmp_db):
        _do_intercept(tmp_db)
        live = self._live_completions()
        assert live.create.__name__ == "create"


# Logging behaviour                                                    

class TestInterceptorLogging:

    def test_successful_call_logged(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response())
        assert len(query_logs(db_path=tmp_db)) == 1

    def test_tokens_recorded(self, tmp_db):
        _invoke_wrapper(tmp_db,
                        response=make_response(prompt_tokens=15,
                                               completion_tokens=25))
        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  == 15
        assert log.output_tokens == 25

    def test_model_recorded(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response(model="gpt-4o"))
        assert query_logs(db_path=tmp_db)[0].model == "gpt-4o"

    def test_ollama_cost_zero(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response(model="qwen2.5:0.5b"))
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0

    def test_gpt4o_cost_positive(self, tmp_db):
        _invoke_wrapper(tmp_db,
                        response=make_response(model="gpt-4o",
                                               prompt_tokens=1000,
                                               completion_tokens=500))
        assert query_logs(db_path=tmp_db)[0].cost_usd > 0.0

    def test_latency_non_negative(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response())
        assert query_logs(db_path=tmp_db)[0].latency_ms >= 0

    def test_success_true(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response())
        assert query_logs(db_path=tmp_db)[0].success is True

    def test_tag_stored(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response(), tag="my-tag")
        assert query_logs(db_path=tmp_db)[0].tag == "my-tag"

    def test_user_id_stored(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response(), user_id="alice")
        assert query_logs(db_path=tmp_db)[0].user_id == "alice"

    def test_session_id_stored(self, tmp_db):
        _invoke_wrapper(tmp_db, response=make_response(), session_id="sess_1")
        assert query_logs(db_path=tmp_db)[0].session_id == "sess_1"

    def test_multiple_calls_all_logged(self, tmp_db):
        _do_intercept(tmp_db)
        mock_orig = MagicMock(return_value=make_response())
        mock_orig.__name__ = "create"
        mock_orig.__doc__ = None
        mock_orig.__module__ = "openai.resources.chat.completions"
        mock_orig.__qualname__ = "Completions.create"
        imod._originals["openai.Completions.create"] = mock_orig
        FakeCompletions.create = _make_wrapper(mock_orig, tmp_db)

        fake_self = MagicMock()
        for _ in range(3):
            FakeCompletions.create(fake_self, model="qwen2.5:0.5b", messages=[])

        assert len(query_logs(db_path=tmp_db)) == 3


# Exception handling                                                   

class TestInterceptorExceptions:

    def test_exception_reraised(self, tmp_db):
        with pytest.raises(ConnectionError, match="API down"):
            _invoke_wrapper(tmp_db, side_effect=ConnectionError("API down"))

    def test_exception_logged(self, tmp_db):
        with pytest.raises(ConnectionError):
            _invoke_wrapper(tmp_db, side_effect=ConnectionError("API down"))
        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].success is False

    def test_error_message_stored(self, tmp_db):
        with pytest.raises(ConnectionError):
            _invoke_wrapper(tmp_db, side_effect=ConnectionError("API down"))
        assert "API down" in query_logs(db_path=tmp_db)[0].error_msg

    def test_failed_call_zero_tokens(self, tmp_db):
        with pytest.raises(RuntimeError):
            _invoke_wrapper(tmp_db, side_effect=RuntimeError("timeout"))
        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  == 0
        assert log.output_tokens == 0
        assert log.cost_usd      == 0.0

    def test_failed_call_latency_recorded(self, tmp_db):
        with pytest.raises(Exception):
            _invoke_wrapper(tmp_db, side_effect=Exception("fail"))
        assert query_logs(db_path=tmp_db)[0].latency_ms >= 0


# Public API exports                                                   

class TestTopLevelExports:

    def test_intercept_importable_from_core(self):
        from infertrack.core.interceptor import intercept, stop, is_active
        assert callable(intercept)
        assert callable(stop)
        assert callable(is_active)

    def test_make_wrapper_is_callable(self):
        from infertrack.core.interceptor import _make_wrapper
        assert callable(_make_wrapper)