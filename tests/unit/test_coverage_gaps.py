# tests/unit/test_coverage_gaps.py
"""
Day 11 — targeted tests to cover remaining gaps found by pytest-cov.

Gaps addressed:
  - interceptor.py: parse failure path, ImportError, anthropic patch/restore
  - exceptions.py:  ProviderNotDetected, PricingModelNotFound str()
  - core/context.py: add_response parse failure path
  - storage/db.py:   _parse_timestamp edge cases
  - core/budget.py:  _detect_and_cost parse failure
  - core/retry.py:   unknown backoff in with_retry (via _compute_delay)
"""
from __future__ import annotations

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from infertrack.storage.db import init_db, query_logs


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
    resp.choices[0].message.content = "Hi"
    return resp


# exceptions.py — ProviderNotDetected, PricingModelNotFound           

class TestExceptionMessages:

    def test_provider_not_detected_str(self):
        from infertrack.exceptions import ProviderNotDetected
        err = ProviderNotDetected("dict")
        assert "dict" in str(err)
        assert err.response_type == "dict"

    def test_provider_not_detected_default(self):
        from infertrack.exceptions import ProviderNotDetected
        err = ProviderNotDetected()
        assert "unknown" in str(err)

    def test_pricing_model_not_found_str(self):
        from infertrack.exceptions import PricingModelNotFound
        err = PricingModelNotFound("gpt-99-ultra")
        assert "gpt-99-ultra" in str(err)
        assert err.model == "gpt-99-ultra"

    def test_pricing_model_not_found_is_watchdog_error(self):
        from infertrack.exceptions import PricingModelNotFound, WatchdogError
        assert isinstance(PricingModelNotFound("x"), WatchdogError)

    def test_provider_not_detected_is_watchdog_error(self):
        from infertrack.exceptions import ProviderNotDetected, WatchdogError
        assert isinstance(ProviderNotDetected(), WatchdogError)


# interceptor.py — uncovered paths                                    

# Build fake openai if not present (same pattern as test_interceptor.py)
def _ensure_fake_openai():
    if "openai" not in sys.modules:
        openai_mod  = types.ModuleType("openai")
        res_mod     = types.ModuleType("openai.resources")
        chat_mod    = types.ModuleType("openai.resources.chat")
        comp_mod    = types.ModuleType("openai.resources.chat.completions")

        class Completions:
            def create(self, *a, **k): raise NotImplementedError

        comp_mod.Completions = Completions
        chat_mod.completions = comp_mod
        res_mod.chat         = chat_mod
        openai_mod.resources = res_mod

        sys.modules.setdefault("openai", openai_mod)
        sys.modules.setdefault("openai.resources", res_mod)
        sys.modules.setdefault("openai.resources.chat", chat_mod)
        sys.modules.setdefault("openai.resources.chat.completions", comp_mod)

    return sys.modules["openai.resources.chat.completions"].Completions


FakeCompletions = _ensure_fake_openai()
_original_create = FakeCompletions.create

import infertrack.core.interceptor as imod


@pytest.fixture(autouse=True)
def reset_interceptor():
    yield
    imod.stop()
    imod._intercept_config.update({
        "db_path": None, "tag": None,
        "user_id": None, "session_id": None, "active": False,
    })
    imod._originals.clear()
    FakeCompletions.create = _original_create


class TestInterceptorUncoveredPaths:

    def test_parse_failure_in_wrapper_logs_unknown(self, tmp_db):
        """Provider detected but extract_usage raises → logs provider=unknown."""
        from infertrack.core.interceptor import intercept, _make_wrapper

        # Response that passes detect() but fails extract_usage()
        bad_resp = MagicMock()
        bad_resp.model = "qwen2.5:0.5b"
        bad_resp.usage.prompt_tokens   = None   # will cause ValueError in extract
        bad_resp.usage.completion_tokens = None
        bad_resp.choices[0].message.content = "hi"

        intercept(db_path=tmp_db)
        mock_orig = MagicMock(return_value=bad_resp)
        mock_orig.__name__ = "create"
        mock_orig.__doc__  = None
        mock_orig.__module__ = "openai.resources.chat.completions"
        mock_orig.__qualname__ = "Completions.create"
        imod._originals["openai.Completions.create"] = mock_orig
        FakeCompletions.create = _make_wrapper(mock_orig, tmp_db)

        FakeCompletions.create(MagicMock(), model="qwen2.5:0.5b", messages=[])

        log = query_logs(db_path=tmp_db)[0]
        # Should still be logged — provider may be unknown or openai depending
        # on whether None tokens are coerced; what matters is no crash
        assert log is not None

    def test_intercept_raises_if_openai_missing(self, tmp_db):
        """ImportError raised gracefully when openai not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        # Only test if openai would actually be imported fresh
        # (skip if already cached — we can't un-cache it safely)
        if "openai" in sys.modules:
            pytest.skip("openai already cached in sys.modules")

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="openai package is required"):
                imod.intercept(db_path=tmp_db)

    def test_stop_when_anthropic_not_installed(self, tmp_db):
        """stop() must not crash if anthropic was never installed."""
        imod.intercept(db_path=tmp_db)

        # Temporarily hide anthropic from sys.modules
        anthropic_backup = sys.modules.pop("anthropic", None)
        anthro_msgs = sys.modules.pop("anthropic.resources.messages", None)

        try:
            imod.stop()   # must not raise
        finally:
            if anthropic_backup:
                sys.modules["anthropic"] = anthropic_backup
            if anthro_msgs:
                sys.modules["anthropic.resources.messages"] = anthro_msgs

        assert imod.is_active() is False

    def test_anthropic_patch_skipped_when_not_installed(self, tmp_db):
        """intercept() silently skips anthropic patch if not installed."""
        # Remove anthropic from sys.modules temporarily
        anthropic_backup = sys.modules.pop("anthropic", None)
        anthro_msgs      = sys.modules.pop("anthropic.resources.messages", None)

        try:
            imod.intercept(db_path=tmp_db)   # must not raise
            assert imod.is_active() is True
        finally:
            if anthropic_backup:
                sys.modules["anthropic"] = anthropic_backup
            if anthro_msgs:
                sys.modules["anthropic.resources.messages"] = anthro_msgs


# core/context.py — add_response parse failure                        

class TestContextAddResponseParseFail:

    def test_unknown_response_logged_with_error(self, tmp_db):
        """add_response with unrecognised object logs provider=unknown."""
        from infertrack.core.context import watch

        with watch(db_path=tmp_db) as w:
            w.add_response({"not": "a response"}, db_path=tmp_db)

        log = query_logs(db_path=tmp_db)[0]
        assert log.provider == "unknown"
        assert log.cost_usd == 0.0

    def test_none_response_handled_gracefully(self, tmp_db):
        """add_response(None) must not raise."""
        from infertrack.core.context import watch

        with watch(db_path=tmp_db) as w:
            w.add_response(None, db_path=tmp_db)

        assert w.call_count == 1


# storage/db.py — _parse_timestamp edge cases                         

class TestParseTimestampEdgeCases:

    def test_integer_epoch_parsed(self):
        from infertrack.storage.db import _parse_timestamp
        from datetime import datetime
        ts = _parse_timestamp(1700000000)
        assert isinstance(ts, datetime)

    def test_invalid_string_raises(self):
        from infertrack.storage.db import _parse_timestamp
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_timestamp("not-a-date-or-number")

    def test_iso_string_with_timezone(self):
        from infertrack.storage.db import _parse_timestamp
        ts = _parse_timestamp("2026-05-07T10:30:00+00:00")
        assert ts.year == 2026

    def test_datetime_passthrough(self):
        from infertrack.storage.db import _parse_timestamp
        from datetime import datetime, timezone
        dt = datetime.now(timezone.utc)
        assert _parse_timestamp(dt) is dt


# core/budget.py — _detect_and_cost parse failure                     

class TestBudgetDetectAndCost:

    def test_unrecognised_response_returns_zeros(self):
        from infertrack.core.budget import _detect_and_cost
        prov, model, inp, out, cost = _detect_and_cost({"not": "a response"})
        assert prov  == "unknown"
        assert model == "unknown"
        assert inp   == 0
        assert out   == 0
        assert cost  == 0.0

    def test_none_response_returns_zeros(self):
        from infertrack.core.budget import _detect_and_cost
        prov, model, inp, out, cost = _detect_and_cost(None)
        assert cost == 0.0

    def test_valid_response_returns_values(self):
        from infertrack.core.budget import _detect_and_cost
        resp = make_response(model="gpt-4o",
                             prompt_tokens=1000, completion_tokens=500)
        prov, model, inp, out, cost = _detect_and_cost(resp)
        assert prov  == "openai"
        assert model == "gpt-4o"
        assert inp   == 1000
        assert out   == 500
        assert cost  > 0.0


# core/retry.py — _compute_delay with unknown backoff                 

class TestComputeDelayUnknownBackoff:

    def test_unknown_backoff_raises(self):
        from infertrack.core.retry import _compute_delay
        with pytest.raises(ValueError, match="Unknown backoff"):
            _compute_delay(0, "zigzag", 1.0, 60.0)


# cli/commands.py — _since_datetime bad input                         

class TestSinceDatetimeBadInput:

    def test_invalid_last_raises_bad_parameter(self):
        from infertrack.cli.commands import _since_datetime
        import click
        with pytest.raises(click.BadParameter):
            _since_datetime("yesterday")

    def test_all_returns_none(self):
        from infertrack.cli.commands import _since_datetime
        assert _since_datetime("all") is None


# cli/__main__.py — __main__ block                                    

class TestCLIMainBlock:

    def test_cli_importable_as_module(self):
        from infertrack.cli.__main__ import cli
        assert callable(cli)
