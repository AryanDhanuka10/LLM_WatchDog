# tests/unit/test_budget.py
"""
Day 4 tests: Budget context manager + BudgetExceeded exception.
All tests use mocked responses and a temp SQLite DB — no real LLM calls.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from llm_ledger.core.budget import Budget, BudgetContext
from llm_ledger.exceptions import BudgetExceeded
from llm_ledger.storage.db import init_db, query_logs, insert_log, get_total_cost
from llm_ledger.storage.models import CallLog


# Fixtures                                                             

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def make_response(model="qwen2.5:0.5b", prompt_tokens=10, completion_tokens=20):
    """Zero-cost Ollama response — safe for budget tests that don't want to trip."""
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = "Hi"
    return resp


def make_paid_response(model="gpt-4o", prompt_tokens=1000, completion_tokens=500):
    """Non-zero cost response for tests that need to accumulate spend."""
    return make_response(model=model,
                         prompt_tokens=prompt_tokens,
                         completion_tokens=completion_tokens)


# BudgetExceeded exception                                             

class TestBudgetExceededException:

    def test_attributes_stored(self):
        err = BudgetExceeded(spent=0.15, limit=0.10, user_id="alice")
        assert err.spent   == 0.15
        assert err.limit   == 0.10
        assert err.user_id == "alice"

    def test_str_contains_user_id(self):
        err = BudgetExceeded(spent=0.15, limit=0.10, user_id="alice")
        assert "alice" in str(err)

    def test_str_contains_amounts(self):
        err = BudgetExceeded(spent=0.15, limit=0.10, user_id="alice")
        assert "0.15" in str(err)
        assert "0.10" in str(err)

    def test_no_user_id_still_works(self):
        err = BudgetExceeded(spent=0.05, limit=0.01)
        assert err.user_id is None
        assert "session" in str(err)

    def test_is_watchdog_error(self):
        from llm_ledger.exceptions import WatchdogError
        assert isinstance(BudgetExceeded(spent=0.1, limit=0.05), WatchdogError)

    def test_is_exception(self):
        assert isinstance(BudgetExceeded(spent=0.1, limit=0.05), Exception)


# Budget — basic usage                                                 

class TestBudgetBasic:

    def test_yields_budget_context(self, tmp_db):
        with Budget(max_usd=1.0, db_path=tmp_db) as b:
            assert isinstance(b, BudgetContext)

    def test_free_calls_dont_raise(self, tmp_db):
        with Budget(max_usd=0.01, db_path=tmp_db) as b:
            for _ in range(5):
                b.add_response(make_response())   # Ollama = $0.00

    def test_spent_starts_at_zero_no_user(self, tmp_db):
        with Budget(max_usd=1.0, db_path=tmp_db) as b:
            assert b.spent_usd == 0.0

    def test_spent_accumulates(self, tmp_db):
        with Budget(max_usd=100.0, db_path=tmp_db) as b:
            b.add_response(make_paid_response(prompt_tokens=1000, completion_tokens=500))
            b.add_response(make_paid_response(prompt_tokens=1000, completion_tokens=500))
            assert b.spent_usd > 0.0
            assert b.call_count == 2

    def test_remaining_decreases(self, tmp_db):
        with Budget(max_usd=1.0, db_path=tmp_db) as b:
            before = b.remaining_usd
            b.add_response(make_paid_response())
            assert b.remaining_usd < before

    def test_calls_logged_to_db(self, tmp_db):
        with Budget(max_usd=100.0, db_path=tmp_db) as b:
            b.add_response(make_response())
            b.add_response(make_response())
        assert len(query_logs(db_path=tmp_db)) == 2

    def test_tag_forwarded_to_logs(self, tmp_db):
        with Budget(max_usd=100.0, tag="my-feature", db_path=tmp_db) as b:
            b.add_response(make_response())
        assert query_logs(db_path=tmp_db)[0].tag == "my-feature"

    def test_user_id_forwarded_to_logs(self, tmp_db):
        with Budget(max_usd=100.0, user_id="alice", db_path=tmp_db) as b:
            b.add_response(make_response())
        assert query_logs(db_path=tmp_db)[0].user_id == "alice"

    def test_session_id_forwarded_to_logs(self, tmp_db):
        with Budget(max_usd=100.0, session_id="sess_xyz", db_path=tmp_db) as b:
            b.add_response(make_response())
        assert query_logs(db_path=tmp_db)[0].session_id == "sess_xyz"


# Budget — enforcement                                                 

class TestBudgetEnforcement:

    def test_raises_budget_exceeded_when_over(self, tmp_db):
        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.001, db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))

    def test_exception_carries_spent(self, tmp_db):
        with pytest.raises(BudgetExceeded) as exc_info:
            with Budget(max_usd=0.001, db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))
        assert exc_info.value.spent > 0.001

    def test_exception_carries_limit(self, tmp_db):
        with pytest.raises(BudgetExceeded) as exc_info:
            with Budget(max_usd=0.001, db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))
        assert exc_info.value.limit == 0.001

    def test_exception_carries_user_id(self, tmp_db):
        with pytest.raises(BudgetExceeded) as exc_info:
            with Budget(max_usd=0.001, user_id="bob", db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))
        assert exc_info.value.user_id == "bob"

    def test_call_that_trips_budget_is_still_logged(self, tmp_db):
        """The call that pushes over budget must be in the DB."""
        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.001, db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))
        assert len(query_logs(db_path=tmp_db)) == 1

    def test_multiple_calls_trip_on_cumulative_spend(self, tmp_db):
        """Budget trips only after enough calls accumulate past the limit."""
        # Each gpt-4o call at 500/250 tokens ≈ $0.0062
        # With limit $0.01, first call passes, second trips
        tripped_on = None
        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.01, db_path=tmp_db) as b:
                for i in range(10):
                    b.add_response(make_paid_response(
                        prompt_tokens=500, completion_tokens=250
                    ))
                    tripped_on = i

        # Must have gotten through at least one call before tripping
        assert tripped_on is not None
        assert tripped_on >= 0

    def test_free_calls_never_trip_budget(self, tmp_db):
        """Ollama / free model calls must never trigger BudgetExceeded."""
        with Budget(max_usd=0.0001, db_path=tmp_db) as b:
            for _ in range(100):
                b.add_response(make_response())   # $0.00 each
        # No exception raised


# Budget — prior spend (user_id scoping)                               

class TestBudgetPriorSpend:

    def test_prior_spend_counted_for_user(self, tmp_db):
        """Pre-existing spend in DB for same user_id counts against budget."""
        # Insert a historical log that puts alice at $0.09
        existing = CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=5000, output_tokens=1000,
            cost_usd=0.09, latency_ms=100.0,
            success=True, user_id="alice",
        )
        insert_log(existing, db_path=tmp_db)

        # Budget of $0.10 — alice already spent $0.09, so almost at limit
        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.10, user_id="alice",
                        period="all", db_path=tmp_db) as b:
                # Even a small paid call should push her over
                b.add_response(make_paid_response(
                    prompt_tokens=1000, completion_tokens=500
                ))

    def test_prior_spend_starts_on_context_entry(self, tmp_db):
        """spent_usd on entry reflects pre-existing DB spend for the user."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=500,
            cost_usd=0.05, latency_ms=50.0,
            success=True, user_id="carol",
        ), db_path=tmp_db)

        with Budget(max_usd=1.0, user_id="carol",
                    period="all", db_path=tmp_db) as b:
            assert abs(b.spent_usd - 0.05) < 1e-9

    def test_different_users_dont_share_budgets(self, tmp_db):
        """alice's spend must not count against bob's budget."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=10000, output_tokens=5000,
            cost_usd=0.50, latency_ms=100.0,
            success=True, user_id="alice",
        ), db_path=tmp_db)

        # bob has a fresh $0.10 budget — alice's spend must not affect him
        with Budget(max_usd=0.10, user_id="bob",
                    period="all", db_path=tmp_db) as b:
            assert b.spent_usd == 0.0

    def test_no_user_id_ignores_prior_spend(self, tmp_db):
        """When user_id is None, prior DB spend is not counted."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=10000, output_tokens=5000,
            cost_usd=0.50, latency_ms=100.0,
            success=True, user_id="alice",
        ), db_path=tmp_db)

        with Budget(max_usd=0.10, db_path=tmp_db) as b:
            assert b.spent_usd == 0.0

    def test_already_over_budget_raises_immediately(self, tmp_db):
        """If prior spend already exceeds limit, raises before yield."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=10000, output_tokens=5000,
            cost_usd=0.20, latency_ms=100.0,
            success=True, user_id="dave",
        ), db_path=tmp_db)

        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.10, user_id="dave",
                        period="all", db_path=tmp_db):
                pass   # should never reach here


# Budget — period parameter                                            

class TestBudgetPeriod:

    def test_period_session_ignores_prior_spend(self, tmp_db):
        """period='session' starts from 0 regardless of DB history."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=10000, output_tokens=5000,
            cost_usd=0.99, latency_ms=100.0,
            success=True, user_id="eve",
        ), db_path=tmp_db)

        # Even with $0.99 in the DB, session budget starts fresh
        with Budget(max_usd=0.10, user_id="eve",
                    period="session", db_path=tmp_db) as b:
            assert b.spent_usd == 0.0

    def test_invalid_period_raises_value_error(self, tmp_db):
        with pytest.raises(ValueError, match="period"):
            with Budget(max_usd=1.0, period="weekly", db_path=tmp_db):
                pass

    def test_period_all_includes_all_history(self, tmp_db):
        """period='all' should pick up all historical spend."""
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=500,
            cost_usd=0.07, latency_ms=100.0,
            success=True, user_id="frank",
        ), db_path=tmp_db)

        with Budget(max_usd=1.0, user_id="frank",
                    period="all", db_path=tmp_db) as b:
            assert abs(b.spent_usd - 0.07) < 1e-9


# Budget — validation                                                  

class TestBudgetValidation:

    def test_zero_max_usd_raises(self, tmp_db):
        with pytest.raises(ValueError):
            with Budget(max_usd=0.0, db_path=tmp_db):
                pass

    def test_negative_max_usd_raises(self, tmp_db):
        with pytest.raises(ValueError):
            with Budget(max_usd=-1.0, db_path=tmp_db):
                pass


# BudgetContext — properties                                           

class TestBudgetContextProperties:

    def test_remaining_usd_full_at_start(self, tmp_db):
        with Budget(max_usd=0.50, db_path=tmp_db) as b:
            assert abs(b.remaining_usd - 0.50) < 1e-9

    def test_remaining_usd_zero_when_exhausted(self, tmp_db):
        with pytest.raises(BudgetExceeded):
            with Budget(max_usd=0.001, db_path=tmp_db) as b:
                b.add_response(make_paid_response(
                    prompt_tokens=10_000, completion_tokens=10_000
                ))
        assert b.remaining_usd == 0.0

    def test_is_over_budget_false_initially(self, tmp_db):
        with Budget(max_usd=1.0, db_path=tmp_db) as b:
            assert b.is_over_budget is False

    def test_call_count_increments(self, tmp_db):
        with Budget(max_usd=100.0, db_path=tmp_db) as b:
            assert b.call_count == 0
            b.add_response(make_response())
            assert b.call_count == 1
            b.add_response(make_response())
            assert b.call_count == 2