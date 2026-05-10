# tests/unit/test_retry.py
"""
Tests for core/retry.py — with_retry() and @watchdog(retry=N).
No real LLM calls, no sleep (delays mocked to 0).
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from infertrack.core.retry import with_retry, _compute_delay
from infertrack.core.decorator import watchdog
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
    resp.choices[0].message.content = "Hello"
    return resp


# _compute_delay                                                       

class TestComputeDelay:

    def test_exponential_doubles(self):
        assert _compute_delay(0, "exponential", 1.0, 60.0) == 1.0
        assert _compute_delay(1, "exponential", 1.0, 60.0) == 2.0
        assert _compute_delay(2, "exponential", 1.0, 60.0) == 4.0
        assert _compute_delay(3, "exponential", 1.0, 60.0) == 8.0

    def test_linear_increments(self):
        assert _compute_delay(0, "linear", 1.0, 60.0) == 1.0
        assert _compute_delay(1, "linear", 1.0, 60.0) == 2.0
        assert _compute_delay(2, "linear", 1.0, 60.0) == 3.0

    def test_fixed_constant(self):
        assert _compute_delay(0, "fixed", 2.0, 60.0) == 2.0
        assert _compute_delay(5, "fixed", 2.0, 60.0) == 2.0

    def test_max_delay_capped(self):
        assert _compute_delay(10, "exponential", 1.0, 5.0) == 5.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown backoff"):
            _compute_delay(0, "random", 1.0, 60.0)

    def test_base_delay_respected(self):
        assert _compute_delay(0, "exponential", 2.0, 60.0) == 2.0
        assert _compute_delay(1, "exponential", 2.0, 60.0) == 4.0


# with_retry — success cases                                           

class TestWithRetrySuccess:

    def test_succeeds_on_first_try(self):
        fn = MagicMock(return_value="ok")
        result = with_retry(fn, retries=3)
        assert result == "ok"
        assert fn.call_count == 1

    def test_succeeds_after_one_failure(self):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        with patch("time.sleep"):
            result = with_retry(fn, retries=3, base_delay=0.0)
        assert result == "ok"
        assert fn.call_count == 2

    def test_succeeds_after_two_failures(self):
        fn = MagicMock(side_effect=[
            ConnectionError("err"), ConnectionError("err"), "ok"
        ])
        with patch("time.sleep"):
            result = with_retry(fn, retries=3, base_delay=0.0)
        assert result == "ok"
        assert fn.call_count == 3

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        with_retry(fn, args=(1, 2), kwargs={"key": "val"}, retries=0)
        fn.assert_called_once_with(1, 2, key="val")

    def test_zero_retries_calls_once(self):
        fn = MagicMock(return_value="done")
        result = with_retry(fn, retries=0)
        assert result == "done"
        assert fn.call_count == 1


# with_retry — failure cases                                           

class TestWithRetryFailure:

    def test_raises_after_all_retries_exhausted(self):
        fn = MagicMock(side_effect=RuntimeError("always fails"))
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="always fails"):
                with_retry(fn, retries=3, base_delay=0.0)
        assert fn.call_count == 4  # 1 original + 3 retries

    def test_raises_last_exception(self):
        fn = MagicMock(side_effect=[
            ValueError("first"),
            TypeError("second"),
            KeyError("third"),
        ])
        with patch("time.sleep"):
            with pytest.raises(KeyError, match="third"):
                with_retry(fn, retries=2, base_delay=0.0)

    def test_zero_retries_raises_immediately(self):
        fn = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            with_retry(fn, retries=0)
        assert fn.call_count == 1

    def test_negative_retries_raises_value_error(self):
        with pytest.raises(ValueError, match="retries must be"):
            with_retry(MagicMock(), retries=-1)


# with_retry — no-retry exceptions                                     

class TestWithRetryNoRetryExceptions:

    def test_keyboard_interrupt_not_retried(self):
        fn = MagicMock(side_effect=KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            with_retry(fn, retries=3)
        assert fn.call_count == 1

    def test_system_exit_not_retried(self):
        fn = MagicMock(side_effect=SystemExit(1))
        with pytest.raises(SystemExit):
            with_retry(fn, retries=3)
        assert fn.call_count == 1

    def test_retry_on_whitelist_respected(self):
        """Only retry ConnectionError, not ValueError."""
        fn = MagicMock(side_effect=ValueError("not retried"))
        with pytest.raises(ValueError):
            with patch("time.sleep"):
                with_retry(fn, retries=3, retry_on=[ConnectionError],
                           base_delay=0.0)
        assert fn.call_count == 1

    def test_retry_on_matching_exception_retried(self):
        fn = MagicMock(side_effect=[
            ConnectionError("retry me"), "ok"
        ])
        with patch("time.sleep"):
            result = with_retry(fn, retries=3,
                                retry_on=[ConnectionError], base_delay=0.0)
        assert result == "ok"


# with_retry — callbacks and delays                                    

class TestWithRetryCallbacks:

    def test_on_retry_called_with_correct_args(self):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        callback = MagicMock()
        with patch("time.sleep"):
            with_retry(fn, retries=3, on_retry=callback, base_delay=1.0)
        # Called once (first retry)
        assert callback.call_count == 1
        attempt, exc, delay = callback.call_args[0]
        assert attempt == 1
        assert isinstance(exc, ValueError)
        assert delay == 1.0  # exponential: base * 2^0 = 1.0

    def test_sleep_called_with_correct_delay(self):
        fn = MagicMock(side_effect=[RuntimeError(), "ok"])
        with patch("time.sleep") as mock_sleep:
            with_retry(fn, retries=3, backoff="fixed",
                       base_delay=2.5, on_retry=lambda *a: None)
        mock_sleep.assert_called_once_with(2.5)

    def test_exponential_sleep_sequence(self):
        fn = MagicMock(side_effect=[
            RuntimeError(), RuntimeError(), RuntimeError(), "ok"
        ])
        delays = []
        with patch("time.sleep", side_effect=lambda d: delays.append(d)):
            with_retry(fn, retries=3, backoff="exponential",
                       base_delay=1.0, on_retry=lambda *a: None)
        assert delays == [1.0, 2.0, 4.0]


# @watchdog(retry=N) integration                                       

class TestWatchdogRetryIntegration:

    def test_retry_0_no_retry_on_failure(self, tmp_db):
        @watchdog(retry=0, db_path=tmp_db)
        def ask():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            ask()
        assert len(query_logs(db_path=tmp_db)) == 1

    def test_retry_succeeds_eventually(self, tmp_db):
        resp = make_response()
        call_count = 0

        @watchdog(retry=3, base_delay=0.0, db_path=tmp_db)
        def ask():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return resp

        with patch("time.sleep"):
            result = ask()

        assert result is resp
        assert call_count == 3

    def test_retry_exhausted_logs_failure(self, tmp_db):
        @watchdog(retry=2, base_delay=0.0, db_path=tmp_db)
        def ask():
            raise RuntimeError("always fails")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                ask()

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].success is False
        assert "always fails" in logs[0].error_msg

    def test_retry_success_logs_correctly(self, tmp_db):
        resp = make_response(model="gpt-4o", prompt_tokens=100,
                             completion_tokens=50)
        attempts = 0

        @watchdog(retry=3, base_delay=0.0, db_path=tmp_db)
        def ask():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("retry")
            return resp

        with patch("time.sleep"):
            ask()

        log = query_logs(db_path=tmp_db)[0]
        assert log.success is True
        assert log.model == "gpt-4o"
        assert log.input_tokens == 100

    def test_backoff_exponential_accepted(self, tmp_db):
        @watchdog(retry=2, backoff="exponential", base_delay=0.0, db_path=tmp_db)
        def ask():
            raise ValueError("fail")

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                ask()

    def test_backoff_linear_accepted(self, tmp_db):
        @watchdog(retry=2, backoff="linear", base_delay=0.0, db_path=tmp_db)
        def ask():
            raise ValueError("fail")

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                ask()

    def test_no_retry_on_success(self, tmp_db):
        call_count = 0

        @watchdog(retry=5, db_path=tmp_db)
        def ask():
            nonlocal call_count
            call_count += 1
            return make_response()

        ask()
        assert call_count == 1