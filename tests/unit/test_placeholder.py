"""
Placeholder test file — replaced day by day.
This file just verifies the package is importable before Day 1.
"""

def test_package_importable():
    import infertrack
    assert infertrack.__version__ == "1.0.3"

def test_config_importable():
    from infertrack.config import config, configure, reset_config
    assert config.default_tag == "default"

def test_exceptions_importable():
    from infertrack.exceptions import (
        WatchdogError,
        BudgetExceeded,
        ProviderNotDetected,
        PricingModelNotFound,
    )

def test_budget_exceeded_message():
    from infertrack.exceptions import BudgetExceeded
    err = BudgetExceeded(spent=0.15, limit=0.10, user_id="user_42")
    assert err.spent == 0.15
    assert err.limit == 0.10
    assert err.user_id == "user_42"
    assert "user_42" in str(err)

def test_calllog_importable():
    from infertrack.storage.models import CallLog
    log = CallLog()
    assert log.provider == "unknown"
    assert log.success is True
    assert log.cost_usd == 0.0
    assert log.id is None

def test_calllog_total_tokens_auto():
    from infertrack.storage.models import CallLog
    log = CallLog(input_tokens=10, output_tokens=20)
    assert log.total_tokens == 30

def test_config_override():
    from infertrack.config import config, configure, reset_config
    configure(default_tag="my_tag", silent=True)
    assert config.default_tag == "my_tag"
    assert config.silent is True
    reset_config()
    assert config.default_tag == "default"

def test_config_invalid_key_raises():
    from infertrack.config import configure
    import pytest
    with pytest.raises(ValueError, match="Unknown config key"):
        configure(nonexistent_key="value")