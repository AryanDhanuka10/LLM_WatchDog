"""
Placeholder test file — replaced day by day.
This file just verifies the package is importable before Day 1.
"""

def test_package_importable():
    """Package must be importable after pip install -e ."""
    import llm_ledger
    assert llm_ledger.__version__ == "0.1.0"


def test_config_importable():
    """Config module must load without errors."""
    from llm_ledger.config import config, configure, reset_config
    assert config.default_tag == "default"


def test_exceptions_importable():
    """All exceptions must be importable."""
    from llm_ledger.exceptions import (
        WatchdogError,
        BudgetExceeded,
        ProviderNotDetected,
        PricingModelNotFound,
    )


def test_budget_exceeded_message():
    """BudgetExceeded must carry spent/limit/user_id."""
    from llm_ledger.exceptions import BudgetExceeded
    err = BudgetExceeded(spent=0.15, limit=0.10, user_id="user_42")
    assert err.spent == 0.15
    assert err.limit == 0.10
    assert err.user_id == "user_42"
    assert "user_42" in str(err)


def test_calllog_importable():
    """CallLog dataclass must be importable and have correct defaults."""
    from llm_ledger.storage.models import CallLog
    log = CallLog()
    assert log.provider == "unknown"
    assert log.success is True
    assert log.cost_usd == 0.0
    assert log.id is None


def test_calllog_total_tokens_auto():
    """CallLog must auto-compute total_tokens in __post_init__."""
    from llm_ledger.storage.models import CallLog
    log = CallLog(input_tokens=10, output_tokens=20)
    assert log.total_tokens == 30


def test_config_override():
    """configure() must override a valid key without error."""
    from llm_ledger.config import config, configure, reset_config
    configure(default_tag="my_tag", silent=True)
    assert config.default_tag == "my_tag"
    assert config.silent is True
    reset_config()
    assert config.default_tag == "default"


def test_config_invalid_key_raises():
    """configure() must raise ValueError for unknown keys."""
    from llm_ledger.config import configure
    import pytest
    with pytest.raises(ValueError, match="Unknown config key"):
        configure(nonexistent_key="value")
