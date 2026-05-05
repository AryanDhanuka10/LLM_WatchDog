# src/llm_watchdog/__init__.py
"""llm-watchdog: Zero-config LLM call interceptor.

Public API (grows each day):
  Day 1: storage layer only
  Day 2: pricing + providers
  Day 3: @watchdog decorator + watch() context manager
  Day 4: Budget + BudgetExceeded
  Day 5: CLI
"""

__version__ = "0.1.0"