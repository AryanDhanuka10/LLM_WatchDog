# Changelog

All notable changes to `llm-ledger` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.1.0] - 2026-05-07

### Added
- `storage/models.py` — `CallLog` dataclass with all fields and `total_tokens` computed in `__post_init__`
- `storage/db.py` — `init_db()`, `insert_log()`, `query_logs()`, `get_total_cost()` using stdlib `sqlite3`
- `pricing/prices.json` — embedded pricing table for Ollama (free), OpenAI, and Anthropic models
- `pricing/table.py` — `calculate_cost()`, `get_price_entry()`, `known_models()`, `reload()`
- `providers/base.py` — abstract `BaseProvider` interface
- `providers/openai.py` — OpenAI + Ollama provider (duck-typed detection, tokens from `response.usage` only)
- `core/decorator.py` — `@watchdog(tag, user_id, session_id, db_path)` decorator
- `core/context.py` — `watch()` context manager with `WatchContext.add_response()`
- `core/budget.py` — `Budget(max_usd, user_id, period)` context manager with pre-flight spend check
- `exceptions.py` — `WatchdogError`, `BudgetExceeded`, `ProviderNotDetected`, `PricingModelNotFound`
- `config.py` — `WatchdogConfig` singleton, `configure()`, `reset_config()`
- `cli/__main__.py` — `watchdog` Click CLI entry point
- `cli/commands.py` — `summary`, `tail`, `top` subcommands

### Notes
- Token counting reads `response.usage` fields only — no `tiktoken` dependency
- All data stored locally in `~/.llm-ledger/logs.db`
- Ollama supported out of the box (OpenAI-compatible API, zero cost)