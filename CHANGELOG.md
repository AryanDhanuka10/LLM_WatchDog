# Changelog

All notable changes to `infertrack` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.1] - 2026-05-10

### Added
- `core/streaming.py` — `StreamingWrapper` passthrough generator for `stream=True` calls
- `@watchdog` now detects streaming responses and logs after stream exhaustion
- `core/retry.py` — `with_retry()` with `exponential`, `linear`, `fixed` backoff
- `@watchdog(retry=N, backoff=...)` — built-in retry with configurable strategy
- `providers/anthropic.py` — full Anthropic Claude provider support
- `core/interceptor.py` — `infertrack.intercept()` / `stop()` zero-code-change global patch
- `cli/export.py` — `watchdog export --format csv/json` with file and filter support
- `tests/unit/test_coverage_gaps.py` — targeted gap coverage tests
- `.github/workflows/test.yml` — CI matrix on Python 3.10 / 3.11 / 3.12
- `.github/workflows/publish.yml` — auto-publish to PyPI on `v*` tags

### Changed
- `storage/models.py` — `CallLog` all fields have defaults; `id=None` until DB insert; `total_tokens` computed in `__post_init__`
- `storage/db.py` — `insert_log()` auto-assigns UUID; `_parse_timestamp()` handles str, datetime, and Unix epoch float
- `core/decorator.py` — streaming path added; retry params added
- `core/context.py` — `init_db()` called on entry to ensure DB exists
- `core/interceptor.py` — `_PROVIDERS` list includes `AnthropicProvider`; anthropic patch is optional
- All core files call `init_db()` on first use — no more `OperationalError` on fresh installs
- `pyproject.toml` — bumped to `1.0.0`; `pytest-cov` in dev deps; coverage config added

### Fixed
- `_parse_timestamp()` handles Unix epoch floats stored by older schema versions
- `intercept()` / `watchdog()` / `watch()` now call `init_db()` before first write
- `TestInterceptorPatching` tests read `Completions` live from `sys.modules` to avoid cross-module class identity issues

---

## [0.1.0] - 2026-05-07

### Added
- `storage/models.py` — `CallLog` dataclass
- `storage/db.py` — `init_db()`, `insert_log()`, `query_logs()`, `get_total_cost()`
- `pricing/prices.json` — embedded pricing for Ollama (free), OpenAI, Anthropic
- `pricing/table.py` — `calculate_cost()`, `known_models()`, `reload()`
- `providers/base.py` — abstract `BaseProvider`
- `providers/openai.py` — OpenAI + Ollama provider
- `core/decorator.py` — `@watchdog(tag, user_id, session_id, db_path)`
- `core/context.py` — `watch()` context manager with `WatchContext.add_response()`
- `core/budget.py` — `Budget(max_usd, user_id, period)` with pre-flight spend check
- `exceptions.py` — `WatchdogError`, `BudgetExceeded`, `ProviderNotDetected`, `PricingModelNotFound`
- `config.py` — `WatchdogConfig` singleton, `configure()`, `reset_config()`
- `cli/__main__.py` + `cli/commands.py` — `summary`, `tail`, `top` subcommands
- Published to TestPyPI as `infertrack 0.1.0`