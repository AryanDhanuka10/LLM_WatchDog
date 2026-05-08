"""
template.py — llm-meter Project Scaffolder
==============================================
Run this ONCE before Day 1 to create the entire project structure.

Usage:
    python template.py
    python template.py --path /custom/path/llm-meter

What it creates:
    - Full folder structure (src layout)
    - All placeholder Python files with correct imports and docstrings
    - pyproject.toml (ready to build)
    - prices.json (Ollama free + OpenAI/Anthropic paid)
    - conftest.py (mock fixtures)
    - .gitignore
    - CHANGELOG.md
    - README.md stub

After running:
    cd llm-meter
    pip install -r requirements.txt
    pip install -e .          ← installs package in editable mode
    pytest tests/ -v          ← should collect 0 tests (placeholders only)
"""

import argparse
import os
import json
from pathlib import Path
from textwrap import dedent


# ── Helpers ───────────────────────────────────────────────────────────────────

def write(path: Path, content: str) -> None:
    """Write content to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip())
    print(f"  created  {path}")


def touch(path: Path) -> None:
    """Create an empty __init__.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")
    print(f"  touched  {path}")


# ── File Contents ─────────────────────────────────────────────────────────────

def make_pyproject(root: Path) -> None:
    write(root / "pyproject.toml", """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "llm-meter"
        version = "0.1.0"
        description = "Zero-config LLM call interceptor: cost, latency, budget enforcement"
        readme = "README.md"
        license = { text = "MIT" }
        requires-python = ">=3.10"
        authors = [{ name = "Your Name", email = "you@example.com" }]
        keywords = ["llm", "ollama", "openai", "cost", "monitoring", "tokens", "budget"]
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ]
        dependencies = [
            "click>=8.0",
            "openai>=1.0",
        ]

        [project.optional-dependencies]
        anthropic = ["anthropic>=0.25"]
        dev = ["pytest", "pytest-mock", "pytest-cov", "ruff", "mypy", "build", "twine"]

        [project.scripts]
        watchdog = "llm_ledger.cli.__main__:cli"

        [project.urls]
        Repository = "https://github.com/yourname/llm-meter"
        "Bug Tracker" = "https://github.com/yourname/llm-meter/issues"

        [tool.hatch.build.targets.wheel]
        packages = ["src/llm_ledger"]

        [tool.ruff]
        line-length = 100
        target-version = "py310"

        [tool.ruff.lint]
        select = ["E", "F", "I", "UP"]

        [tool.mypy]
        python_version = "3.10"
        strict = false
        ignore_missing_imports = true

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        markers = [
            "integration: marks tests that require Ollama running locally",
            "slow: marks tests that are slow",
        ]
        addopts = "-v --tb=short"
    """)


def make_gitignore(root: Path) -> None:
    write(root / ".gitignore", """
        # Python
        __pycache__/
        *.py[cod]
        *.egg-info/
        .eggs/
        dist/
        build/
        *.whl

        # Virtual env
        .venv/
        venv/
        env/

        # Testing
        .pytest_cache/
        .coverage
        htmlcov/
        .mypy_cache/

        # Package DB (user data — never commit)
        *.db
        *.db-shm
        *.db-wal

        # IDE
        .vscode/
        .idea/
        *.swp

        # OS
        .DS_Store
        Thumbs.db

        # Build artifacts
        dist/
        *.egg-info/
    """)


def make_changelog(root: Path) -> None:
    write(root / "CHANGELOG.md", """
        # Changelog

        All notable changes to `llm-meter` will be documented here.
        Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
        Versioning: [Semantic Versioning](https://semver.org/)

        ---

        ## [Unreleased]

        ### Added
        - Initial project scaffold

        ---

        ## [0.1.0] - TBD
        ### Added
        - CallLog dataclass and SQLite storage layer
    """)


def make_readme(root: Path) -> None:
    write(root / "README.md", """
        # llm-meter

        > Zero-config LLM call interceptor. Track cost, latency, and token usage locally.
        > No cloud. No API key required. Works with Ollama (free), OpenAI, and Anthropic.

        ## Install

        ```bash
        pip install llm-meter
        ```

        ## Quickstart (Free with Ollama)

        ```python
        from openai import OpenAI
        from llm_ledger import watchdog

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        @watchdog()
        def ask(prompt: str) -> str:
            response = client.chat.completions.create(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        ask("Hello!")
        ```

        ```bash
        watchdog summary
        # Last 24h | 1 call | 40 tokens | $0.00 | avg latency 312ms
        ```

        ## Documentation

        WIP — see source code and tests for now.
    """)


def make_prices_json(root: Path) -> None:
    prices = {
        "_last_updated": "2026-05-04",
        "_source": "https://openai.com/pricing + https://www.anthropic.com/pricing",
        "_note": "Ollama models are free (local inference). Update paid models periodically.",

        # ── Ollama (local, always free) ──────────────────────────────────────
        "qwen2.5:0.5b":               {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "qwen2.5:1.5b":               {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "qwen2.5:3b":                 {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "qwen2.5:7b":                 {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "llama3.2":                   {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "llama3.2:1b":                {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "llama3.1":                   {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "phi3.5":                     {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "phi3":                       {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "mistral":                    {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "gemma2:2b":                  {"input_per_1k": 0.0, "output_per_1k": 0.0},
        "deepseek-r1:1.5b":           {"input_per_1k": 0.0, "output_per_1k": 0.0},

        # ── OpenAI (paid, optional) ──────────────────────────────────────────
        "gpt-4o":                     {"input_per_1k": 0.005,   "output_per_1k": 0.015},
        "gpt-4o-mini":                {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
        "gpt-4-turbo":                {"input_per_1k": 0.01,    "output_per_1k": 0.03},
        "gpt-3.5-turbo":              {"input_per_1k": 0.0005,  "output_per_1k": 0.0015},

        # ── Anthropic (paid, optional) ───────────────────────────────────────
        "claude-3-5-sonnet-20241022": {"input_per_1k": 0.003,   "output_per_1k": 0.015},
        "claude-3-5-haiku-20241022":  {"input_per_1k": 0.001,   "output_per_1k": 0.005},
        "claude-3-opus-20240229":     {"input_per_1k": 0.015,   "output_per_1k": 0.075},
    }
    path = root / "src" / "llm_ledger" / "pricing" / "prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prices, indent=2))
    print(f"  created  {path}")


def make_config(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "config.py", """
        \"\"\"
        config.py — Central configuration for llm-meter
        ====================================================
        Single source of truth for all runtime settings.

        Priority order (highest → lowest):
            1. Environment variables  (WATCHDOG_*)
            2. Values set via configure() in user code
            3. Defaults defined here

        Why this file exists:
            Without config.py, settings like DB path, default tags, and
            pricing overrides get hardcoded in 5 different files. When a
            user wants to change the DB path, they'd have no obvious place
            to look. config.py is that obvious place.

        Usage:
            from llm_ledger.config import config

            # Read a setting
            db_path = config.db_path

            # Override at runtime (e.g. in tests)
            from llm_ledger.config import configure
            configure(db_path=":memory:", default_tag="test")
        \"\"\"

        import os
        from dataclasses import dataclass, field
        from pathlib import Path


        @dataclass
        class WatchdogConfig:
            \"\"\"All runtime configuration in one place.\"\"\"

            # ── Storage ──────────────────────────────────────────────────────
            db_path: str = field(
                default_factory=lambda: os.environ.get(
                    "WATCHDOG_DB_PATH",
                    str(Path.home() / ".llm-meter" / "logs.db")
                )
            )
            # Why: Users may want to change DB location (CI, Docker, multi-project)
            # Env var lets them do it without touching code.

            # ── Pricing ──────────────────────────────────────────────────────
            prices_path: str = field(
                default_factory=lambda: os.environ.get(
                    "WATCHDOG_PRICES_PATH",
                    ""   # empty = use bundled prices.json
                )
            )
            # Why: Power users may maintain their own pricing file with
            # enterprise/custom model costs. This lets them plug it in.

            # ── Default Tags ─────────────────────────────────────────────────
            default_tag: str = field(
                default_factory=lambda: os.environ.get(
                    "WATCHDOG_DEFAULT_TAG",
                    "default"
                )
            )
            # Why: In a large app, every call should have a tag.
            # If user forgets, this prevents NULL tag chaos in the DB.

            # ── Budget ───────────────────────────────────────────────────────
            global_budget_usd: float = field(
                default_factory=lambda: float(
                    os.environ.get("WATCHDOG_GLOBAL_BUDGET_USD", "0")
                )
            )
            # Why: 0 means no global budget. Positive value = hard global cap.
            # Useful for CI environments where you want a safety net.

            # ── Retry ────────────────────────────────────────────────────────
            default_retry_count: int = field(
                default_factory=lambda: int(
                    os.environ.get("WATCHDOG_RETRY_COUNT", "0")
                )
            )
            default_retry_backoff: str = field(
                default_factory=lambda: os.environ.get(
                    "WATCHDOG_RETRY_BACKOFF",
                    "exponential"   # or "linear"
                )
            )

            # ── Logging ──────────────────────────────────────────────────────
            silent: bool = field(
                default_factory=lambda: os.environ.get(
                    "WATCHDOG_SILENT", "false"
                ).lower() == "true"
            )
            # Why: In production apps, you may not want any stdout output.
            # silent=True suppresses all watchdog prints.


        # ── Singleton ────────────────────────────────────────────────────────
        # One global config instance used across the entire package.
        # Import this, don't instantiate WatchdogConfig yourself.
        config = WatchdogConfig()


        def configure(**kwargs) -> None:
            \"\"\"
            Override config values at runtime.

            Example:
                from llm_ledger.config import configure
                configure(db_path=":memory:", default_tag="pytest")

            Useful for:
                - Tests (use in-memory DB, never touch ~/.llm-meter/)
                - Multi-tenant apps (switch DB per request)
                - CI/CD (set silent=True, global_budget_usd=0.10)
            \"\"\"
            for key, value in kwargs.items():
                if not hasattr(config, key):
                    raise ValueError(
                        f"Unknown config key: '{key}'. "
                        f"Valid keys: {list(config.__dataclass_fields__.keys())}"
                    )
                setattr(config, key, value)


        def reset_config() -> None:
            \"\"\"
            Reset all config to defaults.
            Call this in test teardown to avoid state leakage between tests.

            Example (conftest.py):
                @pytest.fixture(autouse=True)
                def clean_config():
                    yield
                    reset_config()
            \"\"\"
            new = WatchdogConfig()
            for key in config.__dataclass_fields__:
                setattr(config, key, getattr(new, key))
    """)


def make_exceptions(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "exceptions.py", """
        \"\"\"
        exceptions.py — All custom exceptions for llm-meter
        \"\"\"


        class WatchdogError(Exception):
            \"\"\"Base exception for all llm-meter errors.\"\"\"


        class BudgetExceeded(WatchdogError):
            \"\"\"
            Raised when a Budget context manager's spending limit is crossed.

            Attributes:
                spent:   How much was spent (USD) at time of exception
                limit:   The configured budget limit (USD)
                user_id: The user/session that exceeded the budget
            \"\"\"

            def __init__(self, spent: float, limit: float, user_id: str = "global") -> None:
                self.spent = spent
                self.limit = limit
                self.user_id = user_id
                super().__init__(
                    f"Budget exceeded for '{user_id}': "
                    f"spent ${spent:.6f}, limit ${limit:.6f}"
                )


        class ProviderNotDetected(WatchdogError):
            \"\"\"
            Raised when watchdog cannot identify the LLM provider
            from the response object.
            \"\"\"

            def __init__(self, response_type: str) -> None:
                self.response_type = response_type
                super().__init__(
                    f"Cannot detect provider from response type: '{response_type}'. "
                    f"Supported: OpenAI ChatCompletion, Anthropic Message."
                )


        class PricingModelNotFound(WatchdogError):
            \"\"\"
            Raised when a model name has no entry in prices.json.
            Cost will be logged as 0.0 with a warning instead of crashing.
            \"\"\"

            def __init__(self, model: str) -> None:
                self.model = model
                super().__init__(
                    f"No pricing data for model '{model}'. "
                    f"Cost logged as $0.00. Add it to prices.json to track cost."
                )
    """)


def make_models(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "storage" / "models.py", """
        \"\"\"
        models.py — Core data structures for llm-meter
        ==================================================
        CallLog is the single record written to SQLite for every LLM call.
        Using stdlib dataclasses — no Pydantic, no SQLAlchemy.
        \"\"\"

        from dataclasses import dataclass, field
        from datetime import datetime
        from typing import Optional


        @dataclass
        class CallLog:
            \"\"\"
            One row in the logs table. Represents a single LLM API call.

            Fields:
                id:               Auto-assigned by SQLite (None before insert)
                timestamp:        When the call was made (UTC)
                provider:         'openai', 'anthropic', 'ollama', 'unknown'
                model:            e.g. 'qwen2.5:0.5b', 'gpt-4o'
                input_tokens:     From response.usage.prompt_tokens
                output_tokens:    From response.usage.completion_tokens
                total_tokens:     input + output
                cost_usd:         Calculated from prices.json (0.0 for Ollama)
                latency_ms:       Wall clock time in milliseconds
                tag:              Optional label (e.g. 'summarize', 'search')
                user_id:          Optional user identifier for budget tracking
                session_id:       Optional session grouping
                success:          False if the call raised an exception
                error_msg:        Exception message if success=False, else None
            \"\"\"

            # Identity
            id: Optional[int] = field(default=None)
            timestamp: datetime = field(default_factory=datetime.utcnow)

            # Provider info
            provider: str = "unknown"
            model: str = "unknown"

            # Token usage (from response.usage — no tiktoken)
            input_tokens: int = 0
            output_tokens: int = 0
            total_tokens: int = 0

            # Cost (0.0 for Ollama)
            cost_usd: float = 0.0

            # Performance
            latency_ms: float = 0.0

            # Tagging
            tag: str = "default"
            user_id: str = "anonymous"
            session_id: Optional[str] = None

            # Status
            success: bool = True
            error_msg: Optional[str] = None

            def __post_init__(self) -> None:
                \"\"\"Auto-compute total_tokens if not provided.\"\"\"
                if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
                    self.total_tokens = self.input_tokens + self.output_tokens
    """)


def make_db(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "storage" / "db.py", """
        \"\"\"
        db.py — SQLite storage layer (PLACEHOLDER)
        ==========================================
        Full implementation written on Day 1.
        \"\"\"

        # TODO: Day 1 — implement init_db, insert_log, query_logs
    """)


def make_providers(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "providers" / "base.py", """
        \"\"\"
        base.py — Abstract provider interface (PLACEHOLDER)
        Full implementation on Day 2.
        \"\"\"
        # TODO: Day 2
    """)

    write(root / "src" / "llm_ledger" / "providers" / "openai.py", """
        \"\"\"
        openai.py — OpenAI + Ollama provider (PLACEHOLDER)
        Full implementation on Day 2.
        \"\"\"
        # TODO: Day 2
    """)

    write(root / "src" / "llm_ledger" / "providers" / "anthropic.py", """
        \"\"\"
        anthropic.py — Anthropic provider (PLACEHOLDER)
        Full implementation on Day 8.
        \"\"\"
        # TODO: Day 8
    """)


def make_pricing(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "pricing" / "table.py", """
        \"\"\"
        table.py — Pricing table loader (PLACEHOLDER)
        Full implementation on Day 2.
        \"\"\"
        # TODO: Day 2
    """)


def make_core(root: Path) -> None:
    for name, day in [
        ("decorator.py", 3),
        ("context.py", 3),
        ("budget.py", 4),
        ("interceptor.py", 7),
        ("retry.py", 3),
    ]:
        write(root / "src" / "llm_ledger" / "core" / name, f"""
            \"\"\"
            {name} — PLACEHOLDER
            Full implementation on Day {day}.
            \"\"\"
            # TODO: Day {day}
        """)


def make_cli(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "cli" / "__main__.py", """
        \"\"\"
        __main__.py — CLI entry point (PLACEHOLDER)
        Full implementation on Day 5.
        \"\"\"
        # TODO: Day 5
    """)

    write(root / "src" / "llm_ledger" / "cli" / "commands.py", """
        \"\"\"
        commands.py — CLI commands (PLACEHOLDER)
        Full implementation on Day 5.
        \"\"\"
        # TODO: Day 5
    """)


def make_init(root: Path) -> None:
    write(root / "src" / "llm_ledger" / "__init__.py", """
        \"\"\"
        llm-meter
        ============
        Zero-config LLM call interceptor.
        Track cost, latency, and token usage. Enforce budgets. No cloud required.

        Quickstart:
            from llm_ledger import watchdog

            @watchdog()
            def ask(prompt):
                return client.chat.completions.create(...)
        \"\"\"

        __version__ = "0.1.0"

        # Public API — populated as days progress
        # Day 3: from llm_ledger.core.decorator import watchdog
        # Day 3: from llm_ledger.core.context import watch
        # Day 4: from llm_ledger.core.budget import Budget
        # Day 4: from llm_ledger.exceptions import BudgetExceeded
        # Day 7: from llm_ledger.core.interceptor import intercept
    """)


def make_tests(root: Path) -> None:
    write(root / "tests" / "__init__.py", "")
    write(root / "tests" / "unit" / "__init__.py", "")
    write(root / "tests" / "integration" / "__init__.py", "")

    write(root / "tests" / "conftest.py", """
        \"\"\"
        conftest.py — Shared pytest fixtures for llm-meter tests
        ============================================================

        Two categories:
            Unit fixtures:        Pure mocks. No network. No Ollama. Always fast.
            Integration fixtures: Real Ollama calls. Mark with @pytest.mark.integration.

        Key fixture: tmp_db
            Every unit test gets a fresh in-memory SQLite DB.
            This means tests NEVER touch ~/.llm-meter/logs.db
            and NEVER interfere with each other.
        \"\"\"

        import pytest
        from unittest.mock import MagicMock
        from llm_ledger.config import configure, reset_config


        # ── Config isolation ─────────────────────────────────────────────────

        @pytest.fixture(autouse=True)
        def isolate_config():
            \"\"\"
            Reset config to defaults after every test.
            Prevents state leakage between tests.
            autouse=True means this runs for EVERY test automatically.
            \"\"\"
            yield
            reset_config()


        @pytest.fixture
        def tmp_db(tmp_path):
            \"\"\"
            Point the DB at a temp file for this test only.
            After the test, it's deleted automatically by pytest.

            Usage:
                def test_something(tmp_db):
                    # DB is isolated — does not touch ~/.llm-meter/
                    ...
            \"\"\"
            db_path = str(tmp_path / "test_watchdog.db")
            configure(db_path=db_path)
            return db_path


        # ── Mock LLM responses ───────────────────────────────────────────────

        @pytest.fixture
        def mock_openai_response():
            \"\"\"
            Simulates a real OpenAI/Ollama ChatCompletion response object.
            Matches exact structure of openai.types.chat.ChatCompletion.

            Token counts:
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            \"\"\"
            response = MagicMock()
            response.model = "qwen2.5:0.5b"
            response.object = "chat.completion"
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 20
            response.usage.total_tokens = 30
            response.choices[0].message.content = "Hello! How can I assist you today?"
            response.choices[0].finish_reason = "stop"
            return response


        @pytest.fixture
        def mock_openai_response_gpt4():
            \"\"\"
            Simulates a GPT-4o response (paid model, non-zero cost).
            Use this for cost calculation tests.
            \"\"\"
            response = MagicMock()
            response.model = "gpt-4o"
            response.object = "chat.completion"
            response.usage.prompt_tokens = 100
            response.usage.completion_tokens = 50
            response.usage.total_tokens = 150
            response.choices[0].message.content = "This is a GPT-4o response."
            response.choices[0].finish_reason = "stop"
            return response


        @pytest.fixture
        def mock_anthropic_response():
            \"\"\"
            Simulates a real Anthropic Message response object.
            Matches exact structure of anthropic.types.Message.
            \"\"\"
            response = MagicMock()
            response.model = "claude-3-5-haiku-20241022"
            response.type = "message"
            response.usage.input_tokens = 15
            response.usage.output_tokens = 25
            response.content[0].text = "Hello from Claude!"
            response.stop_reason = "end_turn"
            return response


        @pytest.fixture
        def mock_failed_response():
            \"\"\"Simulates an API call that raises an exception.\"\"\"
            def raise_error(*args, **kwargs):
                raise ConnectionError("Ollama not running")
            return raise_error


        # ── Integration fixtures (require Ollama running) ─────────────────────

        @pytest.fixture
        def ollama_client():
            \"\"\"
            Real OpenAI client pointed at local Ollama.
            Only used in @pytest.mark.integration tests.

            Requires:
                ollama serve  (running in background)
                ollama pull qwen2.5:0.5b

            Usage:
                @pytest.mark.integration
                def test_real_call(ollama_client, tmp_db):
                    response = ollama_client.chat.completions.create(
                        model="qwen2.5:0.5b",
                        messages=[{"role": "user", "content": "hi"}]
                    )
                    assert response.usage.prompt_tokens > 0
            \"\"\"
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
                return client
            except ImportError:
                pytest.skip("openai package not installed")


        @pytest.fixture
        def ollama_model():
            \"\"\"Default model for integration tests — smallest and fastest.\"\"\"
            return "qwen2.5:0.5b"
    """)

    write(root / "tests" / "unit" / "test_placeholder.py", """
        \"\"\"
        Placeholder test file — replaced day by day.
        This file just verifies the package is importable before Day 1.
        \"\"\"

        def test_package_importable():
            \"\"\"Package must be importable after pip install -e .\"\"\"
            import llm_ledger
            assert llm_ledger.__version__ == "0.1.0"


        def test_config_importable():
            \"\"\"Config module must load without errors.\"\"\"
            from llm_ledger.config import config, configure, reset_config
            assert config.default_tag == "default"


        def test_exceptions_importable():
            \"\"\"All exceptions must be importable.\"\"\"
            from llm_ledger.exceptions import (
                WatchdogError,
                BudgetExceeded,
                ProviderNotDetected,
                PricingModelNotFound,
            )


        def test_budget_exceeded_message():
            \"\"\"BudgetExceeded must carry spent/limit/user_id.\"\"\"
            from llm_ledger.exceptions import BudgetExceeded
            err = BudgetExceeded(spent=0.15, limit=0.10, user_id="user_42")
            assert err.spent == 0.15
            assert err.limit == 0.10
            assert err.user_id == "user_42"
            assert "user_42" in str(err)


        def test_calllog_importable():
            \"\"\"CallLog dataclass must be importable and have correct defaults.\"\"\"
            from llm_ledger.storage.models import CallLog
            log = CallLog()
            assert log.provider == "unknown"
            assert log.success is True
            assert log.cost_usd == 0.0
            assert log.id is None


        def test_calllog_total_tokens_auto():
            \"\"\"CallLog must auto-compute total_tokens in __post_init__.\"\"\"
            from llm_ledger.storage.models import CallLog
            log = CallLog(input_tokens=10, output_tokens=20)
            assert log.total_tokens == 30


        def test_config_override():
            \"\"\"configure() must override a valid key without error.\"\"\"
            from llm_ledger.config import config, configure, reset_config
            configure(default_tag="my_tag", silent=True)
            assert config.default_tag == "my_tag"
            assert config.silent is True
            reset_config()
            assert config.default_tag == "default"


        def test_config_invalid_key_raises():
            \"\"\"configure() must raise ValueError for unknown keys.\"\"\"
            from llm_ledger.config import configure
            import pytest
            with pytest.raises(ValueError, match="Unknown config key"):
                configure(nonexistent_key="value")
    """)


# ── Main Scaffold Function ─────────────────────────────────────────────────────

def scaffold(base_path: str = ".") -> None:
    root = Path(base_path) / "llm-meter"

    if root.exists():
        print(f"⚠️  Directory '{root}' already exists. Files will be overwritten.")
    else:
        print(f"🚀 Creating project at: {root.resolve()}")

    print("\n── Core files ───────────────────────────────────")
    make_pyproject(root)
    make_gitignore(root)
    make_changelog(root)
    make_readme(root)

    print("\n── Package source ───────────────────────────────")
    make_init(root)
    make_config(root)
    make_exceptions(root)

    print("\n── Storage layer ────────────────────────────────")
    make_models(root)
    make_db(root)
    for p in ["__init__.py"]:
        touch(root / "src" / "llm_ledger" / "storage" / p)

    print("\n── Providers ────────────────────────────────────")
    make_providers(root)
    for p in ["__init__.py"]:
        touch(root / "src" / "llm_ledger" / "providers" / p)

    print("\n── Pricing ──────────────────────────────────────")
    make_pricing(root)
    make_prices_json(root)
    touch(root / "src" / "llm_ledger" / "pricing" / "__init__.py")

    print("\n── Core (decorators, budget, interceptor) ───────")
    make_core(root)
    touch(root / "src" / "llm_ledger" / "core" / "__init__.py")

    print("\n── CLI ──────────────────────────────────────────")
    make_cli(root)

    print("\n── Tests ────────────────────────────────────────")
    make_tests(root)

    print("\n── Done! ─────────────────────────────────────────")
    print(f"""
Next steps:
    cd {root}
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    pytest tests/ -v

Expected output: 7 passed (pre-Day 1 sanity checks)
Then open a new chat with the starter prompt and say: Today is Day 1
    """)


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaffold the llm-meter project structure"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Parent directory to create llm-meter/ in (default: current dir)"
    )
    args = parser.parse_args()
    scaffold(base_path=args.path)