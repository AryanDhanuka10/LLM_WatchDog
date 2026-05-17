# tests/integration/conftest.py
"""
Shared fixtures for integration tests.
All tests in this folder require Ollama running at http://localhost:11434

Setup:
    ollama serve                  # start Ollama daemon
    ollama pull qwen2.5:0.5b     # pull smallest/fastest model
    pytest tests/integration/ -v
"""
from __future__ import annotations

import pytest
from pathlib import Path
from openai import OpenAI


# Ollama availability — checked once, skips only integration tests     

def _ollama_is_running() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip all integration-marked tests if Ollama is not running.

    This hook runs AFTER collection and only skips tests marked
    'integration' — unit tests are never affected.
    """
    if _ollama_is_running():
        return   # Ollama is up, run everything

    skip_reason = pytest.mark.skip(
        reason="Ollama not running at http://localhost:11434. "
               "Start with: ollama serve && ollama pull qwen2.5:0.5b"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_reason)


# Markers                                                              

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require Ollama running locally"
    )


# Fixtures                                                             

@pytest.fixture(scope="session")
def ollama_client():
    """Real OpenAI-compatible client pointing at local Ollama."""
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


@pytest.fixture(scope="session")
def model():
    """Smallest/fastest Ollama model for testing."""
    return "qwen2.5:0.5b"


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh SQLite DB for each test."""
    from infertrack.storage.db import init_db
    db = tmp_path / "integration_test.db"
    init_db(db)
    return db