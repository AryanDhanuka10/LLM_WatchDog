# tests/conftest.py
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# Unit test fixtures — no real LLM, no cost, always works              

@pytest.fixture
def mock_openai_response():
    """Mimics openai.types.chat.ChatCompletion structure."""
    response = MagicMock()
    response.model = "qwen2.5:0.5b"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.choices[0].message.content = "Hello"
    return response


# Integration test fixtures — needs Ollama running at localhost:11434  

@pytest.fixture
def ollama_client():
    """Real OpenAI-compatible client pointing at local Ollama."""
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")