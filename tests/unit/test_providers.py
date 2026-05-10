# tests/unit/test_providers.py
"""
Day 2 tests: pricing table + OpenAI provider.
All tests use mocked response objects — no real API calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from infertrack.pricing.table import (
    calculate_cost,
    get_price_entry,
    known_models,
    reload,
)
from infertrack.providers.openai import OpenAIProvider


# Helpers                                                              

def make_openai_response(
    model: str = "qwen2.5:0.5b",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a minimal ChatCompletion-shaped mock."""
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = "Hello"
    return resp


# pricing/table.py                                                

class TestCalculateCost:
    """Tests for calculate_cost()."""

    # --- Ollama / free models ---

    def test_ollama_model_is_free(self):
        cost = calculate_cost("qwen2.5:0.5b", 1000, 1000)
        assert cost == 0.0

    def test_all_ollama_models_free(self):
        ollama_models = [
            "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
            "llama3.2", "llama3.2:1b", "llama3.2:3b",
            "llama3.1", "phi3.5", "mistral", "gemma2",
        ]
        for model in ollama_models:
            assert calculate_cost(model, 9999, 9999) == 0.0, (
                f"Expected {model} to be free"
            )

    # --- OpenAI paid models ---

    def test_gpt4o_cost(self):
        # 1000 input @ $0.005/1k + 500 output @ $0.015/1k
        cost = calculate_cost("gpt-4o", 1000, 500)
        expected = 0.005 * 1 + 0.015 * 0.5
        assert abs(cost - expected) < 1e-9

    def test_gpt4o_mini_cost(self):
        cost = calculate_cost("gpt-4o-mini", 1000, 1000)
        expected = 0.00015 + 0.0006
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens_zero_cost(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_only_input_tokens(self):
        cost = calculate_cost("gpt-4o", 1000, 0)
        assert abs(cost - 0.005) < 1e-9

    def test_only_output_tokens(self):
        cost = calculate_cost("gpt-4o", 0, 1000)
        assert abs(cost - 0.015) < 1e-9

    # --- Anthropic models ---

    def test_claude_sonnet_cost(self):
        cost = calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        expected = 0.003 + 0.015
        assert abs(cost - expected) < 1e-9

    def test_claude_haiku_cheaper_than_sonnet(self):
        haiku   = calculate_cost("claude-3-5-haiku-20241022", 1000, 1000)
        sonnet  = calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        assert haiku < sonnet

    # --- Unknown models ---

    def test_unknown_model_returns_zero(self):
        assert calculate_cost("some-unknown-model-xyz", 1000, 1000) == 0.0

    def test_empty_model_string_returns_zero(self):
        assert calculate_cost("", 100, 100) == 0.0

    # --- Large token counts ---

    def test_large_token_count(self):
        # 1M tokens gpt-4o — just checking no overflow / float error
        cost = calculate_cost("gpt-4o", 500_000, 500_000)
        expected = 0.005 * 500 + 0.015 * 500
        assert abs(cost - expected) < 1e-6

    # --- Sanity ranges ---

    def test_gpt4o_cost_in_sane_range(self):
        # 1000 input + 500 output should be between $0.005 and $0.02
        cost = calculate_cost("gpt-4o", 1000, 500)
        assert 0.005 < cost < 0.02

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        mini   = calculate_cost("gpt-4o-mini",  1000, 1000)
        full   = calculate_cost("gpt-4o",        1000, 1000)
        assert mini < full


class TestGetPriceEntry:
    def test_known_model_returns_dict(self):
        entry = get_price_entry("gpt-4o")
        assert entry is not None
        assert "input_per_1k" in entry
        assert "output_per_1k" in entry

    def test_unknown_model_returns_none(self):
        assert get_price_entry("this-model-does-not-exist") is None

    def test_ollama_model_returns_zeros(self):
        entry = get_price_entry("qwen2.5:0.5b")
        assert entry is not None
        assert entry["input_per_1k"] == 0.0
        assert entry["output_per_1k"] == 0.0


class TestKnownModels:
    def test_returns_list_of_strings(self):
        models = known_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    def test_contains_expected_models(self):
        models = known_models()
        for expected in ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "qwen2.5:0.5b"]:
            assert expected in models, f"Expected {expected} in known_models()"

    def test_no_meta_keys(self):
        # Keys starting with "_" must not appear
        models = known_models()
        assert not any(m.startswith("_") for m in models)


class TestReload:
    def test_reload_does_not_raise(self):
        reload()   # just must not raise

    def test_reload_returns_same_data(self):
        before = calculate_cost("gpt-4o", 1000, 500)
        reload()
        after  = calculate_cost("gpt-4o", 1000, 500)
        assert abs(before - after) < 1e-12


# providers/base.py                                               

class TestBaseProviderInterface:
    def test_cannot_instantiate_base(self):
        from infertrack.providers.base import BaseProvider
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore


# providers/openai.py                                                 

class TestOpenAIProviderDetect:
    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_detects_valid_response(self):
        resp = make_openai_response()
        assert self.provider.detect(resp) is True

    def test_rejects_none(self):
        assert self.provider.detect(None) is False

    def test_rejects_string(self):
        assert self.provider.detect("not a response") is False

    def test_rejects_dict(self):
        assert self.provider.detect({"model": "gpt-4o"}) is False

    def test_rejects_missing_usage(self):
        resp = MagicMock(spec=[])        # no attributes at all
        assert self.provider.detect(resp) is False

    def test_rejects_response_without_choices(self):
        resp = MagicMock(spec=["usage", "model"])   # no choices
        resp.usage.prompt_tokens = 1
        resp.usage.completion_tokens = 1
        # spec= means hasattr returns False for unlisted attrs
        assert self.provider.detect(resp) is False


class TestOpenAIProviderExtractUsage:
    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_extracts_correct_tokens(self):
        resp = make_openai_response(prompt_tokens=10, completion_tokens=20)
        inp, out = self.provider.extract_usage(resp)
        assert inp == 10
        assert out == 20

    def test_zero_tokens_valid(self):
        resp = make_openai_response(prompt_tokens=0, completion_tokens=0)
        inp, out = self.provider.extract_usage(resp)
        assert inp == 0 and out == 0

    def test_raises_on_missing_usage(self):
        resp = MagicMock(spec=["model", "choices"])  # no usage
        with pytest.raises(ValueError, match="prompt_tokens"):
            self.provider.extract_usage(resp)

    def test_raises_on_none_tokens(self):
        resp = make_openai_response()
        resp.usage.prompt_tokens = None
        with pytest.raises((ValueError, TypeError)):
            self.provider.extract_usage(resp)

    def test_raises_on_negative_tokens(self):
        resp = make_openai_response(prompt_tokens=-1, completion_tokens=5)
        with pytest.raises(ValueError, match="non-negative"):
            self.provider.extract_usage(resp)

    def test_string_tokens_coerced_to_int(self):
        """API sometimes returns numbers as strings — should still work."""
        resp = make_openai_response()
        resp.usage.prompt_tokens = "15"
        resp.usage.completion_tokens = "25"
        inp, out = self.provider.extract_usage(resp)
        assert inp == 15 and out == 25


class TestOpenAIProviderExtractModel:
    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_extracts_model_name(self):
        resp = make_openai_response(model="gpt-4o")
        assert self.provider.extract_model(resp) == "gpt-4o"

    def test_extracts_ollama_model(self):
        resp = make_openai_response(model="qwen2.5:0.5b")
        assert self.provider.extract_model(resp) == "qwen2.5:0.5b"

    def test_strips_whitespace(self):
        resp = make_openai_response(model="  gpt-4o  ")
        assert self.provider.extract_model(resp) == "gpt-4o"

    def test_raises_on_empty_model(self):
        resp = make_openai_response(model="")
        with pytest.raises(ValueError, match="empty"):
            self.provider.extract_model(resp)

    def test_raises_on_missing_model(self):
        resp = MagicMock(spec=["usage", "choices"])
        with pytest.raises(ValueError):
            self.provider.extract_model(resp)


class TestOpenAIProviderCalculateCost:
    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_ollama_is_free(self):
        assert self.provider.calculate_cost("qwen2.5:0.5b", 9999, 9999) == 0.0

    def test_gpt4o_cost_positive(self):
        cost = self.provider.calculate_cost("gpt-4o", 1000, 500)
        assert cost > 0

    def test_unknown_model_is_free(self):
        assert self.provider.calculate_cost("unknown-xyz", 1000, 1000) == 0.0


class TestOpenAIProviderName:
    def test_provider_name(self):
        assert OpenAIProvider().name == "openai"


class TestOpenAIProviderEndToEnd:
    """Simulate what the decorator will do: detect → extract → cost."""

    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_full_flow_ollama(self):
        resp = make_openai_response(model="qwen2.5:0.5b",
                                    prompt_tokens=10, completion_tokens=20)
        assert self.provider.detect(resp)
        inp, out = self.provider.extract_usage(resp)
        model    = self.provider.extract_model(resp)
        cost     = self.provider.calculate_cost(model, inp, out)

        assert inp   == 10
        assert out   == 20
        assert model == "qwen2.5:0.5b"
        assert cost  == 0.0

    def test_full_flow_gpt4o(self):
        resp = make_openai_response(model="gpt-4o",
                                    prompt_tokens=1000, completion_tokens=500)
        assert self.provider.detect(resp)
        inp, out = self.provider.extract_usage(resp)
        model    = self.provider.extract_model(resp)
        cost     = self.provider.calculate_cost(model, inp, out)

        assert inp  == 1000
        assert out  == 500
        assert cost > 0.0