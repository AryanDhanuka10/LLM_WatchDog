# tests/unit/test_providers_anthropic.py
"""
Day 8 tests: Anthropic provider + updated interceptor.
All tests use mocked response objects — no real Anthropic API calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from infertrack.providers.anthropic import AnthropicProvider
from infertrack.pricing.table import calculate_cost


# Helpers                                                              

def make_anthropic_response(
    model: str = "claude-3-5-sonnet-20241022",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    """Build a minimal anthropic.types.Message-shaped mock."""
    resp = MagicMock(spec=[
        "model", "usage", "content", "stop_reason", "id", "type", "role"
    ])
    resp.model = model
    resp.usage.input_tokens  = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.content = [MagicMock(type="text", text="Hello from Claude")]
    resp.stop_reason = "end_turn"
    # Critically: no .choices attribute → won't be detected as OpenAI
    return resp


def make_openai_response(model="gpt-4o"):
    """OpenAI-shaped mock — should NOT be detected by AnthropicProvider."""
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    resp.choices[0].message.content = "Hello from GPT"
    return resp


# AnthropicProvider.detect                                            

class TestAnthropicProviderDetect:

    def setup_method(self):
        self.p = AnthropicProvider()

    def test_detects_valid_anthropic_response(self):
        resp = make_anthropic_response()
        assert self.p.detect(resp) is True

    def test_rejects_openai_response(self):
        """OpenAI response has .choices — must be rejected."""
        resp = make_openai_response()
        assert self.p.detect(resp) is False

    def test_rejects_none(self):
        assert self.p.detect(None) is False

    def test_rejects_string(self):
        assert self.p.detect("not a response") is False

    def test_rejects_dict(self):
        assert self.p.detect({"model": "claude-3-5-sonnet-20241022"}) is False

    def test_rejects_missing_usage(self):
        resp = MagicMock(spec=["model", "content", "stop_reason"])
        assert self.p.detect(resp) is False

    def test_rejects_missing_input_tokens(self):
        resp = MagicMock(spec=["model", "usage", "content", "stop_reason"])
        resp.usage = MagicMock(spec=["output_tokens"])  # no input_tokens
        assert self.p.detect(resp) is False

    def test_rejects_response_with_choices(self):
        """Anything with .choices is OpenAI — must be rejected."""
        resp = MagicMock(spec=[
            "model", "usage", "content", "stop_reason", "choices"
        ])
        resp.usage.input_tokens  = 10
        resp.usage.output_tokens = 20
        assert self.p.detect(resp) is False


# AnthropicProvider.extract_usage                                     

class TestAnthropicProviderExtractUsage:

    def setup_method(self):
        self.p = AnthropicProvider()

    def test_extracts_correct_tokens(self):
        resp = make_anthropic_response(input_tokens=15, output_tokens=35)
        inp, out = self.p.extract_usage(resp)
        assert inp == 15
        assert out == 35

    def test_zero_tokens_valid(self):
        resp = make_anthropic_response(input_tokens=0, output_tokens=0)
        inp, out = self.p.extract_usage(resp)
        assert inp == 0 and out == 0

    def test_raises_on_missing_usage(self):
        resp = MagicMock(spec=["model", "content"])
        with pytest.raises(ValueError, match="input_tokens"):
            self.p.extract_usage(resp)

    def test_raises_on_none_tokens(self):
        resp = make_anthropic_response()
        resp.usage.input_tokens = None
        with pytest.raises((ValueError, TypeError)):
            self.p.extract_usage(resp)

    def test_raises_on_negative_tokens(self):
        resp = make_anthropic_response(input_tokens=-1, output_tokens=5)
        with pytest.raises(ValueError, match="non-negative"):
            self.p.extract_usage(resp)

    def test_string_tokens_coerced(self):
        resp = make_anthropic_response()
        resp.usage.input_tokens  = "12"
        resp.usage.output_tokens = "34"
        inp, out = self.p.extract_usage(resp)
        assert inp == 12 and out == 34


# AnthropicProvider.extract_model                                     

class TestAnthropicProviderExtractModel:

    def setup_method(self):
        self.p = AnthropicProvider()

    def test_extracts_model_name(self):
        resp = make_anthropic_response(model="claude-3-5-sonnet-20241022")
        assert self.p.extract_model(resp) == "claude-3-5-sonnet-20241022"

    def test_extracts_haiku_model(self):
        resp = make_anthropic_response(model="claude-3-5-haiku-20241022")
        assert self.p.extract_model(resp) == "claude-3-5-haiku-20241022"

    def test_strips_whitespace(self):
        resp = make_anthropic_response(model="  claude-3-opus-20240229  ")
        assert self.p.extract_model(resp) == "claude-3-opus-20240229"

    def test_raises_on_empty_model(self):
        resp = make_anthropic_response(model="")
        with pytest.raises(ValueError, match="empty"):
            self.p.extract_model(resp)

    def test_raises_on_missing_model(self):
        resp = MagicMock(spec=["usage", "content"])
        with pytest.raises(ValueError):
            self.p.extract_model(resp)


# AnthropicProvider.calculate_cost                                    

class TestAnthropicProviderCalculateCost:

    def setup_method(self):
        self.p = AnthropicProvider()

    def test_sonnet_cost_positive(self):
        cost = self.p.calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        assert cost > 0.0

    def test_haiku_cheaper_than_sonnet(self):
        haiku  = self.p.calculate_cost("claude-3-5-haiku-20241022", 1000, 1000)
        sonnet = self.p.calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        assert haiku < sonnet

    def test_opus_most_expensive(self):
        haiku  = self.p.calculate_cost("claude-3-5-haiku-20241022", 1000, 1000)
        sonnet = self.p.calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        opus   = self.p.calculate_cost("claude-3-opus-20240229",     1000, 1000)
        assert haiku < sonnet < opus

    def test_unknown_model_returns_zero(self):
        assert self.p.calculate_cost("claude-99-ultra", 1000, 1000) == 0.0

    def test_zero_tokens_zero_cost(self):
        assert self.p.calculate_cost("claude-3-5-sonnet-20241022", 0, 0) == 0.0


# AnthropicProvider name                                              

class TestAnthropicProviderName:

    def test_provider_name(self):
        assert AnthropicProvider().name == "anthropic"


# End-to-end flow                                                     

class TestAnthropicProviderEndToEnd:

    def setup_method(self):
        self.p = AnthropicProvider()

    def test_full_flow_sonnet(self):
        resp = make_anthropic_response(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000, output_tokens=500,
        )
        assert self.p.detect(resp)
        inp, out = self.p.extract_usage(resp)
        model    = self.p.extract_model(resp)
        cost     = self.p.calculate_cost(model, inp, out)

        assert inp   == 1000
        assert out   == 500
        assert model == "claude-3-5-sonnet-20241022"
        assert cost  > 0.0

    def test_full_flow_haiku(self):
        resp = make_anthropic_response(
            model="claude-3-5-haiku-20241022",
            input_tokens=500, output_tokens=250,
        )
        assert self.p.detect(resp)
        inp, out = self.p.extract_usage(resp)
        model    = self.p.extract_model(resp)
        cost     = self.p.calculate_cost(model, inp, out)

        assert inp  == 500
        assert out  == 250
        assert cost > 0.0


# Provider registry — both providers detected correctly               

class TestProviderRegistry:
    """Verify that the _PROVIDERS list in decorator/context/interceptor
    correctly routes OpenAI vs Anthropic responses."""

    def test_openai_detected_by_openai_provider(self):
        from infertrack.providers.openai import OpenAIProvider
        p = OpenAIProvider()
        assert p.detect(make_openai_response()) is True
        assert p.detect(make_anthropic_response()) is False

    def test_anthropic_detected_by_anthropic_provider(self):
        p = AnthropicProvider()
        assert p.detect(make_anthropic_response()) is True
        assert p.detect(make_openai_response()) is False

    def test_decorator_registry_has_both_providers(self):
        from infertrack.core.decorator import _PROVIDERS
        names = [p.name for p in _PROVIDERS]
        assert "openai"    in names
        assert "anthropic" in names

    def test_context_registry_has_both_providers(self):
        from infertrack.core.context import _PROVIDERS
        names = [p.name for p in _PROVIDERS]
        assert "openai"    in names
        assert "anthropic" in names

    def test_interceptor_registry_has_both_providers(self):
        # interceptor stores providers as a module-level list _PROVIDERS
        # added by the Day 8 patch; verify by checking the source directly
        import infertrack.core.interceptor as imod
        providers = getattr(imod, "_PROVIDERS", None)
        assert providers is not None, "_PROVIDERS not found in interceptor"
        names = [p.name for p in providers]
        assert "openai"    in names
        assert "anthropic" in names

    def test_decorator_routes_anthropic_response(self):
        """@watchdog must log provider='anthropic' for Anthropic responses."""
        from pathlib import Path
        import tempfile
        from infertrack.core.decorator import watchdog
        from infertrack.storage.db import init_db, query_logs

        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "test.db"
            init_db(db)

            @watchdog(db_path=db)
            def ask():
                return make_anthropic_response(
                    model="claude-3-5-sonnet-20241022",
                    input_tokens=100, output_tokens=50
                )

            ask()
            log = query_logs(db_path=db)[0]
            assert log.provider == "anthropic"
            assert log.model    == "claude-3-5-sonnet-20241022"
            assert log.input_tokens  == 100
            assert log.output_tokens == 50
            assert log.cost_usd > 0.0