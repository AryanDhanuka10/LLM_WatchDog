# tests/unit/test_streaming.py
"""
Day 10 tests: streaming response support.
All tests use fake stream objects — no real OpenAI calls.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from infertrack.core.streaming import (
    is_streaming_response,
    StreamingWrapper,
    _extract_chunk_usage,
    _extract_chunk_model,
)
from infertrack.core.decorator import watchdog
from infertrack.storage.db import init_db, query_logs


# Fixtures                                                             

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


# Fake stream helpers                                                  

def make_chunk(
    content: str = "",
    model: str = "gpt-4o",
    usage=None,
) -> MagicMock:
    """Build a fake ChatCompletionChunk."""
    chunk = MagicMock()
    chunk.model = model
    chunk.choices[0].delta.content = content
    chunk.usage = usage
    return chunk


def make_usage_chunk(
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build the final chunk that carries usage info."""
    chunk = make_chunk(content="", model=model)
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens     = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    return chunk


class FakeStream:
    """Minimal object that looks like an openai.Stream."""

    def __init__(self, chunks, *, model="gpt-4o"):
        self._chunks = chunks
        self.model   = model
        # These attributes make is_streaming_response() return True
        self.response = MagicMock()   # openai streams have .response

    def __iter__(self):
        yield from self._chunks

    def __class_getitem__(cls, item):
        return cls


def make_fake_stream(
    texts: list[str],
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    include_usage: bool = True,
) -> FakeStream:
    """Build a fake stream with text chunks + optional usage final chunk."""
    chunks = [make_chunk(t, model=model) for t in texts]
    if include_usage:
        chunks.append(make_usage_chunk(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))
    return FakeStream(chunks, model=model)


def make_non_streaming_response(model="gpt-4o"):
    """Regular ChatCompletion mock — must NOT be detected as streaming."""
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    resp.choices[0].message.content = "Hello"
    return resp


# is_streaming_response                                               

class TestIsStreamingResponse:

    def test_fake_stream_detected(self):
        stream = make_fake_stream(["hi"])
        assert is_streaming_response(stream) is True

    def test_regular_response_not_streaming(self):
        resp = make_non_streaming_response()
        assert is_streaming_response(resp) is False

    def test_none_not_streaming(self):
        assert is_streaming_response(None) is False

    def test_string_not_streaming(self):
        assert is_streaming_response("hello") is False

    def test_dict_not_streaming(self):
        assert is_streaming_response({"model": "gpt-4o"}) is False

    def test_list_not_streaming(self):
        assert is_streaming_response([1, 2, 3]) is False

    def test_plain_generator_without_response_attr(self):
        def gen():
            yield 1
        # Plain generator has no .response attribute → not streaming
        assert is_streaming_response(gen()) is False

    def test_object_named_stream_detected(self):
        """Objects whose class is named 'Stream' are always detected."""
        class Stream:
            def __iter__(self): return iter([])
        obj = Stream()
        assert is_streaming_response(obj) is True


# _extract_chunk_usage                                                

class TestExtractChunkUsage:

    def test_extracts_usage_from_final_chunk(self):
        chunk = make_usage_chunk(prompt_tokens=15, completion_tokens=25)
        result = _extract_chunk_usage(chunk)
        assert result == (15, 25)

    def test_returns_none_for_chunk_without_usage(self):
        chunk = make_chunk("hello")
        chunk.usage = None
        assert _extract_chunk_usage(chunk) is None

    def test_returns_none_for_missing_usage_attr(self):
        chunk = MagicMock(spec=["choices", "model"])
        assert _extract_chunk_usage(chunk) is None

    def test_zero_tokens_valid(self):
        chunk = make_usage_chunk(prompt_tokens=0, completion_tokens=0)
        assert _extract_chunk_usage(chunk) == (0, 0)


# _extract_chunk_model                                                

class TestExtractChunkModel:

    def test_extracts_model(self):
        chunk = make_chunk(model="gpt-4o-mini")
        assert _extract_chunk_model(chunk) == "gpt-4o-mini"

    def test_strips_whitespace(self):
        chunk = make_chunk(model="  gpt-4o  ")
        assert _extract_chunk_model(chunk) == "gpt-4o"

    def test_returns_none_for_empty_model(self):
        chunk = make_chunk(model="")
        assert _extract_chunk_model(chunk) is None

    def test_returns_none_for_none_model(self):
        chunk = MagicMock(spec=["choices", "usage"])
        assert _extract_chunk_model(chunk) is None


# StreamingWrapper                                                    

class TestStreamingWrapperPassthrough:

    def test_yields_all_chunks(self):
        stream  = make_fake_stream(["Hello", " world", "!"])
        calls   = []
        wrapper = StreamingWrapper(stream, on_complete=lambda *a: None,
                                   t_start=0.0)
        for chunk in wrapper:
            calls.append(chunk)
        # 3 text chunks + 1 usage chunk = 4 total
        assert len(calls) == 4

    def test_chunk_content_preserved(self):
        stream  = make_fake_stream(["Hello", " world"])
        texts   = []
        wrapper = StreamingWrapper(stream, on_complete=lambda *a: None,
                                   t_start=0.0)
        for chunk in wrapper:
            texts.append(chunk.choices[0].delta.content)
        assert "Hello" in texts
        assert " world" in texts

    def test_getattr_forwarded_to_stream(self):
        stream  = make_fake_stream(["hi"])
        stream.some_custom_attr = "value"
        wrapper = StreamingWrapper(stream, on_complete=lambda *a: None,
                                   t_start=0.0)
        assert wrapper.some_custom_attr == "value"


class TestStreamingWrapperCallback:

    def test_on_complete_called_after_exhaustion(self):
        stream   = make_fake_stream(["a", "b"])
        received = {}

        def on_complete(model, inp, out, latency_ms, success, error_msg):
            received.update(dict(model=model, inp=inp, out=out,
                                 latency_ms=latency_ms, success=success,
                                 error_msg=error_msg))

        wrapper = StreamingWrapper(stream, on_complete=on_complete, t_start=0.0)
        list(wrapper)   # exhaust

        assert received["model"]   == "gpt-4o"
        assert received["inp"]     == 10
        assert received["out"]     == 20
        assert received["success"] is True
        assert received["error_msg"] is None

    def test_latency_positive(self):
        import time
        stream   = make_fake_stream(["hi"])
        received = {}

        def on_complete(model, inp, out, latency_ms, success, error):
            received["latency_ms"] = latency_ms

        import time as t
        t_start = t.perf_counter()
        wrapper = StreamingWrapper(stream, on_complete=on_complete,
                                   t_start=t_start)
        list(wrapper)
        assert received["latency_ms"] >= 0

    def test_on_complete_called_on_exception(self):
        class BrokenStream:
            response = MagicMock()
            def __iter__(self):
                yield make_chunk("hi")
                raise ConnectionError("dropped")

        received = {}

        def on_complete(model, inp, out, latency_ms, success, error):
            received.update(success=success, error=error)

        wrapper = StreamingWrapper(BrokenStream(), on_complete=on_complete,
                                   t_start=0.0)

        with pytest.raises(ConnectionError):
            list(wrapper)

        assert received["success"] is False
        assert "dropped" in received["error"]

    def test_no_usage_in_stream_gives_zero_tokens(self):
        """Stream without stream_options usage → tokens=0, still logged."""
        stream   = make_fake_stream(["hi"], include_usage=False)
        received = {}

        def on_complete(model, inp, out, latency_ms, success, error):
            received.update(inp=inp, out=out)

        wrapper = StreamingWrapper(stream, on_complete=on_complete, t_start=0.0)
        list(wrapper)

        assert received["inp"] == 0
        assert received["out"] == 0

    def test_model_extracted_from_first_chunk(self):
        stream   = make_fake_stream(["hello"], model="gpt-4o-mini")
        received = {}

        def on_complete(model, inp, out, latency_ms, success, error):
            received["model"] = model

        wrapper = StreamingWrapper(stream, on_complete=on_complete, t_start=0.0)
        list(wrapper)

        assert received["model"] == "gpt-4o-mini"


# @watchdog with streaming                                            

class TestWatchdogStreaming:

    def test_streaming_returns_wrapper(self, tmp_db):
        stream = make_fake_stream(["Hello"])

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        result = ask()
        assert isinstance(result, StreamingWrapper)

    def test_db_empty_before_stream_consumed(self, tmp_db):
        """Log must NOT be written until the stream is fully consumed."""
        stream = make_fake_stream(["Hello"])

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        ask()   # returns wrapper, does NOT exhaust stream yet
        assert len(query_logs(db_path=tmp_db)) == 0

    def test_db_written_after_stream_consumed(self, tmp_db):
        stream = make_fake_stream(["Hello", " world"],
                                  prompt_tokens=10, completion_tokens=20)

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        wrapper = ask()
        list(wrapper)   # exhaust

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1

    def test_tokens_logged_correctly(self, tmp_db):
        stream = make_fake_stream(["hi"],
                                  prompt_tokens=15, completion_tokens=30)

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        list(ask())

        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  == 15
        assert log.output_tokens == 30

    def test_model_logged_correctly(self, tmp_db):
        stream = make_fake_stream(["hi"], model="gpt-4o-mini")

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].model == "gpt-4o-mini"

    def test_ollama_stream_zero_cost(self, tmp_db):
        stream = make_fake_stream(["hi"], model="qwen2.5:0.5b",
                                  prompt_tokens=10, completion_tokens=20)

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0

    def test_paid_model_stream_has_cost(self, tmp_db):
        stream = make_fake_stream(["hi"], model="gpt-4o",
                                  prompt_tokens=1000, completion_tokens=500)

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].cost_usd > 0.0

    def test_tag_stored_on_streaming_log(self, tmp_db):
        stream = make_fake_stream(["hi"])

        @watchdog(tag="stream-test", db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].tag == "stream-test"

    def test_user_id_stored_on_streaming_log(self, tmp_db):
        stream = make_fake_stream(["hi"])

        @watchdog(user_id="alice", db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].user_id == "alice"

    def test_success_true_on_clean_stream(self, tmp_db):
        stream = make_fake_stream(["hi"])

        @watchdog(db_path=tmp_db)
        def ask():
            return stream

        list(ask())
        assert query_logs(db_path=tmp_db)[0].success is True

    def test_stream_exception_logged(self, tmp_db):
        class FailingStream:
            response = MagicMock()
            def __iter__(self):
                yield make_chunk("partial")
                raise RuntimeError("connection dropped")

        @watchdog(db_path=tmp_db)
        def ask():
            return FailingStream()

        with pytest.raises(RuntimeError):
            list(ask())

        log = query_logs(db_path=tmp_db)[0]
        assert log.success is False
        assert "connection dropped" in log.error_msg

    def test_non_streaming_unaffected(self, tmp_db):
        """Regular (non-streaming) calls must still work exactly as before."""
        resp = make_non_streaming_response(model="gpt-4o")

        @watchdog(db_path=tmp_db)
        def ask():
            return resp

        result = ask()
        assert result is resp

        log = query_logs(db_path=tmp_db)[0]
        assert log.model == "gpt-4o"
        assert not isinstance(result, StreamingWrapper)