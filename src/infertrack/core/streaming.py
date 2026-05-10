# src/infertrack/core/streaming.py
"""Streaming response support for infertrack.

When a caller uses ``stream=True``, the OpenAI SDK returns a
``Stream[ChatCompletionChunk]`` generator instead of a ``ChatCompletion``.
Token counts are only available in the **last chunk** when the caller
passes ``stream_options={"include_usage": True}``.

This module provides:
  - ``is_streaming_response(obj)``   — detect a stream
  - ``StreamingWrapper``             — passthrough generator that logs on completion

Usage inside decorator/interceptor::

    response = func(*args, **kwargs)
    if is_streaming_response(response):
        return StreamingWrapper(response, on_complete=_log_fn, t_start=t_start)
    else:
        # existing non-streaming path
        ...
"""
from __future__ import annotations

import time
from typing import Any, Callable, Generator, Iterator, Optional


# Detection                                                            

def is_streaming_response(obj: Any) -> bool:
    """Return True if *obj* looks like an OpenAI streaming response.

    We detect by duck-typing rather than importing openai types, so
    this works whether or not openai is installed.

    An OpenAI ``Stream`` object is iterable and has no ``.usage`` or
    ``.choices`` attribute at the top level (those are on the chunks).
    We specifically check for the openai Stream class name as a
    secondary signal to avoid false-positives on plain generators.
    """
    if obj is None:
        return False

    # Must be iterable but not a plain string/bytes/dict
    if isinstance(obj, (str, bytes, dict, list)):
        return False

    type_name = type(obj).__name__

    # openai SDK stream types
    if type_name in ("Stream", "AsyncStream"):
        return True

    # Duck-type fallback: iterable + has response attribute (openai Stream)
    # or __iter__ with no .usage at root level
    try:
        has_iter   = hasattr(obj, "__iter__") or hasattr(obj, "__anext__")
        no_usage   = not hasattr(obj, "usage")
        no_choices = not hasattr(obj, "choices")
        # Additional signal: openai streams have a .response attribute
        has_response = hasattr(obj, "response")
        return has_iter and no_usage and no_choices and has_response
    except Exception:
        return False


# Usage extraction from chunks                                         

def _extract_chunk_usage(chunk: Any) -> Optional[tuple[int, int]]:
    """Extract (input_tokens, output_tokens) from a stream chunk.

    Returns None if the chunk does not carry usage info.
    The usage appears on the last chunk when
    ``stream_options={"include_usage": True}`` was passed.
    """
    try:
        usage = chunk.usage
        if usage is None:
            return None
        input_tokens  = int(usage.prompt_tokens)
        output_tokens = int(usage.completion_tokens)
        return input_tokens, output_tokens
    except (AttributeError, TypeError, ValueError):
        return None


def _extract_chunk_model(chunk: Any) -> Optional[str]:
    """Extract model name from a stream chunk."""
    try:
        model = str(chunk.model).strip()
        return model if model and model != "None" else None
    except (AttributeError, TypeError):
        return None


# StreamingWrapper                                                     

class StreamingWrapper:
    """A transparent passthrough wrapper around an OpenAI stream.

    Yields every chunk unchanged while collecting usage from the final
    chunk. Calls ``on_complete`` with the collected metrics once the
    stream is exhausted or if an exception occurs.

    Usage::

        def _log(model, inp, out, cost, latency_ms, success, error):
            ...

        wrapped = StreamingWrapper(stream, on_complete=_log, t_start=time.perf_counter())
        for chunk in wrapped:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
        # on_complete is called automatically after the loop
    """

    def __init__(
        self,
        stream: Any,
        *,
        on_complete: Callable[[str, int, int, float, float, bool, Optional[str]], None],
        t_start: float,
    ) -> None:
        """
        Args:
            stream:      The raw OpenAI Stream object.
            on_complete: Callback called once stream is done.
                         Signature: (model, input_tokens, output_tokens,
                                     cost, latency_ms, success, error_msg)
            t_start:     ``time.perf_counter()`` value recorded before
                         the API call was made.
        """
        self._stream      = stream
        self._on_complete = on_complete
        self._t_start     = t_start

        # Collected during iteration
        self._model:         Optional[str] = None
        self._input_tokens:  int           = 0
        self._output_tokens: int           = 0
        self._usage_found:   bool          = False

    # Iteration                                                          

    def __iter__(self) -> Iterator[Any]:
        exc_caught: Optional[BaseException] = None
        try:
            for chunk in self._stream:
                # Collect model name from first chunk that has it
                if self._model is None:
                    self._model = _extract_chunk_model(chunk)

                # Collect usage from final chunk (non-None usage)
                usage = _extract_chunk_usage(chunk)
                if usage is not None:
                    self._input_tokens, self._output_tokens = usage
                    self._usage_found = True

                yield chunk

        except Exception as exc:
            exc_caught = exc
            raise
        finally:
            self._fire_callback(exc_caught)

    def _fire_callback(self, exc: Optional[BaseException]) -> None:
        """Compute latency and call on_complete."""
        latency_ms = (time.perf_counter() - self._t_start) * 1000
        model      = self._model or "unknown"
        success    = exc is None
        error_msg  = str(exc) if exc else None

        try:
            self._on_complete(
                model,
                self._input_tokens,
                self._output_tokens,
                latency_ms,
                success,
                error_msg,
            )
        except Exception:
            pass  # never let logging crash the caller

    # Forward other stream attributes transparently                     

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)