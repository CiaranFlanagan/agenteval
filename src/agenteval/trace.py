"""Instrumentation: capture agent runs without changing how they are written.

The public surface is deliberately two things -- a ``@trace`` decorator around
the agent entry point, and a ``span`` context manager for the interesting work
inside it. Anything more invasive and people stop instrumenting.

Nesting is tracked with a ContextVar rather than a global, so concurrent runs
in the same process do not interleave their spans.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from .models import Run, Span
from .store import Store

_current_run: ContextVar[Run | None] = ContextVar("current_run", default=None)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)

_store: Store | None = None


def configure(store: Store) -> None:
    """Point tracing at a store. Without this, tracing is a no-op."""
    global _store
    _store = store


def current_run() -> Run | None:
    """The run being traced on this context, if any."""
    return _current_run.get()


def trace(_fn: Callable | None = None, *, name: str | None = None, version: str = "dev"):
    """Record every invocation of an agent entry point.

    ``version`` is what later gets compared: tag a prompt or model change with
    a new version and the eval runner can diff the two.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = Run(name=name or fn.__name__, version=version, inputs=_describe(args, kwargs))
            run_token = _current_run.set(run)
            span_token = _current_span.set(None)
            try:
                result = fn(*args, **kwargs)
                run.output = str(result) if result is not None else None
                return result
            except Exception as exc:
                run.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                run.ended_at = datetime.now(UTC)
                _current_span.reset(span_token)
                _current_run.reset(run_token)
                if _store is not None:
                    _store.save_run(run)

        return wrapper

    return decorator(_fn) if _fn is not None else decorator


@contextmanager
def span(
    name: str,
    kind: str = "other",
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    **attributes: Any,
) -> Iterator[Span]:
    """Record one unit of work inside a traced run.

    Token counts and cost are passed in rather than inferred: this library does
    not wrap any particular SDK, so the caller reports what its provider
    returned. Yields the span so callers can amend it after the fact.
    """
    run = _current_run.get()
    if run is None:
        # Outside a traced run there is nothing to attach to. Yield a detached
        # span so instrumented code still runs rather than blowing up.
        yield Span(name=name, kind=kind, run_id="")
        return

    parent = _current_span.get()
    s = Span(
        name=name,
        kind=kind,
        run_id=run.run_id,
        parent_id=parent.span_id if parent else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        attributes=dict(attributes),
    )
    run.spans.append(s)

    token = _current_span.set(s)
    try:
        yield s
    except Exception as exc:
        s.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        s.ended_at = datetime.now(UTC)
        _current_span.reset(token)


def _describe(args: tuple, kwargs: dict) -> dict:
    """Summarise call arguments for storage.

    Values are truncated and stringified because inputs can be entire
    documents, and a trace store that grows without bound is useless.
    """
    out: dict[str, Any] = {}
    for i, a in enumerate(args):
        out[f"arg{i}"] = _truncate(a)
    for k, v in kwargs.items():
        out[k] = _truncate(v)
    return out


def _truncate(value: Any, limit: int = 500) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"
