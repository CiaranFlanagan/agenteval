"""Core data types for traced agent runs.

A Run is one end-to-end agent invocation. It contains Spans, which are the
individual units of work inside it -- an LLM call, a tool call, a retrieval.
Cost and token counts roll up from spans to the run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Span:
    """One unit of work inside a run."""

    name: str
    kind: str  # "llm" | "tool" | "retrieval" | "other"
    run_id: str
    span_id: str = field(default_factory=_new_id)
    parent_id: str | None = None
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Elapsed time, or 0.0 if the span never closed (i.e. it crashed)."""
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000


@dataclass
class Run:
    """One end-to-end agent invocation."""

    name: str
    version: str
    run_id: str = field(default_factory=_new_id)
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    inputs: dict = field(default_factory=dict)
    output: str | None = None
    error: str | None = None
    spans: list[Span] = field(default_factory=list)
    # Set by an eval suite when the run is replayed against a known-good answer.
    case_id: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    @property
    def tool_calls(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "tool"]

    @property
    def failed_tool_calls(self) -> list[Span]:
        return [s for s in self.tool_calls if s.error is not None]
