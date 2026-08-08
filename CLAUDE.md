# CLAUDE.md

## What this is

Tracing and regression evaluation for LLM agents. Records every agent run, then diffs two versions and flags what moved — including the metrics nobody was watching, which is where the damage usually is.

## Build and test

Python 3.13 is installed and on the default PATH. Tooling lives in a local venv, **not** on the global PATH:

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'

.venv/bin/pytest --cov=agenteval --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

Coverage is 96%; don't let it slide. CI runs against 3.11, 3.12, and 3.13, so avoid syntax newer than 3.11.

End-to-end demo (no API key needed):

```bash
.venv/bin/python examples/demo.py
.venv/bin/agenteval compare demo-agent --baseline v1 --candidate v2 --suite examples/demo-suite.yaml
```

## Module layout

| Module | Responsibility |
|---|---|
| `models.py` | `Run` and `Span` dataclasses; cost and tokens roll up from spans |
| `trace.py` | `@trace` decorator and `span` context manager |
| `store.py` | SQLite persistence, indexed on `(name, version)` |
| `metrics.py` | aggregation; each metric declares its direction |
| `runner.py` | suite loading, version comparison, regression detection |
| `cli.py` | `compare` and `versions` commands |

## Decisions already made — don't undo these

- **The public API is two symbols**, `@trace` and `span`. Anything more invasive and people stop instrumenting. Resist adding required setup.
- **Provider-agnostic.** Tokens and cost are passed in, never inferred. This library wraps no SDK.
- **ContextVars, not globals**, so concurrent runs don't interleave spans.
- **A `span` outside a traced run yields a detached span rather than raising.** Instrumented helpers get called from untraced paths; code that explodes there gets its instrumentation deleted.
- **Percentiles are nearest-rank, not interpolated.** At eval-suite sample sizes an interpolated p95 reports a latency no run actually had.
- **Metrics carry `higher_is_better`.** That's what lets a change be labelled improvement vs. regression without a human judging each one.
- **Newly-failing cases are reported separately from aggregate metrics.** Accuracy can hold steady while individual cases break.

## Known gaps

- **`trace.py` is sync-only.** Most production agents are async, so the library currently can't instrument the thing it exists for. **Most limiting flaw.**
- **No statistical significance.** With 20 cases a 2% accuracy delta is noise but gets reported as a change. Confidence intervals would make the "regression" label mean something.
- **`metrics.grade()` is substring matching only.** Honest for extraction tasks, wrong for anything open-ended. Wants pluggable graders selected per case.
- No replay execution — versions are tagged by hand, and `examples/demo.py` does it with an awkward re-save.
- No SDK adapters, OpenTelemetry export, sampling, or web UI.

## Conventions

- Type hints everywhere; `from __future__ import annotations` at the top.
- Tests use `tmp_path` fixtures; no shared state between tests.
- Comments explain *why*. Match the existing density.
- Keep `README.md`'s "Status" section honest as things land.

## Related

Listed in the PROJECTS section of `~/resume`. If you change what this repo does, check whether the resume bullet is still accurate — the 96% coverage figure in particular is quoted there.
