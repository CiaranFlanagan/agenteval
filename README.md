# agenteval

Tracing and regression evaluation for LLM agents.

You changed a prompt. Did it get better? The usual answer is "it seemed fine when I tried it twice." `agenteval` records every run, then diffs two versions and tells you what moved — including the metrics you weren't watching, which is usually where the damage is.

## The problem

An agent change is never one-dimensional. A prompt that improves accuracy can quietly triple latency and quadruple cost, and you find out from the bill. Aggregate numbers can also stay flat while individual cases break.

## Usage

Instrument the agent entry point with a version tag:

```python
from agenteval import Store, configure, trace, span

configure(Store("agenteval.db"))

@trace(version="v2")
def git_explain(question: str) -> str:
    with span("retrieve", kind="tool"):
        commits = blame(question)

    with span("gemini", kind="llm",
              input_tokens=1_200, output_tokens=300, cost_usd=0.0018):
        return explain(commits)
```

Run your suite, tag a new version, run it again, then compare:

```
$ agenteval compare git-explain --baseline v1 --candidate v2 --suite examples/suite.yaml

  git-explain: v1 -> v2

  metric                   baseline    candidate        delta
  ------------------------------------------------------------
  accuracy                   82.00%       89.00%       +8.5% +
  tool_error_rate            11.00%        4.00%      -63.6% +
  p95_latency                 3200ms       7800ms     +143.8% x
  cost_per_run              $0.0110      $0.0380     +245.5% x

  1 newly failing:
    case-47      output did not match expected

  2 regressions
```

More accurate, three times slower, four times more expensive. That is the output the tool exists to produce.

## CI

`--fail-on-regression` exits non-zero, so a prompt change that degrades latency or cost fails the build like any other regression:

```yaml
- run: agenteval compare git-explain --baseline main --candidate ${{ github.sha }} \
         --suite evals/suite.yaml --fail-on-regression
```

## Design notes

**Two-symbol API.** A `@trace` decorator and a `span` context manager. Anything more invasive and people stop instrumenting.

**Provider-agnostic.** Token counts and cost are passed in rather than inferred, so the library wraps no particular SDK. The caller reports what its provider returned.

**ContextVars, not globals.** Concurrent runs in one process don't interleave their spans.

**Spans outside a run are inert.** Instrumented helpers get called from untraced paths; that must not raise, or the instrumentation gets removed.

**Percentiles are not interpolated.** At eval-suite sample sizes, an interpolated p95 reports a latency no run actually had.

**Metrics declare their direction.** Each metric knows whether higher is better, which is what lets regressions be labelled without a human judging each one.

## Development

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check src tests
```

## Status

Early. Working: tracing, SQLite storage, metric aggregation, version comparison, regression detection, CLI.

Not yet: pluggable graders (currently substring matching only — honest for extraction tasks, wrong for open-ended ones), replay execution, a web UI, OpenTelemetry export.

## License

MIT
