"""Aggregate metrics over a set of recorded runs.

Every metric declares whether higher is better. That is what lets the runner
label a change as an improvement or a regression without a human deciding each
time -- accuracy going up is good, latency going up is not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    higher_is_better: bool
    unit: str = ""

    def format(self) -> str:
        if self.unit == "$":
            return f"${self.value:.4f}"
        if self.unit == "%":
            return f"{self.value:.2%}"
        if self.unit == "ms":
            return f"{self.value:.0f}ms"
        return f"{self.value:.3f}"


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with the small sample sizes an eval suite
    produces, an interpolated p95 reports a latency no run actually had.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(p / 100 * len(ordered) + 0.5)))
    return ordered[rank - 1]


def compute(runs: list, expected: dict[str, str] | None = None) -> dict[str, Metric]:
    """Summarise a set of runs.

    ``expected`` maps case_id to the answer a run should have produced. Without
    it, accuracy is not reported -- there is nothing to grade against.
    """
    if not runs:
        return {}

    latencies = [r.duration_ms for r in runs]
    metrics: dict[str, Metric] = {
        "runs": Metric("runs", float(len(runs)), higher_is_better=True),
        "error_rate": Metric(
            "error_rate",
            sum(1 for r in runs if r.error) / len(runs),
            higher_is_better=False,
            unit="%",
        ),
        "p50_latency": Metric(
            "p50_latency", percentile(latencies, 50), higher_is_better=False, unit="ms"
        ),
        "p95_latency": Metric(
            "p95_latency", percentile(latencies, 95), higher_is_better=False, unit="ms"
        ),
        "cost_per_run": Metric(
            "cost_per_run",
            sum(r.total_cost_usd for r in runs) / len(runs),
            higher_is_better=False,
            unit="$",
        ),
        "tokens_per_run": Metric(
            "tokens_per_run",
            sum(r.total_tokens for r in runs) / len(runs),
            higher_is_better=False,
        ),
    }

    total_tool_calls = sum(len(r.tool_calls) for r in runs)
    if total_tool_calls:
        metrics["tool_error_rate"] = Metric(
            "tool_error_rate",
            sum(len(r.failed_tool_calls) for r in runs) / total_tool_calls,
            higher_is_better=False,
            unit="%",
        )
        metrics["tool_calls_per_run"] = Metric(
            "tool_calls_per_run", total_tool_calls / len(runs), higher_is_better=False
        )

    if expected:
        graded = [r for r in runs if r.case_id in expected]
        if graded:
            correct = sum(1 for r in graded if grade(r.output, expected[r.case_id]))
            metrics["accuracy"] = Metric(
                "accuracy", correct / len(graded), higher_is_better=True, unit="%"
            )

    return metrics


def grade(output: str | None, expected: str) -> bool:
    """Whether a run's output counts as correct.

    Substring containment, case-insensitive. This is a placeholder: it is
    honest for extraction-style tasks and wrong for anything open-ended.

    TODO: pluggable graders (exact, regex, LLM-as-judge) selected per case in
    the suite file.
    """
    if output is None:
        return False
    return expected.strip().lower() in output.strip().lower()
