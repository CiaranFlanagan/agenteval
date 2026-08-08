import pytest

from agenteval import Store
from agenteval.metrics import Metric, compute, percentile
from agenteval.models import Run, Span
from agenteval.runner import Suite, compare


def make_run(version, case_id, output, *, latency_ms=100.0, cost=0.01, error=None, tool_error=None):
    from datetime import timedelta

    run = Run(name="agent", version=version, case_id=case_id, output=output, error=error)
    run.ended_at = run.started_at + timedelta(milliseconds=latency_ms)

    tool = Span(name="tool", kind="tool", run_id=run.run_id, cost_usd=cost, error=tool_error)
    tool.ended_at = run.ended_at
    run.spans.append(tool)
    return run


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_percentile_is_not_interpolated():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) in values
    assert percentile(values, 95) == 40.0
    assert percentile([], 95) == 0.0


def test_compute_reports_core_metrics():
    runs = [make_run("v1", "a", "yes", latency_ms=100), make_run("v1", "b", "no", latency_ms=300)]
    m = compute(runs, expected={"a": "yes", "b": "yes"})

    assert m["runs"].value == 2
    assert m["accuracy"].value == 0.5
    assert m["accuracy"].higher_is_better
    assert not m["p95_latency"].higher_is_better
    assert m["cost_per_run"].value == pytest.approx(0.01)


def test_compute_omits_accuracy_without_expectations():
    assert "accuracy" not in compute([make_run("v1", "a", "yes")])


def test_compute_handles_no_runs():
    assert compute([]) == {}


def test_compare_flags_regression_by_direction(store):
    for case in ("a", "b"):
        store.save_run(make_run("v1", case, "yes", latency_ms=100, cost=0.01))
    # Candidate is more accurate but much slower and pricier -- the exact
    # tradeoff that goes unnoticed without this tool.
    store.save_run(make_run("v2", "a", "yes", latency_ms=800, cost=0.05))
    store.save_run(make_run("v2", "b", "yes", latency_ms=900, cost=0.05))

    result = compare(store, "agent", "v1", "v2", Suite("s", []))

    assert result.deltas["p95_latency"].is_regression
    assert result.deltas["cost_per_run"].is_regression
    assert not result.passed


def test_compare_detects_newly_failing_cases(store):
    store.save_run(make_run("v1", "a", "correct"))
    store.save_run(make_run("v1", "b", "correct"))
    store.save_run(make_run("v2", "a", "correct"))
    store.save_run(make_run("v2", "b", "wrong"))

    suite = Suite("s", [])
    suite.cases = [
        type("C", (), {"id": "a", "expected": "correct"})(),
        type("C", (), {"id": "b", "expected": "correct"})(),
    ]

    result = compare(store, "agent", "v1", "v2", suite)

    assert [case_id for case_id, _ in result.newly_failing] == ["b"]
    assert not result.passed


def test_compare_ignores_cases_that_were_already_failing(store):
    store.save_run(make_run("v1", "a", "wrong"))
    store.save_run(make_run("v2", "a", "wrong"))

    suite = Suite("s", [])
    suite.cases = [type("C", (), {"id": "a", "expected": "correct"})()]

    assert compare(store, "agent", "v1", "v2", suite).newly_failing == []


def test_compare_requires_recorded_runs(store):
    store.save_run(make_run("v1", "a", "yes"))
    with pytest.raises(ValueError, match="no recorded runs"):
        compare(store, "agent", "v1", "missing")


def test_delta_relative_handles_zero_baseline(store):
    store.save_run(make_run("v1", "a", "yes", cost=0.0))
    store.save_run(make_run("v2", "a", "yes", cost=0.5))

    result = compare(store, "agent", "v1", "v2")
    assert result.deltas["cost_per_run"].relative is None
    assert result.deltas["cost_per_run"].is_regression


def test_metric_formats_by_unit():
    assert Metric("m", 0.5, True, "%").format() == "50.00%"
    assert Metric("m", 0.0123, False, "$").format() == "$0.0123"
    assert Metric("m", 250.0, False, "ms").format() == "250ms"
