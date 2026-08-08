import pytest

from agenteval import Store, configure, span, trace


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    configure(s)
    yield s
    s.close()


def test_records_a_run(store):
    @trace(version="v1")
    def agent(question):
        return f"answer to {question}"

    assert agent("why?") == "answer to why?"

    runs = store.runs()
    assert len(runs) == 1
    assert runs[0].name == "agent"
    assert runs[0].version == "v1"
    assert runs[0].output == "answer to why?"
    assert runs[0].error is None
    assert runs[0].duration_ms >= 0


def test_records_spans_with_cost_and_tokens(store):
    @trace(version="v1")
    def agent():
        with span("gemini", kind="llm", input_tokens=100, output_tokens=50, cost_usd=0.002):
            pass
        with span("git blame", kind="tool"):
            pass
        return "done"

    agent()

    run = store.runs()[0]
    assert len(run.spans) == 2
    assert run.total_tokens == 150
    assert run.total_cost_usd == pytest.approx(0.002)
    assert len(run.tool_calls) == 1


def test_records_exceptions_and_reraises(store):
    @trace(version="v1")
    def agent():
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError, match="api down"):
        agent()

    run = store.runs()[0]
    assert run.error == "RuntimeError: api down"
    assert run.ended_at is not None


def test_span_error_does_not_lose_the_run(store):
    @trace(version="v1")
    def agent():
        try:
            with span("flaky tool", kind="tool"):
                raise ValueError("malformed json")
        except ValueError:
            pass
        return "recovered"

    agent()

    run = store.runs()[0]
    assert run.output == "recovered"
    assert run.error is None
    assert len(run.failed_tool_calls) == 1
    assert "malformed json" in run.failed_tool_calls[0].error


def test_nested_spans_record_parent(store):
    @trace(version="v1")
    def agent():
        # Kept nested rather than combined: the nesting is what's under test.
        with span("outer") as outer:  # noqa: SIM117
            with span("inner") as inner:
                assert inner.parent_id == outer.span_id
        return "ok"

    agent()

    run = store.runs()[0]
    outer, inner = run.spans[0], run.spans[1]
    assert outer.parent_id is None
    assert inner.parent_id == outer.span_id


def test_span_outside_a_run_is_inert():
    # Instrumented helpers get called from untraced code paths; that must not
    # raise, or people remove the instrumentation.
    with span("orphan", kind="tool") as s:
        assert s.run_id == ""


def test_truncates_large_inputs(store):
    @trace(version="v1")
    def agent(document):
        return "ok"

    agent("x" * 5000)

    stored = store.runs()[0].inputs["arg0"]
    assert len(stored) < 600
    assert "5000 chars" in stored
