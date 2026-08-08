from datetime import timedelta

import pytest

from agenteval import Store
from agenteval.cli import main, render
from agenteval.models import Run, Span
from agenteval.runner import compare


def make_run(version, case_id, output, *, latency_ms=100.0, cost=0.01):
    run = Run(name="agent", version=version, case_id=case_id, output=output)
    run.ended_at = run.started_at + timedelta(milliseconds=latency_ms)
    tool = Span(name="tool", kind="tool", run_id=run.run_id, cost_usd=cost)
    tool.ended_at = run.ended_at
    run.spans.append(tool)
    return run


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    with Store(path) as s:
        s.save_run(make_run("v1", "a", "yes", latency_ms=100, cost=0.01))
        s.save_run(make_run("v2", "a", "yes", latency_ms=900, cost=0.05))
    return str(path)


def test_render_marks_regressions(db):
    with Store(db) as s:
        result = compare(s, "agent", "v1", "v2")

    out = render(result, color=False)
    assert "v1 -> v2" in out
    assert "p95_latency" in out
    assert "regressions" in out


def test_compare_command_succeeds(db, capsys):
    assert main(["--db", db, "compare", "agent", "--baseline", "v1", "--candidate", "v2"]) == 0
    assert "v1 -> v2" in capsys.readouterr().out


def test_fail_on_regression_exits_nonzero(db):
    code = main(
        [
            "--db",
            db,
            "compare",
            "agent",
            "--baseline",
            "v1",
            "--candidate",
            "v2",
            "--fail-on-regression",
        ]
    )
    assert code == 1


def test_missing_version_reports_error(db, capsys):
    code = main(["--db", db, "compare", "agent", "--baseline", "v1", "--candidate", "nope"])
    assert code == 1
    assert "no recorded runs" in capsys.readouterr().err


def test_versions_command(db, capsys):
    assert main(["--db", db, "versions", "agent"]) == 0
    assert capsys.readouterr().out.split() == ["v1", "v2"]


def test_versions_command_with_no_data(tmp_path, capsys):
    code = main(["--db", str(tmp_path / "empty.db"), "versions"])
    assert code == 1
    assert "no runs recorded" in capsys.readouterr().err
