"""Compare two recorded versions of an agent and flag regressions.

The unit of comparison is a version tag. You run a suite, tag it ``v1``, change
a prompt, run it again as ``v2``, and this tells you what moved -- including
the things you were not looking at, which is usually where the damage is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .metrics import Metric, compute, grade
from .models import Run
from .store import Store


@dataclass
class Case:
    """One test input and the answer it should produce."""

    id: str
    input: dict
    expected: str


@dataclass
class Suite:
    name: str
    cases: list[Case]

    @property
    def expected(self) -> dict[str, str]:
        return {c.id: c.expected for c in self.cases}

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or "cases" not in data:
            raise ValueError(f"{path}: suite must be a mapping with a 'cases' key")

        cases = []
        for i, raw in enumerate(data["cases"]):
            missing = {"id", "expected"} - raw.keys()
            if missing:
                raise ValueError(f"{path}: case {i} missing {sorted(missing)}")
            cases.append(
                Case(id=str(raw["id"]), input=raw.get("input", {}), expected=str(raw["expected"]))
            )

        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate case ids")

        return cls(name=data.get("name", Path(path).stem), cases=cases)


@dataclass
class Delta:
    """How one metric moved between two versions."""

    name: str
    baseline: Metric | None
    candidate: Metric | None

    @property
    def absolute(self) -> float:
        if self.baseline is None or self.candidate is None:
            return 0.0
        return self.candidate.value - self.baseline.value

    @property
    def relative(self) -> float | None:
        """Fractional change, or None when the baseline is zero."""
        if self.baseline is None or self.candidate is None or self.baseline.value == 0:
            return None
        return self.absolute / self.baseline.value

    @property
    def is_regression(self) -> bool:
        if self.baseline is None or self.candidate is None or self.absolute == 0:
            return False
        improved = self.absolute > 0 if self.candidate.higher_is_better else self.absolute < 0
        return not improved


@dataclass
class Comparison:
    agent: str
    baseline_version: str
    candidate_version: str
    deltas: dict[str, Delta]
    newly_failing: list[tuple[str, str]]  # (case_id, why)

    @property
    def regressions(self) -> list[Delta]:
        return [d for d in self.deltas.values() if d.is_regression]

    @property
    def passed(self) -> bool:
        return not self.regressions and not self.newly_failing


def compare(
    store: Store,
    agent: str,
    baseline_version: str,
    candidate_version: str,
    suite: Suite | None = None,
) -> Comparison:
    """Diff two recorded versions of an agent."""
    expected = suite.expected if suite else None
    baseline_runs = store.runs(name=agent, version=baseline_version)
    candidate_runs = store.runs(name=agent, version=candidate_version)

    if not baseline_runs:
        raise ValueError(f"no recorded runs for {agent} version {baseline_version!r}")
    if not candidate_runs:
        raise ValueError(f"no recorded runs for {agent} version {candidate_version!r}")

    base_metrics = compute(baseline_runs, expected)
    cand_metrics = compute(candidate_runs, expected)

    deltas = {
        name: Delta(name, base_metrics.get(name), cand_metrics.get(name))
        for name in sorted(base_metrics.keys() | cand_metrics.keys())
    }

    return Comparison(
        agent=agent,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        deltas=deltas,
        newly_failing=_newly_failing(baseline_runs, candidate_runs, expected),
    )


def _newly_failing(
    baseline: list[Run], candidate: list[Run], expected: dict[str, str] | None
) -> list[tuple[str, str]]:
    """Cases that passed on the baseline and fail on the candidate.

    Aggregate metrics can stay flat while individual cases break, so this is
    reported separately rather than folded into accuracy.
    """

    def outcome(runs: list[Run]) -> dict[str, tuple[bool, str]]:
        out: dict[str, tuple[bool, str]] = {}
        for r in runs:
            if r.case_id is None:
                continue
            if r.error:
                out[r.case_id] = (False, r.error)
            elif expected and r.case_id in expected:
                ok = grade(r.output, expected[r.case_id])
                out[r.case_id] = (ok, "" if ok else "output did not match expected")
            else:
                out[r.case_id] = (True, "")
        return out

    before, after = outcome(baseline), outcome(candidate)
    return [
        (case_id, why)
        for case_id, (ok, why) in sorted(after.items())
        if not ok and before.get(case_id, (False, ""))[0]
    ]
