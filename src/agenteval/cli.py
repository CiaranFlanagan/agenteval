"""Command line interface."""

from __future__ import annotations

import argparse
import sys

from .runner import Comparison, Suite, compare
from .store import Store

_GREEN, _RED, _DIM, _RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agenteval", description=__doc__)
    parser.add_argument("--db", default="agenteval.db", help="path to the trace database")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser("compare", help="diff two recorded versions of an agent")
    p_cmp.add_argument("agent")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--candidate", required=True)
    p_cmp.add_argument("--suite", help="suite file, required for accuracy")
    p_cmp.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero if any metric regressed (for CI)",
    )

    p_ls = sub.add_parser("versions", help="list recorded versions")
    p_ls.add_argument("agent", nargs="?")

    args = parser.parse_args(argv)

    with Store(args.db) as store:
        if args.command == "versions":
            versions = store.versions(args.agent)
            if not versions:
                print("no runs recorded", file=sys.stderr)
                return 1
            for v in versions:
                print(v)
            return 0

        suite = Suite.load(args.suite) if args.suite else None
        try:
            result = compare(store, args.agent, args.baseline, args.candidate, suite)
        except ValueError as exc:
            print(f"agenteval: {exc}", file=sys.stderr)
            return 1

        print(render(result))
        return 1 if args.fail_on_regression and not result.passed else 0


def render(c: Comparison, color: bool = True) -> str:
    """Format a comparison as a table."""
    g, r, d, x = (_GREEN, _RED, _DIM, _RESET) if color else ("", "", "", "")

    lines = [
        "",
        f"  {c.agent}: {c.baseline_version} -> {c.candidate_version}",
        "",
        f"  {'metric':<20} {'baseline':>12} {'candidate':>12} {'delta':>12}",
        f"  {'-' * 60}",
    ]

    for name, delta in c.deltas.items():
        if delta.baseline is None or delta.candidate is None:
            continue
        rel = delta.relative
        change = f"{rel:+.1%}" if rel is not None else f"{delta.absolute:+.3f}"
        if delta.absolute == 0:
            mark, tint = " ", d
        elif delta.is_regression:
            mark, tint = "x", r
        else:
            mark, tint = "+", g
        lines.append(
            f"  {name:<20} {delta.baseline.format():>12} {delta.candidate.format():>12} "
            f"{tint}{change:>12} {mark}{x}"
        )

    if c.newly_failing:
        lines += ["", f"  {r}{len(c.newly_failing)} newly failing:{x}"]
        lines += [f"    {case_id:<12} {why}" for case_id, why in c.newly_failing]

    verdict = f"{g}no regressions{x}" if c.passed else f"{r}{len(c.regressions)} regressions{x}"
    lines += ["", f"  {verdict}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
