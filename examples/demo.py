"""End-to-end demo: record two versions of an agent, then diff them.

Runs a fake agent so the demo needs no API key. v2 is the change you would
actually ship -- it is more accurate -- and the point is what else moved.

    python examples/demo.py && agenteval compare demo-agent \
        --baseline v1 --candidate v2 --suite examples/demo-suite.yaml
"""

import random
import time

from agenteval import Store, configure, span, trace

CASES = [f"case-{i}" for i in range(20)]


def build(version: str, *, accuracy: float, latency_s: float, cost: float, tool_failure: float):
    @trace(name="demo-agent", version=version)
    def agent(case_id: str) -> str:
        with span("retrieve", kind="tool") as s:
            time.sleep(latency_s * 0.3)
            if random.random() < tool_failure:
                s.error = "ToolError: malformed JSON in response"

        with span("generate", kind="llm", input_tokens=1200, output_tokens=300, cost_usd=cost):
            time.sleep(latency_s * 0.7)

        return "correct" if random.random() < accuracy else "wrong"

    return agent


def record(agent, version: str, store: Store) -> None:
    for case_id in CASES:
        run_ids_before = {r.run_id for r in store.runs(name="demo-agent", version=version)}
        agent(case_id)
        # Tag the run just written with the case it came from.
        for r in store.runs(name="demo-agent", version=version):
            if r.run_id not in run_ids_before:
                r.case_id = case_id
                store.save_run(r)


def main() -> None:
    random.seed(7)
    store = Store("agenteval.db")
    configure(store)

    # v1: the baseline.
    record(build("v1", accuracy=0.80, latency_s=0.03, cost=0.011, tool_failure=0.15), "v1", store)

    # v2: a bigger model with a longer prompt. More accurate, far slower and
    # pricier -- exactly the tradeoff that goes unnoticed without measurement.
    record(build("v2", accuracy=0.95, latency_s=0.09, cost=0.038, tool_failure=0.05), "v2", store)

    store.close()
    print("recorded 20 runs each for v1 and v2 in agenteval.db")


if __name__ == "__main__":
    main()
