"""Experiment 03 — a loop with state across turns.

Goal: make the loop do something no single call can — carry a fact through
three tool calls and only then answer. Memory here is a tool, not a feature.

    python3 experiments/e03_agent_loop.py
    DEEPSEEK_API_KEY=... python3 experiments/e03_agent_loop.py --real
"""

from __future__ import annotations

import sys

from _common import banner, pick_provider, registry, report

from agent.loop import Agent, AgentConfig
from agent.trace import Trace

TASK = (
    "Compute 12.5 * 8, store the result under the key 'order_total', read it back "
    "to be sure, then tell me the value."
)

SCRIPT = [
    {"tool_calls": [{"name": "calculator", "arguments": {"expression": "12.5 * 8"}}]},
    {
        "tool_calls": [
            {"name": "note_put", "arguments": {"key": "order_total", "value": "100.0"}}
        ]
    },
    {"tool_calls": [{"name": "note_get", "arguments": {"key": "order_total"}}]},
    {"content": "The order total is 100.0.", "completion_tokens": 8},
]


def main(argv: list) -> int:
    banner("E03 · multi-step loop with state", "Four turns, three tool calls, one answer that needs all of them.")
    trace = Trace(run_id="e03-agent-loop")
    agent = Agent(
        provider=pick_provider(argv, SCRIPT, TASK),
        registry=registry(),
        config=AgentConfig(max_turns=6, token_budget=3_000),
        on_event=trace.handler(),
    )
    result = agent.run(TASK)
    trace.finish(stop_reason=result.stop_reason, turns=result.turns)

    print(f"\nanswer        : {result.final}")
    print(f"turns         : {result.turns}")
    print(f"tool calls    : {result.tool_calls}")
    print(f"messages      : {len(result.messages)} (grows every turn — that is the cost of a loop)")
    print(f"tokens        : {result.total_tokens}")
    print(
        "\nlesson        : each turn re-sends the whole history, so context growth is the "
        "real budget of an agent, not the number of steps."
    )
    report(result, trace.timeline())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
