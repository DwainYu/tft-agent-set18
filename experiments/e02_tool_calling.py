"""Experiment 02 — one tool call, round tripped.

Goal: watch a tool call become a message again. The model never computes
anything; it emits a request, we execute it, and the result is fed back as the
next prompt.

    python3 experiments/e02_tool_calling.py
    DEEPSEEK_API_KEY=... python3 experiments/e02_tool_calling.py --real
"""

from __future__ import annotations

import sys

from _common import banner, pick_provider, registry, report

from agent.loop import Agent, AgentConfig
from agent.trace import Trace

TASK = "What is (17 + 28) * 4? Use the calculator tool, then answer in one line."

SCRIPT = [
    {"tool_calls": [{"name": "calculator", "arguments": {"expression": "(17 + 28) * 4"}}]},
    {"content": "(17 + 28) * 4 = 180.", "completion_tokens": 10},
]


def main(argv: list) -> int:
    banner("E02 · one tool round trip", "Model → tool → observation → model. Nothing else is magic.")
    trace = Trace(run_id="e02-tool-calling")
    agent = Agent(
        provider=pick_provider(argv, SCRIPT, TASK),
        registry=registry(),
        config=AgentConfig(max_turns=4, token_budget=2_000),
        on_event=trace.handler(),
    )
    result = agent.run(TASK)
    trace.finish(stop_reason=result.stop_reason, tool_calls=result.tool_calls)

    print(f"\nanswer        : {result.final}")
    print(f"stop reason   : {result.stop_reason}")
    print(f"observations  : {sum(1 for message in result.messages if message.role == 'tool')}")
    print(
        "\nlesson        : the tool result entered the next request as a message; the model "
        "answered because it could read the number, not because it computed it."
    )
    report(result, trace.timeline())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
