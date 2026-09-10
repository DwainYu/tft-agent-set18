"""Experiment 05 — context trimming.

Goal: watch the request the model actually receives shrink while the
conversation keeps growing. This is where "it worked in the notebook and died
in production" usually starts.

    python3 experiments/e05_context_trim.py
"""

from __future__ import annotations

import sys

from _common import banner, pick_provider, registry, report

from agent.context import estimate_tokens, trim
from agent.loop import Agent, AgentConfig
from agent.messages import ToolCall, assistant, system, user
from agent.trace import Trace

TASK = "Using the stored order_total, confirm the final figure."

SCRIPT = [
    {"tool_calls": [{"name": "note_get", "arguments": {"key": "order_total"}}]},
    {"content": "order_total is 100.0.", "completion_tokens": 6},
]

# A fake past: twenty turns of tool chatter nobody would delete by hand.
HISTORY = []
for index in range(20):
    HISTORY.append(user(f"step {index}: " + "context " * 24))
    HISTORY.append(assistant(f"acknowledged step {index}: " + "padding " * 22))


def main(argv: list) -> int:
    banner("E05 · context trimming", "Same task, a long history, and a budget that forces a decision.")
    budget = 900
    whole = [system("You are an agent.")] + HISTORY + [user(TASK)]
    provider = pick_provider(argv, SCRIPT, TASK)
    tools = registry()
    # A fact stored in an earlier session: it lives in tool state, not in context.
    tools.invoke(ToolCall(id="seed", name="note_put", arguments={"key": "order_total", "value": "100.0"}))

    print(f"history       : {len(HISTORY)} messages")
    print(f"full estimate : {estimate_tokens(whole)} tokens (budget {budget})")
    sent_now = trim(whole, budget)
    print(f"trimmed       : {len(sent_now)} messages, {estimate_tokens(sent_now)} tokens")
    print(f"kept          : roles={[message.role for message in sent_now][:3]}...")

    trace = Trace(run_id="e05-context-trim")
    agent = Agent(
        provider=provider,
        registry=tools,
        config=AgentConfig(max_turns=4, token_budget=budget),
        on_event=trace.handler(),
    )
    result = agent.run(TASK, history=HISTORY)
    trace.finish(stop_reason=result.stop_reason)

    print(f"\nanswer        : {result.final}")
    print(f"stop reason   : {result.stop_reason}")
    trimmed_events = [event for event in trace.events["events"] if event["kind"] == "context_trimmed"]
    print(f"trim events   : {len(trimmed_events)} of {result.turns} turns had to drop context")

    first_request = provider.calls[0] if hasattr(provider, "calls") else []
    print(f"request #1    : {len(first_request)} messages actually sent")
    print(f"first message : {first_request[0]['content'][:60] if first_request else 'n/a'}")
    print(
        "\nlesson        : trimming is lossy. The two oldest turns are gone, which means an "
        "agent that needs a fact from turn 3 has to have written it somewhere durable first."
    )
    report(result, trace.timeline())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
