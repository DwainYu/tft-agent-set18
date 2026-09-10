"""Experiment 04 — failure modes, on purpose.

Goal: see the five ways a run can end badly, and what the runtime does about
each. These are not hypothetical: they are the failures a real agent hits in
its first week.

    python3 experiments/e04_failure_modes.py
"""

from __future__ import annotations

import sys

from _common import banner, registry

from agent.loop import Agent, AgentConfig
from agent.provider import ScriptedProvider
from agent.tools import Tool, ToolRegistry


def run(name: str, script: list, tools: ToolRegistry, config: AgentConfig) -> str:
    agent = Agent(provider=ScriptedProvider(script), registry=tools, config=config)
    result = agent.run("Show me what happens when the model gets it wrong.")
    print(f"\n{name}")
    print(f"  ok          : {result.ok}")
    print(f"  stop_reason : {result.stop_reason}")
    print(f"  turns/tools : {result.turns}/{result.tool_calls}")
    for step in result.steps:
        if not step.ok:
            print(f"  failure     : {step.detail}")
    if result.final:
        print(f"  recovery    : {result.final}")
    return result.stop_reason


def crashing_tool(**_kwargs) -> str:  # noqa: ANN003 - registered with no parameters
    raise RuntimeError("database connection died")


def main(argv: list) -> int:
    banner("E04 · failure modes", "Bad arguments, unknown tools, a stuck loop, exhausted budgets, a crash.")
    tools = registry()
    tight = AgentConfig(max_turns=4, token_budget=2_000)
    reasons = {}

    reasons["bad-arguments"] = run(
        "1. invalid tool arguments (required key missing)",
        [
            {"tool_calls": [{"name": "calculator", "arguments": {}}]},
            {"content": "I need an expression to compute anything."},
        ],
        tools,
        tight,
    )

    reasons["unknown-tool"] = run(
        "2. a tool that does not exist",
        [
            {"tool_calls": [{"name": "web_search", "arguments": {"q": "agents"}}]},
            {"content": "That tool is unavailable; I will answer from what I know."},
        ],
        tools,
        tight,
    )

    reasons["stuck-loop"] = run(
        "3. the same call repeated (the classic runaway)",
        [
            {"tool_calls": [{"name": "now", "arguments": {}}]},
            {"tool_calls": [{"name": "now", "arguments": {}}]},
            {"tool_calls": [{"name": "now", "arguments": {}}]},
            {"content": "Repeating myself gets nowhere, so here is the answer."},
        ],
        tools,
        AgentConfig(max_turns=6, repeat_limit=2, token_budget=2_000),
    )

    reasons["provider-error"] = run(
        "4. the model keeps asking with no scripted turn left",
        [
            {"tool_calls": [{"id": "a", "name": "now", "arguments": {}}]},
            {"tool_calls": [{"id": "b", "name": "now", "arguments": {}}]},
        ],
        tools,
        AgentConfig(max_turns=8, repeat_limit=9, token_budget=2_000),
    )

    reasons["tool-budget"] = run(
        "5. more tool calls than the run may spend",
        [
            {
                "tool_calls": [
                    {"id": "a", "name": "now", "arguments": {}},
                    {"id": "b", "name": "now", "arguments": {}},
                ]
            },
            {"tool_calls": [{"id": "c", "name": "now", "arguments": {}}]},
        ],
        tools,
        AgentConfig(max_turns=6, max_tool_calls=2, repeat_limit=9, token_budget=2_000),
    )

    reasons["tool-crash"] = run(
        "6. a tool that raises instead of returning",
        [
            {"tool_calls": [{"name": "crash", "arguments": {}}]},
            {"content": "The tool died; I am reporting that rather than retrying."},
        ],
        ToolRegistry([Tool("crash", "Always crashes", {"type": "object", "properties": {}}, crashing_tool)]),
        tight,
    )

    print("\nstop reasons:")
    for case, reason in reasons.items():
        print(f"  {case:<16} → {reason}")
    print(
        "\nlesson        : every failure has to come back as text the model can read. Raising "
        "out of a tool loses the run; an observation keeps it recoverable — and the budgets are "
        "what stop a recoverable failure from becoming an expensive one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
