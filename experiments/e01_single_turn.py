"""Experiment 01 — one model turn, no tools.

Goal: see exactly what a chat-completions request is made of, and that a model
call is stateless. The "conversation" is only a list we rebuild every time.

    python3 experiments/e01_single_turn.py            # scripted, deterministic
    python3 experiments/e01_single_turn.py --real     # needs DEEPSEEK_API_KEY
"""

from __future__ import annotations

import sys

from _common import banner, pick_provider, registry

from agent.messages import system, user
from agent.provider import ProviderError
from agent.tools import ToolRegistry, default_tools

TASK = "Reply with one short sentence explaining what an LLM agent is."

SCRIPT = [{"content": "An agent is a model that acts through tools until a task is done.", "completion_tokens": 12}]


def main(argv: list) -> int:
    banner("E01 · one model turn", "What does a single completion request/response actually look like?")
    provider = pick_provider(argv, SCRIPT, TASK)
    tools = ToolRegistry(default_tools())
    messages = [system("You reply in one short sentence."), user(TASK)]

    print(f"provider      : {provider.name}")
    print(f"tools offered : {len(tools.specs())} ({', '.join(tools.names)})")
    print(f"messages      : {len(messages)}")
    for message in messages:
        print(f"  - {message.role}: {message.content}")

    try:
        completion = provider.complete(messages, tools.specs())
    except ProviderError as error:  # a real key problem shows up here, not in the loop
        print(f"\ncall failed  : {error}")
        return 1

    print("\nresponse")
    print(f"  role         : {completion.message.role}")
    print(f"  content      : {completion.message.content}")
    print(f"  tool_calls   : {len(completion.message.tool_calls)}")
    print(f"  finish_reason: {completion.finish_reason}")
    print(f"  usage        : {completion.usage}")
    print(
        "\nlesson        : the request is the whole state. Nothing is remembered "
        "between calls, so 'memory' is just us rebuilding this list."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
