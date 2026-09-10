"""Shared scaffolding for the experiments: path setup, provider choice, printing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run these files directly, no install step
    sys.path.insert(0, str(ROOT))

from agent.loop import AgentResult  # noqa: E402
from agent.provider import Provider, ScriptedProvider, deepseek  # noqa: E402
from agent.tools import ToolRegistry, default_tools  # noqa: E402


def banner(title: str, goal: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{goal}\n{'=' * 72}")


def registry() -> ToolRegistry:
    return ToolRegistry(default_tools())


def pick_provider(argv: Sequence[str], script: Sequence[dict], real_task: str) -> Provider:
    """`--real` hits the API with the same task; the default is deterministic."""
    if "--real" in argv:
        provider = deepseek()
        print(f"[real] {provider.name} — task: {real_task!r}")
        return provider
    return ScriptedProvider(list(script))


def report(result: AgentResult, trace_timeline: str) -> None:
    print("\nsteps\n" + result.summary())
    print("\ntrace\n" + trace_timeline)
    print("\nmessages sent back to the model")
    for index, message in enumerate(result.messages, start=1):
        calls = ""
        if message.tool_calls:
            calls = "  tool_calls=" + ", ".join(
                f"{call.name}({call.arguments})" for call in message.tool_calls
            )
        print(f"  {index:>2}. {message.role:<9} {message.content[:96]!r}{calls}")
