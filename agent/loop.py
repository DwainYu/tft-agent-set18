"""The agent loop: model → tools → observations → model, until it stops.

This is the whole idea written out longhand before any framework hides it:
  * state is a message list
  * the model only ever produces text or tool calls
  * tool results are appended as observations and become the next prompt
  * a runtime is mostly stop conditions and failure handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .context import trim
from .messages import Message, ToolCall, system, user
from .provider import Provider, ProviderError
from .tools import ToolError, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent that finishes tasks step by step. Call a tool when you need "
    "information you do not have; never invent a result. When the answer is known, "
    "reply with plain text and no tool call."
)

EventHandler = Callable[[str, Dict[str, Any]], None]


def preview(value: Any, limit: int = 70) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class AgentConfig:
    max_turns: int = 8
    max_tool_calls: int = 12
    repeat_limit: int = 2
    token_budget: int = 4_000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    fail_on_tool_error: bool = False


@dataclass
class Step:
    turn: int
    kind: str  # "tool" | "final" | "error"
    detail: str
    ok: bool = True
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "kind": self.kind, "name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class AgentResult:
    ok: bool
    final: str
    stop_reason: str
    steps: List[Step] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        header = (
            f"stop_reason={self.stop_reason} ok={self.ok} turns={self.turns} "
            f"tool_calls={self.tool_calls} tokens={self.total_tokens}"
        )
        rows = [
            f"  turn {step.turn:>2} {step.kind:<5} {'' if step.ok else 'FAILED '}"
            f"{(step.name + ' ') if step.name else ''}{step.detail}".rstrip()
            for step in self.steps
        ]
        return "\n".join([header] + rows)


class Agent:
    """One provider, one tool registry, one message list, and the guards."""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
        on_event: Optional[EventHandler] = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.config = config or AgentConfig()
        self.on_event = on_event

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)

    def run(self, task: str, history: Sequence[Message] = ()) -> AgentResult:
        config = self.config
        messages: List[Message] = [system(config.system_prompt), *history, user(task)]
        result = AgentResult(ok=False, final="", stop_reason="unfinished", messages=messages)
        seen_calls: List[str] = []

        for turn in range(1, config.max_turns + 1):
            result.turns = turn
            request = trim(messages, config.token_budget)
            if len(request) < len(messages):
                self._emit("context_trimmed", {"turn": turn, "sent_messages": len(request)})
            try:
                completion = self.provider.complete(request, self.registry.specs())
            except ProviderError as error:
                result.steps.append(Step(turn, "error", preview(str(error), 160), ok=False))
                result.stop_reason = "provider-error"
                self._emit("error", {"turn": turn, "message": str(error)})
                return result

            result.prompt_tokens += completion.usage.prompt_tokens
            result.completion_tokens += completion.usage.completion_tokens
            reply = completion.message
            messages.append(reply)
            self._emit(
                "assistant",
                {"turn": turn, "content": preview(reply.content, 160), "tool_calls": len(reply.tool_calls)},
            )

            if not reply.tool_calls:
                result.ok = True
                result.final = reply.content
                result.stop_reason = "final-answer"
                result.steps.append(Step(turn, "final", preview(reply.content, 160)))
                return result

            blocked = False
            for call in reply.tool_calls:
                if result.tool_calls >= config.max_tool_calls:
                    result.stop_reason = "tool-budget"
                    result.steps.append(Step(turn, "error", "tool call budget exhausted", ok=False))
                    return result
                result.tool_calls += 1
                blocked = self._dispatch(turn, call, seen_calls, messages, result) or blocked

            if blocked and config.fail_on_tool_error:
                result.stop_reason = "tool-error"
                return result

        result.stop_reason = "max-turns"
        result.steps.append(Step(result.turns, "error", "turn budget exhausted", ok=False))
        return result

    def _dispatch(
        self,
        turn: int,
        call: ToolCall,
        seen_calls: List[str],
        messages: List[Message],
        result: AgentResult,
    ) -> bool:
        """Run one tool call, append the observation, return True on failure."""
        signature = f"{call.name}:{sorted((k, str(v)) for k, v in call.arguments.items())}"
        repeats = seen_calls.count(signature)
        seen_calls.append(signature)
        if repeats >= self.config.repeat_limit:
            observation = (
                f"REFUSED: identical call to {call.name} already made {repeats} times; "
                "the loop is stuck. Change approach or answer directly."
            )
            result.steps.append(Step(turn, "tool", preview(observation, 160), ok=False, name=call.name))
            self._emit("tool_blocked", {"turn": turn, "name": call.name, "repeats": repeats + 1})
            messages.append(_observation(call, observation))
            return True

        try:
            observation = self.registry.invoke(call)
        except ToolError as error:
            failed = True
            observation = f"ERROR: {error}"
            self._emit("tool_error", {"turn": turn, "name": call.name, "error": str(error)})
        except Exception as error:  # noqa: BLE001 - a tool crash must not crash the run
            failed = True
            observation = f"ERROR: {call.name} raised {type(error).__name__}: {error}"
            self._emit("tool_crash", {"turn": turn, "name": call.name, "error": str(error)})
        else:
            failed = False
            self._emit(
                "tool_result",
                {"turn": turn, "name": call.name, "arguments": call.arguments, "result": preview(observation)},
            )
        result.steps.append(
            Step(turn, "tool", f"{preview(call.arguments)} -> {preview(observation)}", ok=not failed, name=call.name)
        )
        messages.append(_observation(call, observation))
        return failed


def _observation(call: ToolCall, content: str) -> Message:
    return Message("tool", content, tool_call_id=call.id, name=call.name)
