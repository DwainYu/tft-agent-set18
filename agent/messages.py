"""Message and tool-schema types shared by every experiment.

Deliberately tiny: an agent talks to a model with a list of dictionaries, and
the only interesting parts are tool calls and their results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass
class ToolCall:
    """A model's request to run a named tool with structured arguments."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Message":
        from json import loads

        calls = []
        for raw in payload.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments") or "{}"
            parsed = loads(arguments) if isinstance(arguments, str) else arguments
            calls.append(
                ToolCall(id=raw.get("id", ""), name=function.get("name", ""), arguments=parsed)
            )
        return cls(
            role=payload.get("role", "user"),
            content=payload.get("content") or "",
            tool_calls=calls,
            tool_call_id=payload.get("tool_call_id"),
            name=payload.get("name"),
        )


def system(text: str) -> Message:
    return Message("system", text)


def user(text: str) -> Message:
    return Message("user", text)


def assistant(text: str = "", tool_calls: Optional[List[ToolCall]] = None) -> Message:
    return Message("assistant", text, tool_calls or [])


def tool_result(call: ToolCall, content: str) -> Message:
    return Message("tool", content, tool_call_id=call.id, name=call.name)


@dataclass
class ToolSpec:
    """The JSON schema description a model needs to call a tool."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
