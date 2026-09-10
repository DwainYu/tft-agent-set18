"""A tool registry the model can call into.

Three rules this file exists to teach:
1. The model never runs code; it asks for a named tool with JSON arguments.
2. Every argument the model produces is untrusted until validated.
3. A tool failure must come back as an observation, not an exception.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .messages import ToolSpec


class ToolError(Exception):
    """A failure safe to show the model: the message is the observation."""


BinOp = Callable[[Any, Any], Any]
_OPS: Dict[type, BinOp] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UnaryOps: Dict[type, BinOp] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UnaryOps:
        return _UnaryOps[type(node.op)](_eval_node(node.operand))
    raise ToolError("only arithmetic on numbers is allowed")


def calculator(expression: str) -> float:
    """Evaluate arithmetic without eval(): a whitelisted AST walk."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolError(f"invalid expression: {error.msg}") from error
    return _eval_node(tree)


def _arg(arguments: Dict[str, Any], key: str) -> Any:
    if key not in arguments:
        raise ToolError(f"missing required argument: {key}")
    return arguments[key]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def invoke(self, arguments: Dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            raise ToolError(f"{self.name}: arguments must be an object")
        required = self.parameters.get("required") or []
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolError(f"{self.name}: missing argument(s) {', '.join(missing)}")
        allowed = set((self.parameters.get("properties") or {}).keys())
        unexpected = [key for key in arguments if key not in allowed]
        if unexpected:
            raise ToolError(f"{self.name}: unexpected argument(s) {', '.join(unexpected)}")
        result = self.handler(**arguments)
        return result if isinstance(result, str) else str(result)


class ToolRegistry:
    def __init__(self, tools: List[Tool]) -> None:
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [tool.spec().to_dict() for tool in self._tools.values()]

    def invoke(self, call) -> str:  # noqa: ANN001 - ToolCall
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"unknown tool: {call.name}")
        return tool.invoke(call.arguments)


def default_tools() -> List[Tool]:
    """The smallest set that still exercises routing, validation and state."""
    from datetime import datetime, timezone

    def now(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        return datetime.now(timezone.utc).strftime(format)

    notes: Dict[str, str] = {}

    def note_put(key: str, value: str) -> str:
        notes[key] = value
        return f"stored {key}"

    def note_get(key: str) -> str:
        if key not in notes:
            raise ToolError(f"no note stored under {key!r}")
        return notes[key]

    return [
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression. Returns a number.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '(3 + 4) * 12'",
                    }
                },
                "required": ["expression"],
            },
            handler=calculator,
        ),
        Tool(
            name="now",
            description="Current UTC time as a formatted string.",
            parameters={
                "type": "object",
                "properties": {
                    "format": {"type": "string", "description": "strftime format"}
                },
            },
            handler=now,
        ),
        Tool(
            name="note_put",
            description="Persist a short fact under a key for later steps.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            handler=note_put,
        ),
        Tool(
            name="note_get",
            description="Read back a fact previously stored with note_put.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            handler=note_get,
        ),
    ]
