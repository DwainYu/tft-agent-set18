"""A hand-written agent runtime: provider, tools, loop, context, trace."""

from .context import estimate_tokens, trim
from .loop import Agent, AgentConfig, AgentResult, Step
from .messages import Message, ToolCall, ToolSpec, assistant, system, user
from .provider import (
    Completion,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    ScriptedProvider,
    Usage,
    modelscope,
)
from .tools import Tool, ToolError, ToolRegistry, calculator, default_tools
from .trace import Trace

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Completion",
    "Message",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "ScriptedProvider",
    "Step",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "Trace",
    "Usage",
    "assistant",
    "calculator",
    "default_tools",
    "modelscope",
    "estimate_tokens",
    "system",
    "trim",
    "user",
]
