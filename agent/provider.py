"""Providers: the only place that talks to a model.

A scripted provider is not a toy — it is what makes the loop, the failure
modes and the tests deterministic. The real provider adds network, retries and
error classes, which is the difference a mock can never teach you.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .messages import Message, ToolCall, assistant

PROTOCOL_NOTE = (
    "You are an agent that must finish a task step by step. "
    "Call a tool when you need information you do not have; never invent results. "
    "When the answer is known, reply with plain text and no tool call."
)


class ProviderError(Exception):
    """A model call that could not be completed."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    raw: Dict[str, Any] = field(default_factory=dict)


class Provider:
    """Interface every provider implements."""

    name = "provider"

    def complete(self, messages: Sequence[Message], tools: Sequence[Dict[str, Any]]) -> Completion:
        raise NotImplementedError


class ScriptedProvider(Provider):
    """Replays a list of assistant turns, so experiments are reproducible.

    Each entry may be a plain string (final answer) or a dict with `content`
    and/or `tool_calls`, which is exactly the shape an API returns.
    """

    name = "scripted"

    def __init__(self, script: List[Dict[str, Any]], latency_ms: int = 0) -> None:
        self._script = script
        self._index = 0
        self._latency_ms = latency_ms
        self.calls: List[List[Dict[str, Any]]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[Dict[str, Any]]) -> Completion:
        self.calls.append([message.to_dict() for message in messages])
        if self._index >= len(self._script):
            raise ProviderError("script exhausted: the model kept asking for tools")
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)
        entry = self._script[self._index]
        self._index += 1
        calls = [
            ToolCall(
                id=call.get("id") or f"call_{self._index}_{position}",
                name=call["name"],
                arguments=call.get("arguments") or {},
            )
            for position, call in enumerate(entry.get("tool_calls") or [])
        ]
        return Completion(
            message=assistant(entry.get("content", ""), calls),
            usage=Usage(entry.get("prompt_tokens", 0), entry.get("completion_tokens", 0)),
            finish_reason="tool_calls" if calls else "stop",
        )


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible /chat/completions endpoint (DeepSeek, vLLM, ...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{model}@{self.base_url}"
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ProviderError("no API key: set DEEPSEEK_API_KEY or drop --real")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.usage = Usage()

    def _request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Optional[str] = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:400]
                # 4xx other than rate limit will not fix themselves by retrying.
                if error.code < 500 and error.code != 429:
                    raise ProviderError(f"HTTP {error.code}: {detail}") from error
                last_error = f"HTTP {error.code}: {detail}"
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = str(error)
            time.sleep(0.5 * 2**attempt)  # exponential backoff
        raise ProviderError(f"giving up after {self.max_attempts} attempts: {last_error}")

    def complete(self, messages: Sequence[Message], tools: Sequence[Dict[str, Any]]) -> Completion:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        data = self._request(body)
        choice = (data.get("choices") or [{}])[0]
        message = Message.from_dict(choice.get("message") or {"role": "assistant"})
        raw_usage = data.get("usage") or {}
        usage = Usage(
            int(raw_usage.get("prompt_tokens", 0)), int(raw_usage.get("completion_tokens", 0))
        )
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        return Completion(message, usage, choice.get("finish_reason") or "stop", data)


def deepseek(model: str = "deepseek-chat") -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=model,
    )


def build_provider(argv: Sequence[str]) -> Provider:
    """`--real` uses the API, otherwise the deterministic script."""
    if "--real" in argv:
        return deepseek(os.environ.get("AGENT_MODEL", "deepseek-chat"))
    raise SystemExit(
        "mock provider needs a script; pass --real with DEEPSEEK_API_KEY, "
        "or call the experiment module from tests"
    )
