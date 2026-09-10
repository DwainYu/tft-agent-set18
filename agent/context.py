"""Context policy: what actually goes into the next request.

A model has a fixed window, so a runtime needs a cheap token estimate and a
trim rule. Dropping the oldest turns is lossy on purpose — the lesson is that
lossy is a decision you have to make explicitly.
"""

from __future__ import annotations

from typing import List, Sequence

from .messages import Message

# ~4 characters per token is a fine heuristic for English/code; the point is
# not accuracy, it is having *a* number to drive a decision.
CHARS_PER_TOKEN = 4


def estimate_tokens(messages: Sequence[Message]) -> int:
    total = 0
    for message in messages:
        total += len(message.content) // CHARS_PER_TOKEN
        for call in message.tool_calls:
            total += (len(call.name) + len(str(call.arguments))) // CHARS_PER_TOKEN
    return total + len(messages) * 4  # per-message framing overhead


def trim(messages: Sequence[Message], budget: int, keep_recent: int = 6) -> List[Message]:
    """Keep the system prompt, drop the oldest non-system turns until it fits."""
    if not messages:
        return []
    head, body = messages[:1], list(messages[1:])
    while estimate_tokens(list(head) + body) > budget and len(body) > keep_recent:
        body.pop(0)
    if estimate_tokens(list(head) + body) > budget:
        # Even the recent window is too large: drop everything but the last turn.
        dropped = len(body) - 1
        body = body[-1:] if dropped > 0 else body
        head = list(head) + [Message("system", f"[context trimmed: {dropped} earlier turns dropped]")]
    return list(head) + body
