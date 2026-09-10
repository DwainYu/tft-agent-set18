# Runtime architecture

Five modules, one direction of dependency, no third-party imports.

```text
messages.py   ← everyone depends on this; it depends on nothing
   ↑
tools.py      provider.py        (execution | model access, independent of each other)
   ↑              ↑
   └──── loop.py ──┘              (state, turn accounting, stop conditions)
             ↑
        context.py  trace.py      (policy and observability, called by loop)
```

## The layers and why each one is separate

| Layer | File | Responsibility | What it must not know |
|---|---|---|---|
| Wire format | `messages.py` | `Message`, `ToolCall`, `ToolSpec` and their dict shape | any provider, any tool |
| Model access | `provider.py` | one `complete(messages, tools) -> Completion`; HTTP, retries, usage | tools, loops, budget |
| Execution | `tools.py` | schema validation + invoking a python function | the model, the transcript |
| State & control | `loop.py` | the turn, the guards, stop reasons | transport, tool internals |
| Policy | `context.py` | estimate + trim | provider, tools |
| Evidence | `trace.py` | JSONL events, timeline | everything else |

The provider is the only module that knows HTTP exists. The registry is the
only one that knows how to run code. The loop knows neither, which is the point:
a scripted provider and a live DeepSeek endpoint drive exactly the same loop,
and swapping one for the other is a constructor argument in `experiments/_common.py`.

## Where the complexity actually went

After the loop itself (about 200 lines), the code is all guards and accounting:

- `AgentConfig` — four budgets, one policy flag
- the repeat guard — `_key(call)` and a counter, ~10 lines
- `Tool.invoke` — required/unexpected argument validation, ~10 lines
- `OpenAICompatProvider._request` — backoff on 429/5xx, hard-fail on other 4xx
- `trim` — one while loop

None of that is agent intelligence. All of it is what makes a run debuggable,
and none of it is optional in a production system. This is the empirical
finding I came here for: the loop is trivial, the runtime is the work.

## What a framework would give me that this does not

Deliberately unbuilt, and the next thing to study:

- Checkpointing: `messages` lives in memory only; a crash loses the transcript
- Interrupts: no way for a human to approve a tool call before it runs
- Streaming: `complete()` returns one message, so partial tool/text events do
  not exist
- Reducers / parallel branches: every turn appends strictly; nothing merges
- Replay: the JSONL trace records but cannot re-run

LangGraph's headline features map almost one-to-one onto that list.
