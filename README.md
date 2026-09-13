# tft-agent-set18 — Agent Engineering Training Ground

The place where I build agents by hand to find out what a framework is
actually doing for me. Not a product, not a portfolio piece: a gym.

Learning notes live in
[`agent-engineering-lab`](https://github.com/DwainYu/agent-engineering-lab)
(site: <https://dwainyu.github.io/agent-engineering-lab/>). Every experiment
here is referenced by a notebook there, and every note there points back at
code here.

Production work belongs to `tft-agent-set17`. Nothing experimental goes there.

## Rules of this repository

1. **No agent framework.** LangGraph, AutoGen, CrewAI are what I compare
   against later — you cannot judge them before writing the loop yourself.
2. **Standard library only** in `agent/`. Retries, JSON schema validation and
   token estimates are the lesson, not plumbing to delegate.
3. **Every claim runs.** Each behaviour I write about in the lab has a script
   or a test that demonstrates it here.
4. **Mock first, real second.** Every experiment is deterministic by default
   and takes `--real` to hit an actual endpoint.

## Layout

```text
agent/
  messages.py    Message / ToolCall / ToolSpec — the wire format
  provider.py    Provider interface, ScriptedProvider, OpenAI-compatible client
  tools.py       ToolRegistry, argument validation, a safe calculator
  loop.py        Agent.run: turn loop, budgets, repeat guard, error handling
  context.py     token estimate + trim policy
  trace.py       JSONL run trace and timeline rendering
experiments/
  e01_single_turn.py     one completion, no tools
  e02_tool_calling.py    one tool round trip
  e03_agent_loop.py      four turns of state-carrying loop
  e04_failure_modes.py   six ways a run ends badly
  e05_context_trim.py    what the model actually receives when history grows
tests/                   28 unittest cases, stdlib only
docs/architecture.md     why the layers are split this way
```

## Quickstart

```bash
python3 --version          # 3.10+
python3 experiments/e03_agent_loop.py
python3 experiments/e04_failure_modes.py
python3 -m unittest discover -s tests -t .
```

No install step, no virtualenv required: `experiments/*` put the repo root on
`sys.path`, so the code runs straight from a checkout.

### Against a real model

```bash
export MODELSCOPE_API_KEY=***                       # ModelScope SDK token
export MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1   # default
export AGENT_MODEL=Qwen/Qwen3.8-Flash-Next                          # default
python3 experiments/e02_tool_calling.py --real
```

Get the token at `https://modelscope.cn/my/myaccesstoken`. Any model the endpoint
serves works as long as it takes OpenAI-style `tools`; `Org/Model` is the id form.

The `--real` path exercises `OpenAICompatProvider`: HTTP, 429/5xx backoff,
`tool_calls` parsing, usage accounting. Same task, same loop, different
provider — that identity is the point of the exercise.

## Experiments

| # | Script | What it proves |
|---|---|---|
| 01 | `e01_single_turn.py` | A completion request *is* the whole state; nothing is remembered |
| 02 | `e02_tool_calling.py` | A tool result re-enters the conversation as a `tool` message |
| 03 | `e03_agent_loop.py` | Multi-step work costs context, not just steps |
| 04 | `e04_failure_modes.py` | Bad args, unknown tool, stuck loop, budgets, crash — each with its own stop reason |
| 05 | `e05_context_trim.py` | Trimming is lossy, so durable state must live outside the prompt |

Run `python3 experiments/e04_failure_modes.py` first if you want the fastest
overview: it prints the stop reason for six failure classes, which is most of
what an agent runtime exists to handle.

## Traces

Any experiment driven through `Agent` emits structured events:

```text
  0.000s assistant        I will compute that
  0.001s tool_result      calculator 180
  0.002s assistant        (17 + 28) * 4 = 180.
```

`Trace(path="traces/demo.jsonl")` writes one JSON object per line, so a failed
run can be replayed against the exact prompt the model saw.

## Verification

```bash
python3 -m unittest discover -s tests -t .   # 28 tests
python3 -m compileall -q agent experiments tests
```

## Commit convention

`exp: <experiment id> <what changed>` for experiment and runtime code,
`test: <area>` for tests. Learning reflections do not live here — they go in
the lab's `docs/daily/`, which links back to the commit sha.
