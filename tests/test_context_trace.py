"""Context policy and trace records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.context import estimate_tokens, trim
from agent.messages import assistant, system, tool_result, user
from agent.trace import Trace


class Estimate(unittest.TestCase):
    def test_grows_with_content(self) -> None:
        short = estimate_tokens([user("hi")])
        long = estimate_tokens([user("word " * 200)])
        self.assertGreater(long, short * 5)

    def test_counts_tool_calls(self) -> None:
        message = assistant("", [tool_call := _call()])
        self.assertGreater(estimate_tokens([message]), estimate_tokens([assistant("x")]))


def _call():  # noqa: ANN202 - tiny helper
    from agent.messages import ToolCall

    return ToolCall("c1", "calculator", {"expression": "1 + 1"})


class Trim(unittest.TestCase):
    def test_noop_when_under_budget(self) -> None:
        messages = [system("s"), user("a"), assistant("b")]
        self.assertEqual(len(trim(messages, 10_000)), 3)

    def test_drops_oldest_turns_first(self) -> None:
        messages = [system("s")] + [user(f"turn {index} " + "x" * 200) for index in range(20)]
        kept = trim(messages, 400)
        self.assertEqual(kept[0].role, "system")
        self.assertIn("turn 19", kept[-1].content)
        self.assertLess(len(kept), len(messages))

    def test_current_task_survives_extreme_pressure(self) -> None:
        messages = [system("s")] + [user("y" * 4000) for _ in range(6)] + [user("question")]
        kept = trim(messages, 50)
        self.assertEqual(kept[0].role, "system")
        self.assertEqual(kept[-1].content, "question")


class TraceFile(unittest.TestCase):
    def test_events_are_appended_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            trace = Trace(path=path, run_id="unit")
            trace.event("assistant", turn=1, content="thinking")
            trace.event("tool_result", turn=1, name="calculator", result="4")
            trace.finish(stop_reason="final-answer", turns=1)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["kind"] for line in lines], ["assistant", "tool_result", "finish"])
            self.assertEqual(lines[1]["name"], "calculator")
            self.assertIn("timeline", dir(trace))
            self.assertIn("tool_result", trace.timeline())


if __name__ == "__main__":
    unittest.main()
