"""The loop: termination, budgets, repeated-call protection, failure handling."""

from __future__ import annotations

import unittest

from agent.loop import Agent, AgentConfig
from agent.messages import Message
from agent.provider import ScriptedProvider
from agent.tools import Tool, ToolRegistry, default_tools


def tools() -> ToolRegistry:
    return ToolRegistry(default_tools())


def tool_call(name: str, **arguments) -> dict:  # noqa: ANN003 - test shorthand
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


class HappyPath(unittest.TestCase):
    def test_answers_without_tools(self) -> None:
        agent = Agent(ScriptedProvider([{"content": "done"}]), tools())
        result = agent.run("anything")
        self.assertTrue(result.ok)
        self.assertEqual(result.stop_reason, "final-answer")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(result.turns, 1)

    def test_tool_result_reaches_the_next_request(self) -> None:
        provider = ScriptedProvider([tool_call("calculator", expression="2 + 2"), {"content": "4"}])
        result = Agent(provider, tools()).run("compute 2+2")
        self.assertEqual(result.final, "4")
        # The second request must contain the observation from the first turn.
        second_request = provider.calls[1]
        observations = [entry for entry in second_request if entry["role"] == "tool"]
        self.assertEqual([entry["content"] for entry in observations], ["4"])

    def test_assistant_tool_call_message_is_preserved(self) -> None:
        result = Agent(
            ScriptedProvider([tool_call("calculator", expression="3 * 3"), {"content": "9"}]), tools()
        ).run("square three")
        roles = [message.role for message in result.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        script = [
            {
                "tool_calls": [
                    {"id": "a", "name": "note_put", "arguments": {"key": "k1", "value": "v1"}},
                    {"id": "b", "name": "note_put", "arguments": {"key": "k2", "value": "v2"}},
                ]
            },
            {"content": "stored two notes"},
        ]
        result = Agent(ScriptedProvider(script), tools(), AgentConfig(max_turns=4)).run("store two")
        self.assertEqual(result.tool_calls, 2)
        self.assertTrue(result.ok)


class Budgets(unittest.TestCase):
    def script(self, count: int) -> list:
        # Distinct arguments each turn so the repeat guard stays out of the way.
        return [tool_call("calculator", expression=f"{index} + 0") for index in range(count)]

    def test_max_turns_stops_a_runaway_loop(self) -> None:
        result = Agent(ScriptedProvider(self.script(10)), tools(), AgentConfig(max_turns=3)).run("go")
        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "max-turns")
        self.assertEqual(result.turns, 3)

    def test_tool_budget_is_enforced(self) -> None:
        result = Agent(
            ScriptedProvider(self.script(6)), tools(), AgentConfig(max_turns=10, max_tool_calls=2)
        ).run("go")
        self.assertEqual(result.stop_reason, "tool-budget")
        self.assertLessEqual(result.tool_calls, 2)

    def test_identical_repeated_calls_are_refused(self) -> None:
        script = [
            tool_call("now"),
            tool_call("now"),
            tool_call("now"),
            {"content": "giving up"},
        ]
        provider = ScriptedProvider(script)
        result = Agent(provider, tools(), AgentConfig(max_turns=6, repeat_limit=2)).run("time please")
        refusals = [
            message
            for message in result.messages
            if message.role == "tool" and message.content.startswith("REFUSED")
        ]
        self.assertTrue(refusals, "the stuck loop should have been refused")
        self.assertTrue(result.ok, "the model should still get a chance to recover")


class Failures(unittest.TestCase):
    def test_tool_error_becomes_an_observation(self) -> None:
        result = Agent(
            ScriptedProvider([tool_call("calculator"), {"content": "cannot compute"}]), tools()
        ).run("bad arguments")
        observations = [message.content for message in result.messages if message.role == "tool"]
        self.assertTrue(observations[0].startswith("ERROR:"))
        self.assertEqual(result.final, "cannot compute")
        self.assertTrue(result.ok)

    def test_tool_crash_does_not_kill_the_run(self) -> None:
        def explode() -> str:
            raise RuntimeError("db died")

        registry = ToolRegistry([Tool("explode", "crashes", {"type": "object", "properties": {}}, explode)])
        result = Agent(
            ScriptedProvider([tool_call("explode"), {"content": "reported"}]), registry
        ).run("crash for me")
        self.assertEqual(result.final, "reported")
        self.assertTrue(any(not step.ok for step in result.steps))

    def test_fail_on_tool_error_ends_the_run(self) -> None:
        result = Agent(
            ScriptedProvider([tool_call("calculator")]),
            tools(),
            AgentConfig(fail_on_tool_error=True),
        ).run("bad arguments")
        self.assertEqual(result.stop_reason, "tool-error")
        self.assertFalse(result.ok)

    def test_provider_error_is_reported_not_raised(self) -> None:
        # The script runs out on turn three: the loop must degrade, not explode.
        result = Agent(ScriptedProvider([]), tools()).run("nothing scripted")
        self.assertEqual(result.stop_reason, "provider-error")
        self.assertFalse(result.ok)


class Context(unittest.TestCase):
    def test_small_budget_sends_a_shorter_request(self) -> None:
        provider = ScriptedProvider([{"content": "ok"}])
        history = [Message("user", "padding " * 400), Message("assistant", "noted " * 400)]
        Agent(provider, tools(), AgentConfig(token_budget=120)).run("final question", history=history)
        sent = provider.calls[0]
        self.assertLess(len(sent), len(history) + 2)
        self.assertEqual(sent[0]["role"], "system", "the system prompt is never trimmed away")
        self.assertEqual(sent[-1]["content"], "final question", "the current task always survives")


if __name__ == "__main__":
    unittest.main()
