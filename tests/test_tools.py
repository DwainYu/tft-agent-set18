"""Tool validation and execution: the boundary where the model meets real code."""

from __future__ import annotations

import unittest

from agent.messages import ToolCall
from agent.tools import ToolError, ToolRegistry, calculator, default_tools


class Calculator(unittest.TestCase):
    def test_arithmetic(self) -> None:
        self.assertEqual(calculator("(17 + 28) * 4"), 180)
        self.assertEqual(calculator("-2 ** 2"), -4.0)

    def test_rejects_names_and_code(self) -> None:
        for expression in ("__import__('os').system('ls')", "open('x')", "foo + 1"):
            with self.assertRaises(ToolError):
                calculator(expression)

    def test_rejects_strings(self) -> None:
        with self.assertRaises(ToolError):
            calculator("'a' + 'b'")


class Registry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry(default_tools())

    def test_specs_are_openai_shaped(self) -> None:
        spec = self.registry.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertIn("parameters", spec["function"])

    def test_invoke_returns_observation_string(self) -> None:
        observation = self.registry.invoke(ToolCall("c1", "calculator", {"expression": "2 * 3"}))
        self.assertEqual(observation, "6")

    def test_unknown_tool(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.invoke(ToolCall("c1", "nope", {}))

    def test_missing_required_argument(self) -> None:
        with self.assertRaises(ToolError) as caught:
            self.registry.invoke(ToolCall("c1", "calculator", {}))
        self.assertIn("missing argument", str(caught.exception))

    def test_unexpected_argument_is_rejected(self) -> None:
        with self.assertRaises(ToolError) as caught:
            self.registry.invoke(ToolCall("c1", "calculator", {"expression": "1 + 1", "unit": "kg"}))
        self.assertIn("unexpected argument", str(caught.exception))

    def test_state_survives_between_calls(self) -> None:
        self.registry.invoke(ToolCall("c1", "note_put", {"key": "k", "value": "v"}))
        self.assertEqual(self.registry.invoke(ToolCall("c2", "note_get", {"key": "k"})), "v")

    def test_reading_missing_state_is_an_error_not_a_crash(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.invoke(ToolCall("c1", "note_get", {"key": "absent"}))


if __name__ == "__main__":
    unittest.main()
