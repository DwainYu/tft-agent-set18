"""The live provider must not invent an answer out of an empty envelope."""

from __future__ import annotations

import unittest

from agent.provider import OpenAICompatProvider, ProviderError


def provider() -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url="https://example.invalid/v1", model="some/model", api_key="test-key"
    )


class CompletionEnvelope(unittest.TestCase):
    def test_plain_answer(self) -> None:
        data = {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1},
        }
        completion = provider()._completion(data)
        self.assertEqual(completion.message.content, "OK")
        self.assertEqual(completion.finish_reason, "stop")
        self.assertEqual(completion.usage.total_tokens, 7)

    def test_tool_call_turn_has_no_text(self) -> None:
        data = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'}}
                        ],
                    },
                }
            ],
            "usage": {},
        }
        completion = provider()._completion(data)
        self.assertEqual(completion.message.content, "")
        self.assertEqual([call.name for call in completion.message.tool_calls], ["calculator"])
        self.assertEqual(completion.usage.total_tokens, 0)

    def test_http_200_without_choices_is_an_error_not_an_empty_answer(self) -> None:
        # Measured response shape from api-inference.modelscope.cn for a model it
        # would not serve: 200, "choices": null, everything else empty.
        with self.assertRaises(ProviderError) as caught:
            provider()._completion({"choices": None, "model": "meituan-longcat/LongCat-Flash-Lite"})
        self.assertIn("no choices", str(caught.exception))

    def test_extra_body_reaches_the_request(self) -> None:
        sent: dict = {}
        subject = provider()
        subject.extra_body = {"enable_thinking": False, "max_tokens": 1024}
        subject._request = lambda body: sent.update(body) or {  # type: ignore[assignment]
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}]
        }
        subject.complete([], [])
        self.assertIs(sent["enable_thinking"], False)
        self.assertEqual(sent["max_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
