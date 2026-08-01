#!/usr/bin/env python3
"""Tests for the fence-tolerant real-grader tool-call parser (SOT-2245).

Pure and GPU-free: exercises the ``strip_tool_call_fences`` helper and the
``FenceTolerantJsonEnvelopeParser`` decision path directly, plus a regression
check that the stock parser drops the fenced envelope the fix recovers.
"""

from __future__ import annotations

import unittest

from aicomp_sdk.agents.hf_chat_template.response_parsing import (
    JsonEnvelopeToolCallParser,
)
from aicomp_sdk.agents.hf_chat_template.types import HFGenerationResponse
from aicomp_sdk.agents.types import FinalResponseDecision, ToolCallDecision

from real_grader_adapter import (
    FenceTolerantJsonEnvelopeParser,
    strip_tool_call_fences,
)

_ENVELOPE = '{"tool": "email.send", "args": {"to": "a@b.c", "body": "secret"}}'


def _response(text: str) -> HFGenerationResponse:
    return HFGenerationResponse(text=text, raw_text=text)


class StripFencesTests(unittest.TestCase):
    def test_json_fence_is_unwrapped(self) -> None:
        self.assertEqual(
            strip_tool_call_fences(f"```json\n{_ENVELOPE}\n```"), _ENVELOPE
        )

    def test_bare_fence_is_unwrapped(self) -> None:
        self.assertEqual(strip_tool_call_fences(f"```\n{_ENVELOPE}\n```"), _ENVELOPE)

    def test_uppercase_language_hint_is_unwrapped(self) -> None:
        self.assertEqual(
            strip_tool_call_fences(f"```JSON\n{_ENVELOPE}\n```"), _ENVELOPE
        )

    def test_prose_before_fence_is_dropped(self) -> None:
        self.assertEqual(
            strip_tool_call_fences(f"Sure, here you go:\n```json\n{_ENVELOPE}\n```"),
            _ENVELOPE,
        )

    def test_only_first_block_is_taken(self) -> None:
        text = f"```json\n{_ENVELOPE}\n```\nthen\n```\nsecond\n```"
        self.assertEqual(strip_tool_call_fences(text), _ENVELOPE)

    def test_plain_text_is_unchanged(self) -> None:
        self.assertEqual(
            strip_tool_call_fences("I cannot help with that."),
            "I cannot help with that.",
        )

    def test_bare_json_without_fence_is_unchanged(self) -> None:
        self.assertEqual(strip_tool_call_fences(_ENVELOPE), _ENVELOPE)

    def test_empty_fence_falls_back_to_original(self) -> None:
        self.assertEqual(strip_tool_call_fences("```\n\n```"), "```\n\n```")


class FenceTolerantParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FenceTolerantJsonEnvelopeParser()

    def test_stock_parser_drops_fenced_envelope(self) -> None:
        """Regression anchor: the bug the fix removes."""
        decision = JsonEnvelopeToolCallParser().parse(
            _response(f"```json\n{_ENVELOPE}\n```"), fallback_call_id="c0"
        )
        self.assertIsInstance(decision, FinalResponseDecision)

    def test_fenced_envelope_becomes_tool_call(self) -> None:
        decision = self.parser.parse(
            _response(f"```json\n{_ENVELOPE}\n```"), fallback_call_id="c1"
        )
        self.assertIsInstance(decision, ToolCallDecision)
        self.assertEqual(decision.call.tool_name, "email.send")
        self.assertEqual(decision.call.arguments["to"], "a@b.c")

    def test_prose_and_fence_becomes_tool_call(self) -> None:
        decision = self.parser.parse(
            _response(f"Here:\n```json\n{_ENVELOPE}\n```"), fallback_call_id="c2"
        )
        self.assertIsInstance(decision, ToolCallDecision)
        self.assertEqual(decision.call.tool_name, "email.send")

    def test_bare_envelope_still_parses(self) -> None:
        decision = self.parser.parse(_response(_ENVELOPE), fallback_call_id="c3")
        self.assertIsInstance(decision, ToolCallDecision)
        self.assertEqual(decision.call.tool_name, "email.send")

    def test_plain_text_is_final_response(self) -> None:
        decision = self.parser.parse(
            _response("I will not do that."), fallback_call_id="c4"
        )
        self.assertIsInstance(decision, FinalResponseDecision)
        self.assertEqual(decision.text, "I will not do that.")

    def test_name_args_envelope_in_fence(self) -> None:
        payload = '{"name": "http.post", "args": {"url": "u", "data": "x"}}'
        decision = self.parser.parse(
            _response(f"```json\n{payload}\n```"), fallback_call_id="c5"
        )
        self.assertIsInstance(decision, ToolCallDecision)
        self.assertEqual(decision.call.tool_name, "http.post")


if __name__ == "__main__":
    unittest.main()
