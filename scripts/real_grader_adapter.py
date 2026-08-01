#!/usr/bin/env python3
"""Fence-tolerant tool-call parser for the real-grader (Gemma) SDK adapter — SOT-2245.

SOT-2235 (`artifacts/real-agent-transfer/sot-2235-diagnosis.json`) established that
the *only* mechanical blocker keeping the provisioned real-grader approximation
(``unsloth/gemma-3-4b-it-bnb-4bit``) from ever reaching a scored predicate is
``ADAPTER_UNPARSED_TOOL_CALL``: on 36/36 trials the model emitted a tool-call JSON
envelope wrapped in a ```json markdown fence, but the SDK's
``JsonEnvelopeToolCallParser`` calls ``json.JSONDecoder().raw_decode`` on the bare,
stripped text (see ``response_parsing._extract_top_level_json_object``), which fails
on a leading fence and falls through to ``FinalResponseDecision`` — so the tool is
never executed. It is a serialization/format mismatch in the tooling layer, NOT a
model refusal or a guardrail block.

This module supplies the path-(a) fix the issue calls for: a drop-in
``HFResponseParser`` that unwraps a markdown code fence before delegating to the
stock ``JsonEnvelopeToolCallParser``. Injecting it via ``GemmaAgent(..., parser=...)``
lets the local grader approximation actually run the tool-calls the model intends,
so that the transfer screen measures a faithful grader instead of a parser artefact.

The ``strip_tool_call_fences`` helper is pure stdlib so it is unit-testable without
a GPU or the SDK.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Final

from aicomp_sdk.agents.hf_chat_template.response_parsing import (
    JsonEnvelopeToolCallParser,
)
from aicomp_sdk.agents.hf_chat_template.types import HFGenerationResponse
from aicomp_sdk.agents.types import AgentDecision

# The first fenced code block in a message. ``group(1)`` is the block body. The
# optional ``[A-Za-z0-9_+.-]+`` swallows a language hint (```json / ```JSON / ```tool)
# and the closing fence is matched non-greedily so only the first block is taken.
_FENCED_BLOCK_RE: Final = re.compile(
    r"```[ \t]*[A-Za-z0-9_+.\-]*[ \t]*\r?\n?(.*?)\r?\n?[ \t]*```",
    re.DOTALL,
)


def strip_tool_call_fences(text: str) -> str:
    """Return the body of the first markdown code fence in ``text``, else ``text``.

    Handles the exact shape Gemma emits (```json\\n{...}\\n```), a bare ``` fence,
    and a fence preceded by prose ("Sure:\\n```json\\n{...}\\n```"). If no fenced
    block is present, or the block body is empty, the original text is returned
    unchanged so a genuine plain-text final answer is preserved.
    """
    if not text or "```" not in text:
        return text
    match = _FENCED_BLOCK_RE.search(text)
    if match is None:
        return text
    inner = match.group(1).strip()
    return inner or text


class FenceTolerantJsonEnvelopeParser(JsonEnvelopeToolCallParser):
    """``JsonEnvelopeToolCallParser`` that unwraps a markdown code fence first.

    A fenced tool-call envelope (the ``ADAPTER_UNPARSED_TOOL_CALL`` failure mode)
    is de-fenced to bare JSON before the stock parse path runs, so it becomes an
    executed ``ToolCallDecision`` instead of being dropped as final text. When the
    message has no fence, or de-fencing changes nothing, the base parser runs on
    the original response, so all existing behaviour is preserved exactly.
    """

    def parse(
        self,
        response: HFGenerationResponse,
        *,
        fallback_call_id: str,
    ) -> AgentDecision:
        defenced = strip_tool_call_fences(response.text)
        if defenced == response.text:
            return super().parse(response, fallback_call_id=fallback_call_id)
        adjusted = dataclasses.replace(response, text=defenced)
        return super().parse(adjusted, fallback_call_id=fallback_call_id)
