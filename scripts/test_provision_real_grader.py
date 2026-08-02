#!/usr/bin/env python3
"""Pure-logic tests for the multi-target grader provisioner (SOT-2313).

No GPU/network: covers the target registry pinning, the VRAM feasibility gate, and
the aggregate/bool/hash-only artifact schema builder.
"""

from __future__ import annotations

import json
import unittest

from provision_real_grader import (
    MODEL_ID,
    MODEL_REVISION,
    TARGETS,
    build_smoke_record,
    vram_feasible,
    _sha256,
    _target_fingerprint,
)


class RegistryTests(unittest.TestCase):
    def test_backward_compatible_gemma_constants(self) -> None:
        self.assertEqual(TARGETS["gemma"].model_id, MODEL_ID)
        self.assertEqual(TARGETS["gemma"].revision, MODEL_REVISION)
        self.assertEqual(TARGETS["gemma"].kind, "image_text")

    def test_every_target_is_revision_pinned(self) -> None:
        for key, target in TARGETS.items():
            with self.subTest(target=key):
                self.assertRegex(target.revision, r"^[0-9a-f]{40}$")

    def test_gpt_oss_and_qwen_are_gpt_oss_family_adapters(self) -> None:
        for key in ("gpt_oss", "qwen-3b"):
            self.assertEqual(TARGETS[key].family, "gpt_oss")
            self.assertEqual(TARGETS[key].kind, "adapter")
        self.assertEqual(TARGETS["gpt_oss"].model_id, "openai/gpt-oss-20b")
        self.assertEqual(TARGETS["qwen-3b"].model_id, "Qwen/Qwen2.5-3B-Instruct")

    def test_adapter_targets_map_both_adapter_env_vars(self) -> None:
        for key in ("gpt_oss", "qwen-3b"):
            env = TARGETS[key].model_path_env
            self.assertEqual(env["gpt_oss"], "GPT_OSS_MODEL_PATH")
            self.assertEqual(env["gemma"], "GEMMA_MODEL_PATH")


class FeasibilityTests(unittest.TestCase):
    def test_unknown_inputs_are_none(self) -> None:
        self.assertIsNone(vram_feasible(None, 12 * 10**9))
        self.assertIsNone(vram_feasible(6 * 10**9, None))

    def test_20b_weights_do_not_fit_12gb(self) -> None:
        # ~41GB weights vs 12GB VRAM -> infeasible.
        self.assertIs(vram_feasible(41 * 10**9, 12 * 10**9), False)

    def test_3b_weights_fit_12gb(self) -> None:
        self.assertIs(vram_feasible(6 * 10**9, 12 * 10**9), True)

    def test_overhead_margin_is_applied(self) -> None:
        # Exactly at capacity must fail because of the 10% runtime overhead.
        self.assertIs(vram_feasible(12 * 10**9, 12 * 10**9), False)


class SmokeRecordTests(unittest.TestCase):
    def _record(self, **overrides):
        base = dict(
            target=TARGETS["qwen-3b"],
            adapter="gemma",
            loaded=True,
            responded=True,
            response_text="OK",
            parser_class="JsonEnvelopeToolCallParser",
            tool_call_fired=True,
            tool_calls_fired=2,
            tool_prompts_tried=2,
            tool_name="fs.read",
            blocker=None,
        )
        base.update(overrides)
        return build_smoke_record(**base)

    def test_record_never_stores_raw_response_text(self) -> None:
        record = self._record(response_text="super secret model reply")
        self.assertNotIn("super secret", json.dumps(record))
        self.assertEqual(record["response_sha256"], _sha256("super secret model reply"))
        self.assertEqual(record["response_char_len"], len("super secret model reply"))

    def test_fingerprint_binds_id_and_revision(self) -> None:
        record = self._record()
        self.assertEqual(
            record["model_fingerprint_sha256"],
            _target_fingerprint(TARGETS["qwen-3b"]),
        )

    def test_bool_normalization(self) -> None:
        record = self._record(tool_call_fired=None, responded=None)
        self.assertIsNone(record["tool_call_fired"])
        self.assertIsNone(record["responded"])

    def test_load_failure_record_shape(self) -> None:
        record = self._record(
            loaded=False,
            responded=None,
            response_text=None,
            parser_class=None,
            tool_call_fired=None,
            tool_calls_fired=0,
            tool_prompts_tried=0,
            tool_name=None,
            blocker="RuntimeError: CUDA out of memory",
        )
        self.assertFalse(record["loaded"])
        self.assertIsNone(record["response_sha256"])
        self.assertIn("CUDA out of memory", record["blocker"])


if __name__ == "__main__":
    unittest.main()
