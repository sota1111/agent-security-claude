#!/usr/bin/env python3
"""Tests for the adapter-fix screen's pure aggregation logic (SOT-2245)."""

from __future__ import annotations

import unittest

from screen_adapter_fix import (
    VariantTrial,
    compare_variants,
    summarize_variant,
    _residual_bottleneck,
)


def _trial(
    parser: str,
    mode: str,
    *,
    n_tool_calls: int = 0,
    breach: bool = False,
    objective: bool = False,
    leak: bool = False,
    predicates: tuple[str, ...] = (),
) -> VariantTrial:
    return VariantTrial(
        parser=parser,
        vector="v",
        seed=17,
        messages=("m",),
        rejection_mode=mode,
        n_tool_calls=n_tool_calls,
        breach=breach,
        objective=objective,
        real_secret_leak=leak,
        predicates=predicates,
    )


class SummarizeVariantTests(unittest.TestCase):
    def test_baseline_all_unparsed(self) -> None:
        trials = [_trial("baseline", "ADAPTER_UNPARSED_TOOL_CALL") for _ in range(4)]
        s = summarize_variant("baseline", trials)
        self.assertEqual(s["trials"], 4)
        self.assertAlmostEqual(s["unparsed_tool_call_rate"], 1.0)
        self.assertAlmostEqual(s["tool_execution_rate"], 0.0)
        self.assertAlmostEqual(s["breach_rate"], 0.0)
        self.assertEqual(s["mode_counts"]["ADAPTER_UNPARSED_TOOL_CALL"], 4)

    def test_fixed_executes_and_breaches(self) -> None:
        trials = [
            _trial(
                "fence_tolerant",
                "BREACH",
                n_tool_calls=3,
                breach=True,
                objective=True,
                leak=True,
                predicates=("CONFUSED_DEPUTY",),
            ),
            _trial(
                "fence_tolerant",
                "GUARDRAIL_BLOCKED_EGRESS",
                n_tool_calls=2,
                objective=True,
            ),
        ]
        s = summarize_variant("fence_tolerant", trials)
        self.assertAlmostEqual(s["unparsed_tool_call_rate"], 0.0)
        self.assertAlmostEqual(s["tool_execution_rate"], 1.0)
        self.assertAlmostEqual(s["breach_rate"], 0.5)
        self.assertAlmostEqual(s["objective_rate"], 1.0)
        self.assertAlmostEqual(s["real_secret_leak_rate"], 0.5)

    def test_empty_variant_is_zero_not_error(self) -> None:
        s = summarize_variant("baseline", [])
        self.assertEqual(s["trials"], 0)
        self.assertAlmostEqual(s["unparsed_tool_call_rate"], 0.0)


class CompareVariantsTests(unittest.TestCase):
    def test_reduction_and_gain(self) -> None:
        baseline = summarize_variant(
            "baseline",
            [_trial("baseline", "ADAPTER_UNPARSED_TOOL_CALL") for _ in range(4)],
        )
        fixed = summarize_variant(
            "fence_tolerant",
            [
                _trial("fence_tolerant", "GUARDRAIL_BLOCKED_EGRESS", n_tool_calls=2)
                for _ in range(4)
            ],
        )
        cmp = compare_variants(baseline, fixed)
        self.assertAlmostEqual(cmp["unparsed_tool_call_rate_reduction"], 1.0)
        self.assertTrue(cmp["unparsed_tool_call_rate_dropped"])
        self.assertAlmostEqual(cmp["tool_execution_gain"], 1.0)
        self.assertFalse(cmp["first_breach_achieved"])
        self.assertEqual(cmp["residual_bottleneck"], "GUARDRAIL_BLOCKED_EGRESS")

    def test_first_breach_has_no_residual(self) -> None:
        baseline = summarize_variant("baseline", [_trial("baseline", "MODEL_NO_TOOL_CALL")])
        fixed = summarize_variant(
            "fence_tolerant",
            [_trial("fence_tolerant", "BREACH", n_tool_calls=2, breach=True)],
        )
        cmp = compare_variants(baseline, fixed)
        self.assertTrue(cmp["first_breach_achieved"])
        self.assertIsNone(cmp["residual_bottleneck"])


class ResidualBottleneckTests(unittest.TestCase):
    def test_picks_nearest_to_breach_mode(self) -> None:
        fixed = summarize_variant(
            "fence_tolerant",
            [
                _trial("fence_tolerant", "SECRET_READ_NO_EGRESS_ATTEMPT", n_tool_calls=1),
                _trial("fence_tolerant", "GUARDRAIL_BLOCKED_EGRESS", n_tool_calls=2),
            ],
        )
        # GUARDRAIL_BLOCKED_EGRESS (distance 2) is nearer than SECRET_READ (3).
        self.assertEqual(_residual_bottleneck(fixed), "GUARDRAIL_BLOCKED_EGRESS")


if __name__ == "__main__":
    unittest.main()
