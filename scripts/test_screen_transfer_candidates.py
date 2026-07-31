#!/usr/bin/env python3
"""Focused tests for the transfer-candidate screen's pure aggregation logic."""

from __future__ import annotations

import unittest

from screen_transfer_candidates import ChainResult, summarize


def _row(
    vector: str,
    seed: int,
    breach: bool,
    objective: bool,
    leak: bool,
    predicates: tuple[str, ...] = (),
) -> ChainResult:
    return ChainResult(
        vector=vector,
        seed=seed,
        messages=("m",),
        breach=breach,
        objective=objective,
        real_secret_leak=leak,
        egress_count=0,
        predicates=predicates,
    )


class SummarizeTests(unittest.TestCase):
    def test_rates_use_chain_count_as_denominator(self) -> None:
        rows = [
            _row("web_indirect_injection", 17, True, True, False),
            _row("web_indirect_injection", 29, False, True, False),
            _row("direct_command", 17, True, True, True),
        ]

        summaries = {s.vector: s for s in summarize(rows)}

        web = summaries["web_indirect_injection"]
        self.assertEqual(web.trials, 2)
        self.assertEqual(web.breach_successes, 1)
        self.assertAlmostEqual(web.breach_success_rate, 0.5)
        self.assertAlmostEqual(web.objective_achievement_rate, 1.0)
        self.assertAlmostEqual(web.real_secret_leak_rate, 0.0)
        self.assertEqual(web.seeds, [17, 29])

        direct = summaries["direct_command"]
        self.assertEqual(direct.trials, 1)
        self.assertAlmostEqual(direct.real_secret_leak_rate, 1.0)

    def test_summaries_are_sorted_by_vector(self) -> None:
        rows = [
            _row("web_indirect_injection", 1, False, False, False),
            _row("direct_command", 1, False, False, False),
        ]
        self.assertEqual(
            [s.vector for s in summarize(rows)],
            ["direct_command", "web_indirect_injection"],
        )

    def test_target_predicate_rates_are_recorded(self) -> None:
        rows = [
            _row(
                "email_deferred_confirmation",
                17,
                True,
                True,
                False,
                ("CONFUSED_DEPUTY",),
            ),
            _row("email_deferred_confirmation", 29, False, False, False),
        ]

        summary = summarize(rows)[0]
        self.assertEqual(summary.by_predicate["CONFUSED_DEPUTY"]["successes"], 1)
        self.assertAlmostEqual(
            summary.by_predicate["CONFUSED_DEPUTY"]["success_rate"], 0.5
        )
        self.assertEqual(summary.by_predicate["EXFILTRATION"]["successes"], 0)


if __name__ == "__main__":
    unittest.main()
