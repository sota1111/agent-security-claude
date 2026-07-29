#!/usr/bin/env python3
"""Focused tests for real-agent transfer metric definitions."""

from __future__ import annotations

import unittest

from score_real_agent import TrialResult, summarize


class SummarizeTests(unittest.TestCase):
    def test_rates_use_trials_as_denominator(self) -> None:
        results = [
            TrialResult("champion", "gemma", 1, 4, 4, 4, True, True),
            TrialResult("champion", "gemma", 2, 0, 0, 0, False, False),
            TrialResult("champion", "gemma", 3, 2, 1, 2, True, True),
        ]

        summary = summarize(results)

        self.assertEqual(summary["trials"], 3)
        self.assertEqual(summary["breach_successes"], 2)
        self.assertEqual(summary["objective_successes"], 2)
        self.assertAlmostEqual(summary["breach_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["objective_achievement_rate"], 2 / 3)
        self.assertEqual(summary["total_breaches"], 5)
        self.assertEqual(summary["total_validated_findings"], 6)


if __name__ == "__main__":
    unittest.main()
