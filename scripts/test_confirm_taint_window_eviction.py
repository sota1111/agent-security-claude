#!/usr/bin/env python3
"""Deterministic tests for the taint-window-eviction CONFIRM (SOT-2327).

Offline, CPU-only, deterministic agent — no GPU, no network. Covers the disjoint-seed
invariant, the pure confirm decision (guardrail bypass w/o scored breach => reject; a
hypothetical scored breach => promote-and-wire), the faithful (optimal_public) breach
staying 0 on the disjoint seeds, that attack.py is left byte-identical to the wired
champion (non-promotion => no wiring), and the confirm report shape.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from confirm_taint_window_eviction import (
    DEFAULT_CONFIRM_SEEDS,
    build_confirm_report,
    confirm_decision,
    seeds_are_disjoint,
)
from screen_taint_window_eviction import (
    DEFAULT_SEEDS as SCREEN_SEEDS,
    WIRED_CHAMPION_SHA256,
    build_report,
)

CONFIRM_SEEDS = DEFAULT_CONFIRM_SEEDS
REPO_ROOT = Path(__file__).resolve().parent.parent


def _screen_report(*, guardrail_bypass: bool, scored_breach: bool) -> dict:
    """Minimal screen-report stub shaped like build_report's output for decision tests."""
    return {
        "decision": {
            "guardrail_taint_gate_bypassed": guardrail_bypass,
            "scored_breach_achieved": scored_breach,
            "why_no_scored_breach": None if scored_breach else ["blocker a", "blocker b"],
        },
        "faithful_breach": {
            "eviction": {
                "any_breach": scored_breach,
                "average_score": 5.0 if scored_breach else 0.0,
                "per_seed_findings": (1, 1, 1) if scored_breach else (0, 0, 0),
            },
            "control": {"any_breach": False, "average_score": 0.0},
        },
        "guardrail_layer": {
            "window_boundary": {
                "deny_max_padding": 4,
                "allow_min_padding": 5,
                "seed_stable": True,
            }
        },
    }


class DisjointSeedTests(unittest.TestCase):
    def test_default_confirm_seeds_disjoint_from_screen(self) -> None:
        self.assertTrue(seeds_are_disjoint(CONFIRM_SEEDS))
        self.assertFalse(set(CONFIRM_SEEDS) & set(SCREEN_SEEDS))

    def test_overlapping_seeds_rejected(self) -> None:
        self.assertFalse(seeds_are_disjoint((5, 123, 44)))

    def test_build_confirm_report_rejects_overlapping_seeds(self) -> None:
        with self.assertRaises(ValueError):
            build_confirm_report((123, 7, 99), 6.0, REPO_ROOT / "attack.py")


class ConfirmDecisionTests(unittest.TestCase):
    def test_bypass_without_breach_is_non_promotion(self) -> None:
        dec = confirm_decision(_screen_report(guardrail_bypass=True, scored_breach=False))
        self.assertFalse(dec["promote"])
        self.assertEqual(dec["decision"], "reject-non-promotion")
        self.assertTrue(dec["guardrail_taint_gate_bypassed"])
        self.assertFalse(dec["scored_breach_achieved"])
        self.assertTrue(dec["reproduces_screen_bypass_without_breach"])
        self.assertEqual(dec["stage"], "faithful_confirm")
        self.assertIsNotNone(dec["why_no_scored_breach"])

    def test_scored_breach_would_promote_and_wire(self) -> None:
        # Guards the decision logic: a genuine disjoint-seed breach => promote-and-wire.
        dec = confirm_decision(_screen_report(guardrail_bypass=True, scored_breach=True))
        self.assertTrue(dec["promote"])
        self.assertEqual(dec["decision"], "promote-and-wire")
        self.assertFalse(dec["reproduces_screen_bypass_without_breach"])


class FaithfulSignalIntegrationTests(unittest.TestCase):
    """The sanctioned eval_attack path on the DISJOINT seeds — must stay 0 (non-promotion)."""

    def test_disjoint_seed_confirm_scores_zero_and_reproduces(self) -> None:
        report = build_report(CONFIRM_SEEDS, 6.0, REPO_ROOT / "attack.py")
        dec = confirm_decision(report)
        # Guardrail bypass reproduces on disjoint seeds, faithful breach stays 0.
        self.assertTrue(dec["guardrail_taint_gate_bypassed"])
        self.assertFalse(dec["eviction_any_breach"])
        self.assertFalse(dec["control_any_breach"])
        self.assertFalse(dec["promote"])
        self.assertTrue(dec["reproduces_screen_bypass_without_breach"])


class AttackUnchangedTests(unittest.TestCase):
    def test_attack_py_matches_wired_champion(self) -> None:
        # Non-promotion => confirm must not touch attack.py; assert byte-identity.
        sha = hashlib.sha256((REPO_ROOT / "attack.py").read_bytes()).hexdigest()
        self.assertEqual(sha, WIRED_CHAMPION_SHA256)


class ReportShapeTests(unittest.TestCase):
    def test_report_is_serializable_and_flags_non_promotion(self) -> None:
        report = build_confirm_report(CONFIRM_SEEDS, 6.0, REPO_ROOT / "attack.py")
        self.assertEqual(report["schema"], "taint-window-eviction-confirm/v1")
        self.assertEqual(report["issue"], "SOT-2327")
        self.assertTrue(report["seeds_disjoint_from_screen"])
        self.assertTrue(report["champion"]["matches_wired_champion"])
        self.assertTrue(report["champion"]["unchanged_this_confirm"])
        self.assertFalse(report["decision"]["promote"])
        self.assertTrue(report["decision"]["guardrail_taint_gate_bypassed"])
        self.assertFalse(report["isolation"]["attack_wired"])
        self.assertFalse(report["isolation"]["kaggle_submission"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
