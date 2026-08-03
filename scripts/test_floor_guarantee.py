#!/usr/bin/env python3
"""Champion structural gate — single-message exfiltration fill.

The SOT-2081/2082 guaranteed-floor + SOT-2296 genuine-carriage lineage this test
previously pinned was **superseded** by the single-message raw-weighted fill
champion (adopted from the public ``aisec-pilk`` scoring analysis). The floor
lineage tuned to the local deterministic stub agent yet scored public LB 0.000 on
the real grader, because the real reasoning/non-reasoning models — not the stub —
are what the public score measures.

The new champion has no ``STARTER_FLOOR_CHAINS`` invariant to pin. What still must
hold before submit is the fill's structural contract, which this gate asserts
without any real-model dependency:

1. the champion exposes the fill controls (``TEMPLATES``, ``PAYLOAD``,
   ``REPLAY_SAFE`` in (0, 1], a plain fallback template first);
2. every template renders to a non-empty single message carrying the sentinel;
3. ``run`` degrades to a non-empty list of single-message candidates when no env
   is available (the exec-compat / offline path), so the submission never ships
   empty.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_attack():
    spec = importlib.util.spec_from_file_location("attack_champion", REPO_ROOT / "attack.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ChampionFillStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.attack = _load_attack()

    def test_fill_controls_present(self) -> None:
        a = self.attack
        self.assertTrue(a.TEMPLATES, "TEMPLATES must be non-empty")
        self.assertEqual(a.TEMPLATES[a.FALLBACK_INDEX][0], "plain")
        self.assertIsInstance(a.PAYLOAD, str)
        self.assertTrue(a.PAYLOAD)
        self.assertGreater(a.REPLAY_SAFE, 0.0)
        self.assertLessEqual(a.REPLAY_SAFE, 1.0)

    def test_templates_render_sentinel_messages(self) -> None:
        a = self.attack
        for name, template in a.TEMPLATES:
            message = a._message(template, 0)
            self.assertTrue(message, f"template {name} rendered empty")
            self.assertIn(a.PAYLOAD, message, f"template {name} dropped the sentinel")

    def test_run_without_env_returns_single_message_candidates(self) -> None:
        a = self.attack
        found = a.AttackAlgorithm().run(None, None)  # offline / exec-compat path
        self.assertTrue(found, "run(None, ...) must return a non-empty fallback set")
        for candidate in found:
            self.assertEqual(
                len(candidate.user_messages), 1, "champion ships single-message candidates"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
