#!/usr/bin/env python3
"""Submission integrity gate: the guaranteed floor never regresses — SOT-2315.

The champion ``attack.py`` (SOT-2081/2082) carries a *guaranteed per-seed floor*: it
replays ``STARTER_FLOOR_CHAINS`` unconditionally and first, so the returned candidate
set is always a superset of the breaching floor chains for any seed. Because
``score_attack`` is monotonic non-decreasing in the set of validated findings, that
superset property is exactly what makes ``champion score >= starter score`` on every
seed — the SOT-2081 non-regression guarantee.

This gate pins that invariant with the sanctioned scoring path (beatable-rules
``Guardrail``, deterministic agent, gym env). It is deterministic and needs no external
starter file: the floor reference set is harvested directly by replaying
``STARTER_FLOOR_CHAINS`` through a fresh env (the same chains ``scripts/diag_floor.py``
harvested from the SOT-1967 starter), then it asserts the champion's own ``run`` returns
every breaching floor chain. Any drift in ``STARTER_FLOOR_CHAINS`` or the floor replay
that drops a breaching chain is caught here before submit.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Sanctioned confirm seeds; a small budget is plenty because the floor is only
# sub-second deterministic replays and always runs first (never time-box-evicted).
FLOOR_SEEDS = (5, 44)
FLOOR_BUDGET_S = 8.0


def _load_attack():
    spec = importlib.util.spec_from_file_location("attack_floor", REPO_ROOT / "attack.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _breaching_floor_chains(attack, *, seed: int, max_tool_hops: int) -> set[tuple[str, ...]]:
    """Ground truth: which STARTER_FLOOR_CHAINS actually breach on this seed."""
    from aicomp_sdk.agents import build_agent_factory
    from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
    from aicomp_sdk.core.env.gym import GymAttackEnv
    from aicomp_sdk.core.predicates import eval_predicates

    import aicomp_sdk

    fixtures = pathlib.Path(aicomp_sdk.__file__).resolve().parent / "fixtures"
    agent = build_agent_factory("deterministic")()
    breaching: set[tuple[str, ...]] = set()
    for chain in attack.STARTER_FLOOR_CHAINS:
        env = GymAttackEnv(seed=seed, fixtures_dir=fixtures, guardrail=Guardrail(), agent=agent)
        env.reset()
        for msg in chain:
            env.interact(msg, max_tool_hops=max_tool_hops)
        if eval_predicates(env.export_trace_dict()):
            breaching.add(tuple(chain))
    return breaching


def _champion_returned_chains(attack, *, seed: int, budget_s: float) -> set[tuple[str, ...]]:
    from aicomp_sdk.agents import build_agent_factory
    from aicomp_sdk.attacks import AttackRunConfig
    from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
    from aicomp_sdk.core.env.gym import GymAttackEnv

    import aicomp_sdk

    fixtures = pathlib.Path(aicomp_sdk.__file__).resolve().parent / "fixtures"
    env = GymAttackEnv(
        seed=seed,
        fixtures_dir=fixtures,
        guardrail=Guardrail(),
        agent=build_agent_factory("deterministic")(),
    )
    algorithm = attack.AttackAlgorithm({"seed": seed})
    config = AttackRunConfig(time_budget_s=budget_s, max_steps=10**9, max_tool_hops=4)
    candidates = algorithm.run(env, config)
    return {tuple(c.user_messages) for c in candidates}


class FloorGuaranteeTest(unittest.TestCase):
    def test_champion_returns_every_breaching_floor_chain(self) -> None:
        attack = _load_attack()
        self.assertTrue(attack.STARTER_FLOOR_CHAINS, "STARTER_FLOOR_CHAINS is empty")
        for seed in FLOOR_SEEDS:
            floor = _breaching_floor_chains(attack, seed=seed, max_tool_hops=4)
            self.assertTrue(
                floor,
                f"no STARTER_FLOOR_CHAINS breached on seed {seed} — the floor "
                "reference collapsed; the guarantee is meaningless",
            )
            returned = _champion_returned_chains(attack, seed=seed, budget_s=FLOOR_BUDGET_S)
            missing = floor - returned
            self.assertFalse(
                missing,
                f"seed {seed}: champion dropped {len(missing)} breaching floor "
                f"chain(s) — FLOOR REGRESSION: {sorted(missing)}",
            )


if __name__ == "__main__":
    unittest.main()
