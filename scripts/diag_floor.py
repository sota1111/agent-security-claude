#!/usr/bin/env python3
"""Diagnostic: per-seed floor coverage — champion findings ⊇ starter findings.

Runs a strategy through the sanctioned ``eval_attack`` scoring path (beatable
rules Guardrail, deterministic agent, gym env) and reports, per seed, the
validated score plus the set of score-driving cell hashes. Used to prove that
the guaranteed-floor champion never regresses below the prompt-bank starter on
any seed, and to measure whether the starter's findings are budget-sensitive.

Usage:
    python scripts/diag_floor.py saturation --attack attack.py --seeds 44,99 --budgets 6,25
    python scripts/diag_floor.py cover --starter <starter.py> --champion attack.py \
        --seeds 123,7,99,5,21,44 --budget 30
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from aicomp_sdk.agents import build_agent_factory
from aicomp_sdk.attacks import AttackRunConfig
from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
from aicomp_sdk.evaluation.ops import AttackEvalOptions, eval_attack
from aicomp_sdk.scoring import get_score_cell_hash, score_attack


def load_attack(path: Path) -> type:
    spec = importlib.util.spec_from_file_location(f"diag_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "AttackAlgorithm")
    assert isinstance(cls, type)
    return cls


def run_seed(attack_cls: type, *, seed: int, budget_s: float):
    options = AttackEvalOptions(
        run_config=AttackRunConfig(time_budget_s=budget_s, max_steps=10**9, max_tool_hops=4),
        attack_config={"seed": seed},
        env_seed=seed,
        guardrail_factory=Guardrail,
    )
    findings = eval_attack(
        attack_cls,
        budget_s,
        agent_factory=build_agent_factory("deterministic"),
        env_selection="gym",
        options=options,
    )
    cells = {h for f in findings if (h := get_score_cell_hash(f))}
    return score_attack(findings), cells


def cmd_saturation(args) -> int:
    cls = load_attack(Path(args.attack).resolve())
    for seed in args.seeds:
        row = []
        prev: set | None = None
        for b in args.budgets:
            score, cells = run_seed(cls, seed=seed, budget_s=b)
            delta = "" if prev is None else (" SAME" if cells == prev else f" +{len(cells - prev)}/-{len(prev - cells)}")
            row.append(f"b={b}:score={score:.4f},cells={len(cells)}{delta}")
            prev = cells
        print(f"seed {seed}: " + " | ".join(row))
    return 0


def cmd_cover(args) -> int:
    starter = load_attack(Path(args.starter).resolve())
    champion = load_attack(Path(args.champion).resolve())
    all_ok = True
    for seed in args.seeds:
        s_score, s_cells = run_seed(starter, seed=seed, budget_s=args.budget)
        c_score, c_cells = run_seed(champion, seed=seed, budget_s=args.budget)
        superset = s_cells <= c_cells
        score_ok = c_score >= s_score - 1e-9
        ok = superset and score_ok
        all_ok = all_ok and ok
        missing = len(s_cells - c_cells)
        print(
            f"seed {seed}: starter score={s_score:.4f} cells={len(s_cells)} | "
            f"champion score={c_score:.4f} cells={len(c_cells)} | "
            f"⊇={superset} (missing {missing}) score≥={score_ok} => {'OK' if ok else 'REGRESSION'}"
        )
    print(f"ALL_SEEDS_FLOOR_OK={all_ok}")
    return 0 if all_ok else 1


def seeds(value: str) -> tuple[int, ...]:
    return tuple(int(p) for p in value.split(",") if p.strip())


def budgets(value: str) -> tuple[float, ...]:
    return tuple(float(p) for p in value.split(",") if p.strip())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("saturation")
    ps.add_argument("--attack", default="attack.py")
    ps.add_argument("--seeds", type=seeds, default=(44, 99))
    ps.add_argument("--budgets", type=budgets, default=(6.0, 25.0))
    ps.set_defaults(func=cmd_saturation)
    pc = sub.add_parser("cover")
    pc.add_argument("--starter", required=True)
    pc.add_argument("--champion", default="attack.py")
    pc.add_argument("--seeds", type=seeds, default=(123, 7, 99, 5, 21, 44))
    pc.add_argument("--budget", type=float, default=30.0)
    pc.set_defaults(func=cmd_cover)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
