#!/usr/bin/env python3
"""Measure attack transfer from the deterministic proxy to an SDK LLM agent."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from types import ModuleType

from aicomp_sdk.agents import build_agent_factory
from aicomp_sdk.attacks import AttackRunConfig
from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
from aicomp_sdk.core.predicates import is_breach
from aicomp_sdk.evaluation.ops import AttackEvalOptions, eval_attack

from score_proxy import load_attack, parse_seeds

PREDICATE_NAMES = (
    "CONFUSED_DEPUTY",
    "UNTRUSTED_TO_ACTION",
    "EXFILTRATION",
)
STRONG_STAND_INS = {
    "qwen-1.5b": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        Path(".models/qwen2.5-1.5b-instruct"),
    ),
    "qwen-3b": (
        "Qwen/Qwen2.5-3B-Instruct",
        Path(".models/qwen2.5-3b-instruct"),
    ),
}
REAL_GRADERS = {
    "gemma-3-4b-nf4": (
        "unsloth/gemma-3-4b-it-bnb-4bit",
        Path(".models/gemma-3-4b-it-bnb-4bit"),
        "eb03c885bc2cc913fe792994bc766006f14ad72d",
    ),
}


@dataclass(frozen=True)
class TrialResult:
    strategy: str
    agent: str
    seed: int
    findings: int
    breaches: int
    objectives: int
    breach_success: bool
    objective_achieved: bool
    predicate_counts: dict[str, int]


def load_attack_revision(revision: str) -> type:
    if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        raise ValueError("--starter-revision must be a 7-40 character commit hash")
    source = subprocess.run(
        ["git", "show", f"{revision}:attack.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = ModuleType(f"starter_attack_{revision}")
    module.__file__ = f"{revision}:attack.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    attack_cls = getattr(module, "AttackAlgorithm", None)
    if not isinstance(attack_cls, type):
        raise TypeError(f"{revision}:attack.py must define AttackAlgorithm")
    return attack_cls


@functools.lru_cache(maxsize=None)
def _agent_factory(selection: str, model_path: str | None):
    if model_path is not None:
        if selection not in {"gemma", "gpt_oss"}:
            raise ValueError("--model-path is supported only with gemma or gpt_oss")
        env_name = "GEMMA_MODEL_PATH" if selection == "gemma" else "GPT_OSS_MODEL_PATH"
        os.environ[env_name] = model_path
    return build_agent_factory(selection)


def run_trial(
    attack_cls: type,
    *,
    strategy: str,
    agent: str,
    seed: int,
    budget_s: float,
    model_path: Path | None,
) -> TrialResult:
    findings = eval_attack(
        attack_cls,
        budget_s,
        agent_factory=_agent_factory(
            agent, str(model_path.resolve()) if model_path is not None else None
        ),
        env_selection="gym",
        options=AttackEvalOptions(
            run_config=AttackRunConfig(
                time_budget_s=budget_s,
                max_steps=10**9,
                max_tool_hops=4,
            ),
            attack_config={"seed": seed},
            env_seed=seed,
            guardrail_factory=Guardrail,
        ),
    )
    breaches = sum(
        is_breach(finding["trace"], finding["predicates"]) for finding in findings
    )
    # eval_attack retains a finding only when the SDK's canonical security
    # predicates fire. A retained finding therefore represents an achieved
    # attack objective; is_breach applies the stricter sensitive-action check.
    objectives = len(findings)
    predicate_counts = {
        name: sum(
            predicate.get("predicate") == name
            for finding in findings
            for predicate in finding["predicates"]
        )
        for name in PREDICATE_NAMES
    }
    return TrialResult(
        strategy=strategy,
        agent=agent,
        seed=seed,
        findings=len(findings),
        breaches=breaches,
        objectives=objectives,
        breach_success=breaches > 0,
        objective_achieved=objectives > 0,
        predicate_counts=predicate_counts,
    )


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    trials = len(results)
    by_predicate = {
        name: {
            "trial_successes": sum(
                result.predicate_counts.get(name, 0) > 0 for result in results
            ),
            "trial_success_rate": statistics.fmean(
                result.predicate_counts.get(name, 0) > 0 for result in results
            ),
            "total_occurrences": sum(
                result.predicate_counts.get(name, 0) for result in results
            ),
        }
        for name in PREDICATE_NAMES
    }
    return {
        "trials": trials,
        "breach_successes": sum(result.breach_success for result in results),
        "objective_successes": sum(result.objective_achieved for result in results),
        "breach_success_rate": statistics.fmean(
            result.breach_success for result in results
        ),
        "objective_achievement_rate": statistics.fmean(
            result.objective_achieved for result in results
        ),
        "total_validated_findings": sum(result.findings for result in results),
        "total_breaches": sum(result.breaches for result in results),
        "by_predicate": by_predicate,
    }


def hardware_fingerprint() -> dict[str, Any]:
    """Return reproducibility metadata without making CUDA mandatory for tests."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu", "cuda_available": False}
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return {
            "device": f"cuda:{device}",
            "cuda_available": True,
            "gpu_name": properties.name,
            "gpu_vram_bytes": properties.total_memory,
            "cuda_version": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    except (ImportError, RuntimeError) as error:
        return {"device": "unknown", "cuda_available": False, "error": str(error)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, default=Path("attack.py"))
    starter = parser.add_mutually_exclusive_group(required=True)
    starter.add_argument("--starter", type=Path)
    starter.add_argument(
        "--starter-revision",
        help="Commit containing the starter attack.py (for example c91e94f)",
    )
    parser.add_argument(
        "--agent",
        choices=("gemma", "gpt_oss", "openai"),
        default="gemma",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local HF model used through the selected SDK adapter (stand-in allowed)",
    )
    parser.add_argument(
        "--model-id",
        help="Stable model identifier stored in the evaluation fingerprint",
    )
    parser.add_argument(
        "--strong-model",
        choices=tuple(STRONG_STAND_INS),
        help=(
            "Select a stronger ungated stand-in and its conventional .models path; "
            "--model-path/--model-id may override either value"
        ),
    )
    parser.add_argument(
        "--real-grader",
        choices=tuple(REAL_GRADERS),
        help="Select a provisioned grading-model approximation and record its revision",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(17, 29, 43))
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--proxy-score", type=float, default=6.2767)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if args.strong_model:
        strong_model_id, strong_model_path = STRONG_STAND_INS[args.strong_model]
        args.model_id = args.model_id or strong_model_id
        args.model_path = args.model_path or strong_model_path
    model_revision = None
    quantization = None
    if args.real_grader:
        model_id, model_path, model_revision = REAL_GRADERS[args.real_grader]
        args.model_id = args.model_id or model_id
        args.model_path = args.model_path or model_path
        quantization = "bitsandbytes-nf4-double-quant"
    if args.model_path is not None and not args.model_path.is_dir():
        raise SystemExit(
            f"--model-path does not exist: {args.model_path}; download the model first"
        )

    strategies = {
        "champion": load_attack(args.champion.resolve()),
        "starter": (
            load_attack(args.starter.resolve())
            if args.starter
            else load_attack_revision(args.starter_revision)
        ),
    }
    results = [
        run_trial(
            attack_cls,
            strategy=strategy,
            agent=args.agent,
            seed=seed,
            budget_s=args.budget,
            model_path=args.model_path,
        )
        for strategy, attack_cls in strategies.items()
        for seed in args.seeds
    ]
    summaries = {
        strategy: summarize(
            [result for result in results if result.strategy == strategy]
        )
        for strategy in strategies
    }
    payload = {
        "schema": "real-agent-transfer-evaluation/v2",
        "fingerprint": {
            "agent_selection": args.agent,
            "model_path": str(args.model_path.resolve()) if args.model_path else None,
            "model_id": args.model_id
            or (args.model_path.name if args.model_path else args.agent),
            "model_revision": model_revision,
            "quantization": quantization,
            "champion_attack": str(args.champion),
            "starter_attack": (
                str(args.starter)
                if args.starter
                else f"git:{args.starter_revision}:attack.py"
            ),
            "seeds": list(args.seeds),
            "budget_s": args.budget,
            "environment": "gym",
            "guardrail": (
                "aicomp_sdk.attacks.baselines.guardrail_rules.Guardrail"
            ),
            "scoring_path": "aicomp_sdk.evaluation.ops.eval_attack",
            "sdk_version": importlib.metadata.version("aicomp-sdk"),
            "python_version": platform.python_version(),
            "temperature": 0,
            "sampling": False,
            "hardware": hardware_fingerprint(),
        },
        "definitions": {
            "trial": "one eval_attack invocation for one strategy and seed",
            "breach_success_rate": "trials containing >=1 SDK is_breach finding / trials",
            "objective_achievement_rate": (
                "trials containing >=1 canonical predicate finding / trials"
            ),
            "predicate_trial_success_rate": (
                "trials containing >=1 occurrence of the named predicate / trials"
            ),
        },
        "proxy": {
            "agent": "deterministic",
            "champion_normalized_score": args.proxy_score,
        },
        "real_agent": summaries,
        "transfer_gap": {
            "proxy_score_minus_champion_breach_rate": (
                args.proxy_score - summaries["champion"]["breach_success_rate"]
            ),
            "proxy_score_minus_champion_objective_rate": (
                args.proxy_score
                - summaries["champion"]["objective_achievement_rate"]
            ),
            "note": (
                "The proxy normalized score and rates have different units; both "
                "raw values are retained to prevent false equivalence."
            ),
        },
        "trials": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["real_agent"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
