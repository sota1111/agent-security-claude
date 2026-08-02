#!/usr/bin/env python3
"""Kaggle-faithful promotion gate — SOT-2320.

Re-anchors champion promotion on the **Kaggle-faithful** local signal so that a
proxy-only improvement can never promote a champion again.

Background (SOT-2319, ``docs/kpi.md``). The persistent public-LB **0.000** with a
strongly non-zero local ``beatable-rules`` proxy (avg ~6.28) was pinpointed — by a
controlled guardrail-swap experiment — to the **guardrail** layer: the proxy
(``aicomp_sdk.attacks.baselines.guardrail_rules.Guardrail``) has a phrasing-gated
allow-hole the hidden Kaggle guardrail does not have, so it *over-reports*. The one
local signal that agrees with Kaggle is the ``optimal_public`` oracle
(``aicomp_sdk.evaluation.ops.DEFAULT_ATTACK_GUARDRAIL_ID`` =
``aicomp_sdk.guardrails.optimal.Guardrail``). SOT-2319's handoff: *gate promotion on
``optimal_public`` being non-zero; never promote on proxy-only (or gemma-only)
evidence.*

This gate enforces a three-stage decision, short-circuiting in order:

1. **integrity** — exec-compat (the exact ``attack.py`` runs under the Kaggle
   ``exec`` sandbox contract) **and** the SOT-2318 kernel byte-identity guard (the
   submission kernel embeds a byte-identical, non-stale copy of ``attack.py``). A
   failure here means the thing being scored is not the thing that would be
   submitted, so no score is trustworthy → reject.
2. **proxy screen** — the cheap beatable-rules proxy, now *demoted to a screen-only
   filter*. It is necessary but **not sufficient**: passing it alone never promotes.
3. **faithful confirm** — the ``optimal_public`` oracle (the graded-rerun-faithful
   guardrail) must be **non-zero AND strictly better than the champion's** faithful
   score. If the faithful signal does not improve, the challenger is **rejected even
   when the proxy rose** — this is the oracle-drift guard SOT-2320 exists to add.

The decision core (:func:`decide_promotion`) is a pure function over already-measured
signals, so it is directly unit-testable without any model, GPU, or ``eval_attack``
call (see ``scripts/test_promotion_gate.py``). The CLI wires the real measurements
(integrity subchecks, proxy screen, optimal_public confirm) into it.

Isolation / safety: offline, deterministic CPU path; no outbound network and no new
attack authored (evaluation-gate infra only). Emitted artifacts contain only
aggregates, booleans, and hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Sibling scripts (score_proxy, reconcile_submission_oracle, the guard tests) live
# alongside this file; make them importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

# The current champion's faithful (optimal_public) score. SOT-2319 measured it at
# 0.0000 (the strategy does not beat the optimal-class guardrail), so any challenger
# that is *non-zero* on the faithful signal is a strict improvement. Overridable via
# --champion-faithful-score when a future champion records a non-zero faithful score.
DEFAULT_CHAMPION_FAITHFUL_SCORE = 0.0


# --------------------------------------------------------------------------- #
# Pure decision core (unit-testable without eval_attack / models / GPU)       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateInputs:
    """The already-measured signals the gate decides over.

    ``kernel_byte_identical`` / ``exec_compatible`` are the SOT-2318 integrity
    subchecks. ``proxy_screen_passed`` is the cheap beatable-rules screen verdict.
    ``faithful_*`` are the ``optimal_public`` oracle measurements for the challenger,
    and ``champion_faithful_score`` is the incumbent's recorded faithful score.
    """

    kernel_byte_identical: bool
    exec_compatible: bool
    proxy_screen_passed: bool
    proxy_screen_score: float
    faithful_any_breach: bool
    faithful_score: float
    champion_faithful_score: float = DEFAULT_CHAMPION_FAITHFUL_SCORE


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    stage_reached: str  # "integrity" | "proxy_screen" | "faithful_confirm"
    reason: str
    checks: dict = field(default_factory=dict)


def decide_promotion(inp: GateInputs) -> GateDecision:
    """Decide promotion from measured signals, short-circuiting in gate order.

    Order is load-bearing: (1) integrity, (2) proxy screen, (3) faithful confirm.
    The faithful-confirm rule is the SOT-2320 invariant — a proxy-only improvement
    (proxy passes but the faithful signal does not improve) is **rejected**.
    """
    faithful_nonzero = inp.faithful_any_breach and inp.faithful_score > 0.0
    faithful_improves = inp.faithful_score > inp.champion_faithful_score
    checks = {
        "integrity": {
            "kernel_byte_identical": inp.kernel_byte_identical,
            "exec_compatible": inp.exec_compatible,
            "passed": inp.kernel_byte_identical and inp.exec_compatible,
        },
        "proxy_screen": {
            "score": inp.proxy_screen_score,
            "passed": inp.proxy_screen_passed,
        },
        "faithful_confirm": {
            "signal": "optimal_public",
            "score": inp.faithful_score,
            "any_breach": inp.faithful_any_breach,
            "champion_score": inp.champion_faithful_score,
            "non_zero": faithful_nonzero,
            "improves_over_champion": faithful_improves,
            "passed": faithful_nonzero and faithful_improves,
        },
    }

    # Stage 1 — integrity: the scored artifact must equal the submitted artifact.
    if not (inp.kernel_byte_identical and inp.exec_compatible):
        return GateDecision(
            promote=False,
            stage_reached="integrity",
            reason=(
                "REJECT (integrity): "
                f"kernel_byte_identical={inp.kernel_byte_identical}, "
                f"exec_compatible={inp.exec_compatible}. The evaluated attack.py is "
                "not byte-identical to the submission kernel (stale/drifted embed, "
                "SOT-2318) or fails the Kaggle exec sandbox contract, so no score is "
                "trustworthy."
            ),
            checks=checks,
        )

    # Stage 2 — proxy screen: cheap filter, necessary but NOT sufficient.
    if not inp.proxy_screen_passed:
        return GateDecision(
            promote=False,
            stage_reached="proxy_screen",
            reason=(
                "REJECT (proxy screen): the cheap beatable-rules screen did not pass "
                f"(score={inp.proxy_screen_score:.4f}). Screened out before the "
                "faithful confirm."
            ),
            checks=checks,
        )

    # Stage 3 — faithful confirm: optimal_public non-zero AND strictly improving.
    if faithful_nonzero and faithful_improves:
        return GateDecision(
            promote=True,
            stage_reached="faithful_confirm",
            reason=(
                "PROMOTE: Kaggle-faithful signal optimal_public improved "
                f"({inp.faithful_score:.4f} > champion {inp.champion_faithful_score:.4f}) "
                "with a real breach on the graded-rerun-faithful guardrail, after "
                "passing the integrity guards and the proxy screen."
            ),
            checks=checks,
        )

    return GateDecision(
        promote=False,
        stage_reached="faithful_confirm",
        reason=(
            "REJECT (faithful confirm — oracle-drift guard): the beatable-rules proxy "
            f"passed (score={inp.proxy_screen_score:.4f}) but the Kaggle-faithful "
            f"optimal_public signal did NOT improve (score={inp.faithful_score:.4f}, "
            f"any_breach={inp.faithful_any_breach}, champion="
            f"{inp.champion_faithful_score:.4f}). Proxy-only improvements are never "
            "promoted (SOT-2319/SOT-2320): the proxy over-reports against a guardrail "
            "with a phrasing hole the hidden Kaggle guardrail lacks."
        ),
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# Live measurement wiring (integrity subchecks / proxy screen / faithful)     #
# --------------------------------------------------------------------------- #
def check_kernel_byte_identity() -> tuple[bool, dict[str, Any]]:
    """SOT-2318 guard: submission kernel embeds a byte-identical, non-stale attack.py."""
    from test_kernel_payload_identity import (  # noqa: E402
        ATTACK_PY,
        PREVIOUS_SUBMITTED_SHA256,
        _embedded_payload,
        _sha256,
    )

    attack_bytes = ATTACK_PY.read_bytes()
    payload = _embedded_payload()
    attack_sha = _sha256(attack_bytes)
    payload_sha = _sha256(payload)
    byte_identical = payload == attack_bytes
    not_stale = payload_sha != PREVIOUS_SUBMITTED_SHA256
    return byte_identical and not_stale, {
        "attack_sha256_16": attack_sha[:16],
        "kernel_payload_sha256_16": payload_sha[:16],
        "byte_identical": byte_identical,
        "not_stale_vs_previous_submission": not_stale,
    }


def check_exec_compat() -> tuple[bool, dict[str, Any]]:
    """Kaggle exec-sandbox contract: attack.py runs with no __file__/__name__, changed cwd."""
    import test_exec_compat  # noqa: E402

    try:
        test_exec_compat.main()
        return True, {"exec_compat": "PASS"}
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"exec_compat": f"FAIL: {exc}"}


def measure_proxy_screen(
    attack_cls: type, *, seeds: tuple[int, ...], budget_s: float, min_score: float
) -> tuple[bool, float, dict[str, Any]]:
    """Cheap beatable-rules screen (score_proxy), demoted to a screen-only filter."""
    from score_proxy import score_stage  # noqa: E402

    stage = score_stage(
        attack_cls,
        stage="screen",
        seeds=seeds,
        budget_s=budget_s,
        minimum_score=min_score,
    )
    return stage.passed, stage.average_score, {
        "seeds": list(seeds),
        "budget_s": budget_s,
        "min_score": min_score,
        "average_score": stage.average_score,
        "passed": stage.passed,
    }


def measure_faithful_signal(
    attack_cls: type, *, seeds: tuple[int, ...], budget_s: float
) -> tuple[bool, float, dict[str, Any]]:
    """The Kaggle-faithful optimal_public oracle score (OptimalGuardrail via eval_attack)."""
    from reconcile_submission_oracle import (  # noqa: E402
        DEFAULT_ATTACK_GUARDRAIL_ID,
        GUARDRAILS,
        run_guardrail_signal,
    )

    guardrail_path, guardrail_factory = GUARDRAILS[DEFAULT_ATTACK_GUARDRAIL_ID]
    signal = run_guardrail_signal(
        attack_cls,
        signal_id="c",
        guardrail_id=DEFAULT_ATTACK_GUARDRAIL_ID,
        guardrail_path=guardrail_path,
        guardrail_factory=guardrail_factory,
        seeds=seeds,
        budget_s=budget_s,
    )
    return signal.any_breach, signal.average_score, {
        "guardrail_id": DEFAULT_ATTACK_GUARDRAIL_ID,
        "guardrail_path": guardrail_path,
        "seeds": list(seeds),
        "budget_s": budget_s,
        "average_score": signal.average_score,
        "max_score": signal.max_score,
        "any_breach": signal.any_breach,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_seeds(value: str) -> tuple[int, ...]:
    from score_proxy import parse_seeds  # noqa: E402

    return parse_seeds(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-faithful promotion gate: integrity -> proxy screen -> "
            "optimal_public faithful confirm. Never promotes on proxy-only evidence "
            "(SOT-2320)."
        )
    )
    parser.add_argument("--attack", type=Path, default=REPO_ROOT / "attack.py")
    parser.add_argument("--screen-seeds", type=_parse_seeds, default=(123, 7, 99))
    parser.add_argument("--screen-budget", type=float, default=25.0)
    parser.add_argument("--screen-min-score", type=float, default=0.33)
    parser.add_argument("--faithful-seeds", type=_parse_seeds, default=(123, 7, 99))
    parser.add_argument("--faithful-budget", type=float, default=25.0)
    parser.add_argument(
        "--champion-faithful-score",
        type=float,
        default=DEFAULT_CHAMPION_FAITHFUL_SCORE,
        help="incumbent champion's optimal_public score to beat (default 0.0, SOT-2319)",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def run_gate(args: argparse.Namespace) -> tuple[GateDecision, dict[str, Any]]:
    from score_proxy import load_attack  # noqa: E402

    attack_path = args.attack.resolve()
    attack_cls = load_attack(attack_path)

    kernel_ok, kernel_detail = check_kernel_byte_identity()
    exec_ok, exec_detail = check_exec_compat()

    # Short-circuit the expensive eval_attack stages if integrity already failed, but
    # still produce a full decision object for the report.
    if kernel_ok and exec_ok:
        proxy_passed, proxy_score, proxy_detail = measure_proxy_screen(
            attack_cls,
            seeds=args.screen_seeds,
            budget_s=args.screen_budget,
            min_score=args.screen_min_score,
        )
        if proxy_passed:
            faithful_breach, faithful_score, faithful_detail = measure_faithful_signal(
                attack_cls, seeds=args.faithful_seeds, budget_s=args.faithful_budget
            )
        else:
            faithful_breach, faithful_score = False, 0.0
            faithful_detail = {"skipped": "proxy screen did not pass"}
    else:
        proxy_passed, proxy_score, proxy_detail = False, 0.0, {"skipped": "integrity failed"}
        faithful_breach, faithful_score = False, 0.0
        faithful_detail = {"skipped": "integrity failed"}

    inputs = GateInputs(
        kernel_byte_identical=kernel_ok,
        exec_compatible=exec_ok,
        proxy_screen_passed=proxy_passed,
        proxy_screen_score=proxy_score,
        faithful_any_breach=faithful_breach,
        faithful_score=faithful_score,
        champion_faithful_score=args.champion_faithful_score,
    )
    decision = decide_promotion(inputs)
    report = {
        "schema": "promotion-gate/v1",
        "issue": "SOT-2320",
        "attack_path": str(attack_path.relative_to(REPO_ROOT)),
        "promote": decision.promote,
        "stage_reached": decision.stage_reached,
        "reason": decision.reason,
        "inputs": asdict(inputs),
        "measurements": {
            "integrity": {"kernel": kernel_detail, "exec": exec_detail},
            "proxy_screen": proxy_detail,
            "faithful_confirm": faithful_detail,
        },
        "isolation": {
            "network": "offline (no outbound)",
            "artifact_contents": "aggregates, booleans, hashes only",
            "new_attack_authored": False,
        },
    }
    return decision, report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision, report = run_gate(args)
    encoded = json.dumps(report, sort_keys=True)
    verdict = "PROMOTE" if decision.promote else "REJECT"
    print(f"promotion-gate: {verdict} (stage={decision.stage_reached})")
    print(decision.reason)
    print(f"GATE_RESULT_JSON={encoded}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_output}")
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
