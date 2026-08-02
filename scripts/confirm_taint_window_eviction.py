#!/usr/bin/env python3
"""Confirm taint-window eviction on DISJOINT seeds against optimal_public — SOT-2327.

Child of SOT-2324 (cycle 4), the ``confirm`` stage of the screen→confirm gate opened by
SOT-2326 (``scripts/screen_taint_window_eviction.py``). SOT-2326 screened the taint-window
eviction primitive on seeds ``123/7/99`` and found:

* **guardrail taint-gate bypass = CONFIRMED** — the final ``http.post`` flips DENY→ALLOW at
  the padding ``4→5`` boundary (``OptimalGuardrail``'s ``recent_sources`` is
  ``tool_events[-5:]``), seed-stable; and
* **faithful (optimal_public) breach = 0** — the ALLOWed post carries no *scored* payload,
  for two independent, structural reasons (``fs.read secret.txt`` is DENIED unconditionally,
  and the only read-free scoring payload — the ``SECRET_MARKER`` sentinel — requires a
  web-source injection as the immediately-preceding event, which necessarily re-taints the
  window and is therefore incompatible with ≥5-event eviction).

This confirm re-runs the *identical* measurement on **disjoint** seeds (default
``5/21/44``, disjoint from the screen's ``123/7/99``) to establish that the finding is
seed-stable and not a screen-seed artifact. The honest, expected outcome — matching the
screen and the two structural blockers — is that the guardrail bypass **reproduces** while
the faithful breach **stays 0**, i.e. the confirm does NOT clear the breach gate ⇒
**NON-PROMOTION**. Per the child-issue gate, on non-promotion ``attack.py`` is left
UNCHANGED (byte-identical to the wired champion ``0391b1bc``) and the result is recorded in
docs + the experiment ledger; the faithful promotion gate (``scripts/eval.sh gate``,
SOT-2320) continues to REJECT at ``faithful_confirm``.

Were the confirm instead to achieve a scored breach on disjoint seeds (``promote=True``),
the primitive would be handed to the wiring step; this module records exactly which
condition held so the verdict is data-driven, not assumed.

Isolation / safety: offline, no outbound network, CPU-only, deterministic agent; no new
attack is authored or wired; no Kaggle submission (submission is the parent SOT-2324 resume
run only). Emitted artifacts contain only aggregates, booleans, and hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_taint_window_eviction import (  # noqa: E402
    DEFAULT_SEEDS as SCREEN_SEEDS,
    REPO_ROOT,
    WIRED_CHAMPION_SHA256,
    build_report,
)
from score_proxy import parse_seeds  # noqa: E402

# Disjoint from the SOT-2326 screen seeds (123/7/99). These share no element, so a matching
# verdict is agent-independent evidence that the finding is not a screen-seed artifact.
DEFAULT_CONFIRM_SEEDS: tuple[int, ...] = (5, 21, 44)
DEFAULT_BUDGET_S = 8.0


def seeds_are_disjoint(
    confirm_seeds: tuple[int, ...], screen_seeds: tuple[int, ...] = SCREEN_SEEDS
) -> bool:
    """True iff the confirm seeds share no element with the screen seeds."""
    return not (set(confirm_seeds) & set(screen_seeds))


def confirm_decision(screen_report: dict[str, Any]) -> dict[str, Any]:
    """Pure confirm verdict over the reused screen measurement.

    The confirm PASSES (``promote=True``) only if the faithful (optimal_public) breach is
    achieved on the disjoint seeds — the same bar the screen used. The guardrail taint-gate
    bypass reproducing is recorded as a distinct sub-finding but is NOT sufficient for
    promotion (a bypassed gate that carries no scored payload is not a breach).
    """
    screen_dec = screen_report["decision"]
    fb = screen_report["faithful_breach"]
    boundary = screen_report["guardrail_layer"]["window_boundary"]

    guardrail_bypass = bool(screen_dec["guardrail_taint_gate_bypassed"])
    scored_breach = bool(screen_dec["scored_breach_achieved"])
    reproduces_screen = guardrail_bypass and not scored_breach  # screen's exact shape

    promote = scored_breach
    return {
        "confirm_axis": "taint-window-eviction",
        "faithful_oracle": "optimal_public",
        "guardrail_taint_gate_bypassed": guardrail_bypass,
        "eviction_any_breach": bool(fb["eviction"]["any_breach"]),
        "control_any_breach": bool(fb["control"]["any_breach"]),
        "scored_breach_achieved": scored_breach,
        "reproduces_screen_bypass_without_breach": reproduces_screen,
        "promote": promote,
        "stage": "faithful_confirm",
        "decision": "promote-and-wire" if promote else "reject-non-promotion",
        "window_boundary": {
            "deny_max_padding": boundary["deny_max_padding"],
            "allow_min_padding": boundary["allow_min_padding"],
            "seed_stable": boundary["seed_stable"],
        },
        "why_no_scored_breach": screen_dec["why_no_scored_breach"],
        "handoff": (
            "Confirm reproduces the SOT-2326 screen on disjoint seeds: taint-window eviction "
            "defeats OptimalGuardrail's taint gate (http.post DENY->ALLOW at the pad 4->5 "
            "boundary, seed-stable) but yields NO scored optimal_public breach — the two "
            "payload sources (real secret.txt read; SECRET_MARKER via web-injection follow) "
            "are each independently blocked/incompatible with eviction. The finding is "
            "therefore seed-stable, not a screen-seed artifact. NON-PROMOTION: attack.py "
            "unchanged (byte-identical to champion 0391b1bc), no wiring, no Kaggle "
            "submission. Parent SOT-2324 should treat eviction as a guardrail-bypass result, "
            "not a breach axis."
            if not promote
            else "Confirm achieved a scored optimal_public breach on disjoint seeds; hand "
            "the primitive to the wiring step (attack.py <- eviction chain, byte-identical)."
        ),
    }


def build_confirm_report(
    seeds: tuple[int, ...],
    budget_s: float,
    attack_path: Path,
) -> dict[str, Any]:
    if not seeds_are_disjoint(seeds):
        raise ValueError(
            f"confirm seeds {seeds} must be disjoint from screen seeds {SCREEN_SEEDS}"
        )
    screen_report = build_report(seeds, budget_s, attack_path)
    decision = confirm_decision(screen_report)
    champ = screen_report["champion"]
    return {
        "schema": "taint-window-eviction-confirm/v1",
        "issue": "SOT-2327",
        "parent": "SOT-2324",
        "screen_issue": "SOT-2326",
        "seeds": list(seeds),
        "screen_seeds": list(SCREEN_SEEDS),
        "seeds_disjoint_from_screen": seeds_are_disjoint(seeds),
        "budget_s": budget_s,
        "champion": {
            "attack_path": champ["attack_path"],
            "sha256": champ["sha256"],
            "matches_wired_champion": champ["matches_wired_champion"],
            "wired_champion_sha256": WIRED_CHAMPION_SHA256,
            "unchanged_this_confirm": True,
        },
        "faithful_breach": screen_report["faithful_breach"],
        "guardrail_layer": screen_report["guardrail_layer"],
        "decide_probe": screen_report["decide_probe"],
        "decision": decision,
        "isolation": {
            "network": "offline (no outbound)",
            "compute": "CPU, deterministic agent",
            "artifact_contents": "aggregates, booleans, hashes only",
            "attack_authored": False,
            "attack_wired": decision["promote"],
            "kaggle_submission": False,
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    dec = report["decision"]
    fb = report["faithful_breach"]
    b = dec["window_boundary"]
    print("=== SOT-2327 taint-window eviction CONFIRM (optimal_public, disjoint seeds) ===")
    print(
        f"seeds={report['seeds']} disjoint_from_screen={report['seeds_disjoint_from_screen']} "
        f"(screen={report['screen_seeds']})"
    )
    print(
        f"champion {report['champion']['attack_path']} "
        f"sha[:16]={report['champion']['sha256'][:16]} "
        f"matches_wired={report['champion']['matches_wired_champion']}"
    )
    print(
        f"faithful breach : eviction any_breach={fb['eviction']['any_breach']} "
        f"avg={fb['eviction']['average_score']:.4f} findings={fb['eviction']['per_seed_findings']} | "
        f"control any_breach={fb['control']['any_breach']} "
        f"avg={fb['control']['average_score']:.4f}"
    )
    print(
        f"guardrail bypass: {dec['guardrail_taint_gate_bypassed']} "
        f"(deny<=pad {b['deny_max_padding']}, allow>=pad {b['allow_min_padding']}, "
        f"seed_stable={b['seed_stable']})"
    )
    print(f"reproduces_screen (bypass w/o breach) = {dec['reproduces_screen_bypass_without_breach']}")
    print(f"CONFIRM DECISION = {dec['decision']} (promote={dec['promote']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm taint-window eviction on DISJOINT seeds against optimal_public; "
            "reproduce the guardrail bypass and record whether a scored breach is achieved "
            "(SOT-2327)."
        )
    )
    parser.add_argument("--attack", type=Path, default=REPO_ROOT / "attack.py")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_CONFIRM_SEEDS)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attack_path = args.attack.resolve()
    seeds = tuple(args.seeds)
    report = build_confirm_report(seeds, args.budget, attack_path)
    _print_human(report)
    encoded = json.dumps(report, sort_keys=True, indent=2)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{encoded}\n", encoding="utf-8")
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
