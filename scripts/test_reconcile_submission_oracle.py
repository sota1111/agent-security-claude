#!/usr/bin/env python3
"""Unit tests for the SOT-2319 submission-oracle reconciliation harness.

Pure-function coverage only — no model weights, no GPU, no ``eval_attack`` call.
Runs under pytest and standalone (``python scripts/test_reconcile_submission_oracle.py``)
so the fast checks can gate ``eval.sh`` if desired.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconcile_submission_oracle import (  # noqa: E402
    GemmaSignal,
    GuardrailSignal,
    SeedCell,
    _extract_gemma_rates,
    _predicate_counts,
    attribute_divergence,
    guardrail_decision_probe,
)

BEATABLE = "aicomp_sdk.attacks.baselines.guardrail_rules.Guardrail"
OPTIMAL = "aicomp_sdk.guardrails.optimal.Guardrail"


def _proxy_signal(any_breach: bool = True, max_score: float = 3.5) -> GuardrailSignal:
    cell = SeedCell(123, max_score, 700.0, 21 if any_breach else 0, 14, {"UNTRUSTED_TO_ACTION": 42})
    return GuardrailSignal(
        signal_id="a",
        guardrail_id="beatable_rules",
        guardrail_path=BEATABLE,
        agent="deterministic",
        budget_s=20.0,
        seeds=(123,),
        average_score=max_score,
        max_score=max_score,
        any_breach=any_breach,
        cells=(cell,),
    )


def _oracle_signal(any_breach: bool = False) -> GuardrailSignal:
    cell = SeedCell(123, 0.0, 0.0, 0, 0, {})
    return GuardrailSignal(
        signal_id="c",
        guardrail_id="optimal_public",
        guardrail_path=OPTIMAL,
        agent="deterministic",
        budget_s=20.0,
        seeds=(123,),
        average_score=0.0,
        max_score=0.0,
        any_breach=any_breach,
        cells=(cell,),
    )


def _gemma_signal(any_breach: bool = True, breach: float | None = 0.5) -> GemmaSignal:
    return GemmaSignal(
        signal_id="b",
        source="ingested-artifact",
        guardrail_path=BEATABLE,
        model_id="gemma-3-4b-nf4",
        model_revision="eb03c885",
        breach_success_rate=breach,
        real_secret_leak_rate=0.5,
        any_breach=any_breach,
        seeds=[17, 29, 43],
        artifact_path="artifacts/real-agent-transfer/x.json",
        artifact_sha256_16="deadbeefdeadbeef",
        note="",
    )


def test_attribution_points_at_guardrail_layer() -> None:
    """proxy non-zero + oracle zero + gemma non-zero (beatable) => guardrail layer."""
    attr = attribute_divergence(_proxy_signal(), _oracle_signal(), _gemma_signal())
    assert attr["divergence_layer"] == "guardrail"
    assert attr["guardrail_zeroes_signal"] is True
    assert attr["kaggle_faithful_signal"] == "optimal_public"
    assert all(attr["held_constant_between_proxy_and_oracle"].values())
    assert "SOT-2320" in attr["handoff_to_SOT_2320"] or "optimal_public" in attr["handoff_to_SOT_2320"]


def test_attribution_holds_when_gemma_unavailable_offline() -> None:
    """No gemma artifact (breach rate None) must not block the guardrail verdict."""
    attr = attribute_divergence(_proxy_signal(), _oracle_signal(), _gemma_signal(any_breach=False, breach=None))
    assert attr["divergence_layer"] == "guardrail"


def test_attribution_inconclusive_when_no_collapse() -> None:
    """If the oracle also breaches, there is no non-zero->zero collapse to attribute."""
    attr = attribute_divergence(_proxy_signal(), _oracle_signal(any_breach=True), _gemma_signal())
    assert attr["divergence_layer"] == "inconclusive"
    assert attr["guardrail_zeroes_signal"] is False


def test_predicate_counts_aggregates_names() -> None:
    findings = [
        {"predicates": [{"predicate": "UNTRUSTED_TO_ACTION", "severity": 5}]},
        {"predicates": [{"predicate": "EXFILTRATION"}, {"predicate": "UNTRUSTED_TO_ACTION"}]},
        {"predicates": []},
    ]
    assert _predicate_counts(findings) == {"EXFILTRATION": 1, "UNTRUSTED_TO_ACTION": 2}


def test_extract_gemma_rates_from_regression_check() -> None:
    doc = {"gemma_regression_check": {"observed": {"breach_success_rate": 0.5, "real_secret_leak_rate": 0.5}}}
    assert _extract_gemma_rates(doc) == (0.5, 0.5)


def test_extract_gemma_rates_falls_back_to_baseline() -> None:
    doc = {"gemma_regression_check": {"baseline": {"breach_success_rate": 1.0, "real_secret_leak_rate": 1.0}}}
    assert _extract_gemma_rates(doc) == (1.0, 1.0)


def test_extract_gemma_rates_missing_is_none() -> None:
    assert _extract_gemma_rates({}) == (None, None)


def test_decision_probe_shows_the_phrasing_hole() -> None:
    """The mechanism probe must show beatable ALLOW where optimal_public DENY."""
    probe = guardrail_decision_probe()
    assert probe, "probe produced no cases"
    for row in probe:
        assert row["beatable_rules"]["allowed"] is True, row
        assert row["optimal_public"]["allowed"] is False, row


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
