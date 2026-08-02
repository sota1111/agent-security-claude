#!/usr/bin/env python3
"""Unit tests for the SOT-2320 Kaggle-faithful promotion gate.

Pure-function coverage of :func:`decide_promotion` — no model weights, no GPU, no
``eval_attack`` call. The load-bearing assertion is the SOT-2320 invariant: a
**proxy-only** improvement (beatable-rules proxy passes, but the Kaggle-faithful
``optimal_public`` signal does not improve) must be **rejected**.

Runs under pytest and standalone
(``python scripts/test_promotion_gate.py``) so the fast checks can gate ``eval.sh``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from promotion_gate import GateInputs, decide_promotion  # noqa: E402


def _inputs(**overrides) -> GateInputs:
    base = dict(
        kernel_byte_identical=True,
        exec_compatible=True,
        proxy_screen_passed=True,
        proxy_screen_score=6.2767,
        faithful_any_breach=True,
        faithful_score=0.5,
        champion_faithful_score=0.0,
    )
    base.update(overrides)
    return GateInputs(**base)


def test_proxy_only_improvement_is_rejected() -> None:
    """THE SOT-2320 invariant: proxy passes (high) but faithful does not improve => REJECT."""
    decision = decide_promotion(
        _inputs(
            proxy_screen_passed=True,
            proxy_screen_score=6.2767,  # proxy strongly up
            faithful_any_breach=False,  # optimal_public unchanged / zero
            faithful_score=0.0,
        )
    )
    assert decision.promote is False
    assert decision.stage_reached == "faithful_confirm"
    assert decision.checks["proxy_screen"]["passed"] is True
    assert decision.checks["faithful_confirm"]["passed"] is False


def test_faithful_improvement_promotes() -> None:
    """Faithful signal non-zero and strictly above the champion => PROMOTE."""
    decision = decide_promotion(
        _inputs(faithful_any_breach=True, faithful_score=0.5, champion_faithful_score=0.0)
    )
    assert decision.promote is True
    assert decision.stage_reached == "faithful_confirm"
    assert decision.checks["faithful_confirm"]["improves_over_champion"] is True


def test_faithful_non_improvement_over_nonzero_champion_is_rejected() -> None:
    """Faithful non-zero but not strictly better than a non-zero champion => REJECT."""
    decision = decide_promotion(
        _inputs(faithful_any_breach=True, faithful_score=0.4, champion_faithful_score=0.5)
    )
    assert decision.promote is False
    assert decision.stage_reached == "faithful_confirm"
    assert decision.checks["faithful_confirm"]["improves_over_champion"] is False


def test_faithful_breach_flag_required_even_if_score_positive() -> None:
    """A positive score with no recorded breach is not a real faithful pass."""
    decision = decide_promotion(
        _inputs(faithful_any_breach=False, faithful_score=0.3, champion_faithful_score=0.0)
    )
    assert decision.promote is False
    assert decision.checks["faithful_confirm"]["non_zero"] is False


def test_integrity_failure_short_circuits_before_proxy() -> None:
    """Kernel drift / exec-incompat blocks promotion regardless of downstream signals."""
    for bad in ("kernel_byte_identical", "exec_compatible"):
        decision = decide_promotion(
            _inputs(**{bad: False}, faithful_any_breach=True, faithful_score=9.9)
        )
        assert decision.promote is False, bad
        assert decision.stage_reached == "integrity", bad
        assert decision.checks["integrity"]["passed"] is False, bad


def test_proxy_screen_failure_short_circuits_before_faithful() -> None:
    """A challenger that fails the cheap screen never reaches the faithful confirm."""
    decision = decide_promotion(
        _inputs(proxy_screen_passed=False, proxy_screen_score=0.1, faithful_any_breach=True, faithful_score=9.9)
    )
    assert decision.promote is False
    assert decision.stage_reached == "proxy_screen"


def test_integrity_takes_precedence_over_proxy_screen() -> None:
    """When both integrity and proxy fail, the reported stage is the earliest (integrity)."""
    decision = decide_promotion(
        _inputs(kernel_byte_identical=False, proxy_screen_passed=False)
    )
    assert decision.stage_reached == "integrity"


def test_full_pass_reason_mentions_optimal_public() -> None:
    """The promote reason cites the Kaggle-faithful signal by name for auditability."""
    decision = decide_promotion(_inputs())
    assert decision.promote is True
    assert "optimal_public" in decision.reason


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
