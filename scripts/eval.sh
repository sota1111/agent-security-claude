#!/usr/bin/env bash
# Thin wrapper over the aicomp red-team CLI for local evaluation.
#
# Usage:
#   scripts/eval.sh validate                       # aicomp validate redteam attack.py
#   scripts/eval.sh test  [extra aicomp args...]   # aicomp test  redteam attack.py ...
#   scripts/eval.sh evaluate [extra aicomp args...]# aicomp evaluate redteam attack.py ...
#   scripts/eval.sh proxy all [proxy args...]       # cheap beatable-rules screen filter
#   scripts/eval.sh gate [gate args...]             # faithful promotion gate (SOT-2320)
#   scripts/eval.sh real-agent [real-agent args...]  # LLM transfer measurement
#   scripts/eval.sh real-agent --strong-model qwen-1.5b ... # stronger stand-in
#   scripts/eval.sh cross-agent --output <path>       # cross-agent transfer matrix
#   scripts/eval.sh provision-real-grader             # download + GPU smoke test
#
# Prefers ./.venv/bin/aicomp; falls back to aicomp on PATH.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -x "$repo_root/.venv/bin/aicomp" ]]; then
  AICOMP="$repo_root/.venv/bin/aicomp"
  PYTHON="$repo_root/.venv/bin/python"
else
  AICOMP="aicomp"
  PYTHON="${PYTHON:-python3}"
fi

sub="${1:-}"
shift || true

case "$sub" in
  validate)
    # Stale-kernel guard (SOT-2318): before validating attack.py, assert the Kaggle
    # submission kernel embeds a byte-identical copy of it. The graded rerun ships
    # whatever kaggle/kernel/submit.py re-materialises, so a drifted _ATTACK_PY_B64
    # would silently submit a different strategy than the one being validated here
    # (the standing cause of public LB 0.000). Stdlib-only unittest, sub-second.
    "$PYTHON" scripts/test_kernel_payload_identity.py
    exec "$AICOMP" validate redteam attack.py "$@"
    ;;
  test)
    exec "$AICOMP" test redteam attack.py "$@"
    ;;
  evaluate)
    exec "$AICOMP" evaluate redteam attack.py "$@"
    ;;
  proxy)
    # Cheap beatable-rules screen. SOT-2320 demoted this to a screen-only filter:
    # passing the proxy alone does NOT promote. Use `eval.sh gate` for a promotion
    # decision (integrity -> proxy screen -> optimal_public faithful confirm).
    exec "$PYTHON" scripts/score_proxy.py "$@"
    ;;
  gate)
    # Kaggle-faithful promotion gate (SOT-2320). Runs, in order: (1) exec-compat +
    # kernel byte-identity guard (SOT-2318), (2) beatable-rules proxy screen, (3) the
    # optimal_public faithful confirm. Never promotes on proxy-only evidence — the
    # faithful signal must be non-zero AND strictly better than the champion. Exit 0
    # promote / 1 reject.
    exec "$PYTHON" scripts/promotion_gate.py "$@"
    ;;
  real-agent)
    exec "$PYTHON" scripts/score_real_agent.py "$@"
    ;;
  cross-agent)
    exec "$PYTHON" scripts/screen_cross_agent_transfer.py "$@"
    ;;
  provision-real-grader)
    exec "$PYTHON" scripts/provision_real_grader.py "$@"
    ;;
  *)
    echo "usage: scripts/eval.sh {validate|test|evaluate|proxy|gate|real-agent|cross-agent|provision-real-grader} [args...]" >&2
    exit 2
    ;;
esac
