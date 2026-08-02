#!/usr/bin/env bash
# Thin wrapper over the aicomp red-team CLI for local evaluation.
#
# Usage:
#   scripts/eval.sh validate                       # aicomp validate redteam attack.py
#   scripts/eval.sh test  [extra aicomp args...]   # aicomp test  redteam attack.py ...
#   scripts/eval.sh evaluate [extra aicomp args...]# aicomp evaluate redteam attack.py ...
#   scripts/eval.sh proxy all [proxy args...]       # screen, then confirm
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
    exec "$PYTHON" scripts/score_proxy.py "$@"
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
    echo "usage: scripts/eval.sh {validate|test|evaluate|proxy|real-agent|cross-agent|provision-real-grader} [args...]" >&2
    exit 2
    ;;
esac
