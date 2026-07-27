#!/usr/bin/env bash
# Thin wrapper over the aicomp red-team CLI for local evaluation.
#
# Usage:
#   scripts/eval.sh validate                       # aicomp validate redteam attack.py
#   scripts/eval.sh test  [extra aicomp args...]   # aicomp test  redteam attack.py ...
#   scripts/eval.sh evaluate [extra aicomp args...]# aicomp evaluate redteam attack.py ...
#   scripts/eval.sh proxy all [proxy args...]       # screen, then confirm
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
  *)
    echo "usage: scripts/eval.sh {validate|test|evaluate|proxy} [args...]" >&2
    exit 2
    ;;
esac
