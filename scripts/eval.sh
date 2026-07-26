#!/usr/bin/env bash
# Thin wrapper over the aicomp red-team CLI for local evaluation.
#
# Usage:
#   scripts/eval.sh validate                       # aicomp validate redteam attack.py
#   scripts/eval.sh test  [extra aicomp args...]   # aicomp test  redteam attack.py ...
#   scripts/eval.sh evaluate [extra aicomp args...]# aicomp evaluate redteam attack.py ...
#
# Prefers ./.venv/bin/aicomp; falls back to aicomp on PATH.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -x "$repo_root/.venv/bin/aicomp" ]]; then
  AICOMP="$repo_root/.venv/bin/aicomp"
else
  AICOMP="aicomp"
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
  *)
    echo "usage: scripts/eval.sh {validate|test|evaluate} [aicomp args...]" >&2
    exit 2
    ;;
esac
