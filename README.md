# agent-security-claude

Kaggle **AI Agent Security — Multi-Step Tool Attacks** competition
(`ai-agent-security-multi-step-tool-attacks`), red-team / attack-only track —
Claude model variant.

The submission is a single `attack.py` defining
`AttackAlgorithm(AttackAlgorithmBase)`, whose `run(env, config)` returns a
`list[AttackCandidate]`: replayable attacker-message chains that make the blue
agent violate a security predicate (secret exfiltration, destructive write,
confused-deputy, …). The evaluator replays each candidate, recomputes the
predicates/cell signatures, and scores only the replayed traces.

## Layout

| Path | Purpose |
| --- | --- |
| `attack.py` | The submission. Starter = time-boxed prompt-bank search using the SDK's `eval_predicates` as the breach detector. |
| `requirements.txt` | Pins `aicomp-sdk>=3.1.2` (competition SDK). |
| `scripts/eval.sh` | Thin wrapper over `aicomp validate/test/evaluate redteam`. |
| `scripts/test_exec_compat.py` | Executes the exact `attack.py` with no `__file__`/`__name__` and a changed cwd. |
| `kaggle/kernel/` | Kaggle code-competition submission kernel (`submit.py` embeds the champion `attack.py` verbatim via base64 and serves `JEDAttackInferenceServer`). |
| `docs/kpi.md` | KPI log — local eval rows + the Kaggle submission log (real refs/scores). |

## Setup (local, inside the Dev Container)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`aicomp-sdk` pulls heavy transitive deps (torch/transformers/…); the first
install takes a few minutes.

## Local evaluation

All commands go through the wrapper (or call `aicomp` directly). The wrapper
auto-uses `.venv/bin/aicomp` when present.

```bash
# 1. Validate the submission shape (syntax / imports / attack structure).
scripts/eval.sh validate

# 2. Quick functional test against the deterministic blue agent.
scripts/eval.sh test --agent deterministic --budget-s 60

# 3. Scored local evaluation on the Kaggle-equivalent gym path.
scripts/eval.sh evaluate --env gym --budget-s 60
```

Equivalent raw commands:

```bash
.venv/bin/aicomp validate redteam attack.py
.venv/bin/aicomp test     redteam attack.py --agent deterministic --budget-s 60
.venv/bin/aicomp evaluate redteam attack.py --env gym --budget-s 60
```

Record the normalized attack score printed by `evaluate` into `docs/kpi.md`.

Before a real submission, run the Kaggle exec-compatibility gate against the
same `attack.py` that will be uploaded:

```bash
.venv/bin/python scripts/test_exec_compat.py
```

## Improvement loop (screen → confirm)

This repo is the C1 bootstrap for the SOT-1961 順位向上サイクル:

- **C1 (this issue, SOT-1967):** valid `attack.py` starter + local eval harness + KPI log.
- **C2:** establish a champion attack strategy (stronger search than the starter).
- **C3:** real Kaggle submission of the champion.

Screen a candidate strategy with a short `--budget-s` eval; only promote it to
champion after a longer confirm run beats the current KPI baseline.
