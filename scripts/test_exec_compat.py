#!/usr/bin/env python3
"""Exercise the exact submission file under Kaggle-like ``exec`` conditions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from aicomp_sdk.attacks import AttackRunConfig


class _ExecSandboxEnv:
    """Small protocol-compatible environment for the submission import gate."""

    def __init__(self) -> None:
        self._messages: list[str] = []

    def reset(self) -> None:
        self._messages = []

    def interact(self, message: str, *, max_tool_hops: int) -> None:
        del max_tool_hops
        self._messages.append(message)

    def export_trace_dict(self) -> dict[str, Any]:
        return {"user_messages": list(self._messages)}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    submission = repo_root / "attack.py"
    source = submission.read_text(encoding="utf-8")

    # Kaggle loads the submission source from outside the caller's cwd. Do not
    # provide __file__ or __name__, mirroring the strict exec sandbox contract.
    namespace: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as isolated_cwd:
        previous_cwd = Path.cwd()
        try:
            os.chdir(isolated_cwd)
            exec(compile(source, "attack.py", "exec"), namespace)
        finally:
            os.chdir(previous_cwd)

    algorithm_class = namespace.get("AttackAlgorithm")
    if algorithm_class is None:
        raise AssertionError("AttackAlgorithm was not defined by attack.py")

    # Make the deterministic stub trace count as a finding. The purpose of this
    # gate is the submission contract, not predicate correctness (covered by
    # the SDK evaluation).
    namespace["eval_predicates"] = lambda trace: bool(trace.get("user_messages"))
    algorithm = algorithm_class({"seed": 1969, "min_margin_s": 0})
    candidates = algorithm.run(
        _ExecSandboxEnv(),
        AttackRunConfig(time_budget_s=1, max_steps=1, max_tool_hops=1),
    )
    if not candidates:
        raise AssertionError("AttackAlgorithm returned no candidate")

    print(
        "exec compatibility: PASS "
        "(exact attack.py, no __file__/__name__, changed cwd, candidate returned)"
    )


if __name__ == "__main__":
    main()
