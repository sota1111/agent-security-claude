#!/usr/bin/env python3
"""Regression test for the Kaggle scored-rerun entrypoint."""

from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    source = Path("kaggle/kernel/submit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    rerun_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "KAGGLE_IS_COMPETITION_RERUN" in ast.unparse(node.test)
    ]
    assert len(rerun_checks) == 1, "expected exactly one competition-rerun branch"
    check = rerun_checks[0]
    assert isinstance(check.test, ast.Compare), (
        "rerun environment variable must be checked for presence, not truthiness"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "serve"
        for node in ast.walk(ast.Module(body=check.body, type_ignores=[]))
    ), "scored rerun branch must start the inference server"
    print("kernel rerun entrypoint: PASS")


if __name__ == "__main__":
    main()
