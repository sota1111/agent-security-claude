#!/usr/bin/env python3
"""Assert attack.py's wired genuine-carriage chains are byte-identical to the confirmed vector.

SOT-2296 promoted the ``exfil_genuine_deferred_genuine_carriage`` vector (native gemma-3-4b-nf4
grader, adapter-fix OFF, confirmed on disjoint seeds 5/21/44) by wiring its exact steered chains
into ``attack.py``. This gate pins that wiring to the source of truth — the screen harness's
``build_screen_vectors`` — so any drift in the preamble or base chains is caught before submit.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_attack_module():
    spec = importlib.util.spec_from_file_location("attack_champion", REPO_ROOT / "attack.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from screen_genuine_carriage import build_screen_vectors

    confirmed = build_screen_vectors()["exfil_genuine_deferred_genuine_carriage"]
    attack = _load_attack_module()

    assert attack.GENUINE_CARRIAGE_CHAINS == confirmed, (
        "attack.py GENUINE_CARRIAGE_CHAINS drifted from the SOT-2296-confirmed vector "
        "exfil_genuine_deferred_genuine_carriage; re-sync the preamble/base chains."
    )

    print(
        "genuine-carriage wiring: PASS "
        f"({len(confirmed)} chains byte-identical to the confirmed native vector)"
    )


if __name__ == "__main__":
    main()
