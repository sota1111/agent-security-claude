#!/usr/bin/env python3
"""Superseded gate — genuine-carriage wiring is no longer part of the champion.

SOT-2296 wired the ``exfil_genuine_deferred_genuine_carriage`` vector into
``attack.py`` and this gate pinned that wiring byte-for-byte. That lineage was
**replaced** by the single-message raw-weighted fill champion (adopted from the
public ``aisec-pilk`` scoring analysis): the champion no longer carries any
``GENUINE_CARRIAGE_CHAINS``. The genuine-carriage wiring invariant is therefore
retired. The structural contract of the new champion is pinned by
``scripts/test_floor_guarantee.py`` (ChampionFillStructureTest) and the
submission-integrity invariants by ``scripts/test_kernel_payload_identity.py`` /
``scripts/test_exec_compat.py``.

This file is kept (not deleted) as a tombstone so the retirement is auditable; it
asserts the pivot actually happened rather than silently passing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_attack_module():
    spec = importlib.util.spec_from_file_location("attack_champion", REPO_ROOT / "attack.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    attack = _load_attack_module()
    assert not hasattr(attack, "GENUINE_CARRIAGE_CHAINS"), (
        "attack.py still defines GENUINE_CARRIAGE_CHAINS; the single-message-fill "
        "pivot expects the genuine-carriage lineage to be gone."
    )
    assert hasattr(attack, "TEMPLATES") and hasattr(attack, "PAYLOAD"), (
        "attack.py is not the single-message-fill champion (missing TEMPLATES/PAYLOAD)."
    )
    print("genuine-carriage wiring: RETIRED (champion is the single-message fill)")


if __name__ == "__main__":
    main()
