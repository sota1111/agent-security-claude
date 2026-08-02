#!/usr/bin/env python3
"""Submission integrity gate: the kernel payload IS the current champion — SOT-2315.

The Kaggle code kernel (``kaggle/kernel/submit.py``) ships the champion by embedding
``attack.py`` verbatim as a base64 blob and re-materialising it at
``/kaggle/working/attack.py`` on the graded rerun. If that blob drifts from the repo
``attack.py``, the *submitted* strategy silently diverges from the *locally-evaluated*
champion.

That drift is exactly what happened before this gate existed: the kernel kept embedding
the **stale SOT-2082 floor champion** (SHA-256 ``e6d75d4b…`` — the ref 55162748 /
public-0.000 submission) while the promoted SOT-2296 genuine-carriage champion
(SHA-256 ``0391b1bc…``) was only ever wired into the repo ``attack.py``, never into the
kernel. The wired champion had therefore never actually been submitted, and every
resubmission of the same stale payload tripped the fingerprint gate as a duplicate —
a standing cause of the persistent public LB 0.000.

This gate enforces two invariants so the parent's resume run can submit safely:

1. **byte-identity** — the kernel's embedded payload decodes byte-for-byte (SHA-256)
   to the repo ``attack.py`` (same spirit as ``test_genuine_carriage_wiring.py``).
2. **new-artifact / fingerprint-diff** — the embedded payload differs from the last
   submitted artifact (``e6d75d4b…``, public 0.000), so the parent's fingerprint gate
   will see a genuinely new submission rather than skipping it as a duplicate.
"""

from __future__ import annotations

import base64
import hashlib
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ATTACK_PY = REPO_ROOT / "attack.py"
SUBMIT_PY = REPO_ROOT / "kaggle" / "kernel" / "submit.py"

# The last artifact actually submitted to Kaggle (docs/kpi.md: ref 55162748, public
# 0.000) — the stale SOT-2082 floor champion the kernel used to embed. The current
# champion must differ from this for the parent's fingerprint gate to treat it as new.
PREVIOUS_SUBMITTED_SHA256 = (
    "e6d75d4bdd00bb426b1f26836995e2ecfc90296482ce1ade83536f6edac2407c"
)

_B64_RE = re.compile(r'_ATTACK_PY_B64 = "([^"]+)"')


def _embedded_payload() -> bytes:
    match = _B64_RE.search(SUBMIT_PY.read_text(encoding="utf-8"))
    assert match, "could not locate _ATTACK_PY_B64 assignment in submit.py"
    return base64.b64decode(match.group(1))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class KernelPayloadIdentityTest(unittest.TestCase):
    def test_embedded_payload_byte_identical_to_attack(self) -> None:
        attack_bytes = ATTACK_PY.read_bytes()
        payload = _embedded_payload()
        self.assertEqual(
            _sha256(payload),
            _sha256(attack_bytes),
            "kaggle/kernel/submit.py embeds a payload that is NOT byte-identical to "
            "attack.py; re-embed the current champion (re-run the SOT-2315 re-embed).",
        )
        # Byte-level equality (not just the digest) for a defence-in-depth check.
        self.assertEqual(payload, attack_bytes)

    def test_embedded_payload_is_new_artifact_vs_previous_submission(self) -> None:
        payload_sha = _sha256(_embedded_payload())
        self.assertNotEqual(
            payload_sha,
            PREVIOUS_SUBMITTED_SHA256,
            "kernel still embeds the stale SOT-2082 floor champion (e6d75d4b…, the "
            "public-0.000 submission); the parent's fingerprint gate would skip it as "
            "a duplicate. Embed the promoted genuine-carriage champion.",
        )

    def test_current_champion_differs_from_previous_submission(self) -> None:
        # Independent of the kernel: the repo champion itself is a new artifact, so a
        # parent submission of it clears the fingerprint gate.
        self.assertNotEqual(_sha256(ATTACK_PY.read_bytes()), PREVIOUS_SUBMITTED_SHA256)


if __name__ == "__main__":
    unittest.main()
