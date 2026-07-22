#!/usr/bin/env python3
"""Rebuild and verify the frozen KuaiRec protocol bundle; run no experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_protocol.access import (  # noqa: E402
    ProtocolAccessError,
    verify_protocol_bundle,
)


def main() -> int:
    try:
        verification = verify_protocol_bundle(ROOT)
    except ProtocolAccessError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verified": True,
                "protocol_revision": verification.protocol_revision,
                "manifest_sha256": verification.manifest_sha256,
                "canonical_target_sha256": dict(
                    verification.canonical_target_sha256
                ),
                "candidate_membership_sha256": dict(
                    verification.candidate_membership_sha256
                ),
                "experiment_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
