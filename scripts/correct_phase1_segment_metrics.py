#!/usr/bin/env python3
"""Run ERRATUM-001 segment-only correction without opening holdouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_phase1.artifacts import ArtifactError
from kuairec_phase1.erratum import run_segment_membership_erratum
from kuairec_phase1.gates import GateError


def main() -> int:
    try:
        result = run_segment_membership_erratum(ROOT)
    except (ArtifactError, GateError) as exc:
        print(f"ERRATUM-001 ABORTED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
