#!/usr/bin/env python3
"""Internal isolated worker for one resumable ERRATUM-001 selection row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_phase1.artifacts import ArtifactError
from kuairec_phase1.erratum import run_segment_membership_erratum_row
from kuairec_phase1.gates import GateError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", required=True, type=int)
    parser.add_argument("--cache-key", required=True)
    args = parser.parse_args()
    try:
        result = run_segment_membership_erratum_row(
            ROOT, row_index=args.row_index, expected_cache_key=args.cache_key
        )
    except (ArtifactError, GateError) as exc:
        print(f"ERRATUM-001 ROW ABORTED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
