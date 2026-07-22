#!/usr/bin/env python3
"""Run the complete frozen Phase 1 temporal-validation selection plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_phase1.artifacts import ArtifactError
from kuairec_phase1.gates import GateError
from kuairec_phase1.runner import run_selection
from kuairec_protocol.access import ProtocolAccessError


def main() -> int:
    try:
        output = run_selection(ROOT)
    except (ArtifactError, GateError, ProtocolAccessError) as exc:
        print(f"PHASE 1 ABORTED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "rows": len(output["result"]["rows"]),
                "selection_receipt": output["selection_receipt"],
                "temporal_final_accessed": False,
                "small_matrix_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
