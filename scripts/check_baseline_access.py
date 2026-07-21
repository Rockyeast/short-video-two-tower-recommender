#!/usr/bin/env python3
"""Check baseline selection scope without running a baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_protocol.access import (
    ProtocolAccessError,
    authorize_baseline_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-split", action="append", default=None)
    parser.add_argument("--evaluation-split", default="validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fit_splits = tuple(args.fit_split or ["train"])
    try:
        # This full verifier runs before any future baseline loader. It first
        # authenticates the generator, then dynamically loads its independent
        # target/catalog hash rebuilder.
        verification = authorize_baseline_selection(
            fit_splits=fit_splits,
            evaluation_split=args.evaluation_split,
            repo_root=ROOT,
        )
    except ProtocolAccessError as exc:
        print(f"ACCESS DENIED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "access": "selection_only",
                "fit_splits": list(fit_splits),
                "evaluation_split": args.evaluation_split,
                "manifest_sha256": verification.manifest_sha256,
                "baseline_executed": False,
                "message": "Protocol check only; no baseline is implemented or executed.",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
