#!/usr/bin/env python3
"""Fail-closed exactly-once temporal-final entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_phase1.final import FINAL_CONFIRMATION, run_registered_final_once
from kuairec_phase1.gates import GateError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument(
        "--experiment-bundle",
        "--final-method-bundle",
        dest="final_method_bundle",
        type=Path,
        required=True,
        help="Frozen final_method_bundle selected on validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_registered_final_once(
            confirmation=args.confirm,
            repo_root=ROOT,
            bundle_path=args.final_method_bundle,
        )
    except GateError as exc:
        print(f"ACCESS DENIED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
