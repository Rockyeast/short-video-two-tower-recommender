#!/usr/bin/env python3
"""Fail-closed explicit entrypoint reserved for one future final evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_protocol.access import (
    FINAL_CONFIRMATION,
    ProtocolAccessError,
    validate_final_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/split_manifest.json")
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "manifests/FINAL_HOLDOUT_LOCKED.json"
    )
    parser.add_argument("--receipt-dir", type=Path, default=ROOT / "receipts/final")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        request = validate_final_request(
            confirmation=args.confirm,
            method_config_path=args.method_config,
            artifact_manifest_path=args.artifact_manifest,
            manifest_path=args.manifest,
            lock_path=args.lock,
            seed=args.seed,
        )
    except ProtocolAccessError as exc:
        print(f"ACCESS DENIED: {exc}", file=sys.stderr)
        return 2

    # Deliberately do not call run_final_once here.  No final evaluator has
    # been reviewed or attached, so this entrypoint must stop before opening
    # protected labels or consuming the one-shot receipt.
    print(
        json.dumps(
            {
                "access": "explicit_final_protocol_verified",
                "fit_splits": list(request.fit_splits),
                "evaluation_split": request.evaluation_split,
                "manifest_sha256": request.split_manifest_sha256,
                "receipt_dir": str(args.receipt_dir),
                "final_executed": False,
            },
            sort_keys=True,
        )
    )
    print(
        "FINAL EVALUATION BLOCKED: protocol-v2 has no reviewed evaluator; "
        "no final data was opened and no receipt was written.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
