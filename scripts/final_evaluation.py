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
    final_receipt_dir,
    validate_final_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--experiment-bundle", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        request = validate_final_request(
            confirmation=args.confirm,
            repo_root=ROOT,
            experiment_bundle_path=args.experiment_bundle,
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
                "experiment_bundle_sha256": request.experiment_bundle_sha256,
                "methods": list(request.method_names),
                "receipt_dir": str(
                    final_receipt_dir(ROOT, request.split_manifest_sha256)
                ),
                "final_executed": False,
            },
            sort_keys=True,
        )
    )
    print(
        "FINAL EVALUATION BLOCKED: no reviewed evaluator is attached; "
        "no final data was opened and no receipt was written.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
