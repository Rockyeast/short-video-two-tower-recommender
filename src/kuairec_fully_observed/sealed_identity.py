"""Fail-closed identities required before the sealed Small evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import sha256_file


SMALL_MATRIX_SIZE_BYTES = 406_155_844
SMALL_MATRIX_SHA256 = (
    "6b601cd38b2600d8734b4aede309ad73d1e201fdc4bd76d4bc7d2534793d7d15"
)


def verify_file_identity(
    path: Path, *, expected_size_bytes: int, expected_sha256: str, label: str
) -> dict[str, Any]:
    """Verify size before hashing, then require the frozen payload SHA."""

    if not path.is_file():
        raise RuntimeError(f"{label} is missing")
    actual_size = path.stat().st_size
    if actual_size != int(expected_size_bytes):
        raise RuntimeError(
            f"{label} size mismatch: actual={actual_size} "
            f"expected={expected_size_bytes}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: actual={actual_sha} "
            f"expected={expected_sha256}"
        )
    return {
        "size_bytes": actual_size,
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha256,
        "match": True,
    }


def verify_frozen_small_source(
    *, small_path: Path, split_manifest_path: Path
) -> dict[str, Any]:
    """Bind the one-time input to the already-frozen split manifest."""

    manifest = json.loads(split_manifest_path.read_text())
    try:
        record = manifest["dataset"]["source_files"]["small_matrix.csv"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "split_manifest.json has no frozen small_matrix.csv record"
        ) from exc
    expected_record = {
        "relative_path": "KuaiRec 2.0/data/small_matrix.csv",
        "size_bytes": SMALL_MATRIX_SIZE_BYTES,
        "sha256": SMALL_MATRIX_SHA256,
    }
    if record != expected_record:
        raise RuntimeError(
            "split_manifest.json small_matrix.csv identity is not frozen"
        )
    return {
        "source_locator": "KUAIREC_DATA_DIR/small_matrix.csv",
        **verify_file_identity(
            small_path,
            expected_size_bytes=SMALL_MATRIX_SIZE_BYTES,
            expected_sha256=SMALL_MATRIX_SHA256,
            label="small_matrix.csv",
        ),
    }


def _final_refit_record(report: dict[str, Any]) -> dict[str, Any]:
    try:
        return report["remote"]["refit"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Final-refit report structure is invalid") from exc


def verify_final_refit_artifacts(
    *,
    final_refit_report_path: Path,
    popularity_path: Path,
    bpr_checkpoint_path: Path,
    two_tower_checkpoint_path: Path,
) -> dict[str, Any]:
    """Bind all scoring artifacts to the committed final-refit report."""

    report = json.loads(final_refit_report_path.read_text())
    refit = _final_refit_record(report)
    expected_claims = {
        "fit_context": "canonical_big_train_plus_validation",
        "recipe_frozen_before_small": True,
        "selection_performed": False,
    }
    for name, expected in expected_claims.items():
        if refit.get(name) != expected:
            raise RuntimeError(
                f"Final-refit report claim mismatch: {name}"
            )
    records = refit.get("refit")
    if not isinstance(records, dict):
        raise RuntimeError("Final-refit method records are missing")
    if records.get("bpr", {}).get("epochs") != 20:
        raise RuntimeError("Final-refit BPR epoch contract changed")
    if records.get("two_tower", {}).get("epochs") != 1:
        raise RuntimeError("Final-refit Two-Tower epoch contract changed")
    expected = {
        "global_popularity": records.get("global_popularity", {}).get(
            "artifact_sha256"
        ),
        "bpr_epoch_20": records.get("bpr", {}).get("checkpoint_sha256"),
        "two_tower_epoch_1": records.get("two_tower", {}).get(
            "checkpoint_sha256"
        ),
    }
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in expected.values()
    ):
        raise RuntimeError("Final-refit artifact SHA record is invalid")
    paths = {
        "global_popularity": popularity_path,
        "bpr_epoch_20": bpr_checkpoint_path,
        "two_tower_epoch_1": two_tower_checkpoint_path,
    }
    verified = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Final-refit artifact is missing: {name}")
        actual_sha = sha256_file(path)
        if actual_sha != expected[name]:
            raise RuntimeError(
                f"Final-refit artifact SHA256 mismatch: {name} "
                f"actual={actual_sha} expected={expected[name]}"
            )
        verified[name] = {
            "actual_sha256": actual_sha,
            "expected_sha256": expected[name],
            "match": True,
        }
    return {
        "fit_context": refit["fit_context"],
        "recipe_frozen_before_small": True,
        "selection_performed": False,
        "bpr_epochs": 20,
        "two_tower_epochs": 1,
        "artifacts": verified,
    }
