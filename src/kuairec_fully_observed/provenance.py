"""Frozen Phase B2A input and identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE1_PROCESSED_MANIFEST_SHA256 = (
    "1461cc1838877efcebafeb9c6cb93847d94173d435c0221a24aab9b4e11724a5"
)
FROZEN_NORMAL_COUNT = 10_699
FROZEN_NORMAL_MEMBERSHIP_SHA256 = (
    "631a7c7cc93413f250f36f548feb720f8322050010e291afcc88338155f52c8e"
)
RAW_SOURCE_LOCATORS = {
    "big_matrix.csv": "KUAIREC_DATA_DIR/big_matrix.csv",
    "item_daily_features.csv": "KUAIREC_DATA_DIR/item_daily_features.csv",
    "kuairec_caption_category.csv": (
        "KUAIREC_DATA_DIR/kuairec_caption_category.csv"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any, *, label: str) -> str:
    digest = hashlib.sha256(f"{label}\n".encode())
    digest.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    )
    return digest.hexdigest()


def ordered_int_membership_sha256(
    values: Iterable[int] | np.ndarray, *, label: str
) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1 or not np.array_equal(array, np.unique(array)):
        raise ValueError(f"{label} membership must be sorted and unique")
    digest = hashlib.sha256(f"{label}\n".encode())
    for value in array:
        digest.update(f"{int(value)}\n".encode())
    return digest.hexdigest()


def membership_record(
    values: Iterable[int] | np.ndarray, *, label: str
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.int64)
    return {
        "count": int(len(array)),
        "sha256": ordered_int_membership_sha256(array, label=label),
        "hash_scheme": (
            f"sha256({label}\\n + sorted-unique-decimal-id\\n)"
        ),
    }


def normal_membership_record(values: np.ndarray) -> dict[str, Any]:
    record = membership_record(values, label="normal-video-membership-v1")
    if (
        record["count"] != FROZEN_NORMAL_COUNT
        or record["sha256"] != FROZEN_NORMAL_MEMBERSHIP_SHA256
    ):
        raise RuntimeError(
            "Frozen NORMAL membership changed: "
            f"count={record['count']} sha256={record['sha256']}"
        )
    return record


def verify_phase_b2a_inputs(
    *,
    data_dir: Path,
    artifact_dir: Path,
    required_raw_files: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify the frozen manifest before hashing or opening any raw input."""

    manifest_path = artifact_dir / "manifest.json"
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != PHASE1_PROCESSED_MANIFEST_SHA256:
        raise RuntimeError(
            "Processed manifest SHA256 mismatch: "
            f"actual={actual_manifest_sha256} "
            f"expected={PHASE1_PROCESSED_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_sources = manifest.get("fingerprint", {}).get(
        "source_file_sha256"
    )
    if not isinstance(expected_sources, dict):
        raise RuntimeError("Processed manifest has no raw source SHA mapping")

    unknown = sorted(set(required_raw_files) - set(RAW_SOURCE_LOCATORS))
    if unknown:
        raise ValueError(f"Unsupported Phase B2A raw inputs: {unknown}")
    records: dict[str, dict[str, Any]] = {}
    for source_name in required_raw_files:
        expected = expected_sources.get(source_name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(
                f"Processed manifest has no exact SHA for {source_name}"
            )
        path = data_dir / source_name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{source_name} SHA256 mismatch: "
                f"actual={actual} expected={expected}"
            )
        records[source_name] = {
            "source_locator": RAW_SOURCE_LOCATORS[source_name],
            "actual_sha256": actual,
            "expected_sha256": expected,
            "sha256_match": True,
        }
    return manifest, records
