"""Strict final-refit numeric preprocessing sidecar validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256


NUMERIC_PREPROCESSING_LABEL = "phase-b3b-numeric-preprocessing-v1"


def load_final_refit_numeric_sidecar(
    path: Path,
    *,
    checkpoint_sha256: str,
    checkpoint_expected_numeric_sha256: str,
    processed_manifest_sha256: str,
    raw_input_sha256: dict[str, str],
    memberships: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError("Numeric sidecar schema changed")
    checkpoint = payload.get("checkpoint", {})
    if checkpoint != {
        "sha256": checkpoint_sha256,
        "expected_numeric_preprocessing_sha256": (
            checkpoint_expected_numeric_sha256
        ),
    }:
        raise RuntimeError("Numeric sidecar checkpoint identity mismatch")
    if payload.get("processed_manifest") != {
        "sha256": processed_manifest_sha256
    }:
        raise RuntimeError("Numeric sidecar processed manifest mismatch")
    if payload.get("raw_inputs") != raw_input_sha256:
        raise RuntimeError("Numeric sidecar raw input identity mismatch")
    if payload.get("memberships") != memberships:
        raise RuntimeError("Numeric sidecar membership identity mismatch")

    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise RuntimeError("Numeric sidecar preprocessing payload missing")
    actual_sha256 = canonical_json_sha256(
        preprocessing, label=NUMERIC_PREPROCESSING_LABEL
    )
    if (
        actual_sha256 != checkpoint_expected_numeric_sha256
        or payload.get("numeric_preprocessing_sha256") != actual_sha256
    ):
        raise RuntimeError("Numeric sidecar preprocessing SHA mismatch")
    expected_hex = payload.get("preprocessing_float_hex")
    actual_hex = {
        field: [float(value).hex() for value in preprocessing[field]]
        for field in ("medians", "means", "stds")
    }
    if expected_hex != actual_hex:
        raise RuntimeError("Numeric sidecar float payload identity mismatch")
    return payload
