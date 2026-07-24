"""Compact, identity-checked artifacts consumed by the recommendation CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


SERVING_BUNDLE_KEYS = {
    "catalog",
    "popularity_item_ids",
    "popularity_scores",
    "bpr_user_ids",
    "bpr_item_ids",
    "bpr_user_factors",
    "bpr_item_factors",
    "two_tower_user_ids",
    "two_tower_user_vectors",
    "two_tower_user_id_embeddings",
    "two_tower_item_ids",
    "two_tower_item_vectors",
    "two_tower_mlp_input_weight",
    "two_tower_mlp_input_bias",
    "two_tower_mlp_output_weight",
    "two_tower_mlp_output_bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int_membership_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    if array.ndim != 1 or not np.array_equal(array, np.unique(array)):
        raise ValueError("Membership values must be sorted and unique")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(arrays) != SERVING_BUNDLE_KEYS:
        raise ValueError("Serving bundle fields differ from schema v2")
    values = {name: np.asarray(value) for name, value in arrays.items()}
    for name in (
        "catalog",
        "popularity_item_ids",
        "bpr_user_ids",
        "bpr_item_ids",
        "two_tower_user_ids",
        "two_tower_item_ids",
    ):
        array = values[name]
        if (
            array.dtype.kind not in "iu"
            or array.ndim != 1
            or not np.array_equal(array, np.unique(array))
        ):
            raise ValueError(f"{name} must be sorted, unique integer IDs")
    aligned = (
        ("popularity_item_ids", "popularity_scores"),
        ("bpr_user_ids", "bpr_user_factors"),
        ("bpr_item_ids", "bpr_item_factors"),
        ("two_tower_user_ids", "two_tower_user_vectors"),
        ("two_tower_user_ids", "two_tower_user_id_embeddings"),
        ("two_tower_item_ids", "two_tower_item_vectors"),
    )
    for ids_name, values_name in aligned:
        if len(values[ids_name]) != len(values[values_name]):
            raise ValueError(f"{ids_name} and {values_name} differ in length")
    float_names = {
        "popularity_scores",
        "bpr_user_factors",
        "bpr_item_factors",
        "two_tower_user_vectors",
        "two_tower_item_vectors",
        "two_tower_user_id_embeddings",
        "two_tower_mlp_input_weight",
        "two_tower_mlp_input_bias",
        "two_tower_mlp_output_weight",
        "two_tower_mlp_output_bias",
    }
    if any(not np.isfinite(values[name]).all() for name in float_names):
        raise ValueError("Serving bundle contains non-finite values")
    if values["popularity_scores"].ndim != 1:
        raise ValueError("Popularity scores must be one-dimensional")
    for users_name, items_name, label in (
        ("bpr_user_factors", "bpr_item_factors", "bpr"),
        ("two_tower_user_vectors", "two_tower_item_vectors", "two_tower"),
    ):
        users = values[users_name]
        items = values[items_name]
        if (
            users.ndim != 2
            or items.ndim != 2
            or users.shape[1] != items.shape[1]
        ):
            raise ValueError(
                f"{label} vectors need matching rank-2 dimensions"
            )
    if (
        values["two_tower_user_id_embeddings"].ndim != 2
        or values["two_tower_mlp_input_weight"].ndim != 2
        or values["two_tower_mlp_input_bias"].ndim != 1
        or values["two_tower_mlp_output_weight"].ndim != 2
        or values["two_tower_mlp_output_bias"].ndim != 1
    ):
        raise ValueError("Dynamic user-tower parameters must be rank 1/2")
    item_dim = values["two_tower_item_vectors"].shape[1]
    user_id_dim = values["two_tower_user_id_embeddings"].shape[1]
    hidden_dim = len(values["two_tower_mlp_input_bias"])
    if (
        values["two_tower_mlp_input_weight"].shape
        != (hidden_dim, user_id_dim + item_dim)
        or values["two_tower_mlp_output_weight"].shape
        != (item_dim, hidden_dim)
        or values["two_tower_mlp_output_bias"].shape != (item_dim,)
    ):
        raise ValueError("Dynamic user-tower parameter shapes differ")
    catalog = values["catalog"]
    if not set(catalog).issubset(set(values["two_tower_item_ids"])):
        raise ValueError("Two-Tower serving vectors do not cover the catalog")
    return {
        "catalog_count": int(len(catalog)),
        "catalog_sha256": int_membership_sha256(catalog),
        "popularity_item_count": int(len(values["popularity_item_ids"])),
        "bpr_user_count": int(len(values["bpr_user_ids"])),
        "bpr_item_count": int(len(values["bpr_item_ids"])),
        "bpr_dimension": int(values["bpr_item_factors"].shape[1]),
        "two_tower_user_count": int(len(values["two_tower_user_ids"])),
        "two_tower_item_count": int(len(values["two_tower_item_ids"])),
        "two_tower_dimension": int(values["two_tower_item_vectors"].shape[1]),
        "two_tower_user_id_dimension": int(user_id_dim),
        "two_tower_user_hidden_dimension": int(hidden_dim),
    }


def write_serving_bundle(
    *,
    bundle_path: Path,
    metadata_path: Path,
    arrays: Mapping[str, np.ndarray],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one compressed payload and a small identity sidecar."""

    summary = _validate_arrays(arrays)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_path,
        **{
            name: np.asarray(arrays[name])
            for name in sorted(SERVING_BUNDLE_KEYS)
        },
    )
    metadata = {
        "schema_version": 2,
        "bundle_locator": "artifacts/serving/serving_bundle_v1.npz",
        "bundle_sha256": sha256_file(bundle_path),
        "fit_context": "canonical_big_train_plus_validation",
        "user_vector_semantics": "dynamic_weighted_history_user_tower",
        "snapshot_user_vectors_included_for_parity": True,
        "source_identity": dict(source_identity),
        **summary,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def load_serving_bundle(
    *, bundle_path: Path, metadata_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fail closed before returning any arrays to the recommendation engine."""

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 2:
        raise RuntimeError("Unsupported serving bundle metadata schema")
    actual_sha = sha256_file(bundle_path)
    if metadata.get("bundle_sha256") != actual_sha:
        raise RuntimeError("Serving bundle SHA256 mismatch")
    with np.load(bundle_path, allow_pickle=False) as payload:
        arrays = {
            name: payload[name].copy()
            for name in payload.files
        }
    summary = _validate_arrays(arrays)
    for name, value in summary.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"Serving bundle metadata mismatch: {name}")
    if metadata.get("fit_context") != "canonical_big_train_plus_validation":
        raise RuntimeError("Serving bundle fit context changed")
    return arrays, metadata
