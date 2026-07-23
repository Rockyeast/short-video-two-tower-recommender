#!/usr/bin/env python3
"""Compare final-refit and sealed Big-only numeric preprocessing paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.full_training import load_canonical_train_events
from kuairec_fully_observed.provenance import (
    PHASE1_PROCESSED_MANIFEST_SHA256,
    membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.torch_training import (
    final_refit_feature_identity,
    prepare_item_feature_store,
)
from scripts.run_phase_b3b_final_refit import _load_context


CHECKPOINT_SHA256 = (
    "5715a98d47c80f545afdce845c2a96940bc7ce3fec355e4a1156576d2fcab949"
)
NUMERIC_FIELDS = ("medians", "means", "stds")


def _float_hex(values: list[float]) -> list[str]:
    return [float(value).hex() for value in values]


def _path_record(
    *,
    name: str,
    store,
    observed_items: np.ndarray,
    observed_normal_items: np.ndarray,
    expected_numeric_sha256: str,
) -> dict[str, Any]:
    feature_identity = final_refit_feature_identity(store)
    preprocessing = store.preprocessing
    return {
        "name": name,
        "observed_items": membership_record(
            np.unique(observed_items),
            label="phase-b3b-r3-train-observed-items-v1",
        ),
        "observed_normal_items": membership_record(
            np.unique(observed_normal_items),
            label="phase-b3b-r3-train-observed-normal-items-v1",
        ),
        "model_item_universe": membership_record(
            store.item_ids,
            label="phase-b3b-refit-item-universe-v1",
        ),
        "category_vocab_count": feature_identity["category_vocab_count"],
        "category_vocab_sha256": feature_identity[
            "category_vocab_sha256"
        ],
        "upload_type_vocab_count": feature_identity[
            "upload_type_vocab_count"
        ],
        "upload_type_vocab_sha256": feature_identity[
            "upload_type_vocab_sha256"
        ],
        "preprocessing": preprocessing,
        "preprocessing_float_hex": {
            field: _float_hex(preprocessing[field])
            for field in NUMERIC_FIELDS
        },
        "missing_value_counts": preprocessing["missing_value_counts"],
        "numeric_preprocessing_sha256": feature_identity[
            "numeric_preprocessing_sha256"
        ],
        "checkpoint_expected_numeric_preprocessing_sha256": (
            expected_numeric_sha256
        ),
        "matches_checkpoint": (
            feature_identity["numeric_preprocessing_sha256"]
            == expected_numeric_sha256
        ),
    }


def _field_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    left_payload = left["preprocessing"]
    right_payload = right["preprocessing"]
    for key in sorted(set(left_payload) | set(right_payload)):
        if left_payload.get(key) == right_payload.get(key):
            continue
        if key in NUMERIC_FIELDS:
            differences[key] = [
                {
                    "index": index,
                    "training_value": left_payload[key][index],
                    "sealed_value": right_payload[key][index],
                    "training_hex": left[
                        "preprocessing_float_hex"
                    ][key][index],
                    "sealed_hex": right[
                        "preprocessing_float_hex"
                    ][key][index],
                }
                for index in range(len(left_payload[key]))
                if left_payload[key][index] != right_payload[key][index]
            ]
        else:
            differences[key] = {
                "training_value": left_payload.get(key),
                "sealed_value": right_payload.get(key),
            }
    return differences


def run(
    *,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    checkpoint_path: Path,
    process_index: int,
) -> dict[str, Any]:
    checkpoint_actual_sha256 = sha256_file(checkpoint_path)
    if checkpoint_actual_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen Two-Tower checkpoint SHA256 mismatch")
    manifest, raw_sources = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    identity = payload["identity"]
    expected_numeric_sha256 = identity["feature_identity"][
        "numeric_preprocessing_sha256"
    ]
    ordered_items = np.asarray(
        payload["ordered_item_ids"], dtype=np.int64
    )
    static = load_static_item_features(data_dir)
    frame = static.frame.set_index("video_id").reindex(ordered_items)
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=ordered_items,
        expected_model_id=identity["caption_identity"]["model_id"],
        expected_revision=identity["caption_identity"]["resolved_revision"],
        expected_source_sha256=raw_sources[
            "kuairec_caption_category.csv"
        ]["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            ordered_items, frame["caption_text"].astype(str).tolist()
        ),
    )

    training_context = _load_context(
        artifact_dir=artifact_dir,
        normal_item_ids=static.normal_item_ids,
    )
    training_observed = np.unique(
        training_context["item_ids"][training_context["fit"]]
    )
    training_normal = np.intersect1d(
        training_observed,
        static.normal_item_ids,
        assume_unique=True,
    )
    training_universe = np.union1d(
        training_observed, np.unique(static.normal_item_ids)
    ).astype(np.int64)
    if not np.array_equal(training_universe, ordered_items):
        raise RuntimeError(
            "Training-path model item universe differs from checkpoint"
        )
    training_store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=training_universe,
        train_observed_item_ids=training_observed,
        train_observed_normal_item_ids=training_normal,
    )

    with np.load(artifact_dir / "catalog.npz") as catalog:
        validation_end = float(catalog["validation_end"][0])
    canonical = load_canonical_train_events(
        data_dir, train_end=validation_end
    )
    sealed_observed = np.unique(
        canonical["video_id"].to_numpy(np.int64)
    )
    sealed_normal = np.intersect1d(
        sealed_observed,
        static.normal_item_ids,
        assume_unique=True,
    )
    sealed_store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=ordered_items,
        train_observed_item_ids=sealed_observed,
        train_observed_normal_item_ids=sealed_normal,
    )

    training_record = _path_record(
        name="final_refit_training_path",
        store=training_store,
        observed_items=training_observed,
        observed_normal_items=training_normal,
        expected_numeric_sha256=expected_numeric_sha256,
    )
    sealed_record = _path_record(
        name="current_sealed_path",
        store=sealed_store,
        observed_items=sealed_observed,
        observed_normal_items=sealed_normal,
        expected_numeric_sha256=expected_numeric_sha256,
    )
    return {
        "phase": "phase-b3b-r3-numeric-preprocessing-diagnostic",
        "process_index": process_index,
        "small_matrix_accessed": False,
        "checkpoint": {
            "actual_sha256": checkpoint_actual_sha256,
            "expected_sha256": CHECKPOINT_SHA256,
            "expected_numeric_preprocessing_sha256": (
                expected_numeric_sha256
            ),
        },
        "processed_manifest": {
            "actual_sha256": sha256_file(
                artifact_dir / "manifest.json"
            ),
            "expected_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
            "manifest_config_hash": manifest.get("config_hash"),
        },
        "raw_sources": raw_sources,
        "paths": {
            "training": training_record,
            "sealed": sealed_record,
        },
        "membership_equal": {
            "observed_items": (
                training_record["observed_items"]
                == sealed_record["observed_items"]
            ),
            "observed_normal_items": (
                training_record["observed_normal_items"]
                == sealed_record["observed_normal_items"]
            ),
            "model_item_universe": (
                training_record["model_item_universe"]
                == sealed_record["model_item_universe"]
            ),
        },
        "preprocessing_field_differences": _field_differences(
            training_record, sealed_record
        ),
    }


def main() -> None:
    raise SystemExit(
        "Use scripts/modal_phase_b3b_r3_numeric_diagnostic.py"
    )


if __name__ == "__main__":
    main()
