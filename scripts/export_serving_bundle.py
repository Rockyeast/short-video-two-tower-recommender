#!/usr/bin/env python3
"""Export frozen final-refit artifacts into one recommendation serving bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kuairec_fully_observed import (
    load_static_item_features,
    verify_final_refit_artifacts,
)
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.numeric_sidecar import (
    load_final_refit_numeric_sidecar,
)
from kuairec_fully_observed.provenance import (
    membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.serving_bundle import write_serving_bundle
from kuairec_fully_observed.torch_training import (
    encode_query_users_from_precomputed,
    final_refit_feature_identity,
    load_final_refit_checkpoint_compatible,
    preencode_item_universe,
    prepare_final_refit_inference_feature_store,
    resolve_concrete_device,
)
from kuairec_fully_observed.training import _weights_from_arrays
from scripts import audit_phase0


def _snapshot_histories(
    *,
    data_dir: Path,
    user_ids: np.ndarray,
    validation_end: float,
    max_history: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    selected = set(int(user) for user in user_ids)
    histories: dict[int, np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    for user_id, raw in audit_phase0.iter_user_frames(
        data_dir / "big_matrix.csv", audit_phase0.EVENT_COLUMNS
    ):
        if user_id not in selected:
            continue
        canonical, _, _, _ = audit_phase0.canonicalize_behavior_events(raw)
        history = canonical.loc[
            canonical["timestamp"] < validation_end
        ].tail(max_history)
        histories[user_id] = history["video_id"].to_numpy(np.int64)
        weights[user_id] = _weights_from_arrays(
            history["watch_ratio"].to_numpy(np.float64),
            history["play_duration"].to_numpy(np.float64),
            history["video_duration"].to_numpy(np.float64),
            quick_skip_mask=history["_is_quick_skip"].to_numpy(bool),
        )
    missing = selected - set(histories)
    if missing:
        raise RuntimeError(
            f"Final-refit users missing Big histories: {len(missing)}"
        )
    return (
        tuple(histories[int(user)] for user in user_ids),
        tuple(weights[int(user)] for user in user_ids),
    )


def _source_identity(
    *,
    refit_identity: dict[str, Any],
    raw_sources: dict[str, dict[str, Any]],
    processed_manifest: Path,
    caption_cache: Path,
    numeric_sidecar: Path,
    two_tower_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_refit": refit_identity,
        "raw_inputs": {
            name: record["actual_sha256"]
            for name, record in sorted(raw_sources.items())
        },
        "processed_manifest_sha256": sha256_file(processed_manifest),
        "caption_cache_sha256": sha256_file(caption_cache),
        "numeric_sidecar_sha256": sha256_file(numeric_sidecar),
        "two_tower_identity_sha256": two_tower_checkpoint["identity_sha256"],
        "small_matrix_accessed": False,
        "temporal_final_accessed": False,
    }


def export(
    *,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    popularity_path: Path,
    bpr_checkpoint_path: Path,
    two_tower_checkpoint_path: Path,
    final_refit_report_path: Path,
    numeric_sidecar_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    device: str,
) -> dict[str, Any]:
    refit_identity = verify_final_refit_artifacts(
        final_refit_report_path=final_refit_report_path,
        popularity_path=popularity_path,
        bpr_checkpoint_path=bpr_checkpoint_path,
        two_tower_checkpoint_path=two_tower_checkpoint_path,
    )
    _, raw_sources = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    with np.load(artifact_dir / "catalog.npz") as catalog_payload:
        validation_end = float(catalog_payload["validation_end"][0])
        video_ids = catalog_payload["video_ids"].astype(
            np.int64, copy=True
        )
    with np.load(artifact_dir / "events_train_validation.npz") as events:
        event_items = events["item"].astype(np.int64, copy=True)
        event_times = events["timestamp"].astype(np.float64, copy=True)
    observed = np.unique(
        video_ids[event_items[event_times < validation_end]]
    ).astype(np.int64)
    static = load_static_item_features(data_dir)
    observed_normal = np.intersect1d(
        observed, static.normal_item_ids, assume_unique=True
    )

    checkpoint_payload = torch.load(
        two_tower_checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_identity = checkpoint_payload["identity"]
    ordered_items = np.asarray(
        checkpoint_payload["ordered_item_ids"], dtype=np.int64
    )
    touched_items = np.asarray(
        checkpoint_payload["touched_item_ids"], dtype=np.int64
    )
    touched_users = np.asarray(
        checkpoint_payload["touched_user_ids"], dtype=np.int64
    )
    frame = static.frame.set_index("video_id").reindex(ordered_items)
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=ordered_items,
        expected_model_id=checkpoint_identity["caption_identity"]["model_id"],
        expected_revision=checkpoint_identity["caption_identity"][
            "resolved_revision"
        ],
        expected_source_sha256=raw_sources[
            "kuairec_caption_category.csv"
        ]["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            ordered_items, frame["caption_text"].astype(str).tolist()
        ),
    )
    memberships = {
        "train_observed_items": membership_record(
            observed, label="phase-b3b-r3-train-observed-items-v1"
        ),
        "train_observed_normal_items": membership_record(
            observed_normal,
            label="phase-b3b-r3-train-observed-normal-items-v1",
        ),
        "model_item_universe": membership_record(
            ordered_items, label="phase-b3b-refit-item-universe-v1"
        ),
    }
    sidecar = load_final_refit_numeric_sidecar(
        numeric_sidecar_path,
        checkpoint_sha256=refit_identity["artifacts"][
            "two_tower_epoch_1"
        ]["actual_sha256"],
        checkpoint_expected_numeric_sha256=checkpoint_identity[
            "feature_identity"
        ]["numeric_preprocessing_sha256"],
        processed_manifest_sha256=sha256_file(
            artifact_dir / "manifest.json"
        ),
        raw_input_sha256={
            name: record["actual_sha256"]
            for name, record in raw_sources.items()
        },
        memberships=memberships,
    )
    store = prepare_final_refit_inference_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=ordered_items,
        train_observed_item_ids=observed,
        train_observed_normal_item_ids=observed_normal,
        frozen_preprocessing=sidecar["preprocessing"],
    )
    feature_identity = final_refit_feature_identity(store)
    target_device = resolve_concrete_device(device)
    model, loaded_checkpoint = load_final_refit_checkpoint_compatible(
        two_tower_checkpoint_path,
        device=target_device,
        expected_identity=checkpoint_identity,
        reconstructed_feature_identity=feature_identity,
        final_refit_artifact_verified=True,
    )
    model.eval()
    item_vectors = preencode_item_universe(
        model=model,
        store=store,
        touched_item_ids=set(int(item) for item in touched_items),
        device=target_device,
        batch_size=1024,
    )
    histories, history_weights = _snapshot_histories(
        data_dir=data_dir,
        user_ids=touched_users,
        validation_end=validation_end,
        max_history=50,
    )
    user_vectors = encode_query_users_from_precomputed(
        model=model,
        store=store,
        precomputed_item_vectors=item_vectors,
        user_ids=touched_users,
        histories=histories,
        history_weights=history_weights,
        user_positions={
            int(user): index + 1
            for index, user in enumerate(
                loaded_checkpoint["ordered_user_ids"]
            )
        },
        touched_user_ids=set(int(user) for user in touched_users),
        device=target_device,
        batch_size=128,
    ).cpu().numpy()
    catalog = np.intersect1d(
        static.normal_item_ids, ordered_items, assume_unique=True
    ).astype(np.int64)
    catalog_positions = np.asarray(
        [store.positions[int(item)] for item in catalog], dtype=np.int64
    )
    serving_item_vectors = item_vectors[
        torch.as_tensor(
            catalog_positions, dtype=torch.long, device=target_device
        )
    ].cpu().numpy()

    popularity = {
        int(item): float(score)
        for item, score in json.loads(popularity_path.read_text()).items()
    }
    popularity_ids = np.asarray(sorted(popularity), dtype=np.int64)
    popularity_scores = np.asarray(
        [popularity[int(item)] for item in popularity_ids],
        dtype=np.float64,
    )
    with np.load(bpr_checkpoint_path) as bpr:
        if int(bpr["epoch"][0]) != 20:
            raise RuntimeError("Serving export requires final BPR epoch 20")
        arrays = {
            "catalog": catalog,
            "popularity_item_ids": popularity_ids,
            "popularity_scores": popularity_scores,
            "bpr_user_ids": bpr["user_ids"].astype(np.int64, copy=True),
            "bpr_item_ids": bpr["item_ids"].astype(np.int64, copy=True),
            "bpr_user_factors": bpr["user_factors"].astype(
                np.float32, copy=True
            ),
            "bpr_item_factors": bpr["item_factors"].astype(
                np.float32, copy=True
            ),
            "two_tower_user_ids": touched_users,
            "two_tower_user_vectors": user_vectors.astype(
                np.float32, copy=False
            ),
            "two_tower_item_ids": catalog,
            "two_tower_item_vectors": serving_item_vectors.astype(
                np.float32, copy=False
            ),
        }
    metadata = write_serving_bundle(
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        arrays=arrays,
        source_identity=_source_identity(
            refit_identity=refit_identity,
            raw_sources=raw_sources,
            processed_manifest=artifact_dir / "manifest.json",
            caption_cache=caption_cache_path,
            numeric_sidecar=numeric_sidecar_path,
            two_tower_checkpoint=checkpoint_payload,
        ),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument("--caption-metadata", type=Path, required=True)
    parser.add_argument("--popularity", type=Path, required=True)
    parser.add_argument("--bpr-checkpoint", type=Path, required=True)
    parser.add_argument("--two-tower-checkpoint", type=Path, required=True)
    parser.add_argument("--final-refit-report", type=Path, required=True)
    parser.add_argument(
        "--numeric-sidecar",
        type=Path,
        default=Path(
            "manifests/phase_b3b_final_numeric_preprocessing.json"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/serving/serving_bundle_v1.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/serving/serving_bundle_v1.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metadata = export(
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        popularity_path=args.popularity.resolve(),
        bpr_checkpoint_path=args.bpr_checkpoint.resolve(),
        two_tower_checkpoint_path=args.two_tower_checkpoint.resolve(),
        final_refit_report_path=args.final_refit_report.resolve(),
        numeric_sidecar_path=args.numeric_sidecar.resolve(),
        bundle_path=args.bundle.resolve(),
        metadata_path=args.metadata.resolve(),
        device=args.device,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
