#!/usr/bin/env python3
"""Run the bounded Phase B4A Exact/FAISS scalability benchmark."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.data import _history_weights
from kuairec_fully_observed.faiss_benchmark import (
    BENCHMARK_SEED,
    HNSW_CONFIG,
    QUERY_LIMIT,
    SCALE_SPECS,
    THREAD_COUNT,
    TOP_K,
    VECTOR_DIMENSION,
    benchmark_catalog,
    extend_catalog,
    require_unit_rows,
    runtime_identity,
)
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.full_training import load_canonical_train_events
from kuairec_fully_observed.numeric_sidecar import (
    load_final_refit_numeric_sidecar,
)
from kuairec_fully_observed.provenance import (
    membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.sealed_identity import (
    verify_final_refit_artifacts,
)
from kuairec_fully_observed.torch_training import (
    encode_query_users_from_precomputed,
    final_refit_feature_identity,
    load_final_refit_checkpoint_compatible,
    preencode_item_universe,
    prepare_final_refit_inference_feature_store,
    resolve_concrete_device,
)


EXPECTED_REAL_ITEM_COUNT = 10_725


def _stable_user_key(user_id: int) -> tuple[bytes, int]:
    digest = hashlib.sha256(
        f"phase-b4a-query-selection-v1:{BENCHMARK_SEED}:{user_id}".encode()
    ).digest()
    return digest, user_id


def _ordered_query_user_sha256(user_ids: np.ndarray) -> str:
    ordered = np.asarray(user_ids, dtype="<i8")
    if ordered.ndim != 1 or len(np.unique(ordered)) != len(ordered):
        raise ValueError("Phase B4A query users must be one-dimensional and unique")
    return hashlib.sha256(ordered.tobytes(order="C")).hexdigest()


def _load_real_vectors(
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
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    with np.load(artifact_dir / "catalog.npz") as catalog:
        validation_end = float(catalog["validation_end"][0])
    big_context = load_canonical_train_events(
        data_dir, train_end=validation_end
    )
    static = load_static_item_features(data_dir)
    checkpoint_payload = torch.load(
        two_tower_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_identity = checkpoint_payload["identity"]
    ordered_items = np.asarray(
        checkpoint_payload["ordered_item_ids"], dtype=np.int64
    )
    if len(ordered_items) != EXPECTED_REAL_ITEM_COUNT:
        raise RuntimeError("Final Two-Tower item universe count changed")
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
    observed = np.unique(big_context["video_id"].to_numpy(np.int64))
    observed_normal = np.intersect1d(
        observed, static.normal_item_ids, assume_unique=True
    )
    sidecar_memberships = {
        "train_observed_items": membership_record(
            observed,
            label="phase-b3b-r3-train-observed-items-v1",
        ),
        "train_observed_normal_items": membership_record(
            observed_normal,
            label="phase-b3b-r3-train-observed-normal-items-v1",
        ),
        "model_item_universe": membership_record(
            ordered_items,
            label="phase-b3b-refit-item-universe-v1",
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
        memberships=sidecar_memberships,
    )
    store = prepare_final_refit_inference_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=ordered_items,
        train_observed_item_ids=observed,
        train_observed_normal_item_ids=observed_normal,
        frozen_preprocessing=sidecar["preprocessing"],
    )
    reconstructed = final_refit_feature_identity(store)
    if reconstructed["numeric_preprocessing_sha256"] != sidecar[
        "numeric_preprocessing_sha256"
    ]:
        raise RuntimeError("B4A numeric preprocessing identity differs")

    target_device = resolve_concrete_device(device)
    model, checkpoint = load_final_refit_checkpoint_compatible(
        two_tower_checkpoint_path,
        device=target_device,
        expected_identity=checkpoint_identity,
        reconstructed_feature_identity=reconstructed,
        final_refit_artifact_verified=refit_identity["artifacts"][
            "two_tower_epoch_1"
        ]["match"],
    )
    item_vectors = preencode_item_universe(
        model=model,
        store=store,
        touched_item_ids=set(
            int(value) for value in checkpoint["touched_item_ids"]
        ),
        device=target_device,
        batch_size=1024,
    )

    grouped = {
        int(user): rows
        for user, rows in big_context.groupby("user_id", sort=True)
    }
    eligible_users = sorted(grouped, key=_stable_user_key)[:QUERY_LIMIT]
    if len(eligible_users) != QUERY_LIMIT:
        raise RuntimeError("B4A query population is smaller than 256 users")
    user_ids = np.asarray(eligible_users, dtype=np.int64)
    histories = tuple(
        grouped[user].tail(50)["video_id"].to_numpy(np.int64)
        for user in eligible_users
    )
    history_weights = tuple(
        _history_weights(grouped[user].tail(50))
        for user in eligible_users
    )
    user_vectors = encode_query_users_from_precomputed(
        model=model,
        store=store,
        precomputed_item_vectors=item_vectors,
        user_ids=user_ids,
        histories=histories,
        history_weights=history_weights,
        user_positions={
            int(user): index + 1
            for index, user in enumerate(checkpoint["ordered_user_ids"])
        },
        touched_user_ids=set(
            int(value) for value in checkpoint["touched_user_ids"]
        ),
        device=target_device,
        batch_size=128,
    )
    real_items = require_unit_rows(item_vectors.cpu().numpy())
    queries = require_unit_rows(user_vectors.cpu().numpy())
    vector_identity = {
        "two_tower_checkpoint_sha256": refit_identity["artifacts"][
            "two_tower_epoch_1"
        ]["actual_sha256"],
        "numeric_sidecar_file_sha256": sha256_file(numeric_sidecar_path),
        "numeric_preprocessing_sha256": sidecar[
            "numeric_preprocessing_sha256"
        ],
        "real_item_membership": membership_record(
            ordered_items, label="phase-b3b-refit-item-universe-v1"
        ),
        "query_user_membership": membership_record(
            np.sort(user_ids), label="phase-b4a-query-users-v1"
        ),
        "query_user_order_sha256": _ordered_query_user_sha256(user_ids),
        "query_selection_seed": BENCHMARK_SEED,
        "small_matrix_accessed": False,
        "temporal_final_accessed": False,
    }
    return real_items, queries, vector_identity


def run(
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
    report_json: Path,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    real_items, queries, vector_identity = _load_real_vectors(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        caption_cache_path=caption_cache_path,
        caption_metadata_path=caption_metadata_path,
        popularity_path=popularity_path,
        bpr_checkpoint_path=bpr_checkpoint_path,
        two_tower_checkpoint_path=two_tower_checkpoint_path,
        final_refit_report_path=final_refit_report_path,
        numeric_sidecar_path=numeric_sidecar_path,
        device=device,
    )
    results = []
    for scale_name, target_count, synthetic_extension in SCALE_SPECS:
        catalog = extend_catalog(
            real_items,
            target_count=target_count,
            seed=BENCHMARK_SEED,
        )
        results.append(
            benchmark_catalog(
                queries=queries,
                catalog=catalog,
                scale_name=scale_name,
                synthetic_extension=synthetic_extension,
            )
        )
        del catalog

    report = {
        "phase": "phase-b4a-faiss-scalability",
        "scope": {
            "real_10k_catalog": True,
            "synthetic_scale_extension": True,
            "recommendation_effectiveness_claim": False,
            "small_labels_accessed": False,
            "temporal_final_accessed": False,
        },
        "contract": {
            "vector_dimension": VECTOR_DIMENSION,
            "top_k": TOP_K,
            "query_limit": QUERY_LIMIT,
            "seed": BENCHMARK_SEED,
            "thread_count": THREAD_COUNT,
            "hnsw": dict(HNSW_CONFIG),
            "tie_contract": "inner_product_desc_then_item_position_asc",
        },
        "vector_identity": vector_identity,
        "runtime_identity": runtime_identity(),
        "scales": results,
        "total_wall_time_s": time.perf_counter() - started,
        "peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
