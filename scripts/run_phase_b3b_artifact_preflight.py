#!/usr/bin/env python3
"""Inference-only final-refit Two-Tower preflight without Small Matrix."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kuairec_fully_observed import ExactDotProductRetriever
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.data import _history_weights
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.full_training import load_canonical_train_events
from kuairec_fully_observed.provenance import (
    membership_record,
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
    prepare_item_feature_store,
    resolve_concrete_device,
)


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
    report_json: Path,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
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
    store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=ordered_items,
        train_observed_item_ids=observed,
        train_observed_normal_item_ids=observed_normal,
    )
    reconstructed = final_refit_feature_identity(store)
    if membership_record(
        ordered_items,
        label="phase-b2a-ordered-item-store-v1",
    ) != checkpoint_identity["ordered_item_store"]:
        raise RuntimeError("Final-refit ordered item membership differs")
    caption_identity = {
        "model_id": caption.metadata["model_id"],
        "resolved_revision": caption.metadata["resolved_revision"],
        "item_membership_sha256": caption.metadata[
            "ordered_item_membership_sha256"
        ],
        "embedding_payload_sha256": caption.metadata[
            "embedding_payload_sha256"
        ],
    }
    if caption_identity != checkpoint_identity["caption_identity"]:
        raise RuntimeError("Final-refit caption identity differs")

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
    if not torch.isfinite(item_vectors).all():
        raise RuntimeError("Final-refit item vectors are non-finite")
    item_norms = torch.linalg.vector_norm(item_vectors, dim=1)
    if not torch.allclose(
        item_norms,
        torch.ones_like(item_norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Final-refit item vectors are not L2 normalized")

    grouped = list(big_context.groupby("user_id", sort=True))[:8]
    if not grouped:
        raise RuntimeError("Big history has no inference-only users")
    user_ids = np.asarray([int(user) for user, _ in grouped], dtype=np.int64)
    histories = tuple(
        rows.tail(50)["video_id"].to_numpy(np.int64)
        for _, rows in grouped
    )
    history_weights = tuple(
        _history_weights(rows.tail(50)) for _, rows in grouped
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
        batch_size=8,
    )
    if not torch.isfinite(user_vectors).all():
        raise RuntimeError("Final-refit user vectors are non-finite")
    user_norms = torch.linalg.vector_norm(user_vectors, dim=1)
    if not torch.allclose(
        user_norms,
        torch.ones_like(user_norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Final-refit user vectors are not L2 normalized")

    retrieval_items = store.item_ids[:256]
    retrieval_vectors = item_vectors[:256].cpu().numpy()
    candidates = tuple(retrieval_items.copy() for _ in user_ids)
    topk = ExactDotProductRetriever().search(
        user_vectors.cpu().numpy(),
        retrieval_vectors,
        item_ids=retrieval_items,
        candidates=candidates,
        k=20,
        score_block_size=8,
    )
    if topk.shape != (len(user_ids), 20) or np.any(topk < 0):
        raise RuntimeError("Artifact-only Exact Retrieval failed")

    report = {
        "phase": "phase-b3b-r2-artifact-only-preflight",
        "checkpoint_sha256": refit_identity["artifacts"][
            "two_tower_epoch_1"
        ]["actual_sha256"],
        "reconstructed_feature_identity": reconstructed,
        "feature_identity_sha_match": {
            name: (
                reconstructed[name]
                == checkpoint_identity["feature_identity"][name]
            )
            for name in (
                "category_vocab_sha256",
                "upload_type_vocab_sha256",
                "numeric_preprocessing_sha256",
            )
        },
        "model_loaded": True,
        "item_encoding": {
            "passed": True,
            "shape": list(item_vectors.shape),
            "finite": True,
            "l2_normalized": True,
        },
        "user_encoding": {
            "passed": True,
            "shape": list(user_vectors.shape),
            "finite": True,
            "l2_normalized": True,
            "inference_only_user_count": len(user_ids),
        },
        "exact_retrieval": {
            "passed": True,
            "query_count": len(user_ids),
            "candidate_count": len(retrieval_items),
            "top_k": 20,
        },
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "small_matrix_accessed": False,
        "temporal_final_accessed": False,
        "recommendation_effectiveness_metrics_computed": False,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
