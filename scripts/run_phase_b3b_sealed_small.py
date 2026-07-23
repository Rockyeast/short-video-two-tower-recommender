#!/usr/bin/env python3
"""Execute the one-time sealed Small evaluation after explicit review approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kuairec_fully_observed import (
    BPRModel,
    ExactDotProductRetriever,
    PopularityBaseline,
    build_small_observed_queries,
    data_cold_items,
    evaluate_frozen_small_routes,
    load_static_item_features,
    require_sealed_execution,
    stable_random_rank,
    verify_final_refit_artifacts,
    verify_frozen_small_source,
)
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.full_training import load_canonical_train_events
from kuairec_fully_observed.provenance import (
    membership_record,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.torch_training import (
    encode_query_users_from_precomputed,
    final_refit_feature_identity,
    load_final_refit_checkpoint_compatible,
    preencode_item_universe,
    prepare_item_feature_store,
    resolve_concrete_device,
)


SMALL_COLUMNS = (
    "user_id",
    "video_id",
    "watch_ratio",
)


def _load_small_once(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(SMALL_COLUMNS))


def _load_bpr(path: Path) -> BPRModel:
    with np.load(path) as payload:
        if int(payload["epoch"][0]) != 20:
            raise RuntimeError("Final refit BPR checkpoint is not epoch 20")
        return BPRModel(
            user_ids=payload["user_ids"].astype(np.int64, copy=True),
            item_ids=payload["item_ids"].astype(np.int64, copy=True),
            user_factors=payload["user_factors"].astype(np.float32, copy=True),
            item_factors=payload["item_factors"].astype(np.float32, copy=True),
        )


def run(
    *,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    popularity_path: Path,
    bpr_checkpoint: Path,
    two_tower_checkpoint: Path,
    final_refit_report_path: Path,
    split_manifest_path: Path,
    report_json: Path,
    execute_sealed_small: bool,
    device: str = "cpu",
) -> dict:
    require_sealed_execution(execute_sealed_small)
    small_source_identity = verify_frozen_small_source(
        small_path=data_dir / "small_matrix.csv",
        split_manifest_path=split_manifest_path,
    )
    refit_identity = verify_final_refit_artifacts(
        final_refit_report_path=final_refit_report_path,
        popularity_path=popularity_path,
        bpr_checkpoint_path=bpr_checkpoint,
        two_tower_checkpoint_path=two_tower_checkpoint,
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
    small = _load_small_once(data_dir / "small_matrix.csv")
    observed_normal = small[
        small["video_id"].isin(static.normal_item_ids)
    ]
    queries = build_small_observed_queries(
        small,
        big_history_events=big_context,
        normal_item_ids=static.normal_item_ids,
        max_history=50,
    )
    cold_items = data_cold_items(big_context, catalog=queries.catalog)
    popularity = PopularityBaseline(
        {
            int(item): float(score)
            for item, score in json.loads(popularity_path.read_text()).items()
        }
    )
    popularity_topk = popularity.rank(queries, k=500)
    bpr_topk = _load_bpr(bpr_checkpoint).rank(
        queries,
        k=500,
        cold_user_fallback=popularity,
        score_block_size=128,
    )

    checkpoint_payload = torch.load(
        two_tower_checkpoint, map_location="cpu", weights_only=False
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
    reconstructed_feature_identity = final_refit_feature_identity(store)
    if membership_record(
        ordered_items,
        label="phase-b2a-ordered-item-store-v1",
    ) != checkpoint_identity["ordered_item_store"]:
        raise RuntimeError(
            "Final-refit ordered item membership differs"
        )
    reconstructed_caption_identity = {
        "model_id": caption.metadata["model_id"],
        "resolved_revision": caption.metadata["resolved_revision"],
        "item_membership_sha256": caption.metadata[
            "ordered_item_membership_sha256"
        ],
        "embedding_payload_sha256": caption.metadata[
            "embedding_payload_sha256"
        ],
    }
    if (
        reconstructed_caption_identity
        != checkpoint_identity["caption_identity"]
    ):
        raise RuntimeError("Final-refit caption identity differs")
    target_device = resolve_concrete_device(device)
    model, checkpoint = load_final_refit_checkpoint_compatible(
        two_tower_checkpoint,
        device=target_device,
        expected_identity=checkpoint_identity,
        reconstructed_feature_identity=reconstructed_feature_identity,
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
    user_vectors = encode_query_users_from_precomputed(
        model=model,
        store=store,
        precomputed_item_vectors=item_vectors,
        user_ids=queries.user_ids,
        histories=queries.histories,
        history_weights=queries.history_weights,
        user_positions={
            int(user): index + 1
            for index, user in enumerate(checkpoint["ordered_user_ids"])
        },
        touched_user_ids=set(
            int(value) for value in checkpoint["touched_user_ids"]
        ),
        device=target_device,
        batch_size=128,
    ).cpu().numpy()
    positions = np.asarray(
        [store.positions[int(item)] for item in queries.catalog], dtype=np.int64
    )
    catalog_vectors = item_vectors[
        torch.as_tensor(positions, dtype=torch.long, device=target_device)
    ].cpu().numpy()
    two_tower_topk = ExactDotProductRetriever().search(
        user_vectors,
        catalog_vectors,
        item_ids=queries.catalog,
        candidates=queries.candidates,
        k=500,
        warm_user_mask=queries.warm_user_mask,
        fallback_topk=popularity_topk,
        score_block_size=128,
    )
    result = evaluate_frozen_small_routes(
        queries=queries,
        random_topk=stable_random_rank(queries, seed=20260722, k=500),
        popularity_topk=popularity_topk,
        bpr_topk=bpr_topk,
        two_tower_topk=two_tower_topk,
        data_cold_item_ids=cold_items,
    )
    serializable = {
        "phase": "phase-b3b-sealed-small",
        "sealed_attempt_number": 2,
        "prior_attempt_metrics_produced": False,
        "prior_failure_stage": "small_schema_validation",
        "selection_performed": False,
        "small_matrix_accessed_once": True,
        "temporal_final_accessed": False,
        "input_identity": {
            "small_matrix": small_source_identity,
            "final_refit": refit_identity,
        },
        "recipe": result["recipe"],
        "query_count": int(len(queries.user_ids)),
        "warm_user_count": int(queries.warm_user_mask.sum()),
        "cold_user_count": int((~queries.warm_user_mask).sum()),
        "target_count": int(sum(len(row) for row in queries.relevant)),
        "audit_counts": {
            "observed_pair_count": int(
                small[["user_id", "video_id"]].drop_duplicates().shape[0]
            ),
            "observed_normal_pair_count": int(
                observed_normal[["user_id", "video_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "normal_candidate_item_count": int(
                observed_normal["video_id"].nunique()
            ),
            "excluded_zero_relevant_user_count": int(
                queries.diagnostics["zero_relevant_users_excluded"]
            ),
            "data_cold_item_count": int(len(cold_items)),
        },
        "results": {
            name: record["metrics"] for name, record in result["results"].items()
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    return serializable


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
        "--split-manifest",
        type=Path,
        default=Path("manifests/split_manifest.json"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute-sealed-small", action="store_true")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b3b/sealed_small.json"),
    )
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        popularity_path=args.popularity.resolve(),
        bpr_checkpoint=args.bpr_checkpoint.resolve(),
        two_tower_checkpoint=args.two_tower_checkpoint.resolve(),
        final_refit_report_path=args.final_refit_report.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        report_json=args.report_json.resolve(),
        execute_sealed_small=args.execute_sealed_small,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
