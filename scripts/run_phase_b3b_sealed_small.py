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
)
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.full_training import load_canonical_train_events
from kuairec_fully_observed.provenance import verify_phase_b2a_inputs
from kuairec_fully_observed.torch_training import (
    encode_query_users_from_precomputed,
    load_checkpoint,
    preencode_item_universe,
    prepare_item_feature_store,
    resolve_concrete_device,
)


SMALL_COLUMNS = (
    "user_id",
    "video_id",
    "timestamp",
    "play_duration",
    "video_duration",
    "watch_ratio",
)


def _load_small_once(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(SMALL_COLUMNS))
    if frame.duplicated(["user_id", "video_id", "timestamp"]).any():
        raise RuntimeError("Small Matrix is not canonical by event key")
    return frame


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
    report_json: Path,
    execute_sealed_small: bool,
    device: str = "cpu",
) -> dict:
    require_sealed_execution(execute_sealed_small)
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

    payload = torch.load(
        two_tower_checkpoint, map_location="cpu", weights_only=False
    )
    model, checkpoint = load_checkpoint(
        two_tower_checkpoint,
        device=resolve_concrete_device(device),
        expected_identity=payload["identity"],
    )
    ordered_items = np.asarray(
        checkpoint["ordered_item_ids"], dtype=np.int64
    )
    frame = static.frame.set_index("video_id").reindex(ordered_items)
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=ordered_items,
        expected_model_id=checkpoint["identity"]["caption_identity"]["model_id"],
        expected_revision=checkpoint["identity"]["caption_identity"][
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
    target_device = resolve_concrete_device(device)
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
        "selection_performed": False,
        "small_matrix_accessed_once": True,
        "temporal_final_accessed": False,
        "recipe": result["recipe"],
        "query_count": int(len(queries.user_ids)),
        "warm_user_count": int(queries.warm_user_mask.sum()),
        "cold_user_count": int((~queries.warm_user_mask).sum()),
        "target_count": int(sum(len(row) for row in queries.relevant)),
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
        report_json=args.report_json.resolve(),
        execute_sealed_small=args.execute_sealed_small,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
