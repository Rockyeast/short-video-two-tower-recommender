#!/usr/bin/env python3
"""Bounded real-data speed/memory smoke test; never opens final or Small."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kuairec_fully_observed import (
    PopularityBaseline,
    RetrievalQueries,
    build_bpr_training_dataset,
    build_two_tower_training_dataset,
    load_static_item_features,
    train_bpr_sgd,
)


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 4)


def run(data_dir: Path, *, max_interactions: int, max_queries: int) -> dict[str, object]:
    if not 1 <= max_interactions <= 100_000:
        raise ValueError("max_interactions must be in [1, 100000]")
    if not 1 <= max_queries <= 256:
        raise ValueError("max_queries must be in [1, 256]")
    timings: dict[str, float] = {}

    started = time.perf_counter()
    events = pd.read_csv(data_dir / "big_matrix.csv", nrows=max_interactions)
    events = events.drop_duplicates(
        ["user_id", "video_id", "timestamp"], keep="first"
    )
    timings["read_and_canonicalize_s"] = _elapsed(started)

    started = time.perf_counter()
    static = load_static_item_features(
        data_dir,
        item_ids=events["video_id"].unique(),
        chunksize=100_000,
    )
    timings["static_features_s"] = _elapsed(started)

    started = time.perf_counter()
    bpr_data = build_bpr_training_dataset(
        events,
        normal_item_ids=static.normal_item_ids,
        seed=20260722,
    )
    first_negatives = bpr_data.sample_negatives(0)
    timings["bpr_dataset_and_negative_sample_s"] = _elapsed(started)

    started = time.perf_counter()
    two_tower_data = build_two_tower_training_dataset(
        events,
        normal_item_ids=static.normal_item_ids,
    )
    lazy_probe_count = min(1_000, len(two_tower_data))
    history_item_count = sum(
        len(two_tower_data[index].history) for index in range(lazy_probe_count)
    )
    timings["two_tower_lazy_dataset_and_1000_getitem_s"] = _elapsed(started)

    started = time.perf_counter()
    trained = train_bpr_sgd(
        bpr_data,
        embedding_dim=16,
        learning_rate=0.05,
        l2=1e-4,
        epochs=1,
        batch_size=4096,
    )
    timings["bpr_one_epoch_s"] = _elapsed(started)

    started = time.perf_counter()
    known_users = trained.model.user_ids[:max_queries]
    query_users = np.concatenate((known_users, np.asarray([-1], dtype=np.int64)))
    catalog = np.asarray(static.normal_item_ids, dtype=np.int64)
    candidates: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    for user in query_users:
        positives = bpr_data.known_positive_items.get(int(user), frozenset())
        candidate_row = np.asarray(
            [item for item in catalog if int(item) not in positives], dtype=np.int64
        )
        if not len(candidate_row):
            raise ValueError("Smoke query has no candidates")
        candidates.append(candidate_row)
        relevant.append(candidate_row[:1])
    empty_histories = tuple(np.asarray([], dtype=np.int64) for _ in query_users)
    empty_weights = tuple(np.asarray([], dtype=np.float32) for _ in query_users)
    queries = RetrievalQueries(
        user_ids=query_users,
        histories=empty_histories,
        history_weights=empty_weights,
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=catalog,
        warm_user_mask=np.concatenate(
            (np.ones(len(known_users), dtype=bool), np.asarray([False]))
        ),
    )
    fallback = PopularityBaseline.fit(events)
    topk = trained.model.rank(
        queries,
        k=100,
        cold_user_fallback=fallback,
        score_block_size=64,
    )
    timings["blocked_exact_scoring_s"] = _elapsed(started)

    return {
        "scope": "first_big_matrix_rows_only_no_final_no_small",
        "raw_interactions_requested": max_interactions,
        "canonical_interactions": len(events),
        "normal_sample_items": len(catalog),
        "static_source_variant_items": len(static.variant_static_item_ids),
        "bpr_positive_events": len(bpr_data.user_ids),
        "bpr_negative_events": len(first_negatives),
        "two_tower_positive_events": len(two_tower_data),
        "lazy_histories_probed": lazy_probe_count,
        "lazy_history_items_materialized": history_item_count,
        "ranking_queries_including_one_cold_user": len(query_users),
        "topk_shape": list(topk.shape),
        "bpr_epoch_loss": trained.epoch_losses[0],
        "timings": timings,
        "total_timed_s": round(sum(timings.values()), 4),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-interactions", type=int, default=100_000)
    parser.add_argument("--max-queries", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.data_dir.resolve(),
                max_interactions=args.max_interactions,
                max_queries=args.max_queries,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
