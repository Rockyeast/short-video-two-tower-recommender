#!/usr/bin/env python3
"""Run one recommendation from a compact, pre-encoded serving bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kuairec_fully_observed.pipeline import (
    BPRRetriever,
    DynamicTwoTowerRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
)
from kuairec_fully_observed.serving_bundle import load_serving_bundle


def load_engine(
    *, config_path: Path, bundle_path: Path, metadata_path: Path
) -> RecommendationEngine:
    """Load the thin pipeline; model training/export remains outside this CLI."""

    payload, _ = load_serving_bundle(
        bundle_path=bundle_path, metadata_path=metadata_path
    )
    catalog = payload["catalog"].astype(np.int64, copy=True)
    popularity_ids = payload["popularity_item_ids"].astype(
        np.int64, copy=True
    )
    popularity_scores = payload["popularity_scores"].astype(
        np.float64, copy=True
    )
    bpr = BPRRetriever(
        user_ids=payload["bpr_user_ids"].astype(np.int64, copy=True),
        item_ids=payload["bpr_item_ids"].astype(np.int64, copy=True),
        user_factors=payload["bpr_user_factors"].astype(
            np.float32, copy=True
        ),
        item_factors=payload["bpr_item_factors"].astype(
            np.float32, copy=True
        ),
    )
    two_item_ids = payload["two_tower_item_ids"].astype(
        np.int64, copy=True
    )
    two_item_vectors = payload["two_tower_item_vectors"].astype(
        np.float32, copy=True
    )
    two_user_ids = payload["two_tower_user_ids"].astype(
        np.int64, copy=True
    )
    user_id_embeddings = payload[
        "two_tower_user_id_embeddings"
    ].astype(np.float32, copy=True)

    if len(popularity_ids) != len(popularity_scores):
        raise ValueError("Popularity IDs and scores differ in length")

    return RecommendationEngine(
        catalog=catalog,
        two_tower=DynamicTwoTowerRetriever(
            item_ids=two_item_ids,
            item_vectors=two_item_vectors,
            user_ids=two_user_ids,
            user_id_embeddings=user_id_embeddings,
            mlp_input_weight=payload[
                "two_tower_mlp_input_weight"
            ].astype(np.float32, copy=True),
            mlp_input_bias=payload["two_tower_mlp_input_bias"].astype(
                np.float32, copy=True
            ),
            mlp_output_weight=payload[
                "two_tower_mlp_output_weight"
            ].astype(np.float32, copy=True),
            mlp_output_bias=payload["two_tower_mlp_output_bias"].astype(
                np.float32, copy=True
            ),
        ),
        bpr=bpr,
        popularity=PopularityRetriever(
            dict(
                zip(
                    (int(value) for value in popularity_ids),
                    (float(value) for value in popularity_scores),
                    strict=True,
                )
            )
        ),
        config=PipelineConfig.from_yaml(config_path),
    )


def _parse_history(raw: str) -> list[int]:
    return [] if not raw.strip() else [int(value) for value in raw.split(",")]


def _parse_history_weights(raw: str) -> list[float] | None:
    if not raw.strip():
        return None
    return [float(value) for value in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recommendation_pipeline_v1.yaml"),
    )
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--history", default="")
    parser.add_argument(
        "--history-weights",
        default="",
        help="Optional comma-separated weights aligned with --history",
    )
    parser.add_argument("--top-k", type=int, default=None)
    arguments = parser.parse_args()
    engine = load_engine(
        config_path=arguments.config,
        bundle_path=arguments.bundle,
        metadata_path=arguments.metadata,
    )
    result = engine.recommend(
        arguments.user_id,
        _parse_history(arguments.history),
        top_k=arguments.top_k,
        history_weights=_parse_history_weights(arguments.history_weights),
    )
    print(
        json.dumps(
            {
                "user_id": result.user_id,
                "item_ids": result.item_ids,
                "strategy": result.strategy,
                "candidate_count": result.candidate_count,
                "seen_count": result.seen_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
