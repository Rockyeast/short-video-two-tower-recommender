"""Load the frozen recommendation engine shared by CLI and HTTP serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import (
    BPRRetriever,
    DynamicTwoTowerRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
)
from .serving_bundle import load_serving_bundle


def load_recommendation_engine(
    *,
    config_path: Path,
    bundle_path: Path,
    metadata_path: Path,
) -> tuple[RecommendationEngine, dict[str, Any]]:
    """Load one identity-checked serving bundle into immutable route adapters."""

    payload, metadata = load_serving_bundle(
        bundle_path=bundle_path,
        metadata_path=metadata_path,
    )
    popularity_ids = payload["popularity_item_ids"].astype(
        np.int64, copy=True
    )
    popularity_scores = payload["popularity_scores"].astype(
        np.float64, copy=True
    )
    if len(popularity_ids) != len(popularity_scores):
        raise ValueError("Popularity IDs and scores differ in length")

    engine = RecommendationEngine(
        catalog=payload["catalog"].astype(np.int64, copy=True),
        two_tower=DynamicTwoTowerRetriever(
            item_ids=payload["two_tower_item_ids"].astype(
                np.int64, copy=True
            ),
            item_vectors=payload["two_tower_item_vectors"].astype(
                np.float32, copy=True
            ),
            user_ids=payload["two_tower_user_ids"].astype(
                np.int64, copy=True
            ),
            user_id_embeddings=payload[
                "two_tower_user_id_embeddings"
            ].astype(np.float32, copy=True),
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
        bpr=BPRRetriever(
            user_ids=payload["bpr_user_ids"].astype(np.int64, copy=True),
            item_ids=payload["bpr_item_ids"].astype(np.int64, copy=True),
            user_factors=payload["bpr_user_factors"].astype(
                np.float32, copy=True
            ),
            item_factors=payload["bpr_item_factors"].astype(
                np.float32, copy=True
            ),
        ),
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
    return engine, metadata
