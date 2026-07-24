#!/usr/bin/env python3
"""Run one recommendation from a compact, pre-encoded serving bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuairec_fully_observed.pipeline import RecommendationEngine
from kuairec_fully_observed.serving import load_recommendation_engine


def load_engine(
    *,
    config_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    reranker_model_path: Path | None = None,
    reranker_feature_path: Path | None = None,
    reranker_metadata_path: Path | None = None,
) -> RecommendationEngine:
    """Backward-compatible CLI loader used by existing tests and callers."""

    engine, _ = load_recommendation_engine(
        config_path=config_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        reranker_model_path=reranker_model_path,
        reranker_feature_path=reranker_feature_path,
        reranker_metadata_path=reranker_metadata_path,
    )
    return engine


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
    parser.add_argument(
        "--use-reranker",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--reranker-model", type=Path)
    parser.add_argument("--reranker-features", type=Path)
    parser.add_argument("--reranker-metadata", type=Path)
    arguments = parser.parse_args()
    engine = load_engine(
        config_path=arguments.config,
        bundle_path=arguments.bundle,
        metadata_path=arguments.metadata,
        reranker_model_path=arguments.reranker_model,
        reranker_feature_path=arguments.reranker_features,
        reranker_metadata_path=arguments.reranker_metadata,
    )
    result = engine.recommend(
        arguments.user_id,
        _parse_history(arguments.history),
        top_k=arguments.top_k,
        history_weights=_parse_history_weights(arguments.history_weights),
        use_reranker=arguments.use_reranker,
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
