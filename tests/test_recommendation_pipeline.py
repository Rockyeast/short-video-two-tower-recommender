from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kuairec_fully_observed.pipeline import (
    BPRRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
    TwoTowerRetriever,
)
from scripts.recommend import load_engine


def _routes() -> tuple[
    TwoTowerRetriever, BPRRetriever, PopularityRetriever
]:
    items = np.arange(1, 7, dtype=np.int64)
    two_item_vectors = np.asarray(
        [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0], [0, -1]],
        dtype=np.float32,
    )
    two = TwoTowerRetriever(
        item_ids=items,
        item_vectors=two_item_vectors,
        trained_user_ids=frozenset({10}),
        encode_user=lambda user, history, weights: np.asarray(
            [1, 0], dtype=np.float32
        ),
    )
    bpr = BPRRetriever(
        user_ids=np.asarray([10], dtype=np.int64),
        item_ids=items,
        user_factors=np.asarray([[0, 1]], dtype=np.float32),
        item_factors=two_item_vectors,
    )
    popularity = PopularityRetriever(
        {1: 1, 2: 2, 3: 6, 4: 5, 5: 4, 6: 3}
    )
    return two, bpr, popularity


def _engine() -> RecommendationEngine:
    two, bpr, popularity = _routes()
    return RecommendationEngine(
        catalog=np.arange(1, 7, dtype=np.int64),
        two_tower=two,
        bpr=bpr,
        popularity=popularity,
        config=PipelineConfig(route_top_k=6, output_k=3),
    )


def test_warm_user_uses_frozen_hybrid_and_filters_seen_items():
    result = _engine().recommend(10, [1], top_k=3)

    assert result.strategy == "two_tower_bpr_rrf"
    assert result.item_ids == (2, 4, 3)
    assert 1 not in result.item_ids
    assert result.candidate_count == 5
    assert result.seen_count == 1


def test_unknown_user_uses_popularity_for_every_route():
    result = _engine().recommend(999, [3], top_k=3)

    assert result.strategy == "cold_user_popularity"
    assert result.item_ids == (4, 5, 6)
    assert 3 not in result.item_ids


def test_empty_candidate_catalog_returns_an_explicit_empty_result():
    two, bpr, popularity = _routes()
    engine = RecommendationEngine(
        catalog=np.asarray([1, 2], dtype=np.int64),
        two_tower=two,
        bpr=bpr,
        popularity=popularity,
    )

    result = engine.recommend(10, [1, 2])

    assert result.item_ids == ()
    assert result.strategy == "empty_catalog"


def test_request_cannot_expand_the_frozen_output_limit():
    with pytest.raises(ValueError, match="configured output limit"):
        _engine().recommend(10, [], top_k=4)


def test_two_tower_rejects_candidate_without_content_vector():
    two, _, _ = _routes()
    with pytest.raises(ValueError, match="no content vector"):
        two.retrieve(
            user_id=10,
            history=np.asarray([], dtype=np.int64),
            history_weights=np.asarray([], dtype=np.float32),
            candidates=np.asarray([99], dtype=np.int64),
            k=1,
        )


def test_config_rejects_unexpected_sections(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("pipeline: {}\nextra: true\n")
    with pytest.raises(ValueError, match="exactly one pipeline"):
        PipelineConfig.from_yaml(config)


def _write_bundle(path: Path) -> None:
    two, bpr, popularity = _routes()
    np.savez(
        path,
        catalog=np.arange(1, 7, dtype=np.int64),
        popularity_item_ids=np.asarray(
            list(popularity.scores), dtype=np.int64
        ),
        popularity_scores=np.asarray(
            list(popularity.scores.values()), dtype=np.float64
        ),
        bpr_user_ids=bpr.user_ids,
        bpr_item_ids=bpr.item_ids,
        bpr_user_factors=bpr.user_factors,
        bpr_item_factors=bpr.item_factors,
        two_tower_user_ids=np.asarray([10], dtype=np.int64),
        two_tower_user_vectors=np.asarray([[1, 0]], dtype=np.float32),
        two_tower_item_ids=two.item_ids,
        two_tower_item_vectors=two.item_vectors,
    )


def test_serving_bundle_loader_and_cli(tmp_path):
    bundle = tmp_path / "bundle.npz"
    config = tmp_path / "pipeline.yaml"
    _write_bundle(bundle)
    config.write_text(
        "pipeline:\n"
        "  route_top_k: 6\n"
        "  output_k: 3\n"
        "  alpha: 0.75\n"
        "  rank_constant: 60\n"
        "  max_history: 50\n"
    )
    engine = load_engine(config_path=config, bundle_path=bundle)
    assert engine.recommend(10, [1]).item_ids == (2, 4, 3)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/recommend.py",
            "--bundle",
            str(bundle),
            "--config",
            str(config),
            "--user-id",
            "10",
            "--history",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    assert output["item_ids"] == [2, 4, 3]
    assert output["strategy"] == "two_tower_bpr_rrf"
