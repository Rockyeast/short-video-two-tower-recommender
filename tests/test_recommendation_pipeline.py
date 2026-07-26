from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kuairec_fully_observed.pipeline import (
    BPRRetriever,
    DynamicTwoTowerRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
    TwoTowerRetriever,
)
from kuairec_fully_observed.reranker_serving import (
    LocalLightGBMReranker,
    load_local_lightgbm_reranker,
    write_reranker_feature_bundle,
)
from kuairec_fully_observed.reranking import (
    build_rerank_feature_matrix,
)
from kuairec_fully_observed.serving_bundle import (
    load_serving_bundle,
    write_serving_bundle,
)
from kuairec_fully_observed.torch_models import TwoTowerV1
from scripts.recommend import load_engine
from scripts import export_serving_bundle


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


class _ReverseReranker:
    name = "reverse"

    def rerank(self, **kwargs):
        return kwargs["candidates"][::-1].copy()


def test_reranker_is_request_switchable_and_preserves_candidate_set():
    two, bpr, popularity = _routes()
    engine = RecommendationEngine(
        catalog=np.arange(1, 7, dtype=np.int64),
        two_tower=two,
        bpr=bpr,
        popularity=popularity,
        config=PipelineConfig(route_top_k=6, output_k=3),
        reranker=_ReverseReranker(),
    )
    baseline = engine.recommend(10, [1], top_k=3)
    reranked = engine.recommend(
        10, [1], top_k=3, use_reranker=True
    )

    assert baseline.strategy == "two_tower_bpr_rrf"
    assert reranked.strategy == "two_tower_bpr_rrf_lightgbm"
    assert set(baseline.item_ids) == set(reranked.item_ids)
    assert reranked.item_ids == tuple(reversed(baseline.item_ids))


def test_requesting_missing_reranker_fails_closed():
    with pytest.raises(ValueError, match="no compatible artifact"):
        _engine().recommend(10, [], use_reranker=True)


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


def test_history_weights_must_align_and_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _engine().recommend(10, [1, 2], history_weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        _engine().recommend(10, [1], history_weights=[-1.0])


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


def _dynamic_two_tower() -> DynamicTwoTowerRetriever:
    return DynamicTwoTowerRetriever(
        item_ids=np.arange(1, 7, dtype=np.int64),
        item_vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        ),
        user_ids=np.asarray([10], dtype=np.int64),
        user_id_embeddings=np.zeros((1, 2), dtype=np.float32),
        mlp_input_weight=np.asarray(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        mlp_input_bias=np.zeros(2, dtype=np.float32),
        mlp_output_weight=np.eye(2, dtype=np.float32),
        mlp_output_bias=np.zeros(2, dtype=np.float32),
    )


def test_dynamic_user_tower_changes_ranking_when_history_changes():
    route = _dynamic_two_tower()
    candidates = np.asarray([3, 4, 5, 6], dtype=np.int64)

    first = route.retrieve(
        user_id=10,
        history=np.asarray([1], dtype=np.int64),
        history_weights=np.ones(1, dtype=np.float32),
        candidates=candidates,
        k=4,
    )
    second = route.retrieve(
        user_id=10,
        history=np.asarray([2], dtype=np.int64),
        history_weights=np.ones(1, dtype=np.float32),
        candidates=candidates,
        k=4,
    )

    assert first.tolist() == [3, 4, 6, 5]
    assert second.tolist() == [4, 3, 5, 6]
    assert not np.array_equal(first, second)


def test_dynamic_numpy_user_tower_matches_pytorch_user_tower():
    torch.manual_seed(17)
    model = TwoTowerV1(
        num_items=3,
        num_users=1,
        num_category_tokens=1,
        num_upload_types=1,
    ).eval()
    history_vectors = torch.nn.functional.normalize(
        torch.randn(1, 3, 128), p=2, dim=2
    )
    history_weights = torch.tensor([[1.0, 0.25, 2.0]])
    with torch.inference_mode():
        expected = model.encode_users(
            user_indices=torch.tensor([1]),
            history_vectors=history_vectors,
            history_weights=history_weights,
            padding_mask=torch.ones((1, 3), dtype=torch.bool),
            use_id_embedding=torch.tensor([True]),
        )[0].numpy()
    state = model.state_dict()
    route = DynamicTwoTowerRetriever(
        item_ids=np.asarray([11, 12, 13], dtype=np.int64),
        item_vectors=history_vectors[0].numpy(),
        user_ids=np.asarray([7], dtype=np.int64),
        user_id_embeddings=state["user_id_embedding.weight"][1:2].numpy(),
        mlp_input_weight=state["user_mlp.0.weight"].numpy(),
        mlp_input_bias=state["user_mlp.0.bias"].numpy(),
        mlp_output_weight=state["user_mlp.2.weight"].numpy(),
        mlp_output_bias=state["user_mlp.2.bias"].numpy(),
    )

    actual = route.encode_user(
        7,
        np.asarray([11, 12, 13], dtype=np.int64),
        history_weights[0].numpy(),
    )

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_config_rejects_unexpected_sections(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("pipeline: {}\nextra: true\n")
    with pytest.raises(ValueError, match="exactly one pipeline"):
        PipelineConfig.from_yaml(config)


class _FakeBooster:
    def num_feature(self):
        return 10

    def predict(self, features):
        return np.asarray(features)[:, 0]


def test_online_and_offline_rerank_feature_transform_are_identical():
    items = np.asarray([3, 4], dtype=np.int64)
    categories = np.asarray([[1, -1, -1], [2, -1, -1]])
    reranker = LocalLightGBMReranker(
        booster=_FakeBooster(),
        item_ids=np.asarray([1, 3, 4], dtype=np.int64),
        category_ids=np.asarray(
            [[2, -1, -1], [1, -1, -1], [2, -1, -1]]
        ),
        video_duration=np.asarray([4.0, 5.0, 6.0]),
        train_data_cold_mask=np.asarray([False, False, True]),
        train_popularity_scores=np.asarray([2.0, 10.0, 3.0]),
    )
    online = reranker.build_features(
        history=np.asarray([1], dtype=np.int64),
        candidates=items,
        two_tower_ranked=np.asarray([4, 3], dtype=np.int64),
        bpr_ranked=np.asarray([3, 4], dtype=np.int64),
        two_tower_scores=np.asarray([0.2, 0.8]),
        bpr_scores=np.asarray([0.7, 0.1]),
    )
    offline = build_rerank_feature_matrix(
        item_ids=items,
        two_tower_scores=np.asarray([0.2, 0.8]),
        bpr_scores=np.asarray([0.7, 0.1]),
        two_tower_ranked=np.asarray([4, 3], dtype=np.int64),
        bpr_ranked=np.asarray([3, 4], dtype=np.int64),
        popularity_scores=np.asarray([10.0, 3.0]),
        item_categories=categories,
        history_categories={2},
        data_cold_mask=np.asarray([False, True]),
        video_duration=np.asarray([5.0, 6.0]),
        history_length=1,
        alpha=0.75,
        rank_constant=60,
    )
    assert np.array_equal(online, offline)


def test_reranker_feature_bundle_is_bound_to_serving_bundle(
    tmp_path, monkeypatch
):
    model = tmp_path / "model.txt"
    model.write_text("fixture")
    features = tmp_path / "features.npz"
    metadata = tmp_path / "features.json"
    arrays = {
        "item_ids": np.asarray([1, 2], dtype=np.int64),
        "category_ids": np.asarray([[1, -1, -1], [2, -1, -1]]),
        "video_duration": np.asarray([1.0, 2.0]),
        "train_data_cold_mask": np.asarray([False, True]),
        "train_popularity_scores": np.asarray([3.0, 1.0]),
    }
    write_reranker_feature_bundle(
        feature_path=features,
        metadata_path=metadata,
        arrays=arrays,
        model_path=model,
        serving_bundle_sha256="a" * 64,
        source_identity={"fixture": True},
    )
    monkeypatch.setattr(
        "kuairec_fully_observed.reranker_serving.lgb.Booster",
        lambda model_file: _FakeBooster(),
    )
    loaded, record = load_local_lightgbm_reranker(
        model_path=model,
        feature_path=features,
        metadata_path=metadata,
        serving_bundle_sha256="a" * 64,
    )
    assert loaded.item_ids.tolist() == [1, 2]
    assert record["enabled_by_default"] is False
    with pytest.raises(RuntimeError, match="serving_bundle_sha256"):
        load_local_lightgbm_reranker(
            model_path=model,
            feature_path=features,
            metadata_path=metadata,
            serving_bundle_sha256="b" * 64,
        )


def _write_bundle(path: Path) -> Path:
    two, bpr, popularity = _routes()
    arrays = {
        "catalog": np.arange(1, 7, dtype=np.int64),
        "popularity_item_ids": np.asarray(
            list(popularity.scores), dtype=np.int64
        ),
        "popularity_scores": np.asarray(
            list(popularity.scores.values()), dtype=np.float64
        ),
        "bpr_user_ids": bpr.user_ids,
        "bpr_item_ids": bpr.item_ids,
        "bpr_user_factors": bpr.user_factors,
        "bpr_item_factors": bpr.item_factors,
        "two_tower_user_ids": np.asarray([10], dtype=np.int64),
        "two_tower_user_vectors": np.asarray(
            [[1, 0]], dtype=np.float32
        ),
        "two_tower_user_id_embeddings": np.asarray(
            [[1, 0]], dtype=np.float32
        ),
        "two_tower_item_ids": two.item_ids,
        "two_tower_item_vectors": two.item_vectors,
        "two_tower_mlp_input_weight": np.asarray(
            [[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32
        ),
        "two_tower_mlp_input_bias": np.zeros(2, dtype=np.float32),
        "two_tower_mlp_output_weight": np.eye(2, dtype=np.float32),
        "two_tower_mlp_output_bias": np.zeros(2, dtype=np.float32),
    }
    metadata = path.with_suffix(".json")
    write_serving_bundle(
        bundle_path=path,
        metadata_path=metadata,
        arrays=arrays,
        source_identity={"fixture": True},
    )
    return metadata


def test_serving_bundle_loader_and_cli(tmp_path):
    bundle = tmp_path / "bundle.npz"
    config = tmp_path / "pipeline.yaml"
    metadata = _write_bundle(bundle)
    config.write_text(
        "pipeline:\n"
        "  route_top_k: 6\n"
        "  output_k: 3\n"
        "  alpha: 0.75\n"
        "  rank_constant: 60\n"
        "  max_history: 50\n"
    )
    engine = load_engine(
        config_path=config,
        bundle_path=bundle,
        metadata_path=metadata,
    )
    assert engine.recommend(10, [1]).item_ids == (2, 4, 3)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/recommend.py",
            "--bundle",
            str(bundle),
            "--metadata",
            str(metadata),
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
        env={**os.environ, "PYTHONPATH": ".:src"},
    )
    output = json.loads(completed.stdout)
    assert output["item_ids"] == [2, 4, 3]
    assert output["strategy"] == "two_tower_bpr_rrf"


def test_serving_bundle_rejects_payload_or_metadata_tampering(tmp_path):
    bundle = tmp_path / "bundle.npz"
    metadata = _write_bundle(bundle)
    payload, record = load_serving_bundle(
        bundle_path=bundle, metadata_path=metadata
    )
    assert len(payload["catalog"]) == record["catalog_count"] == 6

    metadata_payload = json.loads(metadata.read_text())
    metadata_payload["catalog_count"] = 99
    metadata.write_text(json.dumps(metadata_payload))
    with pytest.raises(RuntimeError, match="catalog_count"):
        load_serving_bundle(bundle_path=bundle, metadata_path=metadata)

    metadata = _write_bundle(bundle)
    with bundle.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="SHA256"):
        load_serving_bundle(bundle_path=bundle, metadata_path=metadata)


def test_export_snapshot_uses_only_history_before_validation_end(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "user_id": np.ones(55, dtype=np.int64),
            "video_id": np.arange(1, 56, dtype=np.int64),
            "timestamp": np.arange(55, dtype=np.float64),
            "watch_ratio": np.ones(55),
            "play_duration": np.full(55, 10.0),
            "video_duration": np.full(55, 10.0),
            "_is_quick_skip": np.zeros(55, dtype=bool),
        }
    )
    monkeypatch.setattr(
        export_serving_bundle.audit_phase0,
        "iter_user_frames",
        lambda path, columns: iter([(1, frame)]),
    )
    monkeypatch.setattr(
        export_serving_bundle.audit_phase0,
        "canonicalize_behavior_events",
        lambda raw: (raw, {}, {}, {}),
    )

    histories, weights = export_serving_bundle._snapshot_histories(
        data_dir=tmp_path,
        user_ids=np.asarray([1], dtype=np.int64),
        validation_end=50.0,
        max_history=50,
    )

    assert histories[0].tolist() == list(range(1, 51))
    assert np.array_equal(weights[0], np.ones(50, dtype=np.float32))
