from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kuairec_fully_observed.pipeline import (
    BPRRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
    TwoTowerRetriever,
)
from kuairec_fully_observed.serving_bundle import (
    load_serving_bundle,
    write_serving_bundle,
)
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
        "two_tower_item_ids": two.item_ids,
        "two_tower_item_vectors": two.item_vectors,
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
