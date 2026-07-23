from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from kuairec_fully_observed import ExactDotProductRetriever, RetrievalQueries  # noqa: E402
from kuairec_fully_observed.caption_embeddings import CaptionCache  # noqa: E402
from kuairec_fully_observed.torch_training import (  # noqa: E402
    prepare_item_feature_store,
    sample_bounded_example_indices,
)
from kuairec_fully_observed.training import build_two_tower_training_dataset  # noqa: E402
from scripts.run_phase_b2a_two_tower_smoke import (  # noqa: E402
    _load_selected_train_events,
    validate_smoke_config,
)


def _config():
    with open("configs/phase_b2a_two_tower_smoke.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_b2a_scope_rejects_small_final_or_changed_claims():
    config = _config()
    validate_smoke_config(config)
    changed = copy.deepcopy(config)
    changed["scope"]["forbidden"].remove("small_matrix")
    with pytest.raises(RuntimeError, match="forbidden scope"):
        validate_smoke_config(changed)
    changed = copy.deepcopy(config)
    changed["claims"]["formal_gate_executed"] = True
    with pytest.raises(RuntimeError, match="claim boundary"):
        validate_smoke_config(changed)


def test_plain_package_import_does_not_import_optional_training_dependencies():
    code = (
        "import sys; import kuairec_fully_observed; "
        "assert 'torch' not in sys.modules; "
        "assert 'sentence_transformers' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_selected_event_loader_resolves_frozen_canonicalizer(tmp_path):
    pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 10, 20],
            "play_duration": [3000.0, 3000.0, 3000.0],
            "video_duration": [1000.0, 1000.0, 1000.0],
            "time": ["t", "t", "t"],
            "date": [20200101, 20200101, 20200101],
            "timestamp": [1.0, 1.0, 1.0],
            "watch_ratio": [3.0, 3.0, 3.0],
        }
    ).to_csv(tmp_path / "big_matrix.csv", index=False)
    loaded = _load_selected_train_events(tmp_path, {1}, train_end=2.0)
    assert len(loaded) == 1
    assert loaded.loc[0, "user_id"] == 1


def test_feature_vocab_and_statistics_fit_train_only_with_zero_unk():
    item_ids = np.asarray([10, 20, 30], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "video_id": item_ids,
            "caption_text": ["a", "b", "c"],
            "category_ids": [(1, 2, 3), (1, 4, 5), (99, 98, 97)],
            "video_duration": [9.0, 19.0, np.nan],
            "video_width": [99.0, 199.0, np.nan],
            "video_height": [49.0, 59.0, np.nan],
            "upload_type": ["A", "B", "UNSEEN"],
            "upload_dt": ["2020-01-01", "2020-01-02", None],
        }
    )
    embeddings = np.zeros((3, 384), dtype=np.float32)
    embeddings[:, 0] = 1
    cache = CaptionCache(item_ids, embeddings, {})
    store = prepare_item_feature_store(
        static_frame=frame,
        caption_cache=cache,
        item_universe=item_ids,
        train_observed_item_ids=np.asarray([10, 20]),
        train_observed_normal_item_ids=np.asarray([10, 20]),
    )
    assert np.array_equal(store.category_indices[2], [0, 0, 0])
    assert store.upload_type_indices[2] == 0
    assert np.isfinite(store.numeric_features).all()
    assert store.preprocessing["missing_value_counts"]["upload_dt"] == 1


def test_bounded_sampling_is_reproducible_and_not_a_prefix():
    rows = []
    for user in range(80):
        for offset in range(40):
            rows.append(
                {
                    "user_id": user,
                    "video_id": user * 100 + offset,
                    "timestamp": float(offset),
                    "play_duration": 5000.0,
                    "video_duration": 1000.0,
                    "watch_ratio": 3.0,
                }
            )
    events = pd.DataFrame(rows)
    dataset = build_two_tower_training_dataset(
        events, normal_item_ids=events["video_id"].unique()
    )
    first, stats = sample_bounded_example_indices(
        dataset,
        seed=20260722,
        max_users=64,
        max_examples_per_user=32,
        max_examples=2048,
        min_users=64,
        min_examples=2000,
    )
    second, second_stats = sample_bounded_example_indices(
        dataset,
        seed=20260722,
        max_users=64,
        max_examples_per_user=32,
        max_examples=2048,
        min_users=64,
        min_examples=2000,
    )
    np.testing.assert_array_equal(first, second)
    assert stats == second_stats
    assert stats["not_csv_prefix"] is True
    assert stats["sampled_examples"] == 2048


def test_exact_smoke_topk_stays_in_candidates_filters_seen_and_has_no_padding():
    item_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    queries = RetrievalQueries(
        user_ids=np.asarray([1]),
        histories=(np.asarray([10]),),
        history_weights=(np.asarray([1.0], dtype=np.float32),),
        candidates=(np.asarray([20, 30, 40]),),
        relevant=(np.asarray([30]),),
        catalog=item_ids,
        warm_user_mask=np.asarray([True]),
    )
    topk = ExactDotProductRetriever().search(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[9, 9], [0.1, 0], [0.8, 0], [0.2, 0]], dtype=np.float32),
        item_ids=item_ids,
        candidates=queries.candidates,
        k=3,
    )
    np.testing.assert_array_equal(topk, [[30, 40, 20]])
    assert 10 not in topk[0]
    assert np.all(topk >= 0)


def test_committed_smoke_reports_have_only_stable_logical_paths():
    report_paths = [
        "reports/phase_b2a/caption_cache_metadata.json",
        "reports/phase_b2a/two_tower_smoke.json",
        "reports/phase_b2a/two_tower_smoke.md",
    ]
    for report_path in report_paths:
        text = open(report_path, encoding="utf-8").read()
        assert "/home/" not in text
        assert "\\\\Users\\" not in text
        if report_path.endswith(".json"):
            json.loads(text)
