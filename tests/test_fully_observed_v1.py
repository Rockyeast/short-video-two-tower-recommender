from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from kuairec_fully_observed import (
    BPRModel,
    NumpyTwoTowerReference,
    PopularityBaseline,
    RetrievalQueries,
    build_big_validation_queries,
    build_fixed_validation_catalog,
    build_in_batch_logit_mask,
    build_small_observed_queries,
    build_two_tower_training_examples,
    data_cold_items,
    evaluate_retrieval,
    in_batch_softmax_loss,
    is_quick_skip,
    is_strong_positive,
    resolve_kuairec_data_dir,
    stable_random_rank,
    validate_model_item_feature_columns,
)


def _events(rows: list[tuple[int, int, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "video_id",
            "timestamp",
            "play_duration",
            "video_duration",
            "watch_ratio",
        ],
    )


def _queries(
    *,
    user_ids: list[int],
    candidates: list[np.ndarray],
    relevant: list[np.ndarray],
    catalog: np.ndarray,
    warm_user_mask: np.ndarray | None = None,
) -> RetrievalQueries:
    empty = tuple(np.asarray([], dtype=np.int64) for _ in user_ids)
    weights = tuple(np.asarray([], dtype=np.float32) for _ in user_ids)
    return RetrievalQueries(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        histories=empty,
        history_weights=weights,
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=catalog,
        warm_user_mask=(
            np.ones(len(user_ids), dtype=bool)
            if warm_user_mask is None
            else warm_user_mask
        ),
    )


def test_label_boundaries_are_strict():
    np.testing.assert_array_equal(
        is_strong_positive([2.0, 2.000001]), [False, True]
    )
    np.testing.assert_array_equal(
        is_quick_skip(
            [2999.0, 3000.0, 999.0, 1000.0],
            [5000.0, 5000.0, 1000.0, 1000.0],
        ),
        [True, False, True, False],
    )


def test_data_directory_is_explicit_and_contains_expected_files(tmp_path):
    names = (
        "big_matrix.csv",
        "small_matrix.csv",
        "item_daily_features.csv",
        "item_categories.csv",
        "kuairec_caption_category.csv",
    )
    for name in names:
        (tmp_path / name).touch()
    assert resolve_kuairec_data_dir({"KUAIREC_DATA_DIR": str(tmp_path)}) == tmp_path


def test_big_builder_creates_one_query_filters_seen_and_keeps_relevant():
    train = _events(
        [
            (1, 10, 1.0, 500.0, 1000.0, 0.5),
            (1, 11, 2.0, 3000.0, 1000.0, 3.0),
            (2, 12, 1.0, 3000.0, 1000.0, 3.0),
        ]
    )
    validation = _events(
        [
            (1, 11, 10.0, 3000.0, 1000.0, 3.0),
            (1, 12, 11.0, 3000.0, 1000.0, 3.0),
            (1, 13, 12.0, 2000.0, 1000.0, 2.0),
            (1, 14, 13.0, 3000.0, 1000.0, 3.0),
            (2, 14, 10.0, 2000.0, 1000.0, 2.0),
        ]
    )
    catalog = build_fixed_validation_catalog(
        train, validation, normal_item_ids=np.asarray([10, 11, 12, 13, 14, 99])
    )
    queries = build_big_validation_queries(
        train, validation, fixed_catalog=catalog
    )

    np.testing.assert_array_equal(catalog, [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(queries.user_ids, [1])
    np.testing.assert_array_equal(queries.histories[0], [10, 11])
    np.testing.assert_array_equal(queries.candidates[0], [12, 13, 14])
    np.testing.assert_array_equal(queries.relevant[0], [12, 14])
    assert queries.diagnostics["zero_relevant_users_excluded"] == 1


def test_big_builder_fails_if_relevant_is_not_in_frozen_catalog():
    train = _events([(1, 10, 1.0, 500.0, 1000.0, 0.5)])
    validation = _events([(1, 20, 2.0, 3000.0, 1000.0, 3.0)])
    with pytest.raises(ValueError, match="outside the frozen catalog"):
        build_big_validation_queries(
            train, validation, fixed_catalog=np.asarray([10], dtype=np.int64)
        )


def test_big_builder_retains_validation_only_cold_user():
    train = _events([(1, 10, 1.0, 500.0, 1000.0, 0.5)])
    validation = _events([(2, 20, 2.0, 3000.0, 1000.0, 3.0)])
    queries = build_big_validation_queries(
        train, validation, fixed_catalog=np.asarray([10, 20])
    )

    np.testing.assert_array_equal(queries.user_ids, [2])
    assert len(queries.histories[0]) == 0
    assert queries.warm_user_mask.tolist() == [False]
    np.testing.assert_array_equal(queries.relevant[0], [20])


def test_small_candidates_are_observed_normal_pairs_only():
    observed = _events(
        [
            (1, 1, 1.0, 3000.0, 1000.0, 3.0),
            (1, 2, 2.0, 3000.0, 1000.0, 3.0),
            (1, 3, 3.0, 1000.0, 1000.0, 1.0),
        ]
    )
    big_history = _events(
        [
            (1, 8, 1.0, 500.0, 1000.0, 0.5),
            (1, 9, 2.0, 3000.0, 1000.0, 3.0),
        ]
    )
    queries = build_small_observed_queries(
        observed,
        big_history_events=big_history,
        normal_item_ids=np.asarray([1, 3, 4]),
    )

    np.testing.assert_array_equal(queries.candidates[0], [1, 3])
    np.testing.assert_array_equal(queries.relevant[0], [1])
    np.testing.assert_array_equal(queries.histories[0], [8, 9])
    assert queries.warm_user_mask.tolist() == [True]
    assert 2 not in queries.candidates[0]
    assert 4 not in queries.candidates[0]


def test_small_retains_cold_users_without_using_small_feedback_as_history():
    observed = _events(
        [
            (7, 1, 10.0, 3000.0, 1000.0, 3.0),
            (8, 2, 11.0, 3000.0, 1000.0, 3.0),
        ]
    )
    big_history = _events([(7, 99, 1.0, 1000.0, 1000.0, 1.0)])
    queries = build_small_observed_queries(
        observed,
        big_history_events=big_history,
        normal_item_ids=np.asarray([1, 2]),
    )

    np.testing.assert_array_equal(queries.user_ids, [7, 8])
    np.testing.assert_array_equal(queries.histories[0], [99])
    assert len(queries.histories[1]) == 0
    assert queries.warm_user_mask.tolist() == [True, False]
    assert queries.diagnostics["cold_users_retained"] == 1


def test_data_cold_uses_any_train_interaction_not_only_positive():
    train = _events(
        [
            (1, 10, 1.0, 100.0, 1000.0, 0.1),
            (1, 11, 2.0, 3000.0, 1000.0, 3.0),
        ]
    )
    cold = data_cold_items(train, catalog=np.asarray([10, 11, 12]))
    np.testing.assert_array_equal(cold, [12])


def test_recall_ndcg_coverage_fixture_matches_hand_calculation():
    catalog = np.arange(101, dtype=np.int64)
    queries = _queries(
        user_ids=[7],
        candidates=[catalog],
        relevant=[np.asarray([0, 100], dtype=np.int64)],
        catalog=catalog,
    )
    topk = np.arange(100, dtype=np.int64)[None, :]
    result = evaluate_retrieval(
        topk, queries, data_cold_item_ids=np.asarray([100])
    )
    ideal = 1.0 + 1.0 / np.log2(3.0)

    assert result["metrics"]["Recall@20"] == 0.5
    assert result["metrics"]["Recall@50"] == 0.5
    assert result["metrics"]["Recall@100"] == 0.5
    assert result["metrics"]["NDCG@20"] == pytest.approx(1.0 / ideal)
    assert result["metrics"]["Coverage@100"] == pytest.approx(100 / 101)
    assert result["metrics"]["Data-Cold Recall@100"] == 0.0
    assert result["denominators"]["data_cold_target_count"] == 1


def test_popularity_counts_only_strong_train_feedback():
    train = _events(
        [
            (1, 1, 1.0, 2000.0, 1000.0, 2.0),
            (1, 2, 2.0, 3000.0, 1000.0, 3.0),
            (2, 1, 3.0, 4000.0, 1000.0, 4.0),
        ]
    )
    queries = _queries(
        user_ids=[1],
        candidates=[np.asarray([1, 2])],
        relevant=[np.asarray([1])],
        catalog=np.asarray([1, 2]),
    )
    ranked = PopularityBaseline.fit(train).rank(queries, k=2)
    np.testing.assert_array_equal(ranked, [[1, 2]])


def test_random_sanity_baseline_is_deterministic_and_candidate_only():
    queries = _queries(
        user_ids=[1],
        candidates=[np.asarray([1, 3, 5])],
        relevant=[np.asarray([3])],
        catalog=np.asarray([1, 2, 3, 4, 5]),
    )
    first = stable_random_rank(queries, seed=7, k=3)
    second = stable_random_rank(queries, seed=7, k=3)
    np.testing.assert_array_equal(first, second)
    assert set(first[0]) == {1, 3, 5}


def test_bpr_interface_ranks_exactly_with_stable_item_ties():
    queries = _queries(
        user_ids=[10],
        candidates=[np.asarray([100, 200, 300])],
        relevant=[np.asarray([200])],
        catalog=np.asarray([100, 200, 300]),
    )
    model = BPRModel(
        user_ids=np.asarray([10]),
        item_ids=np.asarray([100, 200, 300]),
        user_factors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        item_factors=np.asarray(
            [[0.0, 1.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32
        ),
    )
    np.testing.assert_array_equal(model.rank(queries, k=3), [[200, 300, 100]])


def test_bpr_cold_item_is_zero_scored_and_cold_user_uses_popularity():
    queries = _queries(
        user_ids=[10, 99],
        candidates=[np.asarray([100, 200, 300]), np.asarray([100, 200, 300])],
        relevant=[np.asarray([300]), np.asarray([200])],
        catalog=np.asarray([100, 200, 300]),
        warm_user_mask=np.asarray([True, False]),
    )
    model = BPRModel(
        user_ids=np.asarray([10]),
        item_ids=np.asarray([100, 200]),
        user_factors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        item_factors=np.asarray([[-1.0, 0.0], [-2.0, 0.0]], dtype=np.float32),
    )
    fallback = PopularityBaseline({100: 1.0, 200: 3.0, 300: 2.0})

    ranked = model.rank(queries, k=3, cold_user_fallback=fallback)
    np.testing.assert_array_equal(ranked, [[300, 100, 200], [200, 300, 100]])
    result = evaluate_retrieval(ranked, queries)
    assert result["denominators"]["query_count"] == 1
    assert result["denominators"]["all_query_count"] == 2
    assert result["cold_user_denominators"]["query_count"] == 1
    assert result["cold_user_metrics"]["Recall@100"] == 1.0


def test_two_tower_reference_shapes_normalization_and_loss():
    model = NumpyTwoTowerReference(
        num_users=3,
        num_items=4,
        num_categories=2,
        caption_dim=5,
        static_dim=2,
        output_dim=128,
    )
    item_vectors = model.encode_items(
        np.arange(4),
        np.asarray([0, 1, 0, 1]),
        np.ones((4, 5), dtype=np.float32),
        np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
    )
    history = np.stack((item_vectors[:3], item_vectors[1:4]))
    user_vectors = model.encode_users(
        np.asarray([0, 1]),
        history,
        np.asarray([[1.0, 0.25, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([[True, True, False], [True, True, True]]),
    )

    assert item_vectors.shape == (4, 128)
    assert user_vectors.shape == (2, 128)
    np.testing.assert_allclose(np.linalg.norm(item_vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(user_vectors, axis=1), 1.0, atol=1e-5)
    loss = in_batch_softmax_loss(
        user_vectors, item_vectors[:2], temperature=0.07
    )
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_two_tower_cold_item_ignores_untrained_id_embedding():
    model = NumpyTwoTowerReference(
        num_users=1,
        num_items=2,
        num_categories=1,
        caption_dim=2,
        static_dim=1,
        output_dim=8,
    )
    arguments = (
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([[0.5, 0.25]], dtype=np.float32),
        np.asarray([[1.0]], dtype=np.float32),
    )
    before = model.encode_items(
        *arguments, use_id_embedding=np.asarray([False])
    )
    model.item_id_embedding[0] = 10_000.0
    after = model.encode_items(
        *arguments, use_id_embedding=np.asarray([False])
    )
    np.testing.assert_allclose(before, after)


def test_training_examples_are_causal_and_exclude_target_from_history():
    events = _events(
        [
            (1, 4, 0.5, 1000.0, 1000.0, 1.0),
            (1, 1, 1.0, 100.0, 1000.0, 0.1),
            (1, 2, 2.0, 3000.0, 1000.0, 3.0),
            (1, 3, 2.0, 3000.0, 1000.0, 3.0),
            (1, 4, 3.0, 3000.0, 1000.0, 3.0),
            (1, 4, 4.0, 1000.0, 1000.0, 1.0),
        ]
    )
    examples = build_two_tower_training_examples(events)

    np.testing.assert_array_equal(examples.target_item_ids, [2, 3, 4])
    np.testing.assert_array_equal(examples.histories[0], [4, 1])
    np.testing.assert_array_equal(examples.histories[1], [4, 1])
    np.testing.assert_array_equal(examples.histories[2], [1, 2, 3])
    for target, history in zip(
        examples.target_item_ids, examples.histories, strict=True
    ):
        assert int(target) not in set(int(item) for item in history)
    assert examples.history_weights[0][1] == pytest.approx(0.025)


def test_in_batch_mask_removes_duplicate_and_known_positive_false_negatives():
    mask = build_in_batch_logit_mask(
        np.asarray([1, 1, 2]),
        np.asarray([2, 3, 2]),
        {1: frozenset({2, 3}), 2: frozenset({2})},
    )
    np.testing.assert_array_equal(
        mask,
        [
            [True, False, False],
            [False, True, False],
            [False, True, True],
        ],
    )
    vectors = np.eye(3, dtype=np.float32)
    assert np.isfinite(
        in_batch_softmax_loss(
            vectors, vectors, temperature=0.1, valid_logit_mask=mask
        )
    )


def test_static_feature_allowlist_rejects_daily_aggregate_leakage():
    assert validate_model_item_feature_columns(
        ["video_id", "caption_embedding", "category_ids", "video_duration"]
    ) == ("video_id", "caption_embedding", "category_ids", "video_duration")
    for leaked in ("show_cnt", "play_cnt", "like_cnt", "follow_cnt"):
        with pytest.raises(ValueError, match="Disallowed or unknown"):
            validate_model_item_feature_columns(["video_id", leaked])


def test_selection_gate_targets_the_strongest_baseline_with_numeric_thresholds():
    with open("configs/fully_observed_v1.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    gate = config["selection_gate"]

    assert gate["comparison_baseline"] == (
        "max_recall_at_100_of_global_popularity_and_bpr"
    )
    assert gate["coverage_tradeoff"] == {
        "maximum_recall_at_100_deficit_absolute": 0.02,
        "minimum_coverage_at_100_gain_absolute": 0.05,
    }
    assert gate["data_cold_tradeoff"]["minimum_target_denominator"] == 100
