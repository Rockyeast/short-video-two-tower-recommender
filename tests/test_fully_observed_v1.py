from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from kuairec_fully_observed import (
    BPRModel,
    ExactDotProductRetriever,
    NumpyTwoTowerReference,
    PopularityBaseline,
    RetrievalQueries,
    build_big_validation_queries,
    build_bpr_training_dataset,
    build_fixed_validation_catalog,
    build_in_batch_logit_mask,
    build_small_observed_queries,
    build_two_tower_training_examples,
    build_two_tower_training_dataset,
    data_cold_items,
    evaluate_frozen_small_routes,
    evaluate_retrieval,
    in_batch_softmax_loss,
    is_quick_skip,
    is_strong_positive,
    load_static_item_features,
    resolve_kuairec_data_dir,
    stable_random_rank,
    train_bpr_sgd,
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


def test_sealed_small_routes_share_cold_fallback_and_keep_frozen_hybrid():
    observed = _events(
        [
            (7, 1, 10.0, 3000.0, 1000.0, 3.0),
            (7, 2, 11.0, 1000.0, 1000.0, 1.0),
            (7, 3, 12.0, 1000.0, 1000.0, 1.0),
            (8, 1, 10.0, 1000.0, 1000.0, 1.0),
            (8, 2, 11.0, 3000.0, 1000.0, 3.0),
            (8, 3, 12.0, 1000.0, 1000.0, 1.0),
        ]
    )
    big = _events([(7, 99, 1.0, 500.0, 1000.0, 0.5)])
    queries = build_small_observed_queries(
        observed,
        big_history_events=big,
        normal_item_ids=np.asarray([1, 2, 3, 4]),
    )
    popularity = np.asarray([[3, 2, 1], [3, 2, 1]])
    result = evaluate_frozen_small_routes(
        queries=queries,
        random_topk=np.asarray([[2, 3, 1], [1, 3, 2]]),
        popularity_topk=popularity,
        bpr_topk=np.asarray([[1, 2, 3], [2, 1, 3]]),
        two_tower_topk=np.asarray([[1, 3, 2], [2, 3, 1]]),
        data_cold_item_ids=np.asarray([3]),
    )

    assert result["recipe"] == {
        "alpha": 0.75,
        "route_top_k": 500,
        "rank_constant": 60,
        "output_k": 100,
        "cold_user_fallback": "refit_global_popularity",
    }
    for name in ("random", "bpr", "two_tower", "hybrid_alpha_0.75"):
        np.testing.assert_array_equal(
            result["results"][name]["topk"][1, :3], popularity[1]
        )
    np.testing.assert_array_equal(
        result["results"]["hybrid_alpha_0.75"]["topk"][0, :3],
        [1, 3, 2],
    )
    assert (
        result["results"]["two_tower"]["metrics"][
            "cold_user_denominators"
        ]["query_count"]
        == 1
    )


def test_sealed_small_rejects_blocked_or_unobserved_ranked_item():
    queries = _queries(
        user_ids=[1],
        candidates=[np.asarray([1, 2])],
        relevant=[np.asarray([1])],
        catalog=np.asarray([1, 2, 3]),
    )
    with pytest.raises(ValueError, match="unavailable Small pair"):
        evaluate_frozen_small_routes(
            queries=queries,
            random_topk=np.asarray([[1, 2]]),
            popularity_topk=np.asarray([[1, 2]]),
            bpr_topk=np.asarray([[1, 3]]),
            two_tower_topk=np.asarray([[1, 2]]),
            data_cold_item_ids=np.asarray([], dtype=np.int64),
        )


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


def test_block_exact_scoring_and_shared_cold_user_fallback():
    retriever = ExactDotProductRetriever()
    users = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    items = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    candidates = tuple(np.asarray([1, 2, 3]) for _ in range(3))
    fallback = np.asarray([[3, 2, 1], [2, 3, 1], [1, 2, 3]])

    ranked = retriever.search(
        users,
        items,
        item_ids=np.asarray([1, 2]),
        candidates=candidates,
        k=3,
        warm_user_mask=np.asarray([True, False, True]),
        fallback_topk=fallback,
        missing_item_score=0.0,
        score_block_size=2,
    )
    np.testing.assert_array_equal(
        ranked,
        [
            [1, 2, 3],
            [2, 3, 1],
            [2, 1, 3],
        ],
    )


def test_blocked_exact_scoring_matches_naive_randomized_reference():
    rng = np.random.default_rng(20260722)
    retriever = ExactDotProductRetriever()
    for _ in range(10):
        query_count = int(rng.integers(3, 12))
        learned_item_count = int(rng.integers(5, 20))
        dimension = int(rng.integers(2, 10))
        k = 7
        item_ids = np.arange(100, 100 + learned_item_count, dtype=np.int64)
        missing_ids = np.arange(1000, 1005, dtype=np.int64)
        catalog = np.concatenate((item_ids, missing_ids))
        users = rng.normal(size=(query_count, dimension)).astype(np.float32)
        items = rng.normal(size=(learned_item_count, dimension)).astype(np.float32)
        warm = rng.random(query_count) > 0.25
        candidates = tuple(
            np.sort(
                rng.choice(
                    catalog,
                    size=int(rng.integers(1, len(catalog) + 1)),
                    replace=False,
                )
            )
            for _ in range(query_count)
        )
        expected = np.full((query_count, k), -1, dtype=np.int64)
        positions = {int(item): index for index, item in enumerate(item_ids)}
        for row, candidate_row in enumerate(candidates):
            scored = []
            for item in candidate_row:
                position = positions.get(int(item))
                score = (
                    0.0
                    if position is None
                    else float(users[row] @ items[position])
                )
                scored.append((int(item), score))
            ranked = [
                item
                for item, _ in sorted(
                    scored, key=lambda pair: (-pair[1], pair[0])
                )
            ]
            expected[row, : min(k, len(ranked))] = ranked[:k]
        fallback = expected.copy()

        for block_size in (1, 3, 16):
            actual = retriever.search(
                users,
                items,
                item_ids=item_ids,
                candidates=candidates,
                k=k,
                warm_user_mask=warm,
                fallback_topk=fallback,
                missing_item_score=0.0,
                score_block_size=block_size,
            )
            np.testing.assert_array_equal(actual, expected)


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


def test_two_tower_encoded_ranking_uses_shared_cold_user_fallback():
    model = NumpyTwoTowerReference(
        num_users=1,
        num_items=2,
        num_categories=1,
        caption_dim=1,
        static_dim=1,
        output_dim=2,
    )
    queries = _queries(
        user_ids=[1, 2],
        candidates=[np.asarray([10, 20]), np.asarray([10, 20])],
        relevant=[np.asarray([10]), np.asarray([20])],
        catalog=np.asarray([10, 20]),
        warm_user_mask=np.asarray([True, False]),
    )
    ranked = model.rank_encoded(
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        item_ids=np.asarray([10, 20]),
        queries=queries,
        learned_user_mask=np.asarray([True, False]),
        cold_user_fallback=PopularityBaseline({10: 1.0, 20: 2.0}),
        k=2,
    )
    np.testing.assert_array_equal(ranked, [[10, 20], [20, 10]])


def test_training_examples_are_causal_and_require_first_item_contact():
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
    lazy = build_two_tower_training_dataset(events)

    assert len(lazy) == 2
    np.testing.assert_array_equal(examples.target_item_ids, [2, 3])
    np.testing.assert_array_equal(examples.histories[0], [4, 1])
    np.testing.assert_array_equal(examples.histories[1], [4, 1])
    for target, history in zip(
        examples.target_item_ids, examples.histories, strict=True
    ):
        assert int(target) not in set(int(item) for item in history)
    assert 4 in examples.known_positive_items[1]
    assert examples.history_weights[0][1] == pytest.approx(0.025)


def test_bpr_epoch_sampler_excludes_all_known_positives_and_is_deterministic():
    rows = [
        (1, 3, 0.1, 1000.0, 1000.0, 1.0),
        (1, 4, 0.2, 1000.0, 1000.0, 1.0),
        (1, 5, 0.3, 1000.0, 1000.0, 1.0),
    ]
    rows.extend(
        (1, 1 + (index % 2), 1.0 + index, 3000.0, 1000.0, 3.0)
        for index in range(8)
    )
    rows.extend(
        [
            (2, 1, 20.0, 1000.0, 1000.0, 1.0),
            (2, 2, 21.0, 3000.0, 1000.0, 3.0),
        ]
    )
    dataset = build_bpr_training_dataset(
        _events(rows),
        normal_item_ids=np.asarray([1, 2, 3, 4, 5, 99]),
        seed=17,
    )
    epoch_zero = dataset.sample_negatives(0)
    epoch_zero_again = dataset.sample_negatives(0)
    epoch_one = dataset.sample_negatives(1)

    np.testing.assert_array_equal(epoch_zero, epoch_zero_again)
    assert not np.array_equal(epoch_zero, epoch_one)
    assert len(epoch_zero) == len(dataset.positive_item_ids)
    assert 99 not in dataset.negative_catalog
    for user, negative in zip(dataset.user_ids, epoch_zero, strict=True):
        assert int(negative) not in dataset.known_positive_items[int(user)]

    trained = train_bpr_sgd(
        dataset,
        embedding_dim=8,
        learning_rate=0.05,
        epochs=2,
        batch_size=4,
    )
    assert len(trained.epoch_losses) == 2
    assert np.isfinite(trained.epoch_losses).all()
    assert 99 not in trained.model.item_ids


def test_bpr_learns_at_formal_batch_size_and_is_seed_reproducible():
    rows = []
    for user in range(32):
        positive_item = 1 + user % 4
        observed_nonpositive = 10 + user % 4
        for repeat in range(8):
            timestamp = float(repeat * 2)
            rows.extend(
                [
                    (user, positive_item, timestamp, 3000.0, 1000.0, 3.0),
                    (
                        user,
                        observed_nonpositive,
                        timestamp + 1.0,
                        1000.0,
                        1000.0,
                        1.0,
                    ),
                ]
            )
    dataset = build_bpr_training_dataset(
        _events(rows),
        normal_item_ids=np.arange(1, 15, dtype=np.int64),
        seed=17,
    )
    settings = {
        "embedding_dim": 8,
        "learning_rate": 0.25,
        "l2": 1e-4,
        "epochs": 20,
        "batch_size": 4096,
    }
    first = train_bpr_sgd(dataset, **settings)
    second = train_bpr_sgd(dataset, **settings)

    assert first.epoch_losses[-1] < 0.60
    np.testing.assert_array_equal(first.epoch_losses, second.epoch_losses)
    np.testing.assert_array_equal(
        first.model.user_factors, second.model.user_factors
    )
    np.testing.assert_array_equal(
        first.model.item_factors, second.model.item_factors
    )

    user_positions = {
        int(user): index for index, user in enumerate(first.model.user_ids)
    }
    item_positions = {
        int(item): index for index, item in enumerate(first.model.item_ids)
    }
    negatives = dataset.sample_negatives(settings["epochs"] - 1)
    correct = []
    for user, positive, negative in zip(
        dataset.user_ids, dataset.positive_item_ids, negatives, strict=True
    ):
        user_vector = first.model.user_factors[user_positions[int(user)]]
        positive_vector = first.model.item_factors[item_positions[int(positive)]]
        negative_vector = first.model.item_factors[item_positions[int(negative)]]
        correct.append(
            float(user_vector @ positive_vector)
            > float(user_vector @ negative_vector)
        )
    assert np.mean(correct) >= 0.90


def test_bpr_checkpoint_callback_includes_initialization_and_can_stop():
    dataset = build_bpr_training_dataset(
        _events(
            [
                (1, 10, 1.0, 3000.0, 1000.0, 3.0),
                (1, 20, 2.0, 1000.0, 1000.0, 1.0),
                (2, 20, 1.0, 3000.0, 1000.0, 3.0),
                (2, 10, 2.0, 1000.0, 1000.0, 1.0),
            ]
        ),
        normal_item_ids=np.asarray([10, 20]),
        seed=9,
    )
    observed: list[tuple[int, int, int]] = []

    def callback(epoch, model, losses):
        observed.append((epoch, len(model.item_ids), len(losses)))
        return epoch < 1

    result = train_bpr_sgd(
        dataset,
        embedding_dim=4,
        epochs=3,
        batch_size=4096,
        checkpoint_epochs=(0, 1, 2, 3),
        checkpoint_callback=callback,
    )

    assert observed == [(0, 2, 0), (1, 2, 1)]
    assert len(result.epoch_losses) == 1


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


def test_real_static_feature_loader_never_materializes_daily_aggregates(tmp_path):
    pd.DataFrame(
        {
            "video_id": [1, 1, 2],
            "date": [20200101, 20200102, 20200101],
            "video_type": ["NORMAL", "NORMAL", "AD"],
            "upload_dt": ["2019-12-01", "2019-12-01", "2019-12-02"],
            "upload_type": ["ShortImport", "ShortImport", "ShortImport"],
            "video_duration": [1000.0, 1000.0, 2000.0],
            "video_width": [720, 720, 1080],
            "video_height": [1280, 1280, 1920],
            "show_cnt": [10, 9999, 5],
            "like_cnt": [1, 8888, 0],
        }
    ).to_csv(tmp_path / "item_daily_features.csv", index=False)
    pd.DataFrame(
        {
            "video_id": [1, 2],
            "manual_cover_text": ["cover", "UNKNOWN"],
            "caption": ["caption", None],
            "topic_tag": ["[]", "sports"],
            "first_level_category_id": [8, 9],
            "second_level_category_id": [10, 11],
            "third_level_category_id": [12, 13],
        }
    ).to_csv(tmp_path / "kuairec_caption_category.csv", index=False)

    features = load_static_item_features(tmp_path, chunksize=1)
    assert features.frame.columns.tolist() == [
        "video_id",
        "caption_text",
        "category_ids",
        "video_duration",
        "video_width",
        "video_height",
        "upload_type",
        "upload_dt",
    ]
    assert "show_cnt" not in features.frame
    assert "like_cnt" not in features.frame
    np.testing.assert_array_equal(features.normal_item_ids, [1])
    assert len(features.variant_static_item_ids) == 0
    assert features.frame.loc[0, "caption_text"] == "caption"
    assert features.frame.loc[1, "caption_text"] == "sports"


def test_static_feature_loader_reports_corrections_and_uses_earliest_row(tmp_path):
    pd.DataFrame(
        {
            "video_id": [1, 1],
            "date": [20200101, 20200102],
            "video_type": ["NORMAL", "NORMAL"],
            "upload_dt": ["2019-12-01", "2019-12-01"],
            "upload_type": ["ShortImport", "ShortImport"],
            "video_duration": [1000.0, 1000.0],
            "video_width": [720, 721],
            "video_height": [1280, 1281],
        }
    ).to_csv(tmp_path / "item_daily_features.csv", index=False)
    pd.DataFrame(
        {
            "video_id": [1],
            "manual_cover_text": ["cover"],
            "caption": ["caption"],
            "topic_tag": ["[]"],
            "first_level_category_id": [8],
            "second_level_category_id": [10],
            "third_level_category_id": [12],
        }
    ).to_csv(tmp_path / "kuairec_caption_category.csv", index=False)

    features = load_static_item_features(tmp_path)
    np.testing.assert_array_equal(features.variant_static_item_ids, [1])
    assert features.frame.loc[0, "video_width"] == 720
    assert features.frame.loc[0, "video_height"] == 1280


def test_selection_gate_targets_the_strongest_baseline_with_numeric_thresholds():
    with open("configs/fully_observed_v1.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    gate = config["selection_gate"]

    assert gate["comparison_baseline"] == (
        "max_recall_at_100_of_global_popularity_and_bpr"
    )
    assert gate["minimum_direct_recall_gain_absolute"] == 0.002
    assert "direct_recall_improvement" not in gate
    assert gate["coverage_tradeoff"] == {
        "maximum_recall_at_100_deficit_absolute": 0.02,
        "minimum_coverage_at_100_gain_absolute": 0.05,
    }
    assert gate["data_cold_tradeoff"]["minimum_target_denominator"] == 100
    assert gate["ndcg_at_20_protection"]["maximum_drop_absolute"] == 0.01
