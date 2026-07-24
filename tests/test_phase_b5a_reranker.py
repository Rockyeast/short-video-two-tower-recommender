from __future__ import annotations

import copy

import numpy as np
import pytest

from kuairec_fully_observed.data import RetrievalQueries
from kuairec_fully_observed.reranking import (
    RerankFeatureBuilder,
    rank_with_model,
    reranker_gate,
    stable_fit_mask,
    train_lightgbm_ranker,
)
from scripts.run_phase_b5a_lightgbm_reranker import (
    EXPECTED_CONFIG,
    validate_config,
)


def _fixture() -> tuple[RerankFeatureBuilder, np.ndarray]:
    queries = RetrievalQueries(
        user_ids=np.asarray([10, 20], dtype=np.int64),
        histories=(
            np.asarray([1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        ),
        history_weights=(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        ),
        candidates=(
            np.asarray([3, 4, 5], dtype=np.int64),
            np.asarray([3, 4, 5], dtype=np.int64),
        ),
        relevant=(
            np.asarray([4], dtype=np.int64),
            np.asarray([5], dtype=np.int64),
        ),
        catalog=np.asarray([3, 4, 5], dtype=np.int64),
        warm_user_mask=np.asarray([True, True]),
    )
    builder = RerankFeatureBuilder(
        queries=queries,
        two_tower_topk=np.asarray([[4, 3, 5], [5, 4, 3]], dtype=np.int64),
        bpr_topk=np.asarray([[3, 4, 5], [4, 5, 3]], dtype=np.int64),
        two_tower_user_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        two_tower_catalog_vectors=np.asarray(
            [[0.1, 0.1], [0.9, 0.2], [0.2, 0.9]]
        ),
        bpr_user_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        bpr_catalog_vectors=np.asarray(
            [[0.8, 0.1], [0.4, 0.8], [0.1, 0.7]]
        ),
        popularity_scores=np.asarray([10.0, 2.0, 1.0]),
        catalog_categories=np.asarray([[1, -1, -1], [2, -1, -1], [3, -1, -1]]),
        category_lookup={1: (2,), 2: (3,)},
        data_cold_mask=np.asarray([False, True, False]),
        video_duration=np.asarray([5.0, 10.0, 20.0]),
    )
    return builder, np.asarray([0, 1], dtype=np.int64)


def test_stable_user_split_is_deterministic_and_order_independent() -> None:
    users = np.arange(100, 300, dtype=np.int64)
    first = stable_fit_mask(users, salt="fixed", fit_percent=70)
    reverse = stable_fit_mask(users[::-1], salt="fixed", fit_percent=70)
    assert np.array_equal(first, reverse[::-1])
    assert 100 < int(first.sum()) < 180


def test_feature_builder_labels_union_and_history_affinity() -> None:
    builder, indices = _fixture()
    dataset = builder.build(
        indices, training_negative_cap=None, require_retrieved_positive=True
    )
    assert dataset.group_sizes.tolist() == [3, 3]
    assert dataset.labels.tolist() == [0, 1, 0, 0, 0, 1]
    assert dataset.item_ids[:3].tolist() == [3, 4, 5]
    # User 10 watched category 2, so item 4 has full category affinity.
    assert dataset.features[1, 6] == pytest.approx(1.0)
    assert dataset.features[1, 7] == pytest.approx(1.0)


def test_negative_cap_keeps_every_retrieved_positive() -> None:
    builder, indices = _fixture()
    dataset = builder.build(
        indices, training_negative_cap=1, require_retrieved_positive=True
    )
    assert dataset.group_sizes.tolist() == [2, 2]
    assert dataset.labels.sum() == 2


def test_lightgbm_fit_and_rank_on_hand_checkable_groups() -> None:
    builder, indices = _fixture()
    dataset = builder.build(
        indices, training_negative_cap=None, require_retrieved_positive=True
    )
    model = train_lightgbm_ranker(
        dataset,
        parameters={
            "objective": "lambdarank",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "random_state": 9,
            "n_jobs": 1,
            "verbosity": -1,
        },
    )
    ranked = rank_with_model(model, dataset, output_k=3)
    assert ranked.shape == (2, 3)
    assert set(ranked[0]) == {3, 4, 5}
    assert set(ranked[1]) == {3, 4, 5}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("split", "fit_percent"), 71),
        (("model", "n_estimators"), 151),
        (("evaluation", "output_k"), 99),
        (("claims", "no_hyperparameter_search"), False),
    ],
)
def test_frozen_config_rejects_mutations(
    path: tuple[str, str], value: object
) -> None:
    changed = copy.deepcopy(EXPECTED_CONFIG)
    changed[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="frozen configuration"):
        validate_config(changed)


def test_gate_requires_ndcg_and_retained_recall_coverage() -> None:
    baseline = {
        "NDCG@20": 0.10,
        "Recall@100": 0.20,
        "Coverage@100": 0.30,
    }
    passed = reranker_gate(
        reranker_metrics={
            "NDCG@20": 0.11,
            "Recall@100": 0.198,
            "Coverage@100": 0.28,
        },
        hybrid_metrics=baseline,
        recall_retention=0.98,
        coverage_retention=0.90,
    )
    assert passed["passed"] is True
    failed = reranker_gate(
        reranker_metrics={
            "NDCG@20": 0.10,
            "Recall@100": 0.198,
            "Coverage@100": 0.28,
        },
        hybrid_metrics=baseline,
        recall_retention=0.98,
        coverage_retention=0.90,
    )
    assert failed["passed"] is False
