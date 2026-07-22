from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from kuairec_phase1.baselines import (
    load_artifacts,
    rank_bpr,
    rank_causal_streaming_decayed,
    rank_fit_frozen_decayed,
    rank_global_popularity,
    rank_itemcf,
    rank_random,
    train_bpr_checkpoints,
)
from kuairec_phase1.metrics import common_bootstrap_indices, evaluate_topk


def _write_fixture(root: Path) -> dict:
    root.mkdir()
    video_ids = np.arange(10, 16, dtype=np.int32)
    train_counts = np.asarray([3, 2, 1, 0, 0, 0], dtype=np.int64)
    np.savez_compressed(
        root / "catalog.npz",
        video_ids=video_ids,
        dates=np.asarray([20200101], dtype=np.int32),
        eligible_bits=np.asarray([[0b00111111]], dtype=np.uint8),
        train_counts=train_counts,
        warm=train_counts > 0,
        head=np.asarray([True, True, False, False, False, False]),
        tail=np.asarray([False, False, True, False, False, False]),
        cold=train_counts == 0,
        train_end=np.asarray([100.0]),
        validation_end=np.asarray([200.0]),
    )
    np.savez_compressed(
        root / "events_train_validation.npz",
        user_ids=np.asarray([1, 2], dtype=np.int32),
        user=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        item=np.asarray([0, 1, 3, 0, 2, 4], dtype=np.int32),
        timestamp=np.asarray([10, 20, 120, 15, 30, 130], dtype=np.float64),
        strong=np.asarray([True, True, True, True, True, True]),
        user_indptr=np.asarray([0, 3, 6], dtype=np.int64),
    )
    np.savez_compressed(
        root / "targets_train.npz",
        user=np.asarray([0, 0, 1, 1], dtype=np.int32),
        item=np.asarray([0, 1, 0, 2], dtype=np.int32),
        timestamp=np.asarray([10, 20, 15, 30], dtype=np.float64),
    )
    np.savez_compressed(
        root / "queries_validation.npz",
        user=np.asarray([0, 1], dtype=np.int32),
        user_ids=np.asarray([1, 2], dtype=np.int32),
        timestamp=np.asarray([120, 130], dtype=np.float64),
        target_indptr=np.asarray([0, 1, 2], dtype=np.int64),
        target_indices=np.asarray([3, 4], dtype=np.int32),
        candidate_count=np.asarray([4, 4], dtype=np.int32),
        candidate_union_count=np.asarray([6], dtype=np.int32),
    )
    candidates = np.asarray([[0b00111100], [0b00111010]], dtype=np.uint8)
    np.save(root / "candidate_bits_validation.npy", candidates)
    user_item = sparse.csr_matrix(
        (
            np.ones(4, dtype=np.float32),
            (np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 2])),
        ),
        shape=(2, 6),
    )
    sparse.save_npz(root / "itemcf_user_item.npz", user_item)
    cooc = (user_item.T @ user_item).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    sparse.save_npz(root / "itemcf_cooccurrence.npz", cooc)
    np.savez_compressed(
        root / "bpr_negative_indices.npz",
        seed_20260721=np.asarray([2, 2, 1, 1], dtype=np.int32),
        seed_20260722=np.asarray([2, 2, 1, 1], dtype=np.int32),
        seed_20260723=np.asarray([2, 2, 1, 1], dtype=np.int32),
    )
    return load_artifacts(root)


def _causal_streaming_fixture(
    candidate_bits: np.ndarray | None = None,
) -> dict:
    """Four deterministic queries around two validation timestamps.

    Item 0 has one train positive. Item 2 receives its first positive at
    timestamp 110; items 1 and 3 receive positives together at timestamp 120.
    The tiny timestamps keep decay deterministic without changing the intended
    score ordering.
    """

    if candidate_bits is None:
        candidate_bits = np.full((4, 1), 0b00001111, dtype=np.uint8)
    return {
        "catalog": {
            "video_ids": np.asarray([10, 20, 30, 40], dtype=np.int32),
            "train_end": np.asarray([100.0], dtype=np.float64),
        },
        "train": {
            "item": np.asarray([0], dtype=np.int32),
            "timestamp": np.asarray([100.0], dtype=np.float64),
        },
        "queries": {
            "user": np.asarray([0, 1, 2, 3], dtype=np.int32),
            "timestamp": np.asarray([110.0, 120.0, 120.0, 130.0]),
            "target_indptr": np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            "target_indices": np.asarray([2, 1, 3, 0], dtype=np.int32),
        },
        "candidate_bits": candidate_bits,
    }


def _ranked_items(row: np.ndarray) -> list[int]:
    return row[row >= 0].astype(int).tolist()


def test_causal_streaming_future_feedback_does_not_change_past_and_cold_starts_zero():
    ranked = rank_causal_streaming_decayed(_causal_streaming_fixture(), 1)

    # Before item 2's first validation positive, only train-warm item 0 has a
    # nonzero score. Future positives for items 1, 2, and 3 cannot rewrite it.
    assert _ranked_items(ranked[0]) == [0, 1, 2, 3]

    # Item 2 was train-cold, but after its timestamp-110 positive it may gain
    # dynamic popularity and therefore leads both timestamp-120 rankings.
    assert _ranked_items(ranked[1]) == [2, 0, 1, 3]


def test_causal_streaming_scores_equal_timestamp_queries_before_any_update():
    ranked = rank_causal_streaming_decayed(_causal_streaming_fixture(), 1)

    # The two timestamp-120 targets are different, but neither query can use
    # the other's feedback because the entire timestamp is scored atomically.
    assert _ranked_items(ranked[1]) == [2, 0, 1, 3]
    assert _ranked_items(ranked[2]) == [2, 0, 1, 3]


def test_causal_streaming_earlier_feedback_changes_later_ranking():
    ranked = rank_causal_streaming_decayed(_causal_streaming_fixture(), 1)

    # Once timestamp 120 is complete, its item-1 and item-3 positives are
    # available to timestamp 130. Their equal scores use video ID as tie-break.
    assert _ranked_items(ranked[3]) == [1, 3, 2, 0]


def test_causal_streaming_respects_each_query_candidate_membership():
    # Candidate sets by row: {0,2}, {1,2}, {0,3}, and {1,3}.
    candidate_bits = np.asarray(
        [[0b00000101], [0b00000110], [0b00001001], [0b00001010]],
        dtype=np.uint8,
    )
    ranked = rank_causal_streaming_decayed(
        _causal_streaming_fixture(candidate_bits), 1
    )

    assert [_ranked_items(row) for row in ranked] == [
        [0, 2],
        [2, 1],
        [0, 3],
        [1, 3],
    ]


def test_all_five_baseline_paths_share_candidates_and_metrics(tmp_path):
    artifacts = _write_fixture(tmp_path / "artifacts")
    outputs = [
        rank_random(artifacts, 20260721),
        rank_global_popularity(artifacts),
        rank_fit_frozen_decayed(artifacts, 7),
        rank_causal_streaming_decayed(artifacts, 7),
        rank_itemcf(artifacts, neighbor_count=50, shrinkage=10),
    ]
    checkpoints = train_bpr_checkpoints(
        artifacts,
        embedding_dim=4,
        learning_rate=0.005,
        l2=0.0001,
        seed=20260721,
        checkpoints=(1,),
    )
    bpr, fallback = rank_bpr(
        artifacts,
        *checkpoints[1],
        fallback_topk=outputs[2],
    )
    outputs.append(bpr)
    assert fallback == {"fallback_user_count": 0, "fallback_query_count": 0}
    users, bootstrap = common_bootstrap_indices(
        artifacts["queries"]["user"], replicates=20
    )
    for output in outputs:
        assert output.shape == (2, 100)
        for query in range(2):
            allowed = set(
                np.flatnonzero(
                    np.unpackbits(
                        artifacts["candidate_bits"][query], bitorder="little"
                    )[:6]
                )
            )
            assert set(output[query][output[query] >= 0]).issubset(allowed)
        evaluated = evaluate_topk(
            topk=output,
            query_users=artifacts["queries"]["user"],
            target_indptr=artifacts["queries"]["target_indptr"],
            target_indices=artifacts["queries"]["target_indices"],
            candidate_union_count=6,
            candidate_score_count=8,
            warm_mask=artifacts["catalog"]["warm"],
            tail_mask=artifacts["catalog"]["tail"],
            cold_mask=artifacts["catalog"]["cold"],
            bootstrap_users=users,
            bootstrap_indices=bootstrap,
        )
        assert evaluated["denominators"]["query_count"] == 2
        assert 0 <= evaluated["metrics"]["Recall@100"] <= 1


def test_user_cluster_bootstrap_preserves_query_macro_estimator():
    # User 0 contributes three misses and user 1 contributes one hit.  The
    # contract's primary query-macro Recall is therefore 1/4; an incorrect
    # unweighted mean of the two per-user means would be 1/2.
    topk = np.full((4, 100), -1, dtype=np.int32)
    topk[3, 0] = 3
    evaluated = evaluate_topk(
        topk=topk,
        query_users=np.asarray([0, 0, 0, 1], dtype=np.int32),
        target_indptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        target_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
        candidate_union_count=4,
        candidate_score_count=16,
        warm_mask=np.ones(4, dtype=bool),
        tail_mask=np.zeros(4, dtype=bool),
        cold_mask=np.zeros(4, dtype=bool),
        bootstrap_users=np.asarray([0, 1], dtype=np.int32),
        bootstrap_indices=np.tile(
            np.asarray([[0, 1]], dtype=np.int32), (20, 1)
        ),
    )
    assert evaluated["metrics"]["Recall@100"] == 0.25
    assert evaluated["bootstrap_95_percent_intervals"]["Recall@100"] == [
        0.25,
        0.25,
    ]
