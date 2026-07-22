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
