"""Exact retrieval used before any approximate index is introduced."""

from __future__ import annotations

import numpy as np


class ExactDotProductRetriever:
    """Rank fixed per-user candidates by dot product with stable item-ID ties."""

    def search(
        self,
        user_vectors: np.ndarray,
        item_vectors: np.ndarray,
        *,
        item_ids: np.ndarray,
        candidates: tuple[np.ndarray, ...],
        k: int = 100,
        warm_user_mask: np.ndarray | None = None,
        fallback_topk: np.ndarray | None = None,
        missing_item_score: float | None = None,
        score_block_size: int = 256,
    ) -> np.ndarray:
        """Score query blocks with one matrix multiplication per block.

        ``warm_user_mask`` and ``fallback_topk`` form the shared cold-user
        route for BPR and Two-Tower. ``missing_item_score`` lets BPR assign a
        fixed zero to candidates without trained factors while Two-Tower can
        continue to require a content vector for every candidate.
        """

        users = np.asarray(user_vectors, dtype=np.float32)
        items = np.asarray(item_vectors, dtype=np.float32)
        ids = np.asarray(item_ids, dtype=np.int64)
        if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
            raise ValueError("User and item vectors need matching rank-2 dimensions")
        if len(users) != len(candidates) or len(items) != len(ids):
            raise ValueError("Retriever inputs differ in row count")
        if k <= 0:
            raise ValueError("k must be positive")
        if score_block_size <= 0:
            raise ValueError("score_block_size must be positive")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("item_ids must be unique")
        warm = (
            np.ones(len(users), dtype=bool)
            if warm_user_mask is None
            else np.asarray(warm_user_mask)
        )
        if warm.shape != (len(users),) or warm.dtype != np.bool_:
            raise ValueError("warm_user_mask must be one boolean per query")
        fallback = None if fallback_topk is None else np.asarray(fallback_topk)
        if fallback is not None and fallback.shape != (len(users), k):
            raise ValueError("fallback_topk must match query count and K")
        if np.any(~warm) and fallback is None:
            raise ValueError("Cold query users require fallback_topk")
        positions = {int(item): index for index, item in enumerate(ids)}
        output = np.full((len(users), k), -1, dtype=np.int64)
        if fallback is not None:
            output[~warm] = fallback[~warm]
        warm_rows = np.flatnonzero(warm)
        for begin in range(0, len(warm_rows), score_block_size):
            rows = warm_rows[begin : begin + score_block_size]
            score_block = users[rows] @ items.T
            for local_row, row in enumerate(rows):
                candidate_ids = np.asarray(candidates[int(row)], dtype=np.int64)
                candidate_positions = np.fromiter(
                    (positions.get(int(item), -1) for item in candidate_ids),
                    dtype=np.int64,
                    count=len(candidate_ids),
                )
                missing = candidate_positions < 0
                if np.any(missing) and missing_item_score is None:
                    raise ValueError("Candidate has no item vector")
                scores = np.full(
                    len(candidate_ids),
                    0.0 if missing_item_score is None else missing_item_score,
                    dtype=np.float32,
                )
                scores[~missing] = score_block[
                    local_row, candidate_positions[~missing]
                ]
                order = np.lexsort((candidate_ids, -scores))[:k]
                ranked = candidate_ids[order]
                output[int(row), : len(ranked)] = ranked
        return output
