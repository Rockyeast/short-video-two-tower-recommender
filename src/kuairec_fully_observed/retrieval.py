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
    ) -> np.ndarray:
        users = np.asarray(user_vectors, dtype=np.float32)
        items = np.asarray(item_vectors, dtype=np.float32)
        ids = np.asarray(item_ids, dtype=np.int64)
        if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
            raise ValueError("User and item vectors need matching rank-2 dimensions")
        if len(users) != len(candidates) or len(items) != len(ids):
            raise ValueError("Retriever inputs differ in row count")
        if k <= 0:
            raise ValueError("k must be positive")
        positions = {int(item): index for index, item in enumerate(ids)}
        output = np.full((len(users), k), -1, dtype=np.int64)
        for row, candidate_ids in enumerate(candidates):
            try:
                candidate_positions = np.asarray(
                    [positions[int(item)] for item in candidate_ids], dtype=np.int64
                )
            except KeyError as exc:
                raise ValueError("Candidate has no item vector") from exc
            scores = items[candidate_positions] @ users[row]
            order = np.lexsort((candidate_ids, -scores))[:k]
            ranked = np.asarray(candidate_ids, dtype=np.int64)[order]
            output[row, : len(ranked)] = ranked
        return output
