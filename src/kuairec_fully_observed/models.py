"""Minimal baselines and encoder interfaces for the Phase A retrieval route."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import RetrievalQueries, is_strong_positive


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def _xavier(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    scale = np.sqrt(2.0 / (rows + columns))
    return rng.normal(0.0, scale, size=(rows, columns)).astype(np.float32)


def _rank_scores(
    scores: dict[int, float], queries: RetrievalQueries, *, k: int
) -> np.ndarray:
    output = np.full((len(queries.user_ids), k), -1, dtype=np.int64)
    for row, candidates in enumerate(queries.candidates):
        order = sorted(
            (int(item) for item in candidates),
            key=lambda item: (-scores.get(item, 0.0), item),
        )[:k]
        output[row, : len(order)] = order
    return output


@dataclass(frozen=True)
class PopularityBaseline:
    """Train-frozen strong-positive counts with ascending item-ID ties."""

    scores: dict[int, float]

    @classmethod
    def fit(cls, train_events: pd.DataFrame) -> "PopularityBaseline":
        required = {"video_id", "watch_ratio"}
        if not required.issubset(train_events.columns):
            raise ValueError("Popularity fit requires video_id and watch_ratio")
        positives = train_events.loc[
            is_strong_positive(train_events["watch_ratio"]), "video_id"
        ]
        counts = positives.value_counts()
        return cls({int(item): float(count) for item, count in counts.items()})

    def rank(self, queries: RetrievalQueries, *, k: int = 100) -> np.ndarray:
        return _rank_scores(self.scores, queries, k=k)


@dataclass(frozen=True)
class BPRModel:
    """Exact-retrieval interface for Phase B BPR checkpoints."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray

    def __post_init__(self) -> None:
        if self.user_factors.shape[0] != len(self.user_ids):
            raise ValueError("BPR user IDs and factors differ")
        if self.item_factors.shape[0] != len(self.item_ids):
            raise ValueError("BPR item IDs and factors differ")
        if self.user_factors.shape[1] != self.item_factors.shape[1]:
            raise ValueError("BPR factor dimensions differ")

    def encode_users(self, user_ids: np.ndarray) -> np.ndarray:
        positions = {int(user): index for index, user in enumerate(self.user_ids)}
        try:
            return np.asarray(
                [self.user_factors[positions[int(user)]] for user in user_ids],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise ValueError("BPR has no factor for a query user") from exc

    def encode_items(self) -> np.ndarray:
        return np.asarray(self.item_factors, dtype=np.float32)

    def rank(
        self,
        queries: RetrievalQueries,
        *,
        k: int = 100,
        cold_user_fallback: PopularityBaseline | None = None,
    ) -> np.ndarray:
        """Rank without dropping cold users or items.

        Missing item factors receive a fixed score of zero. A query user with
        no learned factor is routed to the explicitly supplied Popularity
        fallback rather than raising or being removed from evaluation.
        """

        user_positions = {
            int(user): index for index, user in enumerate(self.user_ids)
        }
        item_positions = {
            int(item): index for index, item in enumerate(self.item_ids)
        }
        fallback = (
            None
            if cold_user_fallback is None
            else cold_user_fallback.rank(queries, k=k)
        )
        output = np.full((len(queries.user_ids), k), -1, dtype=np.int64)
        for row, (user, candidates) in enumerate(
            zip(queries.user_ids, queries.candidates, strict=True)
        ):
            user_position = user_positions.get(int(user))
            if user_position is None:
                if fallback is None:
                    raise ValueError(
                        "BPR cold query user requires a Popularity fallback"
                    )
                output[row] = fallback[row]
                continue
            user_vector = self.user_factors[user_position]
            scores = {
                int(item): (
                    float(self.item_factors[item_positions[int(item)]] @ user_vector)
                    if int(item) in item_positions
                    else 0.0
                )
                for item in candidates
            }
            ranked = sorted(scores, key=lambda item: (-scores[item], item))[:k]
            output[row, : len(ranked)] = ranked
        return output


class NumpyTwoTowerReference:
    """Deterministic Phase A reference for the trainable Two-Tower interface.

    It validates feature flow, pooling, shapes and exact retrieval without
    claiming training effectiveness. Phase B will train the same interfaces
    with an autodiff backend after this protocol is reviewed.
    """

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        num_categories: int,
        caption_dim: int,
        static_dim: int,
        output_dim: int = 128,
        seed: int = 20260722,
    ) -> None:
        rng = np.random.default_rng(seed)
        id_dim = 16
        category_dim = 8
        user_dim = 16
        item_input = id_dim + category_dim + caption_dim + static_dim
        self.output_dim = output_dim
        self.item_id_embedding = _xavier(rng, num_items, id_dim)
        self.category_embedding = _xavier(rng, num_categories, category_dim)
        self.user_id_embedding = _xavier(rng, num_users, user_dim)
        self.item_hidden = _xavier(rng, item_input, 64)
        self.item_output = _xavier(rng, 64, output_dim)
        self.user_hidden = _xavier(rng, user_dim + output_dim, 64)
        self.user_output = _xavier(rng, 64, output_dim)

    def encode_items(
        self,
        item_indices: np.ndarray,
        category_indices: np.ndarray,
        caption_embeddings: np.ndarray,
        static_features: np.ndarray,
        *,
        use_id_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        item_indices = np.asarray(item_indices, dtype=np.int64)
        category_indices = np.asarray(category_indices, dtype=np.int64)
        id_vectors = self.item_id_embedding[item_indices].copy()
        if use_id_embedding is not None:
            id_mask = np.asarray(use_id_embedding)
            if id_mask.shape != item_indices.shape or id_mask.dtype != np.bool_:
                raise ValueError("use_id_embedding must be one boolean per item")
            id_vectors[~id_mask] = 0.0
        inputs = np.concatenate(
            (
                id_vectors,
                self.category_embedding[category_indices],
                np.asarray(caption_embeddings, dtype=np.float32),
                np.asarray(static_features, dtype=np.float32),
            ),
            axis=1,
        )
        hidden = np.maximum(inputs @ self.item_hidden, 0.0)
        return _normalize(hidden @ self.item_output)

    def encode_users(
        self,
        user_indices: np.ndarray,
        history_item_vectors: np.ndarray,
        history_weights: np.ndarray,
        padding_mask: np.ndarray,
    ) -> np.ndarray:
        users = np.asarray(user_indices, dtype=np.int64)
        history = np.asarray(history_item_vectors, dtype=np.float32)
        weights = np.asarray(history_weights, dtype=np.float32)
        mask = np.asarray(padding_mask, dtype=bool)
        if history.ndim != 3 or history.shape[:2] != weights.shape:
            raise ValueError("History vectors and weights differ in shape")
        if mask.shape != weights.shape or history.shape[2] != self.output_dim:
            raise ValueError("History mask or embedding dimension differs")
        if np.any(weights < 0):
            raise ValueError("History pooling weights must be nonnegative")
        effective = weights * mask
        pooled = (history * effective[:, :, None]).sum(axis=1)
        pooled /= np.maximum(effective.sum(axis=1, keepdims=True), 1e-12)
        inputs = np.concatenate((self.user_id_embedding[users], pooled), axis=1)
        hidden = np.maximum(inputs @ self.user_hidden, 0.0)
        return _normalize(hidden @ self.user_output)


def in_batch_softmax_loss(
    user_vectors: np.ndarray,
    positive_item_vectors: np.ndarray,
    *,
    temperature: float,
    valid_logit_mask: np.ndarray | None = None,
) -> float:
    """Temperature-scaled cross-entropy with false-negative masking."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    users = np.asarray(user_vectors, dtype=np.float64)
    items = np.asarray(positive_item_vectors, dtype=np.float64)
    if users.shape != items.shape or users.ndim != 2:
        raise ValueError("In-batch user/item vectors need equal rank-2 shapes")
    logits = users @ items.T / temperature
    if valid_logit_mask is not None:
        mask = np.asarray(valid_logit_mask)
        if mask.shape != logits.shape or mask.dtype != np.bool_:
            raise ValueError(
                "valid_logit_mask must be boolean with batch-square shape"
            )
        if not np.all(np.diag(mask)):
            raise ValueError("Every diagonal positive must remain valid")
        logits = np.where(mask, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(logits).sum(axis=1))
    return float(np.mean(log_partition - np.diag(logits)))


def stable_random_rank(
    queries: RetrievalQueries, *, seed: int = 20260722, k: int = 100
) -> np.ndarray:
    """Deterministic Random sanity baseline, independent of process hash state."""

    output = np.full((len(queries.user_ids), k), -1, dtype=np.int64)
    for row, (user, candidates) in enumerate(
        zip(queries.user_ids, queries.candidates, strict=True)
    ):
        ranked = sorted(
            (int(item) for item in candidates),
            key=lambda item: (
                hashlib.sha256(f"{seed}|{int(user)}|{item}".encode()).digest(),
                item,
            ),
        )[:k]
        output[row, : len(ranked)] = ranked
    return output
