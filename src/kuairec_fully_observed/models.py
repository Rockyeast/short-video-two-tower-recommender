"""Minimal baselines and encoder interfaces for the Phase A retrieval route."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import RetrievalQueries, is_strong_positive
from .retrieval import ExactDotProductRetriever
from .training import BPRTrainingDataset


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
        score_block_size: int = 256,
    ) -> np.ndarray:
        """Rank without dropping cold users or items.

        Missing item factors receive a fixed score of zero. A query user with
        no learned factor is routed to the explicitly supplied Popularity
        fallback rather than raising or being removed from evaluation.
        """

        user_positions = {int(user): index for index, user in enumerate(self.user_ids)}
        user_vectors = np.zeros(
            (len(queries.user_ids), self.user_factors.shape[1]), dtype=np.float32
        )
        learned_user = np.zeros(len(queries.user_ids), dtype=bool)
        for row, user in enumerate(queries.user_ids):
            position = user_positions.get(int(user))
            if position is not None:
                user_vectors[row] = self.user_factors[position]
                learned_user[row] = True
        warm = learned_user & queries.warm_user_mask
        fallback_topk = (
            None
            if cold_user_fallback is None
            else cold_user_fallback.rank(queries, k=k)
        )
        return ExactDotProductRetriever().search(
            user_vectors,
            self.item_factors,
            item_ids=self.item_ids,
            candidates=queries.candidates,
            k=k,
            warm_user_mask=warm,
            fallback_topk=fallback_topk,
            missing_item_score=0.0,
            score_block_size=score_block_size,
        )


@dataclass(frozen=True)
class BPRTrainingResult:
    model: BPRModel
    epoch_losses: tuple[float, ...]


def _sgd_indexed_update(
    matrix: np.ndarray,
    indices: np.ndarray,
    gradients: np.ndarray,
    *,
    learning_rate: float,
    normalization: int,
) -> None:
    unique, inverse = np.unique(indices, return_inverse=True)
    accumulated = np.zeros((len(unique), matrix.shape[1]), dtype=np.float32)
    np.add.at(accumulated, inverse, gradients)
    matrix[unique] -= learning_rate * accumulated / float(normalization)


def train_bpr_sgd(
    dataset: BPRTrainingDataset,
    *,
    embedding_dim: int = 64,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 4096,
) -> BPRTrainingResult:
    """Train the minimal BPR baseline with epoch-resampled negatives."""

    if embedding_dim <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid BPR optimization hyperparameters")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    user_ids = np.unique(dataset.user_ids)
    item_ids = np.asarray(dataset.negative_catalog, dtype=np.int64)
    user_positions = {int(user): index for index, user in enumerate(user_ids)}
    item_positions = {int(item): index for index, item in enumerate(item_ids)}
    positive_users = np.fromiter(
        (user_positions[int(user)] for user in dataset.user_ids),
        dtype=np.int64,
        count=len(dataset.user_ids),
    )
    positive_items = np.fromiter(
        (item_positions[int(item)] for item in dataset.positive_item_ids),
        dtype=np.int64,
        count=len(dataset.positive_item_ids),
    )
    rng = np.random.default_rng(dataset.seed)
    users = rng.normal(0.0, 0.05, (len(user_ids), embedding_dim)).astype(np.float32)
    items = rng.normal(0.0, 0.05, (len(item_ids), embedding_dim)).astype(np.float32)
    touched_items = np.zeros(len(item_ids), dtype=bool)
    order = np.arange(len(positive_users), dtype=np.int64)
    losses: list[float] = []
    for epoch in range(epochs):
        negative_ids = dataset.sample_negatives(epoch)
        negative_items = np.fromiter(
            (item_positions[int(item)] for item in negative_ids),
            dtype=np.int64,
            count=len(negative_ids),
        )
        touched_items[positive_items] = True
        touched_items[negative_items] = True
        rng.shuffle(order)
        total_loss = 0.0
        total_examples = 0
        for begin in range(0, len(order), batch_size):
            batch = order[begin : begin + batch_size]
            user_index = positive_users[batch]
            positive_index = positive_items[batch]
            negative_index = negative_items[batch]
            user_vector = users[user_index].copy()
            positive_vector = items[positive_index].copy()
            negative_vector = items[negative_index].copy()
            difference = positive_vector - negative_vector
            score = np.sum(user_vector * difference, axis=1)
            coefficient = 1.0 / (1.0 + np.exp(np.clip(score, -30.0, 30.0)))
            user_gradient = -coefficient[:, None] * difference + l2 * user_vector
            positive_gradient = (
                -coefficient[:, None] * user_vector + l2 * positive_vector
            )
            negative_gradient = (
                coefficient[:, None] * user_vector + l2 * negative_vector
            )
            normalization = len(batch)
            _sgd_indexed_update(
                users,
                user_index,
                user_gradient,
                learning_rate=learning_rate,
                normalization=normalization,
            )
            _sgd_indexed_update(
                items,
                positive_index,
                positive_gradient,
                learning_rate=learning_rate,
                normalization=normalization,
            )
            _sgd_indexed_update(
                items,
                negative_index,
                negative_gradient,
                learning_rate=learning_rate,
                normalization=normalization,
            )
            total_loss += float(np.logaddexp(0.0, -score).sum())
            total_examples += len(batch)
        losses.append(total_loss / total_examples)
    return BPRTrainingResult(
        model=BPRModel(
            user_ids=user_ids,
            item_ids=item_ids[touched_items],
            user_factors=users,
            item_factors=items[touched_items],
        ),
        epoch_losses=tuple(losses),
    )


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

    def rank_encoded(
        self,
        query_user_vectors: np.ndarray,
        item_vectors: np.ndarray,
        *,
        item_ids: np.ndarray,
        queries: RetrievalQueries,
        learned_user_mask: np.ndarray,
        cold_user_fallback: PopularityBaseline,
        k: int = 100,
        score_block_size: int = 256,
    ) -> np.ndarray:
        """Use the same blocked cold-user route as BPR after encoding."""

        learned = np.asarray(learned_user_mask)
        if learned.shape != (len(queries.user_ids),) or learned.dtype != np.bool_:
            raise ValueError("learned_user_mask must be one boolean per query")
        return ExactDotProductRetriever().search(
            query_user_vectors,
            item_vectors,
            item_ids=item_ids,
            candidates=queries.candidates,
            k=k,
            warm_user_mask=learned & queries.warm_user_mask,
            fallback_topk=cold_user_fallback.rank(queries, k=k),
            score_block_size=score_block_size,
        )


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
