"""Single-configuration LightGBM reranking over frozen retrieval routes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from .data import RetrievalQueries


RERANK_FEATURE_NAMES = (
    "two_tower_score",
    "bpr_score",
    "two_tower_reciprocal_rank",
    "bpr_reciprocal_rank",
    "frozen_hybrid_rrf_score",
    "log1p_train_popularity",
    "history_category_affinity",
    "data_cold_item",
    "log1p_history_length",
    "log1p_video_duration",
)


def stable_fit_mask(
    user_ids: np.ndarray,
    *,
    salt: str,
    fit_percent: int,
) -> np.ndarray:
    """Split users with a stable hash, independent of row order."""

    if not salt or not 1 <= fit_percent <= 99:
        raise ValueError("Stable split salt/percentage is invalid")
    users = np.asarray(user_ids, dtype=np.int64)
    buckets = np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(f"{salt}:{int(user)}".encode()).digest()[:8],
                "big",
            )
            % 100
            for user in users
        ),
        dtype=np.int64,
        count=len(users),
    )
    return buckets < fit_percent


def subset_queries(
    queries: RetrievalQueries,
    indices: np.ndarray,
) -> RetrievalQueries:
    selected = np.asarray(indices, dtype=np.int64)
    return RetrievalQueries(
        user_ids=queries.user_ids[selected],
        histories=tuple(queries.histories[int(index)] for index in selected),
        history_weights=tuple(
            queries.history_weights[int(index)] for index in selected
        ),
        candidates=tuple(
            queries.candidates[int(index)] for index in selected
        ),
        relevant=tuple(queries.relevant[int(index)] for index in selected),
        catalog=queries.catalog,
        warm_user_mask=queries.warm_user_mask[selected],
        diagnostics=dict(queries.diagnostics),
    )


def _valid_ranked(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.int64)
    valid = values[values >= 0]
    if (
        np.any(values[: len(valid)] < 0)
        or np.any(values[len(valid) :] >= 0)
        or len(np.unique(valid)) != len(valid)
    ):
        raise ValueError("Ranked row has invalid padding or duplicates")
    return valid


@dataclass(frozen=True)
class RerankDataset:
    features: np.ndarray
    labels: np.ndarray
    group_sizes: np.ndarray
    item_ids: np.ndarray
    query_indices: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.features)
        if (
            self.features.shape != (rows, len(RERANK_FEATURE_NAMES))
            or self.labels.shape != (rows,)
            or self.item_ids.shape != (rows,)
            or self.query_indices.ndim != 1
            or self.group_sizes.shape != self.query_indices.shape
            or int(self.group_sizes.sum()) != rows
        ):
            raise ValueError("Rerank dataset arrays do not align")
        if not np.isfinite(self.features).all():
            raise ValueError("Rerank features must be finite")
        if not set(np.unique(self.labels)).issubset({0, 1}):
            raise ValueError("Rerank labels must be binary")


@dataclass(frozen=True)
class RerankFeatureBuilder:
    """Build point-in-time-safe features aligned to the frozen catalog."""

    queries: RetrievalQueries
    two_tower_topk: np.ndarray
    bpr_topk: np.ndarray
    two_tower_user_vectors: np.ndarray
    two_tower_catalog_vectors: np.ndarray
    bpr_user_vectors: np.ndarray
    bpr_catalog_vectors: np.ndarray
    popularity_scores: np.ndarray
    catalog_categories: np.ndarray
    category_lookup: dict[int, tuple[int, ...]]
    data_cold_mask: np.ndarray
    video_duration: np.ndarray
    alpha: float = 0.75
    rank_constant: int = 60
    _catalog_positions: dict[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        query_count = len(self.queries.user_ids)
        catalog_count = len(self.queries.catalog)
        if (
            self.two_tower_topk.shape != self.bpr_topk.shape
            or self.two_tower_topk.shape[0] != query_count
            or self.two_tower_user_vectors.shape[0] != query_count
            or self.bpr_user_vectors.shape[0] != query_count
            or self.two_tower_catalog_vectors.shape[0] != catalog_count
            or self.bpr_catalog_vectors.shape[0] != catalog_count
            or self.popularity_scores.shape != (catalog_count,)
            or self.catalog_categories.shape != (catalog_count, 3)
            or self.data_cold_mask.shape != (catalog_count,)
            or self.video_duration.shape != (catalog_count,)
        ):
            raise ValueError("Rerank feature inputs do not align")
        if not 0.0 <= self.alpha <= 1.0 or self.rank_constant <= 0:
            raise ValueError("RRF configuration is invalid")
        positions = {
            int(item): position
            for position, item in enumerate(self.queries.catalog)
        }
        if len(positions) != catalog_count:
            raise ValueError("Catalog item IDs must be unique")
        object.__setattr__(self, "_catalog_positions", positions)

    def _query_features(
        self,
        query_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        two = _valid_ranked(self.two_tower_topk[query_index])
        bpr = _valid_ranked(self.bpr_topk[query_index])
        candidate_membership = set(
            int(item) for item in self.queries.candidates[query_index]
        )
        if not set(two).issubset(candidate_membership) or not set(
            bpr
        ).issubset(candidate_membership):
            raise ValueError("Retrieved item is outside query candidates")
        items = np.union1d(two, bpr).astype(np.int64)
        catalog_rows = np.asarray(
            [self._catalog_positions[int(item)] for item in items],
            dtype=np.int64,
        )
        two_rank = {int(item): rank for rank, item in enumerate(two, 1)}
        bpr_rank = {int(item): rank for rank, item in enumerate(bpr, 1)}
        two_rr = np.asarray(
            [
                0.0
                if int(item) not in two_rank
                else 1.0 / (self.rank_constant + two_rank[int(item)])
                for item in items
            ],
            dtype=np.float32,
        )
        bpr_rr = np.asarray(
            [
                0.0
                if int(item) not in bpr_rank
                else 1.0 / (self.rank_constant + bpr_rank[int(item)])
                for item in items
            ],
            dtype=np.float32,
        )
        history_categories = {
            category
            for item in self.queries.histories[query_index]
            for category in self.category_lookup.get(int(item), ())
            if category >= 0
        }
        affinity = np.zeros(len(items), dtype=np.float32)
        if history_categories:
            for row, categories in enumerate(
                self.catalog_categories[catalog_rows]
            ):
                valid = {int(value) for value in categories if value >= 0}
                if valid:
                    affinity[row] = len(valid & history_categories) / len(
                        valid
                    )
        features = np.column_stack(
            (
                self.two_tower_catalog_vectors[catalog_rows]
                @ self.two_tower_user_vectors[query_index],
                self.bpr_catalog_vectors[catalog_rows]
                @ self.bpr_user_vectors[query_index],
                two_rr,
                bpr_rr,
                self.alpha * two_rr + (1.0 - self.alpha) * bpr_rr,
                np.log1p(
                    np.maximum(self.popularity_scores[catalog_rows], 0.0)
                ),
                affinity,
                self.data_cold_mask[catalog_rows].astype(np.float32),
                np.full(
                    len(items),
                    np.log1p(len(self.queries.histories[query_index])),
                    dtype=np.float32,
                ),
                np.log1p(
                    np.maximum(self.video_duration[catalog_rows], 0.0)
                ),
            )
        ).astype(np.float32)
        relevant = set(
            int(item) for item in self.queries.relevant[query_index]
        )
        labels = np.asarray(
            [int(int(item) in relevant) for item in items], dtype=np.int8
        )
        return items, features, labels

    def build(
        self,
        query_indices: np.ndarray,
        *,
        training_negative_cap: int | None,
        require_retrieved_positive: bool,
    ) -> RerankDataset:
        if training_negative_cap is not None and training_negative_cap <= 0:
            raise ValueError("Training negative cap must be positive")
        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        item_rows: list[np.ndarray] = []
        accepted_queries: list[int] = []
        group_sizes: list[int] = []
        rrf_index = RERANK_FEATURE_NAMES.index("frozen_hybrid_rrf_score")
        for query_index in np.asarray(query_indices, dtype=np.int64):
            items, features, labels = self._query_features(int(query_index))
            if require_retrieved_positive and not labels.any():
                continue
            if training_negative_cap is not None:
                positive = np.flatnonzero(labels)
                negative = np.flatnonzero(~labels.astype(bool))
                negative_order = np.lexsort(
                    (
                        items[negative],
                        -features[negative, rrf_index],
                    )
                )
                selected = np.concatenate(
                    (
                        positive,
                        negative[negative_order[:training_negative_cap]],
                    )
                )
                selected = selected[
                    np.lexsort(
                        (
                            items[selected],
                            -features[selected, rrf_index],
                        )
                    )
                ]
                items = items[selected]
                features = features[selected]
                labels = labels[selected]
            feature_rows.append(features)
            label_rows.append(labels)
            item_rows.append(items)
            accepted_queries.append(int(query_index))
            group_sizes.append(len(items))
        if not feature_rows:
            raise ValueError("Rerank dataset contains no accepted query")
        return RerankDataset(
            features=np.concatenate(feature_rows),
            labels=np.concatenate(label_rows),
            group_sizes=np.asarray(group_sizes, dtype=np.int32),
            item_ids=np.concatenate(item_rows),
            query_indices=np.asarray(accepted_queries, dtype=np.int64),
        )


def train_lightgbm_ranker(
    dataset: RerankDataset,
    *,
    parameters: dict[str, object],
) -> lgb.LGBMRanker:
    if np.any(
        np.add.reduceat(
            dataset.labels,
            np.r_[0, np.cumsum(dataset.group_sizes)[:-1]],
        )
        <= 0
    ):
        raise ValueError("Every training group needs a retrieved positive")
    model = lgb.LGBMRanker(**parameters)
    model.fit(
        dataset.features,
        dataset.labels,
        group=dataset.group_sizes,
        feature_name=list(RERANK_FEATURE_NAMES),
    )
    return model


def rank_with_model(
    model: lgb.LGBMRanker,
    dataset: RerankDataset,
    *,
    output_k: int,
) -> np.ndarray:
    if output_k <= 0:
        raise ValueError("output_k must be positive")
    predictions = np.asarray(
        model.booster_.predict(dataset.features), dtype=np.float64
    )
    if predictions.shape != (len(dataset.features),) or not np.isfinite(
        predictions
    ).all():
        raise RuntimeError("Reranker produced invalid scores")
    output = np.full(
        (len(dataset.group_sizes), output_k), -1, dtype=np.int64
    )
    offset = 0
    for row, size in enumerate(dataset.group_sizes):
        end = offset + int(size)
        items = dataset.item_ids[offset:end]
        scores = predictions[offset:end]
        order = np.lexsort((items, -scores))[:output_k]
        output[row, : len(order)] = items[order]
        offset = end
    return output


def reranker_gate(
    *,
    reranker_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
    recall_retention: float,
    coverage_retention: float,
) -> dict[str, object]:
    if not 0.0 < recall_retention <= 1.0 or not (
        0.0 < coverage_retention <= 1.0
    ):
        raise ValueError("Reranker retention gates are invalid")
    checks = {
        "ndcg20_strictly_higher": (
            reranker_metrics["NDCG@20"] > hybrid_metrics["NDCG@20"]
        ),
        "recall100_retained": (
            reranker_metrics["Recall@100"]
            >= hybrid_metrics["Recall@100"] * recall_retention
        ),
        "coverage100_retained": (
            reranker_metrics["Coverage@100"]
            >= hybrid_metrics["Coverage@100"] * coverage_retention
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}
