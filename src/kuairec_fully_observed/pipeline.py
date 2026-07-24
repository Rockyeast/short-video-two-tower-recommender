"""Thin online-style orchestration over the frozen retrieval routes.

This module deliberately owns no training logic.  A route only needs to expose
``retrieve`` and ``supports_user``; replacing a model therefore does not change
candidate filtering, cold-user routing, or rank fusion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from .hybrid import weighted_reciprocal_rank_fusion


class CandidateRetriever(Protocol):
    """The small contract implemented by every retrieval route."""

    name: str

    def supports_user(self, user_id: int) -> bool:
        """Return whether this route has a trained representation for a user."""

    def retrieve(
        self,
        *,
        user_id: int,
        history: np.ndarray,
        history_weights: np.ndarray,
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Return unique candidate IDs in descending score order."""


def _validate_ids(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must contain unique IDs")
    return array


def _stable_topk(
    item_ids: np.ndarray, scores: np.ndarray, *, k: int
) -> np.ndarray:
    if scores.shape != (len(item_ids),) or not np.isfinite(scores).all():
        raise ValueError("Retriever scores must be finite and align with items")
    order = np.lexsort((item_ids, -scores))[:k]
    return item_ids[order]


@dataclass(frozen=True)
class PopularityRetriever:
    """Frozen global-popularity fallback."""

    scores: Mapping[int, float]
    name: str = "popularity"

    def supports_user(self, user_id: int) -> bool:
        return True

    def retrieve(
        self,
        *,
        user_id: int,
        history: np.ndarray,
        history_weights: np.ndarray,
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        values = np.asarray(
            [self.scores.get(int(item), 0.0) for item in candidates],
            dtype=np.float64,
        )
        return _stable_topk(candidates, values, k=k)


@dataclass(frozen=True)
class BPRRetriever:
    """Adapter around a trained BPR user/item-factor checkpoint."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    name: str = "bpr"
    _user_positions: dict[int, int] = field(
        init=False, repr=False, compare=False
    )
    _item_positions: dict[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        users = _validate_ids(self.user_ids, name="BPR user_ids")
        items = _validate_ids(self.item_ids, name="BPR item_ids")
        if self.user_factors.shape[0] != len(users):
            raise ValueError("BPR user factors do not align with user IDs")
        if self.item_factors.shape[0] != len(items):
            raise ValueError("BPR item factors do not align with item IDs")
        if (
            self.user_factors.ndim != 2
            or self.item_factors.ndim != 2
            or self.user_factors.shape[1] != self.item_factors.shape[1]
        ):
            raise ValueError("BPR factors need matching rank-2 dimensions")
        object.__setattr__(
            self,
            "_user_positions",
            {int(value): index for index, value in enumerate(users)},
        )
        object.__setattr__(
            self,
            "_item_positions",
            {int(value): index for index, value in enumerate(items)},
        )

    @classmethod
    def from_npz(cls, path: Path) -> "BPRRetriever":
        with np.load(path) as payload:
            return cls(
                user_ids=payload["user_ids"].astype(np.int64, copy=True),
                item_ids=payload["item_ids"].astype(np.int64, copy=True),
                user_factors=payload["user_factors"].astype(
                    np.float32, copy=True
                ),
                item_factors=payload["item_factors"].astype(
                    np.float32, copy=True
                ),
            )

    def supports_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_positions

    def retrieve(
        self,
        *,
        user_id: int,
        history: np.ndarray,
        history_weights: np.ndarray,
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        position = self._user_positions.get(int(user_id))
        if position is None:
            raise ValueError("BPR route received an unknown user")
        candidate_positions = np.fromiter(
            (self._item_positions.get(int(item), -1) for item in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        present = candidate_positions >= 0
        scores = np.zeros(len(candidates), dtype=np.float32)
        scores[present] = (
            self.item_factors[candidate_positions[present]]
            @ self.user_factors[position]
        )
        return _stable_topk(candidates, scores, k=k)


UserEncoder = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TwoTowerRetriever:
    """Adapter over pre-encoded item vectors and the existing user tower."""

    item_ids: np.ndarray
    item_vectors: np.ndarray
    trained_user_ids: frozenset[int]
    encode_user: UserEncoder
    name: str = "two_tower"
    _item_positions: dict[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        items = _validate_ids(self.item_ids, name="Two-Tower item_ids")
        if self.item_vectors.ndim != 2 or len(self.item_vectors) != len(items):
            raise ValueError("Two-Tower item vectors do not align with item IDs")
        if not np.isfinite(self.item_vectors).all():
            raise ValueError("Two-Tower item vectors must be finite")
        object.__setattr__(
            self,
            "_item_positions",
            {int(value): index for index, value in enumerate(items)},
        )

    def supports_user(self, user_id: int) -> bool:
        return int(user_id) in self.trained_user_ids

    def retrieve(
        self,
        *,
        user_id: int,
        history: np.ndarray,
        history_weights: np.ndarray,
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        if not self.supports_user(user_id):
            raise ValueError("Two-Tower route received an unknown user")
        user_vector = np.asarray(
            self.encode_user(user_id, history, history_weights),
            dtype=np.float32,
        )
        if user_vector.shape != (self.item_vectors.shape[1],):
            raise ValueError("User tower returned the wrong vector shape")
        try:
            rows = np.asarray(
                [self._item_positions[int(item)] for item in candidates],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("Two-Tower has no content vector for a candidate") from exc
        scores = self.item_vectors[rows] @ user_vector
        return _stable_topk(candidates, scores, k=k)


@dataclass(frozen=True)
class PipelineConfig:
    route_top_k: int = 500
    output_k: int = 100
    alpha: float = 0.75
    rank_constant: int = 60
    max_history: int = 50

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict) or set(payload) != {"pipeline"}:
            raise ValueError("Pipeline config needs exactly one pipeline section")
        values = payload["pipeline"]
        config = cls(**values)
        if (
            config.route_top_k < config.output_k
            or config.output_k <= 0
            or config.max_history <= 0
        ):
            raise ValueError("Pipeline K/history values are invalid")
        return config


@dataclass(frozen=True)
class RecommendationResult:
    user_id: int
    item_ids: tuple[int, ...]
    strategy: str
    candidate_count: int
    seen_count: int


@dataclass(frozen=True)
class RecommendationEngine:
    """Candidate filtering -> retrieval routes -> frozen fusion/fallback."""

    catalog: np.ndarray
    two_tower: CandidateRetriever
    bpr: CandidateRetriever
    popularity: CandidateRetriever
    config: PipelineConfig = PipelineConfig()

    def __post_init__(self) -> None:
        _validate_ids(self.catalog, name="catalog")

    def recommend(
        self,
        user_id: int,
        recent_history: Sequence[int] | np.ndarray,
        *,
        top_k: int | None = None,
        history_weights: Sequence[float] | np.ndarray | None = None,
    ) -> RecommendationResult:
        history = np.asarray(recent_history, dtype=np.int64)
        if history.ndim != 1:
            raise ValueError("recent_history must be one-dimensional")
        history = history[-self.config.max_history :]
        weights = (
            np.ones(len(history), dtype=np.float32)
            if history_weights is None
            else np.asarray(history_weights, dtype=np.float32)[
                -self.config.max_history :
            ]
        )
        if weights.shape != history.shape or not np.isfinite(weights).all():
            raise ValueError("history_weights must be finite and align with history")
        seen = np.unique(history)
        candidates = self.catalog[~np.isin(self.catalog, seen)]
        requested_k = self.config.output_k if top_k is None else int(top_k)
        if requested_k <= 0:
            raise ValueError("top_k must be positive")
        if requested_k > self.config.output_k:
            raise ValueError("top_k exceeds the configured output limit")
        output_k = min(requested_k, len(candidates))
        if output_k == 0:
            return RecommendationResult(
                int(user_id), (), "empty_catalog", 0, len(seen)
            )

        route_k = min(
            max(self.config.route_top_k, output_k), len(candidates)
        )
        route_arguments = {
            "user_id": int(user_id),
            "history": history,
            "history_weights": weights,
            "candidates": candidates,
            "k": route_k,
        }
        warm = self.two_tower.supports_user(user_id) and self.bpr.supports_user(
            user_id
        )
        if not warm:
            ranked = self.popularity.retrieve(**route_arguments)[:output_k]
            strategy = "cold_user_popularity"
        else:
            two_ranked = self.two_tower.retrieve(**route_arguments)
            bpr_ranked = self.bpr.retrieve(**route_arguments)
            ranked = weighted_reciprocal_rank_fusion(
                two_ranked[None, :],
                bpr_ranked[None, :],
                candidates=(candidates,),
                alpha=self.config.alpha,
                output_k=output_k,
                rank_constant=self.config.rank_constant,
            )[0]
            ranked = ranked[ranked >= 0]
            strategy = "two_tower_bpr_rrf"
        return RecommendationResult(
            user_id=int(user_id),
            item_ids=tuple(int(item) for item in ranked),
            strategy=strategy,
            candidate_count=len(candidates),
            seen_count=len(seen),
        )
