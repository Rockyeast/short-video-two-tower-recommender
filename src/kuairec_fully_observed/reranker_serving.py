"""Identity-checked local LightGBM reranking for the online-style pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np

from .reranking import (
    RERANK_FEATURE_NAMES,
    build_rerank_feature_matrix,
)
from .serving_bundle import sha256_file


RERANKER_FEATURE_KEYS = {
    "item_ids",
    "category_ids",
    "video_duration",
    "train_data_cold_mask",
    "train_popularity_scores",
}


def _validate_feature_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if set(arrays) != RERANKER_FEATURE_KEYS:
        raise ValueError("Reranker feature bundle fields differ")
    item_ids = np.asarray(arrays["item_ids"])
    count = len(item_ids)
    if (
        item_ids.ndim != 1
        or item_ids.dtype.kind not in "iu"
        or not np.array_equal(item_ids, np.unique(item_ids))
        or np.asarray(arrays["category_ids"]).shape != (count, 3)
        or np.asarray(arrays["video_duration"]).shape != (count,)
        or np.asarray(arrays["train_data_cold_mask"]).shape != (count,)
        or np.asarray(arrays["train_popularity_scores"]).shape != (count,)
    ):
        raise ValueError("Reranker feature arrays do not align")
    if not np.isfinite(
        np.asarray(arrays["video_duration"], dtype=np.float64)
    ).all() or not np.isfinite(
        np.asarray(arrays["train_popularity_scores"], dtype=np.float64)
    ).all():
        raise ValueError("Reranker feature bundle contains non-finite values")
    return {"item_count": count}


def write_reranker_feature_bundle(
    *,
    feature_path: Path,
    metadata_path: Path,
    arrays: Mapping[str, np.ndarray],
    model_path: Path,
    serving_bundle_sha256: str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _validate_feature_arrays(arrays)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        **{
            name: np.asarray(arrays[name])
            for name in sorted(RERANKER_FEATURE_KEYS)
        },
    )
    metadata = {
        "schema_version": 1,
        "model_sha256": sha256_file(model_path),
        "feature_bundle_sha256": sha256_file(feature_path),
        "serving_bundle_sha256": serving_bundle_sha256,
        "feature_names": list(RERANK_FEATURE_NAMES),
        "fixed_candidate_count": 100,
        "alpha": 0.75,
        "rank_constant": 60,
        "enabled_by_default": False,
        "base_score_distribution": (
            "trained_on_train_only_routes_served_on_final_refit_routes"
        ),
        "effectiveness_claim_for_serving_routes": False,
        "source_identity": dict(source_identity),
        **summary,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


@dataclass(frozen=True)
class LocalLightGBMReranker:
    booster: lgb.Booster
    item_ids: np.ndarray
    category_ids: np.ndarray
    video_duration: np.ndarray
    train_data_cold_mask: np.ndarray
    train_popularity_scores: np.ndarray
    alpha: float = 0.75
    rank_constant: int = 60
    name: str = "local_lightgbm"
    _positions: dict[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_feature_arrays(
            {
                "item_ids": self.item_ids,
                "category_ids": self.category_ids,
                "video_duration": self.video_duration,
                "train_data_cold_mask": self.train_data_cold_mask,
                "train_popularity_scores": self.train_popularity_scores,
            }
        )
        if self.booster.num_feature() != len(RERANK_FEATURE_NAMES):
            raise ValueError("LightGBM feature dimension differs")
        object.__setattr__(
            self,
            "_positions",
            {
                int(item): position
                for position, item in enumerate(self.item_ids)
            },
        )

    def build_features(
        self,
        *,
        history: np.ndarray,
        candidates: np.ndarray,
        two_tower_ranked: np.ndarray,
        bpr_ranked: np.ndarray,
        two_tower_scores: np.ndarray,
        bpr_scores: np.ndarray,
    ) -> np.ndarray:
        try:
            rows = np.asarray(
                [self._positions[int(item)] for item in candidates],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Reranker has no static features for a candidate"
            ) from exc
        history_categories = {
            int(category)
            for item in history
            if int(item) in self._positions
            for category in self.category_ids[
                self._positions[int(item)]
            ]
            if category >= 0
        }
        return build_rerank_feature_matrix(
            item_ids=candidates,
            two_tower_scores=two_tower_scores,
            bpr_scores=bpr_scores,
            two_tower_ranked=two_tower_ranked,
            bpr_ranked=bpr_ranked,
            popularity_scores=self.train_popularity_scores[rows],
            item_categories=self.category_ids[rows],
            history_categories=history_categories,
            data_cold_mask=self.train_data_cold_mask[rows],
            video_duration=self.video_duration[rows],
            history_length=len(history),
            alpha=self.alpha,
            rank_constant=self.rank_constant,
        )

    def rerank(
        self,
        *,
        user_id: int,
        history: np.ndarray,
        history_weights: np.ndarray,
        candidates: np.ndarray,
        two_tower_ranked: np.ndarray,
        bpr_ranked: np.ndarray,
        two_tower_scores: np.ndarray,
        bpr_scores: np.ndarray,
    ) -> np.ndarray:
        features = self.build_features(
            history=history,
            candidates=candidates,
            two_tower_ranked=two_tower_ranked,
            bpr_ranked=bpr_ranked,
            two_tower_scores=two_tower_scores,
            bpr_scores=bpr_scores,
        )
        scores = np.asarray(
            self.booster.predict(features), dtype=np.float64
        )
        if scores.shape != (len(candidates),) or not np.isfinite(
            scores
        ).all():
            raise RuntimeError("LightGBM reranker produced invalid scores")
        order = np.lexsort((candidates, -scores))
        return np.asarray(candidates[order], dtype=np.int64)


def load_local_lightgbm_reranker(
    *,
    model_path: Path,
    feature_path: Path,
    metadata_path: Path,
    serving_bundle_sha256: str,
) -> tuple[LocalLightGBMReranker, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        raise RuntimeError("Unsupported reranker metadata schema")
    expected = {
        "model_sha256": sha256_file(model_path),
        "feature_bundle_sha256": sha256_file(feature_path),
        "serving_bundle_sha256": serving_bundle_sha256,
        "feature_names": list(RERANK_FEATURE_NAMES),
        "fixed_candidate_count": 100,
        "alpha": 0.75,
        "rank_constant": 60,
        "enabled_by_default": False,
        "effectiveness_claim_for_serving_routes": False,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"Reranker metadata mismatch: {name}")
    with np.load(feature_path, allow_pickle=False) as payload:
        arrays = {
            name: payload[name].copy() for name in payload.files
        }
    summary = _validate_feature_arrays(arrays)
    if metadata.get("item_count") != summary["item_count"]:
        raise RuntimeError("Reranker item count metadata mismatch")
    booster = lgb.Booster(model_file=str(model_path))
    reranker = LocalLightGBMReranker(
        booster=booster,
        item_ids=arrays["item_ids"].astype(np.int64, copy=False),
        category_ids=arrays["category_ids"].astype(
            np.int64, copy=False
        ),
        video_duration=arrays["video_duration"].astype(
            np.float32, copy=False
        ),
        train_data_cold_mask=arrays["train_data_cold_mask"].astype(
            bool, copy=False
        ),
        train_popularity_scores=arrays[
            "train_popularity_scores"
        ].astype(np.float32, copy=False),
        alpha=float(metadata["alpha"]),
        rank_constant=int(metadata["rank_constant"]),
    )
    return reranker, metadata
