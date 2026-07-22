"""Dataset adapters for the fixed, one-query-per-user retrieval protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = {
    "user_id",
    "video_id",
    "timestamp",
    "play_duration",
    "video_duration",
    "watch_ratio",
}
KUAIREC_FILES = (
    "big_matrix.csv",
    "small_matrix.csv",
    "item_daily_features.csv",
    "item_categories.csv",
    "kuairec_caption_category.csv",
)


def resolve_kuairec_data_dir(
    environment: dict[str, str] | None = None,
) -> Path:
    """Resolve the shared raw data directory without copying or modifying it."""

    values = os.environ if environment is None else environment
    configured = values.get("KUAIREC_DATA_DIR")
    if not configured:
        raise ValueError("KUAIREC_DATA_DIR must point to the shared raw data directory")
    root = Path(configured).expanduser().resolve()
    missing = [name for name in KUAIREC_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"KUAIREC_DATA_DIR is missing files: {missing}")
    return root


def is_strong_positive(watch_ratio: Any) -> np.ndarray:
    """Official KuaiRec example boundary: strictly greater than 2.0."""

    return np.asarray(watch_ratio, dtype=np.float64) > 2.0


def is_quick_skip(play_duration: Any, video_duration: Any) -> np.ndarray:
    """Return the strict short-play label from KuaiRec's feature definition."""

    play = np.asarray(play_duration, dtype=np.float64)
    duration = np.asarray(video_duration, dtype=np.float64)
    return play < np.minimum(3000.0, duration)


def _validate_events(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = EVENT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame[list(EVENT_COLUMNS)].isnull().any().any():
        raise ValueError(f"{name} contains missing required values")
    duplicate = frame.duplicated(["user_id", "video_id", "timestamp"])
    if duplicate.any():
        raise ValueError(f"{name} must already contain canonical event keys")
    result = frame.copy()
    result["user_id"] = result["user_id"].astype(np.int64)
    result["video_id"] = result["video_id"].astype(np.int64)
    result["timestamp"] = result["timestamp"].astype(np.float64)
    return result.sort_values(
        ["user_id", "timestamp", "video_id"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class RetrievalQueries:
    """Ragged query inputs shared by every fixed-catalog baseline and model."""

    user_ids: np.ndarray
    histories: tuple[np.ndarray, ...]
    history_weights: tuple[np.ndarray, ...]
    candidates: tuple[np.ndarray, ...]
    relevant: tuple[np.ndarray, ...]
    catalog: np.ndarray
    diagnostics: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.user_ids)
        if len(np.unique(self.user_ids)) != count:
            raise ValueError("Each user may contribute at most one query")
        for values in (
            self.histories,
            self.history_weights,
            self.candidates,
            self.relevant,
        ):
            if len(values) != count:
                raise ValueError("Ragged query fields differ in length")
        catalog = set(int(item) for item in self.catalog)
        for history, weights, candidates, relevant in zip(
            self.histories,
            self.history_weights,
            self.candidates,
            self.relevant,
            strict=True,
        ):
            if len(history) != len(weights):
                raise ValueError("History items and weights differ in length")
            if len(np.unique(candidates)) != len(candidates):
                raise ValueError("Candidate rows must contain unique items")
            if not set(int(item) for item in candidates).issubset(catalog):
                raise ValueError("Query candidate is outside the fixed catalog")
            if not len(relevant):
                raise ValueError("Every evaluated query needs at least one relevant item")
            if not set(int(item) for item in relevant).issubset(
                set(int(item) for item in candidates)
            ):
                raise ValueError("Relevant items must be query candidates")


def build_fixed_validation_catalog(
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
    *,
    normal_item_ids: np.ndarray,
) -> np.ndarray:
    """Freeze NORMAL items observed anywhere in the train/validation windows.

    Validation membership reveals only the fixed item universe, not user-item
    labels or frequencies. It guarantees that an unseen validation positive is
    rankable without a per-timestamp catalog replay.
    """

    train = _validate_events(train_events, name="train_events")
    validation = _validate_events(validation_events, name="validation_events")
    observed = np.union1d(train["video_id"], validation["video_id"])
    return np.intersect1d(
        observed, np.asarray(normal_item_ids, dtype=np.int64), assume_unique=False
    ).astype(np.int64)


def _history_weights(frame: pd.DataFrame) -> np.ndarray:
    ratios = np.clip(frame["watch_ratio"].to_numpy(np.float64), 0.0, 4.0)
    weights = np.maximum(ratios, 0.1)
    quick = is_quick_skip(frame["play_duration"], frame["video_duration"])
    weights[quick] *= 0.25
    return weights.astype(np.float32)


def build_big_validation_queries(
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
    *,
    fixed_catalog: np.ndarray,
    max_history: int = 50,
) -> RetrievalQueries:
    """Build one frozen validation query per user from train-only history."""

    if max_history <= 0:
        raise ValueError("max_history must be positive")
    train = _validate_events(train_events, name="train_events")
    validation = _validate_events(validation_events, name="validation_events")
    catalog = np.unique(np.asarray(fixed_catalog, dtype=np.int64))
    catalog_set = set(int(item) for item in catalog)
    validation = validation.assign(
        strong=is_strong_positive(validation["watch_ratio"])
    )
    train_groups = {int(user): rows for user, rows in train.groupby("user_id")}
    user_ids: list[int] = []
    histories: list[np.ndarray] = []
    history_weights: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    relevant_rows: list[np.ndarray] = []
    skipped_zero_positive = 0
    for user, validation_rows in validation.groupby("user_id", sort=True):
        user = int(user)
        history = train_groups.get(user, train.iloc[0:0])
        seen = set(int(item) for item in history["video_id"])
        relevant = np.unique(
            validation_rows.loc[
                validation_rows["strong"]
                & ~validation_rows["video_id"].isin(seen),
                "video_id",
            ].to_numpy(np.int64)
        )
        if not len(relevant):
            skipped_zero_positive += 1
            continue
        outside = set(int(item) for item in relevant) - catalog_set
        if outside:
            raise ValueError(
                f"Validation relevant items are outside the frozen catalog: {sorted(outside)}"
            )
        candidate_row = np.asarray(
            [item for item in catalog if int(item) not in seen], dtype=np.int64
        )
        history = history.tail(max_history)
        user_ids.append(user)
        histories.append(history["video_id"].to_numpy(np.int64))
        history_weights.append(_history_weights(history))
        candidates.append(candidate_row)
        relevant_rows.append(relevant)
    return RetrievalQueries(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        histories=tuple(histories),
        history_weights=tuple(history_weights),
        candidates=tuple(candidates),
        relevant=tuple(relevant_rows),
        catalog=catalog,
        diagnostics={"zero_relevant_users_excluded": skipped_zero_positive},
    )


def build_small_observed_queries(
    observed_events: pd.DataFrame,
    *,
    normal_item_ids: np.ndarray,
) -> RetrievalQueries:
    """Build sealed-evaluation semantics without treating missing pairs as negatives."""

    observed = _validate_events(observed_events, name="small_observed_events")
    normal = set(int(item) for item in np.asarray(normal_item_ids, dtype=np.int64))
    observed = observed[observed["video_id"].isin(normal)].copy()
    observed["strong"] = is_strong_positive(observed["watch_ratio"])
    users: list[int] = []
    candidates: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    zero_positive = 0
    for user, rows in observed.groupby("user_id", sort=True):
        candidate_row = np.unique(rows["video_id"].to_numpy(np.int64))
        relevant_row = np.unique(
            rows.loc[rows["strong"], "video_id"].to_numpy(np.int64)
        )
        if not len(relevant_row):
            zero_positive += 1
            continue
        users.append(int(user))
        candidates.append(candidate_row)
        relevant.append(relevant_row)
    empty = tuple(np.asarray([], dtype=np.int64) for _ in users)
    empty_weights = tuple(np.asarray([], dtype=np.float32) for _ in users)
    return RetrievalQueries(
        user_ids=np.asarray(users, dtype=np.int64),
        histories=empty,
        history_weights=empty_weights,
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=np.asarray(sorted(normal), dtype=np.int64),
        diagnostics={"zero_relevant_users_excluded": zero_positive},
    )


def data_cold_items(
    train_events: pd.DataFrame, *, catalog: np.ndarray
) -> np.ndarray:
    """Return catalog items with no canonical train interaction of any label."""

    train = _validate_events(train_events, name="train_events")
    seen = np.unique(train["video_id"].to_numpy(np.int64))
    return np.setdiff1d(
        np.asarray(catalog, dtype=np.int64), seen, assume_unique=False
    ).astype(np.int64)
