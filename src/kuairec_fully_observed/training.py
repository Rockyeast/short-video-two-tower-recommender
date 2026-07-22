"""Leakage-safe Two-Tower V1 training-example contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import _history_weights, _validate_events, is_strong_positive


@dataclass(frozen=True)
class TwoTowerTrainingExamples:
    """One causal history and one strong-positive target per example."""

    user_ids: np.ndarray
    target_item_ids: np.ndarray
    target_timestamps: np.ndarray
    histories: tuple[np.ndarray, ...]
    history_weights: tuple[np.ndarray, ...]
    known_positive_items: dict[int, frozenset[int]]


def build_two_tower_training_examples(
    fit_events: pd.DataFrame, *, max_history: int = 50
) -> TwoTowerTrainingExamples:
    """Create positive examples without putting a target in its own history.

    Histories use only events with a strictly earlier timestamp. Quick skips
    remain low-weight history context in V1; they are not explicit negatives.
    """

    if max_history <= 0:
        raise ValueError("max_history must be positive")
    events = _validate_events(fit_events, name="fit_events")
    strong = is_strong_positive(events["watch_ratio"])
    known_positive_items = {
        int(user): frozenset(int(item) for item in rows["video_id"])
        for user, rows in events.loc[strong].groupby("user_id", sort=False)
    }
    user_ids: list[int] = []
    targets: list[int] = []
    timestamps: list[float] = []
    histories: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for user, rows in events.groupby("user_id", sort=True):
        for target in rows.loc[is_strong_positive(rows["watch_ratio"])].itertuples():
            prior = rows.loc[
                (rows["timestamp"] < float(target.timestamp))
                & (rows["video_id"] != int(target.video_id))
            ].tail(max_history)
            user_ids.append(int(user))
            targets.append(int(target.video_id))
            timestamps.append(float(target.timestamp))
            histories.append(prior["video_id"].to_numpy(np.int64))
            weights.append(_history_weights(prior))
    return TwoTowerTrainingExamples(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        target_item_ids=np.asarray(targets, dtype=np.int64),
        target_timestamps=np.asarray(timestamps, dtype=np.float64),
        histories=tuple(histories),
        history_weights=tuple(weights),
        known_positive_items=known_positive_items,
    )


def build_in_batch_logit_mask(
    user_ids: np.ndarray,
    target_item_ids: np.ndarray,
    known_positive_items: Mapping[int, frozenset[int] | set[int]],
) -> np.ndarray:
    """Mask false negatives while preserving each row's diagonal positive."""

    users = np.asarray(user_ids, dtype=np.int64)
    targets = np.asarray(target_item_ids, dtype=np.int64)
    if users.ndim != 1 or targets.shape != users.shape:
        raise ValueError("Batch user and target arrays must be equal rank-1 shapes")
    mask = np.ones((len(users), len(users)), dtype=bool)
    for row, user in enumerate(users):
        positives = set(int(item) for item in known_positive_items.get(int(user), ()))
        for column, target in enumerate(targets):
            if row != column and (
                int(target) == int(targets[row]) or int(target) in positives
            ):
                mask[row, column] = False
    np.fill_diagonal(mask, True)
    return mask
