"""Linear preprocessing and lazy datasets for Phase B0 training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import _validate_events, is_quick_skip, is_strong_positive


def _weights_from_arrays(
    watch_ratio: np.ndarray,
    play_duration: np.ndarray,
    video_duration: np.ndarray,
    *,
    quick_skip_mask: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.maximum(np.clip(watch_ratio, 0.0, 4.0), 0.1)
    quick = (
        is_quick_skip(play_duration, video_duration)
        if quick_skip_mask is None
        else np.asarray(quick_skip_mask, dtype=bool)
    )
    if quick.shape != weights.shape:
        raise ValueError("quick_skip_mask must match the behavior rows")
    weights[quick] *= 0.25
    return weights.astype(np.float32)


@dataclass(frozen=True)
class TwoTowerTrainingExample:
    user_id: int
    target_item_id: int
    target_timestamp: float
    history: np.ndarray
    history_weights: np.ndarray


class TwoTowerTrainingDataset:
    """Lazy causal histories backed by one sorted event-array pass.

    Construction is O(number of events). Fetching an example searches only its
    user's timestamp array and walks backward until ``max_history`` non-target
    items have been collected; it never rescans a pandas user frame per target.
    """

    def __init__(
        self,
        fit_events: pd.DataFrame,
        *,
        max_history: int = 50,
        normal_item_ids: np.ndarray | None = None,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        events = _validate_events(fit_events, name="fit_events")
        self.max_history = int(max_history)
        self.user_ids = events["user_id"].to_numpy(np.int64)
        self.item_ids = events["video_id"].to_numpy(np.int64)
        self.timestamps = events["timestamp"].to_numpy(np.float64)
        self.play_duration = events["play_duration"].to_numpy(np.float64)
        self.video_duration = events["video_duration"].to_numpy(np.float64)
        self.watch_ratio = events["watch_ratio"].to_numpy(np.float64)
        strong = (
            events["_is_strong_positive"].to_numpy(bool)
            if "_is_strong_positive" in events.columns
            else is_strong_positive(self.watch_ratio)
        )
        self.quick_skip = (
            events["_is_quick_skip"].to_numpy(bool)
            if "_is_quick_skip" in events.columns
            else is_quick_skip(self.play_duration, self.video_duration)
        )
        if normal_item_ids is not None:
            strong &= np.isin(
                self.item_ids, np.asarray(normal_item_ids, dtype=np.int64)
            )
        # The online task retrieves unseen items.  A later strong interaction
        # with an already encountered item is therefore not a valid target;
        # dropping only that item from history would fabricate an unseen event.
        first_user_item_contact = ~events.duplicated(
            ["user_id", "video_id"], keep="first"
        ).to_numpy()
        self.positive_event_indices = np.flatnonzero(
            strong & first_user_item_contact
        ).astype(np.int64)
        unique_users, starts = np.unique(self.user_ids, return_index=True)
        self._user_starts = {
            int(user): int(start) for user, start in zip(unique_users, starts, strict=True)
        }
        known: defaultdict[int, set[int]] = defaultdict(set)
        # False-negative masking still needs every known strong-positive item,
        # including later repeats that are intentionally not training targets.
        for event_index in np.flatnonzero(strong):
            known[int(self.user_ids[event_index])].add(
                int(self.item_ids[event_index])
            )
        self.known_positive_items = {
            user: frozenset(items) for user, items in known.items()
        }

    def __len__(self) -> int:
        return len(self.positive_event_indices)

    def __getitem__(self, index: int) -> TwoTowerTrainingExample:
        event_index = int(self.positive_event_indices[index])
        user = int(self.user_ids[event_index])
        target = int(self.item_ids[event_index])
        target_time = float(self.timestamps[event_index])
        start = self._user_starts[user]
        prior_end = start + int(
            np.searchsorted(
                self.timestamps[start:event_index], target_time, side="left"
            )
        )
        selected: list[int] = []
        cursor = prior_end - 1
        while cursor >= start and len(selected) < self.max_history:
            if int(self.item_ids[cursor]) != target:
                selected.append(cursor)
            cursor -= 1
        selected.reverse()
        positions = np.asarray(selected, dtype=np.int64)
        history = self.item_ids[positions].copy()
        weights = _weights_from_arrays(
            self.watch_ratio[positions],
            self.play_duration[positions],
            self.video_duration[positions],
            quick_skip_mask=self.quick_skip[positions],
        )
        return TwoTowerTrainingExample(
            user_id=user,
            target_item_id=target,
            target_timestamp=target_time,
            history=history,
            history_weights=weights,
        )


@dataclass(frozen=True)
class TwoTowerTrainingExamples:
    """Materialized compatibility view used by small tests only."""

    user_ids: np.ndarray
    target_item_ids: np.ndarray
    target_timestamps: np.ndarray
    histories: tuple[np.ndarray, ...]
    history_weights: tuple[np.ndarray, ...]
    known_positive_items: dict[int, frozenset[int]]


def build_two_tower_training_dataset(
    fit_events: pd.DataFrame,
    *,
    max_history: int = 50,
    normal_item_ids: np.ndarray | None = None,
) -> TwoTowerTrainingDataset:
    return TwoTowerTrainingDataset(
        fit_events,
        max_history=max_history,
        normal_item_ids=normal_item_ids,
    )


def build_two_tower_training_examples(
    fit_events: pd.DataFrame, *, max_history: int = 50
) -> TwoTowerTrainingExamples:
    """Materialize the lazy dataset for backwards-compatible synthetic tests."""

    dataset = build_two_tower_training_dataset(
        fit_events, max_history=max_history
    )
    examples = [dataset[index] for index in range(len(dataset))]
    return TwoTowerTrainingExamples(
        user_ids=np.asarray([row.user_id for row in examples], dtype=np.int64),
        target_item_ids=np.asarray(
            [row.target_item_id for row in examples], dtype=np.int64
        ),
        target_timestamps=np.asarray(
            [row.target_timestamp for row in examples], dtype=np.float64
        ),
        histories=tuple(row.history for row in examples),
        history_weights=tuple(row.history_weights for row in examples),
        known_positive_items=dataset.known_positive_items,
    )


@dataclass(frozen=True)
class BPRTrainingDataset:
    """One strong NORMAL positive and one resampled negative per epoch."""

    user_ids: np.ndarray
    positive_item_ids: np.ndarray
    negative_catalog: np.ndarray
    known_positive_items: dict[int, frozenset[int]]
    seed: int

    def sample_negatives(self, epoch: int) -> np.ndarray:
        """Sample uniformly after excluding every fit-known user positive."""

        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        output = np.empty(len(self.user_ids), dtype=np.int64)
        users, starts, counts = np.unique(
            self.user_ids, return_index=True, return_counts=True
        )
        for user, start, count in zip(users, starts, counts, strict=True):
            rows = slice(int(start), int(start + count))
            positives = np.asarray(
                sorted(self.known_positive_items[int(user)]), dtype=np.int64
            )
            pool = np.setdiff1d(
                self.negative_catalog, positives, assume_unique=True
            )
            if not len(pool):
                raise ValueError(f"BPR negative pool is empty for user {int(user)}")
            output[rows] = rng.choice(pool, size=int(count), replace=True)
        return output


def build_bpr_training_dataset(
    fit_events: pd.DataFrame,
    *,
    normal_item_ids: np.ndarray,
    seed: int,
) -> BPRTrainingDataset:
    """Build the locked BPR sampling population in one vectorized pass.

    Positives are canonical ``watch_ratio > 2`` NORMAL events. The negative
    catalog is restricted to NORMAL videos observed somewhere in the fit
    context, so truly data-cold videos remain untrained and receive score zero.
    Every epoch resamples one uniform item after excluding all known positives
    of that user.
    """

    events = _validate_events(fit_events, name="fit_events")
    normal = np.unique(np.asarray(normal_item_ids, dtype=np.int64))
    items = events["video_id"].to_numpy(np.int64)
    users = events["user_id"].to_numpy(np.int64)
    positive_mask = is_strong_positive(events["watch_ratio"]) & np.isin(items, normal)
    positive_users = users[positive_mask]
    positive_items = items[positive_mask]
    if not len(positive_users):
        raise ValueError("BPR training requires at least one strong NORMAL positive")
    negative_catalog = np.intersect1d(normal, np.unique(items), assume_unique=True)
    known_sets: defaultdict[int, set[int]] = defaultdict(set)
    for user, item in zip(positive_users, positive_items, strict=True):
        known_sets[int(user)].add(int(item))
    known = {user: frozenset(items) for user, items in known_sets.items()}
    return BPRTrainingDataset(
        user_ids=positive_users,
        positive_item_ids=positive_items,
        negative_catalog=negative_catalog,
        known_positive_items=known,
        seed=int(seed),
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
