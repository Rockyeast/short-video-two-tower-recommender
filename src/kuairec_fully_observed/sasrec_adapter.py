"""Minimal RecBole SASRec adapter for the frozen Big-validation protocol.

RecBole supplies the model implementation.  This module only adapts the
already-frozen KuaiRec event arrays to causal sequences and maps RecBole's
scores back to the repository's exact candidate/evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import RetrievalQueries


@dataclass(frozen=True)
class SASRecTrainingExample:
    item_sequence: np.ndarray
    target_item: int


class SASRecTrainingDataset(Dataset[SASRecTrainingExample]):
    """Lazy unseen-item next-positive samples over canonical Big-train events."""

    def __init__(
        self,
        *,
        event_users: np.ndarray,
        event_items: np.ndarray,
        event_times: np.ndarray,
        event_strong: np.ndarray,
        user_indptr: np.ndarray,
        normal_item_mask: np.ndarray,
        train_end: float,
        max_history: int,
        max_examples: int | None = None,
    ) -> None:
        users = np.asarray(event_users, dtype=np.int64)
        items = np.asarray(event_items, dtype=np.int64)
        times = np.asarray(event_times, dtype=np.float64)
        strong = np.asarray(event_strong, dtype=bool)
        indptr = np.asarray(user_indptr, dtype=np.int64)
        normal = np.asarray(normal_item_mask, dtype=bool)
        if not (users.shape == items.shape == times.shape == strong.shape):
            raise ValueError("Event arrays must have equal shapes")
        if indptr.ndim != 1 or indptr[0] != 0 or indptr[-1] != len(users):
            raise ValueError("user_indptr does not cover the event arrays")
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        if np.any(items < 0) or np.any(items >= len(normal)):
            raise ValueError("Event item position is outside normal_item_mask")

        first_contact = np.zeros(len(items), dtype=bool)
        for start, end in zip(indptr[:-1], indptr[1:], strict=True):
            seen: set[int] = set()
            for row in range(int(start), int(end)):
                item = int(items[row])
                if item not in seen:
                    first_contact[row] = True
                    seen.add(item)
        target_mask = (
            (times < float(train_end))
            & strong
            & normal[items]
            & first_contact
        )
        target_rows = np.flatnonzero(target_mask).astype(np.int64)
        # A pure sequence model has no user-ID state. A target with no strictly
        # earlier behavior provides no sequence signal and is not trainable.
        user_start_for_row = np.empty(len(users), dtype=np.int64)
        for start, end in zip(indptr[:-1], indptr[1:], strict=True):
            user_start_for_row[int(start) : int(end)] = int(start)
        has_prior = np.asarray(
            [
                np.searchsorted(
                    times[user_start_for_row[row] : row],
                    times[row],
                    side="left",
                )
                > 0
                for row in target_rows
            ],
            dtype=bool,
        )
        target_rows = target_rows[has_prior]
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be positive")
            target_rows = target_rows[: int(max_examples)]
        if not len(target_rows):
            raise ValueError("SASRec training has no causal sequence examples")

        self.event_items = items
        self.event_times = times
        self.user_start_for_row = user_start_for_row
        self.target_rows = target_rows
        self.max_history = int(max_history)

    def __len__(self) -> int:
        return len(self.target_rows)

    def __getitem__(self, index: int) -> SASRecTrainingExample:
        row = int(self.target_rows[index])
        start = int(self.user_start_for_row[row])
        target_time = float(self.event_times[row])
        prior_end = start + int(
            np.searchsorted(
                self.event_times[start:row], target_time, side="left"
            )
        )
        history = self.event_items[max(start, prior_end - self.max_history) : prior_end]
        if not len(history):
            raise RuntimeError("SASRec sample unexpectedly has an empty history")
        # RecBole reserves zero for padding, so event positions are shifted by one.
        return SASRecTrainingExample(
            item_sequence=history.astype(np.int64, copy=True) + 1,
            target_item=int(self.event_items[row]) + 1,
        )


def collate_sasrec_examples(
    examples: list[SASRecTrainingExample], *, max_history: int
) -> dict[str, torch.Tensor]:
    sequences = torch.zeros((len(examples), max_history), dtype=torch.long)
    lengths = torch.empty(len(examples), dtype=torch.long)
    targets = torch.empty(len(examples), dtype=torch.long)
    for row, example in enumerate(examples):
        values = torch.as_tensor(example.item_sequence[-max_history:], dtype=torch.long)
        sequences[row, : len(values)] = values
        lengths[row] = len(values)
        targets[row] = int(example.target_item)
    return {
        "item_id_list": sequences,
        "item_length": lengths,
        "item_id": targets,
    }


class _RecBoleItemDataset:
    def __init__(self, n_items: int) -> None:
        self.n_items = int(n_items)

    def num(self, field: str) -> int:
        if field != "item_id":
            raise KeyError(field)
        return self.n_items


def build_recbole_sasrec(
    *,
    num_event_items: int,
    max_history: int,
    model_config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate RecBole's SASRec rather than a local reimplementation."""

    from recbole.model.sequential_recommender.sasrec import SASRec

    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "MAX_ITEM_LIST_LENGTH": int(max_history),
        "device": str(device),
        **model_config,
    }
    model = SASRec(config, _RecBoleItemDataset(num_event_items + 1))
    return model.to(device)


def train_sasrec_epoch(
    model: torch.nn.Module,
    dataset: SASRecTrainingDataset,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    max_history: int,
    seed: int,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=lambda rows: collate_sasrec_examples(
            rows, max_history=max_history
        ),
    )
    model.train()
    total_loss = 0.0
    examples = 0
    steps = 0
    for interaction in loader:
        if max_steps is not None and steps >= max_steps:
            break
        interaction = {
            name: value.to(device) for name, value in interaction.items()
        }
        optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(interaction)
        if not torch.isfinite(loss):
            raise FloatingPointError("SASRec loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_examples = int(interaction["item_id"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_examples
        examples += batch_examples
        steps += 1
    return {
        "mean_loss": total_loss / examples,
        "optimizer_steps": steps,
        "completed_examples": examples,
    }


def build_validation_sequences(
    *,
    query_user_ids: np.ndarray,
    actual_user_ids: np.ndarray,
    event_items: np.ndarray,
    event_times: np.ndarray,
    user_indptr: np.ndarray,
    train_end: float,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-only, right-padded RecBole sequences for fixed queries."""

    user_positions = {
        int(user): position for position, user in enumerate(actual_user_ids)
    }
    sequences = np.zeros((len(query_user_ids), max_history), dtype=np.int64)
    lengths = np.zeros(len(query_user_ids), dtype=np.int64)
    for row, user in enumerate(query_user_ids):
        position = user_positions.get(int(user))
        if position is None:
            continue
        start = int(user_indptr[position])
        end = int(user_indptr[position + 1])
        train_items = np.asarray(event_items[start:end], dtype=np.int64)[
            np.asarray(event_times[start:end], dtype=np.float64) < float(train_end)
        ]
        train_items = train_items[-max_history:]
        sequences[row, : len(train_items)] = train_items + 1
        lengths[row] = len(train_items)
    return sequences, lengths


def rank_sasrec(
    model: torch.nn.Module,
    *,
    queries: RetrievalQueries,
    sequences: np.ndarray,
    sequence_lengths: np.ndarray,
    video_ids: np.ndarray,
    fallback_topk: np.ndarray,
    device: torch.device,
    k: int = 100,
    batch_size: int = 128,
) -> np.ndarray:
    """Score only the frozen candidate membership with deterministic ties."""

    videos = np.asarray(video_ids, dtype=np.int64)
    if len(np.unique(videos)) != len(videos):
        raise ValueError("video_ids must be unique")
    item_positions = {int(item): index for index, item in enumerate(videos)}
    output = np.full((len(queries.user_ids), k), -1, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(queries.user_ids), batch_size):
            end = min(begin + batch_size, len(queries.user_ids))
            batch_lengths = np.asarray(
                sequence_lengths[begin:end], dtype=np.int64
            ).copy()
            # RecBole gathers at length-1 and cannot accept zero. These rows
            # are routed to the frozen fallback below, so a safe padding
            # position is used only to keep the batched forward well-defined.
            batch_lengths[batch_lengths == 0] = 1
            interaction = {
                "item_id_list": torch.as_tensor(
                    sequences[begin:end], dtype=torch.long, device=device
                ),
                "item_length": torch.as_tensor(
                    batch_lengths,
                    dtype=torch.long,
                    device=device,
                ),
            }
            scores = model.full_sort_predict(interaction).detach().cpu().numpy()
            for local_row, global_row in enumerate(range(begin, end)):
                if sequence_lengths[global_row] == 0:
                    output[global_row] = fallback_topk[global_row]
                    continue
                candidates = np.asarray(
                    queries.candidates[global_row], dtype=np.int64
                )
                positions = np.fromiter(
                    (item_positions[int(item)] + 1 for item in candidates),
                    dtype=np.int64,
                    count=len(candidates),
                )
                candidate_scores = scores[local_row, positions]
                order = np.lexsort((candidates, -candidate_scores))[:k]
                selected = candidates[order]
                output[global_row, : len(selected)] = selected
    return output
