from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from kuairec_fully_observed.data import RetrievalQueries
from kuairec_fully_observed.sasrec_adapter import (
    SASRecTrainingDataset,
    build_recbole_sasrec,
    build_validation_sequences,
    collate_sasrec_examples,
    rank_sasrec,
    train_sasrec_epoch,
)


def _dataset(max_examples: int | None = None) -> SASRecTrainingDataset:
    return SASRecTrainingDataset(
        event_users=np.asarray([0, 0, 0, 0, 1, 1, 1]),
        event_items=np.asarray([0, 1, 2, 1, 1, 3, 4]),
        event_times=np.asarray([1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        event_strong=np.asarray([False, True, True, True, False, True, True]),
        user_indptr=np.asarray([0, 4, 7]),
        normal_item_mask=np.asarray([True, True, True, True, True]),
        train_end=4.0,
        max_history=3,
        max_examples=max_examples,
    )


def test_training_sequences_are_causal_first_contact_targets() -> None:
    dataset = _dataset()
    # User 0's repeated item 1 at t=3 is not another unseen-item target.
    assert len(dataset) == 4
    first = dataset[0]
    assert first.target_item == 2  # zero-based item 1 shifted for RecBole
    assert first.item_sequence.tolist() == [1]
    second = dataset[1]
    assert second.target_item == 3
    # The other event at the target's same timestamp cannot enter history.
    assert second.item_sequence.tolist() == [1]


def test_validation_sequences_use_only_train_events() -> None:
    sequences, lengths = build_validation_sequences(
        query_user_ids=np.asarray([10, 20, 99]),
        actual_user_ids=np.asarray([10, 20]),
        event_items=np.asarray([0, 1, 2, 3, 4]),
        event_times=np.asarray([1.0, 2.0, 5.0, 1.0, 2.0]),
        user_indptr=np.asarray([0, 3, 5]),
        train_end=3.0,
        max_history=3,
    )
    assert lengths.tolist() == [2, 2, 0]
    assert sequences.tolist() == [[1, 2, 0], [4, 5, 0], [0, 0, 0]]


@pytest.mark.skipif(
    importlib.util.find_spec("recbole") is None,
    reason="RecBole is an optional sequential dependency",
)
def test_recbole_sasrec_smoke_learns_and_ranks_candidates() -> None:
    torch.manual_seed(7)
    dataset = _dataset()
    model = build_recbole_sasrec(
        num_event_items=5,
        max_history=3,
        model_config={
            "n_layers": 1,
            "n_heads": 1,
            "hidden_size": 8,
            "inner_size": 16,
            "hidden_dropout_prob": 0.0,
            "attn_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "loss_type": "CE",
        },
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = train_sasrec_epoch(
        model,
        dataset,
        optimizer,
        batch_size=2,
        max_history=3,
        seed=7,
        device=torch.device("cpu"),
    )
    assert result["optimizer_steps"] == 2
    assert np.isfinite(result["mean_loss"])

    queries = RetrievalQueries(
        user_ids=np.asarray([10, 99]),
        histories=(np.asarray([0]), np.asarray([], dtype=np.int64)),
        history_weights=(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        ),
        candidates=(np.asarray([1, 2, 3]), np.asarray([1, 2, 3])),
        relevant=(np.asarray([1]), np.asarray([2])),
        catalog=np.asarray([1, 2, 3]),
        warm_user_mask=np.asarray([True, False]),
    )
    ranked = rank_sasrec(
        model,
        queries=queries,
        sequences=np.asarray([[1, 0, 0], [0, 0, 0]]),
        sequence_lengths=np.asarray([1, 0]),
        video_ids=np.asarray([0, 1, 2, 3, 4]),
        fallback_topk=np.asarray([[1, 2, 3], [3, 2, 1]]),
        device=torch.device("cpu"),
        k=3,
        batch_size=2,
    )
    assert set(ranked[0]) == {1, 2, 3}
    assert ranked[1].tolist() == [3, 2, 1]


def test_collation_right_pads_with_zero() -> None:
    rows = [_dataset(max_examples=2)[index] for index in range(2)]
    batch = collate_sasrec_examples(rows, max_history=3)
    assert batch["item_id_list"].shape == (2, 3)
    assert batch["item_length"].tolist() == [1, 1]
    assert torch.all(batch["item_id_list"][:, 1:] == 0)
