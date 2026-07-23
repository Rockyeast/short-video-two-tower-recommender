"""Full-data Two-Tower runner primitives for the frozen Phase B2B route."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import RetrievalQueries
from .provenance import canonical_json_sha256, membership_record
from .torch_models import TwoTowerV1, masked_in_batch_cross_entropy
from .torch_training import (
    PreparedItemFeatureStore,
    assert_model_device,
    collate_training_batch,
    encode_training_batch,
    load_checkpoint,
    record_successful_step_membership,
    resolve_concrete_device,
    stable_int_membership_sha256,
)
from .training import TwoTowerTrainingDataset, _weights_from_arrays


def load_canonical_train_events(
    data_dir: Path,
    *,
    train_end: float,
    selected_user_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Stream complete users and retain only canonical Big-train events."""

    helper_path = Path(__file__).resolve().parents[2] / "scripts/audit_phase0.py"
    spec = importlib.util.spec_from_file_location(
        "_phase_b2b_audit_helpers", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen canonical-event helpers")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    try:
        spec.loader.exec_module(helper)
        event_columns = helper.EVENT_COLUMNS
        iter_user_frames = helper.iter_user_frames
        canonicalize = helper.canonicalize_behavior_events
        rows: list[pd.DataFrame] = []
        for user_id, raw in iter_user_frames(
            data_dir / "big_matrix.csv", event_columns
        ):
            if selected_user_ids is not None and user_id not in selected_user_ids:
                continue
            canonical, _, _, _ = canonicalize(raw)
            train = canonical.loc[
                canonical["timestamp"] < train_end,
                event_columns + ["_is_strong_positive", "_is_quick_skip"],
            ]
            if len(train):
                rows.append(train)
    finally:
        sys.modules.pop(spec.name, None)
    if not rows:
        raise RuntimeError("No canonical Big-train events were loaded")
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["user_id", "timestamp", "video_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def validation_query_contract_sha256(queries: RetrievalQueries) -> str:
    """Hash ordered users and every exact candidate/target membership."""

    digest = hashlib.sha256(b"phase-b2b-validation-query-contract-v1\n")
    for user, warm, candidates, relevant in zip(
        queries.user_ids,
        queries.warm_user_mask,
        queries.candidates,
        queries.relevant,
        strict=True,
    ):
        digest.update(np.asarray([user], dtype="<i8").tobytes())
        digest.update(b"\x01" if warm else b"\x00")
        candidate_values = np.asarray(candidates, dtype="<i8")
        relevant_values = np.asarray(relevant, dtype="<i8")
        digest.update(np.asarray([len(candidate_values)], dtype="<i8").tobytes())
        digest.update(candidate_values.tobytes())
        digest.update(np.asarray([len(relevant_values)], dtype="<i8").tobytes())
        digest.update(relevant_values.tobytes())
    return digest.hexdigest()


def build_validation_contract(
    *,
    event_users: np.ndarray,
    event_items: np.ndarray,
    event_times: np.ndarray,
    event_strong: np.ndarray,
    user_indptr: np.ndarray,
    actual_user_ids: np.ndarray,
    video_ids: np.ndarray,
    normal_item_mask: np.ndarray,
    train_end: float,
    train_events: pd.DataFrame | None,
) -> tuple[RetrievalQueries, np.ndarray, dict[str, Any]]:
    """Build the exact B1A query/candidate set, optionally attaching histories."""

    fixed_positions = np.unique(event_items[normal_item_mask[event_items]])
    fixed_catalog = video_ids[fixed_positions].astype(np.int64)
    train_observed_positions = np.unique(event_items[event_times < train_end])
    data_cold_positions = np.setdiff1d(
        fixed_positions, train_observed_positions, assume_unique=True
    )
    data_cold_items = video_ids[data_cold_positions].astype(np.int64)
    history_groups = (
        {}
        if train_events is None
        else {
            int(user): frame
            for user, frame in train_events.groupby("user_id", sort=False)
        }
    )
    query_users: list[int] = []
    histories: list[np.ndarray] = []
    history_weights: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    warm: list[bool] = []
    for user_position in range(len(user_indptr) - 1):
        start = int(user_indptr[user_position])
        end = int(user_indptr[user_position + 1])
        times = event_times[start:end]
        items = event_items[start:end]
        strong = event_strong[start:end]
        normal = normal_item_mask[items]
        train_mask = times < train_end
        seen_positions = np.unique(items[train_mask])
        relevant_positions = np.unique(
            items[
                (~train_mask)
                & strong
                & normal
                & ~np.isin(items, seen_positions)
            ]
        )
        if not len(relevant_positions):
            continue
        candidate_positions = np.setdiff1d(
            fixed_positions, seen_positions, assume_unique=True
        )
        actual_user = int(actual_user_ids[user_position])
        history = history_groups.get(actual_user)
        if history is None:
            history_ids = np.asarray([], dtype=np.int64)
            weights = np.asarray([], dtype=np.float32)
        else:
            history = history.tail(50)
            history_ids = history["video_id"].to_numpy(np.int64)
            weights = _weights_from_arrays(
                history["watch_ratio"].to_numpy(np.float64),
                history["play_duration"].to_numpy(np.float64),
                history["video_duration"].to_numpy(np.float64),
                quick_skip_mask=history["_is_quick_skip"].to_numpy(bool),
            )
        query_users.append(actual_user)
        histories.append(history_ids)
        history_weights.append(weights)
        candidates.append(video_ids[candidate_positions].astype(np.int64))
        relevant.append(video_ids[relevant_positions].astype(np.int64))
        warm.append(bool(train_mask.any()))
    queries = RetrievalQueries(
        user_ids=np.asarray(query_users, dtype=np.int64),
        histories=tuple(histories),
        history_weights=tuple(history_weights),
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=fixed_catalog,
        warm_user_mask=np.asarray(warm, dtype=bool),
    )
    warm_mask = queries.warm_user_mask
    counts = {
        "fixed_catalog_count": int(len(fixed_catalog)),
        "query_count": int(len(queries.user_ids)),
        "warm_query_count": int(warm_mask.sum()),
        "target_count": int(sum(len(row) for row in queries.relevant)),
        "warm_target_count": int(
            sum(
                len(row)
                for row, is_warm in zip(
                    queries.relevant, warm_mask, strict=True
                )
                if is_warm
            )
        ),
        "data_cold_item_count": int(len(data_cold_items)),
        "query_contract_sha256": validation_query_contract_sha256(queries),
    }
    return queries, data_cold_items, counts


def verify_validation_contract(
    *,
    queries: RetrievalQueries,
    counts: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual_catalog = membership_record(
        queries.catalog, label="phase-b2a-fixed-retrieval-catalog-v1"
    )
    if actual_catalog["count"] != int(expected["fixed_catalog_count"]):
        raise RuntimeError("Fixed validation catalog count mismatch")
    if actual_catalog["sha256"] != expected["fixed_catalog_sha256"]:
        raise RuntimeError("Fixed validation catalog membership mismatch")
    for name in (
        "query_count",
        "warm_query_count",
        "target_count",
        "warm_target_count",
        "data_cold_item_count",
        "query_contract_sha256",
    ):
        if counts[name] != expected[name]:
            raise RuntimeError(
                f"Validation contract mismatch for {name}: "
                f"actual={counts[name]} expected={expected[name]}"
            )


def attach_train_histories(
    queries: RetrievalQueries,
    train_events: pd.DataFrame,
    *,
    max_history: int = 50,
) -> RetrievalQueries:
    """Attach train-only histories without rebuilding candidate memberships."""

    if max_history <= 0:
        raise ValueError("max_history must be positive")
    groups = {
        int(user): frame
        for user, frame in train_events.groupby("user_id", sort=False)
    }
    histories: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for user in queries.user_ids:
        history = groups.get(int(user))
        if history is None:
            histories.append(np.asarray([], dtype=np.int64))
            weights.append(np.asarray([], dtype=np.float32))
            continue
        history = history.tail(max_history)
        histories.append(history["video_id"].to_numpy(np.int64))
        weights.append(
            _weights_from_arrays(
                history["watch_ratio"].to_numpy(np.float64),
                history["play_duration"].to_numpy(np.float64),
                history["video_duration"].to_numpy(np.float64),
                quick_skip_mask=history["_is_quick_skip"].to_numpy(bool),
            )
        )
    return RetrievalQueries(
        user_ids=queries.user_ids,
        histories=tuple(histories),
        history_weights=tuple(weights),
        candidates=queries.candidates,
        relevant=queries.relevant,
        catalog=queries.catalog,
        warm_user_mask=queries.warm_user_mask,
        diagnostics=dict(queries.diagnostics),
    )


def planned_training_membership(
    dataset: TwoTowerTrainingDataset,
    example_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute planned IDs with array operations, without materializing examples."""

    indices = np.asarray(example_indices, dtype=np.int64)
    event_indices = dataset.positive_event_indices[indices]
    users = np.unique(dataset.user_ids[event_indices]).astype(np.int64)
    user_mask = np.isin(dataset.user_ids, users)
    items = np.unique(dataset.item_ids[user_mask]).astype(np.int64)
    return users, items


def _parameters_are_finite(model: TwoTowerV1) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def _assert_fixed_diagnostic_gradients(
    model: TwoTowerV1, history_vectors: torch.Tensor
) -> None:
    modules = {
        "item_id": model.item_id_embedding,
        "category": model.category_embedding,
        "caption_projection": model.caption_projection,
        "static_projection": model.static_projection,
        "upload_type": model.upload_type_embedding,
        "user_id": model.user_id_embedding,
        "user_mlp": model.user_mlp,
    }
    for name, module in modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not any(
            bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
            for value in gradients
        ):
            raise RuntimeError(
                f"Fixed diagnostic batch did not cover gradient branch {name}"
            )
    if (
        history_vectors.grad is None
        or not bool(torch.isfinite(history_vectors.grad).all())
        or float(history_vectors.grad.norm()) <= 0.0
    ):
        raise RuntimeError("Fixed diagnostic history gradient is invalid")


EpochCheckpointCallback = Callable[
    [
        int,
        TwoTowerV1,
        torch.optim.Optimizer,
        tuple[float, ...],
        np.ndarray,
        np.ndarray,
    ],
    None,
]


def train_full_two_tower(
    *,
    model: TwoTowerV1,
    optimizer: torch.optim.Optimizer,
    dataset: TwoTowerTrainingDataset,
    example_indices: np.ndarray,
    store: PreparedItemFeatureStore,
    ordered_user_ids: np.ndarray,
    planned_item_ids: np.ndarray,
    device: str | torch.device,
    seed: int,
    diagnostic_seed: int,
    start_epoch: int,
    end_epoch: int,
    batch_size: int,
    temperature: float,
    gradient_clip_norm: float,
    prior_epoch_losses: tuple[float, ...] = (),
    touched_user_ids: np.ndarray | None = None,
    touched_item_ids: np.ndarray | None = None,
    checkpoint_callback: EpochCheckpointCallback | None = None,
    max_total_steps: int | None = None,
    log_every_steps: int = 100,
) -> dict[str, Any]:
    """Train lazy deterministic epochs and checkpoint only at epoch boundaries."""

    target_device = assert_model_device(model, resolve_concrete_device(device))
    if start_epoch < 1 or end_epoch < start_epoch:
        raise ValueError("Epoch range is invalid")
    if len(prior_epoch_losses) != start_epoch - 1:
        raise ValueError("Prior loss count does not match resume epoch")
    if batch_size <= 1 or log_every_steps <= 0:
        raise ValueError("Batch/log interval is invalid")
    indices = np.asarray(example_indices, dtype=np.int64)
    ordered_users = np.asarray(ordered_user_ids, dtype=np.int64)
    user_positions = {
        int(user): position + 1
        for position, user in enumerate(ordered_users)
    }
    planned_users_set = set(int(value) for value in ordered_users)
    planned_items_set = set(int(value) for value in planned_item_ids)
    touched_users = set(
        int(value)
        for value in (
            np.asarray([], dtype=np.int64)
            if touched_user_ids is None
            else touched_user_ids
        )
    )
    touched_items = set(
        int(value)
        for value in (
            np.asarray([], dtype=np.int64)
            if touched_item_ids is None
            else touched_item_ids
        )
    )
    torch_store = store.torch_features(target_device)
    encoding = {
        "store": store,
        "torch_store": torch_store,
        "user_positions": user_positions,
        "touched_item_ids": planned_items_set,
        "touched_user_ids": planned_users_set,
        "device": target_device,
    }
    diagnostic_order = sorted(
        indices,
        key=lambda value: (
            hashlib.sha256(
                f"{diagnostic_seed}:{int(value)}".encode()
            ).digest(),
            int(value),
        ),
    )
    diagnostic_indices = np.asarray(
        diagnostic_order[: min(batch_size, len(diagnostic_order))],
        dtype=np.int64,
    )
    diagnostic_batch = collate_training_batch(
        dataset, diagnostic_indices, device=target_device
    )
    optimizer.zero_grad(set_to_none=True)
    diagnostic_users, diagnostic_targets, diagnostic_history = (
        encode_training_batch(model, diagnostic_batch, **encoding)
    )
    diagnostic_loss, _ = masked_in_batch_cross_entropy(
        diagnostic_users,
        diagnostic_targets,
        diagnostic_batch.allowed_logits,
        temperature=temperature,
    )
    if not bool(torch.isfinite(diagnostic_loss)):
        raise FloatingPointError("Fixed diagnostic loss became non-finite")
    diagnostic_loss.backward()
    _assert_fixed_diagnostic_gradients(model, diagnostic_history)
    optimizer.zero_grad(set_to_none=True)
    epoch_losses = list(prior_epoch_losses)
    optimizer_steps = 0
    skipped_batches = 0
    model.train()
    for epoch in range(start_epoch, end_epoch + 1):
        order = indices.copy()
        np.random.default_rng(
            np.random.SeedSequence([seed, epoch])
        ).shuffle(order)
        epoch_loss = 0.0
        epoch_examples = 0
        epoch_steps = 0
        for begin in range(0, len(order), batch_size):
            if max_total_steps is not None and optimizer_steps >= max_total_steps:
                break
            batch_indices = order[begin : begin + batch_size]
            if len(batch_indices) < 2:
                skipped_batches += 1
                continue
            batch = collate_training_batch(
                dataset, batch_indices, device=target_device
            )
            if torch.any(batch.allowed_logits.sum(dim=1) < 2):
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            users, targets, history_vectors = encode_training_batch(
                model, batch, **encoding
            )
            if not bool(torch.isfinite(users).all()) or not bool(
                torch.isfinite(targets).all()
            ):
                raise FloatingPointError("Two-Tower training vector became non-finite")
            loss, _ = masked_in_batch_cross_entropy(
                users,
                targets,
                batch.allowed_logits,
                temperature=temperature,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Two-Tower loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("Two-Tower gradient became non-finite")
            optimizer.step()
            model._zero_padding_rows()
            if not _parameters_are_finite(model):
                raise FloatingPointError("Two-Tower parameter became non-finite")
            record_successful_step_membership(
                batch,
                touched_users=touched_users,
                touched_items=touched_items,
            )
            loss_value = float(loss.detach())
            epoch_loss += loss_value * len(batch_indices)
            epoch_examples += len(batch_indices)
            optimizer_steps += 1
            epoch_steps += 1
            if optimizer_steps % log_every_steps == 0:
                print(
                    f"epoch={epoch} step={optimizer_steps} "
                    f"loss={epoch_loss / epoch_examples:.6f}",
                    flush=True,
                )
        if epoch_examples == 0:
            raise RuntimeError(f"Epoch {epoch} completed no optimizer step")
        completed_full_epoch = begin + len(batch_indices) >= len(order)
        if not completed_full_epoch:
            if checkpoint_callback is not None:
                raise RuntimeError("A partial epoch may not be checkpointed")
            break
        epoch_losses.append(epoch_loss / epoch_examples)
        touched_user_array = np.asarray(sorted(touched_users), dtype=np.int64)
        touched_item_array = np.asarray(sorted(touched_items), dtype=np.int64)
        if checkpoint_callback is not None:
            checkpoint_callback(
                epoch,
                model,
                optimizer,
                tuple(epoch_losses),
                touched_user_array,
                touched_item_array,
            )
    return {
        "epoch_losses": tuple(epoch_losses),
        "optimizer_steps": optimizer_steps,
        "skipped_batches": skipped_batches,
        "touched_user_ids": np.asarray(sorted(touched_users), dtype=np.int64),
        "touched_item_ids": np.asarray(sorted(touched_items), dtype=np.int64),
        "completed_epoch": len(epoch_losses),
        "fixed_diagnostic_loss": float(diagnostic_loss.detach()),
    }


def save_full_epoch_checkpoint(
    path: Path,
    *,
    model: TwoTowerV1,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    epoch_losses: tuple[float, ...],
    order_seed: int,
    model_dimensions: dict[str, int],
    ordered_item_ids: np.ndarray,
    ordered_user_ids: np.ndarray,
    touched_user_ids: np.ndarray,
    touched_item_ids: np.ndarray,
    identity: dict[str, Any],
) -> None:
    """Atomically publish one complete-epoch checkpoint."""

    if completed_epoch < 1 or len(epoch_losses) != completed_epoch:
        raise ValueError("Only a complete epoch may be checkpointed")
    if identity.get("schema_version") != 2:
        raise ValueError("Checkpoint identity schema must remain v2")
    payload = {
        "schema_version": 2,
        "checkpoint_kind": "phase-b2b-full-epoch-v1",
        "model_dimensions": model_dimensions,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "epoch_losses": tuple(float(value) for value in epoch_losses),
        "epoch_order_seed": {
            "base_seed": int(order_seed),
            "completed_epoch": completed_epoch,
        },
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "identity": identity,
        "identity_sha256": canonical_json_sha256(
            identity, label="phase-b2a-checkpoint-identity-v2"
        ),
        "ordered_item_ids": np.asarray(ordered_item_ids, dtype=np.int64),
        "ordered_user_ids": np.asarray(ordered_user_ids, dtype=np.int64),
        "touched_user_ids": np.asarray(touched_user_ids, dtype=np.int64),
        "touched_item_ids": np.asarray(touched_item_ids, dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_full_epoch_checkpoint(
    path: Path,
    *,
    device: str | torch.device,
    expected_identity: dict[str, Any],
    learning_rate: float,
    weight_decay: float,
) -> tuple[TwoTowerV1, torch.optim.AdamW, dict[str, Any]]:
    """Restore model, optimizer and RNG from a complete matching epoch."""

    target_device = resolve_concrete_device(device)
    model, payload = load_checkpoint(
        path, device=target_device, expected_identity=expected_identity
    )
    if payload.get("checkpoint_kind") != "phase-b2b-full-epoch-v1":
        raise RuntimeError("Checkpoint is not a complete B2B epoch")
    completed = payload.get("completed_epoch")
    losses = payload.get("epoch_losses")
    order = payload.get("epoch_order_seed")
    if (
        not isinstance(completed, int)
        or completed < 1
        or not isinstance(losses, tuple)
        or len(losses) != completed
        or order
        != {
            "base_seed": expected_identity["training_seed"],
            "completed_epoch": completed,
        }
    ):
        raise RuntimeError("Checkpoint epoch/order contract is invalid")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for name, value in state.items():
            if (
                torch.is_tensor(value)
                and name != "step"
                and value.device != target_device
            ):
                raise RuntimeError("Optimizer state is on the wrong device")
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    cuda_state = payload.get("torch_cuda_rng_state_all")
    if target_device.type == "cuda":
        if cuda_state is None:
            raise RuntimeError("CUDA checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_state)
    return model, optimizer, payload


def build_checkpoint_identity(
    *,
    base_identity: dict[str, Any],
    model_dimensions: dict[str, int],
    ordered_item_ids: np.ndarray,
    ordered_user_ids: np.ndarray,
    touched_user_ids: np.ndarray,
    touched_item_ids: np.ndarray,
    training_seed: int,
) -> dict[str, Any]:
    identity = dict(base_identity)
    identity.update(
        {
            "schema_version": 2,
            "model_dimensions": dict(model_dimensions),
            "ordered_item_store": membership_record(
                ordered_item_ids,
                label="phase-b2a-ordered-item-store-v1",
            ),
            "ordered_user_position_mapping": membership_record(
                ordered_user_ids,
                label="phase-b2a-ordered-user-position-mapping-v1",
            ),
            "actual_touched_membership": {
                "users": {
                    "count": int(len(touched_user_ids)),
                    "sha256": stable_int_membership_sha256(
                        "phase-b2a-touched-users-v1", touched_user_ids
                    ),
                },
                "items": {
                    "count": int(len(touched_item_ids)),
                    "sha256": stable_int_membership_sha256(
                        "phase-b2a-touched-items-v1", touched_item_ids
                    ),
                },
            },
            "training_seed": int(training_seed),
        }
    )
    return identity


def select_checkpoint_epoch(records: list[dict[str, Any]]) -> int:
    if not records:
        raise ValueError("No validation checkpoints are available")
    def metrics(row: dict[str, Any]) -> dict[str, float]:
        return row.get("metrics", row.get("validation", {}).get("metrics", {}))

    return max(
        records,
        key=lambda row: (
            float(metrics(row)["Recall@100"]),
            float(metrics(row)["NDCG@20"]),
            -int(row["epoch"]),
        ),
    )["epoch"]


def evaluate_frozen_gates(
    metrics: dict[str, float],
    denominators: dict[str, int],
    gate: dict[str, Any],
) -> dict[str, bool]:
    ndcg = metrics["NDCG@20"] >= gate["common_ndcg_minimum"]
    return {
        "A": ndcg
        and metrics["Recall@100"] >= gate["A"]["Recall@100_minimum"],
        "B": ndcg
        and metrics["Recall@100"] >= gate["B"]["Recall@100_minimum"]
        and metrics["Coverage@100"] >= gate["B"]["Coverage@100_minimum"],
        "C": ndcg
        and metrics["Recall@100"] >= gate["C"]["Recall@100_minimum"]
        and denominators["data_cold_target_count"]
        >= gate["C"]["data_cold_target_denominator_minimum"]
        and metrics["Data-Cold Recall@100"]
        >= gate["C"]["Data-Cold_Recall@100_minimum"],
    }
