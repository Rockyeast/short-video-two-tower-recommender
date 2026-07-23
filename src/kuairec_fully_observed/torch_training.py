"""Bounded sampling, feature preprocessing and training for Phase B2A."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .caption_embeddings import CaptionCache
from .provenance import canonical_json_sha256, membership_record
from .torch_models import (
    TorchItemFeatures,
    TwoTowerV1,
    masked_in_batch_cross_entropy,
)
from .training import TwoTowerTrainingDataset, build_in_batch_logit_mask


def _stable_key(seed: int, *values: int) -> bytes:
    body = ":".join(str(int(value)) for value in (seed, *values)).encode()
    return hashlib.sha256(body).digest()


def stable_int_membership_sha256(label: str, values: np.ndarray) -> str:
    digest = hashlib.sha256(f"{label}\n".encode())
    for value in np.asarray(values, dtype=np.int64):
        digest.update(f"{int(value)}\n".encode())
    return digest.hexdigest()


def resolve_concrete_device(
    requested_device: str | torch.device,
    *,
    cuda_available: bool | None = None,
    current_cuda_device: int | None = None,
) -> torch.device:
    """Resolve CPU/CUDA requests to one concrete device used end to end."""

    requested = torch.device(requested_device)
    if requested.type == "cpu":
        return torch.device("cpu")
    if requested.type != "cuda":
        raise ValueError(f"Unsupported Two-Tower device type: {requested.type}")
    available = (
        torch.cuda.is_available()
        if cuda_available is None
        else bool(cuda_available)
    )
    if not available:
        raise RuntimeError(
            f"CUDA device {requested} was requested but CUDA is unavailable"
        )
    index = requested.index
    if index is None:
        index = (
            torch.cuda.current_device()
            if current_cuda_device is None
            else int(current_cuda_device)
        )
    if int(index) < 0:
        raise ValueError(f"CUDA device index must be non-negative: {index}")
    return torch.device("cuda", int(index))


@dataclass(frozen=True)
class PreparedItemFeatureStore:
    item_ids: np.ndarray
    category_indices: np.ndarray
    caption_embeddings: np.ndarray
    caption_present: np.ndarray
    numeric_features: np.ndarray
    upload_type_indices: np.ndarray
    category_vocab: dict[tuple[int, int], int]
    upload_type_vocab: dict[str, int]
    preprocessing: dict[str, Any]

    @property
    def positions(self) -> dict[int, int]:
        return {int(item): index for index, item in enumerate(self.item_ids)}

    def torch_features(self, device: torch.device) -> TorchItemFeatures:
        return TorchItemFeatures(
            item_indices=torch.arange(
                1, len(self.item_ids) + 1, dtype=torch.long, device=device
            ),
            category_indices=torch.as_tensor(
                self.category_indices, dtype=torch.long, device=device
            ),
            caption_embeddings=torch.as_tensor(
                self.caption_embeddings, dtype=torch.float32, device=device
            ),
            caption_present=torch.as_tensor(
                self.caption_present, dtype=torch.bool, device=device
            ),
            numeric_features=torch.as_tensor(
                self.numeric_features, dtype=torch.float32, device=device
            ),
            upload_type_indices=torch.as_tensor(
                self.upload_type_indices, dtype=torch.long, device=device
            ),
        )


def prepare_item_feature_store(
    *,
    static_frame: pd.DataFrame,
    caption_cache: CaptionCache,
    item_universe: np.ndarray,
    train_observed_item_ids: np.ndarray,
    train_observed_normal_item_ids: np.ndarray,
) -> PreparedItemFeatureStore:
    """Fit every vocabulary/statistic only on the frozen train context."""

    item_ids = np.unique(np.asarray(item_universe, dtype=np.int64))
    if not np.array_equal(item_ids, caption_cache.item_ids):
        raise RuntimeError("Caption cache and model item universe differ")
    frame = static_frame.set_index("video_id").reindex(item_ids)
    if frame.index.has_duplicates or frame.index.isna().any():
        raise RuntimeError("Static feature identity is invalid")
    if frame["caption_text"].isna().any():
        frame["caption_text"] = frame["caption_text"].fillna("")
    train_ids = set(int(value) for value in train_observed_item_ids)
    train_normal = set(int(value) for value in train_observed_normal_item_ids)
    train_mask = np.asarray([int(item) in train_ids for item in item_ids])
    train_normal_mask = np.asarray([int(item) in train_normal for item in item_ids])
    if not np.any(train_normal_mask):
        raise ValueError("Numeric preprocessing needs train-observed NORMAL items")

    category_vocab: dict[tuple[int, int], int] = {}
    for item_id, categories in zip(item_ids[train_mask], frame.loc[train_mask, "category_ids"], strict=True):
        del item_id
        for level, raw in enumerate(categories):
            token = (level, int(raw))
            if int(raw) >= 0 and token not in category_vocab:
                category_vocab[token] = len(category_vocab) + 1
    category_indices = np.zeros((len(item_ids), 3), dtype=np.int64)
    for row, categories in enumerate(frame["category_ids"]):
        for level, raw in enumerate(categories):
            category_indices[row, level] = category_vocab.get(
                (level, int(raw)), 0
            )

    upload_values = frame["upload_type"].fillna("").astype(str).str.strip()
    upload_type_vocab: dict[str, int] = {}
    for value in sorted(set(upload_values[train_mask]) - {"", "UNKNOWN"}):
        upload_type_vocab[value] = len(upload_type_vocab) + 1
    upload_indices = np.asarray(
        [upload_type_vocab.get(value, 0) for value in upload_values],
        dtype=np.int64,
    )

    numeric = np.empty((len(item_ids), 4), dtype=np.float64)
    for column, target in zip(
        ("video_duration", "video_width", "video_height"),
        range(3),
        strict=True,
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            np.float64
        )
        numeric[:, target] = np.log1p(np.maximum(values, 0.0))
    dates = pd.to_datetime(frame["upload_dt"], errors="coerce", utc=True)
    numeric[:, 3] = dates.map(
        lambda value: np.nan if pd.isna(value) else float(value.toordinal())
    ).to_numpy(np.float64)
    missing_counts = {
        name: int(np.isnan(numeric[:, index]).sum())
        for index, name in enumerate(
            ("video_duration", "video_width", "video_height", "upload_dt")
        )
    }
    fit = numeric[train_normal_mask].copy()
    medians = np.nanmedian(fit, axis=0)
    if not np.isfinite(medians).all():
        raise RuntimeError("Static numeric train medians are non-finite")
    numeric = np.where(np.isnan(numeric), medians, numeric)
    fit_filled = np.where(np.isnan(fit), medians, fit)
    means = fit_filled.mean(axis=0)
    stds = fit_filled.std(axis=0)
    stds[stds < 1e-12] = 1.0
    normalized = ((numeric - means) / stds).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError("Static numeric preprocessing produced NaN or Inf")

    captions = np.asarray(caption_cache.embeddings, dtype=np.float32)
    present = np.linalg.norm(captions, axis=1) > 0
    return PreparedItemFeatureStore(
        item_ids=item_ids,
        category_indices=category_indices,
        caption_embeddings=captions,
        caption_present=present,
        numeric_features=normalized,
        upload_type_indices=upload_indices,
        category_vocab=category_vocab,
        upload_type_vocab=upload_type_vocab,
        preprocessing={
            "numeric_fields": [
                "log1p(video_duration)",
                "log1p(video_width)",
                "log1p(video_height)",
                "upload_dt_ordinal",
            ],
            "numeric_scaler_fit_scope": "train-observed NORMAL items",
            "category_upload_vocab_fit_scope": (
                "all train-observed history/model-context items"
            ),
            "medians": medians.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "missing_value_counts": missing_counts,
            "category_vocab_size": len(category_vocab),
            "upload_type_vocab_size": len(upload_type_vocab),
            "unseen_category_uses_zero_unk": True,
            "unseen_upload_type_uses_zero_unk": True,
        },
    )


def sample_bounded_example_indices(
    dataset: TwoTowerTrainingDataset,
    *,
    seed: int,
    max_users: int = 256,
    max_examples_per_user: int = 32,
    max_examples: int = 8192,
    min_users: int = 64,
    min_examples: int = 2000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hash-sample across the complete example population, never a row prefix."""

    example_users = dataset.user_ids[dataset.positive_event_indices]
    groups: dict[int, list[int]] = {}
    for index, user in enumerate(example_users):
        groups.setdefault(int(user), []).append(index)
    selected_users = sorted(
        groups, key=lambda user: (_stable_key(seed, user), user)
    )[:max_users]
    chosen: list[int] = []
    for user in selected_users:
        rows = sorted(
            groups[user], key=lambda index: (_stable_key(seed, user, index), index)
        )[:max_examples_per_user]
        chosen.extend(rows)
    chosen = sorted(
        chosen, key=lambda index: (_stable_key(seed, index, 991), index)
    )[:max_examples]
    indices = np.asarray(sorted(chosen), dtype=np.int64)
    chosen_users = np.unique(example_users[indices])
    if len(chosen_users) < min_users or len(indices) < min_examples:
        raise RuntimeError(
            f"Bounded sample is too small: users={len(chosen_users)} "
            f"examples={len(indices)}"
        )
    digest = hashlib.sha256(b"phase-b2a-training-example-sample-v1\n")
    item_values: set[int] = set()
    for index in indices:
        example = dataset[int(index)]
        item_values.add(int(example.target_item_id))
        item_values.update(int(item) for item in example.history)
        digest.update(
            f"{int(example.user_id)}|{int(example.target_item_id)}|"
            f"{example.target_timestamp:.6f}|{int(index)}\n".encode()
        )
    return indices, {
        "source_example_population": int(len(dataset)),
        "sampled_users": int(len(chosen_users)),
        "sampled_examples": int(len(indices)),
        "sampled_items": int(len(item_values)),
        "sample_membership_sha256": digest.hexdigest(),
        "not_csv_prefix": not np.array_equal(
            indices, np.arange(len(indices), dtype=np.int64)
        ),
        "minimum_example_index": int(indices.min()),
        "maximum_example_index": int(indices.max()),
    }


@dataclass(frozen=True)
class TrainingBatch:
    user_ids: np.ndarray
    target_item_ids: np.ndarray
    history_item_ids: np.ndarray
    history_weights: torch.Tensor
    history_mask: torch.Tensor
    allowed_logits: torch.Tensor


def record_successful_step_membership(
    batch: TrainingBatch,
    *,
    touched_users: set[int],
    touched_items: set[int],
) -> None:
    """Record IDs only after a forward/backward/optimizer step succeeds."""

    touched_users.update(int(user) for user in batch.user_ids)
    touched_items.update(int(item) for item in batch.target_item_ids)
    touched_items.update(
        int(item)
        for item in batch.history_item_ids.reshape(-1)
        if int(item) >= 0
    )


def collate_training_batch(
    dataset: TwoTowerTrainingDataset,
    example_indices: np.ndarray,
    *,
    device: torch.device,
) -> TrainingBatch:
    examples = [dataset[int(index)] for index in example_indices]
    users = np.asarray([row.user_id for row in examples], dtype=np.int64)
    targets = np.asarray([row.target_item_id for row in examples], dtype=np.int64)
    width = max(1, max(len(row.history) for row in examples))
    histories = np.full((len(examples), width), -1, dtype=np.int64)
    weights = np.zeros((len(examples), width), dtype=np.float32)
    mask = np.zeros((len(examples), width), dtype=bool)
    for row_index, example in enumerate(examples):
        count = len(example.history)
        histories[row_index, :count] = example.history
        weights[row_index, :count] = example.history_weights
        mask[row_index, :count] = True
    allowed = build_in_batch_logit_mask(
        users, targets, dataset.known_positive_items
    )
    return TrainingBatch(
        user_ids=users,
        target_item_ids=targets,
        history_item_ids=histories,
        history_weights=torch.as_tensor(weights, device=device),
        history_mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
        allowed_logits=torch.as_tensor(allowed, dtype=torch.bool, device=device),
    )


def _positions(values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    try:
        return np.asarray([mapping[int(value)] for value in values], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Item {exc.args[0]} is outside the model universe") from exc


def encode_item_ids(
    model: TwoTowerV1,
    store: PreparedItemFeatureStore,
    torch_store: TorchItemFeatures,
    item_ids: np.ndarray,
    *,
    touched_item_ids: set[int],
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(item_ids, dtype=np.int64)
    flat = values.reshape(-1)
    positions = _positions(flat, store.positions)
    position_tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
    use_id = torch.as_tensor(
        [int(item) in touched_item_ids for item in flat],
        dtype=torch.bool,
        device=device,
    )
    encoded = model.encode_items(
        torch_store.select(position_tensor), use_id_embedding=use_id
    )
    return encoded.reshape(*values.shape, 128)


def assert_model_device(
    model: TwoTowerV1, expected_device: str | torch.device
) -> torch.device:
    expected = resolve_concrete_device(expected_device)
    devices = {parameter.device for parameter in model.parameters()}
    if devices != {expected}:
        raise RuntimeError(
            f"Model parameter devices {sorted(map(str, devices))} "
            f"do not match requested device {expected}"
        )
    return expected


def preencode_item_universe(
    *,
    model: TwoTowerV1,
    store: PreparedItemFeatureStore,
    touched_item_ids: set[int],
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode the complete item universe once without constructing a graph."""

    if batch_size <= 0:
        raise ValueError("Item encoding batch_size must be positive")
    target_device = assert_model_device(model, device)
    torch_store = store.torch_features(target_device)
    output = torch.empty(
        (len(store.item_ids), 128), dtype=torch.float32, device=target_device
    )
    with torch.inference_mode():
        for begin in range(0, len(store.item_ids), batch_size):
            end = min(begin + batch_size, len(store.item_ids))
            positions = torch.arange(
                begin, end, dtype=torch.long, device=target_device
            )
            use_id = torch.as_tensor(
                [
                    int(item) in touched_item_ids
                    for item in store.item_ids[begin:end]
                ],
                dtype=torch.bool,
                device=target_device,
            )
            output[begin:end] = model.encode_items(
                torch_store.select(positions), use_id_embedding=use_id
            )
    if output.requires_grad or output.grad_fn is not None:
        raise RuntimeError("Precomputed item vectors retained an autograd graph")
    return output


def encode_query_users_from_precomputed(
    *,
    model: TwoTowerV1,
    store: PreparedItemFeatureStore,
    precomputed_item_vectors: torch.Tensor,
    user_ids: np.ndarray,
    histories: tuple[np.ndarray, ...],
    history_weights: tuple[np.ndarray, ...],
    user_positions: dict[int, int],
    touched_user_ids: set[int],
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Gather cached history vectors and batch the user tower under inference."""

    if batch_size <= 0:
        raise ValueError("User encoding batch_size must be positive")
    target_device = assert_model_device(model, device)
    if precomputed_item_vectors.device != target_device:
        raise RuntimeError("Precomputed item vectors are on the wrong device")
    if (
        precomputed_item_vectors.shape != (len(store.item_ids), 128)
        or precomputed_item_vectors.requires_grad
        or precomputed_item_vectors.grad_fn is not None
    ):
        raise RuntimeError("Precomputed item-vector contract is invalid")
    if not (
        len(user_ids) == len(histories) == len(history_weights)
    ):
        raise ValueError("Query user/history arrays differ in length")
    positions = store.positions
    output = torch.empty(
        (len(user_ids), 128), dtype=torch.float32, device=target_device
    )
    with torch.inference_mode():
        for begin in range(0, len(user_ids), batch_size):
            end = min(begin + batch_size, len(user_ids))
            batch_histories = histories[begin:end]
            width = max(1, max(len(value) for value in batch_histories))
            history = torch.zeros(
                (end - begin, width, 128),
                dtype=torch.float32,
                device=target_device,
            )
            weights = torch.zeros(
                (end - begin, width),
                dtype=torch.float32,
                device=target_device,
            )
            padding = torch.zeros(
                (end - begin, width), dtype=torch.bool, device=target_device
            )
            for row, (item_ids, values) in enumerate(
                zip(
                    batch_histories,
                    history_weights[begin:end],
                    strict=True,
                )
            ):
                if not len(item_ids):
                    continue
                item_positions = _positions(
                    np.asarray(item_ids, dtype=np.int64), positions
                )
                position_tensor = torch.as_tensor(
                    item_positions, dtype=torch.long, device=target_device
                )
                history[row, : len(item_ids)] = precomputed_item_vectors[
                    position_tensor
                ]
                weights[row, : len(item_ids)] = torch.as_tensor(
                    values, dtype=torch.float32, device=target_device
                )
                padding[row, : len(item_ids)] = True
            batch_users = np.asarray(user_ids[begin:end], dtype=np.int64)
            user_index = torch.as_tensor(
                [user_positions.get(int(user), 0) for user in batch_users],
                dtype=torch.long,
                device=target_device,
            )
            use_id = torch.as_tensor(
                [int(user) in touched_user_ids for user in batch_users],
                dtype=torch.bool,
                device=target_device,
            )
            output[begin:end] = model.encode_users(
                user_indices=user_index,
                history_vectors=history,
                history_weights=weights,
                padding_mask=padding,
                use_id_embedding=use_id,
            )
    if output.requires_grad or output.grad_fn is not None:
        raise RuntimeError("Query vectors retained an autograd graph")
    return output


def encode_training_batch(
    model: TwoTowerV1,
    batch: TrainingBatch,
    *,
    store: PreparedItemFeatureStore,
    torch_store: TorchItemFeatures,
    user_positions: dict[int, int],
    touched_item_ids: set[int],
    touched_user_ids: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = encode_item_ids(
        model,
        store,
        torch_store,
        batch.target_item_ids,
        touched_item_ids=touched_item_ids,
        device=device,
    )
    history_vectors = torch.zeros(
        (*batch.history_item_ids.shape, 128), dtype=torch.float32, device=device
    )
    valid = batch.history_item_ids >= 0
    if np.any(valid):
        valid_vectors = encode_item_ids(
            model,
            store,
            torch_store,
            batch.history_item_ids[valid],
            touched_item_ids=touched_item_ids,
            device=device,
        )
        valid_positions = torch.as_tensor(
            np.flatnonzero(valid), dtype=torch.long, device=device
        )
        history_vectors = history_vectors.reshape(-1, 128).index_copy(
            0, valid_positions, valid_vectors
        ).reshape(*batch.history_item_ids.shape, 128)
    if history_vectors.requires_grad:
        history_vectors.retain_grad()
    user_index = torch.as_tensor(
        [user_positions.get(int(user), 0) for user in batch.user_ids],
        dtype=torch.long,
        device=device,
    )
    use_user_id = torch.as_tensor(
        [int(user) in touched_user_ids for user in batch.user_ids],
        dtype=torch.bool,
        device=device,
    )
    users = model.encode_users(
        user_indices=user_index,
        history_vectors=history_vectors,
        history_weights=batch.history_weights,
        padding_mask=batch.history_mask,
        use_id_embedding=use_user_id,
    )
    return users, targets, history_vectors


def _parameter_checksum(model: TwoTowerV1) -> str:
    digest = hashlib.sha256(b"phase-b2a-parameters-v1\n")
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _diagnostic(
    model: TwoTowerV1,
    batch: TrainingBatch,
    **encoding: Any,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        users, targets, _ = encode_training_batch(model, batch, **encoding)
        loss, stats = masked_in_batch_cross_entropy(
            users, targets, batch.allowed_logits, temperature=0.07
        )
    return {"loss": float(loss.cpu()), **stats}


def train_bounded_two_tower(
    *,
    model: TwoTowerV1,
    dataset: TwoTowerTrainingDataset,
    sampled_indices: np.ndarray,
    store: PreparedItemFeatureStore,
    seed: int,
    diagnostic_seed: int,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    gradient_clip_norm: float = 5.0,
) -> dict[str, Any]:
    """Train one fixed smoke model and return optimization diagnostics."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    examples = [dataset[int(index)] for index in sampled_indices]
    planned_users = {int(row.user_id) for row in examples}
    planned_items = {int(row.target_item_id) for row in examples}
    for row in examples:
        planned_items.update(int(item) for item in row.history)
    user_ids = np.asarray(sorted(planned_users), dtype=np.int64)
    user_positions = {int(user): index + 1 for index, user in enumerate(user_ids)}
    torch_store = store.torch_features(device)
    encoding = {
        "store": store,
        "torch_store": torch_store,
        "user_positions": user_positions,
        "touched_item_ids": planned_items,
        "touched_user_ids": planned_users,
        "device": device,
    }
    diagnostic_order = sorted(
        sampled_indices,
        key=lambda index: (_stable_key(diagnostic_seed, int(index)), int(index)),
    )
    diagnostic_indices = np.asarray(
        diagnostic_order[: min(batch_size, len(diagnostic_order))], dtype=np.int64
    )
    diagnostic_batch = collate_training_batch(
        dataset, diagnostic_indices, device=device
    )
    initial_diagnostic = _diagnostic(model, diagnostic_batch, **encoding)
    before_checksum = _parameter_checksum(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    order = np.asarray(sampled_indices, dtype=np.int64).copy()
    epoch_losses: list[float] = []
    optimizer_steps = 0
    skipped_batches = 0
    actual_touched_users: set[int] = set()
    actual_touched_items: set[int] = set()
    gradient_sums = {
        name: 0.0
        for name in (
            "item_id",
            "category",
            "caption_projection",
            "static_projection",
            "upload_type",
            "user_id",
            "history_path",
        )
    }
    gradient_observations = {name: 0 for name in gradient_sums}
    duplicate_rates: list[float] = []
    valid_negative_counts: list[np.ndarray] = []
    masked_fractions: list[float] = []
    module_parameters = {
        "item_id": tuple(model.item_id_embedding.parameters()),
        "category": tuple(model.category_embedding.parameters()),
        "caption_projection": tuple(model.caption_projection.parameters()),
        "static_projection": tuple(model.static_projection.parameters()),
        "upload_type": tuple(model.upload_type_embedding.parameters()),
        "user_id": tuple(model.user_id_embedding.parameters()),
    }
    model.train()
    for epoch in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_examples = 0
        for begin in range(0, len(order), batch_size):
            indices = order[begin : begin + batch_size]
            if len(indices) < 2:
                skipped_batches += 1
                continue
            batch = collate_training_batch(dataset, indices, device=device)
            valid_negatives = batch.allowed_logits.sum(dim=1).cpu().numpy() - 1
            if np.any(valid_negatives < 1):
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            users, targets, history_vectors = encode_training_batch(
                model, batch, **encoding
            )
            loss, stats = masked_in_batch_cross_entropy(
                users,
                targets,
                batch.allowed_logits,
                temperature=temperature,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Two-Tower loss became non-finite")
            loss.backward()
            for name, parameters in module_parameters.items():
                squared = sum(
                    float(parameter.grad.detach().pow(2).sum().cpu())
                    for parameter in parameters
                    if parameter.grad is not None
                )
                norm = math.sqrt(squared)
                if not math.isfinite(norm):
                    raise FloatingPointError(f"Non-finite {name} gradient")
                gradient_sums[name] += norm
                gradient_observations[name] += 1
            history_norm = (
                0.0
                if history_vectors.grad is None
                else float(history_vectors.grad.detach().norm().cpu())
            )
            if not math.isfinite(history_norm):
                raise FloatingPointError("Non-finite history-path gradient")
            gradient_sums["history_path"] += history_norm
            gradient_observations["history_path"] += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            model._zero_padding_rows()
            record_successful_step_membership(
                batch,
                touched_users=actual_touched_users,
                touched_items=actual_touched_items,
            )
            optimizer_steps += 1
            epoch_loss += float(loss.detach().cpu()) * len(indices)
            epoch_examples += len(indices)
            duplicate_rates.append(
                1.0 - len(np.unique(batch.target_item_ids)) / len(batch.target_item_ids)
            )
            valid_negative_counts.append(valid_negatives.astype(np.int64))
            masked_fractions.append(stats["off_diagonal_masked_fraction"])
        if epoch_examples == 0:
            raise RuntimeError("Every Two-Tower batch was skipped")
        epoch_losses.append(epoch_loss / epoch_examples)
    final_diagnostic = _diagnostic(model, diagnostic_batch, **encoding)
    if not final_diagnostic["loss"] < initial_diagnostic["loss"]:
        raise RuntimeError("Fixed diagnostic loss did not decrease")
    gradient_norms = {
        name: gradient_sums[name] / max(1, gradient_observations[name])
        for name in gradient_sums
    }
    for name, value in gradient_norms.items():
        if value <= 0 or not math.isfinite(value):
            raise RuntimeError(f"Covered module has no finite nonzero gradient: {name}")
    valid = np.concatenate(valid_negative_counts)
    touched_user_ids = np.asarray(
        sorted(actual_touched_users), dtype=np.int64
    )
    touched_items_array = np.asarray(
        sorted(actual_touched_items), dtype=np.int64
    )
    after_checksum = _parameter_checksum(model)
    if before_checksum == after_checksum:
        raise RuntimeError("Two-Tower parameters did not change")
    return {
        "initial_fixed_diagnostic": initial_diagnostic,
        "final_fixed_diagnostic": final_diagnostic,
        "diagnostic_definition": "separate-seed fixed diagnostic",
        "epoch_train_loss": epoch_losses,
        "optimizer_steps": optimizer_steps,
        "skipped_batches": skipped_batches,
        "gradient_norms_mean": gradient_norms,
        "parameter_checksum_before": before_checksum,
        "parameter_checksum_after": after_checksum,
        "touched_user_count": len(actual_touched_users),
        "touched_user_membership_sha256": stable_int_membership_sha256(
            "phase-b2a-touched-users-v1", touched_user_ids
        ),
        "touched_item_count": len(actual_touched_items),
        "touched_item_membership_sha256": stable_int_membership_sha256(
            "phase-b2a-touched-items-v1", touched_items_array
        ),
        "touched_user_ids": touched_user_ids,
        "touched_item_ids": touched_items_array,
        "ordered_user_ids": user_ids,
        "user_positions": user_positions,
        "false_negative_mask": {
            "off_diagonal_masked_fraction_mean": float(np.mean(masked_fractions)),
            "valid_negative_p5": float(np.percentile(valid, 5)),
            "valid_negative_median": float(np.median(valid)),
            "duplicate_target_rate_mean": float(np.mean(duplicate_rates)),
        },
    }


def save_checkpoint(
    path: Path,
    *,
    model: TwoTowerV1,
    model_dimensions: dict[str, int],
    ordered_item_ids: np.ndarray,
    ordered_user_ids: np.ndarray,
    touched_user_ids: np.ndarray,
    touched_item_ids: np.ndarray,
    identity: dict[str, Any],
) -> None:
    if identity.get("schema_version") != 2:
        raise ValueError("Phase B2A.1 checkpoint identity schema must be 2")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "model_dimensions": model_dimensions,
            "state_dict": model.state_dict(),
            "identity": identity,
            "identity_sha256": canonical_json_sha256(
                identity, label="phase-b2a-checkpoint-identity-v2"
            ),
            "ordered_item_ids": np.asarray(
                ordered_item_ids, dtype=np.int64
            ),
            "ordered_user_ids": np.asarray(
                ordered_user_ids, dtype=np.int64
            ),
            "touched_user_ids": np.asarray(touched_user_ids, dtype=np.int64),
            "touched_item_ids": np.asarray(touched_item_ids, dtype=np.int64),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device,
    expected_identity: dict[str, Any],
) -> tuple[TwoTowerV1, dict[str, Any]]:
    target_device = resolve_concrete_device(device)
    payload = torch.load(
        path, map_location=target_device, weights_only=False
    )
    if payload.get("schema_version") != 2:
        raise RuntimeError("Unsupported Two-Tower checkpoint schema")
    actual_identity = payload.get("identity")
    if actual_identity != expected_identity:
        raise RuntimeError("Two-Tower checkpoint identity mismatch")
    expected_identity_sha = canonical_json_sha256(
        expected_identity, label="phase-b2a-checkpoint-identity-v2"
    )
    if payload.get("identity_sha256") != expected_identity_sha:
        raise RuntimeError("Two-Tower checkpoint identity SHA mismatch")
    dimensions = payload.get("model_dimensions")
    expected_dimensions = expected_identity.get("model_dimensions")
    required_dimensions = {
        "num_items",
        "num_users",
        "num_category_tokens",
        "num_upload_types",
    }
    if (
        not isinstance(dimensions, dict)
        or set(dimensions) != required_dimensions
        or dimensions != expected_dimensions
    ):
        raise RuntimeError("Checkpoint model dimensions identity mismatch")
    ordered_items = np.asarray(payload.get("ordered_item_ids"), dtype=np.int64)
    ordered_users = np.asarray(payload.get("ordered_user_ids"), dtype=np.int64)
    touched_users = np.asarray(payload.get("touched_user_ids"), dtype=np.int64)
    touched_items = np.asarray(payload.get("touched_item_ids"), dtype=np.int64)
    if not np.array_equal(ordered_items, np.unique(ordered_items)):
        raise RuntimeError("Checkpoint ordered item mapping is invalid")
    if not np.array_equal(ordered_users, np.unique(ordered_users)):
        raise RuntimeError("Checkpoint ordered user mapping is invalid")
    if not np.array_equal(touched_users, np.unique(touched_users)):
        raise RuntimeError("Checkpoint touched user membership is invalid")
    if not np.array_equal(touched_items, np.unique(touched_items)):
        raise RuntimeError("Checkpoint touched item membership is invalid")
    if not set(touched_users).issubset(set(ordered_users)):
        raise RuntimeError("Checkpoint touched users escape the user mapping")
    if not set(touched_items).issubset(set(ordered_items)):
        raise RuntimeError("Checkpoint touched items escape the item mapping")
    if dimensions["num_items"] != len(ordered_items):
        raise RuntimeError("Checkpoint item mapping and model dimensions differ")
    if dimensions["num_users"] != len(ordered_users):
        raise RuntimeError("Checkpoint user mapping and model dimensions differ")
    feature_identity = expected_identity.get("feature_identity", {})
    if (
        dimensions["num_category_tokens"]
        != feature_identity.get("category_vocab_count")
        or dimensions["num_upload_types"]
        != feature_identity.get("upload_type_vocab_count")
    ):
        raise RuntimeError("Checkpoint feature vocabulary dimensions differ")
    ordered_item_identity = expected_identity["ordered_item_store"]
    if membership_record(
        ordered_items,
        label="phase-b2a-ordered-item-store-v1",
    ) != ordered_item_identity:
        raise RuntimeError("Checkpoint ordered item mapping identity mismatch")
    ordered_identity = expected_identity["ordered_user_position_mapping"]
    if membership_record(
        ordered_users,
        label="phase-b2a-ordered-user-position-mapping-v1",
    ) != ordered_identity:
        raise RuntimeError("Checkpoint ordered user mapping identity mismatch")
    touched_identity = expected_identity["actual_touched_membership"]
    if (
        int(len(touched_users)) != touched_identity["users"]["count"]
        or stable_int_membership_sha256(
            "phase-b2a-touched-users-v1", touched_users
        )
        != touched_identity["users"]["sha256"]
        or int(len(touched_items)) != touched_identity["items"]["count"]
        or stable_int_membership_sha256(
            "phase-b2a-touched-items-v1", touched_items
        )
        != touched_identity["items"]["sha256"]
    ):
        raise RuntimeError("Checkpoint touched membership identity mismatch")
    model = TwoTowerV1(**dimensions).to(target_device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model._zero_padding_rows()
    assert_model_device(model, target_device)
    for value in model.state_dict().values():
        if value.device != target_device:
            raise RuntimeError("Checkpoint state tensor is on the wrong device")
    return model, payload
