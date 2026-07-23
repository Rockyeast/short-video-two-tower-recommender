#!/usr/bin/env python3
"""Run one bounded real-data PyTorch Two-Tower engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from kuairec_fully_observed import (
    ExactDotProductRetriever,
    PopularityBaseline,
    RetrievalQueries,
    evaluate_retrieval,
    load_static_item_features,
)
from kuairec_fully_observed.caption_embeddings import (
    CAPTION_MODEL_ID,
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.provenance import (
    PHASE1_PROCESSED_MANIFEST_SHA256,
    canonical_json_sha256,
    membership_record,
    normal_membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.torch_models import TwoTowerV1
from kuairec_fully_observed.torch_training import (
    assert_model_device,
    encode_query_users_from_precomputed,
    load_checkpoint,
    preencode_item_universe,
    prepare_item_feature_store,
    sample_bounded_example_indices,
    save_checkpoint,
    train_bounded_two_tower,
)
from kuairec_fully_observed.training import (
    _weights_from_arrays,
    build_two_tower_training_dataset,
)

ITEM_ENCODING_BATCH_SIZE = 1024
USER_ENCODING_BATCH_SIZE = 128


def _stable_key(seed: int, *values: int) -> bytes:
    body = ":".join(str(int(value)) for value in (seed, *values)).encode()
    return hashlib.sha256(body).digest()


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 4)


def validate_smoke_config(config: dict[str, Any]) -> None:
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size": 256,
        "epochs": 3,
        "temperature": 0.07,
        "gradient_clip_norm": 5.0,
        "seed": 20260722,
        "diagnostic_seed": 20260723,
        "precision": "FP32",
        "num_workers": 0,
    }
    if config.get("training") != expected_training:
        raise RuntimeError("Phase B2A smoke training configuration is not frozen")
    if config.get("scope", {}).get("forbidden") != [
        "small_matrix",
        "temporal_final",
        "full_two_tower",
    ]:
        raise RuntimeError("Phase B2A forbidden scope changed")
    if config.get("claims") != {
        "sampled_catalog_smoke": True,
        "comparable_to_b1a": False,
        "effectiveness_claim": False,
        "formal_gate_executed": False,
    }:
        raise RuntimeError("Phase B2A claim boundary changed")


def _verify_artifacts(
    artifact_dir: Path, manifest: dict[str, Any]
) -> None:
    if manifest.get("artifact_scope") != "train_and_validation_only":
        raise RuntimeError("Smoke artifacts are not train/validation-only")
    stats = manifest.get("statistics", {})
    if stats.get("small_matrix_rows_read") != 0:
        raise RuntimeError("Smoke artifacts accessed Small Matrix")
    if stats.get("temporal_final_rows_persisted") != 0:
        raise RuntimeError("Smoke artifacts contain temporal final")
    for name in ("events_train_validation.npz", "catalog.npz"):
        if sha256_file(artifact_dir / name) != manifest["files"][name]:
            raise RuntimeError(f"Processed artifact SHA mismatch: {name}")
def _select_proxy_training_users(
    *,
    event_users: np.ndarray,
    event_items: np.ndarray,
    event_times: np.ndarray,
    event_strong: np.ndarray,
    user_indptr: np.ndarray,
    normal_item_mask: np.ndarray,
    train_end: float,
    actual_user_ids: np.ndarray,
    seed: int,
    maximum: int,
) -> np.ndarray:
    eligible: list[int] = []
    for user_position in range(len(user_indptr) - 1):
        start = int(user_indptr[user_position])
        end = int(user_indptr[user_position + 1])
        seen: set[int] = set()
        count = 0
        for item, timestamp, strong in zip(
            event_items[start:end],
            event_times[start:end],
            event_strong[start:end],
            strict=True,
        ):
            if timestamp >= train_end:
                break
            item = int(item)
            first = item not in seen
            seen.add(item)
            if first and strong and normal_item_mask[item]:
                count += 1
        if count:
            eligible.append(int(actual_user_ids[user_position]))
    return np.asarray(
        sorted(eligible, key=lambda user: (_stable_key(seed, user), user))[
            :maximum
        ],
        dtype=np.int64,
    )


def _validation_proxy_rows(
    *,
    event_items: np.ndarray,
    event_times: np.ndarray,
    event_strong: np.ndarray,
    user_indptr: np.ndarray,
    normal_item_mask: np.ndarray,
    video_ids: np.ndarray,
    actual_user_ids: np.ndarray,
    train_end: float,
    seed: int,
    maximum: int,
) -> list[dict[str, Any]]:
    fixed_positions = np.unique(event_items[normal_item_mask[event_items]])
    train_observed = np.unique(event_items[event_times < train_end])
    data_cold_positions = set(
        int(value)
        for value in np.setdiff1d(
            fixed_positions, train_observed, assume_unique=True
        )
    )
    rows: list[dict[str, Any]] = []
    for user_position in range(len(user_indptr) - 1):
        start = int(user_indptr[user_position])
        end = int(user_indptr[user_position + 1])
        times = event_times[start:end]
        items = event_items[start:end]
        strong = event_strong[start:end]
        train = times < train_end
        if not np.any(train):
            continue
        seen_positions = np.unique(items[train])
        relevant_positions = np.unique(
            items[(~train) & strong & normal_item_mask[items] & ~np.isin(items, seen_positions)]
        )
        if not len(relevant_positions):
            continue
        candidate_positions = np.setdiff1d(
            fixed_positions, seen_positions, assume_unique=True
        )
        rows.append(
            {
                "user_position": user_position,
                "user_id": int(actual_user_ids[user_position]),
                "seen": video_ids[seen_positions].astype(np.int64),
                "candidates": video_ids[candidate_positions].astype(np.int64),
                "relevant": video_ids[relevant_positions].astype(np.int64),
                "has_data_cold_relevant": any(
                    int(value) in data_cold_positions for value in relevant_positions
                ),
            }
        )
    cold = sorted(
        (row for row in rows if row["has_data_cold_relevant"]),
        key=lambda row: (_stable_key(seed, row["user_id"], 1), row["user_id"]),
    )
    other = sorted(
        (row for row in rows if not row["has_data_cold_relevant"]),
        key=lambda row: (_stable_key(seed, row["user_id"], 2), row["user_id"]),
    )
    cold_quota = min(32, len(cold), maximum)
    selected = cold[:cold_quota] + other[: maximum - cold_quota]
    return sorted(selected, key=lambda row: row["user_id"])


def _load_selected_train_events(
    data_dir: Path, user_ids: set[int], train_end: float
) -> pd.DataFrame:
    helper_path = Path(__file__).with_name("audit_phase0.py")
    spec = importlib.util.spec_from_file_location(
        "_phase_b2a_audit_helpers", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load canonical event helpers")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    try:
        spec.loader.exec_module(helper)
    finally:
        sys.modules.pop(spec.name, None)
    event_columns = helper.EVENT_COLUMNS
    canonicalize_behavior_events = helper.canonicalize_behavior_events

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        data_dir / "big_matrix.csv",
        usecols=event_columns,
        chunksize=500_000,
    ):
        selected = chunk[chunk["user_id"].isin(user_ids)]
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise RuntimeError("No selected Big Matrix users were found")
    raw = pd.concat(chunks, ignore_index=True)
    canonical_rows: list[pd.DataFrame] = []
    for _, frame in raw.groupby("user_id", sort=True):
        canonical, _, _, _ = canonicalize_behavior_events(frame)
        canonical_rows.append(canonical[list(event_columns)])
    canonical = pd.concat(canonical_rows, ignore_index=True)
    canonical = canonical[canonical["timestamp"] < train_end]
    return canonical.sort_values(
        ["user_id", "timestamp", "video_id"], kind="mergesort"
    ).reset_index(drop=True)


def _sampled_queries(
    rows: list[dict[str, Any]],
    train_events: pd.DataFrame,
    *,
    fixed_catalog: np.ndarray,
    smoke_catalog_size: int,
    minimum_candidates: int,
    seed: int,
) -> tuple[RetrievalQueries, dict[str, Any], tuple[set[int], ...]]:
    groups = {
        int(user): frame
        for user, frame in train_events.groupby("user_id", sort=False)
    }
    histories: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    required: set[int] = set()
    seen_sets: list[set[int]] = []
    for row in rows:
        history = groups[int(row["user_id"])].tail(50)
        history_ids = history["video_id"].to_numpy(np.int64)
        histories.append(history_ids)
        weights.append(
            _weights_from_arrays(
                history["watch_ratio"].to_numpy(np.float64),
                history["play_duration"].to_numpy(np.float64),
                history["video_duration"].to_numpy(np.float64),
            )
        )
        # Histories may legally contain non-NORMAL context and are guaranteed
        # by the larger model item universe. Retrieval catalog membership is
        # NORMAL-only, so only relevant targets are mandatory catalog members.
        required.update(int(item) for item in row["relevant"])
        seen_sets.append(set(int(item) for item in row["seen"]))
    if len(required) > smoke_catalog_size:
        raise RuntimeError(
            f"Required sampled-query items exceed smoke catalog: {len(required)}"
        )
    distractors = sorted(
        (int(item) for item in fixed_catalog if int(item) not in required),
        key=lambda item: (_stable_key(seed, item, 77), item),
    )[: smoke_catalog_size - len(required)]
    smoke_catalog = np.asarray(sorted(required | set(distractors)), dtype=np.int64)
    candidates: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    for row, seen in zip(rows, seen_sets, strict=True):
        original = set(int(item) for item in row["candidates"])
        candidate = np.asarray(
            [item for item in smoke_catalog if int(item) in original],
            dtype=np.int64,
        )
        target = np.asarray(row["relevant"], dtype=np.int64)
        if len(candidate) < minimum_candidates:
            raise RuntimeError("Smoke query has fewer than 100 legal candidates")
        if not np.isin(target, candidate).all():
            raise RuntimeError("Smoke relevant item is absent from candidates")
        if any(int(item) in seen for item in candidate):
            raise RuntimeError("Smoke candidate contains a train-seen item")
        candidates.append(candidate)
        relevant.append(target)
    membership = hashlib.sha256(b"phase-b2a-query-sample-v1\n")
    for row, candidate, target in zip(rows, candidates, relevant, strict=True):
        membership.update(f"{row['user_id']}|".encode())
        membership.update(
            ",".join(str(int(item)) for item in candidate).encode()
        )
        membership.update(b"|")
        membership.update(
            ",".join(str(int(item)) for item in target).encode()
        )
        membership.update(b"\n")
    queries = RetrievalQueries(
        user_ids=np.asarray([row["user_id"] for row in rows], dtype=np.int64),
        histories=tuple(histories),
        history_weights=tuple(weights),
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=smoke_catalog,
        warm_user_mask=np.asarray([len(value) > 0 for value in histories], dtype=bool),
    )
    return queries, {
        "sampled_queries": len(rows),
        "sampled_catalog_items": len(smoke_catalog),
        "sampled_targets": int(sum(len(value) for value in relevant)),
        "queries_with_data_cold_relevant": int(
            sum(bool(row["has_data_cold_relevant"]) for row in rows)
        ),
        "query_membership_sha256": membership.hexdigest(),
        "minimum_candidate_count": int(min(len(value) for value in candidates)),
        "maximum_candidate_count": int(max(len(value) for value in candidates)),
    }, tuple(seen_sets)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    diagnostic = report["diagnostic"]
    retrieval = report["sampled_retrieval"]
    artifacts = report["artifacts"]
    memberships = report["memberships"]
    encoding = report["evaluation_encoding"]
    raw_sources = report["input_provenance"]["raw_sources"]
    text = f"""# Phase B2A PyTorch Two-Tower bounded smoke

This is an engineering smoke, not a formal effectiveness experiment. Small
Matrix and temporal final were not accessed; full-data Two-Tower training was
not run.

- Caption model: `{report['environment']['caption_model_id']}`
- Resolved revision: `{report['environment']['caption_model_revision']}`
- Runtime: PyTorch `{report['environment']['torch_version']}`,
  sentence-transformers `{report['environment']['sentence_transformers_version']}`,
  `{report['environment']['device']}`
- Caption coverage: `{report['caption_cache']['nonempty_coverage']:.4%}`
- Caption cache: `{report['caption_cache']['shape'][0]} x {report['caption_cache']['shape'][1]}`
  `{report['caption_cache']['dtype']}`; SHA256
  `{report['caption_cache']['cache_file_sha256']}`
- Training sample: `{report['training_sample']['sampled_users']}` users,
  `{report['training_sample']['sampled_examples']}` examples and
  `{report['training_sample']['sampled_items']}` items
- Fixed diagnostic loss: `{diagnostic['initial_fixed_diagnostic']['loss']:.6f}`
  -> `{diagnostic['final_fixed_diagnostic']['loss']:.6f}`
- Fixed diagnostic Top-1: `{diagnostic['initial_fixed_diagnostic']['diagonal_top1_rate']:.6f}`
  -> `{diagnostic['final_fixed_diagnostic']['diagonal_top1_rate']:.6f}`
- Mean positive logit: `{diagnostic['initial_fixed_diagnostic']['mean_positive_logit']:.6f}`
  -> `{diagnostic['final_fixed_diagnostic']['mean_positive_logit']:.6f}`
- Mean valid-negative logit: `{diagnostic['initial_fixed_diagnostic']['mean_valid_negative_logit']:.6f}`
  -> `{diagnostic['final_fixed_diagnostic']['mean_valid_negative_logit']:.6f}`
- Optimizer: `{diagnostic['optimizer_steps']}` steps,
  `{diagnostic['skipped_batches']}` skipped batches
- Retrieval smoke: `{retrieval['sampled_queries']}` queries,
  `{retrieval['sampled_targets']}` targets,
  `{retrieval['sampled_catalog_items']}` sampled NORMAL items
- Sampled Recall@100: `{retrieval['metrics']['Recall@100']:.6f}`
- Sampled NDCG@20: `{retrieval['metrics']['NDCG@20']:.6f}`;
  Coverage@100: `{retrieval['metrics']['Coverage@100']:.6f}`
- Smoke wall time: `{report['total_wall_time_s']:.4f} s`; peak RSS
  `{report['peak_rss_mb']:.2f} MB`; GPU memory `{report['peak_gpu_memory_mb']:.2f} MB`
- Code commit at run: `{artifacts['code_commit_at_run']}`
- Input tree clean at start:
  `{str(artifacts['input_tree_clean_at_start']).lower()}`
- Checkpoint identity schema: `{artifacts['checkpoint_identity_schema_version']}`;
  SHA256 `{artifacts['checkpoint_identity_sha256']}`
- NORMAL membership: `{memberships['normal']['count']}` items;
  SHA256 `{memberships['normal']['sha256']}`
- Fixed retrieval catalog: `{memberships['fixed_retrieval_catalog']['count']}`
  items; SHA256 `{memberships['fixed_retrieval_catalog']['sha256']}`
- Model item universe: `{memberships['model_item_universe']['count']}` items;
  SHA256 `{memberships['model_item_universe']['sha256']}`
- Evaluation encoding: item batch `{encoding['item_batch_size']}`, user batch
  `{encoding['user_batch_size']}`, one inference-mode item-universe pass plus
  precomputed history gathers

Raw input identity:

| Logical source | Actual SHA256 | Expected SHA256 | Match |
|---|---|---|---|
| `KUAIREC_DATA_DIR/big_matrix.csv` | `{raw_sources['big_matrix.csv']['actual_sha256']}` | `{raw_sources['big_matrix.csv']['expected_sha256']}` | true |
| `KUAIREC_DATA_DIR/item_daily_features.csv` | `{raw_sources['item_daily_features.csv']['actual_sha256']}` | `{raw_sources['item_daily_features.csv']['expected_sha256']}` | true |
| `KUAIREC_DATA_DIR/kuairec_caption_category.csv` | `{raw_sources['kuairec_caption_category.csv']['actual_sha256']}` | `{raw_sources['kuairec_caption_category.csv']['expected_sha256']}` | true |

Required interpretation flags:

```text
sampled_catalog_smoke = true
comparable_to_b1a = false
effectiveness_claim = false
formal_gate_executed = false
```

All paths in the JSON report are stable logical locators. See the JSON for
gradient, false-negative, cache, timing and resource details.
"""
    if "/home/" in text:
        raise RuntimeError("Generated report contains a host path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    checkpoint_path: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    timings: dict[str, float] = {}
    config_path = repo_root / "configs/phase_b2a_two_tower_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    validate_smoke_config(config)
    code_commit_at_run = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    input_tree_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip()
    if input_tree_status:
        raise RuntimeError(
            "Phase B2A.1 smoke requires a clean input tree before execution"
        )
    manifest, raw_input_traceability = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    _verify_artifacts(artifact_dir, manifest)
    started = time.perf_counter()
    static = load_static_item_features(data_dir)
    normal_membership = normal_membership_record(
        np.unique(static.normal_item_ids)
    )
    with np.load(artifact_dir / "events_train_validation.npz") as event_file, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog_file:
        event_user_positions = event_file["user"].astype(np.int64, copy=True)
        event_item_positions = event_file["item"].astype(np.int64, copy=True)
        event_times = event_file["timestamp"].astype(np.float64, copy=True)
        event_strong = event_file["strong"].astype(bool, copy=True)
        user_indptr = event_file["user_indptr"].astype(np.int64, copy=True)
        actual_user_ids = event_file["user_ids"].astype(np.int64, copy=True)
        video_ids = catalog_file["video_ids"].astype(np.int64, copy=True)
        train_end = float(catalog_file["train_end"][0])
    normal_position = np.isin(video_ids, static.normal_item_ids)
    train_event = event_times < train_end
    fixed_catalog = np.unique(
        video_ids[event_item_positions[normal_position[event_item_positions]]]
    )
    fixed_catalog_membership = membership_record(
        fixed_catalog, label="phase-b2a-fixed-retrieval-catalog-v1"
    )
    train_history_items = np.unique(
        video_ids[event_item_positions[train_event]]
    )
    item_universe = np.union1d(train_history_items, fixed_catalog).astype(np.int64)
    model_item_universe_membership = membership_record(
        item_universe, label="phase-b2a-model-item-universe-v1"
    )
    timings["load_verified_inputs_s"] = _elapsed(started)

    caption_trace = raw_input_traceability[
        "kuairec_caption_category.csv"
    ]
    static_for_universe = static.frame.set_index("video_id").reindex(item_universe)
    cleaned_texts = static_for_universe["caption_text"].astype(str).tolist()
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=item_universe,
        expected_model_id=config["caption"]["model_id"],
        expected_revision=config["caption"]["resolved_revision"],
        expected_source_sha256=caption_trace["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            item_universe, cleaned_texts
        ),
    )
    bounded = config["bounded_sample"]
    training_users = _select_proxy_training_users(
        event_users=event_user_positions,
        event_items=event_item_positions,
        event_times=event_times,
        event_strong=event_strong,
        user_indptr=user_indptr,
        normal_item_mask=normal_position,
        train_end=train_end,
        actual_user_ids=actual_user_ids,
        seed=int(config["training"]["seed"]),
        maximum=int(bounded["max_users"]),
    )
    validation_rows = _validation_proxy_rows(
        event_items=event_item_positions,
        event_times=event_times,
        event_strong=event_strong,
        user_indptr=user_indptr,
        normal_item_mask=normal_position,
        video_ids=video_ids,
        actual_user_ids=actual_user_ids,
        train_end=train_end,
        seed=int(config["training"]["diagnostic_seed"]),
        maximum=int(config["retrieval_smoke"]["max_queries"]),
    )
    selected_user_ids = set(int(value) for value in training_users)
    selected_user_ids.update(int(row["user_id"]) for row in validation_rows)
    started = time.perf_counter()
    selected_train_events = _load_selected_train_events(
        data_dir, selected_user_ids, train_end
    )
    timings["scan_big_and_canonicalize_selected_users_s"] = _elapsed(started)

    training_frame = selected_train_events[
        selected_train_events["user_id"].isin(training_users)
    ]
    dataset = build_two_tower_training_dataset(
        training_frame,
        max_history=int(config["architecture"]["max_history"]),
        normal_item_ids=static.normal_item_ids,
    )
    sampled_indices, sample_stats = sample_bounded_example_indices(
        dataset,
        seed=int(config["training"]["seed"]),
        max_users=int(bounded["max_users"]),
        max_examples_per_user=int(bounded["max_examples_per_user"]),
        max_examples=int(bounded["max_examples"]),
        min_users=int(bounded["min_users"]),
        min_examples=int(bounded["min_examples"]),
    )
    train_observed_normal = np.intersect1d(
        train_history_items, static.normal_item_ids, assume_unique=True
    )
    store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=item_universe,
        train_observed_item_ids=train_history_items,
        train_observed_normal_item_ids=train_observed_normal,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = {
        "num_items": len(store.item_ids),
        "num_users": int(sample_stats["sampled_users"]),
        "num_category_tokens": len(store.category_vocab),
        "num_upload_types": len(store.upload_type_vocab),
    }
    torch.manual_seed(int(config["training"]["seed"]))
    model = TwoTowerV1(**dimensions).to(device)
    assert_model_device(model, device)
    started = time.perf_counter()
    training_result = train_bounded_two_tower(
        model=model,
        dataset=dataset,
        sampled_indices=sampled_indices,
        store=store,
        seed=int(config["training"]["seed"]),
        diagnostic_seed=int(config["training"]["diagnostic_seed"]),
        device=device,
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        temperature=float(config["training"]["temperature"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
    )
    timings["bounded_training_s"] = _elapsed(started)
    if timings["bounded_training_s"] > 30 * 60:
        raise RuntimeError("Bounded smoke training exceeded 30 minutes")
    ordered_user_ids = training_result["ordered_user_ids"]
    checkpoint_identity = {
        "schema_version": 2,
        "config": {
            "locator": "configs/phase_b2a_two_tower_smoke.yaml",
            "sha256": sha256_file(config_path),
        },
        "processed_manifest_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
        "raw_inputs": raw_input_traceability,
        "code_commit": code_commit_at_run,
        "ordered_item_store": membership_record(
            store.item_ids, label="phase-b2a-ordered-item-store-v1"
        ),
        "ordered_user_position_mapping": membership_record(
            ordered_user_ids,
            label="phase-b2a-ordered-user-position-mapping-v1",
        ),
        "memberships": {
            "normal": normal_membership,
            "fixed_retrieval_catalog": fixed_catalog_membership,
            "model_item_universe": model_item_universe_membership,
        },
        "feature_identity": {
            "category_vocab_sha256": canonical_json_sha256(
                [
                    [level, raw, index]
                    for (level, raw), index in sorted(
                        store.category_vocab.items()
                    )
                ],
                label="phase-b2a-category-vocab-v1",
            ),
            "upload_type_vocab_sha256": canonical_json_sha256(
                sorted(store.upload_type_vocab.items()),
                label="phase-b2a-upload-type-vocab-v1",
            ),
            "numeric_preprocessing_sha256": canonical_json_sha256(
                store.preprocessing,
                label="phase-b2a-numeric-preprocessing-v1",
            ),
        },
        "caption_identity": {
            "model_id": caption.metadata["model_id"],
            "resolved_revision": caption.metadata["resolved_revision"],
            "item_membership_sha256": caption.metadata[
                "ordered_item_membership_sha256"
            ],
            "embedding_payload_sha256": caption.metadata[
                "embedding_payload_sha256"
            ],
        },
        "actual_touched_membership": {
            "users": {
                "count": training_result["touched_user_count"],
                "sha256": training_result[
                    "touched_user_membership_sha256"
                ],
            },
            "items": {
                "count": training_result["touched_item_count"],
                "sha256": training_result[
                    "touched_item_membership_sha256"
                ],
            },
        },
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        model_dimensions=dimensions,
        ordered_user_ids=ordered_user_ids,
        touched_user_ids=training_result["touched_user_ids"],
        touched_item_ids=training_result["touched_item_ids"],
        identity=checkpoint_identity,
    )
    restored, restored_payload = load_checkpoint(
        checkpoint_path,
        device=device,
        expected_identity=checkpoint_identity,
    )
    assert_model_device(restored, device)
    if any(
        value.device != device for value in restored.state_dict().values()
    ):
        raise RuntimeError("Restored checkpoint tensors are on the wrong device")
    restored.eval()

    retrieval_config = config["retrieval_smoke"]
    queries, query_stats, seen_sets = _sampled_queries(
        validation_rows,
        selected_train_events,
        fixed_catalog=fixed_catalog,
        smoke_catalog_size=int(retrieval_config["max_catalog_items"]),
        minimum_candidates=int(retrieval_config["minimum_candidates_per_query"]),
        seed=int(config["training"]["seed"]),
    )
    touched_items = set(int(value) for value in training_result["touched_item_ids"])
    touched_users = set(int(value) for value in training_result["touched_user_ids"])
    user_positions = {
        int(user): index + 1
        for index, user in enumerate(ordered_user_ids)
    }
    started = time.perf_counter()
    precomputed_items = preencode_item_universe(
        model=restored,
        store=store,
        touched_item_ids=touched_items,
        device=device,
        batch_size=ITEM_ENCODING_BATCH_SIZE,
    )
    timings["preencode_model_item_universe_s"] = _elapsed(started)
    catalog_positions = np.asarray(
        [store.positions[int(item)] for item in queries.catalog],
        dtype=np.int64,
    )
    with torch.inference_mode():
        item_vectors = precomputed_items[
            torch.as_tensor(
                catalog_positions, dtype=torch.long, device=device
            )
        ].cpu().numpy()
    started = time.perf_counter()
    user_vectors = encode_query_users_from_precomputed(
        model=restored,
        store=store,
        precomputed_item_vectors=precomputed_items,
        user_ids=queries.user_ids,
        histories=queries.histories,
        history_weights=queries.history_weights,
        user_positions=user_positions,
        touched_user_ids=touched_users,
        device=device,
        batch_size=USER_ENCODING_BATCH_SIZE,
    ).cpu().numpy()
    timings["batch_encode_query_users_s"] = _elapsed(started)
    started = time.perf_counter()
    popularity = PopularityBaseline.fit(selected_train_events)
    fallback = popularity.rank(queries, k=int(retrieval_config["k"]))
    topk = ExactDotProductRetriever().search(
        user_vectors,
        item_vectors,
        item_ids=queries.catalog,
        candidates=queries.candidates,
        k=int(retrieval_config["k"]),
        warm_user_mask=queries.warm_user_mask,
        fallback_topk=fallback,
    )
    for row, (ranked, candidates, seen) in enumerate(
        zip(topk, queries.candidates, seen_sets, strict=True)
    ):
        valid = ranked[ranked >= 0]
        if len(valid) != min(len(candidates), int(retrieval_config["k"])):
            raise RuntimeError(f"Retrieval row {row} has early padding")
        if len(np.unique(valid)) != len(valid):
            raise RuntimeError(f"Retrieval row {row} has duplicate items")
        if not set(int(value) for value in valid).issubset(
            set(int(value) for value in candidates)
        ):
            raise RuntimeError(f"Retrieval row {row} escaped candidates")
        if any(int(value) in seen for value in valid):
            raise RuntimeError(f"Retrieval row {row} returned a train-seen item")
    cold_items = np.setdiff1d(
        fixed_catalog, train_history_items, assume_unique=True
    ).astype(np.int64)
    metrics = evaluate_retrieval(
        topk, queries, data_cold_item_ids=cold_items
    )
    timings["sampled_exact_retrieval_s"] = _elapsed(started)
    if sum(
        timings[name]
        for name in (
            "bounded_training_s",
            "preencode_model_item_universe_s",
            "batch_encode_query_users_s",
            "sampled_exact_retrieval_s",
        )
    ) > 30 * 60:
        raise RuntimeError("Smoke training plus retrieval exceeded 30 minutes")
    if not np.isfinite(item_vectors).all() or not np.isfinite(user_vectors).all():
        raise FloatingPointError("Retrieval vectors contain NaN or Inf")
    cold_in_catalog = np.intersect1d(queries.catalog, cold_items)
    if len(cold_in_catalog) and not set(int(value) for value in cold_in_catalog).isdisjoint(touched_items):
        raise RuntimeError("Data-cold smoke item unexpectedly used an ID embedding")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if peak_rss_mb > 4096:
        raise RuntimeError(f"Peak RSS exceeded 4GB: {peak_rss_mb:.2f} MB")
    public_training = {
        key: value
        for key, value in training_result.items()
        if key
        not in {
            "touched_user_ids",
            "touched_item_ids",
            "ordered_user_ids",
            "user_positions",
        }
    }
    report: dict[str, Any] = {
        "phase": "phase-b2a-pytorch-two-tower-smoke",
        "status": "completed",
        "claim_boundary": config["claims"],
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "sentence_transformers_version": importlib.metadata.version(
                "sentence-transformers"
            ),
            "caption_model_id": CAPTION_MODEL_ID,
            "caption_model_revision": config["caption"]["resolved_revision"],
        },
        "caption_cache": caption.metadata,
        "feature_preprocessing": store.preprocessing,
        "input_provenance": {
            "processed_manifest": {
                "locator": "PROCESSED_ARTIFACT_DIR/manifest.json",
                "actual_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
                "expected_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
                "sha256_match": True,
            },
            "raw_sources": raw_input_traceability,
        },
        "memberships": {
            "normal": normal_membership,
            "fixed_retrieval_catalog": fixed_catalog_membership,
            "model_item_universe": model_item_universe_membership,
        },
        "training_sample": sample_stats,
        "diagnostic": public_training,
        "sampled_retrieval": {
            **config["claims"],
            **query_stats,
            "content_only_data_cold_catalog_items": int(len(cold_in_catalog)),
            **metrics,
        },
        "artifacts": {
            "config_locator": "configs/phase_b2a_two_tower_smoke.yaml",
            "caption_cache_locator": "artifacts/phase_b2a/caption_embeddings.npz",
            "caption_metadata_locator": "reports/phase_b2a/caption_cache_metadata.json",
            "checkpoint_locator": "artifacts/phase_b2a/two_tower_smoke.pt",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_identity_schema_version": 2,
            "checkpoint_identity_sha256": restored_payload[
                "identity_sha256"
            ],
            "processed_manifest_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
            "code_commit_at_run": code_commit_at_run,
            "input_tree_clean_at_start": True,
        },
        "evaluation_encoding": {
            "inference_mode": True,
            "item_universe_preencoded_once": True,
            "history_uses_precomputed_gather": True,
            "item_batch_size": ITEM_ENCODING_BATCH_SIZE,
            "user_batch_size": USER_ENCODING_BATCH_SIZE,
            "model_item_count": len(store.item_ids),
        },
        "timings_s": timings,
        "total_wall_time_s": _elapsed(started_total),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "access": {
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "full_two_tower_training": False,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if "/home/" in serialized:
        raise RuntimeError("Generated JSON report contains a host path")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(serialized + "\n")
    _write_markdown(report, report_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--caption-cache",
        type=Path,
        default=Path("artifacts/phase_b2a/caption_embeddings.npz"),
    )
    parser.add_argument(
        "--caption-metadata",
        type=Path,
        default=Path("reports/phase_b2a/caption_cache_metadata.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/phase_b2a/two_tower_smoke.pt"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b2a/two_tower_smoke.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b2a/two_tower_smoke.md"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run(
        repo_root=root,
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=(root / args.caption_cache).resolve(),
        caption_metadata_path=(root / args.caption_metadata).resolve(),
        checkpoint_path=(root / args.checkpoint).resolve(),
        report_json=(root / args.report_json).resolve(),
        report_markdown=(root / args.report_markdown).resolve(),
    )


if __name__ == "__main__":
    main()
