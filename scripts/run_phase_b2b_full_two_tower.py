#!/usr/bin/env python3
"""Run the frozen full Two-Tower route or one bounded B2B0 preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.full_training import (
    attach_train_histories,
    build_checkpoint_identity,
    build_validation_contract,
    evaluate_frozen_gates,
    load_canonical_train_events,
    load_full_epoch_checkpoint,
    planned_training_membership,
    save_full_epoch_checkpoint,
    select_checkpoint_epoch,
    train_full_two_tower,
    verify_validation_contract,
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
    preencode_item_universe,
    prepare_item_feature_store,
    resolve_concrete_device,
    sample_bounded_example_indices,
)
from kuairec_fully_observed.training import (
    build_two_tower_training_dataset,
)

EXPECTED_ARCHITECTURE = {
    "item_id_dim": 64,
    "category_dim": 32,
    "caption_projection_dim": 64,
    "static_projection_dim": 16,
    "upload_type_dim": 8,
    "hidden_dim": 256,
    "output_dim": 128,
    "max_history": 50,
}
EXPECTED_TRAINING = {
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
EXPECTED_TRAINING_CONTRACT = {
    "full_example_count": 574098,
    "training_user_count": 7161,
}
EXPECTED_SCOPE = {
    "fit": "canonical_big_train",
    "select": "canonical_big_validation",
    "forbidden": [
        "small_matrix",
        "temporal_final",
        "faiss",
        "hybrid",
        "reranker",
        "serving",
    ],
}
EXPECTED_CAPTION = {
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "resolved_revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    "dimension": 384,
    "preprocessing_version": "phase-b2a-caption-fallback-v1",
}
EXPECTED_CHECKPOINT = {
    "epochs": [1, 2, 3],
    "directory": "artifacts/phase_b2b",
}
EXPECTED_VALIDATION = {
    "k": 100,
    "item_encoding_batch_size": 1024,
    "user_encoding_batch_size": 128,
    "score_block_size": 128,
    "expected": {
        "fixed_catalog_count": 9365,
        "fixed_catalog_sha256": (
            "8b8e88e2455a27dc0fac79e7bdb2733dc43096bb6d637e549d1ce5853e8ce55b"
        ),
        "query_count": 6818,
        "warm_query_count": 6816,
        "target_count": 118565,
        "warm_target_count": 118539,
        "data_cold_item_count": 1492,
        "query_contract_sha256": (
            "f25df9d235f32b114357a190f69a60e49d8993d2c4e09f975a0b361c7439f877"
        ),
    },
}
EXPECTED_SELECTION = {
    "primary": "Recall@100",
    "tie_break": "NDCG@20",
    "final_tie_break": "earliest_epoch",
}
EXPECTED_FROZEN_BPR = {
    "Recall@100": 0.04843855379304513,
    "NDCG@20": 0.012773561140700995,
    "Coverage@100": 0.33304858515750135,
    "Data-Cold Recall@100": 0.0,
}
EXPECTED_GATE = {
    "common_ndcg_minimum": 0.002773561140700995,
    "A": {"Recall@100_minimum": 0.05043855379304513},
    "B": {
        "Recall@100_minimum": 0.02843855379304513,
        "Coverage@100_minimum": 0.38304858515750135,
    },
    "C": {
        "Recall@100_minimum": 0.02843855379304513,
        "data_cold_target_denominator_minimum": 100,
        "Data-Cold_Recall@100_minimum": 0.05,
    },
}
EXPECTED_PREFLIGHT_CLAIMS = {
    "formal_gate_executed": False,
    "effectiveness_claim": False,
    "full_big_train": False,
    "full_big_validation": False,
}
EXPECTED_PREFLIGHT = {
    "max_training_examples": 2560,
    "max_optimizer_steps": 20,
    "epochs": 2,
    "max_validation_queries": 128,
    "claims": EXPECTED_PREFLIGHT_CLAIMS,
}
FULL_RUN_CLAIMS = {
    "formal_gate_executed": True,
    "effectiveness_claim": False,
    "full_big_train": True,
    "full_big_validation": True,
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "phase-b2b-full-two-tower":
        raise RuntimeError("Phase B2B phase is not frozen")
    frozen_sections = {
        "scope": EXPECTED_SCOPE,
        "caption": EXPECTED_CAPTION,
        "architecture": EXPECTED_ARCHITECTURE,
        "training": EXPECTED_TRAINING,
        "training_contract": EXPECTED_TRAINING_CONTRACT,
        "checkpoint": EXPECTED_CHECKPOINT,
        "validation": EXPECTED_VALIDATION,
        "selection": EXPECTED_SELECTION,
        "frozen_bpr_epoch_20": EXPECTED_FROZEN_BPR,
        "gate": EXPECTED_GATE,
        "preflight": EXPECTED_PREFLIGHT,
    }
    for name, expected in frozen_sections.items():
        if config.get(name) != expected:
            raise RuntimeError(f"Phase B2B {name} is not frozen")
def _report_mode(preflight: bool) -> dict[str, Any]:
    if preflight:
        return {
            "phase": "phase-b2b0-full-runner-preflight",
            "claim_boundary": dict(EXPECTED_PREFLIGHT_CLAIMS),
            "title": "# Phase B2B0 Full Runner Preflight",
            "description": (
                "This is a bounded engineering preflight through the production "
                "runner path. It is not a formal effectiveness experiment."
            ),
        }
    return {
        "phase": "phase-b2b-full-two-tower",
        "claim_boundary": dict(FULL_RUN_CLAIMS),
        "title": "# Phase B2B Full Two-Tower Results",
        "description": (
            "This report records the frozen full Big-train and exact "
            "Big-validation experiment."
        ),
    }


def _execution_mode(
    *,
    preflight: bool,
    resume_completed_epoch: int | None,
    frozen_epoch_count: int,
) -> str:
    if preflight:
        return "preflight"
    if resume_completed_epoch is None:
        return "fresh_full_train"
    if resume_completed_epoch < frozen_epoch_count:
        return "resumed_full_train"
    if resume_completed_epoch == frozen_epoch_count:
        return "finalize_completed_checkpoint"
    raise RuntimeError("Resume checkpoint exceeds the frozen epoch plan")


def _report_temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _repo_relative_path(repo_root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _assert_clean_source_tree(
    *,
    repo_root: Path,
    report_json: Path,
    report_markdown: Path,
) -> None:
    allowed = {
        relative
        for path in (
            report_json,
            report_markdown,
            _report_temporary_path(report_json),
            _report_temporary_path(report_markdown),
        )
        if (relative := _repo_relative_path(repo_root, path)) is not None
    }
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    unexpected: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            unexpected.append(line)
            continue
        status = line[:2]
        locator = line[3:]
        if "R" in status or "C" in status or " -> " in locator:
            unexpected.append(line)
            continue
        if locator.startswith('"') or locator not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "Phase B2B runner requires a clean input tree; unexpected "
            f"changes: {unexpected}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _report_temporary_path(path)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_checkpoint_result(
    restored: dict[str, Any],
) -> dict[str, Any]:
    """Represent an epoch-3 resume without executing another optimizer step."""

    cumulative = dict(restored["cumulative_training_statistics"])
    return {
        "epoch_losses": tuple(restored["epoch_losses"]),
        "optimizer_steps": 0,
        "skipped_batches": 0,
        "completed_examples": 0,
        "process_statistics": {
            "optimizer_steps": 0,
            "skipped_batches": 0,
            "completed_examples": 0,
        },
        "cumulative_statistics": cumulative,
        "touched_user_ids": np.asarray(
            restored["touched_user_ids"], dtype=np.int64
        ),
        "touched_item_ids": np.asarray(
            restored["touched_item_ids"], dtype=np.int64
        ),
        "completed_epoch": int(restored["completed_epoch"]),
        "fixed_diagnostic_loss": None,
    }


def _stable_key(seed: int, *values: int) -> bytes:
    body = ":".join(str(int(value)) for value in (seed, *values)).encode()
    return hashlib.sha256(body).digest()


def _select_preflight_training_users(
    *,
    event_items: np.ndarray,
    event_times: np.ndarray,
    event_strong: np.ndarray,
    user_indptr: np.ndarray,
    actual_user_ids: np.ndarray,
    normal_item_mask: np.ndarray,
    train_end: float,
    seed: int,
    maximum: int = 256,
) -> np.ndarray:
    eligible: list[int] = []
    for position in range(len(user_indptr) - 1):
        start = int(user_indptr[position])
        end = int(user_indptr[position + 1])
        seen: set[int] = set()
        valid = False
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
                valid = True
        if valid:
            eligible.append(int(actual_user_ids[position]))
    return np.asarray(
        sorted(eligible, key=lambda user: (_stable_key(seed, user), user))[
            :maximum
        ],
        dtype=np.int64,
    )


def _subset_queries(
    queries: RetrievalQueries, *, maximum: int, seed: int
) -> RetrievalQueries:
    cold = np.flatnonzero(~queries.warm_user_mask).tolist()
    warm = np.flatnonzero(queries.warm_user_mask).tolist()
    cold = sorted(
        cold,
        key=lambda row: (
            _stable_key(seed, int(queries.user_ids[row]), 1),
            int(queries.user_ids[row]),
        ),
    )
    warm = sorted(
        warm,
        key=lambda row: (
            _stable_key(seed, int(queries.user_ids[row]), 2),
            int(queries.user_ids[row]),
        ),
    )
    selected = np.asarray(
        sorted((cold + warm)[:maximum]), dtype=np.int64
    )
    return RetrievalQueries(
        user_ids=queries.user_ids[selected],
        histories=tuple(queries.histories[int(row)] for row in selected),
        history_weights=tuple(
            queries.history_weights[int(row)] for row in selected
        ),
        candidates=tuple(queries.candidates[int(row)] for row in selected),
        relevant=tuple(queries.relevant[int(row)] for row in selected),
        catalog=queries.catalog,
        warm_user_mask=queries.warm_user_mask[selected],
        diagnostics={"bounded_preflight_query_count": int(len(selected))},
    )


def _processed_popularity(
    *,
    event_items: np.ndarray,
    event_times: np.ndarray,
    event_strong: np.ndarray,
    video_ids: np.ndarray,
    normal_item_mask: np.ndarray,
    train_end: float,
) -> PopularityBaseline:
    mask = (
        (event_times < train_end)
        & event_strong
        & normal_item_mask[event_items]
    )
    counts = np.bincount(event_items[mask], minlength=len(video_ids))
    return PopularityBaseline(
        {
            int(video_ids[position]): float(counts[position])
            for position in np.flatnonzero(counts)
        }
    )


def _evaluate_model(
    *,
    model: TwoTowerV1,
    store,
    queries: RetrievalQueries,
    data_cold_items: np.ndarray,
    ordered_user_ids: np.ndarray,
    touched_user_ids: np.ndarray,
    touched_item_ids: np.ndarray,
    popularity: PopularityBaseline,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    validation = config["validation"]
    started = time.perf_counter()
    item_vectors = preencode_item_universe(
        model=model,
        store=store,
        touched_item_ids=set(int(value) for value in touched_item_ids),
        device=device,
        batch_size=int(validation["item_encoding_batch_size"]),
    )
    item_encoding_s = time.perf_counter() - started
    catalog_positions = np.asarray(
        [store.positions[int(item)] for item in queries.catalog],
        dtype=np.int64,
    )
    with torch.inference_mode():
        catalog_vectors = item_vectors[
            torch.as_tensor(
                catalog_positions, dtype=torch.long, device=device
            )
        ].cpu().numpy()
    user_positions = {
        int(user): index + 1
        for index, user in enumerate(ordered_user_ids)
    }
    started = time.perf_counter()
    user_vectors = encode_query_users_from_precomputed(
        model=model,
        store=store,
        precomputed_item_vectors=item_vectors,
        user_ids=queries.user_ids,
        histories=queries.histories,
        history_weights=queries.history_weights,
        user_positions=user_positions,
        touched_user_ids=set(int(value) for value in touched_user_ids),
        device=device,
        batch_size=int(validation["user_encoding_batch_size"]),
    ).cpu().numpy()
    user_encoding_s = time.perf_counter() - started
    if not np.isfinite(catalog_vectors).all() or not np.isfinite(
        user_vectors
    ).all():
        raise FloatingPointError("Validation vector became non-finite")
    fallback = popularity.rank(queries, k=int(validation["k"]))
    started = time.perf_counter()
    topk = ExactDotProductRetriever().search(
        user_vectors,
        catalog_vectors,
        item_ids=queries.catalog,
        candidates=queries.candidates,
        k=int(validation["k"]),
        warm_user_mask=queries.warm_user_mask,
        fallback_topk=fallback,
        score_block_size=int(validation["score_block_size"]),
    )
    metrics = evaluate_retrieval(
        topk, queries, data_cold_item_ids=data_cold_items
    )
    retrieval_s = time.perf_counter() - started
    return metrics, {
        "item_encoding_s": item_encoding_s,
        "user_encoding_s": user_encoding_s,
        "exact_retrieval_and_metrics_s": retrieval_s,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    mode = _report_mode(
        report["phase"] == "phase-b2b0-full-runner-preflight"
    )
    if report["phase"] != mode["phase"]:
        raise RuntimeError(f"Unknown report phase: {report['phase']}")
    if report["claim_boundary"] != mode["claim_boundary"]:
        raise RuntimeError("Report claims do not match the execution mode")
    rows = []
    for record in report["checkpoints"]:
        metrics = record["validation"]["metrics"]
        rows.append(
            "| {epoch} | {loss:.6f} | {recall:.6f} | {ndcg:.6f} | "
            "{coverage:.6f} |".format(
                epoch=record["epoch"],
                loss=record["epoch_loss"],
                recall=metrics["Recall@100"],
                ndcg=metrics["NDCG@20"],
                coverage=metrics["Coverage@100"],
            )
        )
    claims = report["claim_boundary"]
    process_statistics = report["training"]["process_statistics"]
    cumulative_statistics = report["training"]["cumulative_statistics"]
    estimate = report["estimated_full_run_minutes"]
    if estimate["low"] is None or estimate["high"] is None:
        timing_line = (
            f"- Complete full-run wall time: unavailable from this "
            f"`{report['execution_mode']}` process"
        )
    else:
        timing_line = (
            f"- Estimated/observed full run: `{estimate['low']:.1f}` to "
            f"`{estimate['high']:.1f}` minutes"
        )
    lines = [
        mode["title"],
        "",
        mode["description"],
        "",
        f"- Execution mode: `{report['execution_mode']}`",
        f"- Device: `{report['environment']['device']}`",
        f"- Current-process wall time: `{report['runtime_s']:.2f} s`",
        f"- Peak RSS: `{report['peak_rss_mb']:.2f} MB`",
        f"- Save/load/resume verified: "
        f"`{str(report['resume']['verified']).lower()}`",
        f"- Training examples: `{report['training']['example_count']}`",
        f"- Optimizer steps in this process: "
        f"`{process_statistics['optimizer_steps']}`",
        f"- Cumulative optimizer steps: "
        f"`{cumulative_statistics['optimizer_steps']}`",
        f"- Skipped batches in this process / cumulative: "
        f"`{process_statistics['skipped_batches']} / "
        f"{cumulative_statistics['skipped_batches']}`",
        f"- Completed examples in this process / cumulative: "
        f"`{process_statistics['completed_examples']} / "
        f"{cumulative_statistics['completed_examples']}`",
        f"- Validation queries: `{report['validation']['evaluated_queries']}`",
        timing_line,
    ]
    if report["execution_mode"] == "finalize_completed_checkpoint":
        timings = report["timings_s"]
        lines.extend(
            [
                f"- Checkpoint loading: "
                f"`{timings['checkpoint_loading_s']:.3f} s`",
                f"- Three-checkpoint reevaluation: "
                f"`{timings['checkpoint_reevaluation_s']:.3f} s`",
                f"- Report generation: "
                f"`{timings['report_generation_s']:.3f} s`",
            ]
        )
    lines.extend(
        [
            "",
            "| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 |",
            "|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Required claim boundary:",
            "",
            "```text",
            f"formal_gate_executed={str(claims['formal_gate_executed']).lower()}",
            f"effectiveness_claim={str(claims['effectiveness_claim']).lower()}",
            f"full_big_train={str(claims['full_big_train']).lower()}",
            f"full_big_validation={str(claims['full_big_validation']).lower()}",
            "```",
            "",
            "Small Matrix, temporal final, FAISS and Hybrid were not accessed "
            "or run.",
            "",
        ]
    )
    text = "\n".join(lines)
    if "/home/" in text:
        raise RuntimeError("Generated Markdown contains a host path")
    return text


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    _atomic_write_text(path, _render_markdown(report))


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    config_path: Path,
    checkpoint_dir: Path,
    report_json: Path,
    report_markdown: Path,
    preflight: bool,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    _assert_clean_source_tree(
        repo_root=repo_root,
        report_json=report_json,
        report_markdown=report_markdown,
    )
    manifest, raw_sources = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    if sha256_file(artifact_dir / "manifest.json") != (
        PHASE1_PROCESSED_MANIFEST_SHA256
    ):
        raise RuntimeError("Processed manifest identity changed")
    for name in ("events_train_validation.npz", "catalog.npz"):
        if sha256_file(artifact_dir / name) != manifest["files"][name]:
            raise RuntimeError(f"Processed artifact SHA mismatch: {name}")
    static = load_static_item_features(data_dir)
    normal_membership = normal_membership_record(
        np.unique(static.normal_item_ids)
    )
    if normal_membership != {
        "count": 10699,
        "sha256": (
            "631a7c7cc93413f250f36f548feb720f8322050010e291afcc88338155f52c8e"
        ),
        "hash_scheme": (
            "sha256(normal-video-membership-v1\\n + "
            "sorted-unique-decimal-id\\n)"
        ),
    }:
        raise RuntimeError("Frozen NORMAL membership changed")
    with np.load(artifact_dir / "events_train_validation.npz") as events, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog:
        event_users = events["user"].astype(np.int64, copy=True)
        event_items = events["item"].astype(np.int64, copy=True)
        event_times = events["timestamp"].astype(np.float64, copy=True)
        event_strong = events["strong"].astype(bool, copy=True)
        user_indptr = events["user_indptr"].astype(np.int64, copy=True)
        actual_user_ids = events["user_ids"].astype(np.int64, copy=True)
        video_ids = catalog["video_ids"].astype(np.int64, copy=True)
        train_end = float(catalog["train_end"][0])
    normal_position = np.isin(video_ids, static.normal_item_ids)
    contract_queries, data_cold_items, validation_counts = (
        build_validation_contract(
            event_users=event_users,
            event_items=event_items,
            event_times=event_times,
            event_strong=event_strong,
            user_indptr=user_indptr,
            actual_user_ids=actual_user_ids,
            video_ids=video_ids,
            normal_item_mask=normal_position,
            train_end=train_end,
            train_events=None,
        )
    )
    verify_validation_contract(
        queries=contract_queries,
        counts=validation_counts,
        expected=config["validation"]["expected"],
    )
    training = config["training"]
    if preflight:
        training_users = _select_preflight_training_users(
            event_items=event_items,
            event_times=event_times,
            event_strong=event_strong,
            user_indptr=user_indptr,
            actual_user_ids=actual_user_ids,
            normal_item_mask=normal_position,
            train_end=train_end,
            seed=int(training["seed"]),
        )
        query_subset = _subset_queries(
            contract_queries,
            maximum=int(config["preflight"]["max_validation_queries"]),
            seed=int(training["diagnostic_seed"]),
        )
        selected_raw_users = set(int(value) for value in training_users)
        selected_raw_users.update(int(value) for value in query_subset.user_ids)
    else:
        training_users = actual_user_ids
        query_subset = contract_queries
        selected_raw_users = None
    canonical_train = load_canonical_train_events(
        data_dir,
        train_end=train_end,
        selected_user_ids=selected_raw_users,
    )
    training_frame = canonical_train[
        canonical_train["user_id"].isin(training_users)
    ].reset_index(drop=True)
    dataset = build_two_tower_training_dataset(
        training_frame,
        max_history=int(config["architecture"]["max_history"]),
        normal_item_ids=static.normal_item_ids,
    )
    if not preflight:
        if len(dataset) != int(
            config["training_contract"]["full_example_count"]
        ):
            raise RuntimeError("Full Two-Tower example count changed")
        full_training_users = np.unique(
            dataset.user_ids[dataset.positive_event_indices]
        )
        if len(full_training_users) != int(
            config["training_contract"]["training_user_count"]
        ):
            raise RuntimeError("Full Two-Tower training-user count changed")
    if preflight:
        example_indices, sample_stats = sample_bounded_example_indices(
            dataset,
            seed=int(training["seed"]),
            max_users=256,
            max_examples_per_user=64,
            max_examples=int(config["preflight"]["max_training_examples"]),
            min_users=64,
            min_examples=1000,
        )
    else:
        example_indices = np.arange(len(dataset), dtype=np.int64)
        sample_stats = {
            "source_example_population": int(len(dataset)),
            "sampled_users": int(
                len(
                    np.unique(
                        dataset.user_ids[dataset.positive_event_indices]
                    )
                )
            ),
            "sampled_examples": int(len(dataset)),
            "not_csv_prefix": False,
        }
    ordered_users, planned_items = planned_training_membership(
        dataset, example_indices
    )
    fixed_catalog = contract_queries.catalog
    train_history_items = np.unique(
        video_ids[event_items[event_times < train_end]]
    )
    model_item_universe = np.union1d(
        train_history_items, fixed_catalog
    ).astype(np.int64)
    fixed_membership = membership_record(
        fixed_catalog, label="phase-b2a-fixed-retrieval-catalog-v1"
    )
    universe_membership = membership_record(
        model_item_universe, label="phase-b2a-model-item-universe-v1"
    )
    static_for_universe = static.frame.set_index("video_id").reindex(
        model_item_universe
    )
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=model_item_universe,
        expected_model_id=config["caption"]["model_id"],
        expected_revision=config["caption"]["resolved_revision"],
        expected_source_sha256=raw_sources[
            "kuairec_caption_category.csv"
        ]["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            model_item_universe,
            static_for_universe["caption_text"].astype(str).tolist(),
        ),
    )
    train_observed_normal = np.intersect1d(
        train_history_items, static.normal_item_ids, assume_unique=True
    )
    store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=model_item_universe,
        train_observed_item_ids=train_history_items,
        train_observed_normal_item_ids=train_observed_normal,
    )
    dimensions = {
        "num_items": int(len(store.item_ids)),
        "num_users": int(len(ordered_users)),
        "num_category_tokens": int(len(store.category_vocab)),
        "num_upload_types": int(len(store.upload_type_vocab)),
    }
    device = resolve_concrete_device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(int(training["seed"]))
    model = TwoTowerV1(**dimensions).to(device)
    assert_model_device(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    populated_queries = attach_train_histories(
        contract_queries,
        canonical_train,
        max_history=int(config["architecture"]["max_history"]),
    )
    evaluation_queries = (
        _subset_queries(
            populated_queries,
            maximum=int(config["preflight"]["max_validation_queries"]),
            seed=int(training["diagnostic_seed"]),
        )
        if preflight
        else populated_queries
    )
    for row, is_warm in zip(
        evaluation_queries.histories,
        evaluation_queries.warm_user_mask,
        strict=True,
    ):
        if is_warm and not len(row):
            raise RuntimeError("Warm validation query has no train history")
        if not is_warm and len(row):
            raise RuntimeError("Cold validation query unexpectedly has history")
    popularity = _processed_popularity(
        event_items=event_items,
        event_times=event_times,
        event_strong=event_strong,
        video_ids=video_ids,
        normal_item_mask=normal_position,
        train_end=train_end,
    )
    feature_identity = {
        "category_vocab_count": len(store.category_vocab),
        "category_vocab_sha256": canonical_json_sha256(
            [
                [level, raw, index]
                for (level, raw), index in sorted(
                    store.category_vocab.items()
                )
            ],
            label="phase-b2a-category-vocab-v1",
        ),
        "upload_type_vocab_count": len(store.upload_type_vocab),
        "upload_type_vocab_sha256": canonical_json_sha256(
            sorted(store.upload_type_vocab.items()),
            label="phase-b2a-upload-type-vocab-v1",
        ),
        "numeric_preprocessing_sha256": canonical_json_sha256(
            store.preprocessing,
            label="phase-b2a-numeric-preprocessing-v1",
        ),
    }
    base_identity = {
        "config": {
            "locator": "configs/phase_b2b_full_two_tower.yaml",
            "sha256": sha256_file(config_path),
        },
        "processed_manifest_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
        "raw_inputs": raw_sources,
        "code_commit": code_commit,
        "memberships": {
            "normal": normal_membership,
            "fixed_retrieval_catalog": fixed_membership,
            "model_item_universe": universe_membership,
            "validation_query_contract": {
                "count": validation_counts["query_count"],
                "sha256": validation_counts["query_contract_sha256"],
            },
        },
        "feature_identity": feature_identity,
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
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def checkpoint_callback(
        epoch,
        current_model,
        current_optimizer,
        losses,
        touched_users,
        touched_items,
        cumulative_statistics,
    ):
        identity = build_checkpoint_identity(
            base_identity=base_identity,
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=ordered_users,
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            training_seed=int(training["seed"]),
        )
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_full_epoch_checkpoint(
            checkpoint_path,
            model=current_model,
            optimizer=current_optimizer,
            completed_epoch=epoch,
            epoch_losses=losses,
            order_seed=int(training["seed"]),
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=ordered_users,
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            cumulative_statistics=cumulative_statistics,
            identity=identity,
        )
        validation, timings = _evaluate_model(
            model=current_model,
            store=store,
            queries=evaluation_queries,
            data_cold_items=data_cold_items,
            ordered_user_ids=ordered_users,
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            popularity=popularity,
            device=device,
            config=config,
        )
        records.append(
            {
                "epoch": epoch,
                "epoch_loss": losses[-1],
                "validation": validation,
                "timings_s": timings,
                "checkpoint": {
                    "locator": f"CHECKPOINT_DIR/epoch_{epoch:03d}.pt",
                    "sha256": sha256_file(checkpoint_path),
                    "identity_sha256": canonical_json_sha256(
                        identity,
                        label="phase-b2a-checkpoint-identity-v2",
                    ),
                },
            }
        )

    checkpoint_loading_s = 0.0
    checkpoint_reevaluation_s = 0.0
    resume_completed_epoch: int | None = None
    if resume_checkpoint is not None:
        checkpoint_load_started = time.perf_counter()
        inspection = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        expected_identity = build_checkpoint_identity(
            base_identity=base_identity,
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=ordered_users,
            touched_user_ids=np.asarray(
                inspection["touched_user_ids"], dtype=np.int64
            ),
            touched_item_ids=np.asarray(
                inspection["touched_item_ids"], dtype=np.int64
            ),
            training_seed=int(training["seed"]),
        )
        model, optimizer, restored = load_full_epoch_checkpoint(
            resume_checkpoint,
            device=device,
            expected_identity=expected_identity,
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        checkpoint_loading_s += time.perf_counter() - checkpoint_load_started
        resume_completed_epoch = int(restored["completed_epoch"])
        start_epoch = int(restored["completed_epoch"]) + 1
        prior_losses = tuple(restored["epoch_losses"])
        initial_touched_users = restored["touched_user_ids"]
        initial_touched_items = restored["touched_item_ids"]
        prior_statistics = restored["cumulative_training_statistics"]
    else:
        start_epoch = 1
        prior_losses = ()
        initial_touched_users = None
        initial_touched_items = None
        prior_statistics = {
            "optimizer_steps": 0,
            "skipped_batches": 0,
            "completed_examples": 0,
        }
    execution_mode = _execution_mode(
        preflight=preflight,
        resume_completed_epoch=resume_completed_epoch,
        frozen_epoch_count=int(training["epochs"]),
    )
    training_started = time.perf_counter()
    if preflight and resume_checkpoint is None:
        first = train_full_two_tower(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            example_indices=example_indices,
            store=store,
            ordered_user_ids=ordered_users,
            planned_item_ids=planned_items,
            device=device,
            seed=int(training["seed"]),
            diagnostic_seed=int(training["diagnostic_seed"]),
            start_epoch=1,
            end_epoch=1,
            batch_size=int(training["batch_size"]),
            temperature=float(training["temperature"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            checkpoint_callback=checkpoint_callback,
            max_total_steps=int(config["preflight"]["max_optimizer_steps"]) // 2,
        )
        first_checkpoint = checkpoint_dir / "epoch_001.pt"
        first_identity = build_checkpoint_identity(
            base_identity=base_identity,
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=ordered_users,
            touched_user_ids=first["touched_user_ids"],
            touched_item_ids=first["touched_item_ids"],
            training_seed=int(training["seed"]),
        )
        del model, optimizer
        checkpoint_load_started = time.perf_counter()
        model, optimizer, restored = load_full_epoch_checkpoint(
            first_checkpoint,
            device=device,
            expected_identity=first_identity,
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        checkpoint_loading_s += time.perf_counter() - checkpoint_load_started
        second = train_full_two_tower(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            example_indices=example_indices,
            store=store,
            ordered_user_ids=ordered_users,
            planned_item_ids=planned_items,
            device=device,
            seed=int(training["seed"]),
            diagnostic_seed=int(training["diagnostic_seed"]),
            start_epoch=2,
            end_epoch=2,
            batch_size=int(training["batch_size"]),
            temperature=float(training["temperature"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            prior_epoch_losses=tuple(restored["epoch_losses"]),
            touched_user_ids=restored["touched_user_ids"],
            touched_item_ids=restored["touched_item_ids"],
            prior_optimizer_steps=int(
                restored["cumulative_training_statistics"][
                    "optimizer_steps"
                ]
            ),
            prior_skipped_batches=int(
                restored["cumulative_training_statistics"][
                    "skipped_batches"
                ]
            ),
            prior_completed_examples=int(
                restored["cumulative_training_statistics"][
                    "completed_examples"
                ]
            ),
            checkpoint_callback=checkpoint_callback,
            max_total_steps=int(config["preflight"]["max_optimizer_steps"]) // 2,
        )
        training_result = second
        resume_verified = True
        process_statistics = {
            name: int(first["process_statistics"][name])
            + int(second["process_statistics"][name])
            for name in (
                "optimizer_steps",
                "skipped_batches",
                "completed_examples",
            )
        }
        cumulative_statistics = dict(second["cumulative_statistics"])
    elif start_epoch <= int(training["epochs"]):
        training_result = train_full_two_tower(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            example_indices=example_indices,
            store=store,
            ordered_user_ids=ordered_users,
            planned_item_ids=planned_items,
            device=device,
            seed=int(training["seed"]),
            diagnostic_seed=int(training["diagnostic_seed"]),
            start_epoch=start_epoch,
            end_epoch=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            temperature=float(training["temperature"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            prior_epoch_losses=prior_losses,
            prior_optimizer_steps=int(
                prior_statistics["optimizer_steps"]
            ),
            prior_skipped_batches=int(
                prior_statistics["skipped_batches"]
            ),
            prior_completed_examples=int(
                prior_statistics["completed_examples"]
            ),
            touched_user_ids=initial_touched_users,
            touched_item_ids=initial_touched_items,
            checkpoint_callback=checkpoint_callback,
        )
        resume_verified = resume_checkpoint is not None
        process_statistics = dict(training_result["process_statistics"])
        cumulative_statistics = dict(
            training_result["cumulative_statistics"]
        )
    else:
        if execution_mode != "finalize_completed_checkpoint":
            raise RuntimeError("Terminal resume mode was not resolved")
        training_result = _completed_checkpoint_result(restored)
        resume_verified = True
        process_statistics = dict(training_result["process_statistics"])
        cumulative_statistics = dict(
            training_result["cumulative_statistics"]
        )
    training_wall_s = time.perf_counter() - training_started
    total_optimizer_steps = int(process_statistics["optimizer_steps"])
    if preflight and total_optimizer_steps > int(
        config["preflight"]["max_optimizer_steps"]
    ):
        raise RuntimeError("Preflight optimizer-step bound was exceeded")
    recorded_epochs = {int(row["epoch"]) for row in records}
    for epoch in range(1, int(training_result["completed_epoch"]) + 1):
        if epoch in recorded_epochs:
            continue
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(
                f"Complete checkpoint for epoch {epoch} is missing"
            )
        checkpoint_load_started = time.perf_counter()
        inspection = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        identity = build_checkpoint_identity(
            base_identity=base_identity,
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=ordered_users,
            touched_user_ids=np.asarray(
                inspection["touched_user_ids"], dtype=np.int64
            ),
            touched_item_ids=np.asarray(
                inspection["touched_item_ids"], dtype=np.int64
            ),
            training_seed=int(training["seed"]),
        )
        checkpoint_model, _, checkpoint_payload = load_full_epoch_checkpoint(
            checkpoint_path,
            device=device,
            expected_identity=identity,
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        checkpoint_loading_s += time.perf_counter() - checkpoint_load_started
        reevaluation_started = time.perf_counter()
        validation, timings = _evaluate_model(
            model=checkpoint_model,
            store=store,
            queries=evaluation_queries,
            data_cold_items=data_cold_items,
            ordered_user_ids=ordered_users,
            touched_user_ids=checkpoint_payload["touched_user_ids"],
            touched_item_ids=checkpoint_payload["touched_item_ids"],
            popularity=popularity,
            device=device,
            config=config,
        )
        checkpoint_reevaluation_s += (
            time.perf_counter() - reevaluation_started
        )
        records.append(
            {
                "epoch": epoch,
                "epoch_loss": checkpoint_payload["epoch_losses"][-1],
                "validation": validation,
                "timings_s": timings,
                "checkpoint": {
                    "locator": f"CHECKPOINT_DIR/epoch_{epoch:03d}.pt",
                    "sha256": sha256_file(checkpoint_path),
                    "identity_sha256": checkpoint_payload[
                        "identity_sha256"
                    ],
                },
            }
        )
    records.sort(key=lambda row: int(row["epoch"]))
    selected_epoch = select_checkpoint_epoch(records)
    selected_record = next(
        row for row in records if row["epoch"] == selected_epoch
    )
    gates = (
        {"A": False, "B": False, "C": False}
        if preflight
        else evaluate_frozen_gates(
            selected_record["validation"]["metrics"],
            selected_record["validation"]["denominators"],
            config["gate"],
        )
    )
    if preflight:
        steps_per_second = max(
            total_optimizer_steps / max(training_wall_s, 1e-9), 1e-9
        )
        full_steps = (
            int(
                np.ceil(
                    int(config["training_contract"]["full_example_count"])
                    / int(training["batch_size"])
                )
            )
            * int(training["epochs"])
        )
        estimated_full_run = {
            "method": (
                "preflight_step_rate_scaled_with_1.5x_to_2.5x_overhead"
            ),
            "low": full_steps / steps_per_second / 60.0 * 1.5,
            "high": full_steps / steps_per_second / 60.0 * 2.5,
        }
    elif execution_mode == "fresh_full_train":
        runtime_before_report_s = time.perf_counter() - started_total
        estimated_full_run = {
            "method": "completed_full_run_observed_wall_time",
            "low": runtime_before_report_s / 60.0,
            "high": runtime_before_report_s / 60.0,
        }
    elif execution_mode == "resumed_full_train":
        estimated_full_run = {
            "method": "unavailable_from_partial_resumed_process",
            "low": None,
            "high": None,
        }
    else:
        estimated_full_run = {
            "method": "unavailable_from_checkpoint_finalization_process",
            "low": None,
            "high": None,
        }
    mode = _report_mode(preflight)
    report_generation_started = time.perf_counter()
    report = {
        "phase": mode["phase"],
        "status": "completed",
        "execution_mode": execution_mode,
        "claim_boundary": mode["claim_boundary"],
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "training": {
            "source_example_population": int(len(dataset)),
            "example_count": int(len(example_indices)),
            "ordered_user_count": int(len(ordered_users)),
            "planned_item_count": int(len(planned_items)),
            "current_process_training_and_epoch_validation_s": (
                training_wall_s
            ),
            "completed_epoch": int(training_result["completed_epoch"]),
            "epoch_losses": list(training_result["epoch_losses"]),
            "process_statistics": process_statistics,
            "cumulative_statistics": cumulative_statistics,
            "sample": sample_stats,
        },
        "validation": {
            "frozen_contract": validation_counts,
            "evaluated_queries": int(len(evaluation_queries.user_ids)),
            "evaluated_targets": int(
                sum(len(row) for row in evaluation_queries.relevant)
            ),
            "fixed_catalog_count": int(len(evaluation_queries.catalog)),
        },
        "checkpoints": records,
        "selected_epoch_by_frozen_rule": selected_epoch,
        "formal_gates": gates,
        "resume": {
            "verified": resume_verified,
            "checkpoint_epoch": 1 if preflight else (
                None if resume_checkpoint is None else start_epoch - 1
            ),
            "independent_load_before_continue": bool(preflight),
        },
        "memberships": {
            "normal": normal_membership,
            "fixed_retrieval_catalog": fixed_membership,
            "model_item_universe": universe_membership,
        },
        "input_provenance": {
            "processed_manifest_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
            "raw_sources": raw_sources,
            "caption_cache_sha256": caption.metadata["cache_file_sha256"],
            "config_sha256": sha256_file(config_path),
            "code_commit_at_run": code_commit,
            "input_tree_clean_at_start": True,
        },
        "frozen_bpr_epoch_20": config["frozen_bpr_epoch_20"],
        "runtime_s": 0.0,
        "timings_s": {
            "checkpoint_loading_s": checkpoint_loading_s,
            "checkpoint_reevaluation_s": checkpoint_reevaluation_s,
            "report_generation_s": 0.0,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        ),
        "estimated_full_run_minutes": estimated_full_run,
        "access": {
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "faiss_run": False,
            "hybrid_run": False,
            "full_training_started": execution_mode
            in {"fresh_full_train", "resumed_full_train"},
        },
    }
    json.dumps(report, indent=2, sort_keys=True)
    _render_markdown(report)
    report["timings_s"]["report_generation_s"] = (
        time.perf_counter() - report_generation_started
    )
    report["runtime_s"] = time.perf_counter() - started_total
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if "/home/" in serialized:
        raise RuntimeError("Generated JSON contains a host path")
    _atomic_write_text(report_json, serialized + "\n")
    _write_markdown(report, report_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument("--caption-metadata", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b2b_full_two_tower.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else Path(
            "artifacts/phase_b2b0_preflight"
            if args.preflight
            else "artifacts/phase_b2b"
        )
    )
    report_json = (
        args.report_json
        if args.report_json is not None
        else Path(
            "reports/phase_b2b0/runner_preflight.json"
            if args.preflight
            else "reports/phase_b2b/full_two_tower.json"
        )
    )
    report_markdown = (
        args.report_markdown
        if args.report_markdown is not None
        else Path(
            "reports/phase_b2b0/runner_preflight.md"
            if args.preflight
            else "reports/phase_b2b/full_two_tower.md"
        )
    )
    run(
        repo_root=root,
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        config_path=(root / args.config).resolve(),
        checkpoint_dir=(root / checkpoint_dir).resolve(),
        report_json=(root / report_json).resolve(),
        report_markdown=(root / report_markdown).resolve(),
        preflight=args.preflight,
        resume_checkpoint=(
            None
            if args.resume_checkpoint is None
            else args.resume_checkpoint.resolve()
        ),
    )


if __name__ == "__main__":
    main()
