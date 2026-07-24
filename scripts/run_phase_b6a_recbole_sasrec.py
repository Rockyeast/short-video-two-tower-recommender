#!/usr/bin/env python3
"""Run one fixed RecBole SASRec experiment on frozen Big validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kuairec_fully_observed import PopularityBaseline, evaluate_retrieval
from kuairec_fully_observed.data import RetrievalQueries
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.full_training import (
    build_validation_contract,
    verify_validation_contract,
)
from kuairec_fully_observed.provenance import (
    PHASE1_PROCESSED_MANIFEST_SHA256,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.sasrec_adapter import (
    SASRecTrainingDataset,
    build_recbole_sasrec,
    build_validation_sequences,
    rank_sasrec,
    train_sasrec_epoch,
)
from scripts.run_phase_b2b_full_two_tower import EXPECTED_VALIDATION
from scripts.run_phase_b2b_full_two_tower import (
    _processed_popularity as processed_popularity,
)


BPR_REFERENCE = {
    "Recall@100": 0.04843855379304513,
    "NDCG@20": 0.012773561140700995,
    "Coverage@100": 0.33304858515750135,
    "Data-Cold Recall@100": 0.0,
}
TWO_TOWER_REFERENCE = {
    "Recall@100": 0.0720568501336648,
    "NDCG@20": 0.012112516931264626,
    "Coverage@100": 0.5694607581420181,
    "Data-Cold Recall@100": 0.06515057123608131,
}


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config["implementation"] != {
        "library": "recbole",
        "model": "SASRec",
        "version": "1.2.1",
    }:
        raise RuntimeError("SASRec implementation identity changed")
    if config["scope"] != {
        "fit": "canonical_big_train",
        "evaluate": "reused_big_validation_development",
        "forbidden": ["small_matrix", "temporal_final", "final_refit"],
    }:
        raise RuntimeError("SASRec experiment scope changed")
    if config["claims"]["one_configuration"] is not True:
        raise RuntimeError("Only one SASRec configuration is allowed")
    if config["claims"]["one_seed"] is not True:
        raise RuntimeError("Only one SASRec seed is allowed")
    return config


def _subset_queries(queries: RetrievalQueries, count: int) -> RetrievalQueries:
    indices = np.arange(min(int(count), len(queries.user_ids)))
    return RetrievalQueries(
        user_ids=queries.user_ids[indices],
        histories=tuple(queries.histories[int(i)] for i in indices),
        history_weights=tuple(
            queries.history_weights[int(i)] for i in indices
        ),
        candidates=tuple(queries.candidates[int(i)] for i in indices),
        relevant=tuple(queries.relevant[int(i)] for i in indices),
        catalog=queries.catalog,
        warm_user_mask=queries.warm_user_mask[indices],
        diagnostics=dict(queries.diagnostics),
    )


def _prepare_inputs(
    *,
    data_dir: Path,
    artifact_dir: Path,
    max_history: int,
    max_examples: int | None,
) -> dict[str, Any]:
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
    normal_item_mask = np.isin(video_ids, static.normal_item_ids)
    queries, data_cold_items, validation_counts = build_validation_contract(
        event_users=event_users,
        event_items=event_items,
        event_times=event_times,
        event_strong=event_strong,
        user_indptr=user_indptr,
        actual_user_ids=actual_user_ids,
        video_ids=video_ids,
        normal_item_mask=normal_item_mask,
        train_end=train_end,
        train_events=None,
    )
    verify_validation_contract(
        queries=queries,
        counts=validation_counts,
        expected=EXPECTED_VALIDATION["expected"],
    )
    dataset = SASRecTrainingDataset(
        event_users=event_users,
        event_items=event_items,
        event_times=event_times,
        event_strong=event_strong,
        user_indptr=user_indptr,
        normal_item_mask=normal_item_mask,
        train_end=train_end,
        max_history=max_history,
        max_examples=max_examples,
    )
    sequences, lengths = build_validation_sequences(
        query_user_ids=queries.user_ids,
        actual_user_ids=actual_user_ids,
        event_items=event_items,
        event_times=event_times,
        user_indptr=user_indptr,
        train_end=train_end,
        max_history=max_history,
    )
    popularity = processed_popularity(
        event_items=event_items,
        event_times=event_times,
        event_strong=event_strong,
        video_ids=video_ids,
        normal_item_mask=normal_item_mask,
        train_end=train_end,
    )
    return {
        "dataset": dataset,
        "queries": queries,
        "sequences": sequences,
        "sequence_lengths": lengths,
        "data_cold_items": data_cold_items,
        "video_ids": video_ids,
        "popularity": popularity,
        "validation_counts": validation_counts,
        "raw_sources": raw_sources,
        "processed_manifest": manifest,
    }


def _checkpoint_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = []
    for record in report["epochs"]:
        metrics = record["validation"]["metrics"]
        rows.append(
            "| {epoch} | {loss:.6f} | {r20:.6f} | {r50:.6f} | "
            "{r100:.6f} | {ndcg:.6f} | {coverage:.6f} | {cold:.6f} |".format(
                epoch=record["epoch"],
                loss=record["training"]["mean_loss"],
                r20=metrics["Recall@20"],
                r50=metrics["Recall@50"],
                r100=metrics["Recall@100"],
                ndcg=metrics["NDCG@20"],
                coverage=metrics["Coverage@100"],
                cold=metrics["Data-Cold Recall@100"],
            )
        )
    mode = "bounded smoke" if report["mode"] == "smoke" else "full development run"
    text = "\n".join(
        [
            "# Phase B6A RecBole SASRec",
            "",
            f"This is a **{mode}** on Big validation. Small Matrix and "
            "temporal final were not accessed.",
            "",
            "| Epoch | Loss | Recall@20 | Recall@50 | Recall@100 | "
            "NDCG@20 | Coverage@100 | Data-Cold Recall@100 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected epoch: `{report['selected_epoch']}`.",
            f"Wall time: `{report['runtime']['wall_time_s']:.2f}s`; "
            f"peak RSS: `{report['runtime']['peak_rss_mb']:.1f} MB`.",
            "",
            "The model implementation is RecBole 1.2.1 SASRec. The adapter "
            "reuses the repository's frozen candidate membership and metrics. "
            "This is adaptive Big-validation development, not a sealed test.",
            "",
        ]
    )
    md_path.write_text(text)


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    config_path: Path,
    checkpoint_dir: Path,
    report_json: Path,
    report_markdown: Path,
    smoke: bool,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    inputs = _prepare_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        max_history=int(training["max_history"]),
        max_examples=10_000 if smoke else None,
    )
    queries = (
        _subset_queries(inputs["queries"], 128)
        if smoke
        else inputs["queries"]
    )
    sequence_count = len(queries.user_ids)
    sequences = inputs["sequences"][:sequence_count]
    lengths = inputs["sequence_lengths"][:sequence_count]
    fallback = inputs["popularity"].rank(queries, k=100)

    model = build_recbole_sasrec(
        num_event_items=len(inputs["video_ids"]),
        max_history=int(training["max_history"]),
        model_config=config["model"],
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epoch_count = 2 if smoke else int(training["epochs"])
    records: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epoch_count + 1):
        epoch_started = time.perf_counter()
        train_result = train_sasrec_epoch(
            model,
            inputs["dataset"],
            optimizer,
            batch_size=int(training["batch_size"]),
            max_history=int(training["max_history"]),
            seed=seed + epoch,
            device=device,
            max_steps=20 if smoke else None,
        )
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            },
            checkpoint,
        )
        ranked = rank_sasrec(
            model,
            queries=queries,
            sequences=sequences,
            sequence_lengths=lengths,
            video_ids=inputs["video_ids"],
            fallback_topk=fallback,
            device=device,
            k=100,
            batch_size=128,
        )
        validation = evaluate_retrieval(
            ranked,
            queries,
            data_cold_item_ids=inputs["data_cold_items"],
        )
        records.append(
            {
                "epoch": epoch,
                "training": train_result,
                "validation": validation,
                "checkpoint": {
                    "locator": str(checkpoint.relative_to(repo_root)),
                    "sha256": _checkpoint_sha(checkpoint),
                },
                "epoch_wall_time_s": time.perf_counter() - epoch_started,
            }
        )
    selected = max(
        records,
        key=lambda row: (
            row["validation"]["metrics"]["Recall@100"],
            row["validation"]["metrics"]["NDCG@20"],
            -row["epoch"],
        ),
    )
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report = {
        "phase": "phase-b6a-recbole-sasrec",
        "mode": "smoke" if smoke else "full_big_validation_development",
        "configuration": config,
        "training_examples": len(inputs["dataset"]),
        "validation_counts": inputs["validation_counts"],
        "evaluated_query_count": len(queries.user_ids),
        "model_history_query_count": int(np.count_nonzero(lengths)),
        "fallback_query_count": int(np.count_nonzero(lengths == 0)),
        "epochs": records,
        "selected_epoch": selected["epoch"],
        "selected_metrics": selected["validation"],
        "references": {
            "frozen_bpr_epoch_20": BPR_REFERENCE,
            "two_tower_epoch_1": TWO_TOWER_REFERENCE,
        },
        "claims": {
            "adaptive_validation_development": True,
            "sealed_test_claim": False,
            "effectiveness_claim": not smoke,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "one_configuration": True,
            "one_seed": True,
        },
        "runtime": {
            "device": str(device),
            "wall_time_s": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb,
        },
        "inputs": {
            "processed_manifest_sha256": PHASE1_PROCESSED_MANIFEST_SHA256,
            "raw_sources": inputs["raw_sources"],
        },
    }
    _write_report(report, report_json, report_markdown)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, required=True
    )
    parser.add_argument(
        "--processed-artifact-dir", type=Path, required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b6a_recbole_sasrec.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/phase_b6a/smoke"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b6a/sasrec_smoke.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b6a/sasrec_smoke.md"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        repo_root=Path(__file__).resolve().parents[1],
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        config_path=args.config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
        smoke=bool(args.smoke),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
