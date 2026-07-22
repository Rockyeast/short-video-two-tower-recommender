#!/usr/bin/env python3
"""Run one frozen full-data BPR pilot on Big train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kuairec_fully_observed import (
    BPRTrainingDataset,
    PopularityBaseline,
    RetrievalQueries,
    evaluate_retrieval,
    stable_random_rank,
    train_bpr_sgd,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    expected = {
        "seed": 20260722,
        "embedding_dim": 64,
        "learning_rate": 0.05,
        "l2": 1e-4,
        "batch_size": 4096,
        "max_epochs": 20,
        "checkpoints": [5, 10, 15, 20],
    }
    if config.get("training") != expected:
        raise RuntimeError("Phase B1A training configuration is not frozen")
    if config.get("audit", {}).get("negative_seed") != 20260723:
        raise RuntimeError("Phase B1A audit-negative seed is not frozen")
    return config


def _verify_processed_artifacts(
    repo_root: Path, artifact_dir: Path
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Processed artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    statistics = manifest.get("statistics", {})
    if manifest.get("artifact_scope") != "train_and_validation_only":
        raise RuntimeError("Processed artifacts are not train/validation-only")
    if statistics.get("small_matrix_rows_read") != 0:
        raise RuntimeError("Processed artifacts accessed Small Matrix rows")
    if statistics.get("temporal_final_rows_persisted") != 0:
        raise RuntimeError("Processed artifacts contain temporal-final rows")
    split_sha = _sha256_file(repo_root / "manifests/split_manifest.json")
    if artifact_dir.name != split_sha:
        raise RuntimeError("Processed artifact directory does not match split SHA")
    if manifest.get("fingerprint", {}).get("split_manifest_sha256") != split_sha:
        raise RuntimeError("Processed artifact split fingerprint is stale")
    for name in ("events_train_validation.npz", "catalog.npz"):
        path = artifact_dir / name
        expected = manifest.get("files", {}).get(name)
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"Processed artifact hash mismatch: {name}")
    return manifest


def _normal_video_ids(data_dir: Path) -> np.ndarray:
    path = data_dir / "item_daily_features.csv"
    frame = pd.read_csv(path, usecols=["video_id", "video_type"])
    variation = frame.groupby("video_id")["video_type"].nunique(dropna=False)
    if (variation > 1).any():
        raise RuntimeError("video_type changes across daily snapshots")
    first = frame.drop_duplicates("video_id", keep="first")
    return np.sort(
        first.loc[first["video_type"].eq("NORMAL"), "video_id"].to_numpy(
            np.int64
        )
    )


def _prepare_inputs(
    artifact_dir: Path, data_dir: Path, *, training_seed: int
) -> tuple[
    BPRTrainingDataset,
    RetrievalQueries,
    PopularityBaseline,
    np.ndarray,
    dict[str, int],
]:
    events_file = np.load(artifact_dir / "events_train_validation.npz")
    catalog_file = np.load(artifact_dir / "catalog.npz")
    event_users = events_file["user"].astype(np.int64, copy=False)
    event_items = events_file["item"].astype(np.int64, copy=False)
    event_times = events_file["timestamp"].astype(np.float64, copy=False)
    event_strong = events_file["strong"].astype(bool, copy=False)
    user_indptr = events_file["user_indptr"].astype(np.int64, copy=False)
    video_ids = catalog_file["video_ids"].astype(np.int64, copy=False)
    train_end = float(catalog_file["train_end"][0])

    normal_ids = _normal_video_ids(data_dir)
    normal_item = np.isin(video_ids, normal_ids)
    event_normal = normal_item[event_items]
    train_event = event_times < train_end
    train_normal = train_event & event_normal
    positive = train_normal & event_strong
    negative_catalog = np.unique(event_items[train_normal])
    positive_users = event_users[positive]
    positive_items = event_items[positive]
    known_sets: defaultdict[int, set[int]] = defaultdict(set)
    for user, item in zip(positive_users, positive_items, strict=True):
        known_sets[int(user)].add(int(item))
    dataset = BPRTrainingDataset(
        user_ids=positive_users,
        positive_item_ids=positive_items,
        negative_catalog=negative_catalog,
        known_positive_items={
            user: frozenset(items) for user, items in known_sets.items()
        },
        seed=int(training_seed),
    )

    fixed_catalog = np.unique(event_items[event_normal])
    query_users: list[int] = []
    candidates: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    warm: list[bool] = []
    for user in range(len(user_indptr) - 1):
        start = int(user_indptr[user])
        end = int(user_indptr[user + 1])
        times = event_times[start:end]
        items = event_items[start:end]
        strong = event_strong[start:end]
        normal = event_normal[start:end]
        train_row = times < train_end
        seen = np.unique(items[train_row])
        validation_relevant = np.unique(
            items[(~train_row) & strong & normal & ~np.isin(items, seen)]
        )
        if not len(validation_relevant):
            continue
        candidate_row = np.setdiff1d(
            fixed_catalog, seen, assume_unique=True
        ).astype(np.int64, copy=False)
        if not np.isin(validation_relevant, candidate_row).all():
            raise RuntimeError("Validation target is outside fixed candidates")
        query_users.append(user)
        candidates.append(candidate_row)
        relevant.append(validation_relevant)
        warm.append(bool(train_row.any()))
    empty_histories = tuple(
        np.asarray([], dtype=np.int64) for _ in query_users
    )
    empty_weights = tuple(
        np.asarray([], dtype=np.float32) for _ in query_users
    )
    queries = RetrievalQueries(
        user_ids=np.asarray(query_users, dtype=np.int64),
        histories=empty_histories,
        history_weights=empty_weights,
        candidates=tuple(candidates),
        relevant=tuple(relevant),
        catalog=fixed_catalog,
        warm_user_mask=np.asarray(warm, dtype=bool),
    )
    positive_counts = np.bincount(
        positive_items, minlength=len(video_ids)
    )
    popularity = PopularityBaseline(
        {
            int(item): float(positive_counts[int(item)])
            for item in fixed_catalog
            if positive_counts[int(item)] > 0
        }
    )
    train_observed = np.unique(event_items[train_event])
    data_cold = np.setdiff1d(
        fixed_catalog, train_observed, assume_unique=True
    )
    counts = {
        "canonical_train_validation_events": int(len(event_users)),
        "canonical_train_events": int(train_event.sum()),
        "bpr_positive_events": int(len(positive_users)),
        "training_users": int(len(np.unique(positive_users))),
        "fit_observed_normal_items": int(len(negative_catalog)),
        "fixed_validation_catalog_items": int(len(fixed_catalog)),
        "validation_queries": int(len(query_users)),
        "validation_targets": int(sum(len(row) for row in relevant)),
        "warm_validation_queries": int(np.count_nonzero(warm)),
        "data_cold_items": int(len(data_cold)),
    }
    return dataset, queries, popularity, data_cold, counts


def _audit_pairwise(
    model,
    dataset: BPRTrainingDataset,
    negatives: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> dict[str, float | int]:
    user_position = np.searchsorted(model.user_ids, dataset.user_ids)
    positive_position = np.searchsorted(
        model.item_ids, dataset.positive_item_ids
    )
    negative_position = np.searchsorted(model.item_ids, negatives)
    total_loss = 0.0
    wins = 0
    count = len(dataset.user_ids)
    for begin in range(0, count, chunk_size):
        end = min(begin + chunk_size, count)
        user = model.user_factors[user_position[begin:end]]
        positive = model.item_factors[positive_position[begin:end]]
        negative = model.item_factors[negative_position[begin:end]]
        margin = np.sum(user * (positive - negative), axis=1)
        if not np.isfinite(margin).all():
            raise FloatingPointError("Audit margins became non-finite")
        total_loss += float(np.logaddexp(0.0, -margin).sum())
        wins += int(np.count_nonzero(margin > 0.0))
    return {
        "pair_count": count,
        "pairwise_loss": total_loss / count,
        "win_rate": wins / count,
    }


def _save_checkpoint(path: Path, epoch: int, model) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        epoch=np.asarray([epoch], dtype=np.int64),
        user_ids=model.user_ids,
        item_ids=model.item_ids,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
    )
    return _sha256_file(path)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = []
    for row in report["checkpoints"]:
        metrics = row.get("validation", {}).get("metrics", {})
        rows.append(
            "| {epoch} | {loss:.6f} | {win:.4f} | {r20:.6f} | "
            "{r50:.6f} | {r100:.6f} | {ndcg:.6f} | {coverage:.6f} |".format(
                epoch=row["epoch"],
                loss=row["audit"]["pairwise_loss"],
                win=row["audit"]["win_rate"],
                r20=metrics.get("Recall@20", float("nan")),
                r50=metrics.get("Recall@50", float("nan")),
                r100=metrics.get("Recall@100", float("nan")),
                ndcg=metrics.get("NDCG@20", float("nan")),
                coverage=metrics.get("Coverage@100", float("nan")),
            )
        )
    selected = report.get("selected_checkpoint")
    baselines = report["baselines"]
    random_metrics = baselines["random"]["metrics"]
    popularity_metrics = baselines["global_popularity"]["metrics"]
    text = "\n".join(
        [
            "# Phase B1A Full BPR Pilot",
            "",
            "Big train fit and Big validation selection only. Small Matrix, "
            "temporal final and Two-Tower were not accessed or run.",
            "",
            "## Baselines",
            "",
            "| Method | Recall@100 | NDCG@20 | Coverage@100 |",
            "|---|---:|---:|---:|",
            "| Random | {r:.6f} | {n:.6f} | {c:.6f} |".format(
                r=random_metrics["Recall@100"],
                n=random_metrics["NDCG@20"],
                c=random_metrics["Coverage@100"],
            ),
            "| Global Popularity | {r:.6f} | {n:.6f} | {c:.6f} |".format(
                r=popularity_metrics["Recall@100"],
                n=popularity_metrics["NDCG@20"],
                c=popularity_metrics["Coverage@100"],
            ),
            "",
            "## BPR checkpoints",
            "",
            "| Epoch | Audit loss | Audit win | Recall@20 | Recall@50 | "
            "Recall@100 | NDCG@20 | Coverage@100 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Status: `{report['status']}`.",
            f"Selected checkpoint: `{selected}`.",
            f"Stop reason: `{report.get('stop_reason')}`.",
            "",
            f"Validation queries: `{report['counts']['validation_queries']}`; "
            f"targets: `{report['counts']['validation_targets']}`; runtime: "
            f"`{report['runtime_s'] / 60.0:.2f} minutes`.",
            "",
            "The fixed audit-negative metrics are optimization diagnostics, "
            "not recommendation-effectiveness claims.",
            "",
        ]
    )
    path.write_text(text)


def run(
    repo_root: Path,
    *,
    config_path: Path,
    processed_artifact_dir: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    artifact_manifest = _verify_processed_artifacts(
        repo_root, processed_artifact_dir
    )
    training = config["training"]
    dataset, queries, popularity, data_cold, counts = _prepare_inputs(
        processed_artifact_dir,
        data_dir,
        training_seed=int(training["seed"]),
    )
    audit_dataset = replace(
        dataset, seed=int(config["audit"]["negative_seed"])
    )
    audit_negatives = audit_dataset.sample_negatives(0)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    audit_path = checkpoint_dir / "audit_negatives.npy"
    np.save(audit_path, audit_negatives)

    random_result = evaluate_retrieval(
        stable_random_rank(queries, seed=20260722, k=100),
        queries,
        data_cold_item_ids=data_cold,
    )
    popularity_result = evaluate_retrieval(
        popularity.rank(queries, k=100),
        queries,
        data_cold_item_ids=data_cold,
    )
    records: list[dict[str, Any]] = []
    initialization_audit: dict[str, float | int] | None = None
    stop_reason: str | None = None

    def checkpoint_callback(epoch, model, epoch_losses):
        nonlocal initialization_audit, stop_reason
        audit = _audit_pairwise(model, dataset, audit_negatives)
        if epoch == 0:
            initialization_audit = audit
            return True
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.npz"
        checkpoint_sha = _save_checkpoint(checkpoint_path, epoch, model)
        ranked = model.rank(
            queries,
            k=100,
            cold_user_fallback=popularity,
            score_block_size=128,
        )
        validation = evaluate_retrieval(
            ranked, queries, data_cold_item_ids=data_cold
        )
        record = {
            "epoch": epoch,
            "resampled_training_loss": epoch_losses[-1],
            "audit": audit,
            "validation": validation,
            "checkpoint": {
                "path": _report_path(checkpoint_path, repo_root),
                "sha256": checkpoint_sha,
            },
            "elapsed_s": time.perf_counter() - started,
        }
        if not all(
            np.isfinite(value)
            for value in (
                audit["pairwise_loss"],
                audit["win_rate"],
                *validation["metrics"].values(),
            )
        ):
            stop_reason = f"non_finite_metric_at_epoch_{epoch}"
        if epoch == 5:
            if (
                validation["metrics"]["Recall@100"]
                <= random_result["metrics"]["Recall@100"]
            ):
                stop_reason = "epoch_5_did_not_beat_random_recall_at_100"
            elif (
                audit["win_rate"] - initialization_audit["win_rate"]
                < config["stopping"][
                    "epoch_5_audit_win_rate_minimum_improvement"
                ]
            ):
                stop_reason = "epoch_5_audit_win_rate_improvement_below_0.02"
        record["stop_reason"] = stop_reason
        records.append(record)
        return stop_reason is None

    result = train_bpr_sgd(
        dataset,
        embedding_dim=int(training["embedding_dim"]),
        learning_rate=float(training["learning_rate"]),
        l2=float(training["l2"]),
        epochs=int(training["max_epochs"]),
        batch_size=int(training["batch_size"]),
        checkpoint_epochs=(0, *tuple(training["checkpoints"])),
        checkpoint_callback=checkpoint_callback,
    )
    selected = None
    if records:
        selected = max(
            records,
            key=lambda row: (
                row["validation"]["metrics"]["Recall@100"],
                row["validation"]["metrics"]["NDCG@20"],
            ),
        )["epoch"]
    report = {
        "phase": config["phase"],
        "status": "stopped_by_gate" if stop_reason else "completed",
        "stop_reason": stop_reason,
        "selected_checkpoint": selected,
        "claim_boundary": {
            "big_train_fit": True,
            "big_validation_selection": True,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "two_tower_run": False,
            "configuration_count": 1,
            "seed_count": 1,
        },
        "configuration": config,
        "counts": counts,
        "baselines": {
            "random": random_result,
            "global_popularity": popularity_result,
        },
        "initialization_audit": initialization_audit,
        "checkpoints": records,
        "epochs_completed": len(result.epoch_losses),
        "resampled_training_losses": list(result.epoch_losses),
        "artifacts": {
            "processed_manifest_sha256": _sha256_file(
                processed_artifact_dir / "manifest.json"
            ),
            "processed_events_sha256": artifact_manifest["files"][
                "events_train_validation.npz"
            ],
            "audit_negatives_path": _report_path(audit_path, repo_root),
            "audit_negatives_sha256": _sha256_file(audit_path),
            "config_sha256": _sha256_file(config_path),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
        },
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, report_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b1a_bpr_pilot.yaml")
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/phase_b1a")
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b1a/full_bpr_pilot.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b1a/full_bpr_pilot.md"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = run(
        repo_root,
        config_path=(repo_root / args.config).resolve(),
        processed_artifact_dir=args.processed_artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        checkpoint_dir=(repo_root / args.checkpoint_dir).resolve(),
        report_json=(repo_root / args.report_json).resolve(),
        report_markdown=(repo_root / args.report_markdown).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
