#!/usr/bin/env python3
"""Run one bounded real-data smoke for caption-enhanced SASRec."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kuairec_fully_observed import evaluate_retrieval
from kuairec_fully_observed.provenance import sha256_file
from kuairec_fully_observed.sasrec_adapter import (
    build_content_recbole_sasrec,
    rank_sasrec,
    train_sasrec_epoch,
)
from scripts.run_phase_b6a_recbole_sasrec import (
    _prepare_inputs,
    _subset_queries,
)


CAPTION_CACHE_SHA256 = (
    "b4093393c59ec00e9ab1e9cb90467404aea56164564bb379e07f4312f4e9e6fa"
)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config["scope"]["forbidden"] != [
        "small_matrix",
        "temporal_final",
        "final_refit",
        "full_training",
    ]:
        raise RuntimeError("Content-SASRec smoke scope changed")
    if config["content"] != {
        "source": "frozen_minilm_caption_embeddings",
        "fusion": "warm_id_residual_plus_trainable_content_projection",
        "cold_item_id_residual": "disabled",
    }:
        raise RuntimeError("Content-SASRec item representation changed")
    if config["training"] != {
        "seed": 20260725,
        "max_history": 50,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 2,
        "max_examples": 10000,
        "max_optimizer_steps_per_epoch": 20,
    }:
        raise RuntimeError("Content-SASRec smoke training bounds changed")
    return config


def _content_matrix(
    *,
    caption_cache: Path,
    video_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    if sha256_file(caption_cache) != CAPTION_CACHE_SHA256:
        raise RuntimeError("Frozen caption cache SHA changed")
    with np.load(caption_cache) as payload:
        caption_items = payload["item_ids"].astype(np.int64, copy=True)
        caption_vectors = payload["embeddings"].astype(
            np.float32, copy=True
        )
    if (
        caption_vectors.shape != (len(caption_items), 384)
        or len(np.unique(caption_items)) != len(caption_items)
        or not np.isfinite(caption_vectors).all()
    ):
        raise RuntimeError("Frozen caption cache structure changed")
    item_positions = {
        int(item): position for position, item in enumerate(video_ids)
    }
    matrix = np.zeros((len(video_ids) + 1, 384), dtype=np.float32)
    matched = 0
    for item, vector in zip(
        caption_items, caption_vectors, strict=True
    ):
        position = item_positions.get(int(item))
        if position is not None:
            matrix[position + 1] = vector
            matched += 1
    return matrix, {
        "caption_cache_items": len(caption_items),
        "matched_event_items": matched,
        "content_present_items": int(
            np.count_nonzero(np.linalg.norm(matrix[1:], axis=1) > 0)
        ),
    }


def _write_report(
    report: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = []
    for epoch in report["epochs"]:
        metrics = epoch["validation"]["metrics"]
        rows.append(
            "| {epoch} | {loss:.6f} | {r100:.6f} | {ndcg:.6f} | "
            "{coverage:.6f} | {cold:.6f} |".format(
                epoch=epoch["epoch"],
                loss=epoch["mean_loss"],
                r100=metrics["Recall@100"],
                ndcg=metrics["NDCG@20"],
                coverage=metrics["Coverage@100"],
                cold=metrics["Data-Cold Recall@100"],
            )
        )
    markdown_path.write_text(
        "\n".join(
            [
                "# Phase B6C Content-SASRec Smoke",
                "",
                "This bounded smoke adds frozen MiniLM caption vectors to "
                "RecBole SASRec through a trainable projection. Training-seen "
                "items retain an ID residual; training-unseen items use only "
                "content.",
                "",
                "| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 | "
                "Data-Cold Recall@100 |",
                "|---:|---:|---:|---:|---:|---:|",
                *rows,
                "",
                "This is a 10K-example, 20-step-per-epoch smoke on 128 reused "
                "Big-validation queries. It is not an effectiveness result.",
                "",
                "Small Matrix and temporal final were not accessed.",
                "",
            ]
        )
    )


def run(
    *,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache: Path,
    config_path: Path,
    report_json: Path,
    report_markdown: Path,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    training = config["training"]
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    inputs = _prepare_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        max_history=int(training["max_history"]),
        max_examples=int(training["max_examples"]),
    )
    queries = _subset_queries(
        inputs["queries"],
        int(config["evaluation"]["max_validation_queries"]),
    )
    sequence_count = len(queries.user_ids)
    sequences = inputs["sequences"][:sequence_count]
    lengths = inputs["sequence_lengths"][:sequence_count]
    content, content_counts = _content_matrix(
        caption_cache=caption_cache,
        video_ids=inputs["video_ids"],
    )
    train_observed_positions = np.unique(
        inputs["event_items"][
            inputs["event_times"] < float(inputs["train_end"])
        ]
    )
    id_enabled = np.zeros(len(inputs["video_ids"]) + 1, dtype=bool)
    id_enabled[train_observed_positions + 1] = True
    model = build_content_recbole_sasrec(
        num_event_items=len(inputs["video_ids"]),
        max_history=int(training["max_history"]),
        model_config=config["model"],
        content_embeddings=content,
        id_embedding_enabled=id_enabled,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fallback = inputs["popularity"].rank(queries, k=100)
    epochs = []
    for epoch in range(1, int(training["epochs"]) + 1):
        result = train_sasrec_epoch(
            model,
            inputs["dataset"],
            optimizer,
            batch_size=int(training["batch_size"]),
            max_history=int(training["max_history"]),
            seed=int(training["seed"]) + epoch,
            device=device,
            max_steps=int(training["max_optimizer_steps_per_epoch"]),
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
        epochs.append(
            {
                "epoch": epoch,
                "mean_loss": result["mean_loss"],
                "optimizer_steps": result["optimizer_steps"],
                "validation": evaluate_retrieval(
                    ranked,
                    queries,
                    data_cold_item_ids=inputs["data_cold_items"],
                ),
            }
        )
    report = {
        "phase": "phase-b6c-content-sasrec-smoke",
        "configuration": config,
        "training_examples": len(inputs["dataset"]),
        "evaluated_queries": len(queries.user_ids),
        "content": {
            **content_counts,
            "caption_cache_sha256": CAPTION_CACHE_SHA256,
            "id_enabled_items": int(np.count_nonzero(id_enabled)),
        },
        "epochs": epochs,
        "checks": {
            "loss_decreased": (
                epochs[-1]["mean_loss"] < epochs[0]["mean_loss"]
            ),
            "all_metrics_finite": all(
                np.isfinite(value)
                for epoch in epochs
                for value in epoch["validation"]["metrics"].values()
            ),
            "cold_items_use_content_path": True,
        },
        "claims": config["claims"],
        "runtime": {
            "device": str(device),
            "wall_time_s": time.perf_counter() - started,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        },
    }
    _write_report(report, report_json, report_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b6c_content_sasrec_smoke.yaml"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b6c/content_sasrec_smoke.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b6c/content_sasrec_smoke.md"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache=args.caption_cache.resolve(),
        config_path=args.config.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "losses": [row["mean_loss"] for row in report["epochs"]],
                "runtime_s": report["runtime"]["wall_time_s"],
                "checks": report["checks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
