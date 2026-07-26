#!/usr/bin/env python3
"""Train one fixed caption-enhanced SASRec on full Big train/validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kuairec_fully_observed import evaluate_retrieval
from kuairec_fully_observed.sasrec_adapter import (
    build_content_recbole_sasrec,
    rank_sasrec,
    train_sasrec_epoch,
)
from scripts.run_phase_b6a_recbole_sasrec import _prepare_inputs
from scripts.run_phase_b6c_content_sasrec_smoke import (
    CAPTION_CACHE_SHA256,
    _content_matrix,
)


REFERENCES = {
    "BPR": {
        "Recall@100": 0.04843855379304513,
        "NDCG@20": 0.012773561140700995,
        "Coverage@100": 0.33304858515750135,
        "Data-Cold Recall@100": 0.0,
    },
    "Two-Tower": {
        "Recall@100": 0.0720568501336648,
        "NDCG@20": 0.012112516931264626,
        "Coverage@100": 0.5694607581420181,
        "Data-Cold Recall@100": 0.06515057123608131,
    },
    "SASRec": {
        "Recall@100": 0.07756014970881014,
        "NDCG@20": 0.0374616950131969,
        "Coverage@100": 0.25285638013881473,
        "Data-Cold Recall@100": 0.0,
    },
}


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config["phase"] != "phase-b6d-content-sasrec-full":
        raise RuntimeError("Content-SASRec full phase changed")
    if config["scope"] != {
        "fit": "canonical_big_train",
        "evaluate": "reused_big_validation_development",
        "forbidden": ["small_matrix", "temporal_final", "final_refit"],
    }:
        raise RuntimeError("Content-SASRec full scope changed")
    if config["training"] != {
        "seed": 20260725,
        "max_history": 50,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 5,
    }:
        raise RuntimeError("Content-SASRec full training config changed")
    if config["content"] != {
        "source": "frozen_minilm_caption_embeddings",
        "fusion": "warm_id_residual_plus_trainable_content_projection",
        "cold_item_id_residual": "disabled",
    }:
        raise RuntimeError("Content-SASRec full item representation changed")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(report: dict[str, Any]) -> str:
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
    return "\n".join(
        [
            "# Phase B6D Content-Enhanced SASRec",
            "",
            "One fixed, single-seed development run on reused Big validation.",
            "Frozen MiniLM caption vectors are projected into SASRec; "
            "training-unseen items have their ID residual disabled.",
            "",
            "| Epoch | Loss | Recall@20 | Recall@50 | Recall@100 | "
            "NDCG@20 | Coverage@100 | Data-Cold Recall@100 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected epoch: `{report['selected_epoch']}`.",
            "",
            "Small Matrix and temporal final were not accessed. These are "
            "adaptive Big-validation point estimates, not sealed-test or "
            "significance evidence.",
            "",
        ]
    )


def run(
    *,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache: Path,
    config_path: Path,
    checkpoint_dir: Path,
    report_json: Path,
    report_markdown: Path,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(seed)

    inputs = _prepare_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        max_history=int(training["max_history"]),
        max_examples=None,
    )
    if (
        len(inputs["queries"].user_ids) != 6818
        or len(inputs["queries"].catalog) != 9365
        or len(inputs["dataset"]) != 573104
    ):
        raise RuntimeError("Content-SASRec full data contract changed")
    content, content_counts = _content_matrix(
        caption_cache=caption_cache,
        video_ids=inputs["video_ids"],
    )
    train_observed = np.unique(
        inputs["event_items"][
            inputs["event_times"] < float(inputs["train_end"])
        ]
    )
    id_enabled = np.zeros(len(inputs["video_ids"]) + 1, dtype=bool)
    id_enabled[train_observed + 1] = True
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
    fallback = inputs["popularity"].rank(inputs["queries"], k=100)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = []
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        training_result = train_sasrec_epoch(
            model,
            inputs["dataset"],
            optimizer,
            batch_size=int(training["batch_size"]),
            max_history=int(training["max_history"]),
            seed=seed + epoch,
            device=device,
        )
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "caption_cache_sha256": CAPTION_CACHE_SHA256,
            },
            checkpoint,
        )
        ranked = rank_sasrec(
            model,
            queries=inputs["queries"],
            sequences=inputs["sequences"],
            sequence_lengths=inputs["sequence_lengths"],
            video_ids=inputs["video_ids"],
            fallback_topk=fallback,
            device=device,
            k=100,
            batch_size=128,
        )
        validation = evaluate_retrieval(
            ranked,
            inputs["queries"],
            data_cold_item_ids=inputs["data_cold_items"],
        )
        epochs.append(
            {
                "epoch": epoch,
                "training": training_result,
                "validation": validation,
                "checkpoint": {
                    "name": checkpoint.name,
                    "sha256": _sha256(checkpoint),
                },
                "wall_time_s": time.perf_counter() - epoch_started,
            }
        )
    selected = max(
        epochs,
        key=lambda record: (
            record["validation"]["metrics"]["Recall@100"],
            record["validation"]["metrics"]["NDCG@20"],
            -record["epoch"],
        ),
    )
    report = {
        "phase": "phase-b6d-content-sasrec-full",
        "configuration": config,
        "training_examples": len(inputs["dataset"]),
        "validation_counts": inputs["validation_counts"],
        "content": {
            **content_counts,
            "caption_cache_sha256": CAPTION_CACHE_SHA256,
            "id_enabled_items": int(np.count_nonzero(id_enabled)),
        },
        "epochs": epochs,
        "selected_epoch": selected["epoch"],
        "selected_metrics": selected["validation"],
        "references": REFERENCES,
        "claims": config["claims"],
        "runtime": {
            "device": str(device),
            "wall_time_s": time.perf_counter() - started,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_markdown.write_text(_render(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b6d_content_sasrec_full.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache=args.caption_cache.resolve(),
        config_path=args.config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "selected_epoch": report["selected_epoch"],
                "runtime_s": report["runtime"]["wall_time_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
