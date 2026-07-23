#!/usr/bin/env python3
"""Refit the frozen final recipe on canonical Big train plus validation."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kuairec_fully_observed import PopularityBaseline, train_bpr_sgd
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.full_training import (
    build_checkpoint_identity,
    load_canonical_train_events,
    planned_training_membership,
    save_full_epoch_checkpoint,
    train_full_two_tower,
)
from kuairec_fully_observed.provenance import (
    canonical_json_sha256,
    membership_record,
    normal_membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.torch_models import TwoTowerV1
from kuairec_fully_observed.torch_training import (
    prepare_item_feature_store,
    resolve_concrete_device,
)
from kuairec_fully_observed.training import (
    BPRTrainingDataset,
    build_two_tower_training_dataset,
)


def _load_context(
    *, artifact_dir: Path, normal_item_ids: np.ndarray
) -> dict[str, Any]:
    with np.load(artifact_dir / "events_train_validation.npz") as events, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog:
        event_users = events["user"].astype(np.int64, copy=True)
        event_items = events["item"].astype(np.int64, copy=True)
        event_times = events["timestamp"].astype(np.float64, copy=True)
        event_strong = events["strong"].astype(bool, copy=True)
        actual_users = events["user_ids"].astype(np.int64, copy=True)
        video_ids = catalog["video_ids"].astype(np.int64, copy=True)
        validation_end = float(catalog["validation_end"][0])
    fit = event_times < validation_end
    normal_positions = np.isin(video_ids, normal_item_ids)
    fit_normal = fit & normal_positions[event_items]
    positive = fit_normal & event_strong
    return {
        "user_ids": actual_users[event_users],
        "item_ids": video_ids[event_items],
        "times": event_times,
        "strong": event_strong,
        "fit": fit,
        "fit_normal": fit_normal,
        "positive": positive,
        "validation_end": validation_end,
        "video_ids": video_ids,
    }


def _fit_popularity(context: dict[str, Any]) -> PopularityBaseline:
    positives = context["item_ids"][context["positive"]]
    values, counts = np.unique(positives, return_counts=True)
    return PopularityBaseline(
        {int(item): float(count) for item, count in zip(values, counts, strict=True)}
    )


def _fit_bpr(
    *,
    context: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    users = context["user_ids"][context["positive"]]
    items = context["item_ids"][context["positive"]]
    catalog = np.unique(context["item_ids"][context["fit_normal"]])
    known: defaultdict[int, set[int]] = defaultdict(set)
    for user, item in zip(users, items, strict=True):
        known[int(user)].add(int(item))
    dataset = BPRTrainingDataset(
        user_ids=users,
        positive_item_ids=items,
        negative_catalog=catalog,
        known_positive_items={
            user: frozenset(values) for user, values in known.items()
        },
        seed=int(config["seed"]),
    )
    result = train_bpr_sgd(
        dataset,
        embedding_dim=int(config["embedding_dim"]),
        learning_rate=float(config["learning_rate"]),
        l2=float(config["l2"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        epoch=np.asarray([config["epochs"]], dtype=np.int64),
        user_ids=result.model.user_ids,
        item_ids=result.model.item_ids,
        user_factors=result.model.user_factors,
        item_factors=result.model.item_factors,
    )
    return {
        "fit_interactions": int(context["fit"].sum()),
        "strong_positive_events": int(context["positive"].sum()),
        "training_users": int(len(result.model.user_ids)),
        "training_items": int(len(result.model.item_ids)),
        "epochs": int(config["epochs"]),
        "optimizer_steps": int(
            np.ceil(len(dataset) / int(config["batch_size"])) * config["epochs"]
        ),
        "epoch_losses": list(result.epoch_losses),
        "final_loss": float(result.epoch_losses[-1]),
        "wall_time_s": time.perf_counter() - started,
        "checkpoint_path": "artifacts/phase_b3b/final_bpr_epoch_020.npz",
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _fit_two_tower(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    raw_sources: dict[str, Any],
    static,
    context: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    training = config["training"]
    canonical = load_canonical_train_events(
        data_dir, train_end=float(context["validation_end"])
    )
    dataset = build_two_tower_training_dataset(
        canonical,
        max_history=int(config["architecture"]["max_history"]),
        normal_item_ids=static.normal_item_ids,
    )
    example_indices = np.arange(len(dataset), dtype=np.int64)
    ordered_users, planned_items = planned_training_membership(
        dataset, example_indices
    )
    fit_history_items = np.unique(context["item_ids"][context["fit"]])
    item_universe = np.union1d(
        fit_history_items, np.unique(static.normal_item_ids)
    ).astype(np.int64)
    frame = static.frame.set_index("video_id").reindex(item_universe)
    caption_config = config["caption"]
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=item_universe,
        expected_model_id=caption_config["model_id"],
        expected_revision=caption_config["resolved_revision"],
        expected_source_sha256=raw_sources[
            "kuairec_caption_category.csv"
        ]["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            item_universe, frame["caption_text"].astype(str).tolist()
        ),
    )
    fit_normal_items = np.intersect1d(
        fit_history_items, static.normal_item_ids, assume_unique=True
    )
    store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=item_universe,
        train_observed_item_ids=fit_history_items,
        train_observed_normal_item_ids=fit_normal_items,
    )
    dimensions = {
        "num_items": int(len(store.item_ids)),
        "num_users": int(len(ordered_users)),
        "num_category_tokens": int(len(store.category_vocab)),
        "num_upload_types": int(len(store.upload_type_vocab)),
    }
    target_device = resolve_concrete_device(device)
    torch.manual_seed(int(training["seed"]))
    model = TwoTowerV1(**dimensions).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    base_identity = {
        "config": {
            "locator": "configs/phase_b3b_final_recipe.yaml",
            "sha256": sha256_file(config_path),
        },
        "fit_context": "canonical_big_train_plus_validation",
        "raw_inputs": raw_sources,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "memberships": {
            "normal": normal_membership_record(
                np.unique(static.normal_item_ids)
            ),
            "model_item_universe": membership_record(
                store.item_ids, label="phase-b3b-refit-item-universe-v1"
            ),
        },
        "feature_identity": {
            "category_vocab_sha256": canonical_json_sha256(
                [
                    [level, raw, index]
                    for (level, raw), index in sorted(
                        store.category_vocab.items()
                    )
                ],
                label="phase-b3b-category-vocab-v1",
            ),
            "upload_type_vocab_sha256": canonical_json_sha256(
                sorted(store.upload_type_vocab.items()),
                label="phase-b3b-upload-type-vocab-v1",
            ),
            "numeric_preprocessing_sha256": canonical_json_sha256(
                store.preprocessing,
                label="phase-b3b-numeric-preprocessing-v1",
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
    }
    saved: dict[str, Any] = {}

    def save_epoch(
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
        saved.update(
            {
                "identity_sha256": canonical_json_sha256(
                    identity, label="phase-b2a-checkpoint-identity-v2"
                ),
                "touched_users": int(len(touched_users)),
                "touched_items": int(len(touched_items)),
            }
        )

    result = train_full_two_tower(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        example_indices=example_indices,
        store=store,
        ordered_user_ids=ordered_users,
        planned_item_ids=planned_items,
        device=target_device,
        seed=int(training["seed"]),
        diagnostic_seed=int(training["diagnostic_seed"]),
        start_epoch=1,
        end_epoch=1,
        batch_size=int(training["batch_size"]),
        temperature=float(training["temperature"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        checkpoint_callback=save_epoch,
    )
    return {
        "fit_interactions": int(len(canonical)),
        "strong_positive_examples": int(len(dataset)),
        "training_users": int(len(ordered_users)),
        "model_item_universe": int(len(store.item_ids)),
        "epochs": 1,
        "optimizer_steps": int(
            result["cumulative_statistics"]["optimizer_steps"]
        ),
        "skipped_batches": int(
            result["cumulative_statistics"]["skipped_batches"]
        ),
        "final_loss": float(result["epoch_losses"][-1]),
        "wall_time_s": time.perf_counter() - started,
        "device": str(target_device),
        "checkpoint_path": "artifacts/phase_b3b/final_two_tower_epoch_001.pt",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **saved,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    bpr = report["refit"]["bpr"]
    tower = report["refit"]["two_tower"]
    return "\n".join(
        [
            "# Phase B3B0 Final Recipe Refit",
            "",
            "The recipe was frozen on Big validation. This run refits from "
            "scratch on canonical Big train plus validation and performs no "
            "model selection.",
            "",
            "| Method | Fit interactions | Strong positives/examples | "
            "Optimizer steps | Final loss | Wall time (s) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| BPR epoch 20 | {bpr['fit_interactions']} | "
            f"{bpr['strong_positive_events']} | {bpr['optimizer_steps']} | "
            f"{bpr['final_loss']:.6f} | {bpr['wall_time_s']:.2f} |",
            f"| Two-Tower epoch 1 | {tower['fit_interactions']} | "
            f"{tower['strong_positive_examples']} | "
            f"{tower['optimizer_steps']} | {tower['final_loss']:.6f} | "
            f"{tower['wall_time_s']:.2f} |",
            "",
            f"- BPR checkpoint: `{bpr['checkpoint_path']}`",
            f"- Two-Tower checkpoint: `{tower['checkpoint_path']}`",
            "- Global Popularity uses Big train+validation strong-positive counts.",
            "- Small Matrix accessed: `false`",
            "- Temporal final accessed: `false`",
            "",
        ]
    )


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    config_path: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    output_dir: Path,
    report_json: Path,
    report_markdown: Path,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    if config["fit_context"] != "canonical_big_train_plus_validation":
        raise RuntimeError("Final refit context changed")
    if config["selection_complete"] is not True:
        raise RuntimeError("Final recipe is not frozen")
    _, raw_sources = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    static = load_static_item_features(data_dir)
    context = _load_context(
        artifact_dir=artifact_dir, normal_item_ids=static.normal_item_ids
    )
    popularity = _fit_popularity(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    popularity_path = output_dir / "final_global_popularity.json"
    popularity_path.write_text(
        json.dumps(
            {str(item): score for item, score in sorted(popularity.scores.items())},
            sort_keys=True,
        )
        + "\n"
    )
    bpr = _fit_bpr(
        context=context,
        config=config["bpr"],
        checkpoint_path=output_dir / "final_bpr_epoch_020.npz",
    )
    tower = _fit_two_tower(
        repo_root=repo_root,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        raw_sources=raw_sources,
        static=static,
        context=context,
        config=config["two_tower"],
        config_path=config_path,
        caption_cache_path=caption_cache_path,
        caption_metadata_path=caption_metadata_path,
        checkpoint_path=output_dir / "final_two_tower_epoch_001.pt",
        device=device,
    )
    report = {
        "phase": "phase-b3b0-final-refit",
        "recipe_frozen_before_small": True,
        "selection_performed": False,
        "access": {
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
        },
        "fit_context": "canonical_big_train_plus_validation",
        "refit": {
            "global_popularity": {
                "strong_positive_items_with_nonzero_count": len(popularity.scores),
                "artifact_path": "artifacts/phase_b3b/final_global_popularity.json",
                "artifact_sha256": sha256_file(popularity_path),
            },
            "bpr": bpr,
            "two_tower": tower,
        },
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_markdown.write_text(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument("--caption-metadata", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b3b_final_recipe.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/phase_b3b")
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b3b0/final_refit.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b3b0/final_refit.md"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(
        repo_root=root,
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        config_path=(root / args.config).resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        output_dir=args.output_dir.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
