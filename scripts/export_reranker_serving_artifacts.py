#!/usr/bin/env python3
"""Export static/train-only features required by the optional local reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from kuairec_fully_observed import load_static_item_features
from kuairec_fully_observed.provenance import sha256_file
from kuairec_fully_observed.reranker_serving import (
    write_reranker_feature_bundle,
)
from kuairec_fully_observed.serving_bundle import load_serving_bundle


def export(
    *,
    data_dir: Path,
    artifact_dir: Path,
    serving_bundle_path: Path,
    serving_metadata_path: Path,
    reranker_model_path: Path,
    reranker_report_path: Path,
    feature_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    serving, serving_metadata = load_serving_bundle(
        bundle_path=serving_bundle_path,
        metadata_path=serving_metadata_path,
    )
    report = json.loads(reranker_report_path.read_text())
    if (
        report.get("phase")
        != "phase-b5b-recall-preserving-reranker"
        or report.get("status") != "gate_passed"
        or report.get("retrieval_preservation", {}).get(
            "top100_item_sets_exactly_equal"
        )
        is not True
    ):
        raise RuntimeError("Reranker report is not a passing local-rerank run")
    with np.load(
        artifact_dir / "events_train_validation.npz"
    ) as events, np.load(artifact_dir / "catalog.npz") as catalog:
        event_items = events["item"].astype(np.int64, copy=True)
        event_times = events["timestamp"].astype(np.float64, copy=True)
        event_strong = events["strong"].astype(bool, copy=True)
        video_ids = catalog["video_ids"].astype(np.int64, copy=True)
        train_end = float(catalog["train_end"][0])
    static = load_static_item_features(data_dir)
    item_ids = serving["two_tower_item_ids"].astype(
        np.int64, copy=True
    )
    frame = static.frame.set_index("video_id").reindex(item_ids)
    if frame["category_ids"].isna().any():
        raise RuntimeError("Reranker static features do not cover serving items")
    observed_items = np.unique(
        video_ids[event_items[event_times < train_end]]
    )
    normal_mask = np.isin(video_ids, static.normal_item_ids)
    positive_mask = (
        (event_times < train_end)
        & event_strong
        & normal_mask[event_items]
    )
    counts = np.bincount(
        event_items[positive_mask], minlength=len(video_ids)
    )
    popularity_by_item = {
        int(video_ids[position]): float(counts[position])
        for position in np.flatnonzero(counts)
    }
    arrays = {
        "item_ids": item_ids,
        "category_ids": np.asarray(
            frame["category_ids"].tolist(), dtype=np.int64
        ),
        "video_duration": (
            frame["video_duration"]
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        ),
        "train_data_cold_mask": ~np.isin(item_ids, observed_items),
        "train_popularity_scores": np.asarray(
            [
                popularity_by_item.get(int(item), 0.0)
                for item in item_ids
            ],
            dtype=np.float32,
        ),
    }
    source_identity = {
        "reranker_report_sha256": sha256_file(reranker_report_path),
        "reranker_code_commit": report["artifacts"][
            "code_commit_at_run"
        ],
        "training_two_tower_checkpoint_sha256": report["artifacts"][
            "two_tower_checkpoint_sha256"
        ],
        "training_bpr_checkpoint_sha256": report["artifacts"][
            "bpr_checkpoint_sha256"
        ],
        "inference_fit_context": serving_metadata["fit_context"],
        "processed_manifest_sha256": sha256_file(
            artifact_dir / "manifest.json"
        ),
        "small_matrix_accessed": False,
        "temporal_final_accessed": False,
    }
    return write_reranker_feature_bundle(
        feature_path=feature_path,
        metadata_path=metadata_path,
        arrays=arrays,
        model_path=reranker_model_path,
        serving_bundle_sha256=serving_metadata["bundle_sha256"],
        source_identity=source_identity,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--serving-bundle", type=Path, required=True)
    parser.add_argument("--serving-metadata", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument(
        "--reranker-report",
        type=Path,
        default=Path(
            "reports/phase_b5b/local_reranker_validation.json"
        ),
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=Path("artifacts/serving/reranker_features_v1.npz"),
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("artifacts/serving/reranker_features_v1.json"),
    )
    args = parser.parse_args()
    metadata = export(
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        serving_bundle_path=args.serving_bundle.resolve(),
        serving_metadata_path=args.serving_metadata.resolve(),
        reranker_model_path=args.reranker_model.resolve(),
        reranker_report_path=args.reranker_report.resolve(),
        feature_path=args.output_features.resolve(),
        metadata_path=args.output_metadata.resolve(),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
