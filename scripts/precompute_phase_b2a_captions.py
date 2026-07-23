#!/usr/bin/env python3
"""Precompute the required immutable caption encoder over the full item universe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from kuairec_fully_observed.caption_embeddings import (
    CAPTION_MODEL_ID,
    build_caption_cache,
    cleaned_text_sha256,
    load_caption_cache,
    load_sentence_transformer,
    validate_pinned_revision,
)
from kuairec_fully_observed.features import load_static_item_features
from kuairec_fully_observed.provenance import (
    PHASE1_PROCESSED_MANIFEST_SHA256,
    membership_record,
    normal_membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)


def _verify_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_scope") != "train_and_validation_only":
        raise RuntimeError("Caption cache requires train/validation-only artifacts")
    statistics = manifest.get("statistics", {})
    if statistics.get("small_matrix_rows_read") != 0:
        raise RuntimeError("Processed artifacts accessed Small Matrix")
    if statistics.get("temporal_final_rows_persisted") != 0:
        raise RuntimeError("Processed artifacts contain temporal final")
    for name in ("events_train_validation.npz", "catalog.npz"):
        path = artifact_dir / name
        if sha256_file(path) != manifest.get("files", {}).get(name):
            raise RuntimeError(f"Processed artifact SHA mismatch: {name}")
    return manifest


def model_item_sets(
    artifact_dir: Path, normal_item_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(artifact_dir / "events_train_validation.npz") as events, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog:
        event_items = events["item"].astype(np.int64, copy=False)
        event_times = events["timestamp"].astype(np.float64, copy=False)
        video_ids = catalog["video_ids"].astype(np.int64, copy=False)
        train_end = float(catalog["train_end"][0])
        train_history = np.unique(video_ids[event_items[event_times < train_end]])
        observed = np.unique(video_ids[event_items])
    fixed_retrieval = np.intersect1d(
        observed, np.asarray(normal_item_ids, dtype=np.int64), assume_unique=True
    )
    universe = np.union1d(train_history, fixed_retrieval).astype(np.int64)
    return fixed_retrieval.astype(np.int64), universe


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    cache_path: Path,
    metadata_path: Path,
) -> dict:
    config = yaml.safe_load(
        (repo_root / "configs/phase_b2a_two_tower_smoke.yaml").read_text()
    )
    caption_config = config["caption"]
    if caption_config["model_id"] != CAPTION_MODEL_ID:
        raise RuntimeError("Caption model ID is not frozen")
    revision = validate_pinned_revision(caption_config["resolved_revision"])
    manifest, raw_inputs = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    _verify_manifest(artifact_dir)
    started = time.perf_counter()
    all_static = load_static_item_features(data_dir)
    normal_identity = normal_membership_record(
        np.unique(all_static.normal_item_ids)
    )
    fixed_catalog, item_ids = model_item_sets(
        artifact_dir, all_static.normal_item_ids
    )
    fixed_identity = membership_record(
        fixed_catalog, label="phase-b2a-fixed-retrieval-catalog-v1"
    )
    universe_identity = membership_record(
        item_ids, label="phase-b2a-model-item-universe-v1"
    )
    frame = all_static.frame.set_index("video_id").reindex(item_ids)
    if frame["caption_text"].isna().any():
        raise RuntimeError("Model item universe is missing caption metadata rows")
    cleaned_texts = frame["caption_text"].astype(str).tolist()
    caption_trace = raw_inputs["kuairec_caption_category.csv"]
    text_sha = cleaned_text_sha256(item_ids, cleaned_texts)
    if cache_path.is_file() and metadata_path.is_file():
        cache = load_caption_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            expected_item_ids=item_ids,
            expected_model_id=CAPTION_MODEL_ID,
            expected_revision=revision,
            expected_source_sha256=caption_trace["expected_sha256"],
            expected_cleaned_text_sha256=text_sha,
        )
        cache_reused = True
    else:
        encoder = load_sentence_transformer(CAPTION_MODEL_ID, revision)
        cache = build_caption_cache(
            item_ids=item_ids,
            cleaned_texts=cleaned_texts,
            encoder=encoder,
            cache_path=cache_path,
            metadata_path=metadata_path,
            model_id=CAPTION_MODEL_ID,
            resolved_revision=revision,
            source_actual_sha256=caption_trace["actual_sha256"],
            source_expected_sha256=caption_trace["expected_sha256"],
            versions={
                "torch": torch.__version__,
                "sentence_transformers": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "cuda": torch.version.cuda,
            },
            batch_size=int(caption_config["batch_size"]),
        )
        cache_reused = False
    value = dict(cache.metadata)
    value["cache_reused"] = cache_reused
    value["processed_manifest_sha256"] = PHASE1_PROCESSED_MANIFEST_SHA256
    value["raw_input_traceability"] = raw_inputs
    value["normal_membership"] = normal_identity
    value["fixed_retrieval_catalog_membership"] = fixed_identity
    value["model_item_universe_membership"] = universe_identity
    value["wall_time_s"] = round(time.perf_counter() - started, 4)
    metadata_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/phase_b2a/caption_embeddings.npz"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("reports/phase_b2a/caption_cache_metadata.json"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run(
                repo_root=repo_root,
                data_dir=args.data_dir.resolve(),
                artifact_dir=args.processed_artifact_dir.resolve(),
                cache_path=(repo_root / args.cache_path).resolve(),
                metadata_path=(repo_root / args.metadata_path).resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
