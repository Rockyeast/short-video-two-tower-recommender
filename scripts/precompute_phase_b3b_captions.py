#!/usr/bin/env python3
"""Precompute frozen caption vectors for the refit context and all NORMAL items."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
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
    membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)


def refit_item_universe(
    *, artifact_dir: Path, normal_item_ids: np.ndarray
) -> np.ndarray:
    with np.load(artifact_dir / "events_train_validation.npz") as events, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog:
        video_ids = catalog["video_ids"].astype(np.int64, copy=False)
        event_items = events["item"].astype(np.int64, copy=False)
        event_times = events["timestamp"].astype(np.float64, copy=False)
        validation_end = float(catalog["validation_end"][0])
        fit_history = np.unique(
            video_ids[event_items[event_times < validation_end]]
        )
    return np.union1d(
        fit_history, np.asarray(normal_item_ids, dtype=np.int64)
    ).astype(np.int64)


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    cache_path: Path,
    metadata_path: Path,
) -> dict:
    config = yaml.safe_load(
        (repo_root / "configs/phase_b3b_final_recipe.yaml").read_text()
    )
    caption_config = config["two_tower"]["caption"]
    if caption_config["model_id"] != CAPTION_MODEL_ID:
        raise RuntimeError("Final recipe caption model ID changed")
    revision = validate_pinned_revision(caption_config["resolved_revision"])
    _, raw_inputs = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    static = load_static_item_features(data_dir)
    item_ids = refit_item_universe(
        artifact_dir=artifact_dir,
        normal_item_ids=static.normal_item_ids,
    )
    frame = static.frame.set_index("video_id").reindex(item_ids)
    if frame["caption_text"].isna().any():
        raise RuntimeError("Refit item universe is missing caption metadata")
    texts = frame["caption_text"].astype(str).tolist()
    source = raw_inputs["kuairec_caption_category.csv"]
    text_sha = cleaned_text_sha256(item_ids, texts)
    if cache_path.is_file() and metadata_path.is_file():
        cache = load_caption_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            expected_item_ids=item_ids,
            expected_model_id=CAPTION_MODEL_ID,
            expected_revision=revision,
            expected_source_sha256=source["expected_sha256"],
            expected_cleaned_text_sha256=text_sha,
        )
        reused = True
    else:
        import torch

        encoder = load_sentence_transformer(CAPTION_MODEL_ID, revision)
        cache = build_caption_cache(
            item_ids=item_ids,
            cleaned_texts=texts,
            encoder=encoder,
            cache_path=cache_path,
            metadata_path=metadata_path,
            model_id=CAPTION_MODEL_ID,
            resolved_revision=revision,
            source_actual_sha256=source["actual_sha256"],
            source_expected_sha256=source["expected_sha256"],
            versions={
                "torch": torch.__version__,
                "sentence_transformers": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "cuda": torch.version.cuda,
            },
            batch_size=256,
        )
        reused = False
    report = dict(cache.metadata)
    report.update(
        {
            "cache_reused": reused,
            "purpose": "big-train-validation-refit-plus-all-normal-content-fallback",
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "item_universe": membership_record(
                item_ids, label="phase-b3b-refit-item-universe-v1"
            ),
            "normal_items": membership_record(
                np.unique(static.normal_item_ids),
                label="phase-b3b-normal-items-v1",
            ),
            "cache_file_sha256": sha256_file(cache_path),
        }
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/phase_b3b/caption_embeddings.npz"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("reports/phase_b3b0/caption_cache_metadata.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(
        repo_root=root,
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        cache_path=(root / args.cache_path).resolve(),
        metadata_path=(root / args.metadata_path).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
