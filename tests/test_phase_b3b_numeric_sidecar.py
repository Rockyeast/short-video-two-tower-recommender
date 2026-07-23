from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kuairec_fully_observed.caption_embeddings import CaptionCache
from kuairec_fully_observed.numeric_sidecar import (
    NUMERIC_PREPROCESSING_LABEL,
    load_final_refit_numeric_sidecar,
)
from kuairec_fully_observed.provenance import canonical_json_sha256
from kuairec_fully_observed.torch_training import (
    prepare_final_refit_inference_feature_store,
    prepare_item_feature_store,
)


SIDECAR_PATH = Path("manifests/phase_b3b_final_numeric_preprocessing.json")


def _sidecar() -> dict:
    return json.loads(SIDECAR_PATH.read_text())


def _load(path: Path, payload: dict) -> dict:
    return load_final_refit_numeric_sidecar(
        path,
        checkpoint_sha256=payload["checkpoint"]["sha256"],
        checkpoint_expected_numeric_sha256=payload["checkpoint"][
            "expected_numeric_preprocessing_sha256"
        ],
        processed_manifest_sha256=payload["processed_manifest"]["sha256"],
        raw_input_sha256=payload["raw_inputs"],
        memberships=payload["memberships"],
    )


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "numeric-sidecar.json"
    path.write_text(json.dumps(payload))
    return path


def test_committed_sidecar_matches_checkpoint_numeric_identity():
    payload = _sidecar()
    loaded = _load(SIDECAR_PATH, payload)
    actual = canonical_json_sha256(
        loaded["preprocessing"], label=NUMERIC_PREPROCESSING_LABEL
    )
    assert actual == payload["checkpoint"][
        "expected_numeric_preprocessing_sha256"
    ]
    assert actual == payload["numeric_preprocessing_sha256"]


@pytest.mark.parametrize("field", ["medians", "means", "stds"])
def test_sidecar_rejects_numeric_statistic_mutation(tmp_path, field):
    payload = _sidecar()
    changed = copy.deepcopy(payload)
    changed["preprocessing"][field][0] = np.nextafter(
        changed["preprocessing"][field][0], np.inf
    )
    with pytest.raises(RuntimeError, match="preprocessing SHA mismatch"):
        _load(_write(tmp_path, changed), payload)


def test_sidecar_rejects_checkpoint_input_and_membership_mismatch(tmp_path):
    payload = _sidecar()
    path = _write(tmp_path, payload)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        load_final_refit_numeric_sidecar(
            path,
            checkpoint_sha256="0" * 64,
            checkpoint_expected_numeric_sha256=payload["checkpoint"][
                "expected_numeric_preprocessing_sha256"
            ],
            processed_manifest_sha256=payload["processed_manifest"]["sha256"],
            raw_input_sha256=payload["raw_inputs"],
            memberships=payload["memberships"],
        )
    wrong_inputs = dict(payload["raw_inputs"])
    wrong_inputs["big_matrix.csv"] = "1" * 64
    with pytest.raises(RuntimeError, match="raw input identity"):
        load_final_refit_numeric_sidecar(
            path,
            checkpoint_sha256=payload["checkpoint"]["sha256"],
            checkpoint_expected_numeric_sha256=payload["checkpoint"][
                "expected_numeric_preprocessing_sha256"
            ],
            processed_manifest_sha256=payload["processed_manifest"]["sha256"],
            raw_input_sha256=wrong_inputs,
            memberships=payload["memberships"],
        )
    wrong_memberships = copy.deepcopy(payload["memberships"])
    wrong_memberships["train_observed_items"]["count"] += 1
    with pytest.raises(RuntimeError, match="membership identity"):
        load_final_refit_numeric_sidecar(
            path,
            checkpoint_sha256=payload["checkpoint"]["sha256"],
            checkpoint_expected_numeric_sha256=payload["checkpoint"][
                "expected_numeric_preprocessing_sha256"
            ],
            processed_manifest_sha256=payload["processed_manifest"]["sha256"],
            raw_input_sha256=payload["raw_inputs"],
            memberships=wrong_memberships,
        )


def test_final_refit_store_uses_frozen_stats_without_refitting():
    item_ids = np.asarray([10, 20, 30], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "video_id": item_ids,
            "caption_text": ["a", "b", "c"],
            "category_ids": [(1, 2, 3), (1, 4, 5), (1, 6, 7)],
            "video_duration": [9.0, 19.0, 29.0],
            "video_width": [99.0, 199.0, 299.0],
            "video_height": [49.0, 59.0, 69.0],
            "upload_type": ["A", "B", "A"],
            "upload_dt": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    caption = CaptionCache(
        item_ids,
        np.ones((3, 384), dtype=np.float32),
        {},
    )
    fitted = prepare_item_feature_store(
        static_frame=frame,
        caption_cache=caption,
        item_universe=item_ids,
        train_observed_item_ids=item_ids,
        train_observed_normal_item_ids=item_ids,
    )
    frozen = copy.deepcopy(fitted.preprocessing)
    frozen["means"][0] += 0.25
    restored = prepare_final_refit_inference_feature_store(
        static_frame=frame,
        caption_cache=caption,
        item_universe=item_ids,
        train_observed_item_ids=item_ids,
        train_observed_normal_item_ids=item_ids,
        frozen_preprocessing=frozen,
    )
    assert restored.preprocessing == frozen
    assert not np.array_equal(
        restored.numeric_features, fitted.numeric_features
    )


def test_artifact_only_runner_contains_no_small_input_path():
    source = Path("scripts/run_phase_b3b_artifact_preflight.py").read_text()
    assert "small_matrix.csv" not in source
    assert "small_matrix_path" not in source
