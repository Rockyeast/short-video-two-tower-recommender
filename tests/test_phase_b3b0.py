from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.precompute_phase_b3b_captions import refit_item_universe
from kuairec_fully_observed import (
    require_sealed_execution,
    verify_file_identity,
    verify_final_refit_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_refit_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    artifacts = {
        "popularity": tmp_path / "global_popularity.json",
        "bpr": tmp_path / "bpr_epoch_020.npz",
        "two_tower": tmp_path / "two_tower_epoch_001.pt",
    }
    for name, path in artifacts.items():
        path.write_bytes(f"synthetic-{name}".encode())
    report = {
        "remote": {
            "refit": {
                "fit_context": "canonical_big_train_plus_validation",
                "recipe_frozen_before_small": True,
                "selection_performed": False,
                "refit": {
                    "global_popularity": {
                        "artifact_sha256": _sha256(
                            artifacts["popularity"]
                        )
                    },
                    "bpr": {
                        "epochs": 20,
                        "checkpoint_sha256": _sha256(artifacts["bpr"]),
                    },
                    "two_tower": {
                        "epochs": 1,
                        "checkpoint_sha256": _sha256(
                            artifacts["two_tower"]
                        ),
                    },
                },
            }
        }
    }
    report_path = tmp_path / "final_refit.json"
    report_path.write_text(json.dumps(report))
    return report_path, artifacts


def test_final_recipe_is_locked_before_small():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/phase_b3b_final_recipe.yaml").read_text()
    )
    assert config["fit_context"] == "canonical_big_train_plus_validation"
    assert config["selection_complete"] is True
    assert config["bpr"]["epochs"] == 20
    assert config["two_tower"]["training"]["epochs"] == 1
    assert config["hybrid"] == {
        "routes": ["two_tower", "bpr"],
        "route_top_k": 500,
        "rank_constant": 60,
        "alpha": 0.75,
        "output_k": 100,
    }


def test_refit_item_universe_uses_fit_history_plus_all_normal(tmp_path):
    events = tmp_path / "events_train_validation.npz"
    catalog = tmp_path / "catalog.npz"
    np.savez(
        events,
        item=np.asarray([0, 1, 2, 3]),
        timestamp=np.asarray([1.0, 2.0, 3.0, 9.0]),
    )
    np.savez(
        catalog,
        video_ids=np.asarray([10, 11, 12, 13]),
        validation_end=np.asarray([5.0]),
    )
    actual = refit_item_universe(
        artifact_dir=tmp_path, normal_item_ids=np.asarray([11, 13, 20])
    )
    np.testing.assert_array_equal(actual, [10, 11, 12, 13, 20])


def test_sealed_runner_refuses_before_reading_any_input(tmp_path):
    with pytest.raises(RuntimeError, match="requires --execute-sealed-small"):
        require_sealed_execution(False)
    assert list(tmp_path.iterdir()) == []


def test_synthetic_small_wrong_sha_is_rejected(tmp_path):
    small_path = tmp_path / "small_matrix.csv"
    small_path.write_bytes(b"synthetic Small fixture only")
    with pytest.raises(RuntimeError, match="small_matrix.csv SHA256 mismatch"):
        verify_file_identity(
            small_path,
            expected_size_bytes=small_path.stat().st_size,
            expected_sha256="0" * 64,
            label="small_matrix.csv",
        )


@pytest.mark.parametrize(
    ("artifact_name", "error_name"),
    [
        ("popularity", "global_popularity"),
        ("bpr", "bpr_epoch_20"),
        ("two_tower", "two_tower_epoch_1"),
    ],
)
def test_wrong_final_refit_artifact_is_rejected(
    tmp_path, artifact_name, error_name
):
    report_path, artifacts = _write_refit_fixture(tmp_path)
    artifacts[artifact_name].write_bytes(b"wrong artifact")
    with pytest.raises(RuntimeError, match=error_name):
        verify_final_refit_artifacts(
            final_refit_report_path=report_path,
            popularity_path=artifacts["popularity"],
            bpr_checkpoint_path=artifacts["bpr"],
            two_tower_checkpoint_path=artifacts["two_tower"],
        )


def test_correct_final_refit_artifacts_are_accepted(tmp_path):
    report_path, artifacts = _write_refit_fixture(tmp_path)
    actual = verify_final_refit_artifacts(
        final_refit_report_path=report_path,
        popularity_path=artifacts["popularity"],
        bpr_checkpoint_path=artifacts["bpr"],
        two_tower_checkpoint_path=artifacts["two_tower"],
    )
    assert actual["fit_context"] == "canonical_big_train_plus_validation"
    assert actual["recipe_frozen_before_small"] is True
    assert actual["selection_performed"] is False
    assert actual["bpr_epochs"] == 20
    assert actual["two_tower_epochs"] == 1
    assert all(
        artifact["match"] for artifact in actual["artifacts"].values()
    )
