from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_b1a_bpr_pilot import run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_b1a_synthetic_e2e_uses_only_train_validation(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    split_sha = _sha(repo_root / "manifests/split_manifest.json")
    artifact_dir = tmp_path / split_sha
    artifact_dir.mkdir()
    users = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    items = np.asarray([0, 2, 1, 1, 3, 0, 2, 0, 3, 3, 1, 2])
    timestamps = np.asarray([1.0, 2.0, 11.0] * 4)
    strong = np.asarray([True, False, True] * 4)
    np.savez_compressed(
        artifact_dir / "events_train_validation.npz",
        user_ids=np.arange(4),
        user=users,
        item=items,
        timestamp=timestamps,
        strong=strong,
        user_indptr=np.asarray([0, 3, 6, 9, 12]),
    )
    np.savez_compressed(
        artifact_dir / "catalog.npz",
        video_ids=np.asarray([10, 20, 30, 40]),
        train_end=np.asarray([10.0]),
        validation_end=np.asarray([20.0]),
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "video_id": [10, 20, 30, 40],
            "video_type": ["NORMAL"] * 4,
        }
    ).to_csv(data_dir / "item_daily_features.csv", index=False)
    manifest = {
        "artifact_scope": "train_and_validation_only",
        "fingerprint": {
            "split_manifest_sha256": split_sha,
            "source_file_sha256": {
                "item_daily_features.csv": _sha(
                    data_dir / "item_daily_features.csv"
                )
            },
        },
        "statistics": {
            "small_matrix_rows_read": 0,
            "temporal_final_rows_persisted": 0,
        },
        "files": {
            name: _sha(artifact_dir / name)
            for name in ("events_train_validation.npz", "catalog.npz")
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))

    report = run(
        repo_root,
        config_path=repo_root / "configs/phase_b1a_bpr_pilot.yaml",
        processed_artifact_dir=artifact_dir,
        data_dir=data_dir,
        checkpoint_dir=tmp_path / "checkpoints",
        report_json=tmp_path / "report.json",
        report_markdown=tmp_path / "report.md",
    )

    assert report["claim_boundary"]["small_matrix_accessed"] is False
    assert report["claim_boundary"]["temporal_final_accessed"] is False
    assert report["claim_boundary"]["two_tower_run"] is False
    assert report["checkpoints"][0]["epoch"] == 5
    assert report["initialization_audit"]["pair_count"] == 4
    trace = report["artifacts"]["item_daily_features_traceability"]
    assert trace["sha256_match"] is True
    assert trace["normal_membership_count"] == 4
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()

    daily_path = data_dir / "item_daily_features.csv"
    daily_path.write_text(daily_path.read_text().replace("NORMAL", "AD", 1))
    with pytest.raises(
        RuntimeError,
        match=r"item_daily_features\.csv SHA256 mismatch: .*actual=.*expected=",
    ):
        run(
            repo_root,
            config_path=repo_root / "configs/phase_b1a_bpr_pilot.yaml",
            processed_artifact_dir=artifact_dir,
            data_dir=data_dir,
            checkpoint_dir=tmp_path / "rejected-checkpoints",
            report_json=tmp_path / "rejected-report.json",
            report_markdown=tmp_path / "rejected-report.md",
        )
    assert not (tmp_path / "rejected-checkpoints").exists()
    assert not (tmp_path / "rejected-report.json").exists()
