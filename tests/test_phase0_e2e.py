from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

import scripts.audit_phase0 as audit_phase0


ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_kuairec(root: Path) -> tuple[Path, Path]:
    data_root = root / "data"
    extracted = data_root / "synthetic"
    extracted.mkdir(parents=True)
    archive = data_root / "KuaiRec.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("README.txt", "synthetic protocol fixture only\n")

    base = pd.Timestamp("2020-07-07 12:00:00", tz="Asia/Shanghai").timestamp()
    big_rows = []
    for offset in range(10):
        timestamp = base + offset * 86400
        local = pd.Timestamp(timestamp, unit="s", tz="UTC").tz_convert(
            "Asia/Shanghai"
        )
        big_rows.append(
            [
                0,
                offset + 1,
                4000,
                1000,
                local.strftime("%Y-%m-%d %H:%M:%S"),
                int(local.strftime("%Y%m%d")),
                timestamp,
                4.0,
            ]
        )
    pd.DataFrame(
        big_rows,
        columns=[
            "user_id",
            "video_id",
            "play_duration",
            "video_duration",
            "time",
            "date",
            "timestamp",
            "watch_ratio",
        ],
    ).to_csv(extracted / "big_matrix.csv", index=False)

    small_rows = []
    for user_id in (0, 1):
        for video_id in range(1, 6):
            if (user_id, video_id) == (0, 5):
                continue
            timestamp = base + video_id
            small_rows.append(
                [
                    user_id,
                    video_id,
                    3000,
                    1000,
                    "2020-07-07 12:00:00",
                    20200707,
                    timestamp,
                    3.0 if video_id % 2 else 1.0,
                ]
            )
    pd.DataFrame(
        small_rows,
        columns=[
            "user_id",
            "video_id",
            "play_duration",
            "video_duration",
            "time",
            "date",
            "timestamp",
            "watch_ratio",
        ],
    ).to_csv(extracted / "small_matrix.csv", index=False)

    pd.DataFrame(
        {"video_id": range(1, 11), "feat": ["[1]"] * 10}
    ).to_csv(extracted / "item_categories.csv", index=False)
    pd.DataFrame(
        {
            "video_id": range(1, 11),
            "caption": [f"video {value}" for value in range(1, 11)],
            "manual_cover_text": ["cover"] * 10,
            "topic_tag": ["topic"] * 10,
            "first_level_category_id": [1] * 10,
            "second_level_category_id": [2] * 10,
            "third_level_category_id": [3] * 10,
        }
    ).to_csv(extracted / "kuairec_caption_category.csv", index=False)

    daily_rows = []
    for date in pd.date_range("2020-07-05", "2020-07-17", freq="D"):
        for video_id in range(1, 11):
            daily_rows.append(
                [
                    video_id,
                    int(date.strftime("%Y%m%d")),
                    video_id,
                    "AD" if video_id == 5 else "NORMAL",
                    "2020-07-04",
                    "synthetic",
                    "public",
                    1000,
                ]
            )
    pd.DataFrame(
        daily_rows,
        columns=[
            "video_id",
            "date",
            "author_id",
            "video_type",
            "upload_dt",
            "upload_type",
            "visible_status",
            "video_duration",
        ],
    ).to_csv(extracted / "item_daily_features.csv", index=False)
    pd.DataFrame({"user_id": [0, 1], "feature": [1, 2]}).to_csv(
        extracted / "user_features.csv", index=False
    )
    pd.DataFrame({"user_id": [0, 1], "friend_id": [1, 0]}).to_csv(
        extracted / "social_network.csv", index=False
    )

    config = yaml.safe_load((ROOT / "configs/phase0.yaml").read_text())
    config["dataset"]["archive_md5"] = hashlib.md5(
        archive.read_bytes(), usedforsecurity=False
    ).hexdigest()
    config["split"].update(
        {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "temporal_final_fraction": 0.2,
        }
    )
    config_path = root / "phase0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return data_root, config_path


def test_synthetic_generate_then_temp_recompute_verify(tmp_path):
    data_root, config = _write_synthetic_kuairec(tmp_path)
    report_dir = tmp_path / "reports" / "phase0"
    manifest = tmp_path / "manifests" / "split_manifest.json"
    generate = subprocess.run(
        [
            sys.executable,
            "scripts/audit_phase0.py",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--report-dir",
            str(report_dir),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generate.returncode == 0, generate.stderr

    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == 2
    assert payload["protocol_revision"] == "protocol-v2.1.1"
    assert payload["split_algorithm"]["fraction_validation"] == {
        "keys": [
            "train_fraction",
            "validation_fraction",
            "temporal_final_fraction",
        ],
        "each_fraction_strictly_between_zero_and_one": True,
        "required_sum": 1.0,
        "absolute_tolerance": 1.0e-12,
    }
    for split in ("train", "validation", "temporal_final"):
        assert len(payload["splits"][split]["canonical_target_sha256"]) == 64
        assert len(payload["splits"][split]["candidate_membership_sha256"]) == 64
    rebuilt = audit_phase0.recompute_protocol_derived_hashes(
        config_path=config,
        data_root=data_root,
        manifest=payload,
    )
    assert rebuilt == {
        "canonical_targets": {
            split: payload["splits"][split]["canonical_target_sha256"]
            for split in ("train", "validation", "temporal_final")
        },
        "candidate_membership": {
            split: payload["splits"][split]["candidate_membership_sha256"]
            for split in ("train", "validation", "temporal_final")
        },
    }
    assert payload["locks"]["ordinary_baseline_scripts_may_run_final"] is False
    audit = json.loads((report_dir / "audit.json").read_text())
    assert audit["model_or_baseline_executed"] is False
    assert audit["final_ranking_evaluation_executed"] is False
    small = audit["small_matrix_observation_coverage"]
    assert small["video_type_breakdown"]["NORMAL"]["observed_pairs"] == 8
    assert small["video_type_breakdown"]["AD"]["observed_pairs"] == 1
    assert small["normal_plus_ad_observed_pairs"] == small[
        "observed_unique_pairs"
    ]
    assert small["primary_candidate_size_per_user"]["quantiles"]["p50"] == 4
    assert small["secondary_full_catalog_size"] == 5
    assert payload["small_matrix_audit"]["primary_observed_normal_pairs"] == 8
    assert payload["small_matrix_audit"]["video_type_breakdown"]["AD"][
        "observed_pairs"
    ] == 1
    assert payload["small_matrix_audit"]["time_decayed_popularity"] == {
        "static_score_timestamp": "validation_end_exclusive",
        "state_source": "frozen_train_plus_validation",
        "small_matrix_replay_or_update": False,
    }
    assert payload["cold_start_contexts"]["small_matrix"][
        "target_definition"
    ].startswith("observed NORMAL")
    assert payload["cold_start_contexts"]["small_matrix"]["target_item_count"] == 2
    assert payload["cold_start_contexts"]["small_matrix_ad_diagnostic"][
        "target_definition"
    ].startswith("observed AD")
    assert payload["cold_start_contexts"]["small_matrix_ad_diagnostic"][
        "target_item_count"
    ] == 1
    assert audit["baseline_cost_estimates"][
        "small_matrix_primary_observed_pairs"
    ] == 8
    assert audit["baseline_cost_estimates"][
        "small_matrix_secondary_full_ranking_pairs"
    ] == 10

    verify = subprocess.run(
        [
            sys.executable,
            "scripts/audit_phase0.py",
            "--mode",
            "verify",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--reference-report-dir",
            str(report_dir),
            "--reference-manifest",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
    assert "protocol-v2.1.1 bundle" in verify.stdout
    assert "match byte-for-byte" in verify.stdout
