from pathlib import Path

import yaml

import numpy as np

from scripts.audit_phase0 import (
    label_bucket_counts,
    load_upload_availability_epochs,
    scan_big_splits,
    small_matrix_observation_coverage,
    timestamp_quantile_boundaries,
)


def test_strict_positive_boundary_is_not_positive():
    import pandas as pd

    counts = label_bucket_counts(pd.Series([1.99, 2.0, 2.0001]))
    assert counts["1 <= watch_ratio <= 2"] == 2
    assert counts["watch_ratio > 2"] == 1


def test_timestamp_boundaries_are_global_and_keep_boundary_groups_atomic():
    timestamps = np.arange(10, dtype=float)
    boundaries = timestamp_quantile_boundaries(
        timestamps, train_fraction=0.7, validation_fraction=0.15
    )
    assert boundaries["train_end_exclusive"] == 7.0
    assert boundaries["validation_end_exclusive"] == 8.0
    assert (timestamps < 7.0).sum() == 7
    assert ((timestamps >= 7.0) & (timestamps < 8.0)).sum() == 1
    assert (timestamps >= 8.0).sum() == 2


def test_contracts_lock_final_and_group_equal_timestamps():
    temporal = yaml.safe_load(Path("contracts/temporal_evaluation_v1.yaml").read_text())
    fully_observed = yaml.safe_load(
        Path("contracts/fully_observed_audit_v1.yaml").read_text()
    )
    cold_start = yaml.safe_load(
        Path("contracts/two_tower_cold_start_v1.yaml").read_text()
    )
    negative = yaml.safe_load(
        Path("contracts/negative_sampling_v1.yaml").read_text()
    )
    assert temporal["query"]["unit"] == "user_next_strong_positive_timestamp_group"
    assert temporal["candidate_catalog"]["target_must_be_eligible"] is True
    assert temporal["candidate_catalog"]["available_rule"] == (
        "upload_date_plus_one_local_midnight <= query_target_timestamp"
    )
    assert "previously seen" in temporal["candidate_catalog"][
        "ineligible_target_policy"
    ]
    assert temporal["leakage_guards"]["raw_date_field_is_not_a_split_source"] is True
    assert temporal["final_holdout"]["locked_by_default"] is True
    assert fully_observed["ground_truth"]["enters_user_history"] is False
    assert fully_observed["candidate_catalog"]["ranking_scope"] == (
        "all_3327_videos_for_every_user"
    )
    assert fully_observed["lock"]["locked_by_default"] is True
    assert cold_start["cold_item_path"]["independent_untrained_id_embedding"] == "forbidden"
    assert cold_start["cold_item_path"]["fallback"] == "content_only_video_tower"
    assert negative["popular_negatives"]["exclude"] == (
        "current_positive_target_set"
    )
    assert "excluding_current_positive_target_set" in negative[
        "candidate_hard_negative_distribution"
    ]["pool"]


def test_date_only_upload_becomes_available_at_next_local_midnight(tmp_path):
    import pandas as pd

    path = tmp_path / "item_daily_features.csv"
    pd.DataFrame(
        [[10, "2020-07-05"], [10, "2020-07-05"]],
        columns=["video_id", "upload_dt"],
    ).to_csv(path, index=False)

    uploaded, available = load_upload_availability_epochs(path)

    assert available[10] - uploaded[10] == 24 * 60 * 60


def test_temporal_targets_are_uploaded_unseen_and_deduplicated(tmp_path):
    import pandas as pd

    path = tmp_path / "big_matrix.csv"
    pd.DataFrame(
        [
            [0, 1, 10.0, 0.5],
            [0, 1, 10.0, 3.0],
            [0, 1, 10.0, 4.0],
            [0, 2, 10.0, 3.0],
            [0, 1, 12.0, 3.0],
            [0, 3, 13.0, 3.0],
            [0, 4, 14.0, 3.0],
            [0, 5, 15.0, 3.0],
        ],
        columns=["user_id", "video_id", "timestamp", "watch_ratio"],
    ).to_csv(path, index=False)

    stats = scan_big_splits(
        path,
        {"train_end_exclusive": 12.0, "validation_end_exclusive": 14.0},
        {1: 0.0, 2: 0.0, 3: 14.0, 5: 14.0},
        {1: 1.0, 2: 1.0, 3: 15.0, 5: 16.0},
    )

    assert stats["train"]["positive_count"] == 3
    assert stats["train"]["eligible_positive_count"] == 3
    assert stats["train"]["unique_eligible_target_count"] == 2
    assert stats["train"]["temporal_query_count"] == 1
    assert stats["train"]["multi_target_query_count"] == 1
    assert stats["validation"]["positive_previously_seen_count"] == 1
    assert stats["validation"]["positive_before_declared_upload_date_count"] == 1
    assert stats["validation"]["temporal_query_count"] == 0
    assert stats["temporal_final"]["positive_missing_upload_count"] == 1
    assert (
        stats["temporal_final"][
            "positive_same_day_upload_time_unverifiable_count"
        ]
        == 1
    )


def test_small_matrix_reports_unobserved_pairs(tmp_path):
    import pandas as pd

    path = tmp_path / "small_matrix.csv"
    pd.DataFrame(
        [[0, 10], [0, 11], [0, 11], [1, 10]],
        columns=["user_id", "video_id"],
    ).to_csv(path, index=False)

    coverage = small_matrix_observation_coverage(path)

    assert coverage["expected_complete_pairs"] == 4
    assert coverage["observed_unique_pairs"] == 3
    assert coverage["duplicate_rows"] == 1
    assert coverage["missing_pairs"] == 1
