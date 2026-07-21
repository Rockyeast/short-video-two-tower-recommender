from pathlib import Path

import yaml

import numpy as np

from scripts.audit_phase0 import (
    label_bucket_counts,
    load_candidate_catalog_policy,
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
    temporal = yaml.safe_load(Path("contracts/temporal_evaluation_v2.yaml").read_text())
    fully_observed = yaml.safe_load(
        Path("contracts/fully_observed_audit_v2.yaml").read_text()
    )
    cold_start = yaml.safe_load(
        Path("contracts/two_tower_cold_start_v2.yaml").read_text()
    )
    negative = yaml.safe_load(
        Path("contracts/negative_sampling_v2.yaml").read_text()
    )
    assert temporal["query"]["unit"] == "user_next_strong_positive_timestamp_group"
    assert temporal["candidate_catalog"]["target_must_be_eligible"] is True
    assert temporal["candidate_catalog"]["contract"] == (
        "contracts/candidate_catalog_v1.yaml"
    )
    assert temporal["candidate_catalog"]["filter_seen_items"] is True
    assert temporal["leakage_guards"]["raw_date_field_is_not_a_split_source"] is True
    assert temporal["temporal_final"]["ordinary_baseline_entrypoint_access"] == (
        "forbidden"
    )
    assert "user_history" in fully_observed["blocked_pairs"]["forbidden_use"]
    assert fully_observed["primary_evaluation"]["candidate_set"] == "catalog minus B_u"
    assert fully_observed["primary_evaluation"]["equivalent_candidate_set"] == "O_u"
    assert fully_observed["secondary_safety_audit"][
        "model_quality_claim_from_secondary_audit"
    ] == "forbidden"
    assert fully_observed["lock"]["locked_by_default"] is True
    assert cold_start["cold_or_untouched_item_path"][
        "independent_untrained_id_embedding"
    ] == "forbidden"
    assert cold_start["cold_or_untouched_item_path"]["fallback"] == (
        "content_only_video_tower"
    )
    assert negative["popular_negatives"]["exclude"] == "current_positive_target_set"
    assert "excluding current targets" in negative[
        "candidate_hard_negative_distribution"
    ]["pool"]


def test_causal_catalog_uses_prior_day_visibility_and_excludes_ads(tmp_path):
    import pandas as pd

    path = tmp_path / "item_daily_features.csv"
    pd.DataFrame(
        [
            [1, 20200705, "2020-07-04", "NORMAL", "public"],
            [1, 20200707, "2020-07-04", "NORMAL", "private"],
            [2, 20200705, "2020-07-04", "NORMAL", "private"],
            [2, 20200707, "2020-07-04", "NORMAL", "public"],
            [3, 20200705, "2020-07-04", "AD", "public"],
            [4, 20200705, "2020-07-07", "NORMAL", "public"],
        ],
        columns=["video_id", "date", "upload_dt", "video_type", "visible_status"],
    ).to_csv(path, index=False)
    start = pd.Timestamp("2020-07-05 12:00", tz="Asia/Shanghai").timestamp()
    end = pd.Timestamp("2020-07-08 12:00", tz="Asia/Shanghai").timestamp()

    policy = load_candidate_catalog_policy(path, start, end)
    daily = policy["audit"]["daily_primary_catalog_size"]

    assert daily["20200705"] == 0  # no causally prior snapshot
    assert daily["20200706"] == 1  # video 1 only
    assert daily["20200707"] == 1  # same-day status update is not visible yet
    assert daily["20200708"] == 2  # video 2 becomes public; video 4 is uploaded
    assert policy["audit"]["video_type_video_counts"] == {"NORMAL": 3, "AD": 1}


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
            [0, 1, 1000, 2000, "t0", 19700101, 10.0, 3.0],
            [0, 1, 1000, 2000, "t0", 19700101, 10.0, 3.0],
            [0, 2, 1000, 2000, "t0", 19700101, 10.0, 3.0],
            [0, 1, 1000, 2000, "t1", 19700101, 12.0, 3.0],
            [0, 3, 1000, 2000, "t2", 19700101, 13.0, 3.0],
            [0, 4, 1000, 2000, "t3", 19700101, 14.0, 3.0],
            [0, 5, 1000, 2000, "t4", 19700101, 15.0, 3.0],
        ],
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
    assert (
        stats["train"]["target_deduplication_audit"]
        ["exact_duplicate_positive_rows_removed"]
        == 1
    )
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
