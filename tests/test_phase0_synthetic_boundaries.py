from __future__ import annotations

import pandas as pd
import pytest

import scripts.audit_phase0 as audit


def _assert_split_stats_equal(left, right):
    for split_name in left:
        pd.testing.assert_frame_equal(
            left[split_name]["_canonical_targets"].reset_index(drop=True),
            right[split_name]["_canonical_targets"].reset_index(drop=True),
        )
        left_public = {
            key: value
            for key, value in left[split_name].items()
            if key != "_canonical_targets"
        }
        right_public = {
            key: value
            for key, value in right[split_name].items()
            if key != "_canonical_targets"
        }
        assert left_public == right_public


def test_chunk_boundary_does_not_change_equal_timestamp_dedup(tmp_path, monkeypatch):
    path = tmp_path / "big_matrix.csv"
    pd.DataFrame(
        [
            [0, 99, 1000, 5000, "t90", 19700101, 90.0, 0.1],
            [0, 10, 1000, 5000, "t100", 19700101, 100.0, 0.5],
            # Same key with a conflicting binary label, split across chunks.
            [0, 10, 1000, 5000, "t100", 19700101, 100.0, 3.0],
            [0, 11, 1000, 5000, "t100", 19700101, 100.0, 3.0],
            # Exact positive duplicate.
            [0, 11, 1000, 5000, "t100", 19700101, 100.0, 3.0],
            [0, 12, 1000, 5000, "t200", 19700101, 200.0, 3.0],
            [0, 13, 1000, 5000, "t300", 19700101, 300.0, 3.0],
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
    upload = {video: 0.0 for video in (10, 11, 12, 13, 99)}
    available = {video: 1.0 for video in upload}
    boundaries = {
        "train_end_exclusive": 150.0,
        "validation_end_exclusive": 250.0,
    }

    monkeypatch.setattr(audit, "CHUNK_SIZE", 250_000)
    one_chunk = audit.scan_big_splits(path, boundaries, upload, available)
    monkeypatch.setattr(audit, "CHUNK_SIZE", 2)
    many_chunks = audit.scan_big_splits(path, boundaries, upload, available)

    _assert_split_stats_equal(many_chunks, one_chunk)
    train = many_chunks["train"]
    assert train["positive_count"] == 3
    assert train["eligible_positive_count"] == 3
    assert train["unique_eligible_target_count"] == 1
    assert train["temporal_query_count"] == 1
    assert train["multi_target_query_count"] == 0
    assert train["targets_per_query"]["quantiles"]["p100"] == 1.0
    dedup = train["target_deduplication_audit"]
    assert dedup["exact_duplicate_positive_rows_removed"] == 1
    assert dedup["binary_label_conflict_keys_excluded"] == 1
    assert dedup["positive_rows_in_conflict_keys_excluded"] == 1
    assert dedup["reconciliation_ok"] is True


def test_hard_pool_respects_equal_time_window_cutoff_and_uniform_denominator(
    tmp_path, monkeypatch
):
    path = tmp_path / "big_matrix.csv"
    pd.DataFrame(
        [
            [0, 1, 4000, 1000, "t100", 19700101, 100.0, 3.0],
            # Equal-time quick skip: not future and therefore not hard.
            [0, 2, 100, 1000, "t100", 19700101, 100.0, 0.1],
            # Exactly 30 minutes later: same session and included.
            [0, 3, 100, 1000, "t1900", 19700101, 1900.0, 0.1],
            # Same hard item later becomes positive: diagnostic risk only.
            [0, 3, 4000, 1000, "t1999", 19700101, 1999.0, 3.0],
            # Exact train cutoff: forbidden to a selection-time pool.
            [0, 4, 100, 1000, "t2000", 19700101, 2000.0, 0.1],
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
    monkeypatch.setattr(audit, "CHUNK_SIZE", 2)
    catalog = {
        "video_index": {1: 0, 2: 1, 3: 2, 4: 3},
        "state_by_date": {
            19700101: {
                "eligible_mask": 0b1111,
            }
        },
    }
    targets = {
        "train": pd.DataFrame(
            [[0, 1, 100.0]], columns=["user_id", "video_id", "timestamp"]
        ),
        "validation": pd.DataFrame(
            columns=["user_id", "video_id", "timestamp"]
        ),
        "temporal_final": pd.DataFrame(
            columns=["user_id", "video_id", "timestamp"]
        ),
    }

    candidate, hard = audit.audit_candidate_sizes_and_hard_negatives(
        path,
        targets,
        catalog,
        {"train_end_exclusive": 2000.0, "validation_end_exclusive": 3000.0},
        session_gap_minutes=30,
        hard_window_minutes=30,
    )

    train = candidate["per_split"]["train"]
    assert train["available_unseen_candidate_size"]["quantiles"]["p50"] == 4
    assert train["uniform_pool_excluding_targets_size"]["quantiles"]["p50"] == 3
    assert hard["unique_hard_query_item_pairs"] == 1
    assert hard["pool_size_per_query"]["quantiles"]["p50"] == 1
    assert hard["false_negative_risk"]["pair_counts"]["remaining_session"] == 1

    # Membership serialization is stable across CSV chunk boundaries.
    monkeypatch.setattr(audit, "CHUNK_SIZE", 250_000)
    candidate_one_chunk, _ = audit.audit_candidate_sizes_and_hard_negatives(
        path,
        targets,
        catalog,
        {"train_end_exclusive": 2000.0, "validation_end_exclusive": 3000.0},
        session_gap_minutes=30,
        hard_window_minutes=30,
    )
    assert candidate["membership_hash_format"]["algorithm"] == "sha256"
    assert candidate["per_split"]["train"]["candidate_membership_sha256"] == (
        candidate_one_chunk["per_split"]["train"]["candidate_membership_sha256"]
    )


def test_hard_pool_excludes_cutoff_risk_and_quick_positive_conflict(tmp_path):
    path = tmp_path / "big_matrix.csv"
    pd.DataFrame(
        [
            [0, 1, 4000, 1000, "query", 19700101, 100.0, 3.0],
            # Two non-identical rows at one key: both quick-skip and strong-positive.
            [0, 2, 100, 1000, "conflict-a", 19700101, 150.0, 0.1],
            [0, 2, 4000, 1000, "conflict-b", 19700101, 150.0, 3.0],
            # A valid hard negative whose later strong positive is exactly at cutoff.
            [0, 3, 100, 1000, "hard", 19700101, 160.0, 0.1],
            [0, 3, 4000, 1000, "cutoff", 19700101, 200.0, 3.0],
        ],
        columns=audit.EVENT_COLUMNS,
    ).to_csv(path, index=False)
    targets = {
        "train": pd.DataFrame(
            [[0, 1, 100.0]], columns=["user_id", "video_id", "timestamp"]
        ),
        "validation": pd.DataFrame(
            columns=["user_id", "video_id", "timestamp"]
        ),
        "temporal_final": pd.DataFrame(
            columns=["user_id", "video_id", "timestamp"]
        ),
    }
    catalog = {
        "video_ids": [1, 2, 3],
        "video_index": {1: 0, 2: 1, 3: 2},
        "state_by_date": {19700101: {"eligible_mask": 0b111}},
    }

    candidate, hard = audit.audit_candidate_sizes_and_hard_negatives(
        path,
        targets,
        catalog,
        {"train_end_exclusive": 200.0, "validation_end_exclusive": 300.0},
        session_gap_minutes=30,
        hard_window_minutes=30,
    )

    # Item 2 is excluded due to the ambiguous key; item 3 remains hard.
    assert hard["unique_hard_query_item_pairs"] == 1
    assert hard["filter_counts"]["quick_skip_strong_positive_conflict"] == 1
    # Every diagnostic is strict at the train cutoff, including remaining-session.
    assert set(hard["false_negative_risk"]["pair_counts"].values()) == {0}
    # Items first occurring at the query timestamp remain unseen (strict < query).
    assert candidate["per_split"]["train"][
        "available_unseen_candidate_size"
    ]["quantiles"]["p50"] == 3


def test_all_eight_field_event_audit_and_frozen_boundary_sensitivity(tmp_path):
    path = tmp_path / "big_matrix.csv"
    pd.DataFrame(
        [
            [0, 1, 100, 1000, "exact", 19700101, 10.0, 0.1],
            [0, 1, 100, 1000, "exact", 19700101, 10.0, 0.1],
            # Same event key, but other source fields and both labels differ.
            [0, 2, 100, 1000, "nonexact-a", 19700101, 20.0, 0.1],
            [0, 2, 4000, 1000, "nonexact-b", 19700101, 20.0, 3.0],
            [0, 3, 4000, 1000, "third", 19700101, 30.0, 3.0],
            [0, 4, 4000, 1000, "fourth", 19700101, 40.0, 3.0],
        ],
        columns=audit.EVENT_COLUMNS,
    ).to_csv(path, index=False)

    result = audit.audit_canonical_behavior_events(
        path,
        {"train_end_exclusive": 25.0, "validation_end_exclusive": 35.0},
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    assert result["raw_rows"] == 6
    assert result["exact_duplicate_rows_removed"] == 1
    assert result["exact_duplicate_full_row_group_count"] == 1
    assert result["same_key_nonexact_rows_removed"] == 1
    assert result["same_key_nonexact_key_count"] == 1
    assert result["binary_positive_conflict_key_count"] == 1
    assert result["quick_skip_strong_positive_conflict_key_count"] == 1
    assert result["canonical_event_count"] == 4
    assert result["reconciliation_ok"] is True
    assert result["exact_duplicate_distribution"]["affected_users"] == 1
    assert result["exact_duplicate_distribution"]["by_user"] == {"0": 1}
    assert result["same_key_nonexact_distribution"]["affected_users"] == 1
    assert result["same_key_nonexact_distribution"]["by_user"] == {"0": 1}
    assert result["boundary_sensitivity"]["decision"] == (
        "preserve previously frozen raw-count cutoffs"
    )
    assert result["downstream_event_inputs"]["last_50"].startswith(
        "last 50 canonical events"
    )
    assert result["downstream_event_inputs"]["seen_filter"].startswith(
        "canonical events strictly before"
    )
    assert sum(
        result["boundary_sensitivity"][
            "frozen_boundary_canonical_event_counts"
        ].values()
    ) == result["canonical_event_count"]
    assert sum(
        result["boundary_sensitivity"][
            "counterfactual_boundary_canonical_event_counts"
        ].values()
    ) == result["canonical_event_count"]


def test_canonical_event_representative_uses_typed_numeric_ordering():
    frame = pd.DataFrame(
        [
            [0, 1, 10, 1000, "same", 19700101, 10.0, 0.1],
            [0, 1, 2, 1000, "same", 19700101, 10.0, 0.1],
        ],
        columns=audit.EVENT_COLUMNS,
    )

    canonical, summary, _, _ = audit.canonicalize_behavior_events(frame)

    assert summary["same_key_nonexact_rows_removed"] == 1
    # Numeric ordering selects 2; serialized lexical ordering would select 10.
    assert canonical.loc[0, "play_duration"] == 2


def test_split_fraction_policy_is_config_driven_and_fail_closed():
    valid = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "temporal_final_fraction": 0.2,
        "fraction_validation": {
            "keys": [
                "train_fraction",
                "validation_fraction",
                "temporal_final_fraction",
            ],
            "each_fraction_strictly_between_zero_and_one": True,
            "required_sum": 1.0,
            "absolute_tolerance": 1.0e-9,
        },
    }
    assert audit.validate_split_fractions(valid) == (0.6, 0.2, 0.2)

    bad_sum = {**valid, "temporal_final_fraction": 0.21}
    with pytest.raises(RuntimeError, match="required_sum"):
        audit.validate_split_fractions(bad_sum)

    bad_range = {**valid, "train_fraction": 0.0}
    with pytest.raises(RuntimeError, match="strictly between"):
        audit.validate_split_fractions(bad_range)
