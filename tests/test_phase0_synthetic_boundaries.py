from __future__ import annotations

import pandas as pd

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
            [0, 1, 4000, 1000, 100.0, 3.0],
            # Equal-time quick skip: not future and therefore not hard.
            [0, 2, 100, 1000, 100.0, 0.1],
            # Exactly 30 minutes later: same session and included.
            [0, 3, 100, 1000, 1900.0, 0.1],
            # Same hard item later becomes positive: diagnostic risk only.
            [0, 3, 4000, 1000, 1999.0, 3.0],
            # Exact train cutoff: forbidden to a selection-time pool.
            [0, 4, 100, 1000, 2000.0, 0.1],
        ],
        columns=[
            "user_id",
            "video_id",
            "play_duration",
            "video_duration",
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
