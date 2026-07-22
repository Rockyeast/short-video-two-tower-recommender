from __future__ import annotations

import hashlib

import numpy as np
import pytest

from kuairec_phase1.artifacts import (
    ArtifactError,
    build_selection_segment_membership,
)


def _fixture() -> dict[str, np.ndarray | float]:
    return {
        "video_ids": np.asarray([101, 102, 103, 104], dtype=np.int32),
        # Item 101 has a non-positive train interaction. Item 102 has a train
        # interaction and is the only eligible strong-positive train target.
        # Item 103 appears only in validation; item 104 never appears.
        "event_items": np.asarray([0, 1, 2], dtype=np.int32),
        "event_timestamps": np.asarray([1.0, 2.0, 11.0]),
        "positive_target_items": np.asarray([1], dtype=np.int32),
        "train_end_exclusive": 10.0,
    }


def test_membership_separates_interactions_positive_targets_and_model_updates():
    membership = build_selection_segment_membership(**_fixture())

    np.testing.assert_array_equal(membership["interaction_count"], [1, 1, 0, 0])
    np.testing.assert_array_equal(
        membership["positive_target_count"], [0, 1, 0, 0]
    )
    np.testing.assert_array_equal(membership["data_warm"], [True, True, False, False])
    np.testing.assert_array_equal(membership["data_cold"], [False, False, True, True])
    np.testing.assert_array_equal(membership["head"], [False, True, False, False])
    np.testing.assert_array_equal(membership["tail"], [True, False, False, False])

    # Optimizer-touched IDs are a future model-run artifact, not something that
    # may be inferred from either interaction or positive-target membership.
    model_id_touched = np.asarray([True, False, False, False])
    assert "model_id_trained" not in membership
    assert not np.array_equal(model_id_touched, membership["data_warm"])
    assert not np.array_equal(model_id_touched, membership["positive_target_count"] > 0)


def test_validation_events_do_not_enter_selection_data_warm_membership():
    membership = build_selection_segment_membership(**_fixture())

    assert membership["interaction_count"][2] == 0
    assert not membership["data_warm"][2]
    assert membership["data_cold"][2]


def test_positive_target_mask_cannot_masquerade_as_data_warm_mask():
    membership = build_selection_segment_membership(**_fixture())

    positive_mask = membership["positive_target_count"] > 0
    assert membership["data_warm"][0]
    assert not positive_mask[0]
    assert not np.array_equal(positive_mask, membership["data_warm"])


def test_membership_count_and_hash_mismatch_fail_closed():
    values = _fixture()
    expected_hash = hashlib.sha256(b"101\n102\n").hexdigest()
    valid = build_selection_segment_membership(
        **values,
        expected_data_warm_count=2,
        expected_data_warm_sha256=expected_hash,
    )
    assert valid["data_warm_sha256"] == expected_hash

    with pytest.raises(ArtifactError, match="data-warm item count mismatch"):
        build_selection_segment_membership(
            **values,
            expected_data_warm_count=3,
            expected_data_warm_sha256=expected_hash,
        )
    with pytest.raises(ArtifactError, match="data-warm membership SHA256 mismatch"):
        build_selection_segment_membership(
            **values,
            expected_data_warm_count=2,
            expected_data_warm_sha256="0" * 64,
        )
