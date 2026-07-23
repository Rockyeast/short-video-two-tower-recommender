from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.run_phase_b3b_sealed_small import _small_audit_counts


def test_small_audit_counts_keep_observed_frame_separate() -> None:
    small = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 11, 12],
            "watch_ratio": [3.0, 0.5, 4.0],
        }
    )
    observed_normal = small[small["video_id"].isin([10, 12])]
    counts = _small_audit_counts(
        small=small,
        small_observed_normal=observed_normal,
        queries=SimpleNamespace(
            diagnostics={"zero_relevant_users_excluded": 1}
        ),
        cold_items=np.asarray([12], dtype=np.int64),
    )
    assert counts == {
        "observed_pair_count": 3,
        "observed_normal_pair_count": 2,
        "normal_candidate_item_count": 2,
        "excluded_zero_relevant_user_count": 1,
        "data_cold_item_count": 1,
    }
