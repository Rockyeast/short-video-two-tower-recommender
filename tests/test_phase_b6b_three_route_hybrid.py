from __future__ import annotations

import numpy as np
import pytest

from kuairec_fully_observed.three_route_hybrid import (
    FROZEN_THREE_ROUTE_WEIGHTS,
    select_three_route_hybrid,
    weighted_three_route_rrf,
)


def test_three_route_rrf_matches_hand_calculation() -> None:
    fused = weighted_three_route_rrf(
        np.asarray([[10, 20, 30]]),
        np.asarray([[30, 20, 40]]),
        np.asarray([[40, 20, 10]]),
        candidates=(np.asarray([10, 20, 30, 40]),),
        weights=(0.50, 0.35, 0.15),
        output_k=4,
    )

    # 20 receives all three routes; 30 receives TT rank 3 + SASRec rank 1.
    np.testing.assert_array_equal(fused, [[20, 30, 10, 40]])


def test_three_route_rrf_missing_route_contributes_zero() -> None:
    fused = weighted_three_route_rrf(
        np.asarray([[10, 20, -1, -1]]),
        np.asarray([[30, 40, -1, -1]]),
        np.asarray([[20, 40, -1, -1]]),
        candidates=(np.asarray([10, 20, 30, 40]),),
        weights=(0.60, 0.25, 0.15),
        output_k=4,
    )

    np.testing.assert_array_equal(fused, [[20, 10, 40, 30]])


def test_three_route_rrf_rejects_unregistered_weights() -> None:
    with pytest.raises(ValueError, match="frozen"):
        weighted_three_route_rrf(
            np.asarray([[1]]),
            np.asarray([[1]]),
            np.asarray([[1]]),
            candidates=(np.asarray([1]),),
            weights=(0.34, 0.33, 0.33),
        )


def test_three_route_selection_preserves_all_three_retrieval_metrics() -> None:
    reference = {
        "Recall@100": 0.10,
        "Coverage@100": 0.50,
        "Data-Cold Recall@100": 0.04,
    }
    metrics = {
        (0.60, 0.25, 0.15): {
            "Recall@100": 0.099,
            "Coverage@100": 0.46,
            "Data-Cold Recall@100": 0.038,
            "NDCG@20": 0.02,
        },
        (0.50, 0.35, 0.15): {
            "Recall@100": 0.099,
            "Coverage@100": 0.46,
            "Data-Cold Recall@100": 0.037,
            "NDCG@20": 0.03,
        },
        (0.40, 0.45, 0.15): {
            "Recall@100": 0.099,
            "Coverage@100": 0.46,
            "Data-Cold Recall@100": 0.02,
            "NDCG@20": 0.10,
        },
    }

    selected = select_three_route_hybrid(
        reference_metrics=reference, candidate_metrics=metrics
    )

    assert selected.eligible_weights == FROZEN_THREE_ROUTE_WEIGHTS[:2]
    assert selected.selected_weights == (0.50, 0.35, 0.15)
