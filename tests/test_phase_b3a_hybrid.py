from __future__ import annotations

import numpy as np
import pytest

from kuairec_fully_observed.hybrid import (
    FROZEN_HYBRID_ALPHAS,
    select_frozen_hybrid,
    weighted_reciprocal_rank_fusion,
)


def test_weighted_rrf_matches_hand_calculation() -> None:
    two_tower = np.asarray([[10, 20, 30]], dtype=np.int64)
    bpr = np.asarray([[30, 20, 40]], dtype=np.int64)

    fused = weighted_reciprocal_rank_fusion(
        two_tower,
        bpr,
        candidates=(np.asarray([10, 20, 30, 40]),),
        alpha=0.50,
        output_k=4,
    )

    # 30: .5/63 + .5/61; 20: .5/62 + .5/62;
    # 10: .5/61; 40: .5/63.
    np.testing.assert_array_equal(fused, [[30, 20, 10, 40]])


def test_weighted_rrf_missing_route_contributes_zero() -> None:
    fused = weighted_reciprocal_rank_fusion(
        np.asarray([[10, 20, -1]]),
        np.asarray([[30, 40, -1]]),
        candidates=(np.asarray([10, 20, 30, 40]),),
        alpha=0.75,
        output_k=4,
    )

    np.testing.assert_array_equal(fused, [[10, 20, 30, 40]])


def test_weighted_rrf_rejects_grid_expansion() -> None:
    with pytest.raises(ValueError, match="frozen"):
        weighted_reciprocal_rank_fusion(
            np.asarray([[1]]),
            np.asarray([[1]]),
            candidates=(np.asarray([1]),),
            alpha=0.60,
        )


def test_selection_uses_constraints_then_highest_ndcg() -> None:
    two_tower = {"Recall@100": 0.10, "Coverage@100": 0.50}
    metrics = {
        0.25: {"Recall@100": 0.097, "Coverage@100": 0.50, "NDCG@20": 0.30},
        0.50: {"Recall@100": 0.099, "Coverage@100": 0.46, "NDCG@20": 0.20},
        0.75: {"Recall@100": 0.099, "Coverage@100": 0.44, "NDCG@20": 0.40},
    }

    selected = select_frozen_hybrid(
        two_tower_metrics=two_tower, hybrid_metrics=metrics
    )

    assert selected.recall_minimum == pytest.approx(0.098)
    assert selected.coverage_minimum == pytest.approx(0.45)
    assert selected.eligible_alphas == (0.50,)
    assert selected.selected_alpha == 0.50


def test_selection_reports_no_passing_hybrid_without_expanding_grid() -> None:
    metrics = {
        alpha: {
            "Recall@100": 0.09,
            "Coverage@100": 0.40,
            "NDCG@20": 1.0,
        }
        for alpha in FROZEN_HYBRID_ALPHAS
    }

    selected = select_frozen_hybrid(
        two_tower_metrics={"Recall@100": 0.10, "Coverage@100": 0.50},
        hybrid_metrics=metrics,
    )

    assert selected.eligible_alphas == ()
    assert selected.selected_alpha is None
