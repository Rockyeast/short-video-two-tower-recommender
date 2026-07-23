from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.precompute_phase_b3b_captions import refit_item_universe
from kuairec_fully_observed import require_sealed_execution


def test_final_recipe_is_locked_before_small():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/phase_b3b_final_recipe.yaml").read_text()
    )
    assert config["fit_context"] == "canonical_big_train_plus_validation"
    assert config["selection_complete"] is True
    assert config["bpr"]["epochs"] == 20
    assert config["two_tower"]["training"]["epochs"] == 1
    assert config["hybrid"] == {
        "routes": ["two_tower", "bpr"],
        "route_top_k": 500,
        "rank_constant": 60,
        "alpha": 0.75,
        "output_k": 100,
    }


def test_refit_item_universe_uses_fit_history_plus_all_normal(tmp_path):
    events = tmp_path / "events_train_validation.npz"
    catalog = tmp_path / "catalog.npz"
    np.savez(
        events,
        item=np.asarray([0, 1, 2, 3]),
        timestamp=np.asarray([1.0, 2.0, 3.0, 9.0]),
    )
    np.savez(
        catalog,
        video_ids=np.asarray([10, 11, 12, 13]),
        validation_end=np.asarray([5.0]),
    )
    actual = refit_item_universe(
        artifact_dir=tmp_path, normal_item_ids=np.asarray([11, 13, 20])
    )
    np.testing.assert_array_equal(actual, [10, 11, 12, 13, 20])


def test_sealed_runner_refuses_before_reading_any_input(tmp_path):
    with pytest.raises(RuntimeError, match="requires --execute-sealed-small"):
        require_sealed_execution(False)
    assert list(tmp_path.iterdir()) == []
