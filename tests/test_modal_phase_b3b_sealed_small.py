from __future__ import annotations

from pathlib import Path


def test_modal_sealed_wrapper_freezes_single_l4_and_one_call() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "modal_phase_b3b_sealed_small.py"
    ).read_text()
    assert 'gpu="L4"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "run_sealed.remote()" in source
    assert "--full-run" not in source
    assert "temporal_final" in source


def test_sealed_runner_reports_frozen_audit_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_phase_b3b_sealed_small.py"
    ).read_text()
    for field in (
        "observed_pair_count",
        "observed_normal_pair_count",
        "normal_candidate_item_count",
        "excluded_zero_relevant_user_count",
        "data_cold_item_count",
    ):
        assert field in source
