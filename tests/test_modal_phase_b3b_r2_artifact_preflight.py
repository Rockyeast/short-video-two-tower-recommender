from __future__ import annotations

from pathlib import Path


def test_artifact_preflight_wrapper_is_single_l4_without_small() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "modal_phase_b3b_r2_artifact_preflight.py"
    ).read_text()
    assert 'gpu="L4"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert source.count("run_artifact_preflight.remote()") == 1
    assert "SMALL_VOLUME" not in source
    assert "small_matrix.csv" not in source
    assert "execute_sealed_small" not in source
    assert "compute_metrics" not in source
    assert "evaluate_rankings" not in source
