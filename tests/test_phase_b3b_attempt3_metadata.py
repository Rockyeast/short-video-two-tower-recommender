from __future__ import annotations

from pathlib import Path


def test_sealed_runner_declares_attempt_three_and_prior_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_phase_b3b_sealed_small.py"
    ).read_text()
    assert "SEALED_ATTEMPT_NUMBER = 3" in source
    assert "small_schema_validation" in source
    assert "two_tower_checkpoint_feature_vocab_validation" in source
    assert "sealed_small_failure.md" in source
    assert "sealed_small_attempt2_failure.md" in source
    assert '"formal_metrics_produced_or_observed": False' in source
    assert '"small_used_for_post_attempt_tuning": False' in source
