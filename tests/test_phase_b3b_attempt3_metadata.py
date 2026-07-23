from __future__ import annotations

from pathlib import Path


def test_sealed_runner_declares_attempt_four_and_prior_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_phase_b3b_sealed_small.py"
    ).read_text()
    assert "SEALED_ATTEMPT_NUMBER = 4" in source
    assert "small_schema_validation" in source
    assert "two_tower_checkpoint_feature_vocab_validation" in source
    assert "sealed_small_failure.md" in source
    assert "sealed_small_attempt2_failure.md" in source
    assert "sealed_small_attempt3_failure.md" in source
    assert '"formal_metrics_produced_or_observed": False' in source
    assert '"small_used_for_post_attempt_tuning": False' in source


def test_modal_wrapper_pins_compatible_runner() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (
        root / "scripts" / "modal_phase_b3b_sealed_small.py"
    ).read_text()
    runner = (
        root / "scripts" / "run_phase_b3b_sealed_small.py"
    ).read_text()
    assert (
        'RUNNER_COMMIT = "90eced9e062004b5954fab257989b96f2a43339c"'
        in wrapper
    )
    assert "SEALED_ATTEMPT_NUMBER = 3" in wrapper
    assert "SEALED_ATTEMPT_NUMBER = 4" in runner
    assert "load_final_refit_checkpoint_compatible" in runner
    for failure_report in (
        "sealed_small_failure.md",
        "sealed_small_attempt2_failure.md",
    ):
        assert failure_report in wrapper
        assert failure_report in runner
