from __future__ import annotations

from pathlib import Path


def test_sealed_runner_declares_attempt_five_and_prior_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_phase_b3b_sealed_small.py"
    ).read_text()
    assert "SEALED_ATTEMPT_NUMBER = 5" in source
    assert "small_schema_validation" in source
    assert "two_tower_checkpoint_feature_vocab_validation" in source
    assert "sealed_small_failure.md" in source
    assert "sealed_small_attempt2_failure.md" in source
    assert "sealed_small_attempt3_failure.md" in source
    assert "sealed_small_attempt4_failure.md" in source
    assert (
        "two_tower_checkpoint_numeric_preprocessing_identity_validation"
        in source
    )
    assert '"formal_metrics_computed": True' in source
    assert '"formal_metrics_exposed": False' in source
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
        'RUNNER_COMMIT = "82e3a462d1c71828a4fb0ffd53ca77da1fad28e2"'
        in wrapper
    )
    assert "SEALED_ATTEMPT_NUMBER = 5" in wrapper
    assert "SEALED_ATTEMPT_NUMBER = 5" in runner
    assert "load_final_refit_checkpoint_compatible" in runner
    assert "prepare_final_refit_inference_feature_store" in runner
    assert (
        'REPOSITORY_DIR\n'
        '                / "manifests/phase_b3b_final_numeric_preprocessing.json"'
        in wrapper
    )
    for failure_report in (
        "sealed_small_failure.md",
        "sealed_small_attempt2_failure.md",
        "sealed_small_attempt3_failure.md",
        "sealed_small_attempt4_failure.md",
    ):
        assert failure_report in wrapper
        assert failure_report in runner


def test_attempt_five_sidecar_matches_checkpoint_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    sidecar = (
        root / "manifests" / "phase_b3b_final_numeric_preprocessing.json"
    ).read_text()
    expected = (
        "71217e8965b59915874ab879e157eacd61a1c55813604bcdb9e748e08458c489"
    )
    assert (
        f'"expected_numeric_preprocessing_sha256": "{expected}"'
        in sidecar
    )
    assert f'"numeric_preprocessing_sha256": "{expected}"' in sidecar
