from __future__ import annotations

from pathlib import Path

from scripts.diagnose_phase_b3b_numeric_preprocessing import (
    _field_differences,
)


def test_numeric_diagnostic_reports_float_hex_difference() -> None:
    left = {
        "preprocessing": {"means": [1.0], "missing_value_counts": {"x": 0}},
        "preprocessing_float_hex": {"means": [float(1.0).hex()]},
    }
    right = {
        "preprocessing": {
            "means": [1.0 + 2**-52],
            "missing_value_counts": {"x": 0},
        },
        "preprocessing_float_hex": {
            "means": [float(1.0 + 2**-52).hex()]
        },
    }
    differences = _field_differences(left, right)
    assert differences["means"][0]["training_hex"] == "0x1.0000000000000p+0"
    assert differences["means"][0]["sealed_hex"] == "0x1.0000000000001p+0"


def test_numeric_diagnostic_has_no_small_path() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "diagnose_phase_b3b_numeric_preprocessing.py"
    ).read_text()
    assert "small_matrix.csv" not in source
    assert "_load_small_once" not in source


def test_modal_numeric_diagnostic_uses_three_single_use_processes() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "modal_phase_b3b_r3_numeric_diagnostic.py"
    ).read_text()
    assert "single_use_containers=True" in source
    assert "retries=0" in source
    assert "results = [reconstruct.remote(index) for index in range(3)]" in source
    assert '"reconstructions_identical": reconstructions_identical' in source
    assert "SMALL_VOLUME" not in source
    assert "small_matrix.csv" not in source
