from __future__ import annotations

import ast
from pathlib import Path


def test_modal_full_run_wrapper_freezes_scope_and_resources() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "modal_phase_b2b_full_run.py"
    ).read_text()
    tree = ast.parse(source)

    assert (
        'B2B_RUNNER_COMMIT = "7feb5675b7fa6577c68a3775d943c0a32b94f603"'
        in source
    )
    assert 'gpu="L4"' in source
    assert 'gpu="any"' not in source
    assert "memory=16384" in source
    assert "timeout=14400" in source
    assert "startup_timeout=600" in source
    assert "retries=0" in source
    assert "single_use_containers=True" in source
    assert "INPUT_MOUNT: input_volume.read_only()" in source
    assert "OUTPUT_MOUNT: output_volume" in source
    assert '"--full-run"' in source
    assert '"--preflight"' not in source
    assert "FORMAL_STEP_COUNT = 6729" in source
    assert "validation[\"evaluated_queries\"] == 6818" in source
    assert "validation[\"evaluated_targets\"] == 118565" in source
    assert "validation[\"fixed_catalog_count\"] == 9365" in source
    assert '"small_matrix_accessed": False' in source
    assert '"temporal_final_accessed": False' in source
    remote_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
    ]
    assert len(remote_calls) == 1
