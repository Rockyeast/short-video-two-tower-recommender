from pathlib import Path


def test_modal_sasrec_wrapper_has_bounded_single_l4_contract() -> None:
    source = Path("scripts/modal_phase_b6a_sasrec.py").read_text()
    assert 'RUNNER_COMMIT = "ac0a380341d21abb70932b8fad7cb53615929b9e"' in source
    assert 'gpu="L4"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "smoke=False" in source
    assert 'device_name="cuda:0"' in source
    assert 'report["training_examples"] != 573104' in source
    assert "small_matrix.csv" not in source
    assert "temporal_final" in source
