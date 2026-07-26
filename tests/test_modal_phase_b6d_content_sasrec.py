from pathlib import Path


def test_modal_content_sasrec_wrapper_is_single_fixed_l4_run() -> None:
    source = Path("scripts/modal_phase_b6d_content_sasrec.py").read_text()
    assert (
        'RUNNER_COMMIT = "4a74a44e4c6ef440c06a4856703213e60cf02025"'
        in source
    )
    assert 'gpu="L4"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert 'device_name="cuda:0"' in source
    assert 'len(report["epochs"]) != 5' in source
    assert 'report["training_examples"] != 573104' in source
    assert "small_matrix.csv" not in source
    assert "temporal_final" in source


def test_full_runner_uses_content_model_without_search_grid() -> None:
    source = Path("scripts/run_phase_b6d_content_sasrec_full.py").read_text()
    config = Path("configs/phase_b6d_content_sasrec_full.yaml").read_text()
    assert "build_content_recbole_sasrec" in source
    assert "build_recbole_sasrec(" not in source
    assert "epochs: 5" in config
    assert "one_configuration: true" in config
    assert "one_seed: true" in config
