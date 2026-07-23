from pathlib import Path


def test_modal_b4a_wrapper_is_bounded_and_has_no_small_input():
    source = Path(
        "scripts/modal_phase_b4a_faiss_benchmark.py"
    ).read_text()
    assert (
        'RUNNER_COMMIT = "af4fa9aebba1cdbcea0cdbb7983fd99952db3db7"'
        in source
    )
    assert 'gpu="L4"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "small_matrix" not in source.lower()
    assert "SMALL_MOUNT" not in source
    runner = Path("scripts/run_phase_b4a_faiss_benchmark.py").read_text()
    assert '"small_matrix_accessed": False' in runner
    assert '"temporal_final_accessed": False' in runner
