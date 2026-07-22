from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from kuairec_phase1.artifacts import ArtifactError
from kuairec_phase1.erratum import (
    LIVENESS_INTERVAL_SECONDS,
    MAX_WALL_SECONDS,
    ROW_TIMEOUT_SECONDS,
    _atomic_write_topk,
    _bpr_model_path,
    _build_ranking_input_manifest,
    _load_cached_row,
    _row_paths,
    _topk_metadata,
    _validate_ranking_manifest_structure,
    _verify_ranking_manifest_files,
    _write_row_checkpoint,
)
from kuairec_phase1.gates import METRICS, SEGMENT_METRICS, GateError
from kuairec_phase1.watchdog import (
    WatchdogTimeout,
    run_supervised,
    terminate_process_group_id,
)


def _row_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    planned: dict[str, object] = {
        "method": "itemcf",
        "config_id": "config-1",
        "hyperparameters": {"neighbor_count": 50, "shrinkage": 10},
        "seed": None,
    }
    original: dict[str, object] = {
        **planned,
        "status": "completed",
        "metrics": {name: 0.1 for name in METRICS},
        "coverage": {"Coverage@100": {"numerator": 1, "denominator": 2}},
    }
    corrected = json.loads(json.dumps(original))
    for name in SEGMENT_METRICS:
        corrected["metrics"][name] = 0.2
    return planned, original, corrected


def _artifact_fixture() -> dict[str, object]:
    return {
        "queries": {"user": np.asarray([0, 1], dtype=np.int32)},
        "catalog": {"video_ids": np.asarray([10, 11, 12, 13], dtype=np.int32)},
        "candidate_bits": np.asarray([[0b0011], [0b1100]], dtype=np.uint8),
    }


def _valid_topk() -> np.ndarray:
    topk = np.full((2, 100), -1, dtype=np.int32)
    topk[0, :2] = [0, 1]
    topk[1, :2] = [2, 3]
    return topk


def _metadata(planned: dict[str, object], artifacts: dict[str, object]):
    return _topk_metadata(
        row_index=0,
        planned=planned,
        binding={
            "cache_key": "bound-cache",
            "ranking_input_manifest_sha256": "ranking-sha",
            "processed_artifact_manifest_sha256": "artifact-sha",
        },
        ranking_input_manifest={
            "candidate_membership": {"sha256": "candidate-sha"}
        },
        artifacts=artifacts,
    )


def test_erratum_row_checkpoint_persists_topk_and_resumes(tmp_path: Path):
    planned, original, corrected = _row_fixture()
    artifacts = _artifact_fixture()
    topk = _valid_topk()
    metadata = _metadata(planned, artifacts)

    _write_row_checkpoint(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        corrected=corrected,
        topk=topk,
        cache_key="bound-cache",
        topk_metadata=metadata,
    )
    cached, partial = _load_cached_row(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        original=original,
        cache_key="bound-cache",
        topk_metadata=metadata,
        artifacts=artifacts,
    )

    assert partial is None
    assert cached == corrected
    topk_path, row_path = _row_paths(tmp_path, 0, planned)
    assert topk_path.is_file()
    assert row_path.is_file()


def test_erratum_topk_only_checkpoint_resumes_without_ranking(tmp_path: Path):
    planned, original, _ = _row_fixture()
    artifacts = _artifact_fixture()
    metadata = _metadata(planned, artifacts)
    topk_path, _ = _row_paths(tmp_path, 0, planned)
    _atomic_write_topk(topk_path, _valid_topk(), metadata)

    cached, partial = _load_cached_row(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        original=original,
        cache_key="bound-cache",
        topk_metadata=metadata,
        artifacts=artifacts,
    )

    assert cached is None
    np.testing.assert_array_equal(partial, _valid_topk())


def test_erratum_checkpoint_tampering_fails_closed(tmp_path: Path):
    planned, original, corrected = _row_fixture()
    artifacts = _artifact_fixture()
    metadata = _metadata(planned, artifacts)
    _write_row_checkpoint(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        corrected=corrected,
        topk=_valid_topk(),
        cache_key="bound-cache",
        topk_metadata=metadata,
    )
    topk_path, _ = _row_paths(tmp_path, 0, planned)
    topk_path.write_bytes(b"tampered")

    with pytest.raises(GateError, match="Top-K hash mismatch"):
        _load_cached_row(
            cache_dir=tmp_path,
            index=0,
            planned=planned,
            original=original,
            cache_key="bound-cache",
            topk_metadata=metadata,
            artifacts=artifacts,
        )


def test_topk_only_checkpoint_rejects_wrong_identity(tmp_path: Path):
    planned, original, _ = _row_fixture()
    artifacts = _artifact_fixture()
    expected = _metadata(planned, artifacts)
    wrong = json.loads(json.dumps(expected))
    wrong["planned"]["config_id"] = "wrong-config"
    topk_path, _ = _row_paths(tmp_path, 0, planned)
    _atomic_write_topk(topk_path, _valid_topk(), wrong)

    with pytest.raises(GateError, match="identity or input binding"):
        _load_cached_row(
            cache_dir=tmp_path,
            index=0,
            planned=planned,
            original=original,
            cache_key="bound-cache",
            topk_metadata=expected,
            artifacts=artifacts,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda topk: topk.astype(np.int64), "dtype"),
        (lambda topk: topk[:, :99], "shape"),
        (
            lambda topk: np.where(
                np.indices(topk.shape)[1] == 0, np.int32(99), topk
            ).astype(np.int32),
            "out-of-range",
        ),
        (
            lambda topk: np.vstack(
                (
                    np.concatenate((np.asarray([2], dtype=np.int32), topk[0, 1:])),
                    topk[1],
                )
            ),
            "outside query candidates",
        ),
    ],
)
def test_topk_only_checkpoint_rejects_invalid_array(
    tmp_path: Path, mutate, message: str
):
    planned, original, _ = _row_fixture()
    artifacts = _artifact_fixture()
    metadata = _metadata(planned, artifacts)
    invalid = mutate(_valid_topk())
    stored_metadata = dict(metadata)
    stored_metadata["shape"] = list(invalid.shape)
    stored_metadata["dtype"] = str(invalid.dtype)
    topk_path, _ = _row_paths(tmp_path, 0, planned)
    _atomic_write_topk(topk_path, invalid, stored_metadata)

    expected = dict(metadata)
    if message in {"dtype", "shape"}:
        # Identity metadata itself is part of the contract for these failures.
        expected = stored_metadata
    with pytest.raises(GateError, match=message):
        _load_cached_row(
            cache_dir=tmp_path,
            index=0,
            planned=planned,
            original=original,
            cache_key="bound-cache",
            topk_metadata=expected,
            artifacts=artifacts,
        )


def test_supervisor_emits_real_liveness_pulses(tmp_path: Path):
    pulses = []
    returncode = run_supervised(
        [sys.executable, "-c", "import time; time.sleep(0.35)"],
        cwd=tmp_path,
        timeout_seconds=2.0,
        liveness_interval_seconds=0.05,
        liveness_callback=pulses.append,
        poll_seconds=0.01,
    )

    assert returncode == 0
    assert len(pulses) >= 3
    assert all(value.pid > 0 for value in pulses)
    assert any(value.state in {"R", "S"} for value in pulses)
    assert any(value.rss_mb is not None for value in pulses)


def test_supervisor_timeout_kills_entire_process_group(tmp_path: Path):
    child_pid_file = tmp_path / "child.pid"
    program = (
        "import pathlib,subprocess,time; "
        "p=subprocess.Popen(['sleep','30']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(WatchdogTimeout, match="exceeded"):
        run_supervised(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            timeout_seconds=0.5,
            liveness_interval_seconds=0.05,
            poll_seconds=0.01,
        )
    assert time.monotonic() - started < 3.0
    child_pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{child_pid}/stat")
        if not stat.exists() or stat.read_text().split(")", 1)[1].strip().startswith("Z"):
            break
        time.sleep(0.02)
    else:
        pytest.fail("grandchild remained runnable after process-group timeout")


def test_outer_timeout_cleans_independent_row_process_group(tmp_path: Path):
    row_group_file = tmp_path / "row_group.txt"
    program = (
        "import os,pathlib,subprocess,time; "
        "p=subprocess.Popen(['sleep','30'], start_new_session=True); "
        f"pathlib.Path({str(row_group_file)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )

    def cleanup(_orchestrator_pid: int) -> None:
        deadline = time.monotonic() + 1.0
        while not row_group_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        terminate_process_group_id(int(row_group_file.read_text()), grace_seconds=0.1)

    with pytest.raises(WatchdogTimeout):
        run_supervised(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            timeout_seconds=0.5,
            liveness_interval_seconds=0.05,
            before_terminate=cleanup,
            poll_seconds=0.01,
        )
    row_pid = int(row_group_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{row_pid}/stat")
        if not stat.exists() or stat.read_text().split(")", 1)[1].strip().startswith("Z"):
            break
        time.sleep(0.02)
    else:
        pytest.fail("independent row process group survived the outer timeout")


def test_ranking_input_manifest_binds_historical_topk_and_bpr(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "topk").mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("manifest")
    (artifact_dir / "candidate_bits_validation.npy").write_text("candidates")
    fallback = artifact_dir / "topk" / "fallback.npz"
    fallback.write_text("fallback")
    time_row = {
        "method": "time_decayed_popularity",
        "config_id": "time-config",
        "hyperparameters": {"variant": "fit_frozen", "half_life_days": 1},
        "seed": None,
    }
    (artifact_dir / "topk" / "time-config.npz").write_text("time-topk")
    bpr_row = {
        "method": "bpr_mf",
        "config_id": "bpr-config",
        "hyperparameters": {
            "embedding_dim": 32,
            "learning_rate": 0.001,
            "l2": 0.0001,
            "epoch": 5,
        },
        "seed": 7,
    }
    bpr_prefix = __import__("hashlib").sha256(fallback.read_bytes()).hexdigest()[:8]
    model = _bpr_model_path(artifact_dir, bpr_row, bpr_prefix)
    model.parent.mkdir(parents=True)
    model.write_text("bpr-checkpoint")
    random_row = {
        "method": "random",
        "config_id": "random-config",
        "hyperparameters": {},
        "seed": 7,
    }
    rows = (time_row, bpr_row, random_row)

    manifest = _build_ranking_input_manifest(
        root=tmp_path,
        artifact_dir=artifact_dir,
        plan_rows=rows,
        fallback_file=fallback,
    )
    _validate_ranking_manifest_structure(manifest, rows)
    _verify_ranking_manifest_files(tmp_path, manifest, row_index=None)

    assert len(manifest["rows"][0]["files"]) == 1
    assert len(manifest["rows"][1]["files"]) == 2
    assert manifest["rows"][2]["files"] == []
    model.write_text("tampered")
    with pytest.raises(ArtifactError, match="Ranking input hash mismatch"):
        _verify_ranking_manifest_files(tmp_path, manifest, row_index=1)


def test_formal_execution_deadlines_are_locked():
    assert ROW_TIMEOUT_SECONDS == 10 * 60
    assert LIVENESS_INTERVAL_SECONDS == 30
    assert MAX_WALL_SECONDS == 3 * 60 * 60
