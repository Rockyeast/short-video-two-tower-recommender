from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from kuairec_phase1.erratum import (
    HEARTBEAT_SECONDS,
    MAX_WALL_SECONDS,
    ROW_TIMEOUT_SECONDS,
    _load_cached_row,
    _row_paths,
    _write_row_checkpoint,
)
from kuairec_phase1.gates import METRICS, SEGMENT_METRICS, GateError
from kuairec_phase1.watchdog import WatchdogTimeout, run_supervised


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


def test_erratum_row_checkpoint_persists_topk_and_resumes(tmp_path: Path):
    planned, original, corrected = _row_fixture()
    topk = np.arange(24, dtype=np.int32).reshape(3, 8)

    _write_row_checkpoint(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        corrected=corrected,
        topk=topk,
        cache_key="bound-cache",
    )
    cached, partial = _load_cached_row(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        original=original,
        cache_key="bound-cache",
    )

    assert partial is None
    assert cached == corrected
    topk_path, row_path = _row_paths(tmp_path, 0, planned)
    assert topk_path.is_file()
    assert row_path.is_file()


def test_erratum_topk_only_checkpoint_resumes_without_ranking(tmp_path: Path):
    planned, original, _ = _row_fixture()
    topk_path, _ = _row_paths(tmp_path, 0, planned)
    topk_path.parent.mkdir(parents=True)
    with topk_path.open("wb") as handle:
        np.savez_compressed(handle, topk=np.asarray([[3, 2, 1]], dtype=np.int32))

    cached, partial = _load_cached_row(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        original=original,
        cache_key="bound-cache",
    )

    assert cached is None
    np.testing.assert_array_equal(partial, [[3, 2, 1]])


def test_erratum_checkpoint_tampering_fails_closed(tmp_path: Path):
    planned, original, corrected = _row_fixture()
    _write_row_checkpoint(
        cache_dir=tmp_path,
        index=0,
        planned=planned,
        corrected=corrected,
        topk=np.asarray([[1, 2, 3]], dtype=np.int32),
        cache_key="bound-cache",
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
        )


def test_supervisor_emits_real_heartbeats(tmp_path: Path):
    heartbeats = []
    returncode = run_supervised(
        [sys.executable, "-c", "import time; time.sleep(0.35)"],
        cwd=tmp_path,
        timeout_seconds=2.0,
        heartbeat_seconds=0.05,
        heartbeat=heartbeats.append,
        poll_seconds=0.01,
    )

    assert returncode == 0
    assert len(heartbeats) >= 3
    assert all(value.pid > 0 for value in heartbeats)
    assert any(value.state in {"R", "S"} for value in heartbeats)
    assert any(value.rss_mb is not None for value in heartbeats)


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
            heartbeat_seconds=0.05,
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


def test_formal_execution_deadlines_are_locked():
    assert ROW_TIMEOUT_SECONDS == 10 * 60
    assert HEARTBEAT_SECONDS == 30
    assert MAX_WALL_SECONDS == 3 * 60 * 60
