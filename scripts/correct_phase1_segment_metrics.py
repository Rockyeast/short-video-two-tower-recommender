#!/usr/bin/env python3
"""Run ERRATUM-001 segment-only correction without opening holdouts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairec_phase1.artifacts import ArtifactError
from kuairec_phase1.erratum import (
    ERRATUM_ID,
    LIVENESS_INTERVAL_SECONDS,
    MAX_WALL_SECONDS,
    run_segment_membership_erratum,
)
from kuairec_phase1.gates import GateError, sha256_file
from kuairec_phase1.publish import PublishTransactionError
from kuairec_phase1.watchdog import (
    ProcessLivenessPulse,
    WatchdogTimeout,
    run_supervised,
    terminate_process_group_id,
)


def _run_orchestrator() -> int:
    def stop_on_signal(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        result = run_segment_membership_erratum(ROOT)
    except (ArtifactError, GateError, PublishTransactionError) as exc:
        print(f"ERRATUM-001 ABORTED: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print(
            "ERRATUM-001 ABORTED: supervisor interruption",
            file=sys.stderr,
            flush=True,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _supervisor_liveness_pulse(sample: ProcessLivenessPulse) -> None:
    print(
        json.dumps(
            {
                "pulse_type": "process_liveness_not_progress",
                "stage": "external_three_hour_supervisor",
                "pid": sample.pid,
                "process_group_id": sample.process_group_id,
                "elapsed_seconds": round(sample.elapsed_seconds, 2),
                "cpu_percent": (
                    round(sample.cpu_percent, 2)
                    if sample.cpu_percent is not None
                    else None
                ),
                "rss_mb": round(sample.rss_mb, 2) if sample.rss_mb is not None else None,
                "process_state": sample.state,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _cleanup_active_row(orchestrator_pid: int) -> None:
    manifest_sha = sha256_file(ROOT / "manifests/split_manifest.json")
    liveness = (
        ROOT
        / "artifacts"
        / "phase1"
        / manifest_sha
        / "errata"
        / ERRATUM_ID
        / "ACTIVE_ROW_LIVENESS.json"
    )
    try:
        payload = json.loads(liveness.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if (
        payload.get("status") != "running"
        or payload.get("orchestrator_pid") != orchestrator_pid
    ):
        return
    process_group_id = payload.get("worker_process_group_id")
    if isinstance(process_group_id, int):
        terminate_process_group_id(process_group_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.orchestrate:
        return _run_orchestrator()

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return run_supervised(
            [sys.executable, str(Path(__file__).resolve()), "--orchestrate"],
            cwd=ROOT,
            timeout_seconds=float(MAX_WALL_SECONDS),
            liveness_interval_seconds=float(LIVENESS_INTERVAL_SECONDS),
            liveness_callback=_supervisor_liveness_pulse,
            environment=environment,
            before_terminate=_cleanup_active_row,
        )
    except WatchdogTimeout as exc:
        print(f"ERRATUM-001 ABORTED: {exc}", file=sys.stderr, flush=True)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
