"""OS-level supervision helpers for long-running experiment subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


class WatchdogTimeout(RuntimeError):
    """Raised after a supervised process group exceeds its hard deadline."""


@dataclass(frozen=True)
class ProcessLivenessPulse:
    pid: int
    process_group_id: int
    elapsed_seconds: float
    cpu_percent: float | None
    rss_mb: float | None
    state: str | None


def _process_sample(pid: int) -> tuple[float, float, str] | None:
    """Return cumulative CPU seconds, RSS MiB and Linux process state."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        closing = stat.rfind(")")
        fields = stat[closing + 2 :].split()
        state = fields[0]
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_seconds = (float(fields[11]) + float(fields[12])) / float(ticks)
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        rss_mb = resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
        return cpu_seconds, rss_mb, state
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def terminate_process_group(
    process: subprocess.Popen[object], *, grace_seconds: float = 5.0
) -> None:
    """Terminate, then forcibly kill, the isolated group owned by ``process``."""

    if process.poll() is not None:
        return
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(grace_seconds, 1.0))


def terminate_process_group_id(
    process_group_id: int, *, grace_seconds: float = 1.0
) -> None:
    """Best-effort cleanup for a nested group not owned by this supervisor."""

    if process_group_id <= 1:
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_supervised(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    liveness_interval_seconds: float,
    liveness_callback: Callable[[ProcessLivenessPulse], None] | None = None,
    environment: Mapping[str, str] | None = None,
    poll_seconds: float = 0.25,
    before_terminate: Callable[[int], None] | None = None,
) -> int:
    """Run one isolated process group with liveness pulses and a hard timeout."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if liveness_interval_seconds <= 0:
        raise ValueError("liveness_interval_seconds must be positive")
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment) if environment is not None else None,
        start_new_session=True,
    )
    started = time.monotonic()
    next_liveness_pulse = started
    last_sample_time = started
    last_cpu_seconds = 0.0
    cleanup_called = False

    def cleanup_nested() -> None:
        nonlocal cleanup_called
        if not cleanup_called and before_terminate is not None:
            cleanup_called = True
            before_terminate(process.pid)

    try:
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if returncode is not None:
                return int(returncode)
            if now - started >= timeout_seconds:
                cleanup_nested()
                terminate_process_group(process)
                raise WatchdogTimeout(
                    f"Process group {process.pid} exceeded {timeout_seconds:.1f} seconds"
                )
            if now >= next_liveness_pulse:
                sample = _process_sample(process.pid)
                cpu_percent: float | None = None
                rss_mb: float | None = None
                state: str | None = None
                if sample is not None:
                    cpu_seconds, rss_mb, state = sample
                    interval = now - last_sample_time
                    if interval > 0:
                        cpu_percent = max(
                            0.0, (cpu_seconds - last_cpu_seconds) / interval * 100.0
                        )
                    last_cpu_seconds = cpu_seconds
                    last_sample_time = now
                if liveness_callback is not None:
                    liveness_callback(
                        ProcessLivenessPulse(
                            pid=process.pid,
                            process_group_id=process.pid,
                            elapsed_seconds=now - started,
                            cpu_percent=cpu_percent,
                            rss_mb=rss_mb,
                            state=state,
                        )
                    )
                next_liveness_pulse = now + liveness_interval_seconds
            time.sleep(min(poll_seconds, max(0.01, timeout_seconds - (now - started))))
    except BaseException:
        cleanup_nested()
        terminate_process_group(process)
        raise
