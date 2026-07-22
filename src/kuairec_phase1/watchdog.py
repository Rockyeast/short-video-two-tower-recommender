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
class ProcessHeartbeat:
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


def run_supervised(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    heartbeat_seconds: float,
    heartbeat: Callable[[ProcessHeartbeat], None] | None = None,
    environment: Mapping[str, str] | None = None,
    poll_seconds: float = 0.25,
) -> int:
    """Run one isolated process group with real heartbeats and a hard timeout."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment) if environment is not None else None,
        start_new_session=True,
    )
    started = time.monotonic()
    next_heartbeat = started
    last_sample_time = started
    last_cpu_seconds = 0.0
    try:
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if returncode is not None:
                return int(returncode)
            if now - started >= timeout_seconds:
                terminate_process_group(process)
                raise WatchdogTimeout(
                    f"Process group {process.pid} exceeded {timeout_seconds:.1f} seconds"
                )
            if now >= next_heartbeat:
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
                if heartbeat is not None:
                    heartbeat(
                        ProcessHeartbeat(
                            pid=process.pid,
                            process_group_id=process.pid,
                            elapsed_seconds=now - started,
                            cpu_percent=cpu_percent,
                            rss_mb=rss_mb,
                            state=state,
                        )
                    )
                next_heartbeat = now + heartbeat_seconds
            time.sleep(min(poll_seconds, max(0.01, timeout_seconds - (now - started))))
    except BaseException:
        terminate_process_group(process)
        raise
