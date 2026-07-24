#!/usr/bin/env python3
"""Run the single fixed RecBole SASRec development experiment on one L4."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modal_preflight_helpers import (  # noqa: E402
    build_input_allowlist,
    input_bundle_manifest,
    modal_volume_file_paths,
)

RUNNER_COMMIT = "ac0a380341d21abb70932b8fad7cb53615929b9e"
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
OUTPUT_VOLUME_NAME = "kuairec-b6a-sasrec-artifacts"
INPUT_MOUNT = Path("/inputs")
OUTPUT_MOUNT = Path("/outputs")
REMOTE_REPOSITORY = Path("/opt/repository")
INPUT_VOLUME = modal.Volume.from_name(
    INPUT_VOLUME_NAME, create_if_missing=False
)
OUTPUT_VOLUME = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME, create_if_missing=True
)

app = modal.App("short-video-recbole-sasrec-l4")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "modal==1.4.1",
        "numpy==2.2.6",
        "pandas==2.3.3",
        "PyYAML==6.0.3",
        "scipy==1.16.3",
        "torch==2.11.0",
        "colorlog==4.7.2",
        "colorama==0.4.4",
        "scikit-learn==1.9.0",
        "tensorboard==2.21.0",
        "thop==0.1.1.post2209072238",
        "tabulate==0.10.0",
        "texttable==1.7.0",
        "tqdm==4.69.1",
        "plotly==6.9.0",
        "psutil==7.2.2",
    )
    # RecBole's SASRec path does not use Ray. RecBole 1.2.1 nevertheless
    # declares an old Ray upper bound that has no Python 3.12 wheel.
    .run_commands("python -m pip install --no-deps recbole==1.2.1")
    .add_local_dir(
        REPOSITORY_ROOT / "src",
        str(REMOTE_REPOSITORY / "src"),
        copy=True,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "scripts",
        str(REMOTE_REPOSITORY / "scripts"),
        copy=True,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "configs",
        str(REMOTE_REPOSITORY / "configs"),
        copy=True,
    )
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _runner_inputs_match_commit() -> bool:
    paths = (
        "configs/phase_b6a_recbole_sasrec.yaml",
        "scripts/run_phase_b6a_recbole_sasrec.py",
        "src/kuairec_fully_observed/sasrec_adapter.py",
    )
    result = subprocess.run(
        ["git", "diff", "--quiet", RUNNER_COMMIT, "--", *paths],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return result.returncode == 0


def _read_volume_file(path: str) -> bytes:
    return b"".join(INPUT_VOLUME.read_file(path))


def _verify_existing_bundle(files, manifest: dict[str, Any]) -> str:
    version_root = f"bundles/{manifest['bundle_sha256']}"
    entries = INPUT_VOLUME.listdir(version_root, recursive=True)
    actual = modal_volume_file_paths(entries)
    expected = {
        f"{version_root}/{record.logical_path}" for record in files
    } | {f"{version_root}/allowlist.json"}
    if actual != expected:
        raise RuntimeError("Existing Modal input bundle membership changed")
    remote_manifest = json.loads(
        _read_volume_file(f"{version_root}/allowlist.json")
    )
    if remote_manifest != manifest:
        raise RuntimeError("Existing Modal input bundle identity changed")
    return version_root


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=3600,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT: INPUT_VOLUME.read_only(),
        OUTPUT_MOUNT: OUTPUT_VOLUME,
    },
    include_source=False,
    serialized=True,
)
def run_l4(
    *, bundle_sha256: str, input_manifest: dict[str, Any]
) -> dict[str, Any]:
    import hashlib
    import os
    import resource
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")

    sys.path.insert(0, str(REMOTE_REPOSITORY))
    sys.path.insert(0, str(REMOTE_REPOSITORY / "src"))
    from scripts.modal_preflight_helpers import verify_remote_inputs
    from scripts.run_phase_b6a_recbole_sasrec import run

    bundle_root = INPUT_MOUNT / "bundles" / bundle_sha256
    stored_manifest = json.loads(
        (bundle_root / "allowlist.json").read_text()
    )
    if stored_manifest != input_manifest:
        raise RuntimeError("Mounted input manifest changed")
    remote_inputs = verify_remote_inputs(bundle_root, input_manifest)

    output_root = Path("/tmp/phase_b6a")
    checkpoints = output_root / "checkpoints"
    report_json = output_root / "sasrec_full.json"
    report_markdown = output_root / "sasrec_full.md"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(REMOTE_REPOSITORY)
        report = run(
            repo_root=REMOTE_REPOSITORY,
            data_dir=bundle_root / "raw",
            artifact_dir=bundle_root / "processed",
            config_path=REMOTE_REPOSITORY
            / "configs/phase_b6a_recbole_sasrec.yaml",
            checkpoint_dir=checkpoints,
            report_json=report_json,
            report_markdown=report_markdown,
            smoke=False,
            device_name="cuda:0",
        )
    finally:
        os.chdir(previous_cwd)
    if (
        report["evaluated_query_count"] != 6818
        or report["training_examples"] != 574091
        or report["claims"]["small_matrix_accessed"]
        or report["claims"]["temporal_final_accessed"]
    ):
        raise RuntimeError("Full SASRec report contract failed")

    persisted = OUTPUT_MOUNT / "phase-b6a-recbole-sasrec-v1"
    persisted.mkdir(parents=True, exist_ok=True)
    artifact_records = []
    for source in sorted(checkpoints.glob("epoch_*.pt")):
        target = persisted / source.name
        shutil.copyfile(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        artifact_records.append(
            {
                "locator": f"MODAL_VOLUME/{persisted.name}/{source.name}",
                "sha256": digest,
                "size_bytes": target.stat().st_size,
            }
        )
    OUTPUT_VOLUME.commit()
    total_wall = time.perf_counter() - started
    return {
        "runner_commit": RUNNER_COMMIT,
        "gpu": {
            "name": gpu_name,
            "peak_cuda_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
            ),
            "peak_cuda_reserved_mb": (
                torch.cuda.max_memory_reserved() / 1024**2
            ),
        },
        "runtime": {
            "remote_total_wall_time_s": total_wall,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            ),
        },
        "input_verification": remote_inputs,
        "checkpoint_artifacts": artifact_records,
        "runner_report": report,
        "runner_markdown": report_markdown.read_text(),
    }


@app.local_entrypoint()
def main(
    raw_dir: str,
    processed_dir: str,
    caption_cache: str,
    caption_metadata: str,
    report_json: str = "reports/phase_b6a/sasrec_full_modal_l4.json",
    report_markdown: str = "reports/phase_b6a/sasrec_full_modal_l4.md",
) -> None:
    if not _runner_inputs_match_commit():
        raise RuntimeError("Modal runner inputs differ from RUNNER_COMMIT")
    if _git_output("status", "--porcelain"):
        raise RuntimeError("Modal SASRec run requires a clean worktree")
    files = build_input_allowlist(
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        caption_cache=Path(caption_cache),
        caption_metadata=Path(caption_metadata),
    )
    manifest = input_bundle_manifest(files)
    _verify_existing_bundle(files, manifest)
    remote = run_l4.remote(
        bundle_sha256=manifest["bundle_sha256"],
        input_manifest=manifest,
    )
    result = {
        "phase": "phase-b6a-recbole-sasrec-modal-l4",
        "runner_commit": RUNNER_COMMIT,
        "input_manifest": manifest,
        "remote": remote,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("Modal result contains a host path or secret")
    json_path = REPOSITORY_ROOT / report_json
    markdown_path = REPOSITORY_ROOT / report_markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialized + "\n")
    markdown_path.write_text(
        remote["runner_markdown"]
        + "\n## Modal L4 resources\n\n"
        + f"- GPU: `{remote['gpu']['name']}`\n"
        + f"- Remote wall time: "
        + f"`{remote['runtime']['remote_total_wall_time_s']:.2f}s`\n"
        + f"- Peak CUDA allocated/reserved: "
        + f"`{remote['gpu']['peak_cuda_allocated_mb']:.1f}/"
        + f"{remote['gpu']['peak_cuda_reserved_mb']:.1f} MB`\n"
        + f"- Peak RSS: `{remote['runtime']['peak_rss_mb']:.1f} MB`\n"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "selected_epoch": remote["runner_report"]["selected_epoch"],
                "report_json": report_json,
                "report_markdown": report_markdown,
            },
            sort_keys=True,
        ),
        flush=True,
    )
