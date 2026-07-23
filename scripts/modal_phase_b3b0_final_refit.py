#!/usr/bin/env python3
"""Run the frozen Big train+validation final refit on one Modal L4."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modal_preflight_helpers import (  # noqa: E402
    build_input_allowlist,
    input_bundle_manifest,
    modal_volume_file_paths,
)

B3B_RUNNER_COMMIT = "9f1c3222dc111227e64d57c88f5e792ce23c5529"
REPOSITORY_URL = (
    "https://github.com/Rockyeast/short-video-two-tower-recommender.git"
)
REPOSITORY_DIR = Path("/opt/short-video-two-tower-recommender")
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
OUTPUT_VOLUME_NAME = "kuairec-b3b-final-refit-artifacts"
INPUT_MOUNT = Path("/inputs")
OUTPUT_MOUNT = Path("/outputs")
EXPERIMENT_ROOT = OUTPUT_MOUNT / "phase-b3b-final-v1"

app = modal.App("short-video-two-tower-b3b-final-refit")
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "modal==1.4.1",
        "numpy==2.2.6",
        "pandas==2.3.3",
        "PyYAML==6.0.3",
        "scipy==1.16.3",
        "torch==2.11.0",
        "sentence-transformers==5.6.0",
    )
    .run_commands(
        f"git clone --filter=blob:none {REPOSITORY_URL} {REPOSITORY_DIR}",
        f"cd {REPOSITORY_DIR} && git checkout --detach {B3B_RUNNER_COMMIT}",
        (
            f"cd {REPOSITORY_DIR} && "
            f'test "$(git rev-parse HEAD)" = "{B3B_RUNNER_COMMIT}"'
        ),
    )
    .add_local_file(
        Path(__file__),
        "/opt/modal-wrapper/modal_phase_b3b0_final_refit.py",
        copy=True,
    )
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def _read_volume_file(path: str) -> bytes:
    return b"".join(input_volume.read_file(path))


def _prepare_volume(files, manifest: dict[str, Any]) -> dict[str, Any]:
    version_root = f"bundles/{manifest['bundle_sha256']}"
    manifest_path = f"{version_root}/allowlist.json"
    try:
        entries = input_volume.listdir(version_root, recursive=True)
    except Exception as exc:
        if exc.__class__.__name__ not in {"NotFoundError", "GRPCError"}:
            raise
        entries = []
    if entries:
        expected = {
            f"{version_root}/{record.logical_path}" for record in files
        } | {manifest_path}
        if modal_volume_file_paths(entries) != expected:
            raise RuntimeError("Existing Modal input bundle is incomplete")
        if json.loads(_read_volume_file(manifest_path)) != manifest:
            raise RuntimeError("Existing Modal input bundle identity changed")
        return {"version_root": version_root, "bundle_reused": True}
    manifest_bytes = io.BytesIO(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    with input_volume.batch_upload(force=False) as upload:
        for record in files:
            upload.put_file(
                record.local_path, f"{version_root}/{record.logical_path}"
            )
        upload.put_file(manifest_bytes, manifest_path)
    return {"version_root": version_root, "bundle_reused": False}


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=14400,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT: input_volume.read_only(),
        OUTPUT_MOUNT: output_volume,
    },
    include_source=False,
    serialized=True,
)
def run_refit(*, bundle_sha256: str, manifest: dict[str, Any]) -> dict[str, Any]:
    import os
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Final refit requires exactly one CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
        ).strip()
        != B3B_RUNNER_COMMIT
    ):
        raise RuntimeError("Final refit runner commit changed")
    bundle = INPUT_MOUNT / "bundles" / bundle_sha256
    if json.loads((bundle / "allowlist.json").read_text()) != manifest:
        raise RuntimeError("Mounted input manifest changed")
    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    cache = EXPERIMENT_ROOT / "caption_embeddings.npz"
    cache_metadata = EXPERIMENT_ROOT / "caption_cache_metadata.json"
    report_json = EXPERIMENT_ROOT / "final_refit.json"
    report_markdown = EXPERIMENT_ROOT / "final_refit.md"
    artifacts = EXPERIMENT_ROOT / "artifacts"
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(REPOSITORY_DIR)
        from scripts.precompute_phase_b3b_captions import run as precompute
        from scripts.run_phase_b3b_final_refit import run as runner

        caption_report = precompute(
            repo_root=REPOSITORY_DIR,
            data_dir=bundle / "raw",
            artifact_dir=bundle / "processed",
            cache_path=cache,
            metadata_path=cache_metadata,
        )
        refit_report = runner(
            repo_root=REPOSITORY_DIR,
            data_dir=bundle / "raw",
            artifact_dir=bundle / "processed",
            config_path=REPOSITORY_DIR
            / "configs"
            / "phase_b3b_final_recipe.yaml",
            caption_cache_path=cache,
            caption_metadata_path=cache_metadata,
            output_dir=artifacts,
            report_json=report_json,
            report_markdown=report_markdown,
            device="cuda:0",
        )
    finally:
        os.chdir(previous)
    output_volume.commit()
    return {
        "gpu": gpu_name,
        "caption": caption_report,
        "refit": refit_report,
        "output_volume": OUTPUT_VOLUME_NAME,
        "output_root": "phase-b3b-final-v1",
        "checkpoint_files": [
            "phase-b3b-final-v1/artifacts/final_bpr_epoch_020.npz",
            "phase-b3b-final-v1/artifacts/final_two_tower_epoch_001.pt",
        ],
    }


@app.local_entrypoint()
def main(
    raw_dir: str,
    processed_dir: str,
    caption_cache: str,
    caption_metadata: str,
    report_json: str = "reports/phase_b3b0/final_refit_modal_l4.json",
) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Modal final-refit wrapper requires a clean worktree")
    files = build_input_allowlist(
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        caption_cache=Path(caption_cache),
        caption_metadata=Path(caption_metadata),
    )
    manifest = input_bundle_manifest(files)
    volume = _prepare_volume(files, manifest)
    remote = run_refit.remote(
        bundle_sha256=manifest["bundle_sha256"], manifest=manifest
    )
    report = {
        "phase": "phase-b3b0-final-refit-modal-l4",
        "runner_commit": B3B_RUNNER_COMMIT,
        "wrapper_commit": _git("rev-parse", "HEAD"),
        "input_bundle": {
            "sha256": manifest["bundle_sha256"],
            **volume,
        },
        "remote": remote,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("Final refit report contains a host path or secret")
    target = (ROOT / report_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized + "\n")
    print(json.dumps({"status": "completed", "report": report_json}))
