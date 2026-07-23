#!/usr/bin/env python3
"""Run exactly one frozen Phase B2B preflight on one Modal NVIDIA L4."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modal_preflight_helpers import (
    build_input_allowlist,
    cpu_gpu_differences,
    input_bundle_manifest,
    validate_gpu_preflight_report,
    verify_remote_inputs,
)

MODULE_IMPORT_PERF_S = time.perf_counter()
B2B_RUNNER_COMMIT = "0361b648908acd90a134f379bd39335bcf18d518"
REPOSITORY_URL = (
    "https://github.com/Rockyeast/short-video-two-tower-recommender.git"
)
REPOSITORY_DIR = Path("/opt/short-video-two-tower-recommender")
WRAPPER_REMOTE_DIR = "/opt/modal-wrapper"
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
INPUT_MOUNT = Path("/inputs")
FORMAL_STEP_COUNT = 6729

app = modal.App("short-video-two-tower-b2b-l4-preflight")
input_volume = modal.Volume.from_name(
    INPUT_VOLUME_NAME, create_if_missing=True
)
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
    )
    .run_commands(
        f"git clone --filter=blob:none {REPOSITORY_URL} {REPOSITORY_DIR}",
        f"cd {REPOSITORY_DIR} && git checkout --detach {B2B_RUNNER_COMMIT}",
        (
            f"cd {REPOSITORY_DIR} && "
            f'test "$(git rev-parse HEAD)" = "{B2B_RUNNER_COMMIT}"'
        ),
        f"cd {REPOSITORY_DIR} && test -z \"$(git status --porcelain)\"",
        f"python -m pip install --no-deps {REPOSITORY_DIR}",
    )
    .add_local_file(
        Path(__file__),
        f"{WRAPPER_REMOTE_DIR}/modal_phase_b2b_preflight.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).resolve().parent / "modal_preflight_helpers.py",
        f"{WRAPPER_REMOTE_DIR}/modal_preflight_helpers.py",
        copy=True,
    )
    .env({"PYTHONPATH": WRAPPER_REMOTE_DIR})
)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def _read_volume_file(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _prepare_volume(
    *,
    files,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    version_root = f"bundles/{manifest['bundle_sha256']}"
    manifest_path = f"{version_root}/allowlist.json"
    try:
        entries = input_volume.listdir(version_root, recursive=True)
    except Exception as exc:
        if exc.__class__.__name__ not in {"NotFoundError", "GRPCError"}:
            raise
        entries = []
    if entries:
        actual_paths = {entry.path.lstrip("/") for entry in entries}
        expected_paths = {
            f"{version_root}/{record.logical_path}" for record in files
        } | {manifest_path}
        if actual_paths != expected_paths:
            raise RuntimeError(
                "Existing Modal bundle is partial or has unexpected files"
            )
        remote_manifest = json.loads(
            _read_volume_file(input_volume, manifest_path)
        )
        if remote_manifest != manifest:
            raise RuntimeError(
                "Existing Modal bundle manifest does not match local inputs"
            )
        return {
            "volume_name": INPUT_VOLUME_NAME,
            "version_root": version_root,
            "bundle_reused": True,
        }
    manifest_bytes = io.BytesIO(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    with input_volume.batch_upload(force=False) as upload:
        for record in files:
            upload.put_file(
                record.local_path,
                f"{version_root}/{record.logical_path}",
            )
        upload.put_file(manifest_bytes, manifest_path)
    return {
        "volume_name": INPUT_VOLUME_NAME,
        "version_root": version_root,
        "bundle_reused": False,
    }


def _checkpoint_monitor(
    checkpoint_dir: Path,
    stop: threading.Event,
    observations: dict[int, dict[str, float | None]],
) -> None:
    while not stop.is_set():
        now = time.perf_counter()
        for epoch in (1, 2):
            record = observations.setdefault(
                epoch, {"temporary_first_seen": None, "published": None}
            )
            if record["temporary_first_seen"] is None and any(
                checkpoint_dir.glob(f"epoch_{epoch:03d}.pt.tmp.*")
            ):
                record["temporary_first_seen"] = now
            if (
                record["published"] is None
                and (checkpoint_dir / f"epoch_{epoch:03d}.pt").is_file()
            ):
                record["published"] = now
        stop.wait(0.005)


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=1200,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    max_inputs=1,
    volumes={INPUT_MOUNT: input_volume.read_only()},
    block_network=True,
    restrict_modal_access=True,
    include_source=False,
    serialized=True,
)
def run_l4_preflight(
    *,
    bundle_sha256: str,
    input_manifest: dict[str, Any],
    wrapper_commit: str,
    request_epoch_s: float,
) -> dict[str, Any]:
    remote_started_epoch_s = time.time()
    remote_started = time.perf_counter()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The preflight requires exactly one visible GPU")
    sys.path.insert(0, str(REPOSITORY_DIR))
    from kuairec_fully_observed.torch_training import (
        resolve_concrete_device,
    )

    concrete_device = resolve_concrete_device("cuda")
    if concrete_device != torch.device("cuda:0"):
        raise RuntimeError(f"Expected cuda:0, got {concrete_device}")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")

    repo_head = _git_output(REPOSITORY_DIR, "rev-parse", "HEAD")
    repo_status = _git_output(REPOSITORY_DIR, "status", "--porcelain")
    if repo_head != B2B_RUNNER_COMMIT or repo_status:
        raise RuntimeError("Container repository identity is not frozen/clean")

    bundle_root = INPUT_MOUNT / "bundles" / bundle_sha256
    stored_manifest = json.loads(
        (bundle_root / "allowlist.json").read_text()
    )
    if stored_manifest != input_manifest:
        raise RuntimeError("Mounted Modal input manifest changed")
    read_only_enforced = False
    try:
        (bundle_root / "__write_probe__").write_text("forbidden")
    except OSError:
        read_only_enforced = True
    if not read_only_enforced:
        raise RuntimeError("Modal input Volume is not mounted read-only")
    input_verification_started = time.perf_counter()
    remote_inputs = verify_remote_inputs(bundle_root, input_manifest)
    input_verification_s = time.perf_counter() - input_verification_started

    output_root = Path("/tmp/phase_b2b_modal_l4")
    checkpoint_dir = output_root / "checkpoints"
    report_json = output_root / "runner_preflight.json"
    report_markdown = output_root / "runner_preflight.md"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    observations: dict[int, dict[str, float | None]] = {}
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_checkpoint_monitor,
        args=(checkpoint_dir, stop_monitor, observations),
        daemon=True,
    )
    monitor.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner_started = time.perf_counter()
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPOSITORY_DIR)
        sys.argv = [
            "run_phase_b2b_full_two_tower.py",
            "--data-dir",
            str(bundle_root / "raw"),
            "--processed-artifact-dir",
            str(bundle_root / "processed"),
            "--caption-cache",
            str(bundle_root / "caption" / "caption_embeddings.npz"),
            "--caption-metadata",
            str(
                bundle_root
                / "caption"
                / "caption_cache_metadata.json"
            ),
            "--preflight",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--report-json",
            str(report_json),
            "--report-markdown",
            str(report_markdown),
        ]
        from scripts.run_phase_b2b_full_two_tower import main as runner_main

        runner_main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        stop_monitor.set()
        monitor.join(timeout=2)
    runner_wall_s = time.perf_counter() - runner_started

    report = json.loads(report_json.read_text())
    cpu_report = json.loads(
        (
            REPOSITORY_DIR
            / "reports"
            / "phase_b2b0"
            / "runner_preflight.json"
        ).read_text()
    )
    validate_gpu_preflight_report(report, cpu_report=cpu_report)
    if report["input_provenance"]["code_commit_at_run"] != B2B_RUNNER_COMMIT:
        raise RuntimeError("Runner report code identity changed")

    save_observations: dict[str, Any] = {}
    observed_save_total_s = 0.0
    for epoch, observation in sorted(observations.items()):
        started = observation["temporary_first_seen"]
        published = observation["published"]
        duration = (
            None
            if started is None or published is None
            else max(0.0, published - started)
        )
        if duration is not None:
            observed_save_total_s += duration
        save_observations[str(epoch)] = {
            **observation,
            "observed_save_s": duration,
            "poll_interval_s": 0.005,
        }

    checkpoint_records = report["checkpoints"]
    exact_retrieval_s = sum(
        record["timings_s"]["exact_retrieval_and_metrics_s"]
        for record in checkpoint_records
    )
    full_validation_s = sum(
        sum(record["timings_s"].values())
        for record in checkpoint_records
    )
    stage_s = report["training"][
        "current_process_training_and_epoch_validation_s"
    ]
    checkpoint_loading_s = report["timings_s"]["checkpoint_loading_s"]
    estimated_training_s = max(
        0.0,
        stage_s
        - full_validation_s
        - checkpoint_loading_s
        - observed_save_total_s,
    )
    process_steps = report["training"]["process_statistics"][
        "optimizer_steps"
    ]
    seconds_per_step = estimated_training_s / process_steps
    raw_linear_eta_s = seconds_per_step * FORMAL_STEP_COUNT
    runner_data_preparation_s = max(
        0.0,
        report["runtime_s"]
        - stage_s
        - report["timings_s"]["report_generation_s"],
    )
    pip_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ).splitlines()
    pip_freeze_sha256 = hashlib.sha256(
        ("\n".join(pip_freeze) + "\n").encode()
    ).hexdigest()
    nvidia_smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    remote_result = {
        "status": "passed",
        "resource_contract": {
            "gpu_request": "L4",
            "visible_gpu_count": torch.cuda.device_count(),
            "resolved_device": str(concrete_device),
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "cpu_memory_limit_mib": 16384,
            "function_timeout_s": 1200,
            "startup_timeout_s": 600,
            "retries": 0,
            "single_use_container": True,
            "input_volume_read_only": read_only_enforced,
        },
        "source_identity": {
            "repository": REPOSITORY_URL,
            "b2b_runner_commit": repo_head,
            "wrapper_commit": wrapper_commit,
            "repository_status_clean_at_start": True,
        },
        "versions": {
            "modal_sdk": importlib.metadata.version("modal"),
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_smi": nvidia_smi,
            "pip_freeze_sha256": pip_freeze_sha256,
            "pip_freeze": pip_freeze,
        },
        "inputs": remote_inputs,
        "timings_s": {
            "container_startup": max(
                0.0, remote_started_epoch_s - request_epoch_s
            ),
            "remote_input_sha_verification": input_verification_s,
            "runner_data_preparation": runner_data_preparation_s,
            "estimated_training_compute": estimated_training_s,
            "seconds_per_optimizer_step": seconds_per_step,
            "checkpoint_save_observed_total": observed_save_total_s,
            "checkpoint_loading": checkpoint_loading_s,
            "exact_retrieval": exact_retrieval_s,
            "runner_wall": runner_wall_s,
            "remote_function_wall": time.perf_counter() - remote_started,
            "raw_linear_eta_6729_steps": raw_linear_eta_s,
        },
        "eta_scope": (
            "Raw linear training-only estimate for 6,729 optimizer steps; "
            "excludes full-data preparation and fixed validation overhead."
        ),
        "checkpoint_save_observations": save_observations,
        "memory": {
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_mb": report["peak_rss_mb"],
        },
        "runner_report": report,
        "runner_markdown": report_markdown.read_text(),
        "cpu_gpu_preflight_differences": cpu_gpu_differences(
            report, cpu_report
        ),
    }
    return remote_result


def _render_markdown(report: dict[str, Any]) -> str:
    remote = report["remote"]
    runner = remote["runner_report"]
    selected = next(
        record
        for record in runner["checkpoints"]
        if record["epoch"] == runner["selected_epoch_by_frozen_rule"]
    )
    metrics = selected["validation"]["metrics"]
    timing = remote["timings_s"]
    memory = remote["memory"]
    return "\n".join(
        [
            "# Phase B2B Modal NVIDIA L4 Preflight",
            "",
            "This is the single allowed 20-step GPU preflight. It is not a "
            "formal full-training or effectiveness run.",
            "",
            f"- Status: `{remote['status']}`",
            f"- B2B runner commit: "
            f"`{remote['source_identity']['b2b_runner_commit']}`",
            f"- Modal wrapper commit: "
            f"`{remote['source_identity']['wrapper_commit']}`",
            f"- GPU: `{remote['resource_contract']['gpu_name']}`",
            f"- Device: `{remote['resource_contract']['resolved_device']}`",
            f"- PyTorch/CUDA: `{remote['versions']['pytorch']}` / "
            f"`{remote['versions']['cuda_runtime']}`",
            f"- Modal SDK: `{remote['versions']['modal_sdk']}`",
            f"- Input bundle: `{remote['inputs']['bundle_sha256']}`",
            f"- Input size: `{remote['inputs']['total_size_bytes']}` bytes",
            "",
            "## Result",
            "",
            f"- Examples: `{runner['training']['example_count']}`",
            f"- Optimizer steps: "
            f"`{runner['training']['process_statistics']['optimizer_steps']}`",
            f"- Skipped batches: "
            f"`{runner['training']['process_statistics']['skipped_batches']}`",
            f"- Loss: `{runner['training']['epoch_losses'][0]:.6f}` → "
            f"`{runner['training']['epoch_losses'][1]:.6f}`",
            f"- Recall@100: `{metrics['Recall@100']:.6f}`",
            f"- NDCG@20: `{metrics['NDCG@20']:.6f}`",
            f"- Coverage@100: `{metrics['Coverage@100']:.6f}`",
            "",
            "## Timing and memory",
            "",
            f"- Modal initialization/image build: "
            f"`{report['local']['modal_initialization_and_image_build_s']:.3f} s`",
            f"- Local input preparation/upload: "
            f"`{report['local']['input_preparation_and_upload_s']:.3f} s`",
            f"- Container startup: `{timing['container_startup']:.3f} s`",
            f"- Runner data preparation: "
            f"`{timing['runner_data_preparation']:.3f} s`",
            f"- Estimated training compute: "
            f"`{timing['estimated_training_compute']:.3f} s`",
            f"- Seconds/optimizer step: "
            f"`{timing['seconds_per_optimizer_step']:.6f}`",
            f"- Checkpoint save/load: "
            f"`{timing['checkpoint_save_observed_total']:.3f} / "
            f"{timing['checkpoint_loading']:.3f} s`",
            f"- Exact Retrieval: `{timing['exact_retrieval']:.3f} s`",
            f"- Remote function wall: "
            f"`{timing['remote_function_wall']:.3f} s`",
            f"- Raw 6,729-step linear ETA: "
            f"`{timing['raw_linear_eta_6729_steps'] / 60:.2f} min`",
            f"- Peak CUDA allocated/reserved: "
            f"`{memory['peak_cuda_allocated_bytes'] / 1024**2:.2f} / "
            f"{memory['peak_cuda_reserved_bytes'] / 1024**2:.2f} MiB`",
            f"- Peak RSS: `{memory['peak_rss_mb']:.2f} MiB`",
            "",
            remote["eta_scope"],
            "",
            "CPU/GPU values are not required to be bitwise identical, and no "
            "parameter was changed based on their differences.",
            "",
            "```text",
            "formal_gate_executed=false",
            "effectiveness_claim=false",
            "full_big_train=false",
            "full_big_validation=false",
            "```",
            "",
            "Small Matrix, temporal final, FAISS and Hybrid were not run.",
            "",
        ]
    )


@app.local_entrypoint()
def main(
    raw_dir: str,
    processed_dir: str,
    caption_cache: str,
    caption_metadata: str,
    report_json: str = "reports/phase_b2b0/modal_l4_preflight.json",
    report_markdown: str = "reports/phase_b2b0/modal_l4_preflight.md",
) -> None:
    local_started = time.perf_counter()
    repository_root = REPOSITORY_ROOT
    wrapper_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if _git_output(repository_root, "status", "--porcelain"):
        raise RuntimeError("Modal wrapper must run from a clean commit")
    files = build_input_allowlist(
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        caption_cache=Path(caption_cache),
        caption_metadata=Path(caption_metadata),
    )
    manifest = input_bundle_manifest(files)
    print(
        json.dumps(
            {
                "input_allowlist": manifest["files"],
                "total_size_bytes": manifest["total_size_bytes"],
                "bundle_sha256": manifest["bundle_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    volume_record = _prepare_volume(files=files, manifest=manifest)
    input_preparation_and_upload_s = time.perf_counter() - local_started
    request_epoch_s = time.time()
    remote = run_l4_preflight.remote(
        bundle_sha256=manifest["bundle_sha256"],
        input_manifest=manifest,
        wrapper_commit=wrapper_commit,
        request_epoch_s=request_epoch_s,
    )
    local_wall_s = time.perf_counter() - local_started
    report = {
        "phase": "phase-b2b-modal-l4-preflight",
        "status": "passed",
        "b2b_runner_commit": B2B_RUNNER_COMMIT,
        "wrapper_commit": wrapper_commit,
        "modal_sdk_local": modal.__version__,
        "volume": volume_record,
        "input_manifest": manifest,
        "local": {
            "input_preparation_and_upload_s": (
                input_preparation_and_upload_s
            ),
            "local_entrypoint_wall_s": local_wall_s,
            "modal_initialization_and_image_build_s": (
                local_started - MODULE_IMPORT_PERF_S
            ),
            "image_build_timing_scope": (
                "Local module initialization through Modal local-entrypoint "
                "start; includes image resolution/build and Modal app startup."
            ),
            "remote_function_calls": 1,
        },
        "remote": remote,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if "/home/" in serialized:
        raise RuntimeError("Generated Modal report contains a host path")
    json_path = (repository_root / report_json).resolve()
    markdown_path = (repository_root / report_markdown).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialized + "\n")
    markdown_path.write_text(_render_markdown(report))
    print(
        json.dumps(
            {
                "status": "passed",
                "report_json": report_json,
                "report_markdown": report_markdown,
                "gpu": remote["resource_contract"]["gpu_name"],
                "optimizer_steps": remote["runner_report"]["training"][
                    "process_statistics"
                ]["optimizer_steps"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
