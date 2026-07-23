#!/usr/bin/env python3
"""Run the frozen Phase B2B full Two-Tower experiment on one Modal L4."""

from __future__ import annotations

import importlib.metadata
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

from modal_preflight_helpers import (  # noqa: E402
    build_input_allowlist,
    input_bundle_manifest,
    modal_volume_file_paths,
    verify_remote_inputs,
)

B2B_RUNNER_COMMIT = "7feb5675b7fa6577c68a3775d943c0a32b94f603"
REPOSITORY_URL = (
    "https://github.com/Rockyeast/short-video-two-tower-recommender.git"
)
REPOSITORY_DIR = Path("/opt/short-video-two-tower-recommender")
WRAPPER_REMOTE_DIR = "/opt/modal-wrapper"
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
OUTPUT_VOLUME_NAME = "kuairec-b2b-full-run-artifacts"
INPUT_MOUNT = Path("/inputs")
OUTPUT_MOUNT = Path("/outputs")
EXPERIMENT_ROOT = OUTPUT_MOUNT / "phase-b2b-full-v1"
FORMAL_STEP_COUNT = 6729
FROZEN_POPULARITY = {
    "Recall@100": 0.036642843159775826,
    "NDCG@20": 0.01061519019112034,
    "Coverage@100": 0.0800854244527496,
    "Data-Cold Recall@100": 0.0,
}

app = modal.App("short-video-two-tower-b2b-l4-full-run")
input_volume = modal.Volume.from_name(
    INPUT_VOLUME_NAME, create_if_missing=False
)
output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME, create_if_missing=True
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
    )
    .add_local_file(
        Path(__file__),
        f"{WRAPPER_REMOTE_DIR}/modal_phase_b2b_full_run.py",
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


def _verify_existing_input_bundle(
    *, files, manifest: dict[str, Any]
) -> dict[str, Any]:
    version_root = f"bundles/{manifest['bundle_sha256']}"
    manifest_path = f"{version_root}/allowlist.json"
    entries = input_volume.listdir(version_root, recursive=True)
    actual_paths = modal_volume_file_paths(entries)
    expected_paths = {
        f"{version_root}/{record.logical_path}" for record in files
    } | {manifest_path}
    if actual_paths != expected_paths:
        raise RuntimeError("Frozen Modal input bundle is incomplete")
    if json.loads(_read_volume_file(input_volume, manifest_path)) != manifest:
        raise RuntimeError("Frozen Modal input manifest changed")
    return {
        "volume_name": INPUT_VOLUME_NAME,
        "version_root": version_root,
        "bundle_reused": True,
    }


def _existing_complete_epochs(checkpoint_dir: Path) -> tuple[int, ...]:
    epochs = []
    for epoch in (1, 2, 3):
        if (checkpoint_dir / f"epoch_{epoch:03d}.pt").is_file():
            epochs.append(epoch)
    if epochs not in ([], [1], [1, 2], [1, 2, 3]):
        raise RuntimeError("Persisted checkpoints are not a valid prefix")
    return tuple(epochs)


def _checkpoint_monitor(
    checkpoint_dir: Path,
    stop: threading.Event,
    already_committed: set[int],
    observations: dict[int, dict[str, Any]],
    run_started: float,
) -> None:
    while not stop.is_set():
        for epoch in (1, 2, 3):
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            if epoch in already_committed or not path.is_file():
                continue
            seen = time.perf_counter()
            output_volume.commit()
            committed = time.perf_counter()
            already_committed.add(epoch)
            observations[epoch] = {
                "published_elapsed_s": seen - run_started,
                "volume_commit_s": committed - seen,
                "volume_committed": True,
            }
            print(
                f"checkpoint_saved epoch={epoch} "
                f"elapsed_s={seen - run_started:.3f} "
                f"volume_commit_s={committed - seen:.3f}",
                flush=True,
            )
        stop.wait(0.25)


def _validate_full_report(report: dict[str, Any]) -> None:
    claims = {
        "formal_gate_executed": True,
        "effectiveness_claim": False,
        "full_big_train": True,
        "full_big_validation": True,
    }
    training = report["training"]
    validation = report["validation"]
    checks = {
        "phase": report["phase"] == "phase-b2b-full-two-tower",
        "status": report["status"] == "completed",
        "claims": report["claim_boundary"] == claims,
        "examples": training["example_count"] == 574098,
        "epochs": training["completed_epoch"] == 3,
        "steps": (
            training["cumulative_statistics"]["optimizer_steps"]
            == FORMAL_STEP_COUNT
        ),
        "queries": validation["evaluated_queries"] == 6818,
        "targets": validation["evaluated_targets"] == 118565,
        "catalog": validation["fixed_catalog_count"] == 9365,
        "checkpoints": [
            row["epoch"] for row in report["checkpoints"]
        ]
        == [1, 2, 3],
        "selected_epoch": report["selected_epoch_by_frozen_rule"]
        in (1, 2, 3),
        "access": report["access"]
        == {
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "faiss_run": False,
            "hybrid_run": False,
            "full_training_started": True,
        },
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Formal B2B report contract failed: {failed}")


def _metric_comparison(
    selected_metrics: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    result = {}
    for name in (
        "Recall@100",
        "NDCG@20",
        "Coverage@100",
        "Data-Cold Recall@100",
    ):
        actual = float(selected_metrics[name])
        reference = float(baseline[name])
        result[name] = {
            "two_tower": actual,
            "baseline": reference,
            "absolute_difference": actual - reference,
            "relative_difference": (
                None
                if reference == 0
                else (actual - reference) / reference
            ),
        }
    return result


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=14400,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT: input_volume.read_only(),
        OUTPUT_MOUNT: output_volume,
    },
    block_network=True,
    include_source=False,
    serialized=True,
)
def run_l4_full(
    *,
    bundle_sha256: str,
    input_manifest: dict[str, Any],
    wrapper_commit: str,
    request_epoch_s: float,
) -> dict[str, Any]:
    remote_started_epoch_s = time.time()
    remote_started = time.perf_counter()
    import resource
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required")
    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    from kuairec_fully_observed.torch_training import (
        resolve_concrete_device,
    )

    device = resolve_concrete_device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if device != torch.device("cuda:0") or "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4 cuda:0, got {gpu_name} {device}")
    if (
        _git_output(REPOSITORY_DIR, "rev-parse", "HEAD")
        != B2B_RUNNER_COMMIT
        or _git_output(REPOSITORY_DIR, "status", "--porcelain")
    ):
        raise RuntimeError("Frozen runner checkout identity changed")

    bundle_root = INPUT_MOUNT / "bundles" / bundle_sha256
    stored = json.loads((bundle_root / "allowlist.json").read_text())
    if stored != input_manifest:
        raise RuntimeError("Mounted input manifest changed")
    input_started = time.perf_counter()
    remote_inputs = verify_remote_inputs(bundle_root, input_manifest)
    input_verification_s = time.perf_counter() - input_started

    checkpoint_dir = EXPERIMENT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_epochs = _existing_complete_epochs(checkpoint_dir)
    resume_checkpoint = (
        None
        if not existing_epochs
        else checkpoint_dir / f"epoch_{existing_epochs[-1]:03d}.pt"
    )
    report_json = Path("/tmp/phase_b2b_full.json")
    report_markdown = Path("/tmp/phase_b2b_full.md")
    observations: dict[int, dict[str, Any]] = {}
    stop_monitor = threading.Event()
    run_started = time.perf_counter()
    monitor = threading.Thread(
        target=_checkpoint_monitor,
        args=(
            checkpoint_dir,
            stop_monitor,
            set(existing_epochs),
            observations,
            run_started,
        ),
        daemon=True,
    )
    monitor.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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
            str(bundle_root / "caption" / "caption_cache_metadata.json"),
            "--full-run",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--report-json",
            str(report_json),
            "--report-markdown",
            str(report_markdown),
        ]
        if resume_checkpoint is not None:
            sys.argv.extend(
                ["--resume-checkpoint", str(resume_checkpoint)]
            )
        from scripts.run_phase_b2b_full_two_tower import main as runner_main

        runner_main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        stop_monitor.set()
        monitor.join(timeout=120)
    output_volume.commit()

    report = json.loads(report_json.read_text())
    _validate_full_report(report)
    if report["input_provenance"]["code_commit_at_run"] != B2B_RUNNER_COMMIT:
        raise RuntimeError("Formal report runner identity changed")
    b1a = json.loads(
        (
            REPOSITORY_DIR
            / "reports"
            / "phase_b1a"
            / "full_bpr_pilot.json"
        ).read_text()
    )
    frozen_popularity_actual = {
        name: b1a["baselines"]["global_popularity"]["metrics"][name]
        for name in FROZEN_POPULARITY
    }
    if frozen_popularity_actual != FROZEN_POPULARITY:
        raise RuntimeError("Frozen Popularity baseline changed")
    selected = next(
        row
        for row in report["checkpoints"]
        if row["epoch"] == report["selected_epoch_by_frozen_rule"]
    )
    selected_metrics = selected["validation"]["metrics"]
    comparisons = {
        "global_popularity": _metric_comparison(
            selected_metrics, FROZEN_POPULARITY
        ),
        "bpr_epoch_20": _metric_comparison(
            selected_metrics, report["frozen_bpr_epoch_20"]
        ),
    }
    return {
        "status": "passed",
        "source_identity": {
            "repository": REPOSITORY_URL,
            "b2b_runner_commit": B2B_RUNNER_COMMIT,
            "wrapper_commit": wrapper_commit,
            "repository_status_clean_at_start": True,
        },
        "resource_contract": {
            "gpu_request": "L4",
            "gpu_name": gpu_name,
            "resolved_device": str(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "function_timeout_s": 14400,
            "startup_timeout_s": 600,
            "retries": 0,
            "single_use_container": True,
            "cpu_memory_limit_mib": 16384,
        },
        "versions": {
            "modal_sdk": importlib.metadata.version("modal"),
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "inputs": remote_inputs,
        "checkpoint_persistence": {
            "volume_name": OUTPUT_VOLUME_NAME,
            "experiment_root": "phase-b2b-full-v1",
            "existing_complete_epochs_at_start": list(existing_epochs),
            "resumed_from": (
                None
                if resume_checkpoint is None
                else resume_checkpoint.name
            ),
            "observations": observations,
            "final_complete_epochs": list(
                _existing_complete_epochs(checkpoint_dir)
            ),
        },
        "timings_s": {
            "container_startup": max(
                0.0, remote_started_epoch_s - request_epoch_s
            ),
            "remote_input_sha_verification": input_verification_s,
            "runner_wall": report["runtime_s"],
            "training_and_epoch_validation": report["training"][
                "current_process_training_and_epoch_validation_s"
            ],
            "checkpoint_loading": report["timings_s"][
                "checkpoint_loading_s"
            ],
            "checkpoint_reevaluation": report["timings_s"][
                "checkpoint_reevaluation_s"
            ],
            "remote_function_wall": time.perf_counter() - remote_started,
        },
        "memory": {
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        },
        "comparisons": comparisons,
        "runner_report": report,
        "runner_markdown": report_markdown.read_text(),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    remote = result["remote"]
    report = remote["runner_report"]
    checkpoints = report["checkpoints"]
    selected_epoch = report["selected_epoch_by_frozen_rule"]
    selected = next(row for row in checkpoints if row["epoch"] == selected_epoch)
    metrics = selected["validation"]["metrics"]
    lines = [
        "# Phase B2B Full Two-Tower — Modal L4",
        "",
        f"- Runner commit: `{result['b2b_runner_commit']}`",
        f"- Wrapper commit at run: `{result['wrapper_commit']}`",
        f"- GPU/device: `{remote['resource_contract']['gpu_name']}` / "
        f"`{remote['resource_contract']['resolved_device']}`",
        f"- Execution mode: `{report['execution_mode']}`",
        f"- Selected epoch: `{selected_epoch}`",
        "",
        "## Epoch results",
        "",
        "| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 | "
        "Data-Cold Recall@100 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoints:
        m = row["validation"]["metrics"]
        lines.append(
            f"| {row['epoch']} | {row['epoch_loss']:.6f} | "
            f"{m['Recall@100']:.6f} | {m['NDCG@20']:.6f} | "
            f"{m['Coverage@100']:.6f} | "
            f"{m['Data-Cold Recall@100']:.6f} |"
        )
    lines += [
        "",
        "## Selected result",
        "",
        f"- Recall@100: `{metrics['Recall@100']:.9f}`",
        f"- NDCG@20: `{metrics['NDCG@20']:.9f}`",
        f"- Coverage@100: `{metrics['Coverage@100']:.9f}`",
        f"- Data-Cold Recall@100: "
        f"`{metrics['Data-Cold Recall@100']:.9f}`",
        f"- Gate A/B/C: `{json.dumps(report['formal_gates'], sort_keys=True)}`",
        "",
        "## Runtime",
        "",
        f"- Runner wall: `{remote['timings_s']['runner_wall']:.3f} s`",
        f"- Training + epoch validation: "
        f"`{remote['timings_s']['training_and_epoch_validation']:.3f} s`",
        f"- Checkpoint reevaluation: "
        f"`{remote['timings_s']['checkpoint_reevaluation']:.3f} s`",
        f"- Peak CUDA allocated/reserved: "
        f"`{remote['memory']['peak_cuda_allocated_bytes']/1024**2:.2f} / "
        f"{remote['memory']['peak_cuda_reserved_bytes']/1024**2:.2f} MiB`",
        f"- Peak RSS: `{remote['memory']['peak_rss_mb']:.2f} MiB`",
        "",
        "No Small Matrix, temporal final, FAISS, Hybrid, reranker, serving, "
        "or monitoring run was performed.",
        "",
    ]
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    raw_dir: str,
    processed_dir: str,
    caption_cache: str,
    caption_metadata: str,
    report_json: str = "reports/phase_b2b/full_two_tower_modal_l4.json",
    report_markdown: str = "reports/phase_b2b/full_two_tower_modal_l4.md",
) -> None:
    repository_root = REPOSITORY_ROOT
    wrapper_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if _git_output(repository_root, "status", "--porcelain"):
        raise RuntimeError("Full-run wrapper must start from a clean commit")
    files = build_input_allowlist(
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        caption_cache=Path(caption_cache),
        caption_metadata=Path(caption_metadata),
    )
    manifest = input_bundle_manifest(files)
    volume_record = _verify_existing_input_bundle(
        files=files, manifest=manifest
    )
    request_epoch_s = time.time()
    remote = run_l4_full.remote(
        bundle_sha256=manifest["bundle_sha256"],
        input_manifest=manifest,
        wrapper_commit=wrapper_commit,
        request_epoch_s=request_epoch_s,
    )
    result = {
        "phase": "phase-b2b-full-two-tower-modal-l4",
        "status": "passed",
        "b2b_runner_commit": B2B_RUNNER_COMMIT,
        "wrapper_commit": wrapper_commit,
        "input_manifest": manifest,
        "input_volume": volume_record,
        "remote": remote,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if any(
        token in serialized
        for token in ("/home/", "MODAL_TOKEN", "gho_", "hf_")
    ):
        raise RuntimeError("Generated report contains a host path or secret")
    json_path = (repository_root / report_json).resolve()
    markdown_path = (repository_root / report_markdown).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialized + "\n")
    markdown_path.write_text(_render_markdown(result))
    print(
        json.dumps(
            {
                "status": "passed",
                "selected_epoch": remote["runner_report"][
                    "selected_epoch_by_frozen_rule"
                ],
                "report_json": report_json,
                "report_markdown": report_markdown,
            },
            sort_keys=True,
        ),
        flush=True,
    )
