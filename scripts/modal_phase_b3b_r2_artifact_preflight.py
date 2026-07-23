#!/usr/bin/env python3
"""Run the B3B-R2 final-refit artifact preflight on one Modal L4."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNNER_COMMIT = "b0c5bdf20e55f81fcf992cfa8df194983ced2c4c"
REPOSITORY_URL = (
    "https://github.com/Rockyeast/short-video-two-tower-recommender.git"
)
REPOSITORY_DIR = Path("/opt/short-video-two-tower-recommender")
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
REFIT_VOLUME_NAME = "kuairec-b3b-final-refit-artifacts"
INPUT_BUNDLE_SHA256 = (
    "7a7b8b370335f61d28063c7821600063c28929e2a837473890611cc1315f56a6"
)
INPUT_MOUNT = Path("/inputs")
REFIT_MOUNT = Path("/refit")
REFIT_ROOT = REFIT_MOUNT / "phase-b3b-final-v1"

app = modal.App("short-video-two-tower-b3b-r2-artifact-preflight")
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=False)
refit_volume = modal.Volume.from_name(REFIT_VOLUME_NAME, create_if_missing=False)
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
        f"cd {REPOSITORY_DIR} && git checkout --detach {RUNNER_COMMIT}",
        (
            f"cd {REPOSITORY_DIR} && "
            f'test "$(git rev-parse HEAD)" = "{RUNNER_COMMIT}"'
        ),
        f"cd {REPOSITORY_DIR} && test -z \"$(git status --porcelain)\"",
    )
    .add_local_file(
        Path(__file__),
        "/opt/modal-wrapper/modal_phase_b3b_r2_artifact_preflight.py",
        copy=True,
    )
    .env({"PYTHONPATH": "/opt/modal-wrapper"})
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def _render_markdown(report: dict[str, Any]) -> str:
    result = report["artifact_preflight"]
    feature = result["reconstructed_feature_identity"]
    runtime = report["runtime"]
    return "\n".join(
        [
            "# Phase B3B-R2 Final-Refit Artifact Preflight",
            "",
            "Inference-only verification of the frozen final-refit Two-Tower "
            "artifact. No recommendation-effectiveness metric was computed.",
            "",
            f"- Checkpoint SHA256: `{result['checkpoint_sha256']}`",
            (
                "- Category/upload vocabulary counts: "
                f"`{feature['category_vocab_count']} / "
                f"{feature['upload_type_vocab_count']}`"
            ),
            "- Category vocabulary SHA match: "
            f"`{result['feature_identity_sha_match']['category_vocab_sha256']}`",
            "- Upload vocabulary SHA match: "
            f"`{result['feature_identity_sha_match']['upload_type_vocab_sha256']}`",
            "- Numeric preprocessing SHA match: "
            f"`{result['feature_identity_sha_match']['numeric_preprocessing_sha256']}`",
            f"- Model loaded: `{result['model_loaded']}`",
            (
                "- Item encoding shape/passed: "
                f"`{result['item_encoding']['shape']} / "
                f"{result['item_encoding']['passed']}`"
            ),
            (
                "- User encoding shape/passed: "
                f"`{result['user_encoding']['shape']} / "
                f"{result['user_encoding']['passed']}`"
            ),
            "- Exact retrieval passed: "
            f"`{result['exact_retrieval']['passed']}`",
            f"- GPU: `{runtime['gpu_name']}`",
            f"- Wall time: `{runtime['wall_time_s']:.3f} s`",
            f"- Peak RSS: `{runtime['peak_rss_mb']:.2f} MiB`",
            (
                "- Peak CUDA allocated/reserved: "
                f"`{runtime['peak_cuda_allocated_mb']:.2f} / "
                f"{runtime['peak_cuda_reserved_mb']:.2f} MiB`"
            ),
            f"- Small Matrix accessed: `{result['small_matrix_accessed']}`",
            f"- Temporal final accessed: `{result['temporal_final_accessed']}`",
            "- Effectiveness metrics computed: "
            f"`{result['recommendation_effectiveness_metrics_computed']}`",
            "",
        ]
    )


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=1200,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT: input_volume.read_only(),
        REFIT_MOUNT: refit_volume.read_only(),
    },
    include_source=False,
    serialized=True,
)
def run_artifact_preflight() -> dict[str, Any]:
    import resource
    import sys
    import time

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Artifact preflight requires exactly one CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")
    remote_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
    ).strip()
    if remote_commit != RUNNER_COMMIT:
        raise RuntimeError("Artifact-preflight runner commit changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_DIR,
        text=True,
    ).strip():
        raise RuntimeError("Artifact-preflight checkout is dirty")

    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    from scripts.run_phase_b3b_artifact_preflight import run

    bundle = INPUT_MOUNT / "bundles" / INPUT_BUNDLE_SHA256
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    artifact_preflight = run(
        data_dir=bundle / "raw",
        artifact_dir=bundle / "processed",
        caption_cache_path=REFIT_ROOT / "caption_embeddings.npz",
        caption_metadata_path=REFIT_ROOT / "caption_cache_metadata.json",
        popularity_path=(
            REFIT_ROOT / "artifacts/final_global_popularity.json"
        ),
        bpr_checkpoint_path=(
            REFIT_ROOT / "artifacts/final_bpr_epoch_020.npz"
        ),
        two_tower_checkpoint_path=(
            REFIT_ROOT / "artifacts/final_two_tower_epoch_001.pt"
        ),
        final_refit_report_path=(
            REPOSITORY_DIR
            / "reports/phase_b3b0/final_refit_modal_l4.json"
        ),
        report_json=Path("/tmp/b3b_r2_artifact_preflight.json"),
        device="cuda:0",
    )
    wall_time = time.perf_counter() - started
    return {
        "phase": "phase-b3b-r2-artifact-only-modal-l4-preflight",
        "runner_commit": RUNNER_COMMIT,
        "artifact_preflight": artifact_preflight,
        "runtime": {
            "gpu_name": gpu_name,
            "wall_time_s": wall_time,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "peak_cuda_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024.0**2)
            ),
            "peak_cuda_reserved_mb": (
                torch.cuda.max_memory_reserved() / (1024.0**2)
            ),
        },
    }


@app.local_entrypoint()
def main(
    report_json: str = (
        "reports/phase_b3b_r2/artifact_preflight_modal_l4.json"
    ),
    report_markdown: str = (
        "reports/phase_b3b_r2/artifact_preflight_modal_l4.md"
    ),
) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Artifact preflight requires a clean worktree")
    remote = run_artifact_preflight.remote()
    report = {
        **remote,
        "wrapper_commit": _git("rev-parse", "HEAD"),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("Artifact-preflight report contains host data")
    json_target = (ROOT / report_json).resolve()
    markdown_target = (ROOT / report_markdown).resolve()
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(serialized + "\n")
    markdown_target.write_text(_render_markdown(report))
    print(json.dumps({"status": "completed", "report": report_json}))
