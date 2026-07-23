#!/usr/bin/env python3
"""Execute the one-time sealed Small audit on one Modal NVIDIA L4."""

from __future__ import annotations

import hashlib
import io
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNNER_COMMIT = "178f8df631ffdaa9ff038eb8c9d357e604124cd2"
REPOSITORY_URL = (
    "https://github.com/Rockyeast/short-video-two-tower-recommender.git"
)
REPOSITORY_DIR = Path("/opt/short-video-two-tower-recommender")
INPUT_VOLUME_NAME = "kuairec-b2b-preflight-inputs"
REFIT_VOLUME_NAME = "kuairec-b3b-final-refit-artifacts"
SMALL_VOLUME_NAME = "kuairec-b3b-sealed-small-input"
INPUT_BUNDLE_SHA256 = (
    "7a7b8b370335f61d28063c7821600063c28929e2a837473890611cc1315f56a6"
)
SMALL_SIZE_BYTES = 406_155_844
SMALL_SHA256 = (
    "6b601cd38b2600d8734b4aede309ad73d1e201fdc4bd76d4bc7d2534793d7d15"
)
INPUT_MOUNT = Path("/inputs")
REFIT_MOUNT = Path("/refit")
SMALL_MOUNT = Path("/sealed-small")
SMALL_LOGICAL_PATH = f"sha256/{SMALL_SHA256}/small_matrix.csv"

app = modal.App("short-video-two-tower-b3b-sealed-small")
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=False)
refit_volume = modal.Volume.from_name(REFIT_VOLUME_NAME, create_if_missing=False)
small_volume = modal.Volume.from_name(SMALL_VOLUME_NAME, create_if_missing=True)
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
    )
    .add_local_file(
        Path(__file__),
        "/opt/modal-wrapper/modal_phase_b3b_sealed_small.py",
        copy=True,
    )
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_small_volume(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    actual_size = resolved.stat().st_size
    actual_sha = _sha256_file(resolved)
    if actual_size != SMALL_SIZE_BYTES or actual_sha != SMALL_SHA256:
        raise RuntimeError(
            "Local sealed Small identity mismatch: "
            f"size={actual_size} sha256={actual_sha}"
        )
    manifest_path = f"sha256/{SMALL_SHA256}/identity.json"
    expected_manifest = {
        "logical_path": "KUAIREC_DATA_DIR/small_matrix.csv",
        "size_bytes": SMALL_SIZE_BYTES,
        "sha256": SMALL_SHA256,
    }
    try:
        current = b"".join(small_volume.read_file(manifest_path))
    except Exception as exc:
        if exc.__class__.__name__ not in {
            "FileNotFoundError",
            "NotFoundError",
            "GRPCError",
        }:
            raise
        current = b""
    if current:
        if json.loads(current) != expected_manifest:
            raise RuntimeError("Existing sealed Small volume identity changed")
        reused = True
    else:
        manifest_bytes = io.BytesIO(
            (
                json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
            ).encode()
        )
        with small_volume.batch_upload(force=False) as upload:
            upload.put_file(resolved, SMALL_LOGICAL_PATH)
            upload.put_file(manifest_bytes, manifest_path)
        reused = False
    return {
        "actual_size_bytes": actual_size,
        "actual_sha256": actual_sha,
        "expected_size_bytes": SMALL_SIZE_BYTES,
        "expected_sha256": SMALL_SHA256,
        "match": True,
        "volume_reused": reused,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    result = report["sealed_result"]
    counts = result["audit_counts"]
    lines = [
        "# Phase B3B Sealed Small Matrix Evaluation",
        "",
        "This is the single sealed, nearly-fully-observed audit. It is not a "
        "future-time test. Small was not used for model selection, fitting, "
        "history construction, or route parameters.",
        "",
        "## Audit population",
        "",
        f"- Observed pairs: `{counts['observed_pair_count']}`",
        f"- Observed NORMAL pairs: `{counts['observed_normal_pair_count']}`",
        f"- NORMAL candidate items: `{counts['normal_candidate_item_count']}`",
        f"- Evaluable queries: `{result['query_count']}`",
        (
            "- Excluded zero-relevant users: "
            f"`{counts['excluded_zero_relevant_user_count']}`"
        ),
        f"- Warm / cold users: `{result['warm_user_count']} / "
        f"{result['cold_user_count']}`",
        f"- Targets: `{result['target_count']}`",
        f"- Data-cold items: `{counts['data_cold_item_count']}`",
        "",
        "## Metrics",
        "",
        "| Method | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
        "Coverage@100 | Data-Cold Recall@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, record in result["results"].items():
        metrics = record["metrics"]
        lines.append(
            f"| {method} | {metrics['Recall@20']:.6f} | "
            f"{metrics['Recall@50']:.6f} | "
            f"{metrics['Recall@100']:.6f} | "
            f"{metrics['NDCG@20']:.6f} | "
            f"{metrics['Coverage@100']:.6f} | "
            f"{metrics['Data-Cold Recall@100']:.6f} |"
        )
    first = next(iter(result["results"].values()))
    denominators = first["denominators"]
    lines.extend(
        [
            "",
            "## Denominators and fallback",
            "",
            f"- Warm target denominator: `{denominators['target_count']}`",
            (
                "- Data-cold target denominator: "
                f"`{denominators['data_cold_target_count']}`"
            ),
            (
                "- Cold-user query / target denominator: "
                f"`{first['cold_user_denominators']['query_count']} / "
                f"{first['cold_user_denominators']['target_count']}`"
            ),
            "- Every cold-user route uses the same frozen refit Global "
            "Popularity fallback.",
            "",
            "Cold-user metrics by method:",
            "",
            "| Method | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
            "Coverage@100 | Data-Cold Recall@100 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, record in result["results"].items():
        metrics = record["cold_user_metrics"]
        if metrics is None:
            lines.append(f"| {method} | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {method} | {metrics['Recall@20']:.6f} | "
                f"{metrics['Recall@50']:.6f} | "
                f"{metrics['Recall@100']:.6f} | "
                f"{metrics['NDCG@20']:.6f} | "
                f"{metrics['Coverage@100']:.6f} | "
                f"{metrics['Data-Cold Recall@100']:.6f} |"
            )
    runtime = report["runtime"]
    lines.extend(
        [
            "",
            "## Runtime and claim boundary",
            "",
            f"- GPU: `{runtime['gpu_name']}`",
            f"- Wall time: `{runtime['wall_time_s']:.3f} s`",
            f"- Peak RSS: `{runtime['peak_rss_mb']:.2f} MiB`",
            (
                "- Peak CUDA allocated / reserved: "
                f"`{runtime['peak_cuda_allocated_mb']:.2f} / "
                f"{runtime['peak_cuda_reserved_mb']:.2f} MiB`"
            ),
            "- No statistical-significance or cross-seed claim is made.",
            "- The result is reported as observed; no rerun, retuning, or "
            "post-Small model selection is permitted.",
            "- Temporal final was not accessed.",
            "",
        ]
    )
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=3600,
    startup_timeout=600,
    retries=0,
    min_containers=0,
    max_containers=1,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT: input_volume.read_only(),
        REFIT_MOUNT: refit_volume.read_only(),
        SMALL_MOUNT: small_volume.read_only(),
    },
    include_source=False,
    serialized=True,
)
def run_sealed() -> dict[str, Any]:
    import os
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Sealed Small requires exactly one CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")
    remote_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
    ).strip()
    if remote_commit != RUNNER_COMMIT:
        raise RuntimeError("Sealed runner commit changed")

    bundle = INPUT_MOUNT / "bundles" / INPUT_BUNDLE_SHA256
    data_dir = Path("/tmp/sealed-data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "big_matrix.csv",
        "item_daily_features.csv",
        "kuairec_caption_category.csv",
    ):
        (data_dir / name).symlink_to(bundle / "raw" / name)
    (data_dir / "small_matrix.csv").symlink_to(
        SMALL_MOUNT / SMALL_LOGICAL_PATH
    )

    small_path = data_dir / "small_matrix.csv"
    actual_size = small_path.stat().st_size
    actual_sha = _sha256_file(small_path)
    if actual_size != SMALL_SIZE_BYTES or actual_sha != SMALL_SHA256:
        raise RuntimeError("Remote sealed Small identity mismatch")

    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    from scripts.run_phase_b3b_sealed_small import run

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    previous = Path.cwd()
    try:
        os.chdir(REPOSITORY_DIR)
        sealed_result = run(
            data_dir=data_dir,
            artifact_dir=bundle / "processed",
            caption_cache_path=(
                REFIT_MOUNT / "phase-b3b-final-v1/caption_embeddings.npz"
            ),
            caption_metadata_path=(
                REFIT_MOUNT
                / "phase-b3b-final-v1/caption_cache_metadata.json"
            ),
            popularity_path=(
                REFIT_MOUNT
                / "phase-b3b-final-v1/artifacts/final_global_popularity.json"
            ),
            bpr_checkpoint=(
                REFIT_MOUNT
                / "phase-b3b-final-v1/artifacts/final_bpr_epoch_020.npz"
            ),
            two_tower_checkpoint=(
                REFIT_MOUNT
                / "phase-b3b-final-v1/artifacts/final_two_tower_epoch_001.pt"
            ),
            final_refit_report_path=(
                REPOSITORY_DIR
                / "reports/phase_b3b0/final_refit_modal_l4.json"
            ),
            split_manifest_path=(
                REPOSITORY_DIR / "manifests/split_manifest.json"
            ),
            report_json=Path("/tmp/sealed_small_runner.json"),
            execute_sealed_small=True,
            device="cuda:0",
        )
    finally:
        os.chdir(previous)
    wall_time = time.perf_counter() - started
    return {
        "phase": "phase-b3b-sealed-small-modal-l4",
        "runner_commit": RUNNER_COMMIT,
        "sealed_result": sealed_result,
        "runtime": {
            "gpu_name": gpu_name,
            "wall_time_s": wall_time,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated()
            / (1024.0**2),
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved()
            / (1024.0**2),
        },
    }


@app.local_entrypoint()
def main(
    small_matrix: str,
    report_json: str = "reports/phase_b3b/sealed_small_modal_l4.json",
    report_markdown: str = "reports/phase_b3b/sealed_small_modal_l4.md",
) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Sealed Small wrapper requires a clean worktree")
    small_identity = _prepare_small_volume(Path(small_matrix))
    remote = run_sealed.remote()
    report = {
        **remote,
        "wrapper_commit": _git("rev-parse", "HEAD"),
        "small_upload_identity": small_identity,
        "claim_boundary": {
            "sealed_nearly_fully_observed_audit": True,
            "future_time_test": False,
            "small_used_for_model_selection": False,
            "significance_claim": False,
            "cross_seed_claim": False,
            "rerun_or_retuning_permitted": False,
            "temporal_final_accessed": False,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("Sealed report contains a host path or secret")
    json_target = (ROOT / report_json).resolve()
    markdown_target = (ROOT / report_markdown).resolve()
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(serialized + "\n")
    markdown_target.write_text(_render_markdown(report))
    print(json.dumps({"status": "completed", "report": report_json}))
