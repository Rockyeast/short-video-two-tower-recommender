#!/usr/bin/env python3
"""Run the bounded Phase B4A FAISS benchmark on one fixed Modal host."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNNER_COMMIT = "af4fa9aebba1cdbcea0cdbb7983fd99952db3db7"
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

app = modal.App("short-video-two-tower-b4a-faiss-benchmark")
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
        "faiss-cpu==1.14.3",
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
        "/opt/modal-wrapper/modal_phase_b4a_faiss_benchmark.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/modal-wrapper",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
    )
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def _size_mib(value: int) -> float:
    return value / (1024.0**2)


def _render_markdown(report: dict[str, Any]) -> str:
    result = report["benchmark"]
    lines = [
        "# Phase B4A FAISS Scalability Benchmark",
        "",
        "Engineering-only comparison using frozen final Two-Tower vectors. "
        "The 100K and 1M catalogs add deterministic normalized synthetic "
        "distractors and do not support a recommendation-effectiveness claim.",
        "",
        f"- Runner commit: `{report['runner_commit']}`",
        f"- Wrapper commit: `{report['wrapper_commit']}`",
        f"- GPU used only for vector encoding: `{report['gpu_name']}`",
        f"- CPU: `{result['runtime_identity']['cpu_model']}`",
        (
            "- Fixed threads / seed / query count: "
            f"`{result['contract']['thread_count']} / "
            f"{result['contract']['seed']} / "
            f"{result['contract']['query_limit']}`"
        ),
        (
            "- HNSW M / efConstruction / efSearch: "
            f"`{result['contract']['hnsw']['m']} / "
            f"{result['contract']['hnsw']['ef_construction']} / "
            f"{result['contract']['hnsw']['ef_search']}`"
        ),
        "",
        "## Scale and latency",
        "",
        "| Scope | Items | Exact p50/p95 ms | FlatIP p50/p95 ms | "
        "HNSW p50/p95 ms | HNSW QPS | HNSW Recall@100 | HNSW gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale in result["scales"]:
        exact = scale["numpy_exact"]
        flat = scale["faiss_index_flat_ip"]
        hnsw = scale["faiss_hnsw"]
        lines.append(
            f"| {scale['scale_name']} | {scale['catalog_count']} | "
            f"{exact['p50_ms']:.3f}/{exact['p95_ms']:.3f} | "
            f"{flat['p50_ms']:.3f}/{flat['p95_ms']:.3f} | "
            f"{hnsw['p50_ms']:.3f}/{hnsw['p95_ms']:.3f} | "
            f"{hnsw['qps']:.2f} | "
            f"{hnsw['ann_recall_at_100_vs_index_flat_ip']:.6f} | "
            f"{hnsw['passes_99_percent_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Build time and index size",
            "",
            "| Scope | Flat build s | Flat MiB | HNSW build s | HNSW MiB | "
            "Peak RSS MiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scale in result["scales"]:
        flat = scale["faiss_index_flat_ip"]
        hnsw = scale["faiss_hnsw"]
        lines.append(
            f"| {scale['scale_name']} | {flat['index_build_s']:.3f} | "
            f"{_size_mib(flat['index_size_bytes']):.2f} | "
            f"{hnsw['index_build_s']:.3f} | "
            f"{_size_mib(hnsw['index_size_bytes']):.2f} | "
            f"{scale['peak_rss_mb']:.2f} |"
        )
    real = result["scales"][0]
    real_exact = real["numpy_exact"]["p50_ms"]
    real_hnsw = real["faiss_hnsw"]["p50_ms"]
    if real_exact <= real_hnsw:
        conclusion = (
            "At the real 10K-scale catalog, NumPy Exact was faster than "
            "HNSW, so ANN is not needed at the current dataset scale."
        )
    else:
        conclusion = (
            "At the real 10K-scale catalog, HNSW was faster, but its use still "
            "depends on satisfying the frozen 99% recall gate."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            f"- Total wall time: `{result['total_wall_time_s']:.3f} s`",
            f"- Process peak RSS: `{result['peak_rss_mb']:.2f} MiB`",
            "- `recommendation_effectiveness_claim=false`",
            "- Small labels accessed: `False`",
            "- Temporal final accessed: `False`",
            "",
        ]
    )
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=32768,
    timeout=3600,
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
def run_benchmark() -> dict[str, Any]:
    import sys

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B4A requires exactly one CUDA GPU for encoding")
    gpu_name = torch.cuda.get_device_name(0)
    if "NVIDIA L4" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu_name}")
    remote_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
    ).strip()
    if remote_commit != RUNNER_COMMIT:
        raise RuntimeError("B4A runner commit changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_DIR,
        text=True,
    ).strip():
        raise RuntimeError("B4A checkout is dirty")

    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    from scripts.run_phase_b4a_faiss_benchmark import run

    bundle = INPUT_MOUNT / "bundles" / INPUT_BUNDLE_SHA256
    result = run(
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
        numeric_sidecar_path=(
            REPOSITORY_DIR
            / "manifests/phase_b3b_final_numeric_preprocessing.json"
        ),
        report_json=Path("/tmp/phase_b4a_faiss_scalability.json"),
        device="cuda:0",
    )
    return {
        "runner_commit": RUNNER_COMMIT,
        "gpu_name": gpu_name,
        "benchmark": result,
    }


@app.local_entrypoint()
def main(
    report_json: str = "reports/phase_b4a/faiss_scalability.json",
    report_markdown: str = "reports/phase_b4a/faiss_scalability.md",
) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("B4A benchmark requires a clean worktree")
    remote = run_benchmark.remote()
    report = {**remote, "wrapper_commit": _git("rev-parse", "HEAD")}
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("B4A report contains a host path or secret")
    json_target = (ROOT / report_json).resolve()
    markdown_target = (ROOT / report_markdown).resolve()
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(serialized + "\n")
    markdown_target.write_text(_render_markdown(report))
    print(json.dumps({"status": "completed", "report": report_json}))
