#!/usr/bin/env python3
"""Run three independent Big-only numeric provenance reconstructions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNNER_COMMIT = "b2ed1ea63e137009e131c893c17c05ada32c9040"
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

app = modal.App("short-video-two-tower-b3b-r3-numeric-diagnostic")
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
        "/opt/modal-wrapper/modal_phase_b3b_r3_numeric_diagnostic.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/modal-wrapper",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def _render_markdown(report: dict[str, Any]) -> str:
    first = report["reconstructions"][0]
    training = first["paths"]["training"]
    sealed = first["paths"]["sealed"]
    differences = first["preprocessing_field_differences"]
    lines = [
        "# Phase B3B-R3 Numeric Preprocessing Provenance",
        "",
        "Three independent single-threaded Big-only processes rebuilt both "
        "preprocessing paths. No Small input was mounted or accessed.",
        "",
        f"- Reconstructions identical: "
        f"`{report['reconstructions_identical']}`",
        f"- Checkpoint expected SHA: "
        f"`{first['checkpoint']['expected_numeric_preprocessing_sha256']}`",
        f"- Training-path SHA: "
        f"`{training['numeric_preprocessing_sha256']}`",
        f"- Sealed-path SHA: `{sealed['numeric_preprocessing_sha256']}`",
        f"- Training path matches checkpoint: "
        f"`{training['matches_checkpoint']}`",
        f"- Sealed path matches checkpoint: `{sealed['matches_checkpoint']}`",
        "- Per-process training SHAs: "
        f"`{report['per_process_numeric_sha256']['training']}`",
        "- Per-process sealed SHAs: "
        f"`{report['per_process_numeric_sha256']['sealed']}`",
        "",
        "## Membership",
        "",
        "| Path | Observed | Observed NORMAL | Model universe |",
        "|---|---:|---:|---:|",
        f"| training | {training['observed_items']['count']} | "
        f"{training['observed_normal_items']['count']} | "
        f"{training['model_item_universe']['count']} |",
        f"| sealed | {sealed['observed_items']['count']} | "
        f"{sealed['observed_normal_items']['count']} | "
        f"{sealed['model_item_universe']['count']} |",
        "",
        "## Field-level differences",
        "",
        "```json",
        json.dumps(differences, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


@app.function(
    image=image,
    cpu=4,
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
def reconstruct(process_index: int) -> dict[str, Any]:
    import sys

    remote_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
    ).strip()
    if remote_commit != RUNNER_COMMIT:
        raise RuntimeError("Numeric diagnostic runner commit changed")
    sys.path.insert(0, str(REPOSITORY_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "src"))
    from scripts.diagnose_phase_b3b_numeric_preprocessing import run

    bundle = INPUT_MOUNT / "bundles" / INPUT_BUNDLE_SHA256
    return run(
        data_dir=bundle / "raw",
        artifact_dir=bundle / "processed",
        caption_cache_path=REFIT_ROOT / "caption_embeddings.npz",
        caption_metadata_path=REFIT_ROOT / "caption_cache_metadata.json",
        checkpoint_path=(
            REFIT_ROOT / "artifacts/final_two_tower_epoch_001.pt"
        ),
        process_index=process_index,
    )


@app.local_entrypoint()
def main(
    report_json: str = (
        "reports/phase_b3b_r3/numeric_provenance_diagnostic.json"
    ),
    report_markdown: str = (
        "reports/phase_b3b_r3/numeric_provenance_diagnostic.md"
    ),
) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Numeric diagnostic requires a clean worktree")
    results = [reconstruct.remote(index) for index in range(3)]
    comparable = [
        {
            key: value
            for key, value in result.items()
            if key != "process_index"
        }
        for result in results
    ]
    reconstructions_identical = comparable[1:] == comparable[:-1]
    report = {
        "phase": "phase-b3b-r3-numeric-preprocessing-provenance",
        "runner_commit": RUNNER_COMMIT,
        "wrapper_commit": _git("rev-parse", "HEAD"),
        "process_count": 3,
        "single_threaded": True,
        "small_matrix_accessed": False,
        "reconstructions_identical": reconstructions_identical,
        "per_process_numeric_sha256": {
            "training": [
                result["paths"]["training"][
                    "numeric_preprocessing_sha256"
                ]
                for result in results
            ],
            "sealed": [
                result["paths"]["sealed"][
                    "numeric_preprocessing_sha256"
                ]
                for result in results
            ],
        },
        "reconstructions": results,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(value in serialized for value in ("/home/", "MODAL_TOKEN", "gho_")):
        raise RuntimeError("Numeric diagnostic report contains host data")
    json_target = (ROOT / report_json).resolve()
    markdown_target = (ROOT / report_markdown).resolve()
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(serialized + "\n")
    markdown_target.write_text(_render_markdown(report))
    print(json.dumps({"status": "completed", "report": report_json}))
