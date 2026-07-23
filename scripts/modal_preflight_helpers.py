"""Pure input and acceptance helpers for the bounded Modal L4 preflight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputFile:
    logical_path: str
    local_path: Path
    size_bytes: int
    sha256: str

    def public_record(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_allowlist(
    *,
    raw_dir: Path,
    processed_dir: Path,
    caption_cache: Path,
    caption_metadata: Path,
) -> tuple[InputFile, ...]:
    candidates = (
        ("raw/big_matrix.csv", raw_dir / "big_matrix.csv"),
        ("raw/item_daily_features.csv", raw_dir / "item_daily_features.csv"),
        (
            "raw/kuairec_caption_category.csv",
            raw_dir / "kuairec_caption_category.csv",
        ),
        ("processed/manifest.json", processed_dir / "manifest.json"),
        (
            "processed/events_train_validation.npz",
            processed_dir / "events_train_validation.npz",
        ),
        ("processed/catalog.npz", processed_dir / "catalog.npz"),
        ("caption/caption_embeddings.npz", caption_cache),
        ("caption/caption_cache_metadata.json", caption_metadata),
    )
    records: list[InputFile] = []
    for logical_path, local_path in candidates:
        resolved = local_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Required Modal preflight input is missing: {logical_path}"
            )
        records.append(
            InputFile(
                logical_path=logical_path,
                local_path=resolved,
                size_bytes=resolved.stat().st_size,
                sha256=sha256_file(resolved),
            )
        )
    return tuple(records)


def input_bundle_manifest(files: tuple[InputFile, ...]) -> dict[str, Any]:
    public_files = [record.public_record() for record in files]
    canonical = json.dumps(
        public_files, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "bundle_sha256": hashlib.sha256(
            b"phase-b2b-modal-input-bundle-v1\n" + canonical
        ).hexdigest(),
        "total_size_bytes": sum(record.size_bytes for record in files),
        "files": public_files,
    }


def modal_volume_file_paths(entries: list[Any]) -> set[str]:
    """Return only file paths from Modal's recursive file-and-directory list."""
    return {
        entry.path.lstrip("/")
        for entry in entries
        if getattr(entry.type, "name", None) == "FILE"
    }


def verify_remote_inputs(
    root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    expected_paths = {
        record["logical_path"] for record in manifest["files"]
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "allowlist.json"
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "Modal input file membership mismatch: "
            f"expected={sorted(expected_paths)} actual={sorted(actual_paths)}"
        )
    verified = []
    for record in manifest["files"]:
        path = root / record["logical_path"]
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if (
            actual_size != record["size_bytes"]
            or actual_sha != record["sha256"]
        ):
            raise RuntimeError(
                f"Modal input identity mismatch: {record['logical_path']}"
            )
        verified.append(
            {
                **record,
                "remote_size_bytes": actual_size,
                "remote_sha256": actual_sha,
                "match": True,
            }
        )
    return {
        "bundle_sha256": manifest["bundle_sha256"],
        "total_size_bytes": manifest["total_size_bytes"],
        "files": verified,
    }


def validate_gpu_preflight_report(
    report: dict[str, Any],
    *,
    cpu_report: dict[str, Any],
) -> None:
    expected_claims = {
        "formal_gate_executed": False,
        "effectiveness_claim": False,
        "full_big_train": False,
        "full_big_validation": False,
    }
    process = report["training"]["process_statistics"]
    checks = {
        "phase": report["phase"] == "phase-b2b0-full-runner-preflight",
        "mode": report["execution_mode"] == "preflight",
        "examples": report["training"]["example_count"] == 2560,
        "steps": process["optimizer_steps"] == 20,
        "skips": process["skipped_batches"] == 0,
        "queries": report["validation"]["evaluated_queries"] == 128,
        "catalog": report["validation"]["fixed_catalog_count"] == 9365,
        "resume": report["resume"]["verified"] is True,
        "claims": report["claim_boundary"] == expected_claims,
        "loss": (
            len(report["training"]["epoch_losses"]) == 2
            and report["training"]["epoch_losses"][1]
            < report["training"]["epoch_losses"][0]
        ),
        "checkpoints": len(report["checkpoints"]) == 2,
        "memberships": report["memberships"] == cpu_report["memberships"],
        "validation_contract": (
            report["validation"]["frozen_contract"]
            == cpu_report["validation"]["frozen_contract"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Modal L4 preflight gate failed: {failed}")


def cpu_gpu_differences(
    gpu_report: dict[str, Any], cpu_report: dict[str, Any]
) -> dict[str, Any]:
    cpu_epochs = {
        int(record["epoch"]): record for record in cpu_report["checkpoints"]
    }
    rows = []
    for gpu in gpu_report["checkpoints"]:
        epoch = int(gpu["epoch"])
        cpu = cpu_epochs[epoch]
        gpu_metrics = gpu["validation"]["metrics"]
        cpu_metrics = cpu["validation"]["metrics"]
        rows.append(
            {
                "epoch": epoch,
                "loss_gpu": gpu["epoch_loss"],
                "loss_cpu": cpu["epoch_loss"],
                "loss_delta_gpu_minus_cpu": (
                    gpu["epoch_loss"] - cpu["epoch_loss"]
                ),
                "Recall@100_delta_gpu_minus_cpu": (
                    gpu_metrics["Recall@100"]
                    - cpu_metrics["Recall@100"]
                ),
                "NDCG@20_delta_gpu_minus_cpu": (
                    gpu_metrics["NDCG@20"] - cpu_metrics["NDCG@20"]
                ),
                "Coverage@100_delta_gpu_minus_cpu": (
                    gpu_metrics["Coverage@100"]
                    - cpu_metrics["Coverage@100"]
                ),
            }
        )
    return {
        "bitwise_equality_required": False,
        "parameters_changed_due_to_difference": False,
        "epochs": rows,
    }
