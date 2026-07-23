from __future__ import annotations

import ast
import copy
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.modal_preflight_helpers import (
    build_input_allowlist,
    cpu_gpu_differences,
    input_bundle_manifest,
    modal_volume_file_paths,
    validate_gpu_preflight_report,
    verify_remote_inputs,
)


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir()
    processed.mkdir()
    for name in (
        "big_matrix.csv",
        "item_daily_features.csv",
        "kuairec_caption_category.csv",
    ):
        (raw / name).write_text(f"{name}\n")
    (processed / "manifest.json").write_text("{}\n")
    (processed / "events_train_validation.npz").write_bytes(b"events")
    (processed / "catalog.npz").write_bytes(b"catalog")
    caption = root / "caption_embeddings.npz"
    metadata = root / "caption_cache_metadata.json"
    caption.write_bytes(b"caption")
    metadata.write_text("{}\n")
    return raw, processed, caption, metadata


def _runner_report() -> dict:
    membership = {
        "normal": {"count": 10699, "sha256": "normal"},
        "fixed_retrieval_catalog": {"count": 9365, "sha256": "fixed"},
        "model_item_universe": {"count": 9388, "sha256": "universe"},
    }
    frozen = {
        "fixed_catalog_count": 9365,
        "query_count": 6818,
        "warm_query_count": 6816,
        "target_count": 118565,
        "warm_target_count": 118539,
        "data_cold_item_count": 1492,
        "query_contract_sha256": "queries",
    }
    return {
        "phase": "phase-b2b0-full-runner-preflight",
        "execution_mode": "preflight",
        "claim_boundary": {
            "formal_gate_executed": False,
            "effectiveness_claim": False,
            "full_big_train": False,
            "full_big_validation": False,
        },
        "training": {
            "example_count": 2560,
            "epoch_losses": [5.8, 5.1],
            "process_statistics": {
                "optimizer_steps": 20,
                "skipped_batches": 0,
            },
        },
        "validation": {
            "evaluated_queries": 128,
            "fixed_catalog_count": 9365,
            "frozen_contract": frozen,
        },
        "resume": {"verified": True},
        "memberships": membership,
        "checkpoints": [
            {
                "epoch": epoch,
                "epoch_loss": loss,
                "validation": {
                    "metrics": {
                        "Recall@100": recall,
                        "NDCG@20": ndcg,
                        "Coverage@100": coverage,
                    }
                },
            }
            for epoch, loss, recall, ndcg, coverage in (
                (1, 5.8, 0.01, 0.02, 0.03),
                (2, 5.1, 0.02, 0.03, 0.04),
            )
        ],
    }


def test_input_allowlist_is_exact_and_host_paths_are_not_serialized(
    tmp_path: Path,
) -> None:
    raw, processed, caption, metadata = _write_inputs(tmp_path)
    files = build_input_allowlist(
        raw_dir=raw,
        processed_dir=processed,
        caption_cache=caption,
        caption_metadata=metadata,
    )
    manifest = input_bundle_manifest(files)

    assert [record.logical_path for record in files] == [
        "raw/big_matrix.csv",
        "raw/item_daily_features.csv",
        "raw/kuairec_caption_category.csv",
        "processed/manifest.json",
        "processed/events_train_validation.npz",
        "processed/catalog.npz",
        "caption/caption_embeddings.npz",
        "caption/caption_cache_metadata.json",
    ]
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest == input_bundle_manifest(files)


def test_input_allowlist_rejects_missing_file(tmp_path: Path) -> None:
    raw, processed, caption, metadata = _write_inputs(tmp_path)
    (raw / "big_matrix.csv").unlink()
    with pytest.raises(FileNotFoundError, match="raw/big_matrix.csv"):
        build_input_allowlist(
            raw_dir=raw,
            processed_dir=processed,
            caption_cache=caption,
            caption_metadata=metadata,
        )


def test_modal_volume_membership_ignores_directory_entries() -> None:
    class EntryType(Enum):
        FILE = 1
        DIRECTORY = 2

    entries = [
        SimpleNamespace(path="bundle/raw", type=EntryType.DIRECTORY),
        SimpleNamespace(
            path="bundle/raw/big_matrix.csv", type=EntryType.FILE
        ),
    ]
    assert modal_volume_file_paths(entries) == {
        "bundle/raw/big_matrix.csv"
    }


def test_remote_input_verification_rejects_membership_and_sha_changes(
    tmp_path: Path,
) -> None:
    raw, processed, caption, metadata = _write_inputs(tmp_path)
    files = build_input_allowlist(
        raw_dir=raw,
        processed_dir=processed,
        caption_cache=caption,
        caption_metadata=metadata,
    )
    manifest = input_bundle_manifest(files)
    remote = tmp_path / "remote"
    for record in files:
        destination = remote / record.logical_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(record.local_path.read_bytes())
    verified = verify_remote_inputs(remote, manifest)
    assert all(record["match"] for record in verified["files"])

    (remote / "unexpected.txt").write_text("no")
    with pytest.raises(RuntimeError, match="membership mismatch"):
        verify_remote_inputs(remote, manifest)
    (remote / "unexpected.txt").unlink()
    (remote / "raw" / "big_matrix.csv").write_text("changed")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        verify_remote_inputs(remote, manifest)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training", "example_count"), 2559),
        (("training", "process_statistics", "optimizer_steps"), 19),
        (("training", "process_statistics", "skipped_batches"), 1),
        (("validation", "evaluated_queries"), 127),
        (("validation", "fixed_catalog_count"), 9364),
        (("resume", "verified"), False),
        (("claim_boundary", "formal_gate_executed"), True),
        (("training", "epoch_losses"), [5.0, 5.1]),
    ],
)
def test_gpu_report_gate_rejects_contract_mutations(
    path: tuple[str, ...], value: object
) -> None:
    cpu = _runner_report()
    gpu = copy.deepcopy(cpu)
    target = gpu
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match="preflight gate failed"):
        validate_gpu_preflight_report(gpu, cpu_report=cpu)


def test_gpu_report_gate_and_cpu_difference_summary() -> None:
    cpu = _runner_report()
    gpu = copy.deepcopy(cpu)
    gpu["checkpoints"][0]["epoch_loss"] -= 0.1
    gpu["checkpoints"][0]["validation"]["metrics"]["Recall@100"] += 0.001
    validate_gpu_preflight_report(gpu, cpu_report=cpu)
    summary = cpu_gpu_differences(gpu, cpu)
    assert summary["bitwise_equality_required"] is False
    assert summary["parameters_changed_due_to_difference"] is False
    assert summary["epochs"][0]["loss_delta_gpu_minus_cpu"] == pytest.approx(
        -0.1
    )


def test_modal_wrapper_has_frozen_single_l4_contract() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "modal_phase_b2b_preflight.py"
    )
    source = script.read_text()
    tree = ast.parse(source)
    assert 'B2B_RUNNER_COMMIT = "0361b648908acd90a134f379bd39335bcf18d518"' in source
    assert 'gpu="L4"' in source
    assert "gpu=\"any\"" not in source
    assert "memory=16384" in source
    assert "timeout=1200" in source
    assert "startup_timeout=600" in source
    assert "retries=0" in source
    assert "single_use_containers=True" in source
    assert "input_volume.read_only()" in source
    assert 'sys.path.insert(0, str(REPOSITORY_DIR / "src"))' in source
    assert "pip install --no-deps" not in source
    assert source.count(".add_local_file(") == 2
    assert "modal_phase_b2b_preflight.py" in source
    assert "modal_preflight_helpers.py" in source
    assert '.env({"PYTHONPATH": WRAPPER_REMOTE_DIR})' in source
    assert "include_source=False" in source
    assert '"--preflight"' in source
    assert '"--full-run"' not in source
    remote_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
    ]
    assert len(remote_calls) == 1
