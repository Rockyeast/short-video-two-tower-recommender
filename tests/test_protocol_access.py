from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kuairec_protocol.access import (
    FINAL_CONFIRMATION,
    ProtocolAccessError,
    ReceiptAlreadyExistsError,
    authorize_baseline_selection,
    run_final_once,
    sha256_file,
    validate_final_request,
    verify_final_request_consistency,
    verify_manifest_lock,
    write_exclusive_json,
)


ROOT = Path(__file__).resolve().parents[1]


def write_lock_bundle(root: Path) -> tuple[Path, Path]:
    manifest = root / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "locks": {
                    "temporal_final_locked": True,
                    "small_matrix_audit_locked": True,
                    "ordinary_baseline_scripts_may_run_final": False,
                    "unlock_requires_separate_explicit_final_command": True,
                },
            },
            indent=2,
        )
        + "\n"
    )
    lock = root / "FINAL_HOLDOUT_LOCKED.json"
    lock.write_text(
        json.dumps(
            {
                "locked": True,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "protected": ["temporal_final", "small_matrix_audit"],
                "ordinary_baseline_access": False,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest, lock


def write_frozen_inputs(root: Path) -> tuple[Path, Path]:
    method = root / "method.yaml"
    method.write_text("model: frozen\nseed: 17\n")
    artifacts = root / "artifacts.json"
    artifacts.write_text('{"model_sha256": "synthetic"}\n')
    return method, artifacts


def test_manifest_lock_hash_and_selection_scope(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)

    verification = authorize_baseline_selection(
        fit_splits=("train",),
        evaluation_split="validation",
        manifest_path=manifest,
        lock_path=lock,
    )

    assert verification.manifest_sha256 == sha256_file(manifest)
    assert verification.protected_scopes == (
        "small_matrix_audit",
        "temporal_final",
    )


@pytest.mark.parametrize("protected", ["temporal_final", "small_matrix_audit"])
def test_ordinary_baseline_rejects_protected_scope(tmp_path, protected):
    manifest, lock = write_lock_bundle(tmp_path)

    with pytest.raises(ProtocolAccessError, match="cannot access protected"):
        authorize_baseline_selection(
            fit_splits=("train",),
            evaluation_split=protected,
            manifest_path=manifest,
            lock_path=lock,
        )


def test_tampered_manifest_is_rejected(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    manifest.write_text(manifest.read_text() + " ")

    with pytest.raises(ProtocolAccessError, match="does not match"):
        verify_manifest_lock(manifest, lock)


def test_manifest_policy_cannot_enable_ordinary_final(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["locks"]["ordinary_baseline_scripts_may_run_final"] = True
    manifest.write_text(json.dumps(payload) + "\n")
    lock_payload = json.loads(lock.read_text())
    lock_payload["manifest_sha256"] = sha256_file(manifest)
    lock.write_text(json.dumps(lock_payload) + "\n")

    with pytest.raises(ProtocolAccessError, match="must be False"):
        verify_manifest_lock(manifest, lock)


def test_final_request_requires_confirmation_and_hashes_frozen_inputs(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    method, artifacts = write_frozen_inputs(tmp_path)

    with pytest.raises(ProtocolAccessError, match="Explicit final confirmation"):
        validate_final_request(
            confirmation="yes",
            method_config_path=method,
            artifact_manifest_path=artifacts,
            manifest_path=manifest,
            lock_path=lock,
            seed=17,
        )

    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        method_config_path=method,
        artifact_manifest_path=artifacts,
        manifest_path=manifest,
        lock_path=lock,
        seed=17,
        now_utc="2026-01-02T03:04:05+00:00",
    )
    assert request.fit_splits == ("train", "validation")
    assert request.evaluation_split == "temporal_final"
    assert request.method_config_sha256 == sha256_file(method)
    assert request.artifact_manifest_sha256 == sha256_file(artifacts)


def test_exclusive_json_never_overwrites(tmp_path):
    receipt = tmp_path / "receipt.json"
    write_exclusive_json(receipt, {"attempt": 1})
    original = receipt.read_bytes()

    with pytest.raises(ReceiptAlreadyExistsError, match="Refusing to overwrite"):
        write_exclusive_json(receipt, {"attempt": 2})

    assert receipt.read_bytes() == original
    assert receipt.stat().st_mode & stat.S_IWUSR == 0


def test_final_attempt_is_one_shot_and_second_callback_never_runs(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    method, artifacts = write_frozen_inputs(tmp_path)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        method_config_path=method,
        artifact_manifest_path=artifacts,
        manifest_path=manifest,
        lock_path=lock,
        seed=17,
        now_utc="2026-01-02T03:04:05+00:00",
    )
    calls = []

    result = run_final_once(
        request=request,
        receipt_dir=tmp_path / "receipts",
        evaluator=lambda: calls.append("first") or {"synthetic_metric": 0.5},
    )
    assert result == {"synthetic_metric": 0.5}
    assert calls == ["first"]

    with pytest.raises(ReceiptAlreadyExistsError):
        run_final_once(
            request=request,
            receipt_dir=tmp_path / "receipts",
            evaluator=lambda: calls.append("second") or {},
        )
    assert calls == ["first"]


def test_final_request_rechecks_frozen_files_before_claim(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    method, artifacts = write_frozen_inputs(tmp_path)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        method_config_path=method,
        artifact_manifest_path=artifacts,
        manifest_path=manifest,
        lock_path=lock,
        seed=17,
    )
    artifacts.write_text('{"model_sha256": "changed"}\n')

    with pytest.raises(ProtocolAccessError, match="artifact manifest hash changed"):
        verify_final_request_consistency(request)
    calls = []
    with pytest.raises(ProtocolAccessError, match="artifact manifest hash changed"):
        run_final_once(
            request=request,
            receipt_dir=tmp_path / "never-created",
            evaluator=lambda: calls.append("called") or {},
        )
    assert calls == []
    assert not (tmp_path / "never-created").exists()


def test_failed_final_attempt_leaves_claim_and_blocks_retry(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    method, artifacts = write_frozen_inputs(tmp_path)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        method_config_path=method,
        artifact_manifest_path=artifacts,
        manifest_path=manifest,
        lock_path=lock,
        seed=17,
    )

    def fail():
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        run_final_once(
            request=request,
            receipt_dir=tmp_path / "failed",
            evaluator=fail,
        )
    assert (tmp_path / "failed/FINAL_ATTEMPT_CLAIM.json").exists()

    with pytest.raises(ReceiptAlreadyExistsError):
        run_final_once(
            request=request,
            receipt_dir=tmp_path / "failed",
            evaluator=lambda: {},
        )


def test_baseline_guard_cli_denies_final_and_runs_no_baseline(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_baseline_access.py",
            "--manifest",
            str(manifest),
            "--lock",
            str(lock),
            "--evaluation-split",
            "temporal_final",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "ACCESS DENIED" in completed.stderr
    assert "baseline_executed" not in completed.stdout


def test_final_cli_validates_protocol_then_fails_closed_without_receipt(tmp_path):
    manifest, lock = write_lock_bundle(tmp_path)
    method, artifacts = write_frozen_inputs(tmp_path)
    receipt_dir = tmp_path / "real-final-receipts"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/final_evaluation.py",
            "--confirm",
            FINAL_CONFIRMATION,
            "--method-config",
            str(method),
            "--artifact-manifest",
            str(artifacts),
            "--seed",
            "17",
            "--manifest",
            str(manifest),
            "--lock",
            str(lock),
            "--receipt-dir",
            str(receipt_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 3
    assert "FINAL EVALUATION BLOCKED" in completed.stderr
    assert '"final_executed": false' in completed.stdout
    assert not receipt_dir.exists()
