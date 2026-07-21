from __future__ import annotations

import json
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from kuairec_protocol.access import (
    FINAL_CONFIRMATION,
    PROTOCOL_REVISION,
    ProtocolAccessError,
    ReceiptAlreadyExistsError,
    authorize_baseline_selection,
    receipt_root,
    run_final_once,
    sha256_file,
    validate_experiment_bundle,
    validate_final_request,
    validate_selection_receipt,
    verify_final_request_consistency,
    verify_manifest_lock,
    verify_protocol_bundle,
    write_exclusive_json,
    write_selection_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "temporal_final")
GENERATOR_SOURCE = '''from __future__ import annotations

import hashlib
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute_protocol_derived_hashes(*, config_path, data_root, manifest):
    derived = Path(data_root).parent / "derived"
    return {
        "canonical_targets": {
            split: _sha(derived / f"{split}.targets")
            for split in ("train", "validation", "temporal_final")
        },
        "candidate_membership": {
            split: _sha(derived / f"{split}.candidates")
            for split in ("train", "validation", "temporal_final")
        },
    }
'''


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_lock(root: Path) -> None:
    manifest_path = root / "manifests/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _write_json(
        root / "manifests/FINAL_HOLDOUT_LOCKED.json",
        {
            "locked": True,
            "protocol_revision": manifest["protocol_revision"],
            "manifest": "manifests/split_manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "protected": ["temporal_final", "small_matrix_audit"],
            "ordinary_baseline_access": False,
        },
    )


def _derived_hashes(root: Path) -> dict[str, dict[str, str]]:
    return {
        "canonical_targets": {
            split: sha256_file(root / "data/derived" / f"{split}.targets")
            for split in SPLITS
        },
        "candidate_membership": {
            split: sha256_file(root / "data/derived" / f"{split}.candidates")
            for split in SPLITS
        },
    }


def make_protocol_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    (root / "contracts").mkdir()
    (root / "scripts").mkdir()
    (root / "manifests").mkdir()
    (root / "data/raw").mkdir(parents=True)
    (root / "data/derived").mkdir()

    archive = root / "data/raw/KuaiRec.zip"
    source = root / "data/raw/source.csv"
    archive.write_bytes(b"synthetic-kuairec-archive")
    source.write_text("user_id,video_id,timestamp\n1,10,1.0\n")
    contract = root / "contracts/main.yaml"
    contract.write_text(f"protocol_revision: {PROTOCOL_REVISION}\n")
    generator = root / "scripts/audit_phase0.py"
    generator.write_text(GENERATOR_SOURCE)
    for split in SPLITS:
        (root / "data/derived" / f"{split}.targets").write_text(
            f"target|{split}\n"
        )
        (root / "data/derived" / f"{split}.candidates").write_text(
            f"membership|{split}\n"
        )

    fraction_validation = {
        "keys": [
            "train_fraction",
            "validation_fraction",
            "temporal_final_fraction",
        ],
        "required_sum": 1.0,
        "absolute_tolerance": 1.0e-12,
    }
    config = {
        "dataset": {"expected_files": ["source.csv"]},
        "protocol": {
            "revision": PROTOCOL_REVISION,
            "active_contracts": {"main": "contracts/main.yaml"},
        },
        "split": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "temporal_final_fraction": 0.15,
            "fraction_validation": fraction_validation,
        },
    }
    config_path = root / "configs/phase0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    derived = _derived_hashes(root)
    manifest = {
        "schema_version": 2,
        "protocol_revision": PROTOCOL_REVISION,
        "immutable": True,
        "dataset": {
            "archive_sha256": sha256_file(archive),
            "source_files": {
                "source.csv": {
                    "relative_path": "source.csv",
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            },
        },
        "config_sha256": sha256_file(config_path),
        "active_contracts": {
            "main": {
                "path": "contracts/main.yaml",
                "sha256": sha256_file(contract),
            }
        },
        "generation_code": {
            "path": "scripts/audit_phase0.py",
            "sha256": sha256_file(generator),
        },
        "split_algorithm": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "temporal_final_fraction": 0.15,
            "fraction_validation": fraction_validation,
        },
        "splits": {
            split: {
                "canonical_target_sha256": derived["canonical_targets"][split],
                "candidate_membership_sha256": derived["candidate_membership"][split],
            }
            for split in SPLITS
        },
        "candidate_catalog": {
            "membership_hash_format": {
                "algorithm": "sha256",
                "version": "synthetic-membership-v1",
            }
        },
        "locks": {
            "temporal_final_locked": True,
            "small_matrix_audit_locked": True,
            "ordinary_baseline_scripts_may_run_final": False,
            "unlock_requires_separate_explicit_final_command": True,
        },
    }
    _write_json(root / "manifests/split_manifest.json", manifest)
    _write_lock(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Protocol Test")
    _git(root, "config", "user.email", "protocol-test@example.com")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "synthetic protocol bundle")
    return root


def write_experiment_bundle(root: Path) -> Path:
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    path = root / "experiments/final_table.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "bundle_scope": "complete_final_method_table",
            "protocol_revision": PROTOCOL_REVISION,
            "split_manifest_sha256": manifest_sha,
            "code_commit": _git(root, "rev-parse", "HEAD"),
            "methods": [
                {"name": "random", "hyperparameters": {}, "seeds": [11, 17]},
                {
                    "name": "bpr",
                    "hyperparameters": {"factors": 64, "epochs": 10},
                    "seeds": [11, 17],
                },
            ],
        },
    )
    return path


def prepare_selection(root: Path):
    verification = verify_protocol_bundle(root)
    bundle_path = write_experiment_bundle(root)
    bundle_payload = json.loads(bundle_path.read_text())
    selection_result = root / "reports/selection.json"
    _write_json(
        selection_result,
        {
            "schema_version": 1,
            "result_scope": "selection_validation",
            "protocol_revision": PROTOCOL_REVISION,
            "split_manifest_sha256": verification.manifest_sha256,
            "experiment_bundle_sha256": sha256_file(bundle_path),
            "code_commit": bundle_payload["code_commit"],
            "fit_splits": ["train"],
            "evaluation_split": "validation",
            "methods": bundle_payload["methods"],
            "selected": "bpr",
        },
    )
    receipt = write_selection_receipt(
        verification=verification,
        experiment_bundle_path=bundle_path,
        selection_result_path=selection_result,
        selected_at_utc="2026-01-02T03:04:05+00:00",
    )
    return verification, bundle_path, selection_result, receipt


def test_complete_bundle_verifies_with_dynamic_generator_rebuilder(tmp_path):
    root = make_protocol_repo(tmp_path)
    verification = verify_protocol_bundle(root)

    assert verification.protocol_revision == PROTOCOL_REVISION
    assert dict(verification.canonical_target_sha256) == _derived_hashes(root)[
        "canonical_targets"
    ]
    assert dict(verification.candidate_membership_sha256) == _derived_hashes(root)[
        "candidate_membership"
    ]


@pytest.mark.parametrize("tamper", ["config", "contract", "generator"])
def test_static_protocol_input_tampering_is_rejected(tmp_path, tamper):
    root = make_protocol_repo(tmp_path)
    paths = {
        "config": root / "configs/phase0.yaml",
        "contract": root / "contracts/main.yaml",
        "generator": root / "scripts/audit_phase0.py",
    }
    paths[tamper].write_text(paths[tamper].read_text() + "\n# tampered\n")

    with pytest.raises(ProtocolAccessError, match="hash"):
        verify_protocol_bundle(root)


def test_rebuilt_candidate_membership_tampering_is_rejected(tmp_path):
    root = make_protocol_repo(tmp_path)
    candidate = root / "data/derived/validation.candidates"
    candidate.write_text(candidate.read_text() + "tampered\n")

    with pytest.raises(
        ProtocolAccessError,
        match=r"candidate_membership\.validation hash does not match",
    ):
        verify_protocol_bundle(root)


def test_missing_candidate_membership_hash_fails_closed(tmp_path):
    root = make_protocol_repo(tmp_path)
    manifest_path = root / "manifests/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["splits"]["validation"]["candidate_membership_sha256"]
    _write_json(manifest_path, manifest)
    _write_lock(root)

    with pytest.raises(ProtocolAccessError, match="must be a lowercase SHA256"):
        verify_protocol_bundle(root)


def test_fraction_contract_is_read_from_config_and_sum_is_enforced(tmp_path):
    root = make_protocol_repo(tmp_path)
    config_path = root / "configs/phase0.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["split"].update(
        {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "temporal_final_fraction": 0.1,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest_path = root / "manifests/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = sha256_file(config_path)
    manifest["split_algorithm"].update(
        {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "temporal_final_fraction": 0.1,
        }
    )
    _write_json(manifest_path, manifest)
    _write_lock(root)

    with pytest.raises(ProtocolAccessError, match="required_sum"):
        verify_protocol_bundle(root)


def test_lock_must_bind_fixed_manifest_path_and_exact_bytes(tmp_path):
    root = make_protocol_repo(tmp_path)
    manifest = root / "manifests/split_manifest.json"
    lock = root / "manifests/FINAL_HOLDOUT_LOCKED.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ProtocolAccessError, match="does not match"):
        verify_manifest_lock(manifest, lock)

    _write_lock(root)
    payload = json.loads(lock.read_text())
    payload["manifest"] = "other.json"
    _write_json(lock, payload)
    with pytest.raises(ProtocolAccessError, match="must point"):
        verify_manifest_lock(manifest, lock)


@pytest.mark.parametrize("protected", ["temporal_final", "small_matrix_audit"])
def test_ordinary_baseline_runs_full_verifier_then_rejects_protected_scope(
    tmp_path, protected
):
    root = make_protocol_repo(tmp_path)
    with pytest.raises(ProtocolAccessError, match="cannot access protected"):
        authorize_baseline_selection(
            fit_splits=("train",),
            evaluation_split=protected,
            repo_root=root,
        )


def test_ordinary_baseline_cannot_bypass_tampered_bundle_with_scope_error(tmp_path):
    root = make_protocol_repo(tmp_path)
    contract = root / "contracts/main.yaml"
    contract.write_text(contract.read_text() + "# tampered\n")

    with pytest.raises(ProtocolAccessError, match="Active contract hash mismatch"):
        authorize_baseline_selection(
            fit_splits=("train",),
            evaluation_split="temporal_final",
            repo_root=root,
        )


def test_selection_receipt_is_fixed_immutable_and_binds_selection_context(tmp_path):
    root = make_protocol_repo(tmp_path)
    verification, bundle_path, _, receipt = prepare_selection(root)
    expected_root = root / "receipts" / verification.manifest_sha256

    assert Path(receipt.path) == expected_root / "SELECTION_RECEIPT.json"
    payload = json.loads(Path(receipt.path).read_text())
    assert payload["fit_splits"] == ["train"]
    assert payload["evaluation_split"] == "validation"
    assert payload["experiment_bundle_sha256"] == sha256_file(bundle_path)
    assert Path(receipt.path).stat().st_mode & stat.S_IWUSR == 0

    bundle = validate_experiment_bundle(root, bundle_path, verification)
    assert validate_selection_receipt(
        verification=verification, experiment_bundle=bundle
    ).sha256 == receipt.sha256


def test_experiment_bundle_requires_unique_methods_and_nonempty_seeds(tmp_path):
    root = make_protocol_repo(tmp_path)
    verification = verify_protocol_bundle(root)
    bundle_path = write_experiment_bundle(root)
    payload = json.loads(bundle_path.read_text())
    payload["methods"][1]["name"] = "random"
    payload["methods"][1]["seeds"] = []
    _write_json(bundle_path, payload)

    with pytest.raises(ProtocolAccessError, match="Duplicate experiment bundle method"):
        validate_experiment_bundle(root, bundle_path, verification)


def test_experiment_bundle_commit_must_exist_match_head_and_have_clean_tracked_tree(
    tmp_path,
):
    root = make_protocol_repo(tmp_path)
    verification = verify_protocol_bundle(root)
    bundle_path = write_experiment_bundle(root)
    original_commit = _git(root, "rev-parse", "HEAD")

    tracked = root / "tracked.txt"
    tracked.write_text("second commit\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "second commit")
    assert _git(root, "rev-parse", "HEAD") != original_commit
    with pytest.raises(ProtocolAccessError, match="does not match HEAD"):
        validate_experiment_bundle(root, bundle_path, verification)

    payload = json.loads(bundle_path.read_text())
    payload["code_commit"] = _git(root, "rev-parse", "HEAD")
    _write_json(bundle_path, payload)
    tracked.write_text("dirty tracked worktree\n")
    with pytest.raises(ProtocolAccessError, match="Tracked worktree/index must be clean"):
        validate_experiment_bundle(root, bundle_path, verification)


def test_selection_result_method_semantics_must_match_bundle(tmp_path):
    root = make_protocol_repo(tmp_path)
    verification = verify_protocol_bundle(root)
    bundle_path = write_experiment_bundle(root)
    bundle_payload = json.loads(bundle_path.read_text())
    result_path = root / "reports/selection.json"
    methods = json.loads(json.dumps(bundle_payload["methods"]))
    methods[1]["seeds"] = [999]
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "result_scope": "selection_validation",
            "protocol_revision": PROTOCOL_REVISION,
            "split_manifest_sha256": verification.manifest_sha256,
            "experiment_bundle_sha256": sha256_file(bundle_path),
            "code_commit": bundle_payload["code_commit"],
            "fit_splits": ["train"],
            "evaluation_split": "validation",
            "methods": methods,
        },
    )

    with pytest.raises(ProtocolAccessError, match="methods/hyperparameters/seeds differ"):
        write_selection_receipt(
            verification=verification,
            experiment_bundle_path=bundle_path,
            selection_result_path=result_path,
        )


def test_final_request_requires_selection_and_freezes_whole_method_table(tmp_path):
    root = make_protocol_repo(tmp_path)
    _, bundle_path, _, selection = prepare_selection(root)

    with pytest.raises(ProtocolAccessError, match="Explicit final confirmation"):
        validate_final_request(
            confirmation="yes",
            repo_root=root,
            experiment_bundle_path=bundle_path,
        )

    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        repo_root=root,
        experiment_bundle_path=bundle_path,
        now_utc="2026-01-02T03:04:05+00:00",
    )
    assert request.method_names == ("random", "bpr")
    assert request.selection_receipt_sha256 == selection.sha256
    assert request.fit_splits == ("train", "validation")
    assert request.evaluation_split == "temporal_final"

    calls: list[str] = []
    final_table = {
        "methods": {
            "random": {"recall_at_100": 0.1},
            "bpr": {"recall_at_100": 0.2},
        }
    }
    result = run_final_once(
        request=request,
        evaluator=lambda: calls.append("whole-table") or final_table,
    )
    assert result == final_table
    assert calls == ["whole-table"]
    fixed_root = receipt_root(root, request.split_manifest_sha256)
    assert (fixed_root / "FINAL_ATTEMPT_CLAIM.json").exists()
    result_receipt = json.loads(
        (fixed_root / "FINAL_RESULT_RECEIPT.json").read_text()
    )
    assert result_receipt["methods"] == ["random", "bpr"]

    with pytest.raises(ReceiptAlreadyExistsError):
        run_final_once(
            request=request,
            evaluator=lambda: calls.append("second") or {},
        )
    assert calls == ["whole-table"]


def test_final_request_rechecks_bundle_and_selection_result(tmp_path):
    root = make_protocol_repo(tmp_path)
    _, bundle_path, selection_result, _ = prepare_selection(root)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        repo_root=root,
        experiment_bundle_path=bundle_path,
    )
    selection_result.write_text('{"tampered": true}\n')
    with pytest.raises(ProtocolAccessError, match="Selection result"):
        verify_final_request_consistency(request)


@pytest.mark.parametrize(
    "relative_path", ["src/untracked_bypass.py", "artifacts/untracked_bypass.py"]
)
def test_untracked_code_cannot_bypass_frozen_commit(tmp_path, relative_path):
    root = make_protocol_repo(tmp_path)
    verification = verify_protocol_bundle(root)
    bundle_path = write_experiment_bundle(root)
    bypass = root / relative_path
    bypass.parent.mkdir(parents=True, exist_ok=True)
    bypass.write_text("BYPASS = True\n")

    with pytest.raises(ProtocolAccessError, match="could bypass code_commit"):
        validate_experiment_bundle(root, bundle_path, verification)


def test_forged_final_request_method_subset_is_rejected_before_evaluator(tmp_path):
    root = make_protocol_repo(tmp_path)
    _, bundle_path, _, _ = prepare_selection(root)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        repo_root=root,
        experiment_bundle_path=bundle_path,
    )
    forged = replace(request, method_names=("random",))
    calls: list[str] = []
    with pytest.raises(ProtocolAccessError, match="method names changed"):
        run_final_once(
            request=forged,
            evaluator=lambda: calls.append("called") or {},
        )
    forged_commit = replace(request, code_commit="b" * 40)
    with pytest.raises(ProtocolAccessError, match="code_commit changed"):
        run_final_once(
            request=forged_commit,
            evaluator=lambda: calls.append("called") or {},
        )
    assert calls == []


def test_final_result_must_cover_every_frozen_method_exactly_once(tmp_path):
    root = make_protocol_repo(tmp_path)
    _, bundle_path, _, _ = prepare_selection(root)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        repo_root=root,
        experiment_bundle_path=bundle_path,
    )
    with pytest.raises(ProtocolAccessError, match="exactly cover"):
        run_final_once(
            request=request,
            evaluator=lambda: {"methods": {"random": {"metric": 0.1}}},
        )
    claim = receipt_root(root, request.split_manifest_sha256) / "FINAL_ATTEMPT_CLAIM.json"
    assert claim.exists()


def test_failed_final_attempt_leaves_claim_and_blocks_retry(tmp_path):
    root = make_protocol_repo(tmp_path)
    _, bundle_path, _, _ = prepare_selection(root)
    request = validate_final_request(
        confirmation=FINAL_CONFIRMATION,
        repo_root=root,
        experiment_bundle_path=bundle_path,
    )

    def fail():
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        run_final_once(request=request, evaluator=fail)
    claim = receipt_root(root, request.split_manifest_sha256) / "FINAL_ATTEMPT_CLAIM.json"
    assert claim.exists()
    with pytest.raises(ReceiptAlreadyExistsError):
        run_final_once(request=request, evaluator=lambda: {})


def test_exclusive_json_never_overwrites(tmp_path):
    receipt = tmp_path / "receipt.json"
    write_exclusive_json(receipt, {"attempt": 1})
    original = receipt.read_bytes()
    with pytest.raises(ReceiptAlreadyExistsError, match="Refusing to overwrite"):
        write_exclusive_json(receipt, {"attempt": 2})
    assert receipt.read_bytes() == original


def test_final_cli_has_no_receipt_directory_override():
    completed = subprocess.run(
        [sys.executable, "scripts/final_evaluation.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--receipt-dir" not in completed.stdout
    assert "--experiment-bundle" in completed.stdout
