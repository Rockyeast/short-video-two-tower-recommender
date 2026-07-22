from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from kuairec_phase1.gates import (
    CI_METRICS,
    METRICS,
    GateError,
    archive_and_replace_selection_receipt,
    derive_final_method_bundle,
    load_and_validate_selection_plan,
    sha256_file,
    validate_final_method_bundle,
    validate_final_result_coverage,
    validate_selection_erratum_invariants,
    validate_selection_result,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_result(tmp_path: Path) -> tuple[Path, object]:
    plan = load_and_validate_selection_plan(ROOT)
    manifest = json.loads((ROOT / "manifests/split_manifest.json").read_text())
    rows = []
    for planned in plan.rows:
        rows.append(
            {
                **planned,
                "status": "completed",
                "metrics": {name: 0.1 for name in METRICS},
                "bootstrap_95_percent_intervals": {
                    name: [0.05, 0.15] for name in CI_METRICS
                },
                "denominators": {
                    "query_count": 10,
                    "user_count": 4,
                    "target_count": 10,
                    "candidate_union_count": 20,
                    "candidate_score_count": 200,
                    "warm_query_count": 8,
                    "warm_user_count": 4,
                    "warm_target_count": 8,
                    "tail_query_count": 3,
                    "tail_user_count": 2,
                    "tail_target_count": 3,
                    "cold_query_count": 1,
                    "cold_user_count": 1,
                    "cold_target_count": 1,
                },
                "runtime": {
                    "seconds": 1.0,
                    "peak_memory_mb": 100.0,
                    "processed_cache_hit": True,
                },
                "extra": {},
            }
        )
    result = {
        "schema_version": 1,
        "result_scope": "selection_validation",
        "protocol_revision": plan.protocol_revision,
        "selection_plan_sha256": plan.sha256,
        "split_manifest_sha256": sha256_file(ROOT / "manifests/split_manifest.json"),
        "fit_splits": ["train"],
        "evaluation_split": "validation",
        "code_commit": "a" * 40,
        "rows": rows,
        "hashes": {
            "processed_artifact_manifest_sha256": "b" * 64,
            "contracts": {
                name: entry["sha256"]
                for name, entry in manifest["active_contracts"].items()
            },
            "evaluator": {
                "path": "scripts/run_phase1_baselines.py",
                "sha256": sha256_file(ROOT / "scripts/run_phase1_baselines.py"),
            },
        },
    }
    path = tmp_path / "selection_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path, plan


def test_selection_plan_exactly_matches_frozen_contracts():
    plan = load_and_validate_selection_plan(ROOT)
    assert len(plan.rows) == 97
    assert {row["method"] for row in plan.rows} == {
        "random",
        "global_popularity",
        "time_decayed_popularity",
        "itemcf",
        "bpr_mf",
    }


def test_selection_plan_rejects_missing_method_and_replaced_seed(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "contracts", root / "contracts")
    plan_path = root / "configs/phase1_selection_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    del plan["methods"]["itemcf"]
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    with pytest.raises(GateError, match="methods must be exactly"):
        load_and_validate_selection_plan(root)

    shutil.rmtree(root)
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "contracts", root / "contracts")
    plan_path = root / "configs/phase1_selection_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    plan["methods"]["random"]["seeds"][0] = 7
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    with pytest.raises(GateError, match="grid or seeds"):
        load_and_validate_selection_plan(root)


def test_selection_result_rejects_missing_row_metric_and_failed_seed(tmp_path):
    path, plan = _valid_result(tmp_path)
    validate_selection_result(ROOT, path, plan=plan, verify_bindings=False)

    value = json.loads(path.read_text())
    value["rows"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(GateError, match="missing 1 planned rows"):
        validate_selection_result(ROOT, path, plan=plan, verify_bindings=False)

    path, plan = _valid_result(tmp_path)
    value = json.loads(path.read_text())
    del value["rows"][0]["metrics"]["Recall@100"]
    path.write_text(json.dumps(value))
    with pytest.raises(GateError, match="incomplete metrics"):
        validate_selection_result(ROOT, path, plan=plan, verify_bindings=False)

    path, plan = _valid_result(tmp_path)
    value = json.loads(path.read_text())
    value["rows"][0]["status"] = "failed"
    path.write_text(json.dumps(value))
    with pytest.raises(GateError, match="did not complete"):
        validate_selection_result(ROOT, path, plan=plan, verify_bindings=False)


def test_final_bundle_binds_tracked_evaluator_and_all_seeds(tmp_path):
    result_path, _ = _valid_result(tmp_path)
    bundle_path = tmp_path / "final_bundle.json"
    result = json.loads(result_path.read_text())
    result["code_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    result_path.write_text(json.dumps(result))
    bundle = derive_final_method_bundle(
        ROOT, result_path, bundle_path, verify_bindings=False
    )
    validate_final_method_bundle(ROOT, bundle_path)

    tampered = dict(bundle)
    tampered["evaluator"] = dict(bundle["evaluator"])
    tampered["evaluator"]["path"] = "external.py"
    bundle_path.write_text(json.dumps(tampered))
    with pytest.raises(GateError, match="fixed version-controlled"):
        validate_final_method_bundle(ROOT, bundle_path)

    missing_seed_result = {
        "rows": [
            {
                "method": method["name"],
                "config_id": method["config_id"],
                "seed": seed,
                "status": "completed",
            }
            for method in bundle["methods"]
            for seed in method["seeds"]
        ][:-1]
    }
    with pytest.raises(GateError, match="exactly cover"):
        validate_final_result_coverage(bundle, missing_seed_result)


def test_erratum_rejects_overall_or_selected_configuration_changes(tmp_path):
    result_path, plan = _valid_result(tmp_path)
    original = json.loads(result_path.read_text())
    corrected = copy.deepcopy(original)
    corrected["rows"][0]["metrics"]["ColdRecall@100"] = 0.2
    methods = []
    for method in ("random", "global_popularity", "time_decayed_popularity", "itemcf", "bpr_mf"):
        row = next(value for value in plan.rows if value["method"] == method)
        seeds = []
        for value in plan.rows:
            if value["method"] == method and value["config_id"] == row["config_id"]:
                seeds.append(value["seed"])
        methods.append(
            {
                "name": method,
                "config_id": row["config_id"],
                "hyperparameters": row["hyperparameters"],
                "seeds": seeds,
            }
        )
    bundle = {"methods": methods}
    validate_selection_erratum_invariants(original, corrected, bundle, bundle)

    changed_overall = copy.deepcopy(corrected)
    changed_overall["rows"][0]["metrics"]["Recall@100"] = 0.2
    with pytest.raises(GateError, match="protected overall metric"):
        validate_selection_erratum_invariants(
            original, changed_overall, bundle, bundle
        )

    changed_bundle = copy.deepcopy(bundle)
    changed_bundle["methods"][0]["config_id"] = "changed"
    with pytest.raises(GateError, match="selected configuration"):
        validate_selection_erratum_invariants(
            original, corrected, bundle, changed_bundle
        )


def test_receipt_supersession_archives_exact_old_bytes_and_rejects_wrong_hash(tmp_path):
    receipt = tmp_path / "receipts" / "manifest" / "SELECTION_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    old_payload = {
        "selection_result_sha256": "1" * 64,
        "final_method_bundle_sha256": "2" * 64,
    }
    old_bytes = (json.dumps(old_payload, indent=2, sort_keys=True) + "\n").encode()
    receipt.write_bytes(old_bytes)
    old_receipt_sha = hashlib.sha256(old_bytes).hexdigest()
    new_payload = {
        "erratum_id": "ERRATUM-001",
        "erratum_reason": "correct segment membership",
        "supersedes_selection_receipt_sha256": old_receipt_sha,
        "selection_result_sha256": "3" * 64,
        "final_method_bundle_sha256": "4" * 64,
    }

    with pytest.raises(GateError, match="known superseded receipt"):
        archive_and_replace_selection_receipt(
            receipt_path=receipt,
            new_payload=new_payload,
            expected_old_receipt_sha256="0" * 64,
            expected_old_selection_result_sha256="1" * 64,
            expected_old_final_bundle_sha256="2" * 64,
            erratum_id="ERRATUM-001",
            reason="correct segment membership",
        )
    assert receipt.read_bytes() == old_bytes

    archive = archive_and_replace_selection_receipt(
        receipt_path=receipt,
        new_payload=new_payload,
        expected_old_receipt_sha256=old_receipt_sha,
        expected_old_selection_result_sha256="1" * 64,
        expected_old_final_bundle_sha256="2" * 64,
        erratum_id="ERRATUM-001",
        reason="correct segment membership",
    )
    assert archive.read_bytes() == old_bytes
    assert json.loads(receipt.read_text()) == new_payload
