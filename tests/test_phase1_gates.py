from __future__ import annotations

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
    derive_final_method_bundle,
    load_and_validate_selection_plan,
    sha256_file,
    validate_final_method_bundle,
    validate_final_result_coverage,
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
