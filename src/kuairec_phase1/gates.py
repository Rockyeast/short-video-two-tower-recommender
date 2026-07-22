"""Fail-closed Phase 1 selection and final-execution gates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METHODS = (
    "random",
    "global_popularity",
    "time_decayed_popularity",
    "itemcf",
    "bpr_mf",
)
METRICS = (
    "Recall@20",
    "Recall@50",
    "Recall@100",
    "NDCG@10",
    "NDCG@20",
    "Coverage@20",
    "Coverage@50",
    "Coverage@100",
    "WarmRecall@20",
    "WarmRecall@50",
    "WarmRecall@100",
    "TailRecall@20",
    "TailRecall@50",
    "TailRecall@100",
    "ColdRecall@20",
    "ColdRecall@50",
    "ColdRecall@100",
)
CI_METRICS = (
    "Recall@20",
    "Recall@50",
    "Recall@100",
    "NDCG@10",
    "NDCG@20",
    "TailRecall@20",
    "TailRecall@50",
    "TailRecall@100",
    "ColdRecall@20",
    "ColdRecall@50",
    "ColdRecall@100",
)
REQUIRED_CONTRACTS = (
    "event_canonicalization",
    "temporal_evaluation",
    "small_matrix_audit",
    "fit_contexts",
    "candidate_catalog",
    "target_deduplication",
    "cold_start",
    "metrics",
    "baselines",
    "negative_sampling",
)
FIXED_FINAL_EVALUATOR = "scripts/evaluate_temporal_final.py"
ERRATUM_SELECTION_EVALUATOR = "scripts/correct_phase1_segment_metrics.py"
SEGMENT_METRICS = tuple(
    metric
    for metric in METRICS
    if metric.startswith(("WarmRecall@", "TailRecall@", "ColdRecall@"))
)
UNCHANGED_ERRATUM_METRICS = tuple(
    metric for metric in METRICS if metric not in SEGMENT_METRICS
)


class GateError(RuntimeError):
    """Raised before incomplete or tampered experiment state is accepted."""


@dataclass(frozen=True)
class PlanVerification:
    path: str
    sha256: str
    protocol_revision: str
    rows: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise GateError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    return value


def _grid_rows(method: str, search_space: Mapping[str, Any], seeds: list[Any]) -> list[dict[str, Any]]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    combinations = itertools.product(*values) if keys else [()]
    rows: list[dict[str, Any]] = []
    for combination in combinations:
        hyperparameters = dict(zip(keys, combination, strict=True))
        config_id = hashlib.sha256(
            f"{method}:{canonical_json(hyperparameters)}".encode()
        ).hexdigest()[:16]
        for seed in seeds:
            rows.append(
                {
                    "method": method,
                    "config_id": config_id,
                    "hyperparameters": hyperparameters,
                    "seed": seed,
                }
            )
    return rows


def _expected_plan_from_contracts(root: Path) -> dict[str, dict[str, Any]]:
    baseline = _load_yaml(root / "contracts/baselines_v1.yaml", "baseline contract")
    metrics = _load_yaml(root / "contracts/metrics_v1.yaml", "metrics contract")
    methods = baseline["baselines"]
    expected = {
        "random": {
            "search_space": {},
            "seeds": methods["random"]["seeds"],
        },
        "global_popularity": {"search_space": {}, "seeds": [None]},
        "time_decayed_popularity": {
            "search_space": {
                "variant": methods["time_decayed_popularity"]["variants"],
                "half_life_days": methods["time_decayed_popularity"][
                    "half_life_days_grid"
                ],
            },
            "seeds": [None],
        },
        "itemcf": {
            "search_space": {
                "neighbor_count": methods["itemcf"]["neighbor_count_grid"],
                "shrinkage": methods["itemcf"]["shrinkage_grid"],
            },
            "seeds": [None],
        },
        "bpr_mf": {
            "search_space": {
                "embedding_dim": methods["bpr_mf"]["embedding_dim_grid"],
                "learning_rate": methods["bpr_mf"]["learning_rate_grid"],
                "l2": methods["bpr_mf"]["l2_grid"],
                "epoch": methods["bpr_mf"]["epoch_checkpoints"],
            },
            "seeds": methods["bpr_mf"]["seeds"],
        },
    }
    expected["time_decayed_popularity"]["search_space"]["variant"] = list(
        methods["time_decayed_popularity"]["variants"]
    )
    if expected["random"]["seeds"] != metrics["randomness"]["random_baseline_seeds"]:
        raise GateError("Random seeds disagree between baseline and metrics contracts")
    if expected["bpr_mf"]["seeds"] != metrics["randomness"]["bpr_training_seeds"]:
        raise GateError("BPR seeds disagree between baseline and metrics contracts")
    return expected


def load_and_validate_selection_plan(
    repo_root: str | Path,
    plan_path: str | Path = "configs/phase1_selection_plan.yaml",
) -> PlanVerification:
    root = Path(repo_root).resolve()
    path = (root / plan_path).resolve() if not Path(plan_path).is_absolute() else Path(plan_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise GateError("Selection plan must be a repository file")
    plan = _load_yaml(path, "selection plan")
    required = {
        "schema_version": 1,
        "plan_scope": "temporal_validation_selection",
        "protocol_revision": "protocol-v2.1.1",
        "fit_splits": ["train"],
        "evaluation_split": "validation",
        "model_selection": "lexicographic",
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise GateError(f"Selection plan field {key!r} is invalid")
    methods = plan.get("methods")
    if not isinstance(methods, Mapping) or tuple(methods) != METHODS:
        raise GateError(f"Selection plan methods must be exactly {list(METHODS)}")
    expected_methods = _expected_plan_from_contracts(root)
    if canonical_json(methods) != canonical_json(expected_methods):
        raise GateError("Selection plan grid or seeds differ from frozen contracts")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        definition = methods[method]
        rows.extend(_grid_rows(method, definition["search_space"], definition["seeds"]))
    keys = {(row["method"], row["config_id"], row["seed"]) for row in rows}
    if len(keys) != len(rows):
        raise GateError("Selection plan expands to duplicate method/config/seed rows")
    return PlanVerification(
        path=str(path),
        sha256=sha256_file(path),
        protocol_revision=plan["protocol_revision"],
        rows=tuple(rows),
    )


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GateError(f"{label} must be a SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GateError(f"{label} must be hexadecimal") from exc
    return value


def _require_finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise GateError(f"{label} must be finite and non-negative")
    return converted


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, Any]:
    return row.get("method"), row.get("config_id"), row.get("seed")


def _git_file_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GateError(f"Cannot read {relative} from code commit {commit}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_selection_bindings(
    root: Path, result: Mapping[str, Any], hashes: Mapping[str, Any]
) -> None:
    manifest = _load_json(root / "manifests/split_manifest.json", "split manifest")
    contract_hashes = hashes["contracts"]
    active = manifest.get("active_contracts")
    if not isinstance(active, Mapping):
        raise GateError("Split manifest active contracts are missing")
    for name, entry in active.items():
        if not isinstance(entry, Mapping) or contract_hashes.get(name) != sha256_file(
            root / entry["path"]
        ):
            raise GateError(f"Selection result contract binding changed: {name}")
    artifact_manifest = (
        root
        / "artifacts"
        / "phase1"
        / result["split_manifest_sha256"]
        / "manifest.json"
    )
    if not artifact_manifest.is_file() or sha256_file(artifact_manifest) != hashes.get(
        "processed_artifact_manifest_sha256"
    ):
        raise GateError("Selection result processed-artifact binding changed")
    evaluator = hashes["evaluator"]
    evaluator_path = root / evaluator["path"]
    if not evaluator_path.is_file() or sha256_file(evaluator_path) != evaluator["sha256"]:
        raise GateError("Selection evaluator content hash changed")
    if _git_file_sha256(root, result["code_commit"], evaluator["path"]) != evaluator[
        "sha256"
    ]:
        raise GateError("Selection evaluator is not bound to code_commit")


def validate_selection_result(
    repo_root: str | Path,
    result_path: str | Path,
    *,
    plan: PlanVerification | None = None,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    plan = plan or load_and_validate_selection_plan(root)
    path = Path(result_path)
    if not path.is_absolute():
        path = root / path
    result = _load_json(path, "selection result")
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    required = {
        "schema_version": 1,
        "result_scope": "selection_validation",
        "protocol_revision": plan.protocol_revision,
        "selection_plan_sha256": plan.sha256,
        "split_manifest_sha256": manifest_sha,
        "fit_splits": ["train"],
        "evaluation_split": "validation",
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise GateError(f"Selection result field {key!r} is invalid")
    code_commit = result.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise GateError("Selection result code_commit must be a full commit SHA")
    hashes = result.get("hashes")
    if not isinstance(hashes, Mapping):
        raise GateError("Selection result hashes must be a mapping")
    _require_hash(hashes.get("processed_artifact_manifest_sha256"), "artifact manifest hash")
    contract_hashes = hashes.get("contracts")
    if not isinstance(contract_hashes, Mapping) or set(contract_hashes) != set(REQUIRED_CONTRACTS):
        raise GateError("Selection result must bind every required contract hash")
    for name, digest in contract_hashes.items():
        _require_hash(digest, f"contract hash {name}")
    evaluator = hashes.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise GateError("Selection result must bind the evaluator")
    evaluator_path = evaluator.get("path")
    erratum = result.get("erratum")
    expected_evaluator = (
        ERRATUM_SELECTION_EVALUATOR
        if isinstance(erratum, Mapping)
        else "scripts/run_phase1_baselines.py"
    )
    if evaluator_path != expected_evaluator:
        raise GateError("Selection evaluator path is not the registered entrypoint")
    _require_hash(evaluator.get("sha256"), "selection evaluator hash")
    if isinstance(erratum, Mapping):
        if erratum.get("id") != "ERRATUM-001":
            raise GateError("Selection erratum ID is invalid")
        if not isinstance(erratum.get("reason"), str) or not erratum["reason"].strip():
            raise GateError("Selection erratum reason must be non-empty")
        segment = hashes.get("segment_membership")
        if not isinstance(segment, Mapping):
            raise GateError("Corrected selection result must bind segment membership")
        reference = _load_json(
            root / "manifests/split_manifest.json", "split manifest"
        )["cold_start_contexts"]["validation"]
        if segment.get("data_warm_item_count") != reference["reference_item_count"]:
            raise GateError("Corrected data-warm item count differs from Phase 0")
        if (
            segment.get("data_warm_membership_sha256")
            != reference["reference_membership_sha256"]
        ):
            raise GateError("Corrected data-warm membership hash differs from Phase 0")
        _require_hash(
            segment.get("data_warm_membership_sha256"),
            "corrected data-warm membership hash",
        )
    if verify_bindings:
        _verify_selection_bindings(root, result, hashes)
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise GateError("Selection result rows must be a list")
    expected_rows = {_row_key(row): row for row in plan.rows}
    actual_rows: dict[tuple[str, str, Any], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GateError(f"Selection result row {index} must be an object")
        key = _row_key(row)
        if key in actual_rows:
            raise GateError(f"Duplicate selection result row: {key}")
        if key not in expected_rows:
            raise GateError(f"Unplanned selection result row: {key}")
        if canonical_json(row.get("hyperparameters")) != canonical_json(
            expected_rows[key]["hyperparameters"]
        ):
            raise GateError(f"Selection row {key} uses grid-external parameters")
        if row.get("status") != "completed":
            raise GateError(f"Selection row {key} did not complete")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
            raise GateError(f"Selection row {key} has incomplete metrics")
        for metric, value in metrics.items():
            numeric = _require_finite_nonnegative(value, f"{key} {metric}")
            if numeric > 1.0:
                raise GateError(f"{key} {metric} must be at most 1")
        denominators = row.get("denominators")
        required_denominators = {
            "query_count",
            "user_count",
            "target_count",
            "candidate_union_count",
            "candidate_score_count",
            "warm_query_count",
            "warm_user_count",
            "warm_target_count",
            "tail_query_count",
            "tail_user_count",
            "tail_target_count",
            "cold_query_count",
            "cold_user_count",
            "cold_target_count",
        }
        if not isinstance(denominators, Mapping) or set(denominators) != required_denominators:
            raise GateError(f"Selection row {key} has incomplete denominators")
        for name, value in denominators.items():
            _require_finite_nonnegative(value, f"{key} denominator {name}")
        runtime = row.get("runtime")
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "seconds",
            "peak_memory_mb",
            "processed_cache_hit",
        }:
            raise GateError(f"Selection row {key} has incomplete runtime metadata")
        _require_finite_nonnegative(runtime["seconds"], f"{key} runtime")
        _require_finite_nonnegative(runtime["peak_memory_mb"], f"{key} memory")
        if not isinstance(runtime["processed_cache_hit"], bool):
            raise GateError(f"{key} cache-hit flag must be boolean")
        intervals = row.get("bootstrap_95_percent_intervals")
        if not isinstance(intervals, Mapping) or set(intervals) != set(CI_METRICS):
            raise GateError(f"Selection row {key} has incomplete bootstrap intervals")
        for metric, interval in intervals.items():
            if not isinstance(interval, list) or len(interval) != 2:
                raise GateError(f"{key} {metric} CI must contain [low, high]")
            low = _require_finite_nonnegative(interval[0], f"{key} {metric} CI low")
            high = _require_finite_nonnegative(interval[1], f"{key} {metric} CI high")
            if low > high or high > 1.0:
                raise GateError(f"{key} {metric} CI is invalid")
        actual_rows[key] = row
    missing = set(expected_rows) - set(actual_rows)
    if missing:
        raise GateError(f"Selection result is missing {len(missing)} planned rows")
    if len(actual_rows) != len(expected_rows):
        raise GateError("Selection result coverage differs from the selection plan")
    return result


def validate_selection_erratum_invariants(
    original_result: Mapping[str, Any],
    corrected_result: Mapping[str, Any],
    original_bundle: Mapping[str, Any],
    corrected_bundle: Mapping[str, Any],
) -> None:
    """Reject a segment-only correction that changes ranking or selection."""

    original_rows = {
        _row_key(row): row for row in original_result.get("rows", [])
    }
    corrected_rows = {
        _row_key(row): row for row in corrected_result.get("rows", [])
    }
    if original_rows.keys() != corrected_rows.keys() or len(original_rows) != 97:
        raise GateError("Erratum must preserve all 97 method/config/seed rows")
    for key, original in original_rows.items():
        corrected = corrected_rows[key]
        if canonical_json(original.get("hyperparameters")) != canonical_json(
            corrected.get("hyperparameters")
        ):
            raise GateError(f"Erratum changed hyperparameters for {key}")
        for metric in UNCHANGED_ERRATUM_METRICS:
            if original["metrics"][metric] != corrected["metrics"][metric]:
                raise GateError(f"Erratum changed protected overall metric {metric} for {key}")
        for name in (
            "query_count",
            "user_count",
            "target_count",
            "candidate_union_count",
            "candidate_score_count",
        ):
            if original["denominators"][name] != corrected["denominators"][name]:
                raise GateError(f"Erratum changed protected denominator {name} for {key}")
        if original.get("coverage") != corrected.get("coverage"):
            raise GateError(f"Erratum changed coverage evidence for {key}")
    original_methods = [
        (row["name"], row["config_id"], row["hyperparameters"], row["seeds"])
        for row in original_bundle.get("methods", [])
    ]
    corrected_methods = [
        (row["name"], row["config_id"], row["hyperparameters"], row["seeds"])
        for row in corrected_bundle.get("methods", [])
    ]
    if canonical_json(original_methods) != canonical_json(corrected_methods):
        raise GateError("Erratum changed a selected configuration or seed")


def _complexity(method: str, hyperparameters: Mapping[str, Any]) -> tuple[Any, ...]:
    if method == "random" or method == "global_popularity":
        return (0,)
    if method == "time_decayed_popularity":
        return (0 if hyperparameters["variant"] == "fit_frozen" else 1, hyperparameters["half_life_days"])
    if method == "itemcf":
        return (hyperparameters["neighbor_count"], hyperparameters["shrinkage"])
    return (
        hyperparameters["embedding_dim"],
        hyperparameters["epoch"],
        hyperparameters["learning_rate"],
        hyperparameters["l2"],
    )


def derive_final_method_bundle(
    repo_root: str | Path,
    result_path: str | Path,
    output_path: str | Path,
    *,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    plan = load_and_validate_selection_plan(root)
    result = validate_selection_result(
        root, result_path, plan=plan, verify_bindings=verify_bindings
    )
    selected: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in result["rows"] if row["method"] == method]
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in method_rows:
            grouped.setdefault(row["config_id"], []).append(row)
        candidates = []
        for config_id, rows in grouped.items():
            hyperparameters = dict(rows[0]["hyperparameters"])
            mean_recall = sum(row["metrics"]["Recall@100"] for row in rows) / len(rows)
            mean_ndcg = sum(row["metrics"]["NDCG@20"] for row in rows) / len(rows)
            mean_coverage = sum(row["metrics"]["Coverage@100"] for row in rows) / len(rows)
            key = (
                -mean_recall,
                -mean_ndcg,
                -mean_coverage,
                _complexity(method, hyperparameters),
                canonical_json(hyperparameters),
            )
            candidates.append((key, config_id, hyperparameters, rows))
        _, config_id, hyperparameters, rows = min(candidates, key=lambda value: value[0])
        selected.append(
            {
                "name": method,
                "config_id": config_id,
                "hyperparameters": hyperparameters,
                "seeds": [row["seed"] for row in rows],
            }
        )
    evaluator_path = root / FIXED_FINAL_EVALUATOR
    if not evaluator_path.is_file():
        raise GateError(f"Missing fixed final evaluator: {FIXED_FINAL_EVALUATOR}")
    bundle = {
        "schema_version": 1,
        "bundle_scope": "complete_final_method_table",
        "protocol_revision": plan.protocol_revision,
        "split_manifest_sha256": sha256_file(root / "manifests/split_manifest.json"),
        "selection_plan_sha256": plan.sha256,
        "selection_result_sha256": sha256_file(root / result_path),
        "code_commit": result["code_commit"],
        "model_selection_objective": [
            "maximize Recall@100",
            "maximize NDCG@20",
            "maximize Coverage@100",
            "minimize model_complexity",
            "minimize canonical_hyperparameter_json",
        ],
        "evaluator": {
            "path": FIXED_FINAL_EVALUATOR,
            "sha256": sha256_file(evaluator_path),
            "external_override": "forbidden",
        },
        "methods": selected,
    }
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle


def validate_final_method_bundle(
    repo_root: str | Path, bundle_path: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(bundle_path)
    if not path.is_absolute():
        path = root / path
    bundle = _load_json(path, "final method bundle")
    plan = load_and_validate_selection_plan(root)
    required = {
        "schema_version": 1,
        "bundle_scope": "complete_final_method_table",
        "protocol_revision": plan.protocol_revision,
        "split_manifest_sha256": sha256_file(root / "manifests/split_manifest.json"),
        "selection_plan_sha256": plan.sha256,
    }
    for key, expected in required.items():
        if bundle.get(key) != expected:
            raise GateError(f"Final method bundle field {key!r} is invalid")
    evaluator = bundle.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("path") != FIXED_FINAL_EVALUATOR:
        raise GateError("Final evaluator must be the fixed version-controlled entrypoint")
    evaluator_path = (root / FIXED_FINAL_EVALUATOR).resolve()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", FIXED_FINAL_EVALUATOR],
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise GateError("Final evaluator entrypoint is not version controlled")
    if evaluator.get("sha256") != sha256_file(evaluator_path):
        raise GateError("Final evaluator content hash changed")
    code_commit = bundle.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise GateError("Final method bundle code_commit must be a full commit SHA")
    if _git_file_sha256(root, code_commit, FIXED_FINAL_EVALUATOR) != evaluator[
        "sha256"
    ]:
        raise GateError("Final evaluator is not bound to the recorded code_commit")
    if evaluator.get("external_override") != "forbidden":
        raise GateError("External final evaluator replacement must be forbidden")
    methods = bundle.get("methods")
    if not isinstance(methods, list) or [row.get("name") for row in methods] != list(METHODS):
        raise GateError("Final bundle must contain every method exactly once")
    expected_seed_map: dict[str, list[Any]] = {}
    for row in plan.rows:
        expected_seed_map.setdefault(row["method"], [])
        if row["seed"] not in expected_seed_map[row["method"]]:
            expected_seed_map[row["method"]].append(row["seed"])
    for method in methods:
        if method.get("seeds") != expected_seed_map[method["name"]]:
            raise GateError(f"Final method {method['name']} has incomplete seed coverage")
        if not isinstance(method.get("hyperparameters"), Mapping):
            raise GateError(f"Final method {method['name']} lacks hyperparameters")
    return bundle


def validate_final_result_coverage(
    bundle: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise GateError("Final result rows must be a list")
    expected = {
        (method["name"], method["config_id"], seed)
        for method in bundle["methods"]
        for seed in method["seeds"]
    }
    actual = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "completed":
            raise GateError("Every final method/seed row must complete")
        actual.add((row.get("method"), row.get("config_id"), row.get("seed")))
    if actual != expected or len(rows) != len(expected):
        raise GateError("Final result must exactly cover every frozen method and seed")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GateError(f"Refusing to overwrite selection receipt: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_selection_receipt(
    repo_root: str | Path,
    result_path: str | Path,
    final_bundle_path: str | Path,
    *,
    selected_at_utc: str | None = None,
) -> Path:
    root = Path(repo_root).resolve()
    plan = load_and_validate_selection_plan(root)
    result = validate_selection_result(root, result_path, plan=plan)
    bundle = validate_final_method_bundle(root, final_bundle_path)
    result_file = Path(result_path)
    if not result_file.is_absolute():
        result_file = root / result_file
    bundle_file = Path(final_bundle_path)
    if not bundle_file.is_absolute():
        bundle_file = root / bundle_file
    if bundle.get("selection_result_sha256") != sha256_file(result_file):
        raise GateError("Final method bundle is not bound to the selection result")
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    receipt = root / "receipts" / manifest_sha / "SELECTION_RECEIPT.json"
    payload = {
        "schema_version": 1,
        "receipt_type": "selection",
        "status": "completed",
        "protocol_revision": plan.protocol_revision,
        "split_manifest_sha256": manifest_sha,
        "selection_plan_path": str(Path(plan.path).relative_to(root)),
        "selection_plan_sha256": plan.sha256,
        "selection_result_path": str(result_file.relative_to(root)),
        "selection_result_sha256": sha256_file(result_file),
        "final_method_bundle_path": str(bundle_file.relative_to(root)),
        "final_method_bundle_sha256": sha256_file(bundle_file),
        "code_commit": result["code_commit"],
        "methods": list(METHODS),
        "selected_at_utc": selected_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(receipt, payload)
    return receipt


def _write_bytes_exclusive(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GateError(f"Refusing to overwrite archived receipt: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def archive_and_replace_selection_receipt(
    *,
    receipt_path: str | Path,
    new_payload: Mapping[str, Any],
    expected_old_receipt_sha256: str,
    expected_old_selection_result_sha256: str,
    expected_old_final_bundle_sha256: str,
    erratum_id: str,
    reason: str,
) -> Path:
    """Explicitly archive one known receipt, then atomically replace it."""

    if not isinstance(erratum_id, str) or not erratum_id.strip():
        raise GateError("Erratum ID must be non-empty")
    if not isinstance(reason, str) or not reason.strip():
        raise GateError("Erratum reason must be non-empty")
    receipt = Path(receipt_path)
    if not receipt.is_file():
        raise GateError("Canonical selection receipt is missing")
    old_bytes = receipt.read_bytes()
    old_receipt_sha = hashlib.sha256(old_bytes).hexdigest()
    if old_receipt_sha != expected_old_receipt_sha256:
        raise GateError("Canonical receipt hash does not match the known superseded receipt")
    try:
        old_payload = json.loads(old_bytes)
    except json.JSONDecodeError as exc:
        raise GateError("Canonical selection receipt is invalid JSON") from exc
    if (
        old_payload.get("selection_result_sha256")
        != expected_old_selection_result_sha256
        or old_payload.get("final_method_bundle_sha256")
        != expected_old_final_bundle_sha256
    ):
        raise GateError("Canonical receipt does not bind the known old result and bundle")
    if new_payload.get("erratum_id") != erratum_id:
        raise GateError("Replacement receipt is not bound to the requested erratum")
    if new_payload.get("erratum_reason") != reason:
        raise GateError("Replacement receipt reason differs from the requested reason")
    if new_payload.get("supersedes_selection_receipt_sha256") != old_receipt_sha:
        raise GateError("Replacement receipt does not identify the superseded receipt")

    archive = (
        receipt.parent
        / "superseded"
        / expected_old_selection_result_sha256
        / receipt.name
    )
    _write_bytes_exclusive(archive, old_bytes)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != old_receipt_sha:
        raise GateError("Archived receipt bytes differ from the canonical old receipt")

    encoded = (json.dumps(new_payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = receipt.parent / f".{receipt.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def supersede_selection_receipt(
    repo_root: str | Path,
    result_path: str | Path,
    final_bundle_path: str | Path,
    *,
    expected_old_receipt_sha256: str,
    expected_old_selection_result_sha256: str,
    expected_old_final_bundle_sha256: str,
    erratum_id: str,
    reason: str,
    selected_at_utc: str | None = None,
) -> tuple[Path, Path]:
    """Validate corrected artifacts and replace only the known old receipt."""

    root = Path(repo_root).resolve()
    plan = load_and_validate_selection_plan(root)
    result = validate_selection_result(root, result_path, plan=plan)
    bundle = validate_final_method_bundle(root, final_bundle_path)
    result_file = Path(result_path)
    if not result_file.is_absolute():
        result_file = root / result_file
    bundle_file = Path(final_bundle_path)
    if not bundle_file.is_absolute():
        bundle_file = root / bundle_file
    if bundle.get("selection_result_sha256") != sha256_file(result_file):
        raise GateError("Corrected bundle is not bound to the corrected result")
    if result.get("erratum", {}).get("id") != erratum_id:
        raise GateError("Corrected result does not identify the requested erratum")
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    receipt = root / "receipts" / manifest_sha / "SELECTION_RECEIPT.json"
    payload = {
        "schema_version": 1,
        "receipt_type": "selection",
        "status": "completed",
        "protocol_revision": plan.protocol_revision,
        "split_manifest_sha256": manifest_sha,
        "selection_plan_path": str(Path(plan.path).relative_to(root)),
        "selection_plan_sha256": plan.sha256,
        "selection_result_path": str(result_file.relative_to(root)),
        "selection_result_sha256": sha256_file(result_file),
        "final_method_bundle_path": str(bundle_file.relative_to(root)),
        "final_method_bundle_sha256": sha256_file(bundle_file),
        "code_commit": result["code_commit"],
        "methods": list(METHODS),
        "selected_at_utc": selected_at_utc or datetime.now(timezone.utc).isoformat(),
        "erratum_id": erratum_id,
        "erratum_reason": reason,
        "supersedes_selection_receipt_sha256": expected_old_receipt_sha256,
        "supersedes_selection_result_sha256": expected_old_selection_result_sha256,
        "supersedes_final_method_bundle_sha256": expected_old_final_bundle_sha256,
    }
    archive = archive_and_replace_selection_receipt(
        receipt_path=receipt,
        new_payload=payload,
        expected_old_receipt_sha256=expected_old_receipt_sha256,
        expected_old_selection_result_sha256=expected_old_selection_result_sha256,
        expected_old_final_bundle_sha256=expected_old_final_bundle_sha256,
        erratum_id=erratum_id,
        reason=reason,
    )
    return receipt, archive
