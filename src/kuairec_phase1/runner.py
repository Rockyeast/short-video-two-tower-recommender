"""Single-orchestrator Phase 1 temporal validation experiment."""

from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kuairec_protocol.access import authorize_baseline_selection, sha256_file

from .artifacts import (
    ARTIFACT_FILES,
    ArtifactError,
    build_processed_artifacts,
    load_full_verification_receipt,
    write_full_verification_receipt,
)
from .baselines import (
    load_artifacts,
    rank_bpr,
    rank_causal_streaming_decayed,
    rank_fit_frozen_decayed,
    rank_global_popularity,
    rank_itemcf,
    rank_random,
    train_bpr_checkpoints,
)
from .gates import (
    METHODS,
    GateError,
    canonical_json,
    derive_final_method_bundle,
    load_and_validate_selection_plan,
    validate_selection_result,
    write_selection_receipt,
)
from .metrics import common_bootstrap_indices, evaluate_topk


def _git_head_clean(root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise GateError("Tracked worktree must be clean before formal validation")
    return head


def _artifact_directory(root: Path, manifest_sha: str) -> Path:
    return root / "artifacts" / "phase1" / manifest_sha


def _verify_reusable_processed_artifacts(
    root: Path, directory: Path
) -> dict[str, Any]:
    """Reuse data artifacts across baseline-only commits, fail closed otherwise.

    The immutable cache is bound to its raw inputs, frozen contracts, Phase 0
    configuration, manifest, and the three files which actually generated it.
    A later change to ranking or reporting code is recorded separately as the
    selection code commit and must not force the expensive data build to repeat.
    """

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("Processed artifact manifest is missing")
    artifact_manifest = json.loads(manifest_path.read_text())
    fingerprint = artifact_manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ArtifactError("Processed artifact fingerprint is missing")
    phase0_manifest_path = root / "manifests/split_manifest.json"
    phase0_manifest = json.loads(phase0_manifest_path.read_text())
    if fingerprint.get("split_manifest_sha256") != sha256_file(
        phase0_manifest_path
    ):
        raise ArtifactError("Processed artifact manifest binding changed")
    if fingerprint.get("phase0_config_sha256") != sha256_file(
        root / "configs/phase0.yaml"
    ):
        raise ArtifactError("Processed artifact Phase 0 config changed")
    generation = phase0_manifest["generation_code"]
    if fingerprint.get("phase0_generation_code_sha256") != sha256_file(
        root / generation["path"]
    ):
        raise ArtifactError("Processed artifact Phase 0 generator changed")
    expected_generators = fingerprint.get("generator_file_sha256")
    if not isinstance(expected_generators, dict):
        raise ArtifactError("Processed artifact generator hashes are missing")
    for path, digest in expected_generators.items():
        if sha256_file(root / path) != digest:
            raise ArtifactError(f"Processed artifact generator changed: {path}")
    current_contracts = {
        name: sha256_file(root / entry["path"])
        for name, entry in phase0_manifest["active_contracts"].items()
    }
    if current_contracts != fingerprint.get("contract_sha256"):
        raise ArtifactError("Processed artifact contracts changed")
    expected_sources = fingerprint.get("source_file_sha256")
    if not isinstance(expected_sources, dict):
        raise ArtifactError("Processed artifact raw-source hashes are missing")
    for name, digest in expected_sources.items():
        entry = phase0_manifest["dataset"]["source_files"].get(name)
        if not isinstance(entry, dict):
            raise ArtifactError(f"Processed artifact source disappeared: {name}")
        if sha256_file(root / "data/raw" / entry["relative_path"]) != digest:
            raise ArtifactError(f"Processed artifact raw source changed: {name}")
    file_hashes = artifact_manifest.get("files")
    if not isinstance(file_hashes, dict) or set(file_hashes) != set(ARTIFACT_FILES):
        raise ArtifactError("Processed artifact file table is incomplete")
    for name, digest in file_hashes.items():
        if sha256_file(directory / name) != digest:
            raise ArtifactError(f"Processed artifact changed: {name}")
    return artifact_manifest


def _row_path(checkpoint_dir: Path, row: Mapping[str, Any]) -> Path:
    seed = "deterministic" if row["seed"] is None else str(row["seed"])
    return checkpoint_dir / f"{row['method']}__{row['config_id']}__{seed}.json"


def _save_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_row(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    for key in ("method", "config_id", "hyperparameters", "seed"):
        if canonical_json(value.get(key)) != canonical_json(expected[key]):
            raise GateError(f"Checkpoint does not match selection plan: {path}")
    if value.get("status") != "completed":
        raise GateError(f"Incomplete checkpoint must not be reused: {path}")
    return value


def _peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _evaluate(
    *,
    planned: Mapping[str, Any],
    topk: np.ndarray,
    artifacts: dict[str, Any],
    bootstrap_users: np.ndarray,
    bootstrap_indices: np.ndarray,
    elapsed: float,
    artifact_cache_hit: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    evaluated = evaluate_topk(
        topk=topk,
        query_users=queries["user"],
        target_indptr=queries["target_indptr"],
        target_indices=queries["target_indices"],
        candidate_union_count=int(queries["candidate_union_count"][0]),
        candidate_score_count=int(queries["candidate_count"].sum()),
        warm_mask=catalog["warm"],
        tail_mask=catalog["tail"],
        cold_mask=catalog["cold"],
        bootstrap_users=bootstrap_users,
        bootstrap_indices=bootstrap_indices,
    )
    return {
        **dict(planned),
        "status": "completed",
        **evaluated,
        "runtime": {
            "seconds": float(elapsed),
            "peak_memory_mb": float(_peak_memory_mb()),
            "processed_cache_hit": bool(artifact_cache_hit),
        },
        "extra": dict(extra or {}),
    }


def _select_decayed_row(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    decayed = [row for row in rows if row["method"] == "time_decayed_popularity"]
    return min(
        decayed,
        key=lambda row: (
            -row["metrics"]["Recall@100"],
            -row["metrics"]["NDCG@20"],
            -row["metrics"]["Coverage@100"],
            0 if row["hyperparameters"]["variant"] == "fit_frozen" else 1,
            row["hyperparameters"]["half_life_days"],
            canonical_json(row["hyperparameters"]),
        ),
    )


def _run_non_bpr_rows(
    *,
    plan_rows: list[Mapping[str, Any]],
    artifacts: dict[str, Any],
    checkpoint_dir: Path,
    topk_dir: Path,
    bootstrap_users: np.ndarray,
    bootstrap_indices: np.ndarray,
    artifact_cache_hit: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for planned in plan_rows:
        if planned["method"] == "bpr_mf":
            continue
        checkpoint = _row_path(checkpoint_dir, planned)
        cached = _load_row(checkpoint, planned)
        if cached is not None:
            rows.append(cached)
            continue
        started = time.perf_counter()
        method = planned["method"]
        hp = planned["hyperparameters"]
        if method == "random":
            topk = rank_random(artifacts, int(planned["seed"]))
        elif method == "global_popularity":
            topk = rank_global_popularity(artifacts)
        elif method == "time_decayed_popularity":
            if hp["variant"] == "fit_frozen":
                topk = rank_fit_frozen_decayed(artifacts, hp["half_life_days"])
            else:
                topk = rank_causal_streaming_decayed(
                    artifacts, hp["half_life_days"]
                )
        elif method == "itemcf":
            topk = rank_itemcf(
                artifacts,
                neighbor_count=hp["neighbor_count"],
                shrinkage=hp["shrinkage"],
            )
        else:
            raise GateError(f"Unsupported planned baseline: {method}")
        elapsed = time.perf_counter() - started
        row = _evaluate(
            planned=planned,
            topk=topk,
            artifacts=artifacts,
            bootstrap_users=bootstrap_users,
            bootstrap_indices=bootstrap_indices,
            elapsed=elapsed,
            artifact_cache_hit=artifact_cache_hit,
        )
        _save_row(checkpoint, row)
        if method == "time_decayed_popularity":
            topk_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(topk_dir / f"{planned['config_id']}.npz", topk=topk)
        rows.append(row)
    return rows


def _run_bpr_rows(
    *,
    plan_rows: list[Mapping[str, Any]],
    existing_rows: list[Mapping[str, Any]],
    artifacts: dict[str, Any],
    checkpoint_dir: Path,
    model_dir: Path,
    topk_dir: Path,
    bootstrap_users: np.ndarray,
    bootstrap_indices: np.ndarray,
    artifact_cache_hit: bool,
) -> list[dict[str, Any]]:
    winner = _select_decayed_row(list(existing_rows))
    fallback_file = topk_dir / f"{winner['config_id']}.npz"
    if not fallback_file.is_file():
        raise GateError("Selected time-decayed fallback Top-K cache is missing")
    fallback_topk = np.load(fallback_file)["topk"]
    planned_bpr = [row for row in plan_rows if row["method"] == "bpr_mf"]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in planned_bpr:
        hp = row["hyperparameters"]
        key = (
            hp["embedding_dim"],
            hp["learning_rate"],
            hp["l2"],
            row["seed"],
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, planned_group in groups.items():
        embedding_dim, learning_rate, l2, seed = key
        group_id = sha256_file(fallback_file)[:8] + "_" + "_".join(map(str, key))
        group_dir = model_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        missing_epochs = [
            row["hyperparameters"]["epoch"]
            for row in planned_group
            if _load_row(_row_path(checkpoint_dir, row), row) is None
        ]
        if missing_epochs:
            model_paths = {
                epoch: group_dir / f"epoch_{epoch}.npz" for epoch in (5, 10, 20)
            }
            if not all(path.is_file() for path in model_paths.values()):
                trained = train_bpr_checkpoints(
                    artifacts,
                    embedding_dim=int(embedding_dim),
                    learning_rate=float(learning_rate),
                    l2=float(l2),
                    seed=int(seed),
                )
                for epoch, (users, items) in trained.items():
                    np.savez_compressed(model_paths[epoch], users=users, items=items)
            for planned in sorted(
                planned_group, key=lambda row: row["hyperparameters"]["epoch"]
            ):
                checkpoint = _row_path(checkpoint_dir, planned)
                cached = _load_row(checkpoint, planned)
                if cached is not None:
                    output.append(cached)
                    continue
                started = time.perf_counter()
                model = np.load(model_paths[planned["hyperparameters"]["epoch"]])
                topk, fallback = rank_bpr(
                    artifacts,
                    model["users"],
                    model["items"],
                    fallback_topk=fallback_topk,
                )
                row = _evaluate(
                    planned=planned,
                    topk=topk,
                    artifacts=artifacts,
                    bootstrap_users=bootstrap_users,
                    bootstrap_indices=bootstrap_indices,
                    elapsed=time.perf_counter() - started,
                    artifact_cache_hit=artifact_cache_hit,
                    extra={
                        **fallback,
                        "fallback_method": "time_decayed_popularity",
                        "fallback_config_id": winner["config_id"],
                    },
                )
                _save_row(checkpoint, row)
                output.append(row)
        else:
            output.extend(
                _load_row(_row_path(checkpoint_dir, row), row)
                for row in planned_group
            )
    return [dict(row) for row in output if row is not None]


def _write_markdown(path: Path, result: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    selected = {method["name"]: method for method in bundle["methods"]}
    lines = [
        "# Phase 1 Temporal Validation Baselines",
        "",
        "This report uses train-fit / validation-only selection. Temporal final and Small Matrix were not opened by the experiment runner.",
        "",
        "## Run summary",
        "",
        f"- Completed selection rows: **{len(result['rows'])}**",
        f"- Orchestrator runtime: **{result['run']['seconds']:.2f} seconds**",
        f"- Peak resident memory: **{result['run']['peak_memory_mb']:.2f} MB**",
        f"- Processed artifact cache hit: **{str(result['run']['processed_cache_hit']).lower()}**",
        f"- Full protocol verifications during this run: **{result['run']['full_protocol_verification_count_this_run']}**",
        "",
        "## Selected configurations",
        "",
        "Metrics for stochastic methods are means across their frozen seeds.",
        "",
        "| Method | Hyperparameters | Recall@100 | NDCG@20 | Coverage@100 | WarmRecall@100 | TailRecall@100 | ColdRecall@100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        chosen = selected[method]
        rows = [
            row
            for row in result["rows"]
            if row["method"] == method and row["config_id"] == chosen["config_id"]
        ]
        means = {
            metric: sum(row["metrics"][metric] for row in rows) / len(rows)
            for metric in (
                "Recall@100",
                "NDCG@20",
                "Coverage@100",
                "WarmRecall@100",
                "TailRecall@100",
                "ColdRecall@100",
            )
        }
        lines.append(
            f"| {method} | `{canonical_json(chosen['hyperparameters'])}` | "
            f"{means['Recall@100']:.6f} | {means['NDCG@20']:.6f} | "
            f"{means['Coverage@100']:.6f} | {means['WarmRecall@100']:.6f} | "
            f"{means['TailRecall@100']:.6f} | {means['ColdRecall@100']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selected configuration confidence intervals",
            "",
            "Intervals are paired user-cluster bootstrap intervals for the primary query-macro estimator. WarmRecall is reported above as a point estimate; the frozen metrics contract requires intervals for overall, tail, and cold metrics.",
            "",
            "| Method | Seed | Recall@100 | 95% CI | NDCG@20 | 95% CI | TailRecall@100 95% CI | ColdRecall@100 95% CI |",
            "|---|---:|---:|---|---:|---|---|---|",
        ]
    )
    for method in METHODS:
        chosen = selected[method]
        for row in result["rows"]:
            if row["method"] != method or row["config_id"] != chosen["config_id"]:
                continue
            intervals = row["bootstrap_95_percent_intervals"]
            recall_ci = intervals["Recall@100"]
            ndcg_ci = intervals["NDCG@20"]
            tail_ci = intervals["TailRecall@100"]
            cold_ci = intervals["ColdRecall@100"]
            lines.append(
                f"| {method} | {row['seed']} | {row['metrics']['Recall@100']:.6f} | "
                f"[{recall_ci[0]:.6f}, {recall_ci[1]:.6f}] | "
                f"{row['metrics']['NDCG@20']:.6f} | "
                f"[{ndcg_ci[0]:.6f}, {ndcg_ci[1]:.6f}] | "
                f"[{tail_ci[0]:.6f}, {tail_ci[1]:.6f}] | "
                f"[{cold_ci[0]:.6f}, {cold_ci[1]:.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Complete configuration and seed results",
            "",
            "| Method | Config | Seed | Recall@100 | NDCG@20 | Coverage@100 | TailRecall@100 | ColdRecall@100 | Runtime s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['method']} | `{row['config_id']}` | {row['seed']} | "
            f"{row['metrics']['Recall@100']:.6f} | {row['metrics']['NDCG@20']:.6f} | "
            f"{row['metrics']['Coverage@100']:.6f} | {row['metrics']['TailRecall@100']:.6f} | "
            f"{row['metrics']['ColdRecall@100']:.6f} | {row['runtime']['seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Protocol: `{result['protocol_revision']}`",
            f"- Manifest: `{result['split_manifest_sha256']}`",
            f"- Processed artifacts: `{result['hashes']['processed_artifact_manifest_sha256']}`",
            f"- Code commit: `{result['code_commit']}`",
            "- Temporal final accessed: **no**",
            "- Small Matrix accessed: **no**",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_selection(repo_root: str | Path) -> dict[str, Any]:
    """Run all 97 planned validation rows under one verified orchestrator."""

    root = Path(repo_root).resolve()
    code_commit = _git_head_clean(root)
    plan = load_and_validate_selection_plan(root)
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    artifact_dir = _artifact_directory(root, manifest_sha)
    if artifact_dir.exists():
        artifact_manifest = _verify_reusable_processed_artifacts(root, artifact_dir)
        artifact_cache_hit = True
    else:
        try:
            verification = load_full_verification_receipt(root)
        except ArtifactError:
            verification = authorize_baseline_selection(
                fit_splits=("train",), evaluation_split="validation", repo_root=root
            )
            write_full_verification_receipt(root, verification)
        artifact_manifest, artifact_cache_hit = build_processed_artifacts(
            root, verification
        )
    artifacts = load_artifacts(artifact_dir)
    bootstrap_users, bootstrap_indices = common_bootstrap_indices(
        artifacts["queries"]["user"]
    )
    checkpoints = artifact_dir / "selection_checkpoints"
    topk_dir = artifact_dir / "topk"
    started = time.perf_counter()
    rows = _run_non_bpr_rows(
        plan_rows=list(plan.rows),
        artifacts=artifacts,
        checkpoint_dir=checkpoints,
        topk_dir=topk_dir,
        bootstrap_users=bootstrap_users,
        bootstrap_indices=bootstrap_indices,
        artifact_cache_hit=artifact_cache_hit,
    )
    rows.extend(
        _run_bpr_rows(
            plan_rows=list(plan.rows),
            existing_rows=rows,
            artifacts=artifacts,
            checkpoint_dir=checkpoints,
            model_dir=artifact_dir / "bpr_models",
            topk_dir=topk_dir,
            bootstrap_users=bootstrap_users,
            bootstrap_indices=bootstrap_indices,
            artifact_cache_hit=artifact_cache_hit,
        )
    )
    row_map = {
        (row["method"], row["config_id"], row["seed"]): row for row in rows
    }
    ordered_rows = [
        row_map[(row["method"], row["config_id"], row["seed"])]
        for row in plan.rows
    ]
    phase0_manifest = json.loads((root / "manifests/split_manifest.json").read_text())
    contract_hashes = {
        name: entry["sha256"]
        for name, entry in phase0_manifest["active_contracts"].items()
    }
    result = {
        "schema_version": 1,
        "result_scope": "selection_validation",
        "protocol_revision": plan.protocol_revision,
        "selection_plan_sha256": plan.sha256,
        "split_manifest_sha256": manifest_sha,
        "fit_splits": ["train"],
        "evaluation_split": "validation",
        "code_commit": code_commit,
        "rows": ordered_rows,
        "hashes": {
            "processed_artifact_manifest_sha256": sha256_file(
                artifact_dir / "manifest.json"
            ),
            "contracts": contract_hashes,
            "evaluator": {
                "path": "scripts/run_phase1_baselines.py",
                "sha256": sha256_file(root / "scripts/run_phase1_baselines.py"),
            },
        },
        "run": {
            "seconds": float(time.perf_counter() - started),
            "peak_memory_mb": float(_peak_memory_mb()),
            "processed_cache_hit": bool(artifact_cache_hit),
            "full_protocol_verification_count_this_run": 0 if artifact_cache_hit else 1,
        },
        "scope_access": {
            "temporal_final": False,
            "small_matrix": False,
        },
    }
    report_dir = root / "reports/phase1"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_path = report_dir / "validation_baselines.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    validate_selection_result(root, result_path, plan=plan)
    bundle_path = report_dir / "final_method_bundle.json"
    bundle = derive_final_method_bundle(root, result_path, bundle_path)
    receipt = write_selection_receipt(root, result_path, bundle_path)
    _write_markdown(report_dir / "validation_baselines.md", result, bundle)
    return {
        "result": result,
        "final_method_bundle": bundle,
        "selection_receipt": str(receipt),
    }


def evaluate_frozen_final(
    *, repo_root: Path, bundle: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Fixed final evaluator hook; Phase 1 intentionally has no final artifacts."""

    raise GateError(
        "Temporal-final evaluator is registered and hash-bound, but final artifact "
        "construction is intentionally disabled during Phase 1 selection"
    )
