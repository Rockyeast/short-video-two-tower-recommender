#!/usr/bin/env python3
"""Train one frozen LightGBM reranker and evaluate held-out validation users."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kuairec_fully_observed import evaluate_retrieval
from kuairec_fully_observed.hybrid import (
    RRF_RANK_CONSTANT,
    weighted_reciprocal_rank_fusion,
)
from kuairec_fully_observed.reranking import (
    RERANK_FEATURE_NAMES,
    RerankFeatureBuilder,
    rank_with_model,
    reranker_gate,
    stable_fit_mask,
    subset_queries,
    train_lightgbm_ranker,
)
from scripts.run_phase_b3a_hybrid import (
    BPR_CHECKPOINT_SHA256,
    TWO_TOWER_CHECKPOINT_SHA256,
    load_frozen_validation_routes,
)


EXPECTED_CONFIG: dict[str, Any] = {
    "phase": "phase-b5a-lightgbm-reranker",
    "scope": {
        "base_models_fit": "canonical_big_train",
        "reranker_fit": "stable_70_percent_of_warm_big_validation_users",
        "reranker_evaluate": (
            "held_out_30_percent_of_warm_big_validation_users"
        ),
        "forbidden": [
            "small_matrix",
            "temporal_final",
            "final_refit_artifacts",
        ],
    },
    "split": {
        "salt": "phase-b5a-reranker-split-v1",
        "fit_percent": 70,
    },
    "candidates": {
        "two_tower_top_k": 500,
        "bpr_top_k": 500,
        "union": "stable_unique_union",
        "training_negative_cap_per_query": 256,
        "training_negative_order": (
            "frozen_hybrid_score_then_item_id"
        ),
        "retain_all_retrieved_positives": True,
    },
    "features": list(RERANK_FEATURE_NAMES),
    "model": {
        "implementation": "lightgbm.LGBMRanker",
        "objective": "lambdarank",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "random_state": 20260724,
        "n_jobs": 8,
        "deterministic": True,
        "force_col_wise": True,
    },
    "evaluation": {
        "output_k": 100,
        "tie_break": "ascending_item_id",
        "baseline": "frozen_hybrid_alpha_0.75",
        "gate": {
            "ndcg20_strictly_higher": True,
            "recall100_retention": 0.98,
            "coverage100_retention": 0.90,
        },
    },
    "claims": {
        "single_configuration": True,
        "single_split": True,
        "no_hyperparameter_search": True,
        "effectiveness_claim_scope": (
            "held_out_big_validation_users_only"
        ),
    },
}


def validate_config(config: dict[str, Any]) -> None:
    if config != EXPECTED_CONFIG:
        raise ValueError("Phase B5A frozen configuration changed")


def _assert_clean_tree(repo_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if status:
        raise RuntimeError("Phase B5A must start from a clean source commit")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _membership_sha(values: np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _metric_change(
    actual: dict[str, float], baseline: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for name, value in actual.items():
        reference = float(baseline[name])
        result[name] = {
            "reranker": float(value),
            "hybrid": reference,
            "absolute_change": float(value - reference),
            "relative_change": (
                None if reference == 0.0 else float((value - reference) / reference)
            ),
        }
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, result in report["results"].items():
        metrics = result["metrics"]
        rows.append(
            "| {name} | {r20:.6f} | {r50:.6f} | {r100:.6f} | "
            "{ndcg:.6f} | {coverage:.6f} | {cold:.6f} |".format(
                name=name,
                r20=metrics["Recall@20"],
                r50=metrics["Recall@50"],
                r100=metrics["Recall@100"],
                ndcg=metrics["NDCG@20"],
                coverage=metrics["Coverage@100"],
                cold=metrics["Data-Cold Recall@100"],
            )
        )
    gate = report["gate"]
    lines = [
        "# Phase B5A LightGBM Reranker Validation",
        "",
        "A single frozen LambdaRank configuration was trained on a stable 70% "
        "user split of Big validation and evaluated on the disjoint 30% split.",
        "",
        "| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
        "Coverage@100 | Data-Cold Recall@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Gate",
        "",
        f"- Passed: `{str(gate['passed']).lower()}`",
        f"- Checks: `{gate['checks']}`",
        "- Objective: improve NDCG@20 while retaining 98% of Hybrid "
        "Recall@100 and 90% of Hybrid Coverage@100.",
        "",
        "## Split and training",
        "",
        f"- Reranker-fit users: `{report['split']['fit_user_count']}`",
        f"- Held-out evaluation users: `{report['split']['eval_user_count']}`",
        f"- Training groups accepted: "
        f"`{report['training']['accepted_query_count']}`",
        f"- Fit queries without a retrieved positive: "
        f"`{report['training']['skipped_no_retrieved_positive']}`",
        f"- Training candidate rows: `{report['training']['candidate_row_count']}`",
        "",
        "## Claim boundary",
        "",
        "This is a validation-development experiment, not a new sealed test. "
        "The base retrieval models were fit only on Big train; reranker labels "
        "came only from the fit-user subset. Small Matrix, temporal final, and "
        "final-refit artifacts were not accessed.",
        "",
        f"- Total wall time: `{report['runtime']['wall_time_s']:.3f} s`",
        f"- Peak RSS: `{report['runtime']['peak_rss_mb']:.2f} MiB`",
        "",
    ]
    text = "\n".join(lines)
    if "/home/" in text:
        raise RuntimeError("Generated Markdown contains a host path")
    return text


def run(
    *,
    repo_root: Path,
    config_path: Path,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    two_tower_checkpoint: Path,
    bpr_checkpoint: Path,
    model_output: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    code_commit = _assert_clean_tree(repo_root)
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)

    routes = load_frozen_validation_routes(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        caption_cache_path=caption_cache_path,
        caption_metadata_path=caption_metadata_path,
        two_tower_checkpoint=two_tower_checkpoint,
        bpr_checkpoint=bpr_checkpoint,
    )
    warm_indices = np.flatnonzero(routes.queries.warm_user_mask)
    warm_users = routes.queries.user_ids[warm_indices]
    fit_mask = stable_fit_mask(
        warm_users,
        salt=config["split"]["salt"],
        fit_percent=config["split"]["fit_percent"],
    )
    fit_indices = warm_indices[fit_mask]
    eval_indices = warm_indices[~fit_mask]
    if not len(fit_indices) or not len(eval_indices):
        raise RuntimeError("Stable reranker split produced an empty partition")
    if set(routes.queries.user_ids[fit_indices]) & set(
        routes.queries.user_ids[eval_indices]
    ):
        raise RuntimeError("Reranker fit/evaluation users overlap")

    static_by_item = routes.static.frame.set_index("video_id")
    catalog_static = static_by_item.reindex(routes.queries.catalog)
    if catalog_static.index.has_duplicates or catalog_static["category_ids"].isna().any():
        raise RuntimeError("Static catalog features do not align")
    catalog_categories = np.asarray(
        catalog_static["category_ids"].tolist(), dtype=np.int64
    )
    category_lookup = {
        int(item): tuple(int(value) for value in categories)
        for item, categories in zip(
            routes.static.frame["video_id"],
            routes.static.frame["category_ids"],
            strict=True,
        )
    }
    popularity_scores = np.asarray(
        [
            routes.popularity.scores.get(int(item), 0.0)
            for item in routes.queries.catalog
        ],
        dtype=np.float32,
    )
    video_duration = (
        catalog_static["video_duration"]
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    builder = RerankFeatureBuilder(
        queries=routes.queries,
        two_tower_topk=routes.two_tower_top500,
        bpr_topk=routes.bpr_top500,
        two_tower_user_vectors=routes.two_tower_user_vectors,
        two_tower_catalog_vectors=routes.two_tower_catalog_vectors,
        bpr_user_vectors=routes.bpr_user_vectors,
        bpr_catalog_vectors=routes.bpr_catalog_vectors,
        popularity_scores=popularity_scores,
        catalog_categories=catalog_categories,
        category_lookup=category_lookup,
        data_cold_mask=np.isin(
            routes.queries.catalog, routes.data_cold_items
        ),
        video_duration=video_duration,
        alpha=0.75,
        rank_constant=RRF_RANK_CONSTANT,
    )
    train_dataset = builder.build(
        fit_indices,
        training_negative_cap=config["candidates"][
            "training_negative_cap_per_query"
        ],
        require_retrieved_positive=True,
    )
    eval_dataset = builder.build(
        eval_indices,
        training_negative_cap=None,
        require_retrieved_positive=False,
    )
    if not np.array_equal(eval_dataset.query_indices, eval_indices):
        raise RuntimeError("Held-out evaluation queries changed")

    model_parameters = dict(config["model"])
    model_parameters.pop("implementation")
    model = train_lightgbm_ranker(
        train_dataset, parameters=model_parameters
    )
    reranked_top100 = rank_with_model(
        model,
        eval_dataset,
        output_k=config["evaluation"]["output_k"],
    )
    eval_queries = subset_queries(routes.queries, eval_indices)
    hybrid_top100 = weighted_reciprocal_rank_fusion(
        routes.two_tower_top500[eval_indices],
        routes.bpr_top500[eval_indices],
        candidates=eval_queries.candidates,
        alpha=0.75,
        output_k=config["evaluation"]["output_k"],
        rank_constant=RRF_RANK_CONSTANT,
    )
    hybrid_result = evaluate_retrieval(
        hybrid_top100,
        eval_queries,
        data_cold_item_ids=routes.data_cold_items,
    )
    reranker_result = evaluate_retrieval(
        reranked_top100,
        eval_queries,
        data_cold_item_ids=routes.data_cold_items,
    )
    gate = reranker_gate(
        reranker_metrics=reranker_result["metrics"],
        hybrid_metrics=hybrid_result["metrics"],
        recall_retention=config["evaluation"]["gate"][
            "recall100_retention"
        ],
        coverage_retention=config["evaluation"]["gate"][
            "coverage100_retention"
        ],
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(model_output)
    importances = {
        name: float(value)
        for name, value in zip(
            RERANK_FEATURE_NAMES,
            model.booster_.feature_importance(importance_type="gain"),
            strict=True,
        )
    }
    report: dict[str, Any] = {
        "phase": config["phase"],
        "status": "gate_passed" if gate["passed"] else "gate_failed",
        "claim_boundary": {
            "validation_development_result": True,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "final_refit_artifacts_used": False,
            "hyperparameter_search_performed": False,
            "configuration_count": 1,
            "split_count": 1,
        },
        "configuration": config,
        "split": {
            "salt": config["split"]["salt"],
            "fit_percent": config["split"]["fit_percent"],
            "warm_user_count": int(len(warm_indices)),
            "fit_user_count": int(len(fit_indices)),
            "eval_user_count": int(len(eval_indices)),
            "fit_user_membership_sha256": _membership_sha(
                routes.queries.user_ids[fit_indices]
            ),
            "eval_user_membership_sha256": _membership_sha(
                routes.queries.user_ids[eval_indices]
            ),
            "user_overlap_count": 0,
        },
        "training": {
            "accepted_query_count": int(len(train_dataset.query_indices)),
            "skipped_no_retrieved_positive": int(
                len(fit_indices) - len(train_dataset.query_indices)
            ),
            "candidate_row_count": int(len(train_dataset.labels)),
            "positive_row_count": int(train_dataset.labels.sum()),
            "negative_row_count": int(
                len(train_dataset.labels) - train_dataset.labels.sum()
            ),
        },
        "results": {
            "Frozen Hybrid alpha=0.75": hybrid_result,
            "LightGBM LambdaRank": reranker_result,
        },
        "gate": gate,
        "comparison": _metric_change(
            reranker_result["metrics"], hybrid_result["metrics"]
        ),
        "feature_importance_gain": importances,
        "artifacts": {
            "code_commit_at_run": code_commit,
            "input_tree_clean_at_start": True,
            "two_tower_checkpoint_sha256": (
                TWO_TOWER_CHECKPOINT_SHA256
            ),
            "bpr_checkpoint_sha256": BPR_CHECKPOINT_SHA256,
            "model_locator": "ARTIFACT_DIR/phase_b5a/lightgbm_reranker.txt",
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(
        token in serialized
        for token in ("/home/", "gho_", "hf_", "MODAL_TOKEN")
    ):
        raise RuntimeError("Generated report contains host path or credential")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(serialized + "\n")
    report_markdown.write_text(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b5a_lightgbm_reranker.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument("--caption-metadata", type=Path, required=True)
    parser.add_argument("--two-tower-checkpoint", type=Path, required=True)
    parser.add_argument("--bpr-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/phase_b5a/lightgbm_reranker.txt"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b5a/reranker_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b5a/reranker_validation.md"),
    )
    args = parser.parse_args()
    report = run(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        two_tower_checkpoint=args.two_tower_checkpoint.resolve(),
        bpr_checkpoint=args.bpr_checkpoint.resolve(),
        model_output=args.model_output.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
