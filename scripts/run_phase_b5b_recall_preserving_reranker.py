#!/usr/bin/env python3
"""Rerank only the frozen Hybrid Top-100, preserving retrieval coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
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
    rank_with_model,
    stable_fit_mask,
    subset_queries,
    train_lightgbm_ranker,
)
from scripts.run_phase_b3a_hybrid import (
    BPR_CHECKPOINT_SHA256,
    TWO_TOWER_CHECKPOINT_SHA256,
    load_frozen_validation_routes,
)
from scripts.run_phase_b5a_lightgbm_reranker import (
    _assert_clean_tree,
    _membership_sha,
    _metric_change,
    build_feature_builder,
)


CONFIG_SHA256 = (
    "e030de7ca74fff85b2602c6b48da001f3f33fad7bfa52e46aa05142be6631adf"
)


def load_config(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Phase B5B frozen configuration changed")
    config = yaml.safe_load(path.read_text())
    if (
        config["phase"] != "phase-b5b-recall-preserving-reranker"
        or config["candidates"]["fixed_rerank_top_k"] != 100
        or config["candidates"]["hybrid_alpha"] != 0.75
        or config["evaluation"]["gate"]
        != {
            "ndcg20_strictly_higher": True,
            "recall100_exactly_equal": True,
            "coverage100_exactly_equal": True,
        }
    ):
        raise ValueError("Phase B5B frozen semantics changed")
    return config


def topk_sets_match(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    return all(
        np.array_equal(
            np.sort(left[row][left[row] >= 0]),
            np.sort(right[row][right[row] >= 0]),
        )
        for row in range(len(left))
    )


def recall_preserving_gate(
    reranker_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "ndcg20_strictly_higher": (
            reranker_metrics["NDCG@20"] > hybrid_metrics["NDCG@20"]
        ),
        "recall100_exactly_equal": bool(
            np.isclose(
                reranker_metrics["Recall@100"],
                hybrid_metrics["Recall@100"],
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "coverage100_exactly_equal": bool(
            np.isclose(
                reranker_metrics["Coverage@100"],
                hybrid_metrics["Coverage@100"],
                rtol=0.0,
                atol=1e-15,
            )
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {name: bool(value) for name, value in checks.items()},
    }


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
    return "\n".join(
        [
            "# Phase B5B Recall-Preserving Local Reranker",
            "",
            "The frozen Hybrid selects Top-100 candidates. LightGBM may only "
            "change their order; it cannot add or remove a Top-100 item.",
            "",
            "| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
            "Coverage@100 | Data-Cold Recall@100 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"- Gate passed: `{str(report['gate']['passed']).lower()}`",
            f"- Checks: `{report['gate']['checks']}`",
            f"- Fit users: `{report['split']['fit_user_count']}`",
            f"- Evaluation users: `{report['split']['eval_user_count']}`",
            f"- Accepted training groups: "
            f"`{report['training']['accepted_query_count']}`",
            f"- Wall time: `{report['runtime']['wall_time_s']:.3f} s`",
            f"- Peak RSS: `{report['runtime']['peak_rss_mb']:.2f} MiB`",
            "",
            "This is an adaptive Big-validation development iteration after "
            "the B5A failure. It is not a sealed or untouched test. Small "
            "Matrix, temporal final, and final-refit artifacts were not used.",
            "",
        ]
    )


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
    config = load_config(config_path)
    routes = load_frozen_validation_routes(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        caption_cache_path=caption_cache_path,
        caption_metadata_path=caption_metadata_path,
        two_tower_checkpoint=two_tower_checkpoint,
        bpr_checkpoint=bpr_checkpoint,
    )
    hybrid_top100 = weighted_reciprocal_rank_fusion(
        routes.two_tower_top500,
        routes.bpr_top500,
        candidates=routes.queries.candidates,
        alpha=config["candidates"]["hybrid_alpha"],
        output_k=config["candidates"]["fixed_rerank_top_k"],
        rank_constant=config["candidates"]["rrf_rank_constant"],
    )
    warm_indices = np.flatnonzero(routes.queries.warm_user_mask)
    fit_mask = stable_fit_mask(
        routes.queries.user_ids[warm_indices],
        salt=config["split"]["salt"],
        fit_percent=config["split"]["fit_percent"],
    )
    fit_indices = warm_indices[fit_mask]
    eval_indices = warm_indices[~fit_mask]
    builder = build_feature_builder(
        routes, restricted_candidates=hybrid_top100
    )
    train_dataset = builder.build(
        fit_indices,
        training_negative_cap=None,
        require_retrieved_positive=True,
    )
    eval_dataset = builder.build(
        eval_indices,
        training_negative_cap=None,
        require_retrieved_positive=False,
    )
    if not np.array_equal(eval_dataset.query_indices, eval_indices):
        raise RuntimeError("Held-out evaluation queries changed")
    parameters = dict(config["model"])
    parameters.pop("implementation")
    model = train_lightgbm_ranker(
        train_dataset, parameters=parameters
    )
    reranked = rank_with_model(
        model,
        eval_dataset,
        output_k=config["evaluation"]["output_k"],
    )
    baseline_top100 = hybrid_top100[eval_indices]
    if not topk_sets_match(reranked, baseline_top100):
        raise RuntimeError("Local reranker changed the frozen Top-100 set")
    eval_queries = subset_queries(routes.queries, eval_indices)
    baseline_result = evaluate_retrieval(
        baseline_top100,
        eval_queries,
        data_cold_item_ids=routes.data_cold_items,
    )
    reranker_result = evaluate_retrieval(
        reranked,
        eval_queries,
        data_cold_item_ids=routes.data_cold_items,
    )
    gate = recall_preserving_gate(
        reranker_result["metrics"], baseline_result["metrics"]
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(model_output)
    report: dict[str, Any] = {
        "phase": config["phase"],
        "status": "gate_passed" if gate["passed"] else "gate_failed",
        "claim_boundary": {
            "adaptive_validation_development": True,
            "sealed_test_claim": False,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "final_refit_artifacts_used": False,
            "hyperparameter_search_performed": False,
            "configuration_count": 1,
        },
        "configuration": config,
        "split": {
            "fit_user_count": int(len(fit_indices)),
            "eval_user_count": int(len(eval_indices)),
            "fit_user_membership_sha256": _membership_sha(
                routes.queries.user_ids[fit_indices]
            ),
            "eval_user_membership_sha256": _membership_sha(
                routes.queries.user_ids[eval_indices]
            ),
        },
        "training": {
            "accepted_query_count": int(len(train_dataset.query_indices)),
            "skipped_no_top100_positive": int(
                len(fit_indices) - len(train_dataset.query_indices)
            ),
            "candidate_row_count": int(len(train_dataset.labels)),
            "positive_row_count": int(train_dataset.labels.sum()),
        },
        "retrieval_preservation": {
            "top100_item_sets_exactly_equal": True,
            "query_count": int(len(eval_indices)),
        },
        "results": {
            "Frozen Hybrid alpha=0.75": baseline_result,
            "Local LightGBM LambdaRank": reranker_result,
        },
        "gate": gate,
        "comparison": _metric_change(
            reranker_result["metrics"], baseline_result["metrics"]
        ),
        "feature_importance_gain": {
            name: float(value)
            for name, value in zip(
                RERANK_FEATURE_NAMES,
                model.booster_.feature_importance(importance_type="gain"),
                strict=True,
            )
        },
        "artifacts": {
            "code_commit_at_run": code_commit,
            "input_tree_clean_at_start": True,
            "two_tower_checkpoint_sha256": TWO_TOWER_CHECKPOINT_SHA256,
            "bpr_checkpoint_sha256": BPR_CHECKPOINT_SHA256,
            "model_locator": (
                "ARTIFACT_DIR/phase_b5b/local_lightgbm_reranker.txt"
            ),
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if "/home/" in serialized:
        raise RuntimeError("Generated report contains a host path")
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
        default=Path(
            "configs/phase_b5b_recall_preserving_reranker.yaml"
        ),
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
        default=Path(
            "artifacts/phase_b5b/local_lightgbm_reranker.txt"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b5b/local_reranker_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b5b/local_reranker_validation.md"),
    )
    args = parser.parse_args()
    result = run(
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
