#!/usr/bin/env python3
"""Evaluate a bounded Two-Tower + SASRec + BPR weighted-RRF grid."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kuairec_fully_observed import evaluate_retrieval
from kuairec_fully_observed.provenance import sha256_file
from kuairec_fully_observed.sasrec_adapter import (
    build_recbole_sasrec,
    rank_sasrec,
)
from kuairec_fully_observed.three_route_hybrid import (
    FROZEN_THREE_ROUTE_WEIGHTS,
    RRF_RANK_CONSTANT,
    select_three_route_hybrid,
    weighted_three_route_rrf,
)
from scripts.run_phase_b3a_hybrid import (
    BPR_REFERENCE,
    TOPK_PER_ROUTE,
    TWO_TOWER_REFERENCE,
    load_frozen_validation_routes,
)
from scripts.run_phase_b6a_recbole_sasrec import (
    _load_config as load_sasrec_config,
)
from scripts.run_phase_b6a_recbole_sasrec import _prepare_inputs


SASREC_CHECKPOINT_SHA256 = (
    "914a27656b880655587e8b83f84ab99d6832c93913bc23906b0ede59b6d7ffd3"
)
SASREC_REFERENCE = {
    "Recall@20": 0.030656688963654945,
    "Recall@50": 0.052638749987949346,
    "Recall@100": 0.07756014970064312,
    "NDCG@20": 0.03746169500439598,
    "Coverage@100": 0.2528563801388147,
    "Data-Cold Recall@100": 0.0,
}
TWO_ROUTE_REFERENCE = {
    "Recall@20": 0.015642887713368626,
    "Recall@50": 0.03700175626274963,
    "Recall@100": 0.07221290435417195,
    "NDCG@20": 0.015340913271705538,
    "Coverage@100": 0.5711692471970101,
    "Data-Cold Recall@100": 0.060356233382505925,
}
OUTPUT_K = 100


def _load_experiment_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    expected_weights = [
        {"two_tower": weights[0], "sasrec": weights[1], "bpr": weights[2]}
        for weights in FROZEN_THREE_ROUTE_WEIGHTS
    ]
    if config != {
        "phase": "phase-b6b-three-route-hybrid",
        "scope": {
            "evaluate": "reused_big_validation_development",
            "forbidden": [
                "small_matrix",
                "temporal_final",
                "final_refit",
                "model_training",
            ],
        },
        "routes": [
            "two_tower_epoch_1",
            "sasrec_epoch_5",
            "bpr_epoch_20",
        ],
        "route_top_k": 500,
        "output_k": 100,
        "rank_constant": 60,
        "weights": expected_weights,
        "selection": {
            "reference": "frozen_two_route_hybrid_alpha_0.75",
            "minimum_recall_fraction": 0.98,
            "minimum_coverage_fraction": 0.90,
            "minimum_data_cold_recall_fraction": 0.90,
            "objective": "highest_ndcg_at_20",
        },
        "claims": {
            "adaptive_validation_development": True,
            "sealed_test_claim": False,
            "significance_claim": False,
            "one_seed_per_route": True,
        },
    }:
        raise RuntimeError("Phase B6B frozen experiment configuration changed")
    return config


def _assert_queries_aligned(left: Any, right: Any) -> None:
    for field in ("user_ids", "catalog", "warm_user_mask"):
        if not np.array_equal(getattr(left, field), getattr(right, field)):
            raise RuntimeError(f"route query alignment changed: {field}")
    for field in ("candidates", "relevant"):
        left_rows = getattr(left, field)
        right_rows = getattr(right, field)
        if len(left_rows) != len(right_rows) or any(
            not np.array_equal(a, b)
            for a, b in zip(left_rows, right_rows, strict=True)
        ):
            raise RuntimeError(f"route query alignment changed: {field}")


def _assert_metrics(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    route: str,
) -> None:
    for name, value in expected.items():
        if not np.isclose(actual[name], value, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{route} {name} changed: {actual[name]} != {value}"
            )


def _comparison(
    candidate: dict[str, float], reference: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    names = (
        "Recall@20",
        "Recall@50",
        "Recall@100",
        "NDCG@20",
        "Coverage@100",
        "Data-Cold Recall@100",
    )
    return {
        name: {
            "absolute": candidate[name] - reference[name],
            "relative_percent": (
                None
                if reference[name] == 0
                else (candidate[name] / reference[name] - 1.0) * 100.0
            ),
        }
        for name in names
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
    selected = report["selection"]["selected_weights"]
    conclusion = (
        "No three-route candidate passed all frozen preservation gates."
        if selected is None
        else f"Selected weights (TT, SASRec, BPR): `{selected}`."
    )
    return "\n".join(
        [
            "# Phase B6B Three-Route Hybrid",
            "",
            "This is adaptive development on the already reused Big validation "
            "set. It is not sealed-test or statistical-significance evidence.",
            "",
            "| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
            "Coverage@100 | Data-Cold Recall@100 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            conclusion,
            "",
            "Selection first preserves 98% of Recall@100, 90% of "
            "Coverage@100, and 90% of Data-Cold Recall@100 from the frozen "
            "Two-Tower+BPR hybrid; it then maximizes NDCG@20.",
            "",
            "No model was trained. Small Matrix and temporal final were not "
            "accessed.",
            "",
        ]
    )


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    two_tower_checkpoint: Path,
    bpr_checkpoint: Path,
    sasrec_checkpoint: Path,
    sasrec_config_path: Path,
    experiment_config_path: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    _load_experiment_config(experiment_config_path)
    sasrec_config = load_sasrec_config(sasrec_config_path)
    if sha256_file(sasrec_checkpoint) != SASREC_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen SASRec epoch 5 checkpoint SHA changed")

    routes = load_frozen_validation_routes(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        caption_cache_path=caption_cache_path,
        caption_metadata_path=caption_metadata_path,
        two_tower_checkpoint=two_tower_checkpoint,
        bpr_checkpoint=bpr_checkpoint,
    )
    sasrec_inputs = _prepare_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        max_history=int(sasrec_config["training"]["max_history"]),
        max_examples=None,
    )
    _assert_queries_aligned(routes.queries, sasrec_inputs["queries"])
    if not np.array_equal(
        routes.data_cold_items, sasrec_inputs["data_cold_items"]
    ):
        raise RuntimeError("SASRec data-cold membership changed")

    payload = torch.load(
        sasrec_checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("epoch") != 5 or payload.get("config") != sasrec_config:
        raise RuntimeError("SASRec checkpoint identity changed")
    model = build_recbole_sasrec(
        num_event_items=len(sasrec_inputs["video_ids"]),
        max_history=int(sasrec_config["training"]["max_history"]),
        model_config=sasrec_config["model"],
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model"], strict=True)
    fallback = routes.popularity.rank(routes.queries, k=TOPK_PER_ROUTE)
    sasrec_top500 = rank_sasrec(
        model,
        queries=routes.queries,
        sequences=sasrec_inputs["sequences"],
        sequence_lengths=sasrec_inputs["sequence_lengths"],
        video_ids=sasrec_inputs["video_ids"],
        fallback_topk=fallback,
        device=torch.device("cpu"),
        k=TOPK_PER_ROUTE,
        batch_size=128,
    )

    route_results = {
        "BPR": evaluate_retrieval(
            routes.bpr_top500,
            routes.queries,
            data_cold_item_ids=routes.data_cold_items,
        ),
        "Two-Tower": evaluate_retrieval(
            routes.two_tower_top500,
            routes.queries,
            data_cold_item_ids=routes.data_cold_items,
        ),
        "SASRec": evaluate_retrieval(
            sasrec_top500,
            routes.queries,
            data_cold_item_ids=routes.data_cold_items,
        ),
    }
    _assert_metrics(route_results["BPR"]["metrics"], BPR_REFERENCE, route="BPR")
    _assert_metrics(
        route_results["Two-Tower"]["metrics"],
        TWO_TOWER_REFERENCE,
        route="Two-Tower",
    )
    _assert_metrics(
        route_results["SASRec"]["metrics"],
        SASREC_REFERENCE,
        route="SASRec",
    )

    candidate_results: dict[
        tuple[float, float, float], dict[str, Any]
    ] = {}
    for weights in FROZEN_THREE_ROUTE_WEIGHTS:
        fused = weighted_three_route_rrf(
            routes.two_tower_top500,
            sasrec_top500,
            routes.bpr_top500,
            candidates=routes.queries.candidates,
            weights=weights,
            output_k=OUTPUT_K,
            rank_constant=RRF_RANK_CONSTANT,
        )
        candidate_results[weights] = evaluate_retrieval(
            fused,
            routes.queries,
            data_cold_item_ids=routes.data_cold_items,
        )
    selection = select_three_route_hybrid(
        reference_metrics=TWO_ROUTE_REFERENCE,
        candidate_metrics={
            weights: result["metrics"]
            for weights, result in candidate_results.items()
        },
    )
    selected_result = (
        None
        if selection.selected_weights is None
        else candidate_results[selection.selected_weights]
    )
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    report: dict[str, Any] = {
        "phase": "phase-b6b-three-route-hybrid",
        "status": (
            "hybrid_selected"
            if selection.selected_weights is not None
            else "hybrid_not_selected"
        ),
        "configuration": {
            "route_top_k": TOPK_PER_ROUTE,
            "output_k": OUTPUT_K,
            "rank_constant": RRF_RANK_CONSTANT,
            "weights": [list(values) for values in FROZEN_THREE_ROUTE_WEIGHTS],
            "route_order": ["two_tower", "sasrec", "bpr"],
            "tie_break": "ascending_item_id",
        },
        "counts": routes.validation_counts,
        "results": {
            **route_results,
            **{
                "Hybrid TT={:.2f} SASRec={:.2f} BPR={:.2f}".format(*weights):
                result
                for weights, result in candidate_results.items()
            },
        },
        "selection": {
            "reference": {
                "name": "frozen_two_route_hybrid_alpha_0.75",
                "metrics": TWO_ROUTE_REFERENCE,
            },
            "recall_minimum": selection.recall_minimum,
            "coverage_minimum": selection.coverage_minimum,
            "data_cold_minimum": selection.data_cold_minimum,
            "eligible_weights": [
                list(values) for values in selection.eligible_weights
            ],
            "selected_weights": (
                None
                if selection.selected_weights is None
                else list(selection.selected_weights)
            ),
            "objective": "highest_NDCG@20_among_eligible",
            "selected_comparison_to_two_route": (
                None
                if selected_result is None
                else _comparison(
                    selected_result["metrics"], TWO_ROUTE_REFERENCE
                )
            ),
        },
        "artifacts": {
            "code_commit_at_run": code_commit,
            "sasrec_checkpoint": {
                "locator": (
                    "MODAL_VOLUME:kuairec-b6a-sasrec-artifacts/"
                    "phase-b6a-recbole-sasrec-v1/epoch_005.pt"
                ),
                "sha256": SASREC_CHECKPOINT_SHA256,
            },
        },
        "claims": {
            "model_training_executed": False,
            "adaptive_validation_development": True,
            "sealed_test_claim": False,
            "significance_claim": False,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "grid_expanded_after_results": False,
        },
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(token in serialized for token in ("/home/", "gho_", "hf_")):
        raise RuntimeError("Generated report contains a host path or credential")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(serialized + "\n")
    report_markdown.write_text(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-cache", type=Path, required=True)
    parser.add_argument("--caption-metadata", type=Path, required=True)
    parser.add_argument("--two-tower-checkpoint", type=Path, required=True)
    parser.add_argument("--bpr-checkpoint", type=Path, required=True)
    parser.add_argument("--sasrec-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sasrec-config",
        type=Path,
        default=Path("configs/phase_b6a_recbole_sasrec.yaml"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/phase_b6b_three_route_hybrid.yaml"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b6b/three_route_hybrid.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b6b/three_route_hybrid.md"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = run(
        repo_root=repo_root,
        data_dir=args.data_dir.resolve(),
        artifact_dir=args.processed_artifact_dir.resolve(),
        caption_cache_path=args.caption_cache.resolve(),
        caption_metadata_path=args.caption_metadata.resolve(),
        two_tower_checkpoint=args.two_tower_checkpoint.resolve(),
        bpr_checkpoint=args.bpr_checkpoint.resolve(),
        sasrec_checkpoint=args.sasrec_checkpoint.resolve(),
        sasrec_config_path=(repo_root / args.sasrec_config).resolve(),
        experiment_config_path=(repo_root / args.experiment_config).resolve(),
        report_json=(repo_root / args.report_json).resolve(),
        report_markdown=(repo_root / args.report_markdown).resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_weights": report["selection"]["selected_weights"],
                "runtime_s": report["runtime_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
