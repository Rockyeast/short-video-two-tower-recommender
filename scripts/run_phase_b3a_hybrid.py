#!/usr/bin/env python3
"""Evaluate the frozen Phase B3A Two-Tower + BPR weighted-RRF grid."""

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

from kuairec_fully_observed import (
    BPRModel,
    ExactDotProductRetriever,
    evaluate_retrieval,
    load_static_item_features,
)
from kuairec_fully_observed.caption_embeddings import (
    cleaned_text_sha256,
    load_caption_cache,
)
from kuairec_fully_observed.full_training import (
    attach_train_histories,
    build_validation_contract,
    load_canonical_train_events,
    verify_validation_contract,
)
from kuairec_fully_observed.hybrid import (
    FROZEN_HYBRID_ALPHAS,
    RRF_RANK_CONSTANT,
    select_frozen_hybrid,
    weighted_reciprocal_rank_fusion,
)
from kuairec_fully_observed.provenance import (
    PHASE1_PROCESSED_MANIFEST_SHA256,
    normal_membership_record,
    sha256_file,
    verify_phase_b2a_inputs,
)
from kuairec_fully_observed.torch_training import (
    encode_query_users_from_precomputed,
    load_checkpoint,
    preencode_item_universe,
    prepare_item_feature_store,
)
from scripts.run_phase_b2b_full_two_tower import (
    EXPECTED_CAPTION,
    EXPECTED_VALIDATION,
    _processed_popularity,
)

TWO_TOWER_CHECKPOINT_SHA256 = (
    "76c72a3bef0321e719f1db16fa11c48b81d4c50a5cbf21d0fa54e1748b3cf42d"
)
BPR_CHECKPOINT_SHA256 = (
    "2c7bad508a019fc7f9d14e83aceb96170fc4a3b04da6a7412da09c650b4fd737"
)
TWO_TOWER_REFERENCE = {
    "Recall@20": 0.014870122464355342,
    "Recall@50": 0.036037581300587575,
    "Recall@100": 0.0720568501336648,
    "NDCG@20": 0.012112516931264626,
    "Coverage@100": 0.5694607581420181,
    "Data-Cold Recall@100": 0.06515057123608131,
}
BPR_REFERENCE = {
    "Recall@20": 0.013890591784459262,
    "Recall@50": 0.030343695584530817,
    "Recall@100": 0.04843855379304513,
    "NDCG@20": 0.012773561140700995,
    "Coverage@100": 0.33304858515750135,
    "Data-Cold Recall@100": 0.0,
}
TOPK_PER_ROUTE = 500
OUTPUT_K = 100


def _assert_clean_tree(repo_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if status:
        raise RuntimeError("Phase B3A must start from a clean source commit")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _assert_metrics(
    actual: dict[str, float], expected: dict[str, float], *, route: str
) -> None:
    for name, value in expected.items():
        if not np.isclose(
            float(actual[name]), value, rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(
                f"{route} reference metric changed: {name} "
                f"actual={actual[name]} expected={value}"
            )


def _load_bpr(path: Path) -> BPRModel:
    if sha256_file(path) != BPR_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen BPR epoch 20 checkpoint SHA changed")
    with np.load(path) as payload:
        if int(payload["epoch"][0]) != 20:
            raise RuntimeError("BPR checkpoint is not epoch 20")
        return BPRModel(
            user_ids=payload["user_ids"].astype(np.int64, copy=True),
            item_ids=payload["item_ids"].astype(np.int64, copy=True),
            user_factors=payload["user_factors"].astype(np.float32, copy=True),
            item_factors=payload["item_factors"].astype(np.float32, copy=True),
        )


def _relative_change(actual: float, baseline: float) -> float | None:
    return None if baseline == 0.0 else (actual - baseline) / baseline


def _comparison(
    metrics: dict[str, float], baseline: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "hybrid": float(metrics[name]),
            "baseline": float(baseline[name]),
            "absolute_change": float(metrics[name] - baseline[name]),
            "relative_change": _relative_change(
                float(metrics[name]), float(baseline[name])
            ),
        }
        for name in TWO_TOWER_REFERENCE
    }


def _render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, record in report["results"].items():
        metrics = record["metrics"]
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
    selection = report["selection"]
    selected = selection["selected_alpha"]
    if selected is None:
        conclusion = (
            "No frozen Hybrid alpha met both the Recall and Coverage "
            "constraints. The Hybrid gate did not pass; no grid expansion "
            "was performed."
        )
    else:
        conclusion = (
            f"`alpha={selected:.2f}` met the frozen Recall/Coverage "
            "constraints and had the highest NDCG@20 among eligible Hybrid "
            "configurations."
        )
    lines = [
        "# Phase B3A Minimal Two-Tower + BPR Hybrid Validation",
        "",
        "Big validation only. This experiment trains no model and evaluates "
        "only the frozen weighted-RRF alpha grid.",
        "",
        "| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | "
        "Coverage@100 | Data-Cold Recall@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Selection",
        "",
        f"- Recall@100 minimum: `{selection['recall_minimum']:.9f}`",
        f"- Coverage@100 minimum: `{selection['coverage_minimum']:.9f}`",
        f"- Eligible alphas: `{selection['eligible_alphas']}`",
        f"- Selected alpha: `{selected}`",
        f"- NDCG gap reduced: `{str(selection['ndcg_gap_reduced']).lower()}`",
        "",
        conclusion,
        "",
        "## Runtime",
        "",
        f"- Total wall time: `{report['runtime_s']:.3f} s`",
        f"- Peak RSS: `{report['peak_rss_mb']:.2f} MiB`",
        "",
        "No Small Matrix, temporal final, FAISS, LightGBM, training, service, "
        "or monitoring execution occurred.",
        "",
    ]
    text = "\n".join(lines)
    if "/home/" in text:
        raise RuntimeError("Generated Markdown contains a host path")
    return text


def run(
    *,
    repo_root: Path,
    data_dir: Path,
    artifact_dir: Path,
    caption_cache_path: Path,
    caption_metadata_path: Path,
    two_tower_checkpoint: Path,
    bpr_checkpoint: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    code_commit = _assert_clean_tree(repo_root)
    if sha256_file(two_tower_checkpoint) != TWO_TOWER_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen Two-Tower epoch 1 checkpoint SHA changed")
    manifest, raw_sources = verify_phase_b2a_inputs(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        required_raw_files=(
            "big_matrix.csv",
            "item_daily_features.csv",
            "kuairec_caption_category.csv",
        ),
    )
    if sha256_file(artifact_dir / "manifest.json") != (
        PHASE1_PROCESSED_MANIFEST_SHA256
    ):
        raise RuntimeError("Processed manifest identity changed")
    for name in ("events_train_validation.npz", "catalog.npz"):
        if sha256_file(artifact_dir / name) != manifest["files"][name]:
            raise RuntimeError(f"Processed artifact SHA mismatch: {name}")

    static = load_static_item_features(data_dir)
    normal_membership = normal_membership_record(
        np.unique(static.normal_item_ids)
    )
    with np.load(artifact_dir / "events_train_validation.npz") as events, np.load(
        artifact_dir / "catalog.npz"
    ) as catalog:
        event_users = events["user"].astype(np.int64, copy=True)
        event_items = events["item"].astype(np.int64, copy=True)
        event_times = events["timestamp"].astype(np.float64, copy=True)
        event_strong = events["strong"].astype(bool, copy=True)
        user_indptr = events["user_indptr"].astype(np.int64, copy=True)
        actual_user_ids = events["user_ids"].astype(np.int64, copy=True)
        video_ids = catalog["video_ids"].astype(np.int64, copy=True)
        train_end = float(catalog["train_end"][0])
    normal_position = np.isin(video_ids, static.normal_item_ids)
    contract_queries, data_cold_items, validation_counts = (
        build_validation_contract(
            event_users=event_users,
            event_items=event_items,
            event_times=event_times,
            event_strong=event_strong,
            user_indptr=user_indptr,
            actual_user_ids=actual_user_ids,
            video_ids=video_ids,
            normal_item_mask=normal_position,
            train_end=train_end,
            train_events=None,
        )
    )
    verify_validation_contract(
        queries=contract_queries,
        counts=validation_counts,
        expected=EXPECTED_VALIDATION["expected"],
    )
    canonical_train = load_canonical_train_events(
        data_dir, train_end=train_end
    )
    queries = attach_train_histories(
        contract_queries, canonical_train, max_history=50
    )
    popularity = _processed_popularity(
        event_items=event_items,
        event_times=event_times,
        event_strong=event_strong,
        video_ids=video_ids,
        normal_item_mask=normal_position,
        train_end=train_end,
    )

    checkpoint_payload = torch.load(
        two_tower_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        checkpoint_payload.get("checkpoint_kind")
        != "phase-b2b-full-epoch-v1"
        or checkpoint_payload.get("completed_epoch") != 1
    ):
        raise RuntimeError("Two-Tower checkpoint is not complete epoch 1")
    ordered_items = np.asarray(
        checkpoint_payload["ordered_item_ids"], dtype=np.int64
    )
    ordered_users = np.asarray(
        checkpoint_payload["ordered_user_ids"], dtype=np.int64
    )
    touched_items = np.asarray(
        checkpoint_payload["touched_item_ids"], dtype=np.int64
    )
    touched_users = np.asarray(
        checkpoint_payload["touched_user_ids"], dtype=np.int64
    )
    train_history_items = np.unique(
        video_ids[event_items[event_times < train_end]]
    )
    expected_universe = np.union1d(
        train_history_items, queries.catalog
    ).astype(np.int64)
    if not np.array_equal(ordered_items, expected_universe):
        raise RuntimeError("Checkpoint item universe differs from validation")
    static_for_universe = static.frame.set_index("video_id").reindex(
        ordered_items
    )
    caption = load_caption_cache(
        cache_path=caption_cache_path,
        metadata_path=caption_metadata_path,
        expected_item_ids=ordered_items,
        expected_model_id=EXPECTED_CAPTION["model_id"],
        expected_revision=EXPECTED_CAPTION["resolved_revision"],
        expected_source_sha256=raw_sources[
            "kuairec_caption_category.csv"
        ]["expected_sha256"],
        expected_cleaned_text_sha256=cleaned_text_sha256(
            ordered_items,
            static_for_universe["caption_text"].astype(str).tolist(),
        ),
    )
    train_observed_normal = np.intersect1d(
        train_history_items, static.normal_item_ids, assume_unique=True
    )
    store = prepare_item_feature_store(
        static_frame=static.frame,
        caption_cache=caption,
        item_universe=ordered_items,
        train_observed_item_ids=train_history_items,
        train_observed_normal_item_ids=train_observed_normal,
    )
    if not np.array_equal(store.item_ids, ordered_items):
        raise RuntimeError("Prepared item store differs from checkpoint mapping")
    two_tower, loaded_payload = load_checkpoint(
        two_tower_checkpoint,
        device="cpu",
        expected_identity=checkpoint_payload["identity"],
    )
    if not np.array_equal(
        loaded_payload["ordered_user_ids"], ordered_users
    ):
        raise RuntimeError("Two-Tower ordered user mapping changed")
    two_tower.eval()
    item_vectors = preencode_item_universe(
        model=two_tower,
        store=store,
        touched_item_ids=set(int(value) for value in touched_items),
        device="cpu",
        batch_size=1024,
    )
    catalog_positions = np.asarray(
        [store.positions[int(item)] for item in queries.catalog],
        dtype=np.int64,
    )
    catalog_vectors = item_vectors[catalog_positions].numpy()
    user_positions = {
        int(user): position + 1
        for position, user in enumerate(ordered_users)
    }
    user_vectors = encode_query_users_from_precomputed(
        model=two_tower,
        store=store,
        precomputed_item_vectors=item_vectors,
        user_ids=queries.user_ids,
        histories=queries.histories,
        history_weights=queries.history_weights,
        user_positions=user_positions,
        touched_user_ids=set(int(value) for value in touched_users),
        device="cpu",
        batch_size=128,
    ).numpy()
    fallback = popularity.rank(queries, k=TOPK_PER_ROUTE)
    two_tower_top500 = ExactDotProductRetriever().search(
        user_vectors,
        catalog_vectors,
        item_ids=queries.catalog,
        candidates=queries.candidates,
        k=TOPK_PER_ROUTE,
        warm_user_mask=queries.warm_user_mask,
        fallback_topk=fallback,
        score_block_size=128,
    )
    bpr = _load_bpr(bpr_checkpoint)
    bpr_top500 = bpr.rank(
        queries,
        k=TOPK_PER_ROUTE,
        cold_user_fallback=popularity,
        score_block_size=128,
    )

    two_reference = evaluate_retrieval(
        two_tower_top500,
        queries,
        data_cold_item_ids=data_cold_items,
    )
    bpr_reference = evaluate_retrieval(
        bpr_top500,
        queries,
        data_cold_item_ids=data_cold_items,
    )
    _assert_metrics(
        two_reference["metrics"], TWO_TOWER_REFERENCE, route="Two-Tower"
    )
    _assert_metrics(bpr_reference["metrics"], BPR_REFERENCE, route="BPR")
    hybrid_results: dict[float, dict[str, Any]] = {}
    for alpha in FROZEN_HYBRID_ALPHAS:
        fused = weighted_reciprocal_rank_fusion(
            two_tower_top500,
            bpr_top500,
            candidates=queries.candidates,
            alpha=alpha,
            output_k=OUTPUT_K,
            rank_constant=RRF_RANK_CONSTANT,
        )
        hybrid_results[alpha] = evaluate_retrieval(
            fused, queries, data_cold_item_ids=data_cold_items
        )
    selection = select_frozen_hybrid(
        two_tower_metrics=two_reference["metrics"],
        hybrid_metrics={
            alpha: hybrid_results[alpha]["metrics"]
            for alpha in FROZEN_HYBRID_ALPHAS
        },
    )
    selected_metrics = (
        None
        if selection.selected_alpha is None
        else hybrid_results[selection.selected_alpha]["metrics"]
    )
    tt_shortfall = max(
        0.0,
        BPR_REFERENCE["NDCG@20"] - TWO_TOWER_REFERENCE["NDCG@20"],
    )
    selected_shortfall = (
        None
        if selected_metrics is None
        else max(0.0, BPR_REFERENCE["NDCG@20"] - selected_metrics["NDCG@20"])
    )
    report: dict[str, Any] = {
        "phase": "phase-b3a-minimal-hybrid-validation",
        "status": (
            "hybrid_selected"
            if selection.selected_alpha is not None
            else "hybrid_not_selected"
        ),
        "claim_boundary": {
            "model_training_executed": False,
            "big_validation_accessed": True,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
            "faiss_run": False,
            "lightgbm_run": False,
            "service_run": False,
            "monitoring_run": False,
            "alpha_grid_expanded": False,
            "seed_count": 1,
        },
        "configuration": {
            "route_top_k": TOPK_PER_ROUTE,
            "output_k": OUTPUT_K,
            "rank_constant": RRF_RANK_CONSTANT,
            "alphas": list(FROZEN_HYBRID_ALPHAS),
            "formula": (
                "alpha/(60+rank_two_tower) + "
                "(1-alpha)/(60+rank_bpr)"
            ),
            "rank_indexing": "one_based",
            "missing_route_contribution": 0.0,
            "tie_break": "ascending_item_id",
        },
        "counts": validation_counts,
        "results": {
            "BPR alpha=0": bpr_reference,
            "Two-Tower alpha=1": two_reference,
            **{
                f"Hybrid alpha={alpha:.2f}": hybrid_results[alpha]
                for alpha in FROZEN_HYBRID_ALPHAS
            },
        },
        "selection": {
            "recall_minimum": selection.recall_minimum,
            "coverage_minimum": selection.coverage_minimum,
            "eligible_alphas": list(selection.eligible_alphas),
            "selected_alpha": selection.selected_alpha,
            "objective": "highest_NDCG@20_among_eligible",
            "two_tower_ndcg_shortfall_to_bpr": tt_shortfall,
            "selected_ndcg_shortfall_to_bpr": selected_shortfall,
            "ndcg_gap_reduced": (
                False
                if selected_shortfall is None
                else selected_shortfall < tt_shortfall
            ),
        },
        "selected_comparisons": (
            None
            if selected_metrics is None
            else {
                "versus_two_tower": _comparison(
                    selected_metrics, TWO_TOWER_REFERENCE
                ),
                "versus_bpr": _comparison(selected_metrics, BPR_REFERENCE),
            }
        ),
        "artifacts": {
            "code_commit_at_run": code_commit,
            "input_tree_clean_at_start": True,
            "processed_manifest_sha256": (
                PHASE1_PROCESSED_MANIFEST_SHA256
            ),
            "raw_inputs": raw_sources,
            "normal_membership": normal_membership,
            "two_tower_checkpoint": {
                "locator": (
                    "MODAL_VOLUME:kuairec-b2b-full-run-artifacts/"
                    "phase-b2b-full-v1/checkpoints/epoch_001.pt"
                ),
                "sha256": TWO_TOWER_CHECKPOINT_SHA256,
            },
            "bpr_checkpoint": {
                "locator": "B1A_ARTIFACT_DIR/epoch_020.npz",
                "sha256": BPR_CHECKPOINT_SHA256,
            },
        },
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if any(token in serialized for token in ("/home/", "gho_", "hf_", "MODAL_TOKEN")):
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
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/phase_b3a/hybrid_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/phase_b3a/hybrid_validation.md"),
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
        report_json=(repo_root / args.report_json).resolve(),
        report_markdown=(repo_root / args.report_markdown).resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_alpha": report["selection"]["selected_alpha"],
                "runtime_s": report["runtime_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
