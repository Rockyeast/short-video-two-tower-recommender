#!/usr/bin/env python3
"""Verify real-artifact offline, pipeline, and HTTP reranker parity."""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from kuairec_fully_observed.api import create_app
from kuairec_fully_observed.hybrid import (
    weighted_reciprocal_rank_fusion,
)
from kuairec_fully_observed.reranker_serving import (
    LocalLightGBMReranker,
)
from kuairec_fully_observed.reranking import (
    build_rerank_feature_matrix,
)
from kuairec_fully_observed.serving import load_recommendation_engine


def _assert_clean(repo_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if status:
        raise RuntimeError("Parity verification requires a clean source tree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _direct_features(
    reranker: LocalLightGBMReranker,
    *,
    history: np.ndarray,
    candidates: np.ndarray,
    two_ranked: np.ndarray,
    bpr_ranked: np.ndarray,
    two_scores: np.ndarray,
    bpr_scores: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(
        [reranker._positions[int(item)] for item in candidates],
        dtype=np.int64,
    )
    history_categories = {
        int(category)
        for item in history
        if int(item) in reranker._positions
        for category in reranker.category_ids[
            reranker._positions[int(item)]
        ]
        if category >= 0
    }
    return build_rerank_feature_matrix(
        item_ids=candidates,
        two_tower_scores=two_scores,
        bpr_scores=bpr_scores,
        two_tower_ranked=two_ranked,
        bpr_ranked=bpr_ranked,
        popularity_scores=reranker.train_popularity_scores[positions],
        item_categories=reranker.category_ids[positions],
        history_categories=history_categories,
        data_cold_mask=reranker.train_data_cold_mask[positions],
        video_duration=reranker.video_duration[positions],
        history_length=len(history),
        alpha=reranker.alpha,
        rank_constant=reranker.rank_constant,
    )


async def _http_checks(
    *,
    config_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    reranker_model_path: Path,
    reranker_feature_path: Path,
    reranker_metadata_path: Path,
    requests: list[dict[str, Any]],
    expected: list[list[int]],
) -> dict[str, Any]:
    app = create_app(
        config_path=config_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        reranker_model_path=reranker_model_path,
        reranker_feature_path=reranker_feature_path,
        reranker_metadata_path=reranker_metadata_path,
    )
    enabled_latency: list[float] = []
    disabled_latency: list[float] = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            health = (await client.get("/healthz")).json()
            for index, payload in enumerate(requests):
                started = time.perf_counter()
                enabled = await client.post(
                    "/v1/recommend",
                    json={**payload, "use_reranker": True},
                )
                enabled_latency.append(
                    (time.perf_counter() - started) * 1000.0
                )
                if enabled.status_code != 200:
                    raise RuntimeError(
                        f"Enabled HTTP request failed: {enabled.text}"
                    )
                if enabled.json()["item_ids"] != expected[index]:
                    raise RuntimeError("HTTP and pipeline rankings differ")
                started = time.perf_counter()
                disabled = await client.post(
                    "/v1/recommend",
                    json={**payload, "use_reranker": False},
                )
                disabled_latency.append(
                    (time.perf_counter() - started) * 1000.0
                )
                if disabled.status_code != 200:
                    raise RuntimeError(
                        f"Disabled HTTP request failed: {disabled.text}"
                    )
    return {
        "health": health,
        "enabled_latency_ms": {
            "p50": _percentile(enabled_latency, 50),
            "p95": _percentile(enabled_latency, 95),
            "mean": float(np.mean(enabled_latency)),
        },
        "disabled_latency_ms": {
            "p50": _percentile(disabled_latency, 50),
            "p95": _percentile(disabled_latency, 95),
            "mean": float(np.mean(disabled_latency)),
        },
    }


def run(
    *,
    repo_root: Path,
    config_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    reranker_model_path: Path,
    reranker_feature_path: Path,
    reranker_metadata_path: Path,
    report_json: Path,
    report_markdown: Path,
    query_count: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    code_commit = _assert_clean(repo_root)
    engine, metadata = load_recommendation_engine(
        config_path=config_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        reranker_model_path=reranker_model_path,
        reranker_feature_path=reranker_feature_path,
        reranker_metadata_path=reranker_metadata_path,
    )
    if not isinstance(engine.reranker, LocalLightGBMReranker):
        raise RuntimeError("Expected the local LightGBM reranker")
    warm_users = np.intersect1d(
        engine.two_tower.user_ids, engine.bpr.user_ids
    )[:query_count]
    if len(warm_users) != query_count:
        raise RuntimeError("Not enough warm users for parity verification")
    feature_max_abs = 0.0
    requests: list[dict[str, Any]] = []
    expected: list[list[int]] = []
    top100_set_matches = 0
    for offset, user in enumerate(warm_users):
        history = np.roll(engine.catalog, offset)[:5].copy()
        candidates = engine.catalog[~np.isin(engine.catalog, history)]
        weights = np.ones(len(history), dtype=np.float32)
        route_k = min(engine.config.route_top_k, len(candidates))
        route_args = {
            "user_id": int(user),
            "history": history,
            "history_weights": weights,
            "candidates": candidates,
            "k": route_k,
        }
        two_ranked = engine.two_tower.retrieve(**route_args)
        bpr_ranked = engine.bpr.retrieve(**route_args)
        hybrid = weighted_reciprocal_rank_fusion(
            two_ranked[None, :],
            bpr_ranked[None, :],
            candidates=(candidates,),
            alpha=engine.config.alpha,
            output_k=engine.config.output_k,
            rank_constant=engine.config.rank_constant,
        )[0]
        hybrid = hybrid[hybrid >= 0]
        two_scores = engine.two_tower.score_candidates(
            user_id=int(user),
            history=history,
            history_weights=weights,
            candidates=hybrid,
        )
        bpr_scores = engine.bpr.score_candidates(
            user_id=int(user),
            history=history,
            history_weights=weights,
            candidates=hybrid,
        )
        online_features = engine.reranker.build_features(
            history=history,
            candidates=hybrid,
            two_tower_ranked=two_ranked,
            bpr_ranked=bpr_ranked,
            two_tower_scores=two_scores,
            bpr_scores=bpr_scores,
        )
        offline_features = _direct_features(
            engine.reranker,
            history=history,
            candidates=hybrid,
            two_ranked=two_ranked,
            bpr_ranked=bpr_ranked,
            two_scores=two_scores,
            bpr_scores=bpr_scores,
        )
        feature_max_abs = max(
            feature_max_abs,
            float(np.max(np.abs(online_features - offline_features))),
        )
        manual = engine.reranker.rerank(
            user_id=int(user),
            history=history,
            history_weights=weights,
            candidates=hybrid,
            two_tower_ranked=two_ranked,
            bpr_ranked=bpr_ranked,
            two_tower_scores=two_scores,
            bpr_scores=bpr_scores,
        )
        pipeline = engine.recommend(
            int(user),
            history,
            top_k=engine.config.output_k,
            history_weights=weights,
            use_reranker=True,
        )
        if list(pipeline.item_ids) != manual.tolist():
            raise RuntimeError("Manual and pipeline rankings differ")
        if set(manual) != set(hybrid):
            raise RuntimeError("Reranker changed a Top-100 candidate set")
        top100_set_matches += 1
        requests.append(
            {
                "user_id": int(user),
                "history": history.tolist(),
                "history_weights": weights.tolist(),
                "top_k": engine.config.output_k,
            }
        )
        expected.append(manual.tolist())
    if feature_max_abs != 0.0:
        raise RuntimeError("Offline and online rerank features differ")
    http = asyncio.run(
        _http_checks(
            config_path=config_path,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            reranker_model_path=reranker_model_path,
            reranker_feature_path=reranker_feature_path,
            reranker_metadata_path=reranker_metadata_path,
            requests=requests,
            expected=expected,
        )
    )
    report = {
        "phase": "reranker-pipeline-parity-v1",
        "status": "passed",
        "query_count": query_count,
        "offline_online_feature_max_abs": feature_max_abs,
        "manual_pipeline_rankings_exact": True,
        "pipeline_http_rankings_exact": True,
        "top100_candidate_sets_preserved": top100_set_matches,
        "reranker_default_enabled": False,
        "http": http,
        "identity": {
            "code_commit": code_commit,
            "serving_bundle_sha256": metadata["bundle_sha256"],
            "reranker_model_sha256": metadata[
                "reranker_model_sha256"
            ],
        },
        "claim_boundary": {
            "real_artifacts_used": True,
            "request_histories_are_deterministic_synthetic_inputs": True,
            "recommendation_effectiveness_claim": False,
            "small_matrix_accessed": False,
            "temporal_final_accessed": False,
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
        raise RuntimeError("Parity report contains a host path")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(serialized + "\n")
    report_markdown.write_text(
        "\n".join(
            [
                "# Optional Reranker Pipeline Parity",
                "",
                f"- Queries: `{query_count}`",
                f"- Offline/online feature max abs: `{feature_max_abs}`",
                "- Manual/Pipeline rankings exact: `true`",
                "- Pipeline/HTTP rankings exact: `true`",
                f"- Preserved Top-100 sets: `{top100_set_matches}`",
                f"- HTTP reranker enabled P50/P95: "
                f"`{http['enabled_latency_ms']['p50']:.3f}/"
                f"{http['enabled_latency_ms']['p95']:.3f} ms`",
                f"- HTTP reranker disabled P50/P95: "
                f"`{http['disabled_latency_ms']['p50']:.3f}/"
                f"{http['disabled_latency_ms']['p95']:.3f} ms`",
                "",
                "Real serving artifacts were used with deterministic synthetic "
                "request histories. This validates feature and orchestration "
                "parity, not recommendation effectiveness on final-refit routes.",
                "",
            ]
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--reranker-features", type=Path, required=True)
    parser.add_argument("--reranker-metadata", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/pipeline/reranker_parity.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/pipeline/reranker_parity.md"),
    )
    args = parser.parse_args()
    report = run(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        bundle_path=args.bundle.resolve(),
        metadata_path=args.metadata.resolve(),
        reranker_model_path=args.reranker_model.resolve(),
        reranker_feature_path=args.reranker_features.resolve(),
        reranker_metadata_path=args.reranker_metadata.resolve(),
        report_json=args.report_json.resolve(),
        report_markdown=args.report_markdown.resolve(),
        query_count=args.query_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
