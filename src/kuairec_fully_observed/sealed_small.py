"""Frozen Small-Matrix evaluator assembled before the sealed file is opened."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import RetrievalQueries
from .evaluation import evaluate_retrieval
from .hybrid import RRF_RANK_CONSTANT, weighted_reciprocal_rank_fusion


FROZEN_SMALL_ALPHA = 0.75
FROZEN_ROUTE_TOP_K = 500
FROZEN_OUTPUT_K = 100
FROZEN_SMALL_METHODS = (
    "random",
    "global_popularity",
    "bpr",
    "two_tower",
    "hybrid_alpha_0.75",
)


def require_sealed_execution(execute_sealed_small: bool) -> None:
    """Fail before any data access unless the one-time run is explicit."""

    if not execute_sealed_small:
        raise RuntimeError(
            "The one-time Small evaluation requires --execute-sealed-small"
        )


def _validate_route(
    values: np.ndarray,
    queries: RetrievalQueries,
    *,
    name: str,
) -> np.ndarray:
    ranked = np.asarray(values, dtype=np.int64)
    if ranked.ndim != 2 or ranked.shape[0] != len(queries.user_ids):
        raise ValueError(f"{name} must have one rank-2 row per Small query")
    for row_index, candidates in enumerate(queries.candidates):
        row = ranked[row_index]
        valid = row[row >= 0]
        required = min(ranked.shape[1], len(candidates))
        if len(valid) != required:
            raise ValueError(f"{name} row does not contain the required Top-K")
        if np.any(row[:required] < 0) or np.any(row[required:] >= 0):
            raise ValueError(f"{name} padding is not a trailing -1 suffix")
        if len(np.unique(valid)) != len(valid):
            raise ValueError(f"{name} row contains duplicate items")
        if not set(int(item) for item in valid).issubset(
            set(int(item) for item in candidates)
        ):
            raise ValueError(f"{name} row contains an unavailable Small pair")
    return ranked


def _force_shared_cold_fallback(
    rankings: np.ndarray,
    popularity: np.ndarray,
    warm_user_mask: np.ndarray,
) -> np.ndarray:
    result = rankings.copy()
    cold = ~np.asarray(warm_user_mask, dtype=bool)
    replacement = np.full(
        (int(cold.sum()), result.shape[1]), -1, dtype=np.int64
    )
    copy_width = min(result.shape[1], popularity.shape[1])
    replacement[:, :copy_width] = popularity[cold, :copy_width]
    result[cold] = replacement
    return result


def evaluate_frozen_small_routes(
    *,
    queries: RetrievalQueries,
    random_topk: np.ndarray,
    popularity_topk: np.ndarray,
    bpr_topk: np.ndarray,
    two_tower_topk: np.ndarray,
    data_cold_item_ids: np.ndarray,
) -> dict[str, Any]:
    """Apply the frozen route/fallback policy and compute the sealed report table.

    Route rankings are produced by already-refit models. This function never
    fits or updates a model. In particular, Small labels are only consumed by
    ``evaluate_retrieval`` after all rankings have been fixed.
    """

    popularity = _validate_route(
        popularity_topk, queries, name="global_popularity"
    )
    routes = {
        "random": _validate_route(random_topk, queries, name="random"),
        "global_popularity": popularity,
        "bpr": _validate_route(bpr_topk, queries, name="bpr"),
        "two_tower": _validate_route(
            two_tower_topk, queries, name="two_tower"
        ),
    }
    for name in ("random", "bpr", "two_tower"):
        routes[name] = _force_shared_cold_fallback(
            routes[name], popularity, queries.warm_user_mask
        )

    hybrid = weighted_reciprocal_rank_fusion(
        routes["two_tower"],
        routes["bpr"],
        candidates=queries.candidates,
        alpha=FROZEN_SMALL_ALPHA,
        output_k=FROZEN_OUTPUT_K,
        rank_constant=RRF_RANK_CONSTANT,
    )
    popularity_output = popularity[:, :FROZEN_OUTPUT_K]
    routes["hybrid_alpha_0.75"] = _force_shared_cold_fallback(
        hybrid, popularity_output, queries.warm_user_mask
    )

    results = {}
    for name in FROZEN_SMALL_METHODS:
        topk = routes[name][:, :FROZEN_OUTPUT_K]
        results[name] = {
            "metrics": evaluate_retrieval(
                topk,
                queries,
                data_cold_item_ids=np.asarray(
                    data_cold_item_ids, dtype=np.int64
                ),
            ),
            "topk": topk,
        }
    return {
        "recipe": {
            "alpha": FROZEN_SMALL_ALPHA,
            "route_top_k": FROZEN_ROUTE_TOP_K,
            "rank_constant": RRF_RANK_CONSTANT,
            "output_k": FROZEN_OUTPUT_K,
            "cold_user_fallback": "refit_global_popularity",
        },
        "results": results,
    }
