"""Bounded three-route rank fusion for Big-validation development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FROZEN_THREE_ROUTE_WEIGHTS = (
    (0.60, 0.25, 0.15),
    (0.50, 0.35, 0.15),
    (0.40, 0.45, 0.15),
)
RRF_RANK_CONSTANT = 60


def _valid_row(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.int64)
    valid_count = int(np.count_nonzero(values >= 0))
    if np.any(values[:valid_count] < 0) or np.any(values[valid_count:] >= 0):
        raise ValueError("Ranked-list padding must be a trailing -1 suffix")
    valid = values[:valid_count]
    if len(np.unique(valid)) != len(valid):
        raise ValueError("Ranked lists may not contain duplicate items")
    return valid


def weighted_three_route_rrf(
    two_tower_topk: np.ndarray,
    sasrec_topk: np.ndarray,
    bpr_topk: np.ndarray,
    *,
    candidates: tuple[np.ndarray, ...],
    weights: tuple[float, float, float],
    output_k: int = 100,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> np.ndarray:
    """Fuse Two-Tower, SASRec and BPR lists with one-based weighted RRF."""

    if weights not in FROZEN_THREE_ROUTE_WEIGHTS:
        raise ValueError("weights are outside the frozen three-route grid")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("route weights must sum to one")
    if output_k <= 0 or rank_constant <= 0:
        raise ValueError("output_k and rank_constant must be positive")
    routes = tuple(
        np.asarray(values, dtype=np.int64)
        for values in (two_tower_topk, sasrec_topk, bpr_topk)
    )
    if any(values.ndim != 2 for values in routes):
        raise ValueError("all route rankings must have rank-2 shape")
    if len({values.shape for values in routes}) != 1:
        raise ValueError("all route rankings must have equal shape")
    if len(candidates) != routes[0].shape[0]:
        raise ValueError("candidate rows must align with ranking rows")

    output = np.full((routes[0].shape[0], output_k), -1, dtype=np.int64)
    for row_index, candidate_values in enumerate(candidates):
        candidate_set = set(
            int(item) for item in np.asarray(candidate_values, dtype=np.int64)
        )
        scores: dict[int, float] = {}
        for route, weight in zip(routes, weights, strict=True):
            ranked = _valid_row(route[row_index])
            if not set(int(item) for item in ranked).issubset(candidate_set):
                raise ValueError("route ranking contains a non-candidate item")
            for rank, item in enumerate(ranked, start=1):
                key = int(item)
                scores[key] = scores.get(key, 0.0) + weight / (
                    rank_constant + rank
                )
        required = min(output_k, len(candidate_set))
        if len(scores) < required:
            raise ValueError("Top-K union is too small for requested output")
        ranked_items = sorted(
            scores, key=lambda item: (-scores[item], item)
        )[:required]
        output[row_index, :required] = ranked_items
    return output


@dataclass(frozen=True)
class ThreeRouteSelection:
    selected_weights: tuple[float, float, float] | None
    eligible_weights: tuple[tuple[float, float, float], ...]
    recall_minimum: float
    coverage_minimum: float
    data_cold_minimum: float


def select_three_route_hybrid(
    *,
    reference_metrics: dict[str, float],
    candidate_metrics: dict[
        tuple[float, float, float], dict[str, float]
    ],
) -> ThreeRouteSelection:
    """Preserve retrieval quality, then select the highest NDCG candidate."""

    if tuple(candidate_metrics) != FROZEN_THREE_ROUTE_WEIGHTS:
        raise ValueError("results do not cover the frozen three-route grid")
    recall_minimum = float(reference_metrics["Recall@100"]) * 0.98
    coverage_minimum = float(reference_metrics["Coverage@100"]) * 0.90
    data_cold_minimum = (
        float(reference_metrics["Data-Cold Recall@100"]) * 0.90
    )
    eligible = tuple(
        weights
        for weights in FROZEN_THREE_ROUTE_WEIGHTS
        if candidate_metrics[weights]["Recall@100"] >= recall_minimum
        and candidate_metrics[weights]["Coverage@100"] >= coverage_minimum
        and candidate_metrics[weights]["Data-Cold Recall@100"]
        >= data_cold_minimum
    )
    selected = (
        None
        if not eligible
        else max(
            eligible,
            key=lambda weights: candidate_metrics[weights]["NDCG@20"],
        )
    )
    return ThreeRouteSelection(
        selected_weights=selected,
        eligible_weights=eligible,
        recall_minimum=recall_minimum,
        coverage_minimum=coverage_minimum,
        data_cold_minimum=data_cold_minimum,
    )
