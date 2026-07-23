"""Minimal deterministic rank fusion for Phase B3A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FROZEN_HYBRID_ALPHAS = (0.25, 0.50, 0.75)
RRF_RANK_CONSTANT = 60


def _valid_row(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.int64)
    valid = values[values >= 0]
    if np.any(values[: len(valid)] < 0) or np.any(values[len(valid) :] >= 0):
        raise ValueError("Ranked-list padding must be a trailing -1 suffix")
    if len(np.unique(valid)) != len(valid):
        raise ValueError("Ranked lists may not contain duplicate items")
    return valid


def weighted_reciprocal_rank_fusion(
    two_tower_topk: np.ndarray,
    bpr_topk: np.ndarray,
    *,
    candidates: tuple[np.ndarray, ...],
    alpha: float,
    output_k: int = 100,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> np.ndarray:
    """Fuse two ranked lists using the frozen one-based weighted RRF formula."""

    if alpha not in FROZEN_HYBRID_ALPHAS:
        raise ValueError("alpha is outside the frozen Phase B3A grid")
    if output_k <= 0 or rank_constant <= 0:
        raise ValueError("output_k and rank_constant must be positive")
    two = np.asarray(two_tower_topk, dtype=np.int64)
    bpr = np.asarray(bpr_topk, dtype=np.int64)
    if two.ndim != 2 or bpr.ndim != 2 or two.shape != bpr.shape:
        raise ValueError("Two-Tower and BPR rankings must have equal rank-2 shapes")
    if len(candidates) != two.shape[0]:
        raise ValueError("Candidate rows must align with ranking rows")

    output = np.full((two.shape[0], output_k), -1, dtype=np.int64)
    for row_index, candidate_values in enumerate(candidates):
        candidate_set = set(
            int(item) for item in np.asarray(candidate_values, dtype=np.int64)
        )
        two_row = _valid_row(two[row_index])
        bpr_row = _valid_row(bpr[row_index])
        if not set(int(item) for item in two_row).issubset(candidate_set):
            raise ValueError("Two-Tower ranking contains a non-candidate item")
        if not set(int(item) for item in bpr_row).issubset(candidate_set):
            raise ValueError("BPR ranking contains a non-candidate item")

        scores: dict[int, float] = {}
        for rank, item in enumerate(two_row, start=1):
            key = int(item)
            scores[key] = scores.get(key, 0.0) + alpha / (
                rank_constant + rank
            )
        for rank, item in enumerate(bpr_row, start=1):
            key = int(item)
            scores[key] = scores.get(key, 0.0) + (1.0 - alpha) / (
                rank_constant + rank
            )
        required = min(output_k, len(candidate_set))
        if len(scores) < required:
            raise ValueError("Top-K union is too small for the requested output")
        ranked = sorted(scores, key=lambda item: (-scores[item], item))[:required]
        output[row_index, :required] = ranked
    return output


@dataclass(frozen=True)
class HybridSelection:
    selected_alpha: float | None
    eligible_alphas: tuple[float, ...]
    recall_minimum: float
    coverage_minimum: float


def select_frozen_hybrid(
    *,
    two_tower_metrics: dict[str, float],
    hybrid_metrics: dict[float, dict[str, float]],
) -> HybridSelection:
    """Apply the preregistered Recall/Coverage constraints and NDCG objective."""

    if tuple(hybrid_metrics) != FROZEN_HYBRID_ALPHAS:
        raise ValueError("Hybrid results do not cover the frozen alpha grid")
    recall_minimum = float(two_tower_metrics["Recall@100"]) * 0.98
    coverage_minimum = float(two_tower_metrics["Coverage@100"]) * 0.90
    eligible = tuple(
        alpha
        for alpha in FROZEN_HYBRID_ALPHAS
        if hybrid_metrics[alpha]["Recall@100"] >= recall_minimum
        and hybrid_metrics[alpha]["Coverage@100"] >= coverage_minimum
    )
    selected = (
        None
        if not eligible
        else max(eligible, key=lambda alpha: hybrid_metrics[alpha]["NDCG@20"])
    )
    return HybridSelection(
        selected_alpha=selected,
        eligible_alphas=eligible,
        recall_minimum=recall_minimum,
        coverage_minimum=coverage_minimum,
    )
