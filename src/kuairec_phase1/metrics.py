"""Shared exact-ranking metrics for every Phase 1 baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .gates import CI_METRICS, METRICS


KS = (10, 20, 50, 100)


def _per_user_components(
    values: np.ndarray, users: np.ndarray, eligible: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_users, inverse = np.unique(users, return_inverse=True)
    if eligible is None:
        eligible = np.ones(len(values), dtype=bool)
    sums = np.bincount(
        inverse[eligible], weights=values[eligible], minlength=len(unique_users)
    )
    counts = np.bincount(inverse[eligible], minlength=len(unique_users))
    means = np.full(len(unique_users), np.nan, dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present]
    return unique_users, sums, counts, means


def _bootstrap_ci(
    per_user_sums: np.ndarray,
    per_user_counts: np.ndarray,
    sample_indices: np.ndarray,
) -> list[float]:
    valid = per_user_counts > 0
    if not valid.any():
        return [0.0, 0.0]
    # Resample whole user clusters, then recompute the contract's primary
    # query-macro estimator over every query carried by those clusters.  Taking
    # an unweighted mean of per-user means would instead bootstrap the separate
    # secondary user-macro metric and can produce an interval far from the
    # reported primary point estimate when history lengths are imbalanced.
    sampled_sums = per_user_sums[sample_indices].sum(axis=1)
    sampled_counts = per_user_counts[sample_indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sampled_sums / sampled_counts
    means = means[np.isfinite(means)]
    if means.size == 0:
        return [0.0, 0.0]
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def common_bootstrap_indices(
    users: np.ndarray, *, replicates: int = 2000, seed: int = 20260721
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(users)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(replicates, len(unique)), dtype=np.int32)
    return unique, indices


def evaluate_topk(
    *,
    topk: np.ndarray,
    query_users: np.ndarray,
    target_indptr: np.ndarray,
    target_indices: np.ndarray,
    candidate_union_count: int,
    candidate_score_count: int,
    warm_mask: np.ndarray,
    tail_mask: np.ndarray,
    cold_mask: np.ndarray,
    bootstrap_users: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one exact Top-100 matrix under the frozen metric contract."""

    query_count = len(query_users)
    if topk.shape != (query_count, 100):
        raise ValueError("topk must have shape (query_count, 100)")
    recall_values = {k: np.zeros(query_count, dtype=np.float64) for k in (20, 50, 100)}
    ndcg_values = {k: np.zeros(query_count, dtype=np.float64) for k in (10, 20)}
    segment_values = {
        name: {k: np.zeros(query_count, dtype=np.float64) for k in (20, 50, 100)}
        for name in ("Warm", "Tail", "Cold")
    }
    segment_eligible = {
        name: np.zeros(query_count, dtype=bool) for name in ("Warm", "Tail", "Cold")
    }
    segment_masks = {"Warm": warm_mask, "Tail": tail_mask, "Cold": cold_mask}
    target_counts = np.diff(target_indptr)
    discounts = 1.0 / np.log2(np.arange(2, 102, dtype=np.float64))

    for query in range(query_count):
        targets = target_indices[target_indptr[query] : target_indptr[query + 1]]
        target_set = set(int(value) for value in targets)
        ranked = topk[query]
        relevance = np.fromiter(
            (int(item) in target_set if item >= 0 else False for item in ranked),
            dtype=bool,
            count=100,
        )
        for k in (20, 50, 100):
            recall_values[k][query] = float(relevance[:k].sum()) / len(targets)
        for k in (10, 20):
            dcg = float((relevance[:k] * discounts[:k]).sum())
            ideal = float(discounts[: min(k, len(targets))].sum())
            ndcg_values[k][query] = dcg / ideal
        for name, mask in segment_masks.items():
            segment_targets = targets[mask[targets]]
            if not len(segment_targets):
                continue
            segment_eligible[name][query] = True
            segment_set = set(int(value) for value in segment_targets)
            for k in (20, 50, 100):
                hits = sum(int(item) in segment_set for item in ranked[:k] if item >= 0)
                segment_values[name][k][query] = hits / len(segment_targets)

    metrics: dict[str, float] = {}
    per_user_metric: dict[str, np.ndarray] = {}
    bootstrap_components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, values in recall_values.items():
        name = f"Recall@{k}"
        metrics[name] = float(values.mean())
        _, sums, counts, means = _per_user_components(values, query_users)
        per_user_metric[name] = means
        bootstrap_components[name] = (sums, counts)
    for k, values in ndcg_values.items():
        name = f"NDCG@{k}"
        metrics[name] = float(values.mean())
        _, sums, counts, means = _per_user_components(values, query_users)
        per_user_metric[name] = means
        bootstrap_components[name] = (sums, counts)
    recommended_by_k = {
        k: np.unique(topk[:, :k][topk[:, :k] >= 0]) for k in (20, 50, 100)
    }
    for k, values in recommended_by_k.items():
        metrics[f"Coverage@{k}"] = float(len(values) / candidate_union_count)
    for name in ("Warm", "Tail", "Cold"):
        eligible = segment_eligible[name]
        for k, values in segment_values[name].items():
            metric_name = f"{name}Recall@{k}"
            metrics[metric_name] = float(values[eligible].mean()) if eligible.any() else 0.0
            _, sums, counts, means = _per_user_components(
                values, query_users, eligible
            )
            per_user_metric[metric_name] = means
            bootstrap_components[metric_name] = (sums, counts)
    if set(metrics) != set(METRICS):
        raise RuntimeError("Metric implementation does not match the execution gate")

    unique_query_users = np.unique(query_users)
    if not np.array_equal(unique_query_users, bootstrap_users):
        raise RuntimeError("Bootstrap users differ from evaluated users")
    intervals = {
        name: _bootstrap_ci(*bootstrap_components[name], bootstrap_indices)
        for name in CI_METRICS
    }
    denominators: dict[str, int] = {
        "query_count": int(query_count),
        "user_count": int(len(unique_query_users)),
        "target_count": int(target_counts.sum()),
        "candidate_union_count": int(candidate_union_count),
        "candidate_score_count": int(candidate_score_count),
    }
    for name, mask in segment_masks.items():
        relevant = mask[target_indices]
        per_query = np.add.reduceat(relevant.astype(np.int64), target_indptr[:-1])
        eligible = per_query > 0
        prefix = name.lower()
        denominators[f"{prefix}_query_count"] = int(eligible.sum())
        denominators[f"{prefix}_user_count"] = int(np.unique(query_users[eligible]).size)
        denominators[f"{prefix}_target_count"] = int(relevant.sum())

    secondary_user_macro = {
        name: float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        for name, values in per_user_metric.items()
    }
    return {
        "metrics": metrics,
        "bootstrap_95_percent_intervals": intervals,
        "denominators": denominators,
        "secondary_user_macro": secondary_user_macro,
        "coverage": {
            f"Coverage@{k}": {
                "numerator": int(len(values)),
                "denominator": int(candidate_union_count),
            }
            for k, values in recommended_by_k.items()
        },
    }
