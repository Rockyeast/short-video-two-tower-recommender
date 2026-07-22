"""Small, shared evaluator for fixed-catalog retrieval methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import RetrievalQueries


def _ranked_rows(topk: np.ndarray, queries: RetrievalQueries) -> tuple[np.ndarray, ...]:
    values = np.asarray(topk)
    if values.ndim != 2 or values.shape[0] != len(queries.user_ids):
        raise ValueError("topk must have one rank-2 row per query")
    rows: list[np.ndarray] = []
    for index, candidates in enumerate(queries.candidates):
        row = values[index]
        ranked = row[row >= 0].astype(np.int64)
        expected = min(values.shape[1], len(candidates))
        if len(ranked) != expected:
            raise ValueError("Each Top-K row must contain min(K, candidate_count) items")
        if len(np.unique(ranked)) != len(ranked):
            raise ValueError("Top-K rows may not contain duplicate items")
        if not set(int(item) for item in ranked).issubset(
            set(int(item) for item in candidates)
        ):
            raise ValueError("Top-K item is outside the query candidates")
        if np.any(row[: len(ranked)] < 0) or np.any(row[len(ranked) :] >= 0):
            raise ValueError("Top-K padding must be a trailing -1 suffix")
        rows.append(ranked)
    return tuple(rows)


def evaluate_retrieval(
    topk: np.ndarray,
    queries: RetrievalQueries,
    *,
    data_cold_item_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the locked V1 query-macro metrics without bootstrap machinery."""

    ranked_rows = _ranked_rows(topk, queries)
    if not len(ranked_rows):
        raise ValueError("At least one evaluable query is required")
    warm_indices = np.flatnonzero(queries.warm_user_mask)
    cold_indices = np.flatnonzero(~queries.warm_user_mask)
    if not len(warm_indices):
        raise ValueError("Primary evaluation requires at least one warm-user query")
    primary = _evaluate_rows(
        ranked_rows, queries, warm_indices, data_cold_item_ids=data_cold_item_ids
    )
    cold_user = (
        _evaluate_rows(
            ranked_rows,
            queries,
            cold_indices,
            data_cold_item_ids=data_cold_item_ids,
        )
        if len(cold_indices)
        else None
    )
    primary["cold_user_metrics"] = None if cold_user is None else cold_user["metrics"]
    primary["cold_user_denominators"] = (
        {"query_count": 0, "target_count": 0}
        if cold_user is None
        else cold_user["denominators"]
    )
    primary["denominators"]["all_query_count"] = len(ranked_rows)
    primary["denominators"]["all_target_count"] = int(
        sum(len(row) for row in queries.relevant)
    )
    return primary


def _evaluate_rows(
    ranked_rows: tuple[np.ndarray, ...],
    queries: RetrievalQueries,
    indices: np.ndarray,
    *,
    data_cold_item_ids: np.ndarray | None,
) -> dict[str, Any]:
    """Evaluate one predeclared user segment without changing its rankings."""

    query_count = len(indices)
    recall = {k: [] for k in (20, 50, 100)}
    ndcg20: list[float] = []
    recommended: set[int] = set()
    cold = set(
        int(item)
        for item in (
            np.asarray([], dtype=np.int64)
            if data_cold_item_ids is None
            else np.asarray(data_cold_item_ids, dtype=np.int64)
        )
    )
    cold_recall: list[float] = []
    cold_target_count = 0
    discounts = 1.0 / np.log2(np.arange(2, 22, dtype=np.float64))
    for index in indices:
        ranked = ranked_rows[int(index)]
        relevant_values = queries.relevant[int(index)]
        relevant = set(int(item) for item in relevant_values)
        for k in (20, 50, 100):
            hits = sum(int(item) in relevant for item in ranked[:k])
            recall[k].append(hits / len(relevant))
        relevance = np.asarray(
            [int(item) in relevant for item in ranked[:20]], dtype=np.float64
        )
        dcg = float((relevance * discounts[: len(relevance)]).sum())
        ideal = float(discounts[: min(20, len(relevant))].sum())
        ndcg20.append(dcg / ideal)
        recommended.update(int(item) for item in ranked[:100])
        cold_relevant = relevant & cold
        if cold_relevant:
            cold_target_count += len(cold_relevant)
            cold_hits = sum(int(item) in cold_relevant for item in ranked[:100])
            cold_recall.append(cold_hits / len(cold_relevant))
    candidate_union = set(
        int(item) for index in indices for item in queries.candidates[int(index)]
    )
    return {
        "metrics": {
            "Recall@20": float(np.mean(recall[20])),
            "Recall@50": float(np.mean(recall[50])),
            "Recall@100": float(np.mean(recall[100])),
            "NDCG@20": float(np.mean(ndcg20)),
            "Coverage@100": float(len(recommended) / len(candidate_union)),
            "Data-Cold Recall@100": (
                float(np.mean(cold_recall)) if cold_recall else 0.0
            ),
        },
        "denominators": {
            "query_count": query_count,
            "user_count": len(np.unique(queries.user_ids[indices])),
            "target_count": int(
                sum(len(queries.relevant[int(index)]) for index in indices)
            ),
            "candidate_union_count": len(candidate_union),
            "data_cold_query_count": len(cold_recall),
            "data_cold_target_count": cold_target_count,
        },
        "data_cold_is_descriptive": True,
    }
