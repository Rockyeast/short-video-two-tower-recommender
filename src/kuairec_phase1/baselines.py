"""Exact full-candidate implementations of the five frozen baselines."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def load_artifacts(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    return {
        "root": root,
        "catalog": np.load(root / "catalog.npz"),
        "events": np.load(root / "events_train_validation.npz"),
        "train": np.load(root / "targets_train.npz"),
        "queries": np.load(root / "queries_validation.npz"),
        "candidate_bits": np.load(root / "candidate_bits_validation.npy", mmap_mode="r"),
        "user_item": sparse.load_npz(root / "itemcf_user_item.npz").tocsr(),
        "cooccurrence": sparse.load_npz(root / "itemcf_cooccurrence.npz").tocsr(),
        "bpr_negatives": np.load(root / "bpr_negative_indices.npz"),
    }


def candidate_positions(bits: np.ndarray, item_count: int) -> np.ndarray:
    return np.flatnonzero(np.unpackbits(bits, bitorder="little")[:item_count])


def _contains(bits: np.ndarray, position: int) -> bool:
    return bool(int(bits[position >> 3]) & (1 << (position & 7)))


def topk_from_order(
    order: np.ndarray | list[int], bits: np.ndarray, *, k: int = 100
) -> np.ndarray:
    result = np.full(k, -1, dtype=np.int32)
    cursor = 0
    for value in order:
        item = int(value)
        if _contains(bits, item):
            result[cursor] = item
            cursor += 1
            if cursor == k:
                break
    return result


def popularity_order(scores: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((video_ids, -scores))


def rank_global_popularity(artifacts: dict[str, Any]) -> np.ndarray:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    order = popularity_order(catalog["train_counts"], catalog["video_ids"])
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    for query in range(len(output)):
        output[query] = topk_from_order(order, bits[query])
    return output


def rank_random(artifacts: dict[str, Any], seed: int) -> np.ndarray:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    item_count = len(catalog["video_ids"])
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    for query in range(len(output)):
        candidates = candidate_positions(bits[query], item_count)
        query_id = f"{int(queries['user_ids'][query])}|{float(queries['timestamp'][query]):.6f}"
        ranked = heapq.nlargest(
            min(100, len(candidates)),
            (int(item) for item in candidates),
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}|{query_id}|{int(catalog['video_ids'][item])}".encode()
                ).digest(),
                -int(catalog["video_ids"][item]),
            ),
        )
        output[query].fill(-1)
        output[query, : len(ranked)] = ranked
    return output


def _fit_decayed_scores(artifacts: dict[str, Any], half_life_days: float) -> np.ndarray:
    train = artifacts["train"]
    item_count = len(artifacts["catalog"]["video_ids"])
    reference = float(artifacts["catalog"]["train_end"][0])
    scale = math.log(2.0) / (half_life_days * 86400.0)
    weights = np.exp(-scale * (reference - train["timestamp"]))
    return np.bincount(train["item"], weights=weights, minlength=item_count)


def rank_fit_frozen_decayed(
    artifacts: dict[str, Any], half_life_days: float
) -> np.ndarray:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    scores = _fit_decayed_scores(artifacts, half_life_days)
    order = popularity_order(scores, catalog["video_ids"])
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    for query in range(len(output)):
        output[query] = topk_from_order(order, bits[query])
    return output


def _insert_updated_item(
    order: list[int], item: int, scores: np.ndarray, video_ids: np.ndarray
) -> None:
    old = order.index(item)
    order.pop(old)
    key_score = float(scores[item])
    key_video = int(video_ids[item])
    low = 0
    high = old
    while low < high:
        middle = (low + high) // 2
        other = order[middle]
        before = (float(scores[other]) > key_score) or (
            float(scores[other]) == key_score and int(video_ids[other]) < key_video
        )
        if before:
            low = middle + 1
        else:
            high = middle
    order.insert(low, item)


def rank_causal_streaming_decayed(
    artifacts: dict[str, Any], half_life_days: float
) -> np.ndarray:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    video_ids = catalog["video_ids"]
    scores = _fit_decayed_scores(artifacts, half_life_days)
    order = popularity_order(scores, video_ids).astype(int).tolist()
    scale = math.log(2.0) / (half_life_days * 86400.0)
    reference = float(catalog["train_end"][0])
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    chronological = np.argsort(queries["timestamp"], kind="mergesort")
    cursor = 0
    while cursor < len(chronological):
        query_time = float(queries["timestamp"][chronological[cursor]])
        end = cursor + 1
        while end < len(chronological) and float(
            queries["timestamp"][chronological[end]]
        ) == query_time:
            end += 1
        decay = math.exp(-scale * (query_time - reference))
        scores *= decay
        reference = query_time
        group = chronological[cursor:end]
        for query in group:
            output[query] = topk_from_order(order, bits[query])
        updates: Counter[int] = Counter()
        for query in group:
            targets = queries["target_indices"][
                queries["target_indptr"][query] : queries["target_indptr"][query + 1]
            ]
            updates.update(int(item) for item in targets)
        for item, count in updates.items():
            scores[item] += count
            _insert_updated_item(order, item, scores, video_ids)
        cursor = end
    return output


def _itemcf_neighbors(
    artifacts: dict[str, Any], *, neighbor_count: int, shrinkage: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cooc = artifacts["cooccurrence"]
    counts = np.asarray(artifacts["user_item"].sum(axis=0)).ravel()
    video_ids = artifacts["catalog"]["video_ids"]
    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for item in range(cooc.shape[0]):
        start, end = cooc.indptr[item], cooc.indptr[item + 1]
        neighbors = cooc.indices[start:end]
        values = cooc.data[start:end].astype(np.float64)
        if len(neighbors):
            cosine = values / np.sqrt(counts[item] * counts[neighbors])
            similarity = cosine * values / (values + shrinkage)
            order = np.lexsort((video_ids[neighbors], -similarity))[:neighbor_count]
            indices.append(neighbors[order].astype(np.int32))
            weights.append(similarity[order].astype(np.float32))
        else:
            indices.append(np.asarray([], dtype=np.int32))
            weights.append(np.asarray([], dtype=np.float32))
    return indices, weights


def _add_history_item(
    score: np.ndarray,
    item: int,
    seen_history: set[int],
    neighbors: list[np.ndarray],
    weights: list[np.ndarray],
) -> None:
    if item in seen_history:
        return
    seen_history.add(item)
    score[neighbors[item]] += weights[item]


def _top_itemcf(
    score: np.ndarray,
    bits: np.ndarray,
    fallback_order: np.ndarray,
    video_ids: np.ndarray,
) -> np.ndarray:
    positive = np.flatnonzero(score > 0)
    positive = np.asarray(
        [item for item in positive if _contains(bits, int(item))], dtype=np.int32
    )
    if len(positive):
        positive = positive[np.lexsort((video_ids[positive], -score[positive]))]
    result: list[int] = [int(item) for item in positive[:100]]
    selected = set(result)
    if len(result) < 100:
        for value in fallback_order:
            item = int(value)
            if item not in selected and _contains(bits, item):
                result.append(item)
                if len(result) == 100:
                    break
    output = np.full(100, -1, dtype=np.int32)
    output[: len(result)] = result
    return output


def rank_itemcf(
    artifacts: dict[str, Any], *, neighbor_count: int, shrinkage: float
) -> np.ndarray:
    catalog = artifacts["catalog"]
    events = artifacts["events"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    video_ids = catalog["video_ids"]
    fallback = popularity_order(catalog["train_counts"], video_ids)
    neighbors, weights = _itemcf_neighbors(
        artifacts, neighbor_count=neighbor_count, shrinkage=shrinkage
    )
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    query_by_user: dict[int, list[int]] = defaultdict(list)
    for query, user in enumerate(queries["user"]):
        query_by_user[int(user)].append(query)
    train_end = float(catalog["train_end"][0])
    for user, user_queries in query_by_user.items():
        start = int(events["user_indptr"][user])
        end = int(events["user_indptr"][user + 1])
        times = events["timestamp"][start:end]
        items = events["item"][start:end]
        strong = events["strong"][start:end]
        score = np.zeros(len(video_ids), dtype=np.float32)
        history: set[int] = set()
        initial = np.flatnonzero(strong & (times < train_end))
        for position in initial:
            _add_history_item(score, int(items[position]), history, neighbors, weights)
        cursor = int(np.searchsorted(times, train_end, side="left"))
        for query in user_queries:
            query_time = float(queries["timestamp"][query])
            left = int(np.searchsorted(times, query_time, side="left"))
            for position in range(cursor, left):
                if strong[position]:
                    _add_history_item(
                        score, int(items[position]), history, neighbors, weights
                    )
            cursor = left
            output[query] = _top_itemcf(
                score, bits[query], fallback, video_ids
            )
            right = int(np.searchsorted(times, query_time, side="right"))
            for position in range(cursor, right):
                if strong[position]:
                    _add_history_item(
                        score, int(items[position]), history, neighbors, weights
                    )
            cursor = right
    return output


def _adam_step(
    values: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: int,
    learning_rate: float,
) -> None:
    first *= 0.9
    first += 0.1 * gradient
    second *= 0.999
    second += 0.001 * gradient * gradient
    first_hat = first / (1.0 - 0.9**step)
    second_hat = second / (1.0 - 0.999**step)
    values -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)


def train_bpr_checkpoints(
    artifacts: dict[str, Any],
    *,
    embedding_dim: int,
    learning_rate: float,
    l2: float,
    seed: int,
    checkpoints: tuple[int, ...] = (5, 10, 20),
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    train = artifacts["train"]
    negatives = artifacts["bpr_negatives"][f"seed_{seed}"]
    user_count = artifacts["user_item"].shape[0]
    item_count = artifacts["user_item"].shape[1]
    rng = np.random.default_rng(seed)
    users = rng.normal(0, 0.05, size=(user_count, embedding_dim)).astype(np.float32)
    items = rng.normal(0, 0.05, size=(item_count, embedding_dim)).astype(np.float32)
    user_m = np.zeros_like(users)
    user_v = np.zeros_like(users)
    item_m = np.zeros_like(items)
    item_v = np.zeros_like(items)
    batch_size = 4096
    step = 0
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    order = np.arange(len(train["user"]), dtype=np.int64)
    for epoch in range(1, max(checkpoints) + 1):
        rng.shuffle(order)
        for begin in range(0, len(order), batch_size):
            batch = order[begin : begin + batch_size]
            user_idx = train["user"][batch]
            positive_idx = train["item"][batch]
            negative_idx = negatives[batch]
            user_vec = users[user_idx]
            positive_vec = items[positive_idx]
            negative_vec = items[negative_idx]
            difference = positive_vec - negative_vec
            logits = np.sum(user_vec * difference, axis=1)
            coefficient = 1.0 / (1.0 + np.exp(np.clip(logits, -30, 30)))
            user_grad = -coefficient[:, None] * difference + l2 * user_vec
            positive_grad = -coefficient[:, None] * user_vec + l2 * positive_vec
            negative_grad = coefficient[:, None] * user_vec + l2 * negative_vec
            step += 1
            # Accumulate repeated indices before the Adam update.
            for index_array, gradient, matrix, first, second in (
                (user_idx, user_grad, users, user_m, user_v),
                (positive_idx, positive_grad, items, item_m, item_v),
                (negative_idx, negative_grad, items, item_m, item_v),
            ):
                unique, inverse = np.unique(index_array, return_inverse=True)
                accumulated = np.zeros((len(unique), embedding_dim), dtype=np.float32)
                np.add.at(accumulated, inverse, gradient)
                # Advanced indexing returns copies; write the updated state back.
                values = matrix[unique].copy()
                moments = first[unique].copy()
                variances = second[unique].copy()
                _adam_step(
                    values,
                    accumulated,
                    moments,
                    variances,
                    step,
                    learning_rate,
                )
                matrix[unique] = values
                first[unique] = moments
                second[unique] = variances
        if epoch in checkpoints:
            output[epoch] = (users.copy(), items.copy())
    return output


def rank_bpr(
    artifacts: dict[str, Any],
    users: np.ndarray,
    items: np.ndarray,
    *,
    fallback_topk: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    bits = artifacts["candidate_bits"]
    video_ids = catalog["video_ids"]
    trained_users = np.asarray(artifacts["user_item"].sum(axis=1)).ravel() > 0
    trained_items = np.asarray(artifacts["user_item"].sum(axis=0)).ravel() > 0
    output = np.empty((len(queries["user"]), 100), dtype=np.int32)
    query_by_user: dict[int, list[int]] = defaultdict(list)
    for query, user in enumerate(queries["user"]):
        query_by_user[int(user)].append(query)
    fallback_queries = 0
    fallback_users = 0
    for user, query_rows in query_by_user.items():
        if not trained_users[user]:
            fallback_users += 1
            fallback_queries += len(query_rows)
            for query in query_rows:
                output[query] = fallback_topk[query]
            continue
        scores = items @ users[user]
        scores = scores.astype(np.float64)
        scores[~trained_items] = -np.inf
        order = np.lexsort((video_ids, -scores))
        for query in query_rows:
            output[query] = topk_from_order(order, bits[query])
    return output, {
        "fallback_user_count": fallback_users,
        "fallback_query_count": fallback_queries,
    }
