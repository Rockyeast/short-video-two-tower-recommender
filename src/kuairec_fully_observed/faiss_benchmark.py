"""Bounded exact-versus-FAISS retrieval scalability benchmark."""

from __future__ import annotations

import gc
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


VECTOR_DIMENSION = 128
TOP_K = 100
BENCHMARK_SEED = 20260724
QUERY_LIMIT = 256
THREAD_COUNT = 8
SCALE_SPECS = (
    ("real_10k_catalog", 10_725, False),
    ("synthetic_scale_extension_100k", 100_000, True),
    ("synthetic_scale_extension_1m", 1_000_000, True),
)
HNSW_CONFIG = {
    "m": 32,
    "ef_construction": 200,
    "ef_search": 512,
}


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised by deployment
        raise RuntimeError("Phase B4A requires faiss-cpu") from exc
    return faiss


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return finite, contiguous FP32 unit vectors."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != VECTOR_DIMENSION:
        raise ValueError("Vectors must have shape [n, 128]")
    if not np.isfinite(array).all():
        raise ValueError("Vectors contain NaN or Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero vectors cannot be normalized")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def require_unit_rows(values: np.ndarray) -> np.ndarray:
    """Validate already-normalized contiguous FP32 vectors without copying."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != VECTOR_DIMENSION:
        raise ValueError("Vectors must have shape [n, 128]")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if not np.isfinite(array).all():
        raise ValueError("Vectors contain NaN or Inf")
    if not np.allclose(
        np.linalg.norm(array, axis=1), 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Benchmark vectors must be L2 normalized")
    return array


def extend_catalog(
    real_vectors: np.ndarray,
    *,
    target_count: int,
    seed: int = BENCHMARK_SEED,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Append deterministic normalized distractors to the real item vectors."""

    real = normalize_rows(real_vectors)
    if target_count < len(real):
        raise ValueError("Target catalog cannot truncate real item vectors")
    output = np.empty((target_count, VECTOR_DIMENSION), dtype=np.float32)
    output[: len(real)] = real
    generator = np.random.default_rng(seed)
    for begin in range(len(real), target_count, chunk_size):
        end = min(begin + chunk_size, target_count)
        distractors = generator.standard_normal(
            (end - begin, VECTOR_DIMENSION), dtype=np.float32
        )
        output[begin:end] = normalize_rows(distractors)
    return output


def stable_topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Select score-descending, item-position-ascending Top-K."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be one finite vector")
    count = min(int(k), len(values))
    if count <= 0:
        raise ValueError("Top-K must be positive")
    cutoff = np.partition(values, len(values) - count)[len(values) - count]
    candidates = np.flatnonzero(values >= cutoff)
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order[:count]].astype(np.int64, copy=False)


def numpy_exact_search(
    queries: np.ndarray,
    catalog: np.ndarray,
    *,
    k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure single-query NumPy exact inner-product retrieval."""

    query_vectors = require_unit_rows(queries)
    item_vectors = require_unit_rows(catalog)
    topk = np.empty((len(query_vectors), min(k, len(item_vectors))), np.int64)
    latencies_ms = np.empty(len(query_vectors), np.float64)
    for row, query in enumerate(query_vectors):
        started = time.perf_counter()
        scores = item_vectors @ query
        topk[row] = stable_topk_from_scores(scores, k)
        latencies_ms[row] = (time.perf_counter() - started) * 1000.0
    return topk, latencies_ms


def _stable_faiss_rows(
    distances: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    output = np.empty_like(positions, dtype=np.int64)
    for row, (scores, items) in enumerate(
        zip(distances, positions, strict=True)
    ):
        if np.any(items < 0):
            raise RuntimeError("FAISS returned padding for a full catalog")
        order = np.lexsort((items, -scores))
        output[row] = items[order]
    return output


def faiss_search(
    queries: np.ndarray,
    index,
    *,
    k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure single-query FAISS search under the shared output tie contract."""

    query_vectors = require_unit_rows(queries)
    topk = np.empty((len(query_vectors), k), np.int64)
    latencies_ms = np.empty(len(query_vectors), np.float64)
    for row, query in enumerate(query_vectors):
        started = time.perf_counter()
        distances, positions = index.search(query[None, :], k)
        topk[row] = _stable_faiss_rows(distances, positions)[0]
        latencies_ms[row] = (time.perf_counter() - started) * 1000.0
    return topk, latencies_ms


def topk_overlap(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Macro-average fraction of reference Top-K IDs recovered."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Top-K arrays must have the same two-dimensional shape")
    return float(
        np.mean(
            [
                len(set(left.tolist()) & set(right.tolist())) / len(left)
                for left, right in zip(reference, candidate, strict=True)
            ]
        )
    )


def _latency_summary(latencies_ms: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "qps": float(1000.0 / np.mean(latencies_ms)),
        "total_search_s": float(latencies_ms.sum() / 1000.0),
    }


def _index_size_bytes(index) -> int:
    faiss = _require_faiss()
    with tempfile.TemporaryDirectory(prefix="phase-b4a-faiss-") as directory:
        path = Path(directory) / "index.faiss"
        faiss.write_index(index, str(path))
        return path.stat().st_size


def benchmark_catalog(
    *,
    queries: np.ndarray,
    catalog: np.ndarray,
    scale_name: str,
    synthetic_extension: bool,
) -> dict[str, Any]:
    """Benchmark Exact, FlatIP and one frozen HNSW configuration."""

    faiss = _require_faiss()
    faiss.omp_set_num_threads(THREAD_COUNT)
    query_vectors = require_unit_rows(queries)
    item_vectors = require_unit_rows(catalog)
    if len(query_vectors) > QUERY_LIMIT:
        raise ValueError("Query limit exceeds the frozen Phase B4A contract")

    exact_started = time.perf_counter()
    exact_topk, exact_latency = numpy_exact_search(
        query_vectors, item_vectors, k=TOP_K
    )
    exact_runtime = time.perf_counter() - exact_started

    flat = faiss.IndexFlatIP(VECTOR_DIMENSION)
    started = time.perf_counter()
    flat.add(item_vectors)
    flat_build_s = time.perf_counter() - started
    flat_topk, flat_latency = faiss_search(query_vectors, flat, k=TOP_K)
    flat_overlap = topk_overlap(exact_topk, flat_topk)
    flat_size = _index_size_bytes(flat)

    hnsw = faiss.IndexHNSWFlat(
        VECTOR_DIMENSION,
        HNSW_CONFIG["m"],
        faiss.METRIC_INNER_PRODUCT,
    )
    hnsw.hnsw.efConstruction = HNSW_CONFIG["ef_construction"]
    hnsw.hnsw.efSearch = HNSW_CONFIG["ef_search"]
    started = time.perf_counter()
    hnsw.add(item_vectors)
    hnsw_build_s = time.perf_counter() - started
    hnsw_topk, hnsw_latency = faiss_search(query_vectors, hnsw, k=TOP_K)
    hnsw_overlap = topk_overlap(flat_topk, hnsw_topk)
    hnsw_size = _index_size_bytes(hnsw)

    report = {
        "scale_name": scale_name,
        "catalog_count": int(len(item_vectors)),
        "query_count": int(len(query_vectors)),
        "vector_dimension": VECTOR_DIMENSION,
        "top_k": TOP_K,
        "data_scope": (
            "synthetic_scale_extension"
            if synthetic_extension
            else "real_10k_catalog"
        ),
        "recommendation_effectiveness_claim": False,
        "numpy_exact": {
            "index_build_s": 0.0,
            "index_size_bytes": int(item_vectors.nbytes),
            "runtime_s": exact_runtime,
            **_latency_summary(exact_latency),
            "top100_overlap_vs_exact": 1.0,
        },
        "faiss_index_flat_ip": {
            "index_build_s": flat_build_s,
            "index_size_bytes": flat_size,
            **_latency_summary(flat_latency),
            "top100_overlap_vs_exact": flat_overlap,
        },
        "faiss_hnsw": {
            "config": dict(HNSW_CONFIG),
            "index_build_s": hnsw_build_s,
            "index_size_bytes": hnsw_size,
            **_latency_summary(hnsw_latency),
            "top100_overlap_vs_exact": hnsw_overlap,
            "ann_recall_at_100_vs_index_flat_ip": hnsw_overlap,
            "passes_99_percent_gate": hnsw_overlap >= 0.99,
        },
        "peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }
    del hnsw, flat, exact_topk, flat_topk, hnsw_topk
    gc.collect()
    return report


def runtime_identity() -> dict[str, Any]:
    faiss = _require_faiss()
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "fixed_thread_count": THREAD_COUNT,
        "numpy_version": np.__version__,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
