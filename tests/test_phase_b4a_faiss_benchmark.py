from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from kuairec_fully_observed.faiss_benchmark import (  # noqa: E402
    HNSW_CONFIG,
    VECTOR_DIMENSION,
    benchmark_catalog,
    extend_catalog,
    faiss_search,
    normalize_rows,
    numpy_exact_search,
    stable_topk_from_scores,
    topk_overlap,
)


def test_stable_topk_uses_item_position_as_tie_break():
    scores = np.asarray([0.5, 0.8, 0.8, 0.1, 0.8], dtype=np.float32)
    np.testing.assert_array_equal(stable_topk_from_scores(scores, 3), [1, 2, 4])


def test_synthetic_extension_is_deterministic_normalized_and_keeps_real_prefix():
    real = normalize_rows(
        np.arange(3 * VECTOR_DIMENSION, dtype=np.float32).reshape(3, -1) + 1
    )
    first = extend_catalog(real, target_count=19, seed=17)
    second = extend_catalog(real, target_count=19, seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[: len(real)], real)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_flat_and_frozen_hnsw_match_exact_on_separated_toy_vectors():
    generator = np.random.default_rng(9)
    catalog = normalize_rows(
        generator.standard_normal((128, VECTOR_DIMENSION), dtype=np.float32)
    )
    queries = normalize_rows(
        catalog[:16]
        + 0.01
        * generator.standard_normal(
            (16, VECTOR_DIMENSION), dtype=np.float32
        )
    )
    exact, _ = numpy_exact_search(queries, catalog, k=10)

    flat = faiss.IndexFlatIP(VECTOR_DIMENSION)
    flat.add(catalog)
    flat_topk, _ = faiss_search(queries, flat, k=10)
    assert topk_overlap(exact, flat_topk) == 1.0

    hnsw = faiss.IndexHNSWFlat(
        VECTOR_DIMENSION,
        HNSW_CONFIG["m"],
        faiss.METRIC_INNER_PRODUCT,
    )
    hnsw.hnsw.efConstruction = HNSW_CONFIG["ef_construction"]
    hnsw.hnsw.efSearch = HNSW_CONFIG["ef_search"]
    hnsw.add(catalog)
    hnsw_topk, _ = faiss_search(queries, hnsw, k=10)
    assert topk_overlap(exact, hnsw_topk) == 1.0


def test_benchmark_report_separates_synthetic_scope_and_ann_gate():
    generator = np.random.default_rng(31)
    catalog = normalize_rows(
        generator.standard_normal((512, VECTOR_DIMENSION), dtype=np.float32)
    )
    queries = normalize_rows(
        generator.standard_normal((8, VECTOR_DIMENSION), dtype=np.float32)
    )
    report = benchmark_catalog(
        queries=queries,
        catalog=catalog,
        scale_name="synthetic_scale_extension_test",
        synthetic_extension=True,
    )
    assert report["data_scope"] == "synthetic_scale_extension"
    assert report["recommendation_effectiveness_claim"] is False
    assert report["faiss_index_flat_ip"]["top100_overlap_vs_exact"] == 1.0
    assert report["faiss_hnsw"]["passes_99_percent_gate"] is True
