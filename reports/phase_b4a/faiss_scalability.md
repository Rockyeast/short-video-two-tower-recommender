# Phase B4A FAISS Scalability Benchmark

Engineering-only comparison using frozen final Two-Tower vectors. The 100K and 1M catalogs add deterministic normalized synthetic distractors and do not support a recommendation-effectiveness claim.

- Runner commit: `af4fa9aebba1cdbcea0cdbb7983fd99952db3db7`
- Wrapper commit: `04079513b2ef45fd76681467c2607ddb2732b43f`
- GPU used only for vector encoding: `NVIDIA L4`
- CPU: `unknown`
- Fixed threads / seed / query count: `8 / 20260724 / 256`
- HNSW M / efConstruction / efSearch: `32 / 200 / 512`

## Scale and latency

| Scope | Items | Exact p50/p95 ms | FlatIP p50/p95 ms | HNSW p50/p95 ms | HNSW QPS | HNSW Recall@100 | HNSW gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_10k_catalog | 10725 | 0.071/0.090 | 0.314/0.600 | 0.291/0.506 | 3147.00 | 0.985234 | False |
| synthetic_scale_extension_100k | 100000 | 0.420/0.568 | 1.273/1.856 | 0.712/1.893 | 1134.97 | 0.842773 | False |
| synthetic_scale_extension_1m | 1000000 | 9.665/11.297 | 13.370/14.911 | 3.902/7.956 | 230.50 | 0.546563 | False |

## Build time and index size

| Scope | Flat build s | Flat MiB | HNSW build s | HNSW MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| real_10k_catalog | 0.001 | 5.24 | 0.110 | 8.02 | 6713.50 |
| synthetic_scale_extension_100k | 0.013 | 48.83 | 9.540 | 74.79 | 6713.50 |
| synthetic_scale_extension_1m | 0.673 | 488.28 | 305.955 | 747.80 | 6713.50 |

## Conclusion

At the real 10K-scale catalog, NumPy Exact was faster than HNSW, so ANN is not needed at the current dataset scale.

- Total wall time: `538.347 s`
- Process peak RSS: `6713.50 MiB`
- `recommendation_effectiveness_claim=false`
- Small labels accessed: `False`
- Temporal final accessed: `False`
