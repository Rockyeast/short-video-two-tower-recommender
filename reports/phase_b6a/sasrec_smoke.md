# Phase B6A RecBole SASRec

This is a **bounded smoke** on Big validation. Small Matrix and temporal final were not accessed.

| Epoch | Loss | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.160416 | 0.002842 | 0.014411 | 0.024594 | 0.005374 | 0.057982 | 0.000000 |
| 2 | 8.440892 | 0.006613 | 0.016988 | 0.030393 | 0.009224 | 0.046556 | 0.000000 |

Selected epoch: `2`.
Wall time: `60.91s`; peak RSS: `1781.2 MB`.

The model implementation is RecBole 1.2.1 SASRec. The adapter reuses the repository's frozen candidate membership and metrics. This is adaptive Big-validation development, not a sealed test.
