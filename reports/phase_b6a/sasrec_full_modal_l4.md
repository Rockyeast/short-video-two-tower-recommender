# Phase B6A RecBole SASRec

This is a **full development run** on Big validation. Small Matrix and temporal final were not accessed.

| Epoch | Loss | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.210746 | 0.023380 | 0.038598 | 0.057039 | 0.028799 | 0.180139 | 0.000000 |
| 2 | 6.681440 | 0.027468 | 0.047473 | 0.070035 | 0.032985 | 0.224026 | 0.000000 |
| 3 | 6.550585 | 0.027150 | 0.048268 | 0.073477 | 0.034159 | 0.238868 | 0.000000 |
| 4 | 6.481129 | 0.029196 | 0.051562 | 0.075772 | 0.034441 | 0.248478 | 0.000000 |
| 5 | 6.436673 | 0.030657 | 0.052639 | 0.077560 | 0.037462 | 0.252856 | 0.000000 |

Selected epoch: `5`.
Wall time: `1216.78s`; peak RSS: `4959.0 MB`.

The model implementation is RecBole 1.2.1 SASRec. The adapter reuses the repository's frozen candidate membership and metrics. This is adaptive Big-validation development, not a sealed test.

## Comparison with existing retrieval models

| Model | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|
| BPR epoch 20 | 0.048439 | 0.012774 | 0.333049 | 0.000000 |
| Two-Tower epoch 1 | 0.072057 | 0.012113 | 0.569461 | 0.065151 |
| RecBole SASRec epoch 5 | **0.077560** | **0.037462** | 0.252856 | 0.000000 |

Relative to Two-Tower, SASRec improved Recall@100 by `7.64%` and
NDCG@20 by `209.28%`, but reduced Coverage@100 by `55.60%` and did not
retrieve any train-unseen Data-Cold target. The result supports SASRec as a
warm-item sequential route; it does not replace the content-aware Two-Tower.

The experiment used one fixed configuration and one seed on reused Big
validation development data. It is not a sealed result and makes no
statistical-significance or cross-dataset claim.

## Modal L4 resources

- GPU: `NVIDIA L4`
- Remote wall time: `1219.23s`
- Peak CUDA allocated/reserved: `260.0/374.0 MB`
- Peak RSS: `4959.0 MB`
