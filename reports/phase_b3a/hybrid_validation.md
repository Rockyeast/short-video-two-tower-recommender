# Phase B3A Minimal Two-Tower + BPR Hybrid Validation

Big validation only. This experiment trains no model and evaluates only the frozen weighted-RRF alpha grid.

| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| BPR alpha=0 | 0.013891 | 0.030344 | 0.048439 | 0.012774 | 0.333049 | 0.000000 |
| Two-Tower alpha=1 | 0.014870 | 0.036038 | 0.072057 | 0.012113 | 0.569461 | 0.065151 |
| Hybrid alpha=0.25 | 0.016996 | 0.032119 | 0.051845 | 0.017972 | 0.339669 | 0.000000 |
| Hybrid alpha=0.50 | 0.017336 | 0.035796 | 0.063496 | 0.020698 | 0.582488 | 0.024423 |
| Hybrid alpha=0.75 | 0.015643 | 0.037002 | 0.072213 | 0.015341 | 0.571169 | 0.060356 |

## Selection

- Recall@100 minimum: `0.070615713`
- Coverage@100 minimum: `0.512514682`
- Eligible alphas: `[0.75]`
- Selected alpha: `0.75`
- NDCG gap reduced: `true`

`alpha=0.75` met the frozen Recall/Coverage constraints and had the highest NDCG@20 among eligible Hybrid configurations.

## Runtime

- Total wall time: `710.456 s`
- Peak RSS: `4164.63 MiB`

No Small Matrix, temporal final, FAISS, LightGBM, training, service, or monitoring execution occurred.
