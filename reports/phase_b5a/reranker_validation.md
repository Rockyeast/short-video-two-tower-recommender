# Phase B5A LightGBM Reranker Validation

A single frozen LambdaRank configuration was trained on a stable 70% user split of Big validation and evaluated on the disjoint 30% split.

| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Hybrid alpha=0.75 | 0.015980 | 0.035032 | 0.070289 | 0.016535 | 0.498131 | 0.057113 |
| LightGBM LambdaRank | 0.009867 | 0.020573 | 0.039180 | 0.010985 | 0.521089 | 0.050554 |

## Gate

- Passed: `false`
- Checks: `{'ndcg20_strictly_higher': False, 'recall100_retained': False, 'coverage100_retained': True}`
- Objective: improve NDCG@20 while retaining 98% of Hybrid Recall@100 and 90% of Hybrid Coverage@100.

## Split and training

- Reranker-fit users: `4797`
- Held-out evaluation users: `2019`
- Training groups accepted: `4223`
- Fit queries without a retrieved positive: `574`
- Training candidate rows: `1106943`

## Claim boundary

This is a validation-development experiment, not a new sealed test. The base retrieval models were fit only on Big train; reranker labels came only from the fit-user subset. Small Matrix, temporal final, and final-refit artifacts were not accessed.

- Total wall time: `747.484 s`
- Peak RSS: `4258.79 MiB`
