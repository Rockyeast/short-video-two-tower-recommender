# Phase B5B Recall-Preserving Local Reranker

The frozen Hybrid selects Top-100 candidates. LightGBM may only change their order; it cannot add or remove a Top-100 item.

| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Hybrid alpha=0.75 | 0.015980 | 0.035032 | 0.070289 | 0.016535 | 0.498131 | 0.057113 |
| Local LightGBM LambdaRank | 0.051985 | 0.064973 | 0.070289 | 0.059983 | 0.498131 | 0.057113 |

- Gate passed: `true`
- Checks: `{'ndcg20_strictly_higher': True, 'recall100_exactly_equal': True, 'coverage100_exactly_equal': True}`
- Fit users: `4797`
- Evaluation users: `2019`
- Accepted training groups: `2154`
- Wall time: `594.420 s`
- Peak RSS: `4272.46 MiB`

This is an adaptive Big-validation development iteration after the B5A failure. It is not a sealed or untouched test. Small Matrix, temporal final, and final-refit artifacts were not used.
