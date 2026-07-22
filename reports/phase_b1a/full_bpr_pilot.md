# Phase B1A Full BPR Pilot

Big train fit and Big validation selection only. Small Matrix, temporal final and Two-Tower were not accessed or run.

## Baselines

| Method | Recall@100 | NDCG@20 | Coverage@100 |
|---|---:|---:|---:|
| Random | 0.012930 | 0.002675 | 1.000000 |
| Global Popularity | 0.036643 | 0.010615 | 0.080085 |

## BPR checkpoints

| Epoch | Diagnostic loss | Diagnostic win | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.265469 | 0.8880 | 0.014670 | 0.027817 | 0.041036 | 0.012690 | 0.239402 |
| 10 | 0.214120 | 0.9092 | 0.012320 | 0.027641 | 0.045177 | 0.010328 | 0.197758 |
| 15 | 0.184048 | 0.9226 | 0.012261 | 0.029452 | 0.047351 | 0.010869 | 0.257768 |
| 20 | 0.158740 | 0.9342 | 0.013891 | 0.030344 | 0.048439 | 0.012774 | 0.333049 |

Status: `completed`.
Selected checkpoint: `20`.
Stop reason: `None`.

Primary warm-user queries: `6816`; targets: `118539`. Cold-user audit queries:
`2`; targets: `26`. Runtime: `11.03 minutes`.

The separate-seed fixed diagnostic metrics measure optimization only; they are
not recommendation-effectiveness claims.

The selected BPR checkpoint's Data-Cold Recall@100 was `0` in this run,
consistent with a pure ID model being unable to learn videos unseen during
training; this is an observed result, not a claim of mathematical inevitability.

BPR's Recall@100 improvement over Global Popularity was `32.2%` relative in the
frozen Big-validation protocol for this single seed. No statistical-significance
or cross-dataset-stability claim is made.

## Input traceability

`item_daily_features.csv` was verified before NORMAL membership was read:

- Input: `KUAIREC_DATA_DIR/item_daily_features.csv`
- Actual SHA256: `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`
- Expected SHA256: `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`
- Sorted unique NORMAL items: `10699`
- Membership SHA256: `631a7c7cc93413f250f36f548feb720f8322050010e291afcc88338155f52c8e`

These traceability fields were added without retraining BPR or changing the
frozen metrics, checkpoints or model-selection rule.
