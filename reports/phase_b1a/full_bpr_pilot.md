# Phase B1A Full BPR Pilot

Big train fit and Big validation selection only. Small Matrix, temporal final and Two-Tower were not accessed or run.

## Baselines

| Method | Recall@100 | NDCG@20 | Coverage@100 |
|---|---:|---:|---:|
| Random | 0.012930 | 0.002675 | 1.000000 |
| Global Popularity | 0.036643 | 0.010615 | 0.080085 |

## BPR checkpoints

| Epoch | Audit loss | Audit win | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.265469 | 0.8880 | 0.014670 | 0.027817 | 0.041036 | 0.012690 | 0.239402 |
| 10 | 0.214120 | 0.9092 | 0.012320 | 0.027641 | 0.045177 | 0.010328 | 0.197758 |
| 15 | 0.184048 | 0.9226 | 0.012261 | 0.029452 | 0.047351 | 0.010869 | 0.257768 |
| 20 | 0.158740 | 0.9342 | 0.013891 | 0.030344 | 0.048439 | 0.012774 | 0.333049 |

Status: `completed`.
Selected checkpoint: `20`.
Stop reason: `None`.

Validation queries: `6818`; targets: `118565`; runtime: `11.03 minutes`.

The fixed audit-negative metrics are optimization diagnostics, not recommendation-effectiveness claims.
