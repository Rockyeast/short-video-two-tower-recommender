# Phase B6B Three-Route Hybrid

This is adaptive development on the already reused Big validation set. It is not sealed-test or statistical-significance evidence.

| Route | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| BPR | 0.013891 | 0.030344 | 0.048439 | 0.012774 | 0.333049 | 0.000000 |
| Two-Tower | 0.014870 | 0.036038 | 0.072057 | 0.012113 | 0.569461 | 0.065151 |
| SASRec | 0.030657 | 0.052639 | 0.077560 | 0.037462 | 0.252856 | 0.000000 |
| Hybrid TT=0.60 SASRec=0.25 BPR=0.15 | 0.017737 | 0.041720 | 0.082561 | 0.020006 | 0.557395 | 0.049244 |
| Hybrid TT=0.50 SASRec=0.35 BPR=0.15 | 0.022135 | 0.049237 | 0.085315 | 0.025961 | 0.524933 | 0.032707 |
| Hybrid TT=0.40 SASRec=0.45 BPR=0.15 | 0.031806 | 0.053105 | 0.084606 | 0.037131 | 0.467806 | 0.017436 |

No three-route candidate passed all frozen preservation gates.

Selection first preserves 98% of Recall@100, 90% of Coverage@100, and 90% of Data-Cold Recall@100 from the frozen Two-Tower+BPR hybrid; it then maximizes NDCG@20.

No model was trained. Small Matrix and temporal final were not accessed.
