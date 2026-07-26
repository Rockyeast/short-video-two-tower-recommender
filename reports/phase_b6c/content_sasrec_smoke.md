# Phase B6C Content-SASRec Smoke

This bounded smoke adds frozen MiniLM caption vectors to RecBole SASRec through a trainable projection. Training-seen items retain an ID residual; training-unseen items use only content.

| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.089587 | 0.024337 | 0.005468 | 0.030005 | 0.036719 |
| 2 | 8.601322 | 0.027187 | 0.009972 | 0.037480 | 0.022974 |

This is a 10K-example, 20-step-per-epoch smoke on 128 reused Big-validation queries. It is not an effectiveness result.

Small Matrix and temporal final were not accessed.
