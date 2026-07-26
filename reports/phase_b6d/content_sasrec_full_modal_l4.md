# Phase B6D Content-Enhanced SASRec

One fixed, single-seed development run on reused Big validation.
Frozen MiniLM caption vectors are projected into SASRec; training-unseen items have their ID residual disabled.

| Epoch | Loss | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.269463 | 0.024269 | 0.040917 | 0.061728 | 0.029617 | 0.206834 | 0.000000 |
| 2 | 6.703339 | 0.026492 | 0.044724 | 0.067818 | 0.033792 | 0.233422 | 0.000000 |
| 3 | 6.571044 | 0.026916 | 0.047806 | 0.071089 | 0.033346 | 0.246129 | 0.000000 |
| 4 | 6.507541 | 0.030255 | 0.051792 | 0.074631 | 0.037274 | 0.262787 | 0.000000 |
| 5 | 6.464043 | 0.029176 | 0.051527 | 0.076347 | 0.036381 | 0.266311 | 0.000000 |

Selected epoch: `5`.

Small Matrix and temporal final were not accessed. These are adaptive Big-validation point estimates, not sealed-test or significance evidence.

## Modal L4 resources

- GPU: `NVIDIA L4`
- Remote wall time: `1168.19s`
- Peak CUDA allocated/reserved: `294.4/386.0 MB`
- Peak RSS: `5005.3 MB`
