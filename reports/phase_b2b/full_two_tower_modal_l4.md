# Phase B2B Full Two-Tower — Modal L4

- Runner commit: `7feb5675b7fa6577c68a3775d943c0a32b94f603`
- Wrapper commit at run: `dbf663a98631581efbd801b773e98250b7217b8e`
- GPU/device: `NVIDIA L4` / `cuda:0`
- Execution mode: `fresh_full_train`
- Selected epoch: `1`

## Epoch results

| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.799012 | 0.072057 | 0.012113 | 0.569461 | 0.065151 |
| 2 | 4.212923 | 0.066256 | 0.014936 | 0.531020 | 0.039149 |
| 3 | 4.032735 | 0.059768 | 0.016960 | 0.537320 | 0.026287 |

## Selected result

- Recall@100: `0.072056850`
- NDCG@20: `0.012112517`
- Coverage@100: `0.569460758`
- Data-Cold Recall@100: `0.065150571`
- Gate A/B/C: `{"A": true, "B": true, "C": true}`

## Runtime

- Runner wall: `904.182 s`
- Training + epoch validation: `641.172 s`
- Checkpoint reevaluation: `0.000 s`
- Peak CUDA allocated/reserved: `171.24 / 258.00 MiB`
- Peak RSS: `8089.98 MiB`

No Small Matrix, temporal final, FAISS, Hybrid, reranker, serving, or monitoring run was performed.
