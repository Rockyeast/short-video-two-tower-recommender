# Phase B2B0 Full Runner Preflight

This is a bounded engineering preflight through the production runner path. It is not a formal effectiveness experiment.

- Device: `cpu`
- Runtime: `151.07 s`
- Peak RSS: `1712.71 MB`
- Save/load/resume verified: `true`
- Training examples: `2560`
- Optimizer steps: `20`
- Validation queries: `128`
- Estimated full run: `358.7` to `597.8` minutes

| Epoch | Loss | Recall@100 | NDCG@20 | Coverage@100 |
|---:|---:|---:|---:|---:|
| 1 | 5.877541 | 0.002072 | 0.000000 | 0.056167 |
| 2 | 5.074812 | 0.020326 | 0.003371 | 0.060331 |

Required claim boundary:

```text
formal_gate_executed=false
effectiveness_claim=false
full_big_train=false
full_big_validation=false
```

Small Matrix, temporal final, FAISS and Hybrid were not accessed or run.
