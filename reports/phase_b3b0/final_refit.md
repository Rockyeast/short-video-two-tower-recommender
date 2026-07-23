# Phase B3B0 Final Recipe Refit

The recipe was frozen on Big validation. This run refit the selected methods
from scratch on canonical Big train plus validation and performed no model
selection.

| Method | Fit interactions | Strong positives/examples | Optimizer steps | Final loss | Wall time |
|---|---:|---:|---:|---:|---:|
| BPR epoch 20 | 10,024,291 | 752,130 | 3,680 | 0.159871 | 63.35 s |
| Two-Tower epoch 1 | 10,024,291 | 689,023 | 2,692 | 4.733988 | 379.26 s |

- Global Popularity: Big train+validation strong-positive counts; 9,111 items
  received a nonzero count.
- BPR checkpoint:
  `kuairec-b3b-final-refit-artifacts/phase-b3b-final-v1/artifacts/final_bpr_epoch_020.npz`
- Two-Tower checkpoint:
  `kuairec-b3b-final-refit-artifacts/phase-b3b-final-v1/artifacts/final_two_tower_epoch_001.pt`
- Two-Tower skipped batches: `0`
- Full refit item/content universe: `10,725` items
- Caption cache coverage: `10,725 / 10,725`
- Device: `NVIDIA L4 / cuda:0`
- Total runner wall time: `446.44 s`
- Peak RSS: `8,018.36 MiB`
- Selection performed: `false`
- Small Matrix accessed: `false`
- Temporal final accessed: `false`

The sealed Small evaluator was exercised only with synthetic fixtures. No Small
metric was computed in Phase B3B0.
