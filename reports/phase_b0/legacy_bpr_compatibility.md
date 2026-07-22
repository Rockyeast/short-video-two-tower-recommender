# Legacy BPR Checkpoint Compatibility

Status: **REJECTED for fully-observed V1 reuse**.

This is a metadata/code inspection only. It did not run a model, read temporal
final, or access Small Matrix.

| Requirement | Finding | Compatible |
|---|---|---:|
| Split | Both routes reference committed split manifest `e271c45d...` and use its train window for selection fit. | Yes |
| Positive label | Both use canonical strong positives with strict `watch_ratio > 2.0`. | Yes |
| Positional shapes | Legacy selected checkpoints contain `users=(7176, 64)` and `items=(10728, 64)`, matching their frozen catalog arrays. | Structurally |
| ID binding | Legacy factors depend on the positional `catalog.npz` mapping; fully-observed V1 uses raw IDs explicitly. An adapter could preserve this mapping. | Adaptable |
| Negative population | Legacy BPR samples from each target timestamp's causal available/unseen catalog. Fully-observed V1 samples from fit-observed `NORMAL` items after excluding all fit-known positives. | **No** |
| Epoch sampling | Legacy `bpr_negative_indices.npz` fixes one negative array per seed across training. Fully-observed V1 resamples once per positive every epoch. | **No** |
| Cold-item score | Legacy contract assigns untrained IDs negative infinity. Fully-observed V1 assigns data-cold items zero. | **No** |

Because the optimization examples and cold-item semantics differ, the old
checkpoint is not the same baseline even when split, label, shapes and mapping
can be aligned. Phase B must train one fully-observed-V1 BPR implementation;
the old three-seed grid must not be copied or presented as this result.
