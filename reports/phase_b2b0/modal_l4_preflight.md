# Phase B2B Modal NVIDIA L4 Preflight

This is the successful bounded 20-step GPU preflight. It is not a formal full-training or effectiveness run.

- Status: `passed`
- B2B runner commit: `7feb5675b7fa6577c68a3775d943c0a32b94f603`
- Modal wrapper commit: `2426556a123381b8cd2800bb74d19b748ad3459e`
- GPU: `NVIDIA L4`
- Device: `cuda:0`
- PyTorch/CUDA: `2.11.0+cu130` / `13.0`
- Modal SDK: `1.4.1`
- Input bundle: `7a7b8b370335f61d28063c7821600063c28929e2a837473890611cc1315f56a6`
- Input size: `1256272367` bytes

## Execution history

Six earlier bounded launch attempts failed closed while exposing wrapper and
runner blockers: local module packaging, Modal Volume directory enumeration,
clone contamination during package installation, source-relative helper
resolution, CUDA RNG checkpoint restoration, plus one host-side import failure.
None completed the preflight contract. The metrics below belong only to the
final successful run.

## Result

- Examples: `2560`
- Optimizer steps: `20`
- Skipped batches: `0`
- Loss: `5.877542` → `5.074811`
- Recall@100: `0.020326`
- NDCG@20: `0.003371`
- Coverage@100: `0.060331`

## Timing and memory

- Modal initialization/image build: `12.444 s`
- Local input preparation/upload: `2.524 s`
- Container startup: `4.193 s`
- Runner data preparation: `77.443 s`
- Estimated training compute: `32.840 s`
- Seconds/optimizer step: `1.642007`
- Checkpoint save/load: `0.039 / 0.035 s`
- Exact Retrieval: `1.255 s`
- Remote function wall: `118.270 s`
- Raw 6,729-step linear ETA: `184.15 min`
- Peak CUDA allocated/reserved: `165.24 / 234.00 MiB`
- Peak RSS: `5115.65 MiB`

Raw linear training-only estimate for 6,729 optimizer steps; excludes full-data preparation and fixed validation overhead.

CPU/GPU values are not required to be bitwise identical, and no parameter was changed based on their differences.

```text
formal_gate_executed=false
effectiveness_claim=false
full_big_train=false
full_big_validation=false
```

Small Matrix, temporal final, FAISS and Hybrid were not run.
