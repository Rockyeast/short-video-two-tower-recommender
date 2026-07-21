# Short-Video Recommendation on KuaiRec

This repository is currently limited to **Phase 0: data and evaluation audit**.

No Two-Tower model, ranking model, FAISS index, serving API, or online-feedback
component is implemented at this stage. Baselines remain blocked until the
Phase 0 report and evaluation contracts are reviewed.

## Locked label

The primary strong-positive label follows the KuaiRec documentation example:

```text
watch_ratio > 2.0
```

The threshold must not be changed in response to holdout results.

## Phase 0 outputs

After the official data is downloaded, the audit will produce:

- field and missing-value inventory;
- metadata coverage and time ranges;
- label and user-history distributions;
- immutable split manifest;
- temporal and fully-observed evaluation contracts;
- baseline scale and compute estimates.

The temporal final holdout and the Small Matrix audit are locked by default.

## Reproduce Phase 0

The raw archive is the official [KuaiRec 2.0 Zenodo artifact](https://zenodo.org/records/18164998).
Its expected MD5 is `261550d472c48eff4990fb13c0e5bcf7`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m zipfile -e data/raw/KuaiRec.zip data/raw
.venv/bin/python scripts/audit_phase0.py
.venv/bin/pytest -q
```

The audit command refuses to overwrite an existing report, split manifest, or
holdout lock. An intentional protocol revision must preserve the old bundle and
be reviewed before regenerating it.

Generated artifacts:

- `reports/phase0/audit.md`: human-readable audit;
- `reports/phase0/audit.json`: machine-readable audit;
- `manifests/split_manifest.json`: read-only data/split/config/contract record;
- `manifests/FINAL_HOLDOUT_LOCKED.json`: read-only final-evaluation guard;
- `contracts/*.yaml`: temporal, fully-observed, negative-sampling, and cold-item contracts.
