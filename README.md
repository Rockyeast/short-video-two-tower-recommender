# Short-Video Recommendation on KuaiRec

This repository is currently limited to **Phase 0 / protocol-v2.1.1: data,
candidate-catalog, and evaluation audit**.

No Two-Tower model, ranking model, FAISS index, serving API, or online-feedback
component is implemented at this stage. The five baselines are specified but
remain unimplemented and unexecuted until protocol-v2.1.1 is reviewed.

## Locked label

The primary strong-positive label follows the KuaiRec documentation example:

```text
watch_ratio > 2.0
```

The threshold must not be changed in response to holdout results.

## Phase 0 outputs

The committed protocol-v2.1.1 audit bundle contains:

- field and missing-value inventory;
- metadata coverage and time ranges;
- label and user-history distributions;
- immutable split manifest;
- temporal and fully-observed evaluation contracts;
- baseline scale and compute estimates.

Key aggregate findings (not model metrics):

- the original temporal-final eligible-row gap is exactly reconciled as
  `99,637 raw rows - 15,910 exact duplicate extras - 26 same-key nonexact
  extras = 83,701 canonical keys`;
- the primary catalog contains only causally visible `NORMAL` videos; all
  10,728 item records have consistent per-video `upload_dt` and `video_type`,
  and the 29 `AD` videos are excluded;
- every one of the 497,117 train, 99,248 validation, and 83,661 temporal-final
  formal targets remains inside its query-time available/unseen candidate set;
- protocol-v2.1.1 preserves the already frozen and disclosed protocol-v2 time
  cutoffs, then assigns canonical events to them. Histories, seen filters,
  last-50 sequences, quick-skip pools, and popularity statistics use one event
  per `(user_id, video_id, timestamp)`, never duplicated raw rows. The audit
  separately reports how recomputing cutoffs after deduplication would change
  membership; it does not redefine the holdout.

## Fit and evaluation contexts

| context | fit | select/evaluate | cold-item reference |
|---|---|---|---|
| selection | train | select on validation | train |
| final | refit from scratch on train + validation | temporal final exactly once | train + validation |
| Small Matrix audit | reuse the frozen final-fit artifact | locked static audit | train + validation |

Validation events may update a user's causal runtime history only after the
current query is scored; they never update learned parameters. The same rule
applies to prior temporal-final events during the one-time final replay.

The temporal final split is **not claimed to be untouched**. Phase 0 has already
published aggregate label, query, and data-quality statistics for that split.
It is therefore described as a **holdout frozen after the Phase 0 aggregate
audit**. It may only be evaluated once, through a separate explicit entrypoint,
after the method, feature schema, hyperparameters, and seeds are frozen and the
model is refit from scratch on train plus validation.

## Candidate and target protocol

The primary temporal catalog contains only videos that are:

- `NORMAL`, not `AD`;
- uploaded early enough under the conservative date-only availability rule;
- `public` in the latest daily snapshot strictly before the query's local date;
- unseen by that user strictly before the query timestamp.

`private` and `only friends` videos are excluded because global permission to
show them cannot be established from the dataset. Conflicting per-video
`upload_dt` or `video_type` values are excluded and reported rather than silently
resolved.

Raw positive rows are not training targets. Rows first pass deterministic
deduplication by `(user_id, video_id, timestamp)`. Exact duplicates are
coalesced, a key containing both positive and nonpositive labels is excluded,
and the final canonical target table is hashed. Different eligible videos at
the same user timestamp remain one atomic multi-target query.

## Small Matrix: primary quality audit and secondary safety audit

The [official KuaiRec documentation](https://github.com/chongminggao/KuaiRec)
explains that the 0.4% missing Small Matrix pairs arise because users blocked
videos or their authors. Protocol-v2.1.1 therefore uses deliberately separate
evaluations:

1. **Primary quality audit:** remove each user's blocked/missing pairs, rank only
   physically observed `NORMAL` pairs, and treat `watch_ratio > 2.0` as
   relevant. This matches the temporal task's `NORMAL`-only catalog.
2. **Secondary safety audit:** rank all 3,327 catalog videos and report
   `Blocked@K`, the fraction of Top-K results belonging to that user's inferred
   blocked set, plus the fraction of users receiving at least one blocked item.
3. **AD diagnostic:** report quality on physically observed `AD` pairs
   separately; never pool it into the primary quality metrics.

Blocked/missing information is an evaluation-time availability mask only. It
must never enter training, user history, feature construction, hyperparameter
selection, or negative sampling.

## Locked baseline scope

Random, Global Popularity, Causal Time-Decayed Popularity, ItemCF, and BPR are
specified but not yet implemented or run. Time-decayed popularity must compare
fit-frozen and causal-streaming variants on validation; the stronger registered
variant becomes the baseline in the frozen final bundle. BPR uses the causal
candidate catalog at each positive event time and falls back to the selected
time-decayed popularity policy for users absent from fit data.

Planning scale uses `497,117` canonical train targets. Ten BPR epochs therefore
represent approximately `4,971,170` positive updates before batching. These are
cost estimates, not executed results.

## Active protocol-v2.1.1 contracts

- `contracts/event_canonicalization_v1.yaml`
- `contracts/temporal_evaluation_v2.yaml`
- `contracts/fully_observed_audit_v2.yaml`
- `contracts/fit_contexts_v1.yaml`
- `contracts/candidate_catalog_v1.yaml`
- `contracts/target_deduplication_v1.yaml`
- `contracts/two_tower_cold_start_v2.yaml`
- `contracts/metrics_v1.yaml`
- `contracts/baselines_v1.yaml`
- `contracts/negative_sampling_v2.yaml`

The original `temporal_evaluation_v1.yaml`, `fully_observed_audit_v1.yaml`,
`two_tower_cold_start_v1.yaml`, and `negative_sampling_v1.yaml` files from
commit `97a0f52` are retained only as inactive historical records.
`configs/phase0.yaml` is the authoritative list of active contracts.

The temporal final holdout and the Small Matrix audit are locked by default.
Ordinary baseline entrypoints may access only train and validation.

## Reproduce Phase 0

The raw archive is the official [KuaiRec 2.0 Zenodo artifact](https://zenodo.org/records/18164998).
Its expected MD5 is `261550d472c48eff4990fb13c0e5bcf7`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m zipfile -e data/raw/KuaiRec.zip data/raw
.venv/bin/python scripts/audit_phase0.py --mode verify
.venv/bin/pytest -q
```

Verify mode recomputes the Phase 0 bundle in a temporary directory and compares
it with committed outputs. Generate mode refuses to overwrite an existing
report, split manifest, or holdout lock. An intentional protocol revision must
preserve the old bundle and be reviewed before regenerating it.

Generated artifacts:

- `reports/phase0/audit.md`: human-readable audit;
- `reports/phase0/audit.json`: machine-readable audit;
- `manifests/split_manifest.json`: read-only data/split/config/contract record;
- `manifests/FINAL_HOLDOUT_LOCKED.json`: read-only final-evaluation guard;
- `contracts/*.yaml`: fit-context, candidate, target, metrics, baseline,
  negative-sampling, temporal, Small Matrix, and cold-item contracts.
