# Legacy Protocol-v2.1.1 Record

This document preserves the temporal protocol history previously embedded in
the main README. It is not the active evaluation contract for the primary
fully-observed route.

The active project narrative is in [`../README.md`](../README.md), and the
current evaluation contract is
[`fully_observed_protocol_v1.md`](fully_observed_protocol_v1.md).

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

## Legacy optional temporal-validation baselines

All methods use the same 99,248 validation queries and targets, causal candidate
membership, seen filtering, deterministic tie-breaks, and registered metrics.
The table below reports the selected configuration for each family, plus the
best fit-frozen time-decayed-popularity configuration needed for an
information-condition comparison.

| Information condition | Method / selected configuration | Recall@100 | NDCG@20 | Coverage@100 |
|---|---|---:|---:|---:|
| frozen/static | Random | 0.015287 | 0.001105 | 0.999033 |
| frozen/static | Global Popularity | 0.046721 | 0.004532 | 0.093353 |
| frozen/static | 1-day fit-frozen Time-Decayed Popularity | 0.089987 | 0.011351 | 0.087954 |
| frozen/static | ItemCF (`neighbors=200`, `shrinkage=0`) | 0.067105 | 0.008653 | 0.365201 |
| frozen/static | BPR-MF (`dim=64`, `epoch=20`, `lr=0.001`, `L2=0.0001`) | **0.096103** | 0.011670 | 0.285982 |
| online causal | 1-day causal-streaming Time-Decayed Popularity | **0.462951** | **0.110571** | 0.142616 |

BPR-MF is currently the strongest frozen personalized baseline. The much larger
streaming-popularity number is a different information condition: all queries
at a timestamp are scored first, then that timestamp's canonical strong
positives update popularity for later queries only. It is causal, but it is not
a train-frozen comparison.

ERRATUM-001 corrected the Phase 1 segment-membership implementation. Under the
active contract, an item is `Warm` if it has any canonical Big Matrix
interaction in the train reference window, independent of label, and `Cold` if
it has none. Of the 99,248 validation queries/targets, 72,115 are Warm and
27,133 (27.3%) are Cold; 48,008 target Tail items, a subgroup of the data-warm
catalog. `Cold` still does not mean strict query-time zero-shot: an item may
already have received an earlier validation positive before a later query.

The corrected Recall@100 comparison shows that 1-day causal-streaming
Time-Decayed Popularity leads BPR-MF in every reported segment, not only Cold:

| Method | Warm Recall@100 | Tail Recall@100 | Cold Recall@100 |
|---|---:|---:|---:|
| 1-day causal-streaming Time-Decayed Popularity | **0.448048** | **0.512435** | **0.502561** |
| BPR-MF | 0.132039 | 0.036640 | 0.000590 |

This remains a comparison across different information conditions: streaming
popularity causally absorbs validation feedback after prediction, whereas BPR
is train-frozen. See the committed
[ERRATUM-001 report](../reports/phase1/ERRATUM-001.md) for the corrected
membership, artifact lineage, and invariants.

The original Phase 2 plan used two separate gates:

1. compare a pure Two-Tower against BPR-MF and the other frozen baselines under
   the frozen/static information condition;
2. compare a Two-Tower + Causal Popularity hybrid against the strongest online
   causal baseline.

The formal Phase 1 results, selected configurations, and receipt remain under
`reports/phase1/` and `receipts/`; the explanatory summary is
[`../reports/phase1/interpretation.md`](../reports/phase1/interpretation.md),
and the segment correction is
[`../reports/phase1/ERRATUM-001.md`](../reports/phase1/ERRATUM-001.md).

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

The temporal final holdout remains locked by default. Ordinary baseline
entrypoints may access only train and validation.

## Reproduce Phase 0

The raw archive is the official
[KuaiRec 2.0 Zenodo artifact](https://zenodo.org/records/18164998).
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
