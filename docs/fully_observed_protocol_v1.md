# Fully-Observed Retrieval Protocol V1

## Status and claim boundary

This document freezes the primary KuaiRec retrieval route before full model
experiments. Phase A/A.1 is a protocol and interface skeleton, not a completed
recommendation system and not evidence that a model is effective. We follow
KuaiRec's intended sparse Big-Matrix training and nearly fully-observed
Small-Matrix evaluation design, then define our own fixed Top-K protocol for
Popularity, BPR-MF and a content-aware Two-Tower. There is no official KuaiRec
Two-Tower leaderboard or mandatory model/metric recipe.

Sources:

- [KuaiRec official repository](https://github.com/chongminggao/KuaiRec)
- [KuaiRec CIKM 2022 paper](https://arxiv.org/abs/2202.10842)

The official repository reports 7,176 users, 10,728 items and 16.3% density in
the Big Matrix; and 1,411 users, 3,327 items and 99.6% density in the Small
Matrix. It explains that missing Small pairs correspond to videos/authors the
user blocked. Those missing pairs are never treated as negatives.

## Labels

- Strong positive: `watch_ratio > 2.0` (strict), following the official binary
  feedback example.
- Quick skip: `play_duration < min(3000 ms, video_duration)` (strict).
- Middle interaction: neither a declared positive nor a declared negative.

## Big Matrix fit and selection

The frozen protocol-v2.1.1 time boundaries are reused without opening temporal
final:

- existing train window: fit data;
- existing validation window: model selection;
- final 15% temporal holdout: remains sealed and is not part of this route.

Canonical event keys and existing ID mappings may be reused. Each validation
user contributes at most one query:

1. history is the user's last 50 canonical train-window interactions only;
2. relevant items are validation strong positives absent from train history;
3. the fixed catalog is the union of item IDs observed in canonical train or
   validation, intersected with `NORMAL` item metadata;
4. the query candidates are that catalog minus all train-seen items;
5. users with no unseen validation strong positive are reported and excluded
   from retrieval metrics.

Validation interaction identities define only the fixed item universe. Their
labels, per-item frequencies, order and user association never update a model,
history, candidate set or popularity score. There is no timestamp replay.

## Sealed nearly-fully-observed evaluation

Small Matrix is run once only after model type, hyperparameters, seed and any
hybrid weight are frozen on Big validation. The final model is then refit from
scratch on Big train + validation. Small user representations use only the
last 50 interactions from that Big train + validation refit context. Small
feedback never enters user history, training or feature construction. This
matches the KuaiRec paper: Big contains additional interactions for the Small
users/items while all Small user-item interactions are excluded from Big. For
user `u`:

```text
candidates(u) = physically observed Small pairs intersect NORMAL items
relevant(u)   = candidates(u) where watch_ratio > 2.0
```

Unobserved/blocked pairs are unavailable, not negative. Small does not update
model parameters or Big history and is not called a future-time test. Before
opening it, report user/item overlap with Big train+validation, content-feature
coverage, data-cold items and users without Big history. Coverage audit numbers
must not be used for model selection.

Users with no Big history are retained and reported as cold users. They use the
fit-context Global Popularity fallback for ranking and are reported separately;
they are not silently removed from candidate, target or denominator counts.

## Metrics and segmentation

All models share the same queries, candidates, stable item-ID tie-break and
evaluator. Warm-user metrics are primary. Cold-user metrics and denominators
are reported separately after the declared Popularity fallback:

- Recall@20, Recall@50, Recall@100;
- NDCG@20;
- Coverage@100 = unique recommended items / union of candidate items;
- descriptive Data-Cold Recall@100.

Data-cold means no canonical interaction of any label in the Big train window.
Its query/target denominator is always reported. It is descriptive and has no
win gate when the denominator is small. Metrics are query-macro; because there
is one query per user, this is also user-macro. V1 has no bootstrap CI.

## Methods and budget

- Random: one deterministic sanity seed.
- Global Popularity: train strong-positive counts only.
- BPR-MF: at most three configurations and three final seeds.
- Two-Tower: at most three configurations and three final seeds.
- Two-Tower + Popularity: considered only after the first four methods finish.

BPR positives are canonical `NORMAL` events with `watch_ratio > 2.0`. For each
positive and each epoch, one uniform negative is resampled from fit-observed
`NORMAL` videos after removing every strong-positive video known for that user
in the fit context. The random seed is fixed. Restricting the negative catalog
to fit-observed videos preserves the declared score-zero behavior for truly
data-cold items. Mini-batch SGD averages a sparse user/item row only across the
examples in that batch that address that row; it does not divide every sparse
row update by the full batch size.

Two-Tower V1 uses item ID, category, frozen/precomputed caption vectors and
static features in the item tower. Allowed model inputs are exactly item ID,
caption embedding, category IDs, duration, width, height, upload type and upload
date. Daily engagement aggregates such as show/play/like/follow counts are
forbidden. `video_type` and visibility may filter the catalog but are not model
features. Static fields sourced from the daily table use each video's earliest
available row; any later source correction is counted and reported rather than
silently selecting a future snapshot. The user tower combines user ID with a
masked weighted mean of up to 50 train-history item representations. The loss
is temperature-scaled in-batch softmax over dot products; embeddings are
128-dimensional and L2-normalized. Exact matrix-multiplication retrieval comes
before any FAISS experiment.

### Two-Tower training and cold-start contract

Only a strong-positive event at a user's first interaction with that video may
be a training target. If any earlier interaction with the video exists, the
entire later target example is skipped rather than fabricating an unseen event
by deleting the video from history. For each retained target at time `t`,
history contains at most 50 fit-context events with timestamp strictly less
than `t`; same-timestamp events cannot enter each other's histories. In-batch
logits mask repeated targets and every other known fit-context positive of that
row's user. Quick skips are downweighted history context only in V1, not
explicit negatives.

BPR assigns every candidate without a trained item factor a fixed score of
zero. Two-Tower zeros the ID-embedding contribution for an item whose ID
embedding was not trained, leaving category/caption/static content as the
cold-item path. Neither model may delete a cold positive target.

## Gate and prohibition

The comparison baseline is the stronger Big-validation Recall@100 result from
Global Popularity and BPR, never BPR by assumption. Two-Tower may enter sealed
Small evaluation only if its NDCG@20 is no more than 0.01 absolute below that
baseline and it satisfies one predeclared rule:

1. Recall@100 exceeds that strongest baseline by at least 0.002 absolute; or
2. Recall@100 is within 0.02 absolute and Coverage@100 improves by at least 0.05
   absolute; or
3. Recall@100 is within 0.02 absolute, the data-cold denominator is at least
   100 targets, and Data-Cold Recall@100 improves by at least 0.05 absolute; or
4. the frozen Two-Tower + Popularity hybrid is no worse on both Recall@100 and
   Coverage@100 and improves at least one by 0.01 absolute.

Small results never trigger a protocol or hyperparameter change.

Numbers from this protocol must never be compared in one table with legacy
causal temporal numbers. V1 excludes ItemCF grids, sequence/attention models,
FAISS, reranking, serving, receipts, watchdogs and transaction machinery.
