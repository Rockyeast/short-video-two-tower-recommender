# Fully-Observed Retrieval Protocol V1

## Status and claim boundary

This document freezes the primary KuaiRec retrieval route before full model
experiments. We follow KuaiRec's intended sparse Big-Matrix training and nearly
fully-observed Small-Matrix evaluation design, then define our own fixed Top-K
protocol for Popularity, BPR-MF and a content-aware Two-Tower. There is no
official KuaiRec Two-Tower leaderboard or mandatory model/metric recipe.

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

Small Matrix is run once only after model type, hyperparameters and any hybrid
weight are frozen on Big validation. For user `u`:

```text
candidates(u) = physically observed Small pairs intersect NORMAL items
relevant(u)   = candidates(u) where watch_ratio > 2.0
```

Unobserved/blocked pairs are unavailable, not negative. Small does not update
model parameters or Big history and is not called a future-time test. Before
opening it, report user/item overlap with Big train+validation, content-feature
coverage, data-cold items and users without Big history. Coverage audit numbers
must not be used for model selection.

## Metrics and segmentation

All models share the same queries, candidates, stable item-ID tie-break and
evaluator:

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

Two-Tower V1 uses item ID, category, frozen/precomputed caption vectors and
static features in the item tower. The user tower combines user ID with a
masked weighted mean of up to 50 train-history item representations. The loss
is temperature-scaled in-batch softmax over dot products; embeddings are
128-dimensional and L2-normalized. Exact matrix-multiplication retrieval comes
before any FAISS experiment.

## Gate and prohibition

Two-Tower may enter sealed Small evaluation if it beats BPR Recall@100, is
within two absolute percentage points with clearly higher Coverage@100,
improves a sufficiently supported data-cold slice, or forms a clear hybrid
Recall/Coverage Pareto improvement. Small results never trigger a protocol or
hyperparameter change.

Numbers from this protocol must never be compared in one table with legacy
causal temporal numbers. V1 excludes ItemCF grids, sequence/attention models,
FAISS, reranking, serving, receipts, watchdogs and transaction machinery.
