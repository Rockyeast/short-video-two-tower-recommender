# Phase 1 Result Interpretation

This note explains the completed temporal-validation results. It does not
participate in model selection and does not alter the formal result JSON, final
method bundle, or Selection Receipt.

## Two information conditions

**Frozen/static baselines** fit global model state from train only. Validation
queries may use their causal per-user runtime history where the contract allows
it, but validation events do not update learned parameters or global item
statistics. Under this condition, BPR-MF is the strongest personalized
baseline at Recall@100 `0.096103`; 1-day fit-frozen time-decayed popularity
reaches `0.089987`.

**Online causal baselines** may update global state during validation, but only
after prediction. The 1-day causal-streaming popularity implementation first
scores every query sharing a timestamp, then absorbs that timestamp's canonical
strong-positive targets. Those updates affect only later timestamps. Its
Recall@100 is `0.462951`.

These numbers answer different questions and should not be presented as a
direct frozen-model leaderboard.

## Why streaming popularity is much higher

Validation contains 99,248 queries/targets, of which 60,271 (60.7%) are
train-cold. A rapidly decayed streaming counter can learn that an item has just
started receiving strong feedback and immediately surface it for later
queries. A frozen BPR item representation cannot learn from those validation
events. On warm targets their Recall@100 values are almost identical:
`0.245170` for causal popularity versus approximately `0.244093` for BPR. The
large overall gap is concentrated in the reported Cold group:
`0.603790` versus `0.000398`.

## What `Cold` means

`Cold` means **no canonical strong-positive target in train**. It does not mean
that the item has no information at query time. After an item's first
validation positive, causal streaming may use that event for later queries.
The current Cold metric is therefore not a strict zero-shot cold-start metric.

## Phase 2 implication

Phase 2 needs two separately reported comparisons:

1. Pure Two-Tower versus BPR-MF and other frozen baselines, using the same
   frozen/static information condition.
2. Two-Tower + Causal Popularity hybrid versus the strongest online causal
   baseline.

The objective is a causal hybrid retrieval system. Phase 1 does not establish
that a pure Two-Tower improves retrieval, and no such claim should be made
until its registered validation gate is run.

## Integrity boundary

This document is interpretation only. The 97/97 formal selection rows,
selected configurations, `validation_baselines.json`,
`final_method_bundle.json`, and Selection Receipt remain unchanged. Temporal
final and the Small Matrix audit remain unrun.
