# Phase 1 Result Interpretation

This note explains the completed temporal-validation results after
[ERRATUM-001](ERRATUM-001.md). It does not participate in model selection or
change the registered objective, configurations, or overall ranking metrics.

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

## Corrected segment membership

ERRATUM-001 enforces the active cold-start contract:

- `Warm`: the item has at least one canonical Big Matrix interaction in the
  train reference window, independent of label;
- `Cold`: the item has no canonical Big Matrix interaction in that window;
- `Tail`: a low-frequency subgroup of the data-warm catalog.

The 99,248 validation queries/targets contain 72,115 Warm targets, 27,133 Cold
targets, and 48,008 Tail targets. Tail overlaps Warm rather than forming a third
partition. Cold is not strict query-time zero-shot: after an item's first
validation positive, causal streaming may use that event for later queries.

## Why streaming popularity is much higher

A one-day causal counter reacts to current validation-period demand after each
timestamp is scored. It can therefore surface newly active items and reinforce
short-lived trends for later queries. Frozen BPR cannot update its item
representations from those validation events.

The corrected Recall@100 values show a broad advantage for causal-streaming
Time-Decayed Popularity over BPR-MF:

| Segment | Causal-streaming popularity | BPR-MF |
|---|---:|---:|
| Overall | **0.462951** | 0.096103 |
| Warm | **0.448048** | 0.132039 |
| Tail | **0.512435** | 0.036640 |
| Cold | **0.502561** | 0.000590 |

The gap is therefore present across Warm, Tail, and Cold; it is not
concentrated only in Cold. This does not establish that popularity is a better
frozen personalized model, because the two methods operate under different
information conditions.

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

ERRATUM-001 corrected only Warm/Tail/Cold membership, denominators, segment
Recall, and their bootstrap intervals across all 97 rows. Overall
Recall/NDCG/Coverage values and every selected configuration are unchanged. The
formal result JSON, final method bundle, and Selection Receipt were updated
with new hashes and explicit lineage to their archived predecessors. Temporal
final and the Small Matrix audit remain unrun.
