# Phase B3B sealed Small attempt-2 failure record

Repair attempt 2 stopped after the sealed Small CSV had been identity verified,
read, and converted into queries. It was not retried.

- Modal run:
  `https://modal.com/apps/lzf1416082617/main/ap-i57dfvaXBDBdnkg5bo1ija`
- Runner commit: `518131548478182de285ec8e48b6e3b57330a83e`
- Wrapper commit:
  `98da5c1581bae55b244f56a513d39ac9ca1be213`
- Prior attempt metrics produced: `false`
- Prior failure stage: `small_schema_validation`
- Attempt-2 failure stage:
  `two_tower_checkpoint_feature_vocab_validation`
- Failure function: `load_checkpoint()`
- Exception: `RuntimeError`
- Message: `Checkpoint feature vocabulary dimensions differ`

The frozen Small size/SHA and all three final-refit artifact SHA checks passed.
Small query construction completed. Global Popularity and BPR route rankings
were computed internally, but Two-Tower loading failed before its ranking;
Hybrid was therefore not constructed. No formal metric table or final report
was produced or observed, and no partial rankings or metrics were exposed.

Following the sealed-run rule, there was no retry, checkpoint inspection, Small
inspection, code change, model/rule change, or temporal-final access after the
failure. Further action requires independent review.
