# Phase B3B sealed Small Attempt 4 failure

Attempt 4 was the single recovery run for the Attempt 3 report-serialization
failure. It failed and was not retried.

- Modal run:
  `https://modal.com/apps/lzf1416082617/main/ap-dzkLRVZFq4JnZoXGqcxB8C`
- Runner commit:
  `cc93f2487ce8f3e6e44575b89feb755c0bbb266a`
- Wrapper pin commit:
  `97d8c42be66e95b7d842e723a0d20ce2446bc721`
- Failure stage:
  `two_tower_checkpoint_numeric_preprocessing_identity_validation`
- Exception: `RuntimeError`
- Message:
  `Final-refit reconstructed numeric_preprocessing_sha256 mismatch`

The frozen Small and all three artifact file identities passed. Small was read,
queries were constructed, and Global Popularity and BPR rankings were computed
internally. The strict reconstructed feature identity check then failed before
Two-Tower ranking, Hybrid construction, or formal metric computation.

No metric or ranking value was returned, written, or observed. There was no
automatic retry, post-failure Small/model inspection, code change, retraining,
tuning, or temporal-final access. Further action requires independent review.
