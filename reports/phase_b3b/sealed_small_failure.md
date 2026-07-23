# Phase B3B sealed Small failure record

The one-time Modal L4 run stopped after the sealed Small CSV had been identity
verified and read. It was not retried.

- Modal run:
  `https://modal.com/apps/lzf1416082617/main/ap-sh85kZ2WFXmX5tY9J8wLhd`
- Runner commit: `178f8df631ffdaa9ff038eb8c9d357e604124cd2`
- Wrapper commit at launch: `551cdd9`
- Attempt number: `1`
- Failure stage: `small_schema_validation` in
  `build_small_observed_queries()`
- Exception: `ValueError`
- Message: `small_observed_events contains missing required values`

The raw Small identity and all three final-refit artifact identities passed
before the CSV read. The failure occurred before query construction, ranking,
or metric computation. No formal Small report was written.

Following the sealed-run rule, no automatic retry, post-failure Small scan,
code change, model change, or temporal-final access was performed. Further
action requires independent review.

The approved repair only removes the requirement for unused Small
timestamp/time/date and duration fields. It does not modify any model,
checkpoint, alpha/RRF parameter, observed-pair candidate rule,
`watch_ratio > 2.0` relevance rule, fallback, or metric.
