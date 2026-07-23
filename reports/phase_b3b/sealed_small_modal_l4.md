# Phase B3B Sealed Small Matrix Evaluation

This is the single sealed, nearly-fully-observed audit. It is not a future-time test. Small was not used for model selection, fitting, history construction, or route parameters.

- Sealed attempt number: `5`
- Attempt 1 failed at `small_schema_validation`: [sealed_small_failure.md](sealed_small_failure.md)
- Attempt 2 failed at `two_tower_checkpoint_feature_vocab_validation`: [sealed_small_attempt2_failure.md](sealed_small_attempt2_failure.md)
- Attempt 3 failed at `formal_report_serialization_audit_counts`: [sealed_small_attempt3_failure.md](sealed_small_attempt3_failure.md)
- Attempt 4 failed at `two_tower_checkpoint_numeric_preprocessing_identity_validation`: [sealed_small_attempt4_failure.md](sealed_small_attempt4_failure.md)
- Attempt 3 computed formal metrics in remote memory, but they were not returned, written, or observed.
- Attempts 1, 2, and 4 did not compute formal metrics.
- Attempts 1 through 4 exposed no formal metrics.
- No model, rule, or parameter was changed based on Small.

## Audit population

- Observed pairs: `4676570`
- Observed NORMAL pairs: `4676570`
- NORMAL candidate items: `3327`
- Evaluable queries: `1411`
- Excluded zero-relevant users: `0`
- Warm / cold users: `1411 / 0`
- Targets: `217175`
- Data-cold items: `1334`

## Metrics

| Method | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.005842 | 0.015039 | 0.030570 | 0.045656 | 1.000000 | 0.045455 |
| global_popularity | 0.140601 | 0.212629 | 0.268417 | 0.514947 | 0.032161 | 0.000000 |
| bpr | 0.173284 | 0.251986 | 0.319850 | 0.569454 | 0.403366 | 0.000000 |
| two_tower | 0.034061 | 0.058720 | 0.089741 | 0.159165 | 0.176135 | 0.295455 |
| hybrid_alpha_0.75 | 0.039513 | 0.068028 | 0.103950 | 0.206173 | 0.178840 | 0.250000 |

## Denominators and fallback

- Warm target denominator: `217175`
- Data-cold target denominator: `88`
- Cold-user query / target denominator: `0 / 0`
- Every cold-user route uses the same frozen refit Global Popularity fallback.

Cold-user metrics by method:

| Method | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | Coverage@100 | Data-Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| random | n/a | n/a | n/a | n/a | n/a | n/a |
| global_popularity | n/a | n/a | n/a | n/a | n/a | n/a |
| bpr | n/a | n/a | n/a | n/a | n/a | n/a |
| two_tower | n/a | n/a | n/a | n/a | n/a | n/a |
| hybrid_alpha_0.75 | n/a | n/a | n/a | n/a | n/a | n/a |

## Runtime and claim boundary

- GPU: `NVIDIA L4`
- Wall time: `285.646 s`
- Peak RSS: `8324.69 MiB`
- Peak CUDA allocated / reserved: `57.00 / 66.00 MiB`
- No statistical-significance or cross-seed claim is made.
- The result is reported as observed; no rerun, retuning, or post-Small model selection is permitted.
- Temporal final was not accessed.
