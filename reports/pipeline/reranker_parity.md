# Optional Reranker Pipeline Parity

- Queries: `32`
- Offline/online feature max abs: `0.0`
- Manual/Pipeline rankings exact: `true`
- Pipeline/HTTP rankings exact: `true`
- Preserved Top-100 sets: `32`
- HTTP reranker enabled P50/P95: `26.277/43.064 ms`
- HTTP reranker disabled P50/P95: `24.462/32.893 ms`

Real serving artifacts were used with deterministic synthetic request histories. This validates feature and orchestration parity, not recommendation effectiveness on final-refit routes.
