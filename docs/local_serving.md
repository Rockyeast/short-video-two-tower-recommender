# Local Recommendation Serving

This is a local inference path over the frozen final-refit artifacts. It does
not retrain models or read the Small Matrix or temporal final.

## What is loaded

`scripts/export_serving_bundle.py` verifies the frozen artifact identities and
writes one ignored local NPZ bundle containing:

- the 10,699-item serving catalog and frozen Popularity scores;
- BPR user/item factors;
- pre-encoded Two-Tower item vectors;
- trained user-ID embeddings and user-MLP parameters.

The exporter checks the NumPy dynamic user tower against the original PyTorch
user tower before publishing the bundle. The service then loads and verifies
that bundle once during application startup.

## Request path

```text
request user + last-50 history + optional history weights
                         |
                 filter seen items
                         |
       +-----------------+-----------------+
       |                                   |
warm in Two-Tower and BPR            unknown/cold user
       |                                   |
dynamic user tower + BPR              Popularity fallback
       |                                   |
frozen weighted RRF                         |
       +-----------------+-----------------+
                         |
                       Top-K
```

The current batch endpoint executes the same immutable engine once per request
in input order. It is an API convenience, not a vectorized throughput claim.

## Start the API

Install the serving dependencies:

```bash
.venv/bin/pip install -e '.[serving]'
```

After exporting `artifacts/serving/serving_bundle_v1.npz` and its JSON metadata:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/serve_recommendation.py \
  --bundle artifacts/serving/serving_bundle_v1.npz \
  --metadata artifacts/serving/serving_bundle_v1.json \
  --port 8000
```

Single recommendation:

```bash
curl -X POST http://127.0.0.1:8000/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 0,
    "history": [0, 53],
    "history_weights": [1.0, 0.1],
    "top_k": 10
  }'
```

Batch recommendation:

```bash
curl -X POST http://127.0.0.1:8000/v1/recommend/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "requests": [
      {"user_id": 0, "history": [0], "top_k": 10},
      {"user_id": 999999, "history": [0], "top_k": 10}
    ]
  }'
```

Readiness and loaded bundle identity:

```bash
curl http://127.0.0.1:8000/healthz
```

The API has no authentication, online feature store, feedback ingestion,
autoscaling, production monitoring or concurrent-load benchmark.
