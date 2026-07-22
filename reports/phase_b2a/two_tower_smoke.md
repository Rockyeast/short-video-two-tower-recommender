# Phase B2A PyTorch Two-Tower bounded smoke

This is an engineering smoke, not a formal effectiveness experiment. Small
Matrix and temporal final were not accessed; full-data Two-Tower training was
not run.

- Caption model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Resolved revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Runtime: PyTorch `2.11.0+cu130`, sentence-transformers `5.6.0`, CPU
- Caption coverage: `100.0000%`
- Caption cache: `9388 x 384` float32; SHA256
  `b4093393c59ec00e9ab1e9cb90467404aea56164564bb379e07f4312f4e9e6fa`
- Training sample: `256` users,
  `7436` examples and
  `5941` items
- Fixed diagnostic loss: `6.146019`
  -> `4.587346`
- Fixed diagnostic Top-1: `0.003906` -> `0.054688`
- Mean positive logit: `-0.188858` -> `3.111732`
- Mean valid-negative logit: `-0.160014` -> `1.307620`
- Optimizer: `90` steps, `0` skipped batches
- False-negative mask: mean off-diagonal masked `0.072800`, median valid
  negatives `238`, duplicate-target rate `0.090191`
- Mean gradient norms: item ID `0.035864`, category `0.019916`, caption
  projection `0.027572`, static projection `0.049168`, upload type `0.013380`,
  user ID `0.045307`, history path `0.017497`
- Retrieval smoke: `128` queries, `1067` targets, `4096` sampled NORMAL items
- Sampled Recall@100: `0.147449`
- Sampled NDCG@20: `0.014331`; Coverage@100: `0.177490`
- Wall time: caption encoding `118.5568 s`; smoke `75.0842 s` total
- Peak resources: RSS `1714.07 MB`; GPU memory `0 MB`

Required interpretation flags:

```text
sampled_catalog_smoke = true
comparable_to_b1a = false
effectiveness_claim = false
formal_gate_executed = false
```

All paths in the JSON report are stable logical locators. See the JSON for
gradient, false-negative, cache, timing and resource details.
