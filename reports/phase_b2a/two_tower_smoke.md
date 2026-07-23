# Phase B2A PyTorch Two-Tower bounded smoke

This is an engineering smoke, not a formal effectiveness experiment. Small
Matrix and temporal final were not accessed; full-data Two-Tower training was
not run.

- Caption model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Resolved revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Runtime: PyTorch `2.11.0+cu130`,
  sentence-transformers `5.6.0`,
  `cpu`
- Caption coverage: `100.0000%`
- Caption cache: `9388 x 384`
  `float32`; SHA256
  `b4093393c59ec00e9ab1e9cb90467404aea56164564bb379e07f4312f4e9e6fa`
- Training sample: `256` users,
  `7436` examples and
  `5941` items
- Fixed diagnostic loss: `6.146019`
  -> `4.587346`
- Fixed diagnostic Top-1: `0.003906`
  -> `0.054688`
- Mean positive logit: `-0.188858`
  -> `3.111732`
- Mean valid-negative logit: `-0.160014`
  -> `1.307620`
- Optimizer: `90` steps,
  `0` skipped batches
- Retrieval smoke: `128` queries,
  `1067` targets,
  `4096` sampled NORMAL items
- Sampled Recall@100: `0.147449`
- Sampled NDCG@20: `0.014331`;
  Coverage@100: `0.177490`
- Smoke wall time: `83.4175 s`; peak RSS
  `1700.48 MB`; GPU memory `0.00 MB`
- Code commit at run: `7f2cf9ccc9738c78c284eb77f911c4ea5ade2d49`
- Input tree clean at start:
  `true`
- Checkpoint identity schema: `2`;
  SHA256 `08a68921796ed7fcecbbaff8cd0bbdddf22609844ec46c3ccc1e058c7e3f471a`
- NORMAL membership: `10699` items;
  SHA256 `631a7c7cc93413f250f36f548feb720f8322050010e291afcc88338155f52c8e`
- Fixed retrieval catalog: `9365`
  items; SHA256 `8b8e88e2455a27dc0fac79e7bdb2733dc43096bb6d637e549d1ce5853e8ce55b`
- Model item universe: `9388` items;
  SHA256 `36b4f349b5f290c5117d8c19c2ce1cbb38f5a9431272060f46a875fb3aec5d9b`
- Evaluation encoding: item batch `1024`, user batch
  `128`, one inference-mode item-universe pass plus
  precomputed history gathers

Raw input identity:

| Logical source | Actual SHA256 | Expected SHA256 | Match |
|---|---|---|---|
| `KUAIREC_DATA_DIR/big_matrix.csv` | `4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9` | `4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9` | true |
| `KUAIREC_DATA_DIR/item_daily_features.csv` | `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b` | `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b` | true |
| `KUAIREC_DATA_DIR/kuairec_caption_category.csv` | `08f6ec40059c5a7ecdcebf615d080d9cf8f59f9497f6d66e8d5ccabdc7cab2d9` | `08f6ec40059c5a7ecdcebf615d080d9cf8f59f9497f6d66e8d5ccabdc7cab2d9` | true |

Required interpretation flags:

```text
sampled_catalog_smoke = true
comparable_to_b1a = false
effectiveness_claim = false
formal_gate_executed = false
```

All paths in the JSON report are stable logical locators. See the JSON for
gradient, false-negative, cache, timing and resource details.
