# ERRATUM-001: Phase 1 segment membership

## Error and contract

The active cold-start contract defines a data-warm item as any video with at least one canonical Big Matrix interaction in the train reference window, independent of label. The original Phase 1 implementation instead used eligible strong-positive train-target counts. It therefore mislabeled data-warm-but-positive-untouched items as Cold.

- Original merge commit: `4fab970fe36685f0c23aef49ac713dc100570502`
- Fix code commit: `43a4fc6f9dc9a7aaa283ebdd4f84c4f9b50bcda7`
- Corrected run time: `8019.98` seconds
- Temporal final accessed: **no**
- Small Matrix accessed: **no**

## Scope of correction

Changed: Warm/Tail/Cold Recall, their denominators and bootstrap intervals, and segment membership counts/hashes.

Unchanged and fail-closed checked: candidate membership, every Top-K-derived overall Recall/NDCG/Coverage value, all 97 configurations/seeds, and every selected configuration.

## Membership

- data-warm items: `7896`
- data-cold items: `2832`
- head items: `1668`
- tail items: `6228`
- data-warm SHA256: `765a33ca4362ef7f008fa80ef57ee70707bea7b1331880d0858b8d5afbbec8b6`
- model-ID-touched membership: not inferred; Phase 2 must record actual optimizer updates.

## Artifact lineage

- Original selection result SHA256: `f3dbbba9de5552d8d6bb34ae0fbe58dc50b57726a180b61bd9d216f31927857f`
- Original final bundle SHA256: `c56c83bc96486fb87d4650be59321aeb3dfdf11421d181a50baf1f23448119ac`
- Original selection receipt SHA256: `7acbb6ea4dd9bd88374b479b6aa54f1c97779d027c527f328e9db0835842d57a`
- Corrected selection result SHA256: `fdb9e13385f6db34ece4d2664d151f331ef81c192c4a4a933a9eee5bf08c0e80`
- Corrected final bundle SHA256: `6d2015e3df6e4939b95e85c9b205dd481dde370995c9d44ecbc5e29dea2fbd19`
- Corrected selection receipt SHA256: `c3edc5bae6b6a25862e3d495a8f21c2ca5853bd981c7afd8e28ed8cb539f5eca`
- Archived old receipt: `receipts/e271c45d34651387926e7572e18b8605654fd7dc8720e1f5e9b34a40378eea91/superseded/f3dbbba9de5552d8d6bb34ae0fbe58dc50b57726a180b61bd9d216f31927857f/SELECTION_RECEIPT.json`

## Selected configurations

Original and corrected selected configurations are identical:

- `random`: `0d69fac02d8481c5`
- `global_popularity`: `1517f66a8c07cee0`
- `time_decayed_popularity`: `b0582b145cfe15e4`
- `itemcf`: `28dcfded8b6b922f`
- `bpr_mf`: `dd709f2d4ffcc4c6`
