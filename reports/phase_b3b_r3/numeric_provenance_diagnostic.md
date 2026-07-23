# Phase B3B-R3 Numeric Preprocessing Provenance

Three independent single-threaded Big-only processes rebuilt both
preprocessing paths. No Small input was mounted or accessed.

- Training and sealed paths were identical within every process: `True`
- Cross-process preprocessing payloads were bit-identical: `False`
- Checkpoint expected SHA: `71217e8965b59915874ab879e157eacd61a1c55813604bcdb9e748e08458c489`
- Per-process training SHAs: `['ec0fa898fe79632ae5fa17496f117c83134dbf1eb283b9de526ef05647e81c98', 'ec0fa898fe79632ae5fa17496f117c83134dbf1eb283b9de526ef05647e81c98', '71217e8965b59915874ab879e157eacd61a1c55813604bcdb9e748e08458c489']`
- Per-process sealed SHAs: `['ec0fa898fe79632ae5fa17496f117c83134dbf1eb283b9de526ef05647e81c98', 'ec0fa898fe79632ae5fa17496f117c83134dbf1eb283b9de526ef05647e81c98', '71217e8965b59915874ab879e157eacd61a1c55813604bcdb9e748e08458c489']`
- Process 2 reproduced the checkpoint SHA exactly.

## Membership

| Path | Observed | Observed NORMAL | Model universe |
|---|---:|---:|---:|
| training | 9391 | 9365 | 10725 |
| sealed | 9391 | 9365 | 10725 |

All three processes produced the same membership hashes:

- Observed items:
  `2c972e1fb50abe5077f24c41310421ce7d0b5568dd1cd46ba22df9840e814b94`
- Observed NORMAL items:
  `c13658cdbafe3ff0a4ee778618c2ff94f43d213894b5e7f46b75982b298f8758`
- Model item universe:
  `b9c31f8b2994c9ead960da977fd4e0cc357918cfc491121225689a79c33fa577`

## Field-level diagnosis

Medians, means, missing-value counts, vocabularies, memberships, and the
remaining three standard deviations were identical. The only difference was
the lowest bit of the first standard deviation:

| Processes | Decimal | `float.hex()` |
|---|---:|---|
| 0 and 1 | 0.5884286330248872 | `0x1.2d46848dbe669p-1` |
| 2 / checkpoint match | 0.5884286330248873 | `0x1.2d46848dbe66ap-1` |

Therefore the mismatch is not caused by a different data path or item
membership. It is a one-ULP cross-process floating-reduction difference in
`stds[0]`. The exact checkpoint-matching payload from process 2 is frozen in
`manifests/phase_b3b_final_numeric_preprocessing.json`.
