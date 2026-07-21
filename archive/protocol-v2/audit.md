# KuaiRec Phase 0 Audit

Generated: `2026-07-21T17:17:46.276639+00:00`

> No model or baseline was trained or evaluated in Phase 0.

## Locked label

```text
watch_ratio > 2.0
```

The threshold was not changed in response to these statistics.

## Interaction and label summary

| source/split | rows | users | videos | positives | positive rate | users with zero positives |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 8,771,564 | 7,174 | 7,896 | 658,907 | 7.5119% | 0.1673% |
| `validation` | 1,879,621 | 7,094 | 5,431 | 136,711 | 7.2733% | 3.7074% |
| `temporal_final` | 1,879,621 | 6,911 | 5,246 | 140,772 | 7.4894% | 2.7203% |
| `small_matrix_audit` | 4,676,570 | 1,411 | 3,327 | 217,175 | 4.6439% | 0.0000% |

### Time ranges

- Big Matrix (Asia/Shanghai): `2020-06-23T08:34:11.373000+08:00` to `2020-09-10T07:32:12.427000+08:00`
- Small Matrix (Asia/Shanghai): `2020-07-04T02:23:26.060000+08:00` to `2020-09-05T23:57:23.683000+08:00`

The raw `date` column is not used for splitting because it disagrees
with the localized timestamp on 15,530 Big Matrix rows.

### Watch-ratio buckets

| source | bucket | count |
|---|---|---:|
| `big_matrix` | `missing` | 0 |
| `big_matrix` | `watch_ratio < 0` | 0 |
| `big_matrix` | `0 <= watch_ratio < 0.25` | 2,607,852 |
| `big_matrix` | `0.25 <= watch_ratio < 0.5` | 1,891,163 |
| `big_matrix` | `0.5 <= watch_ratio < 1` | 3,793,563 |
| `big_matrix` | `1 <= watch_ratio <= 2` | 3,301,838 |
| `big_matrix` | `watch_ratio > 2` | 936,390 |
| `small_matrix` | `missing` | 0 |
| `small_matrix` | `watch_ratio < 0` | 0 |
| `small_matrix` | `0 <= watch_ratio < 0.25` | 594,959 |
| `small_matrix` | `0.25 <= watch_ratio < 0.5` | 680,138 |
| `small_matrix` | `0.5 <= watch_ratio < 1` | 1,886,423 |
| `small_matrix` | `1 <= watch_ratio <= 2` | 1,297,875 |
| `small_matrix` | `watch_ratio > 2` | 217,175 |

## Per-user distributions

- **train**: events p50/p90/p99 = 1276/2117/2921; positives p50/p90/p99 = 66/203/407.
- **validation**: events p50/p90/p99 = 230/531/818; positives p50/p90/p99 = 12/45/97.
- **temporal_final**: events p50/p90/p99 = 226/524/880; positives p50/p90/p99 = 14/44/97.
- **small_matrix_audit**: positives p50/p90/p99 = 95/376/671.

### History available at evaluation time

- `validation_history_from_train`: p50/p90/p99 = 1274/2116/2919; zero-history users = 0.0282%.
- `temporal_final_history_from_train_and_validation`: p50/p90/p99 = 1549/2508/3367; zero-history users = 0.0000%.
- `small_audit_history_from_big_first_85_percent`: p50/p90/p99 = 301/413/595; zero-history users = 0.0000%.

## Metadata coverage

| feature | covered videos | catalog videos | coverage |
|---|---:|---:|---:|
| `item_categories_feat` | 10,728 | 10,728 | 100.0000% |
| `caption` | 9,372 | 10,728 | 87.3602% |
| `manual_cover_text` | 10,727 | 10,728 | 99.9907% |
| `topic_tag` | 10,728 | 10,728 | 100.0000% |
| `first_level_category` | 10,728 | 10,728 | 100.0000% |
| `second_level_category` | 10,728 | 10,728 | 100.0000% |
| `third_level_category` | 10,728 | 10,728 | 100.0000% |
| `upload_time` | 10,728 | 10,728 | 100.0000% |

## New users and videos by temporal split

| split | new users | fraction | new videos | fraction |
|---|---:|---:|---:|---:|
| `validation` | 2 | 0.0282% | 1,495 | 27.5272% |
| `temporal_final` | 0 | 0.0000% | 1,337 | 25.4861% |

## Evaluation contracts

- Temporal: `contracts/temporal_evaluation_v2.yaml`
- Small Matrix: `contracts/fully_observed_audit_v2.yaml`
- Fit contexts: `contracts/fit_contexts_v1.yaml`
- Candidate catalog: `contracts/candidate_catalog_v1.yaml`
- Target deduplication: `contracts/target_deduplication_v1.yaml`
- Metrics: `contracts/metrics_v1.yaml`
- Baselines: `contracts/baselines_v1.yaml`
- Negative sampling: `contracts/negative_sampling_v2.yaml`
- Cold-item fallback: `contracts/two_tower_cold_start_v2.yaml`

The temporal final split is not claimed to be untouched. It is frozen
after this Phase 0 aggregate audit; no ranking metric was computed.

Equal-timestamp strong positives are one multi-target query with shared
history ending strictly before that timestamp.

A target must also be unseen before its query timestamp and certainly uploaded.
Because `upload_dt` has date precision only, an item becomes eligible at the
next Asia/Shanghai midnight; same-day events are excluded as unverifiable.
The following exclusion counts are event-level. Formal targets then use
the locked 8-field duplicate/conflict rule shown below.

| split | raw positives | eligible rows | canonical targets | before date | same-day unknown | no prior status | AD | not public | previously seen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 658,907 | 533,835 | 497,117 | 212 | 65,995 | 16,473 | 91 | 20 | 42,281 |
| `validation` | 136,711 | 99,274 | 99,248 | 1 | 16,154 | 16 | 8 | 6 | 21,252 |
| `temporal_final` | 140,772 | 99,566 | 83,661 | 3 | 23,557 | 25 | 13 | 40 | 17,568 |

### Pre-catalog duplicate anomaly reconciliation

This reproduces the original upload/unseen eligible-event stage before
the new NORMAL/public catalog filter, including the reported final gap.

| split | eligible positive rows | exact duplicate extras | same-key nonexact extras | binary-conflict keys | canonical keys |
|---|---:|---:|---:|---:|---:|
| `train` | 549,976 | 36,603 | 131 | 0 | 513,242 |
  - `train`: 5,119 affected users; extras/user p50/p90/p99/max=4/17/44/132; largest date `2020-08-05`=36,568.
| `validation` | 99,296 | 1 | 25 | 0 | 99,270 |
  - `validation`: 7 affected users; extras/user p50/p90/p99/max=1/9/18/19; largest date `2020-08-27`=22.
| `temporal_final` | 99,637 | 15,910 | 26 | 0 | 83,701 |
  - `temporal_final`: 4,161 affected users; extras/user p50/p90/p99/max=2/8/22/73; largest date `2020-08-31`=15,913.

### Formal catalog-eligible target reconciliation

| split | eligible positive rows | exact duplicate extras | same-key nonexact extras | binary-conflict keys | positives excluded by conflict | canonical targets |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 533,835 | 36,593 | 125 | 0 | 0 | 497,117 |
  - `train` duplicate/conflict extras affect 5,119 users; extras/user p50/p90/p99/max=4/17/44/132; largest date is `2020-08-05` with 36,558. Full user/date counts are in audit.json.
| `validation` | 99,274 | 1 | 25 | 0 | 0 | 99,248 |
  - `validation` duplicate/conflict extras affect 7 users; extras/user p50/p90/p99/max=1/9/18/19; largest date is `2020-08-27` with 22. Full user/date counts are in audit.json.
| `temporal_final` | 99,566 | 15,879 | 26 | 0 | 0 | 83,661 |
  - `temporal_final` duplicate/conflict extras affect 4,160 users; extras/user p50/p90/p99/max=2/8/22/73; largest date is `2020-08-31` with 15,882. Full user/date counts are in audit.json.

| split | temporal queries | multi-target queries | fraction | max targets |
|---|---:|---:|---:|---:|
| `train` | 497,117 | 0 | 0.000000% | 1 |
| `validation` | 99,248 | 0 | 0.000000% | 1 |
| `temporal_final` | 83,661 | 0 | 0.000000% | 1 |

### Small Matrix observation coverage

- Full ranking catalog: 3,327 videos for each of 1,411 users (4,694,397 scored pairs).
- Observed feedback pairs: 4,676,570 (99.6202%); blocked/missing pairs: 17,827.
- Missing pairs per user p50/p90/p99/max: 12/23/29/32.
- Officially, missing pairs represent videos/authors blocked by that user.
- Primary audit removes each user's blocked/missing pairs. Full 3,327-item ranking is secondary only and must report `Blocked@K` and user hit rate.
- Blocked information never enters training, history, features, negative sampling, or hyperparameter selection.

## Causal candidate catalog audit

- Primary type: `NORMAL`; excluded `AD` videos: 29.
- Visibility at query date D uses the latest snapshot with `date < D`; same-day or missing status is not treated as visible.
- Per-video inconsistent `upload_dt`: 0; inconsistent `video_type`: 0.
- Visible-status transitions: `{"only friends->private": 3, "only friends->public": 1, "private->only friends": 3, "private->public": 190, "public->only friends": 4, "public->private": 926}`.

### Per-query candidate-size distributions

| split | queries | available p50/p90/p99 | unseen p50/p90/p99 | uniform pool p50/p90/p99 | missing targets |
|---|---:|---:|---:|---:|---:|
| `train` | 497,117 | 5471/6718/7166 | 4828/5667/6379 | 4827/5666/6378 | 0 |
| `validation` | 99,248 | 8248/8633/8633 | 6713/7641/8270 | 6712/7640/8269 | 0 |
| `temporal_final` | 83,661 | 9198/9808/9808 | 7609/8664/9446 | 7608/8663/9445 | 0 |

### Train hard-negative pool audit

- Nonempty-pool query coverage: 40.4778% (201,222/497,117).
- Deduplicated hard `(query,item)` pairs: 686,008; pool p50/p90/p99: 0/4/15.
- Future positive labels are used only for aggregate false-negative risk diagnostics; they do not filter samples or tune the sampler.
- Pair-level risk fractions: `{"before_fit_cutoff": 0.009126715723431796, "remaining_session": 0.006564063392846731, "within_1d": 0.006687968653426782, "within_7d": 0.007081550069386946}`.

## Data quality findings

- Big Matrix raw `date` disagrees with localized `timestamp` on 15,530 rows (0.123935%); splitting uses timestamp.
- Small Matrix has 181,992 missing `time/date/timestamp` rows; it is therefore used only as a static audit.
- Caption CSV contains 4 bare carriage returns. LF-only record parsing preserves all 10,728 video rows.
- Non-empty caption coverage is 87.3602%; cold items must also fall back to category/topic content.
- Big Matrix is verified user-major and timestamp-monotonic within each user; this permits exact first-view target filtering without reordering equal timestamps.
- `upload_dt` has day precision only. Candidate availability is conservatively the following local midnight, so same-day targets with unverifiable upload times are excluded: train=65,995, validation=16,154, temporal_final=23,557.
- Strong positives timestamped before even the declared upload date are also excluded as metadata inconsistencies: train=212, validation=1, temporal_final=3.
- Small Matrix is 99.6202% observed, not literally complete; 17,827 pairs are treated as blocked/missing for the primary audit.

## Baseline scale and estimated cost

> Planning estimates only; no baseline was executed in Phase 0. Temporal query count uses exact (user_id, next-positive timestamp) groups.

| baseline | fit scale | evaluation scale | planning estimate |
|---|---|---|---|
| `random` | none | up to 1,064,732,544 candidate pairs | under 5 CPU minutes with direct seeded top-K sampling |
| `global_popularity` | one pass over 8,771,564 train interactions | one shared ranking plus per-user seen filtering | roughly 1-5 CPU minutes |
| `time_decayed_popularity` | one chronological pass over 8,771,564 interactions | state update plus shared ranking at query times | roughly 3-15 CPU minutes |
| `itemcf` | sparse co-occurrence from 658,907 strong positives over at most 10,728 videos | history-neighbor aggregation for every temporal query | roughly 10-60 CPU minutes and 1-4 GB working memory |
| `bpr_mf` | pre-registered 10 epochs = about 6,589,070 positive-pair updates before batching | up to 1,064,732,544 dot products | roughly 30-120 CPU minutes or 5-20 GPU minutes |

## Complete schema and missingness

### `big_matrix.csv`

Rows: 12,530,806

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `user_id` | `int64` | 0 | 0.000000% |
| `video_id` | `int64` | 0 | 0.000000% |
| `play_duration` | `int64` | 0 | 0.000000% |
| `video_duration` | `int64` | 0 | 0.000000% |
| `time` | `object` | 0 | 0.000000% |
| `date` | `int64` | 0 | 0.000000% |
| `timestamp` | `float64` | 0 | 0.000000% |
| `watch_ratio` | `float64` | 0 | 0.000000% |

### `small_matrix.csv`

Rows: 4,676,570

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `user_id` | `int64` | 0 | 0.000000% |
| `video_id` | `int64` | 0 | 0.000000% |
| `play_duration` | `int64` | 0 | 0.000000% |
| `video_duration` | `int64` | 0 | 0.000000% |
| `time` | `object` | 181,992 | 3.891570% |
| `date` | `float64` | 181,992 | 3.891570% |
| `timestamp` | `float64` | 181,992 | 3.891570% |
| `watch_ratio` | `float64` | 0 | 0.000000% |

### `item_categories.csv`

Rows: 10,728

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `video_id` | `int64` | 0 | 0.000000% |
| `feat` | `object` | 0 | 0.000000% |

### `item_daily_features.csv`

Rows: 343,341

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `video_id` | `int64` | 0 | 0.000000% |
| `date` | `int64` | 0 | 0.000000% |
| `author_id` | `int64` | 0 | 0.000000% |
| `video_type` | `object` | 0 | 0.000000% |
| `upload_dt` | `object` | 0 | 0.000000% |
| `upload_type` | `object` | 0 | 0.000000% |
| `visible_status` | `object` | 0 | 0.000000% |
| `video_duration` | `float64` | 10,598 | 3.086727% |
| `video_width` | `int64` | 0 | 0.000000% |
| `video_height` | `int64` | 0 | 0.000000% |
| `music_id` | `int64` | 0 | 0.000000% |
| `video_tag_id` | `int64` | 0 | 0.000000% |
| `video_tag_name` | `object` | 32,434 | 9.446585% |
| `show_cnt` | `int64` | 0 | 0.000000% |
| `show_user_num` | `int64` | 0 | 0.000000% |
| `play_cnt` | `int64` | 0 | 0.000000% |
| `play_user_num` | `int64` | 0 | 0.000000% |
| `play_duration` | `int64` | 0 | 0.000000% |
| `complete_play_cnt` | `int64` | 0 | 0.000000% |
| `complete_play_user_num` | `int64` | 0 | 0.000000% |
| `valid_play_cnt` | `int64` | 0 | 0.000000% |
| `valid_play_user_num` | `int64` | 0 | 0.000000% |
| `long_time_play_cnt` | `int64` | 0 | 0.000000% |
| `long_time_play_user_num` | `int64` | 0 | 0.000000% |
| `short_time_play_cnt` | `int64` | 0 | 0.000000% |
| `short_time_play_user_num` | `int64` | 0 | 0.000000% |
| `play_progress` | `float64` | 0 | 0.000000% |
| `comment_stay_duration` | `int64` | 0 | 0.000000% |
| `like_cnt` | `int64` | 0 | 0.000000% |
| `like_user_num` | `int64` | 0 | 0.000000% |
| `click_like_cnt` | `int64` | 0 | 0.000000% |
| `double_click_cnt` | `int64` | 0 | 0.000000% |
| `cancel_like_cnt` | `int64` | 0 | 0.000000% |
| `cancel_like_user_num` | `int64` | 0 | 0.000000% |
| `comment_cnt` | `int64` | 0 | 0.000000% |
| `comment_user_num` | `int64` | 0 | 0.000000% |
| `direct_comment_cnt` | `int64` | 0 | 0.000000% |
| `reply_comment_cnt` | `int64` | 0 | 0.000000% |
| `delete_comment_cnt` | `int64` | 0 | 0.000000% |
| `delete_comment_user_num` | `int64` | 0 | 0.000000% |
| `comment_like_cnt` | `int64` | 0 | 0.000000% |
| `comment_like_user_num` | `int64` | 0 | 0.000000% |
| `follow_cnt` | `int64` | 0 | 0.000000% |
| `follow_user_num` | `int64` | 0 | 0.000000% |
| `cancel_follow_cnt` | `int64` | 0 | 0.000000% |
| `cancel_follow_user_num` | `int64` | 0 | 0.000000% |
| `share_cnt` | `int64` | 0 | 0.000000% |
| `share_user_num` | `int64` | 0 | 0.000000% |
| `download_cnt` | `int64` | 0 | 0.000000% |
| `download_user_num` | `int64` | 0 | 0.000000% |
| `report_cnt` | `int64` | 0 | 0.000000% |
| `report_user_num` | `int64` | 0 | 0.000000% |
| `reduce_similar_cnt` | `int64` | 0 | 0.000000% |
| `reduce_similar_user_num` | `int64` | 0 | 0.000000% |
| `collect_cnt` | `float64` | 69,683 | 20.295566% |
| `collect_user_num` | `float64` | 69,683 | 20.295566% |
| `cancel_collect_cnt` | `float64` | 69,683 | 20.295566% |
| `cancel_collect_user_num` | `float64` | 69,683 | 20.295566% |

### `user_features.csv`

Rows: 7,176

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `user_id` | `int64` | 0 | 0.000000% |
| `user_active_degree` | `object` | 0 | 0.000000% |
| `is_lowactive_period` | `int64` | 0 | 0.000000% |
| `is_live_streamer` | `int64` | 0 | 0.000000% |
| `is_video_author` | `int64` | 0 | 0.000000% |
| `follow_user_num` | `int64` | 0 | 0.000000% |
| `follow_user_num_range` | `object` | 0 | 0.000000% |
| `fans_user_num` | `int64` | 0 | 0.000000% |
| `fans_user_num_range` | `object` | 0 | 0.000000% |
| `friend_user_num` | `int64` | 0 | 0.000000% |
| `friend_user_num_range` | `object` | 0 | 0.000000% |
| `register_days` | `int64` | 0 | 0.000000% |
| `register_days_range` | `object` | 0 | 0.000000% |
| `onehot_feat0` | `int64` | 0 | 0.000000% |
| `onehot_feat1` | `int64` | 0 | 0.000000% |
| `onehot_feat2` | `int64` | 0 | 0.000000% |
| `onehot_feat3` | `int64` | 0 | 0.000000% |
| `onehot_feat4` | `float64` | 201 | 2.801003% |
| `onehot_feat5` | `int64` | 0 | 0.000000% |
| `onehot_feat6` | `int64` | 0 | 0.000000% |
| `onehot_feat7` | `int64` | 0 | 0.000000% |
| `onehot_feat8` | `int64` | 0 | 0.000000% |
| `onehot_feat9` | `int64` | 0 | 0.000000% |
| `onehot_feat10` | `int64` | 0 | 0.000000% |
| `onehot_feat11` | `int64` | 0 | 0.000000% |
| `onehot_feat12` | `float64` | 77 | 1.073021% |
| `onehot_feat13` | `float64` | 75 | 1.045151% |
| `onehot_feat14` | `float64` | 75 | 1.045151% |
| `onehot_feat15` | `float64` | 74 | 1.031215% |
| `onehot_feat16` | `float64` | 74 | 1.031215% |
| `onehot_feat17` | `float64` | 74 | 1.031215% |

### `social_network.csv`

Rows: 472

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `user_id` | `int64` | 0 | 0.000000% |
| `friend_list` | `object` | 0 | 0.000000% |

### `kuairec_caption_category.csv`

Rows: 10,728

| field | observed dtype(s) | missing | missing rate |
|---|---|---:|---:|
| `video_id` | `int64` | 0 | 0.000000% |
| `manual_cover_text` | `object` | 0 | 0.000000% |
| `caption` | `object` | 1,355 | 12.630500% |
| `topic_tag` | `object` | 0 | 0.000000% |
| `first_level_category_id` | `int64` | 0 | 0.000000% |
| `first_level_category_name` | `object` | 0 | 0.000000% |
| `second_level_category_id` | `int64` | 0 | 0.000000% |
| `second_level_category_name` | `object` | 0 | 0.000000% |
| `third_level_category_id` | `int64` | 0 | 0.000000% |
| `third_level_category_name` | `object` | 0 | 0.000000% |
