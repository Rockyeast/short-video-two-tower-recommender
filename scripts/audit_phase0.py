#!/usr/bin/env python3
"""Run the KuaiRec Phase 0 data and evaluation audit.

This script intentionally contains no model training or baseline execution.  It
creates a data inventory, a globally chronological split manifest, evaluation
cost estimates, and final-holdout lock files.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


CHUNK_SIZE = 250_000
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
EVENT_COLUMNS = [
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "time",
    "date",
    "timestamp",
    "watch_ratio",
]
EVENT_KEY = ["user_id", "video_id", "timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("generate", "verify"),
        default="generate",
        help=(
            "generate refuses to overwrite a bundle; verify recomputes the bundle "
            "in a temporary directory and compares it with committed outputs"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase0"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/split_manifest.json")
    )
    parser.add_argument(
        "--reference-report-dir",
        type=Path,
        default=Path("reports/phase0"),
        help="Committed report bundle used by --mode verify",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("manifests/split_manifest.json"),
        help="Committed split manifest used by --mode verify",
    )
    parser.add_argument(
        "--supersede-protocol-v1",
        action="store_true",
        help=(
            "One-time reviewed migration: archive the existing schema-v1 bundle "
            "and generate protocol-v2 outputs. Never use for ordinary reruns."
        ),
    )
    parser.add_argument(
        "--supersede-protocol-v2",
        action="store_true",
        help=(
            "One-time reviewed migration: archive the complete protocol-v2 bundle "
            "and generate protocol-v2.1 outputs. Never use for ordinary reruns."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_timestamp(value: float | None) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def local_timestamp(value: float | None, timezone_name: str = "Asia/Shanghai") -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return datetime.fromtimestamp(value, tz=ZoneInfo(timezone_name)).isoformat()


def add_counts(target: Counter[int], values: pd.Series) -> None:
    target.update({int(key): int(value) for key, value in values.items()})


def distribution(values: list[int] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, QUANTILES)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "zero_count": int((array == 0).sum()),
        "zero_fraction": float((array == 0).mean()),
        "quantiles": {
            f"p{int(q * 100):02d}": float(value)
            for q, value in zip(QUANTILES, quantiles, strict=True)
        },
    }


def label_bucket_counts(watch_ratio: pd.Series) -> Counter[str]:
    valid = pd.to_numeric(watch_ratio, errors="coerce")
    result: Counter[str] = Counter()
    result["missing"] = int(valid.isna().sum())
    valid = valid.dropna()
    result["watch_ratio < 0"] = int((valid < 0).sum())
    result["0 <= watch_ratio < 0.25"] = int(
        ((valid >= 0) & (valid < 0.25)).sum()
    )
    result["0.25 <= watch_ratio < 0.5"] = int(
        ((valid >= 0.25) & (valid < 0.5)).sum()
    )
    result["0.5 <= watch_ratio < 1"] = int(
        ((valid >= 0.5) & (valid < 1)).sum()
    )
    result["1 <= watch_ratio <= 2"] = int(
        ((valid >= 1) & (valid <= 2)).sum()
    )
    result["watch_ratio > 2"] = int((valid > 2).sum())
    return result


def find_dataset_files(data_root: Path, expected: list[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in expected:
        matches = sorted(data_root.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {name}, found: {matches}")
        found[name] = matches[0]
    actual_csvs = sorted(data_root.rglob("*.csv"))
    unexpected = [path for path in actual_csvs if path.name not in expected]
    if unexpected:
        raise RuntimeError(f"Unexpected CSV files must be audited explicitly: {unexpected}")
    return found


def scan_csv(path: Path, interaction: bool) -> dict[str, Any]:
    row_count = 0
    null_counts: Counter[str] = Counter()
    dtype_names: dict[str, set[str]] = {}
    users: set[int] = set()
    videos: set[int] = set()
    dates: set[int] = set()
    per_user_events: Counter[int] = Counter()
    per_user_positives: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    quick_skip_count = 0
    watch_sum = 0.0
    watch_count = 0
    watch_min = math.inf
    watch_max = -math.inf
    timestamp_min = math.inf
    timestamp_max = -math.inf
    date_field_mismatch_count = 0
    comparable_date_rows = 0

    read_options: dict[str, Any] = {
        "chunksize": CHUNK_SIZE,
        "low_memory": False,
    }
    if path.name == "kuairec_caption_category.csv":
        # The official file contains bare carriage returns inside four caption
        # fields. Treat only LF as a record terminator so those rows stay intact.
        read_options["lineterminator"] = "\n"
    for chunk in pd.read_csv(path, **read_options):
        row_count += len(chunk)
        for column in chunk.columns:
            null_counts[column] += int(chunk[column].isna().sum())
            dtype_names.setdefault(column, set()).add(str(chunk[column].dtype))

        if not interaction:
            continue

        user = pd.to_numeric(chunk["user_id"], errors="coerce").dropna().astype(int)
        video = pd.to_numeric(chunk["video_id"], errors="coerce").dropna().astype(int)
        users.update(user.unique().tolist())
        videos.update(video.unique().tolist())
        add_counts(per_user_events, user.value_counts())

        date = pd.to_numeric(chunk["date"], errors="coerce").dropna().astype(int)
        dates.update(date.unique().tolist())
        timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce").dropna()
        if not timestamp.empty:
            timestamp_min = min(timestamp_min, float(timestamp.min()))
            timestamp_max = max(timestamp_max, float(timestamp.max()))
        raw_timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")
        local_date = pd.to_datetime(
            raw_timestamp, unit="s", utc=True, errors="coerce"
        ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d")
        local_date = pd.to_numeric(local_date, errors="coerce")
        declared_date = pd.to_numeric(chunk["date"], errors="coerce")
        comparable = local_date.notna() & declared_date.notna()
        comparable_date_rows += int(comparable.sum())
        date_field_mismatch_count += int(
            (local_date[comparable] != declared_date[comparable]).sum()
        )

        watch_ratio = pd.to_numeric(chunk["watch_ratio"], errors="coerce")
        positive = watch_ratio > 2.0
        positive_users = pd.to_numeric(
            chunk.loc[positive, "user_id"], errors="coerce"
        ).dropna().astype(int)
        add_counts(per_user_positives, positive_users.value_counts())
        label_counts.update(label_bucket_counts(watch_ratio))

        valid_watch = watch_ratio.dropna()
        if not valid_watch.empty:
            watch_sum += float(valid_watch.sum())
            watch_count += int(valid_watch.size)
            watch_min = min(watch_min, float(valid_watch.min()))
            watch_max = max(watch_max, float(valid_watch.max()))

        play = pd.to_numeric(chunk["play_duration"], errors="coerce")
        duration = pd.to_numeric(chunk["video_duration"], errors="coerce")
        quick_skip_count += int((play < np.minimum(3000, duration)).sum())

    fields = []
    for column in dtype_names:
        fields.append(
            {
                "name": column,
                "observed_dtypes": sorted(dtype_names[column]),
                "missing_count": int(null_counts[column]),
                "missing_fraction": (
                    float(null_counts[column] / row_count) if row_count else None
                ),
            }
        )

    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "rows": row_count,
        "columns": len(fields),
        "fields": fields,
    }
    if path.name == "kuairec_caption_category.csv":
        result["parser_note"] = {
            "bare_carriage_return_bytes": path.read_bytes().count(b"\r"),
            "record_terminator": "LF only",
            "reason": (
                "The default pandas C parser overflows and the Python parser "
                "misreads embedded bare CR bytes as extra records."
            ),
        }
    if interaction:
        positive_values = [per_user_positives.get(user_id, 0) for user_id in users]
        result["interaction"] = {
            "users": len(users),
            "videos": len(videos),
            "dates": sorted(dates),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "timestamp_min": timestamp_min if math.isfinite(timestamp_min) else None,
            "timestamp_max": timestamp_max if math.isfinite(timestamp_max) else None,
            "timestamp_min_utc": utc_timestamp(timestamp_min),
            "timestamp_max_utc": utc_timestamp(timestamp_max),
            "timestamp_min_asia_shanghai": local_timestamp(timestamp_min),
            "timestamp_max_asia_shanghai": local_timestamp(timestamp_max),
            "date_field_consistency": {
                "comparison": "date versus timestamp converted to Asia/Shanghai calendar date",
                "comparable_rows": comparable_date_rows,
                "mismatch_count": date_field_mismatch_count,
                "mismatch_fraction": (
                    date_field_mismatch_count / comparable_date_rows
                    if comparable_date_rows
                    else None
                ),
                "split_uses_raw_date_field": False,
            },
            "watch_ratio": {
                "mean": watch_sum / watch_count if watch_count else None,
                "min": watch_min if math.isfinite(watch_min) else None,
                "max": watch_max if math.isfinite(watch_max) else None,
                "buckets": dict(label_counts),
            },
            "positive_count": int(sum(per_user_positives.values())),
            "positive_fraction": (
                float(sum(per_user_positives.values()) / row_count) if row_count else 0.0
            ),
            "quick_skip_count": quick_skip_count,
            "quick_skip_fraction": quick_skip_count / row_count if row_count else 0.0,
            "events_per_user": distribution(list(per_user_events.values())),
            "positives_per_user": distribution(positive_values),
            "user_ids": sorted(users),
            "video_ids": sorted(videos),
            "per_user_events": dict(per_user_events),
            "per_user_positives": dict(per_user_positives),
        }
    return result


def small_matrix_observation_coverage(path: Path) -> dict[str, Any]:
    """Audit Small Matrix observed and officially blocked/missing pairs."""

    users: set[int] = set()
    catalog: set[int] = set()
    unique_pairs_per_user: dict[int, int] = {}
    duplicate_rows = 0
    carry_user: int | None = None
    carry_seen_videos: set[int] = set()
    previous_user: int | None = None

    for chunk in pd.read_csv(
        path, usecols=["user_id", "video_id"], chunksize=CHUNK_SIZE
    ):
        user = pd.to_numeric(chunk["user_id"], errors="coerce")
        video = pd.to_numeric(chunk["video_id"], errors="coerce")
        if user.isna().any() or video.isna().any():
            raise RuntimeError("small_matrix user_id or video_id is invalid")
        user = user.astype(int)
        video = video.astype(int)
        user_values = user.to_numpy()
        if (user_values[1:] < user_values[:-1]).any() or (
            previous_user is not None and int(user_values[0]) < previous_user
        ):
            raise RuntimeError("small_matrix must remain grouped by user_id")

        grouped = pd.DataFrame({"user_id": user, "video_id": video}).groupby(
            "user_id", sort=False
        )
        group_ids = list(grouped.indices)
        for position, (user_id, group) in enumerate(grouped):
            user_id = int(user_id)
            values = group["video_id"].astype(int).tolist()
            if user_id == carry_user:
                seen = carry_seen_videos
            else:
                seen = set()
            before = len(seen)
            seen.update(values)
            duplicate_rows += len(values) - (len(seen) - before)
            unique_pairs_per_user[user_id] = len(seen)
            if position == len(group_ids) - 1:
                carry_user = user_id
                carry_seen_videos = seen

        users.update(user.unique().tolist())
        catalog.update(video.unique().tolist())
        previous_user = int(user_values[-1])

    expected_pairs = len(users) * len(catalog)
    observed_unique_pairs = sum(unique_pairs_per_user.values())
    missing_pairs = expected_pairs - observed_unique_pairs
    per_user_missing = [
        len(catalog) - unique_pairs_per_user.get(user_id, 0) for user_id in users
    ]
    return {
        "users": len(users),
        "catalog_videos": len(catalog),
        "expected_complete_pairs": expected_pairs,
        "observed_unique_pairs": observed_unique_pairs,
        "duplicate_rows": duplicate_rows,
        "missing_pairs": missing_pairs,
        "blocked_or_missing_pairs": missing_pairs,
        "observed_pair_fraction": (
            observed_unique_pairs / expected_pairs if expected_pairs else 0.0
        ),
        "missing_pairs_per_user": distribution(per_user_missing),
        "primary_candidate_size_per_user": distribution(
            list(unique_pairs_per_user.values())
        ),
        "secondary_full_catalog_size": len(catalog),
        "official_missing_pair_semantics": (
            "user blocked the video or its author; inferred only at pair level"
        ),
        "primary_policy": (
            "exclude each user's blocked/missing pairs; rank only physically "
            "observed NORMAL-video pairs, including observed nonpositive pairs"
        ),
        "primary_video_type": "NORMAL",
        "advertisement_policy": (
            "exclude AD from primary quality metrics; report AD only as a "
            "separate diagnostic"
        ),
        "secondary_safety_policy": (
            "rank all 3327 videos only to report Blocked@K and "
            "BlockedUserHitRate@K; never claim it as primary model quality"
        ),
        "blocked_information_forbidden_uses": [
            "training",
            "features",
            "user_history",
            "negative_sampling",
            "hyperparameter_selection",
        ],
    }


def timestamp_quantile_boundaries(
    timestamps: np.ndarray, train_fraction: float, validation_fraction: float
) -> dict[str, float | int]:
    values = np.asarray(timestamps, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise RuntimeError("At least three valid timestamps are required")
    train_index = math.floor(values.size * train_fraction)
    validation_index = math.floor(values.size * (train_fraction + validation_fraction))
    if train_index <= 0 or validation_index <= train_index or validation_index >= values.size:
        raise RuntimeError("Invalid temporal quantile indices")
    partitioned = np.partition(values, (train_index, validation_index))
    train_end = float(partitioned[train_index])
    validation_end = float(partitioned[validation_index])
    if train_end >= validation_end:
        raise RuntimeError("Temporal boundaries collapse onto the same timestamp")
    return {
        "valid_timestamp_count": int(values.size),
        "train_target_index": train_index,
        "validation_end_target_index": validation_index,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
    }


def validate_split_fractions(split: Mapping[str, Any]) -> tuple[float, float, float]:
    """Validate the three configured temporal fractions as one complete partition."""

    expected_names = (
        "train_fraction",
        "validation_fraction",
        "temporal_final_fraction",
    )
    policy = split.get("fraction_validation")
    if not isinstance(policy, Mapping):
        raise RuntimeError("split.fraction_validation policy is required")
    names = tuple(str(name) for name in policy.get("keys", ()))
    if names != expected_names:
        raise RuntimeError(
            "split.fraction_validation.keys must exactly match "
            f"{list(expected_names)} in order"
        )
    if policy.get("each_fraction_strictly_between_zero_and_one") is not True:
        raise RuntimeError("Split fraction range validation must remain enabled")
    try:
        required_sum = float(policy["required_sum"])
        absolute_tolerance = float(policy["absolute_tolerance"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid split fraction validation policy: {error}") from error
    if not math.isfinite(required_sum) or not math.isfinite(absolute_tolerance):
        raise RuntimeError("Split fraction sum policy must be finite")
    if absolute_tolerance < 0.0:
        raise RuntimeError("Split fraction absolute tolerance cannot be negative")
    try:
        values = tuple(float(split[name]) for name in names)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid temporal split fractions: {error}") from error
    if any(not math.isfinite(value) or value <= 0.0 or value >= 1.0 for value in values):
        raise RuntimeError(
            "Each temporal split fraction must be finite and strictly between 0 and 1"
        )
    if not math.isclose(
        sum(values), required_sum, rel_tol=0.0, abs_tol=absolute_tolerance
    ):
        raise RuntimeError(
            "Temporal split fractions violate configured required_sum/tolerance; "
            f"expected {required_sum:.17g} +/- {absolute_tolerance:.17g}, "
            f"got {sum(values):.17g}"
        )
    return values


def choose_timestamp_boundaries(
    path: Path, train_fraction: float, validation_fraction: float
) -> dict[str, float | int]:
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=CHUNK_SIZE):
        chunks.append(pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy())
    return timestamp_quantile_boundaries(
        np.concatenate(chunks), train_fraction, validation_fraction
    )


def load_upload_availability_epochs(
    path: Path,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return declared upload-date midnight and conservative availability.

    KuaiRec provides only a calendar date for ``upload_dt``.  Treating that
    date's 00:00 as the real upload instant can leak an item into earlier
    queries on the same day.  The next local midnight is the first timestamp at
    which the item is certainly known to have been uploaded.
    """

    frame = pd.read_csv(path, usecols=["video_id", "upload_dt"])
    upload = pd.to_datetime(frame["upload_dt"], errors="coerce").dt.tz_localize(
        "Asia/Shanghai"
    )
    upload_epoch = upload.map(
        lambda value: value.timestamp() if pd.notna(value) else np.nan
    )
    available_epoch = (upload + pd.Timedelta(days=1)).map(
        lambda value: value.timestamp() if pd.notna(value) else np.nan
    )
    video_index = frame["video_id"].astype(int)
    earliest_upload = pd.Series(upload_epoch.to_numpy(), index=video_index).groupby(
        level=0
    ).min()
    earliest_available = pd.Series(
        available_epoch.to_numpy(), index=video_index
    ).groupby(level=0).min()
    return (
        {
            int(video): float(value)
            for video, value in earliest_upload.dropna().items()
        },
        {
            int(video): float(value)
            for video, value in earliest_available.dropna().items()
        },
    )


def timestamp_to_local_date(value: float) -> int:
    return int(
        datetime.fromtimestamp(float(value), tz=ZoneInfo("Asia/Shanghai")).strftime(
            "%Y%m%d"
        )
    )


def _bool_array_to_mask(values: np.ndarray) -> int:
    """Encode a small catalog boolean vector as a Python integer bitset."""

    packed = np.packbits(values.astype(np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def _stable_int_set_hash(values: set[int]) -> str:
    body = "".join(f"{value}\n" for value in sorted(values)).encode()
    return sha256_bytes(body)


def _stable_target_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return sha256_bytes(b"")
    ordered = frame.sort_values(
        ["user_id", "timestamp", "video_id"], kind="mergesort"
    )
    body = "".join(
        f"{int(row.user_id)},{float(row.timestamp):.6f},{int(row.video_id)}\n"
        for row in ordered.itertuples(index=False)
    ).encode()
    return sha256_bytes(body)


def load_candidate_catalog_policy(
    path: Path, timestamp_min: float, timestamp_max: float
) -> dict[str, Any]:
    """Build causal daily candidate masks and audit item metadata.

    KuaiRec exposes only daily item snapshots.  For a query on Shanghai date
    ``D`` we therefore use the latest snapshot with ``snapshot_date < D``.
    Same-day status is deliberately unavailable.  The primary task contains
    only NORMAL, public videos whose upload date is also strictly before ``D``.
    """

    columns = ["video_id", "date", "upload_dt", "video_type", "visible_status"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["video_id"] = pd.to_numeric(frame["video_id"], errors="raise").astype(int)
    frame["date"] = pd.to_numeric(frame["date"], errors="raise").astype(int)
    frame["upload_dt_parsed"] = pd.to_datetime(frame["upload_dt"], errors="coerce")
    frame["upload_date"] = pd.to_numeric(
        frame["upload_dt_parsed"].dt.strftime("%Y%m%d"), errors="coerce"
    )

    duplicate_video_date_rows = int(
        frame.duplicated(["video_id", "date"], keep=False).sum()
    )
    if duplicate_video_date_rows:
        raise RuntimeError(
            "item_daily_features contains duplicate (video_id, date) snapshots"
        )

    invariant_fields = ("upload_dt", "video_type")
    inconsistent = {
        field: int((frame.groupby("video_id")[field].nunique(dropna=False) > 1).sum())
        for field in invariant_fields
    }
    if any(inconsistent.values()):
        raise RuntimeError(f"Per-video invariant metadata changed: {inconsistent}")

    video_ids = sorted(frame["video_id"].unique().tolist())
    index = {int(video_id): position for position, video_id in enumerate(video_ids)}
    upload_by_video = (
        frame.drop_duplicates("video_id")
        .set_index("video_id")["upload_date"]
        .to_dict()
    )
    type_by_video = (
        frame.drop_duplicates("video_id")
        .set_index("video_id")["video_type"]
        .astype(str)
        .to_dict()
    )

    ordered = frame.sort_values(["date", "video_id"], kind="mergesort")
    snapshots_by_date = {
        int(date): group
        for date, group in ordered.groupby("date", sort=True)
    }
    snapshot_dates = sorted(snapshots_by_date)
    start_date = datetime.fromtimestamp(
        timestamp_min, tz=ZoneInfo("Asia/Shanghai")
    ).date()
    end_date = datetime.fromtimestamp(
        timestamp_max, tz=ZoneInfo("Asia/Shanghai")
    ).date()
    query_dates = [
        int(value.strftime("%Y%m%d"))
        for value in pd.date_range(start_date, end_date, freq="D")
    ]

    known = np.zeros(len(video_ids), dtype=bool)
    public = np.zeros(len(video_ids), dtype=bool)
    normal = np.zeros(len(video_ids), dtype=bool)
    current_status: dict[int, str] = {}
    current_type: dict[int, str] = {}
    cursor = 0
    state_by_date: dict[int, dict[str, Any]] = {}
    for query_date in query_dates:
        while cursor < len(snapshot_dates) and snapshot_dates[cursor] < query_date:
            snapshot = snapshots_by_date[snapshot_dates[cursor]]
            for row in snapshot.itertuples(index=False):
                video_id = int(row.video_id)
                position = index[video_id]
                status = str(row.visible_status)
                video_type = str(row.video_type)
                known[position] = True
                public[position] = status == "public"
                normal[position] = video_type == "NORMAL"
                current_status[video_id] = status
                current_type[video_id] = video_type
            cursor += 1

        uploaded = np.asarray(
            [
                pd.notna(upload_by_video[video_id])
                and int(upload_by_video[video_id]) < query_date
                for video_id in video_ids
            ],
            dtype=bool,
        )
        eligible = known & public & normal & uploaded
        state_by_date[query_date] = {
            "known": known.copy(),
            "public": public.copy(),
            "normal": normal.copy(),
            "uploaded": uploaded,
            "eligible": eligible,
            "eligible_mask": _bool_array_to_mask(eligible),
            "eligible_count": int(eligible.sum()),
        }

    transitions: Counter[str] = Counter()
    for _, group in ordered.groupby("video_id", sort=False):
        statuses = group.sort_values("date", kind="mergesort")[
            "visible_status"
        ].astype(str).tolist()
        for before, after in zip(statuses, statuses[1:]):
            if before != after:
                transitions[f"{before}->{after}"] += 1

    per_video_status_count = frame.groupby("video_id")["visible_status"].nunique()
    audit = {
        "policy_id": "normal-public-prior-day-status-v1",
        "query_timezone": "Asia/Shanghai",
        "snapshot_rule": "latest item_daily_features row with date < query local date",
        "upload_rule": "upload_dt local date < query local date",
        "primary_video_type": "NORMAL",
        "advertisement_policy": "exclude AD from the primary recommendation task",
        "unknown_or_no_prior_snapshot_policy": "exclude",
        "video_count": len(video_ids),
        "snapshot_rows": int(len(frame)),
        "snapshot_date_min": int(frame["date"].min()),
        "snapshot_date_max": int(frame["date"].max()),
        "duplicate_video_date_rows": duplicate_video_date_rows,
        "per_video_inconsistent_fields": inconsistent,
        "video_type_video_counts": {
            str(key): int(value)
            for key, value in Counter(type_by_video.values()).items()
        },
        "visible_status_row_counts": {
            str(key): int(value)
            for key, value in frame["visible_status"].astype(str).value_counts().items()
        },
        "videos_by_distinct_visible_status_count": {
            str(int(key)): int(value)
            for key, value in per_video_status_count.value_counts().sort_index().items()
        },
        "visible_status_transition_counts": dict(sorted(transitions.items())),
        "daily_primary_catalog_size": {
            str(date): state["eligible_count"] for date, state in state_by_date.items()
        },
    }
    return {
        "video_ids": video_ids,
        "video_index": index,
        "state_by_date": state_by_date,
        "audit": audit,
    }


def _catalog_flags(
    video: pd.Series, timestamp: pd.Series, catalog: dict[str, Any]
) -> dict[str, pd.Series]:
    positions = video.map(catalog["video_index"])
    dates = pd.to_datetime(timestamp, unit="s", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    ).dt.strftime("%Y%m%d").astype(int)
    flags = {
        name: pd.Series(False, index=video.index, dtype=bool)
        for name in ("known", "public", "normal", "uploaded", "eligible")
    }
    valid_position = positions.notna()
    for query_date in dates.unique().tolist():
        state = catalog["state_by_date"].get(int(query_date))
        selection = dates.eq(query_date) & valid_position
        if state is None or not selection.any():
            continue
        integer_positions = positions.loc[selection].astype(int).to_numpy()
        for name in flags:
            flags[name].loc[selection] = state[name][integer_positions]
    flags["metadata_present"] = valid_position
    return flags


def _count_distribution_by_user_and_date(
    weighted_keys: pd.DataFrame,
) -> dict[str, Any]:
    if weighted_keys.empty:
        return {
            "affected_users": 0,
            "extras_per_affected_user": distribution([]),
            "by_user": {},
            "top_users": [],
            "by_asia_shanghai_date": {},
        }
    by_user = (
        weighted_keys.groupby("user_id", sort=False)["extra_count"].sum().astype(int)
    )
    local_dates = pd.to_datetime(
        weighted_keys["timestamp"], unit="s", utc=True
    ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    by_date = (
        weighted_keys.assign(local_date=local_dates)
        .groupby("local_date", sort=True)["extra_count"]
        .sum()
        .astype(int)
    )
    top = sorted(
        ((int(user_id), int(count)) for user_id, count in by_user.items()),
        key=lambda item: (-item[1], item[0]),
    )[:20]
    return {
        "affected_users": int(len(by_user)),
        "extras_per_affected_user": distribution(by_user.to_numpy()),
        "by_user": {
            str(int(user_id)): int(count)
            for user_id, count in sorted(
                by_user.items(), key=lambda item: int(item[0])
            )
        },
        "top_users": [
            {"user_id": user_id, "extra_count": count} for user_id, count in top
        ],
        "by_asia_shanghai_date": {
            str(date): int(count) for date, count in by_date.items()
        },
    }


def canonicalize_behavior_events(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Collapse a user's raw Big Matrix rows to one deterministic event key.

    The canonical key is ``(user_id, video_id, timestamp)``. Full eight-field
    duplicates are removed first; remaining non-identical rows sharing the key
    collapse deterministically. Derived strong-positive and quick-skip flags are
    aggregated across the key so a representative row cannot hide a label
    conflict. A quick-skip/strong-positive conflict is never a hard negative.

    The returned weighted frames contain duplicate *extras* and are intentionally
    small; callers aggregate them into the user/date audit without retaining raw
    behavior logs in memory.
    """

    missing = [column for column in EVENT_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Big Matrix event columns are missing: {missing}")
    if frame.empty:
        empty = frame.loc[:, EVENT_COLUMNS].copy()
        for name in (
            "_is_strong_positive",
            "_is_quick_skip",
            "_binary_positive_conflict",
            "_quick_skip_strong_conflict",
        ):
            empty[name] = pd.Series(dtype=bool)
        weighted = pd.DataFrame(columns=EVENT_KEY + ["extra_count"])
        return empty, {
            "raw_rows": 0,
            "exact_duplicate_rows_removed": 0,
            "exact_duplicate_full_row_group_count": 0,
            "same_key_nonexact_rows_removed": 0,
            "same_key_nonexact_key_count": 0,
            "binary_positive_conflict_key_count": 0,
            "quick_skip_strong_positive_conflict_key_count": 0,
            "canonical_event_count": 0,
            "reconciliation_ok": True,
        }, weighted.copy(), weighted.copy()

    context = frame.loc[:, EVENT_COLUMNS].copy()
    for column in (
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "date",
        "timestamp",
        "watch_ratio",
    ):
        context[column] = pd.to_numeric(context[column], errors="coerce")
    if context[["user_id", "video_id", "timestamp"]].isna().any().any():
        raise RuntimeError("Canonical behavior identity contains missing values")
    context["user_id"] = context["user_id"].astype(int)
    context["video_id"] = context["video_id"].astype(int)

    exact_sizes = context.groupby(
        EVENT_COLUMNS, sort=False, dropna=False
    ).size().rename("row_count")
    exact_duplicate_groups = exact_sizes[exact_sizes > 1].reset_index()
    exact_weighted = exact_duplicate_groups.loc[:, EVENT_KEY].copy()
    exact_weighted["extra_count"] = (
        exact_duplicate_groups["row_count"].astype(int) - 1
    )
    exact = context.drop_duplicates(EVENT_COLUMNS, keep="first").copy()

    play = pd.to_numeric(exact["play_duration"], errors="coerce")
    duration = pd.to_numeric(exact["video_duration"], errors="coerce")
    exact["_row_strong_positive"] = (
        pd.to_numeric(exact["watch_ratio"], errors="coerce") > 2.0
    )
    exact["_row_quick_skip"] = play < np.minimum(3000.0, duration)
    grouped = exact.groupby(EVENT_KEY, sort=False, dropna=False)
    key_flags = grouped.agg(
        _strong_any=("_row_strong_positive", "any"),
        _strong_all=("_row_strong_positive", "all"),
        _quick_any=("_row_quick_skip", "any"),
        _key_row_count=("video_id", "size"),
    ).reset_index()
    key_flags["_binary_positive_conflict"] = (
        key_flags["_strong_any"] & ~key_flags["_strong_all"]
    )
    key_flags["_quick_skip_strong_conflict"] = (
        key_flags["_quick_any"] & key_flags["_strong_any"]
    )
    key_flags["_is_strong_positive"] = (
        key_flags["_strong_any"] & ~key_flags["_binary_positive_conflict"]
    )
    key_flags["_is_quick_skip"] = (
        key_flags["_quick_any"] & ~key_flags["_quick_skip_strong_conflict"]
    )

    nonexact_groups = key_flags.loc[key_flags["_key_row_count"] > 1].copy()
    nonexact_weighted = nonexact_groups.loc[:, EVENT_KEY].copy()
    nonexact_weighted["extra_count"] = (
        nonexact_groups["_key_row_count"].astype(int) - 1
    )

    sort_columns = EVENT_KEY + [
        column for column in EVENT_COLUMNS if column not in EVENT_KEY
    ]
    representative = (
        exact.sort_values(sort_columns, kind="mergesort", na_position="last")
        .drop_duplicates(EVENT_KEY, keep="first")
        .drop(columns=["_row_strong_positive", "_row_quick_skip"])
    )
    canonical = representative.merge(
        key_flags[
            EVENT_KEY
            + [
                "_is_strong_positive",
                "_is_quick_skip",
                "_binary_positive_conflict",
                "_quick_skip_strong_conflict",
            ]
        ],
        on=EVENT_KEY,
        how="left",
        validate="one_to_one",
    ).sort_values(["timestamp", "video_id"], kind="mergesort")
    canonical = canonical.reset_index(drop=True)

    exact_removed = int(len(context) - len(exact))
    nonexact_removed = int(len(exact) - len(canonical))
    reconciled = len(context) - exact_removed - nonexact_removed == len(canonical)
    if not reconciled:
        raise RuntimeError("Canonical behavior-event accounting did not reconcile")
    return canonical, {
        "raw_rows": int(len(context)),
        "exact_duplicate_rows_removed": exact_removed,
        "exact_duplicate_full_row_group_count": int(len(exact_duplicate_groups)),
        "same_key_nonexact_rows_removed": nonexact_removed,
        "same_key_nonexact_key_count": int(len(nonexact_groups)),
        "binary_positive_conflict_key_count": int(
            key_flags["_binary_positive_conflict"].sum()
        ),
        "quick_skip_strong_positive_conflict_key_count": int(
            key_flags["_quick_skip_strong_conflict"].sum()
        ),
        "canonical_event_count": int(len(canonical)),
        "reconciliation_ok": bool(reconciled),
    }, exact_weighted, nonexact_weighted


def canonicalize_eligible_targets(
    eligible_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Turn raw eligible rows into one unambiguous target per (u, i, t).

    Exact duplicate rows are removed first.  A key containing both a strong and
    a non-strong label is excluded as ambiguous.  All remaining positive rows
    for one key collapse to one canonical target.  This rule is fixed before
    any baseline or learned model is developed.
    """

    key = ["user_id", "video_id", "timestamp"]
    if eligible_context.empty:
        empty = eligible_context.loc[:, key].copy()
        return empty, {
            "raw_eligible_positive_rows": 0,
            "exact_duplicate_positive_rows_removed": 0,
            "exact_duplicate_positive_key_count": 0,
            "same_key_nonexact_positive_rows_removed": 0,
            "same_key_nonexact_positive_key_count": 0,
            "binary_label_conflict_keys_excluded": 0,
            "positive_rows_in_conflict_keys_excluded": 0,
            "numeric_watch_ratio_disagreement_keys": 0,
            "canonical_target_count": 0,
            "reconciliation_ok": True,
            "difference_distribution": _count_distribution_by_user_and_date(
                pd.DataFrame(columns=key + ["extra_count"])
            ),
        }

    context = eligible_context.copy()
    context["watch_ratio"] = pd.to_numeric(
        context["watch_ratio"], errors="coerce"
    )
    context["is_strong_positive"] = context["watch_ratio"] > 2.0
    original_columns = [
        column for column in context.columns if column != "is_strong_positive"
    ]
    positive = context.loc[context["is_strong_positive"]].copy()
    raw_positive_rows = int(len(positive))
    exact_group_sizes = positive.groupby(
        original_columns, sort=False, dropna=False
    ).size()
    exact_duplicate_key_count = int((exact_group_sizes > 1).sum())
    exact_positive = positive.drop_duplicates(original_columns, keep="first")
    exact_removed = raw_positive_rows - len(exact_positive)

    exact_context = context.drop_duplicates(original_columns, keep="first")
    binary_nunique = exact_context.groupby(key, sort=False)[
        "is_strong_positive"
    ].nunique()
    conflict_index = set(binary_nunique[binary_nunique > 1].index.tolist())
    numeric_nunique = exact_context.groupby(key, sort=False)["watch_ratio"].nunique(
        dropna=False
    )
    numeric_disagreement_count = int((numeric_nunique > 1).sum())

    key_index = pd.MultiIndex.from_frame(exact_positive[key])
    conflict_mask = np.fromiter(
        (tuple(value) in conflict_index for value in key_index),
        dtype=bool,
        count=len(key_index),
    )
    positive_rows_in_conflicts = int(conflict_mask.sum())
    unambiguous_positive = exact_positive.loc[~conflict_mask].copy()
    nonexact_key_sizes = unambiguous_positive.groupby(key, sort=False).size()
    nonexact_key_count = int((nonexact_key_sizes > 1).sum())
    canonical_sort = key + [
        column
        for column in original_columns
        if column not in key and column != "is_strong_positive"
    ]
    unambiguous_positive = unambiguous_positive.sort_values(
        canonical_sort, kind="mergesort", na_position="last"
    )
    canonical = unambiguous_positive.drop_duplicates(key, keep="first")[key].copy()
    same_key_nonexact_removed = len(unambiguous_positive) - len(canonical)

    key_positive_counts = positive.groupby(key, sort=False).size().rename("positive_rows")
    weighted = key_positive_counts.reset_index()
    weighted["is_conflict"] = [
        tuple(value) in conflict_index
        for value in pd.MultiIndex.from_frame(weighted[key])
    ]
    weighted["extra_count"] = np.where(
        weighted["is_conflict"],
        weighted["positive_rows"],
        weighted["positive_rows"] - 1,
    ).astype(int)
    weighted = weighted.loc[weighted["extra_count"] > 0, key + ["extra_count"]]

    reconciled = (
        raw_positive_rows
        - exact_removed
        - same_key_nonexact_removed
        - positive_rows_in_conflicts
        == len(canonical)
    )
    if not reconciled:
        raise RuntimeError("Eligible target deduplication accounting did not reconcile")
    return canonical, {
        "formal_rule": (
            "drop exact duplicate rows; exclude (user_id, video_id, timestamp) "
            "keys with conflicting binary watch_ratio>2 labels; collapse every "
            "remaining positive key to one canonical target"
        ),
        "raw_eligible_positive_rows": raw_positive_rows,
        "exact_duplicate_positive_rows_removed": int(exact_removed),
        "exact_duplicate_positive_key_count": exact_duplicate_key_count,
        "same_key_nonexact_positive_rows_removed": int(
            same_key_nonexact_removed
        ),
        "same_key_nonexact_positive_key_count": nonexact_key_count,
        "binary_label_conflict_keys_excluded": int(len(conflict_index)),
        "positive_rows_in_conflict_keys_excluded": positive_rows_in_conflicts,
        "numeric_watch_ratio_disagreement_keys": numeric_disagreement_count,
        "canonical_target_count": int(len(canonical)),
        "reconciliation_ok": bool(reconciled),
        "difference_distribution": _count_distribution_by_user_and_date(weighted),
    }


def scan_big_splits(
    path: Path,
    boundaries: dict[str, float | int],
    upload_epoch_by_video: dict[int, float],
    available_epoch_by_video: dict[int, float],
    catalog_policy: dict[str, Any] | None = None,
    canonical_behavior_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_order = ("train", "validation", "temporal_final")
    state: dict[str, dict[str, Any]] = {
        name: {
            "rows": 0,
            "positive_count": 0,
            "eligible_positive_count": 0,
            "interaction_before_declared_upload_date_count": 0,
            "interaction_same_day_upload_time_unverifiable_count": 0,
            "positive_before_declared_upload_date_count": 0,
            "positive_same_day_upload_time_unverifiable_count": 0,
            "positive_missing_upload_count": 0,
            "positive_previously_seen_count": 0,
            "positive_missing_catalog_metadata_count": 0,
            "positive_without_prior_status_count": 0,
            "positive_ad_video_count": 0,
            "positive_not_public_count": 0,
            "users": set(),
            "videos": set(),
            "per_user_events": Counter(),
            "per_user_positives": Counter(),
            "eligible_context_parts": [],
            "pre_catalog_context_parts": [],
            "timestamp_min": math.inf,
            "timestamp_max": -math.inf,
        }
        for name in split_order
    }
    carry_user: int | None = None
    carry_first_timestamp_by_video: dict[int, float] = {}
    previous_user: int | None = None
    previous_timestamp: float | None = None
    columns = [
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "time",
        "date",
        "timestamp",
        "watch_ratio",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE):
        user = pd.to_numeric(chunk["user_id"], errors="coerce")
        video = pd.to_numeric(chunk["video_id"], errors="coerce")
        timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")
        if user.isna().any() or video.isna().any() or timestamp.isna().any():
            raise RuntimeError("big_matrix identity or timestamp field is invalid")
        user = user.astype(int)
        video = video.astype(int)

        user_values = user.to_numpy()
        timestamp_values = timestamp.to_numpy()
        if (
            (user_values[1:] < user_values[:-1]).any()
            or (
                (user_values[1:] == user_values[:-1])
                & (timestamp_values[1:] < timestamp_values[:-1])
            ).any()
        ):
            raise RuntimeError(
                "big_matrix must be sorted by user_id and then timestamp"
            )
        if previous_user is not None and (
            int(user_values[0]) < previous_user
            or (
                int(user_values[0]) == previous_user
                and float(timestamp_values[0]) < float(previous_timestamp)
            )
        ):
            raise RuntimeError(
                "big_matrix ordering is broken across CSV chunk boundaries"
            )

        # The source is user-major and chronological within each user. A video
        # is a valid target at t only when that user's first interaction with it
        # is also at t. Equal-timestamp events remain one atomic history group.
        pair_first_timestamp = chunk.groupby(
            ["user_id", "video_id"], sort=False
        )["timestamp"].transform("min")
        first_user = int(user_values[0])
        if carry_user == first_user and carry_first_timestamp_by_video:
            first_user_mask = user == first_user
            previous_first = video.loc[first_user_mask].map(
                carry_first_timestamp_by_video
            )
            pair_first_timestamp.loc[first_user_mask] = np.minimum(
                pair_first_timestamp.loc[first_user_mask].to_numpy(),
                previous_first.fillna(np.inf).to_numpy(),
            )
        first_interaction_at_query_time = timestamp.eq(pair_first_timestamp)

        last_user = int(user_values[-1])
        last_user_frame = pd.DataFrame(
            {
                "video_id": video.loc[user == last_user],
                "timestamp": timestamp.loc[user == last_user],
            }
        )
        last_first = (
            last_user_frame.groupby("video_id", sort=False)["timestamp"]
            .min()
            .to_dict()
        )
        if carry_user == last_user:
            for video_id, first_timestamp in last_first.items():
                old = carry_first_timestamp_by_video.get(int(video_id), math.inf)
                carry_first_timestamp_by_video[int(video_id)] = min(
                    old, float(first_timestamp)
                )
        else:
            carry_first_timestamp_by_video = {
                int(video_id): float(first_timestamp)
                for video_id, first_timestamp in last_first.items()
            }
        carry_user = last_user
        previous_user = last_user
        previous_timestamp = float(timestamp_values[-1])
        split_names = np.select(
            [
                timestamp < float(boundaries["train_end_exclusive"]),
                timestamp < float(boundaries["validation_end_exclusive"]),
            ],
            ["train", "validation"],
            default="temporal_final",
        )
        for split_name in split_order:
            frame = chunk.loc[split_names == split_name]
            if frame.empty:
                continue
            split_index = frame.index
            split_user = user.loc[split_index]
            split_video = video.loc[split_index]
            positive = pd.to_numeric(frame["watch_ratio"], errors="coerce") > 2.0
            upload_epoch = frame["video_id"].map(upload_epoch_by_video)
            available_epoch = frame["video_id"].map(available_epoch_by_video)
            missing_upload = upload_epoch.isna() | available_epoch.isna()
            before_declared_upload_date = frame["timestamp"] < upload_epoch
            same_day_upload_time_unverifiable = (
                ~missing_upload
                & ~before_declared_upload_date
                & (frame["timestamp"] < available_epoch)
            )
            previously_seen = ~first_interaction_at_query_time.loc[split_index]
            if catalog_policy is None:
                metadata_present = ~missing_upload
                prior_status_known = ~missing_upload
                normal_video = pd.Series(True, index=frame.index)
                public_video = pd.Series(True, index=frame.index)
            else:
                catalog_flags = _catalog_flags(
                    split_video, frame["timestamp"], catalog_policy
                )
                metadata_present = catalog_flags["metadata_present"]
                prior_status_known = catalog_flags["known"]
                normal_video = catalog_flags["normal"]
                public_video = catalog_flags["public"]

            # Reasons use an explicit precedence so they reconcile with the
            # canonical target input instead of double counting exclusions.
            missing_catalog = positive & (~metadata_present | missing_upload)
            before_upload = (
                positive & ~missing_catalog & before_declared_upload_date
            )
            same_day_unknown = (
                positive
                & ~missing_catalog
                & ~before_upload
                & same_day_upload_time_unverifiable
            )
            no_prior_status = (
                positive
                & ~missing_catalog
                & ~before_upload
                & ~same_day_unknown
                & ~prior_status_known
            )
            ad_video = (
                positive
                & ~missing_catalog
                & ~before_upload
                & ~same_day_unknown
                & ~no_prior_status
                & ~normal_video
            )
            not_public = (
                positive
                & ~missing_catalog
                & ~before_upload
                & ~same_day_unknown
                & ~no_prior_status
                & ~ad_video
                & ~public_video
            )
            already_seen = (
                positive
                & ~missing_catalog
                & ~before_upload
                & ~same_day_unknown
                & ~no_prior_status
                & ~ad_video
                & ~not_public
                & previously_seen
            )
            eligible_context = (
                ~missing_upload
                & ~before_declared_upload_date
                & ~same_day_upload_time_unverifiable
                & metadata_present
                & prior_status_known
                & normal_video
                & public_video
                & ~previously_seen
            )
            pre_catalog_context = (
                ~missing_upload
                & ~before_declared_upload_date
                & ~same_day_upload_time_unverifiable
                & ~previously_seen
            )
            eligible_positive = positive & eligible_context
            positive_user = frame.loc[positive, "user_id"].astype(int)
            current = state[split_name]
            current["rows"] += len(frame)
            current["positive_count"] += int(positive.sum())
            current["eligible_positive_count"] += int(eligible_positive.sum())
            current["interaction_before_declared_upload_date_count"] += int(
                (~missing_upload & before_declared_upload_date).sum()
            )
            current["interaction_same_day_upload_time_unverifiable_count"] += int(
                same_day_upload_time_unverifiable.sum()
            )
            current["positive_before_declared_upload_date_count"] += int(
                (positive & ~missing_upload & before_declared_upload_date).sum()
            )
            current["positive_same_day_upload_time_unverifiable_count"] += int(
                (positive & same_day_upload_time_unverifiable).sum()
            )
            current["positive_missing_upload_count"] += int(
                (positive & missing_upload).sum()
            )
            current["positive_previously_seen_count"] += int(
                already_seen.sum()
            )
            current["positive_missing_catalog_metadata_count"] += int(
                missing_catalog.sum()
            )
            current["positive_without_prior_status_count"] += int(
                no_prior_status.sum()
            )
            current["positive_ad_video_count"] += int(ad_video.sum())
            current["positive_not_public_count"] += int(not_public.sum())
            current["users"].update(split_user.unique().tolist())
            current["videos"].update(split_video.unique().tolist())
            add_counts(current["per_user_events"], split_user.value_counts())
            add_counts(current["per_user_positives"], positive_user.value_counts())
            if eligible_context.any():
                current["eligible_context_parts"].append(
                    frame.loc[eligible_context, columns].copy()
                )
            if pre_catalog_context.any():
                current["pre_catalog_context_parts"].append(
                    frame.loc[pre_catalog_context, columns].copy()
                )
            current["timestamp_min"] = min(
                current["timestamp_min"], float(frame["timestamp"].min())
            )
            current["timestamp_max"] = max(
                current["timestamp_max"], float(frame["timestamp"].max())
            )

    train_users = state["train"]["users"]
    train_videos = state["train"]["videos"]
    through_validation_users = train_users | state["validation"]["users"]
    through_validation_videos = train_videos | state["validation"]["videos"]

    output: dict[str, Any] = {}
    for split_name in split_order:
        current = state[split_name]
        users = current["users"]
        videos = current["videos"]
        positives = [current["per_user_positives"].get(user_id, 0) for user_id in users]
        if split_name == "train":
            reference_users: set[int] = set()
            reference_videos: set[int] = set()
        elif split_name == "validation":
            reference_users = train_users
            reference_videos = train_videos
        else:
            reference_users = through_validation_users
            reference_videos = through_validation_videos
        new_users = users - reference_users if reference_users else set()
        new_videos = videos - reference_videos if reference_videos else set()
        timestamp_min = float(current["timestamp_min"])
        timestamp_max = float(current["timestamp_max"])
        if current["eligible_context_parts"]:
            eligible_context = pd.concat(
                current["eligible_context_parts"], ignore_index=True
            )
            positive_events, duplicate_audit = canonicalize_eligible_targets(
                eligible_context
            )
            target_group_sizes = (
                positive_events.groupby(["user_id", "timestamp"], sort=False)
                .size()
                .to_numpy()
            )
        else:
            positive_events, duplicate_audit = canonicalize_eligible_targets(
                pd.DataFrame(columns=columns)
            )
            target_group_sizes = np.asarray([], dtype=np.int64)
        if current["pre_catalog_context_parts"]:
            pre_catalog_context = pd.concat(
                current["pre_catalog_context_parts"], ignore_index=True
            )
            _, pre_catalog_duplicate_audit = canonicalize_eligible_targets(
                pre_catalog_context
            )
        else:
            _, pre_catalog_duplicate_audit = canonicalize_eligible_targets(
                pd.DataFrame(columns=columns)
            )
        output[split_name] = {
            "timestamp_start_inclusive": timestamp_min,
            "timestamp_end_inclusive": timestamp_max,
            "time_start_asia_shanghai": local_timestamp(timestamp_min),
            "time_end_asia_shanghai": local_timestamp(timestamp_max),
            "rows": int(current["rows"]),
            "users": len(users),
            "videos": len(videos),
            "positive_count": int(current["positive_count"]),
            "eligible_positive_count": int(current["eligible_positive_count"]),
            "interaction_before_declared_upload_date_count": int(
                current["interaction_before_declared_upload_date_count"]
            ),
            "interaction_same_day_upload_time_unverifiable_count": int(
                current["interaction_same_day_upload_time_unverifiable_count"]
            ),
            "positive_before_declared_upload_date_count": int(
                current["positive_before_declared_upload_date_count"]
            ),
            "positive_same_day_upload_time_unverifiable_count": int(
                current["positive_same_day_upload_time_unverifiable_count"]
            ),
            "positive_missing_upload_count": int(
                current["positive_missing_upload_count"]
            ),
            "positive_previously_seen_count": int(
                current["positive_previously_seen_count"]
            ),
            "positive_missing_catalog_metadata_count": int(
                current["positive_missing_catalog_metadata_count"]
            ),
            "positive_without_prior_status_count": int(
                current["positive_without_prior_status_count"]
            ),
            "positive_ad_video_count": int(current["positive_ad_video_count"]),
            "positive_not_public_count": int(
                current["positive_not_public_count"]
            ),
            "unique_eligible_target_count": int(target_group_sizes.sum()),
            "target_deduplication_audit": duplicate_audit,
            "pre_catalog_target_deduplication_audit": pre_catalog_duplicate_audit,
            "positive_fraction": (
                current["positive_count"] / current["rows"] if current["rows"] else 0.0
            ),
            "temporal_query_count": int(target_group_sizes.size),
            "multi_target_query_count": int((target_group_sizes > 1).sum()),
            "multi_target_query_fraction": (
                float((target_group_sizes > 1).mean())
                if target_group_sizes.size
                else 0.0
            ),
            "targets_per_query": distribution(target_group_sizes),
            "positives_per_active_user": distribution(positives),
            "events_per_active_user": distribution(list(current["per_user_events"].values())),
            "new_user_count": len(new_users) if split_name != "train" else None,
            "new_user_fraction_of_split_users": (
                len(new_users) / len(users) if users and split_name != "train" else None
            ),
            "new_video_count": len(new_videos) if split_name != "train" else None,
            "new_video_fraction_of_split_videos": (
                len(new_videos) / len(videos) if videos and split_name != "train" else None
            ),
            "user_ids": sorted(users),
            "video_ids": sorted(videos),
            "per_user_events": dict(current["per_user_events"]),
            "per_user_positives": dict(current["per_user_positives"]),
            "_canonical_targets": positive_events,
        }
        if canonical_behavior_audit is not None:
            canonical_split = canonical_behavior_audit["per_frozen_split"][
                split_name
            ]
            # Keep raw target-quality counters above for transparent
            # reconciliation, but all sequence/history-facing counters below
            # come from one row per canonical event key.
            output[split_name]["raw_behavior_event_count"] = int(
                current["rows"]
            )
            output[split_name]["canonical_behavior_event_count"] = int(
                canonical_split["canonical_event_count"]
            )
            output[split_name]["behavior_event_source"] = (
                "canonical (user_id,video_id,timestamp) events"
            )
            output[split_name]["events_per_active_user"] = distribution(
                list(canonical_split["per_user_events"].values())
            )
            output[split_name]["per_user_events"] = dict(
                canonical_split["per_user_events"]
            )
    return output


def _candidate_distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    percentiles = (0, 1, 5, 10, 50, 90, 95, 99, 100)
    quantiles = np.percentile(array, percentiles)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "zero_count": int((array == 0).sum()),
        "zero_fraction": float((array == 0).mean()),
        "quantiles": {
            f"p{percentile:02d}": float(value)
            for percentile, value in zip(percentiles, quantiles, strict=True)
        },
    }


def iter_user_frames(path: Path, columns: list[str]):
    """Yield complete user frames even when a user crosses a CSV chunk edge."""

    carry = pd.DataFrame(columns=columns)
    previous_user: int | None = None
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame(columns=columns)
        user = pd.to_numeric(chunk["user_id"], errors="raise").astype(int)
        if (user.to_numpy()[1:] < user.to_numpy()[:-1]).any():
            raise RuntimeError("big_matrix must remain grouped by user_id")
        last_user = int(user.iloc[-1])
        complete = chunk.loc[user.ne(last_user)]
        carry = chunk.loc[user.eq(last_user)].copy()
        for user_id, frame in complete.groupby("user_id", sort=False):
            user_id = int(user_id)
            if previous_user is not None and user_id <= previous_user:
                raise RuntimeError("user frame iterator observed duplicate/out-of-order user")
            previous_user = user_id
            yield user_id, frame.reset_index(drop=True)
    if not carry.empty:
        user_id = int(carry["user_id"].iloc[0])
        if previous_user is not None and user_id <= previous_user:
            raise RuntimeError("user frame iterator ended out of order")
        yield user_id, carry.reset_index(drop=True)


def _split_name_for_timestamp(
    value: float, boundaries: Mapping[str, float | int]
) -> str:
    if value < float(boundaries["train_end_exclusive"]):
        return "train"
    if value < float(boundaries["validation_end_exclusive"]):
        return "validation"
    return "temporal_final"


def audit_canonical_behavior_events(
    path: Path,
    raw_boundaries: Mapping[str, float | int],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    """Audit all Big Matrix rows and summarize canonical history inputs.

    Protocol-v2.1 deliberately preserves the already disclosed protocol-v2
    cutoffs, which were selected from raw row counts. All event-consuming
    inputs use the canonical event table. We also calculate the counterfactual
    canonical-count cutoffs and quantify event reassignment rather than silently
    redefining the frozen holdout.
    """

    aggregate: Counter[str] = Counter()
    exact_weighted_parts: list[pd.DataFrame] = []
    nonexact_weighted_parts: list[pd.DataFrame] = []
    canonical_timestamp_parts: list[np.ndarray] = []
    raw_split_rows: Counter[str] = Counter()
    canonical_split_rows: Counter[str] = Counter()
    per_split: dict[str, dict[str, Any]] = {
        name: {
            "users": set(),
            "videos": set(),
            "per_user_events": Counter(),
            "per_user_strong_positives": Counter(),
        }
        for name in ("train", "validation", "temporal_final")
    }
    canonical_digest = hashlib.sha256()

    for user_id, raw_frame in iter_user_frames(path, EVENT_COLUMNS):
        raw_times = pd.to_numeric(raw_frame["timestamp"], errors="raise").to_numpy(
            dtype=np.float64
        )
        for value in raw_times:
            raw_split_rows[_split_name_for_timestamp(float(value), raw_boundaries)] += 1

        canonical, partial, exact_weighted, nonexact_weighted = (
            canonicalize_behavior_events(raw_frame)
        )
        for name, value in partial.items():
            if name.endswith("count") or name.endswith("rows") or name.endswith(
                "removed"
            ):
                aggregate[name] += int(value)
        if not exact_weighted.empty:
            exact_weighted_parts.append(exact_weighted)
        if not nonexact_weighted.empty:
            nonexact_weighted_parts.append(nonexact_weighted)

        timestamps = canonical["timestamp"].to_numpy(dtype=np.float64)
        canonical_timestamp_parts.append(timestamps)
        for row in canonical.itertuples(index=False):
            values = [getattr(row, column) for column in EVENT_COLUMNS]
            canonical_digest.update(
                json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str).encode()
            )
            canonical_digest.update(b"\n")
        for split_name in ("train", "validation", "temporal_final"):
            if split_name == "train":
                mask = timestamps < float(raw_boundaries["train_end_exclusive"])
            elif split_name == "validation":
                mask = (
                    timestamps >= float(raw_boundaries["train_end_exclusive"])
                ) & (
                    timestamps
                    < float(raw_boundaries["validation_end_exclusive"])
                )
            else:
                mask = timestamps >= float(
                    raw_boundaries["validation_end_exclusive"]
                )
            if not mask.any():
                continue
            selected = canonical.loc[mask]
            current = per_split[split_name]
            canonical_split_rows[split_name] += int(len(selected))
            current["users"].add(int(user_id))
            current["videos"].update(selected["video_id"].astype(int).tolist())
            current["per_user_events"][int(user_id)] += int(len(selected))
            strong_count = int(selected["_is_strong_positive"].sum())
            if strong_count:
                current["per_user_strong_positives"][int(user_id)] += strong_count

    canonical_timestamps = (
        np.concatenate(canonical_timestamp_parts)
        if canonical_timestamp_parts
        else np.asarray([], dtype=np.float64)
    )
    canonical_boundaries = timestamp_quantile_boundaries(
        canonical_timestamps, train_fraction, validation_fraction
    )
    frozen_assignment = np.where(
        canonical_timestamps < float(raw_boundaries["train_end_exclusive"]),
        "train",
        np.where(
            canonical_timestamps
            < float(raw_boundaries["validation_end_exclusive"]),
            "validation",
            "temporal_final",
        ),
    )
    counterfactual_assignment = np.where(
        canonical_timestamps
        < float(canonical_boundaries["train_end_exclusive"]),
        "train",
        np.where(
            canonical_timestamps
            < float(canonical_boundaries["validation_end_exclusive"]),
            "validation",
            "temporal_final",
        ),
    )
    transitions = Counter(
        f"{before}->{after}"
        for before, after in zip(
            frozen_assignment, counterfactual_assignment, strict=True
        )
    )
    frozen_assignment_counts = Counter(str(value) for value in frozen_assignment)
    counterfactual_assignment_counts = Counter(
        str(value) for value in counterfactual_assignment
    )
    changed = int((frozen_assignment != counterfactual_assignment).sum())

    empty_weighted = pd.DataFrame(columns=EVENT_KEY + ["extra_count"])
    exact_weighted_all = (
        pd.concat(exact_weighted_parts, ignore_index=True)
        if exact_weighted_parts
        else empty_weighted
    )
    nonexact_weighted_all = (
        pd.concat(nonexact_weighted_parts, ignore_index=True)
        if nonexact_weighted_parts
        else empty_weighted
    )
    split_payload: dict[str, Any] = {}
    for name, current in per_split.items():
        split_payload[name] = {
            "raw_rows_assigned": int(raw_split_rows[name]),
            "canonical_event_count": int(canonical_split_rows[name]),
            "users": len(current["users"]),
            "videos": len(current["videos"]),
            "user_ids": sorted(current["users"]),
            "video_ids": sorted(current["videos"]),
            "per_user_events": dict(current["per_user_events"]),
            "per_user_strong_positives": dict(
                current["per_user_strong_positives"]
            ),
        }

    raw_rows = int(aggregate["raw_rows"])
    canonical_count = int(aggregate["canonical_event_count"])
    reconciliation_ok = (
        raw_rows
        - int(aggregate["exact_duplicate_rows_removed"])
        - int(aggregate["same_key_nonexact_rows_removed"])
        == canonical_count
    )
    if not reconciliation_ok:
        raise RuntimeError("Global canonical behavior-event accounting failed")
    return {
        "canonical_key": EVENT_KEY,
        "exact_duplicate_definition": EVENT_COLUMNS,
        "canonical_representative_rule": (
            "remove full eight-field exact duplicates, then sort each "
            "(user_id,video_id,timestamp) key by a typed tuple of all remaining "
            "fields (numeric values numerically, text lexicographically, nulls "
            "last) and retain one; "
            "aggregate label flags across the complete key"
        ),
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": int(
            aggregate["exact_duplicate_rows_removed"]
        ),
        "exact_duplicate_full_row_group_count": int(
            aggregate["exact_duplicate_full_row_group_count"]
        ),
        "same_key_nonexact_rows_removed": int(
            aggregate["same_key_nonexact_rows_removed"]
        ),
        "same_key_nonexact_key_count": int(
            aggregate["same_key_nonexact_key_count"]
        ),
        "binary_positive_conflict_key_count": int(
            aggregate["binary_positive_conflict_key_count"]
        ),
        "quick_skip_strong_positive_conflict_key_count": int(
            aggregate["quick_skip_strong_positive_conflict_key_count"]
        ),
        "canonical_event_count": canonical_count,
        "canonical_event_sha256": canonical_digest.hexdigest(),
        "reconciliation_ok": reconciliation_ok,
        "exact_duplicate_distribution": _count_distribution_by_user_and_date(
            exact_weighted_all
        ),
        "same_key_nonexact_distribution": _count_distribution_by_user_and_date(
            nonexact_weighted_all
        ),
        "downstream_event_inputs": {
            "split_cutoffs": (
                "retain frozen protocol-v2 raw-count cutoffs; apply them to "
                "canonical events"
            ),
            "history": "canonical events strictly before query timestamp",
            "seen_filter": "canonical events strictly before query timestamp",
            "last_50": "last 50 canonical events, not raw duplicate rows",
            "quick_skip": (
                "canonical key flag; exclude any key containing both quick-skip "
                "and strong-positive evidence"
            ),
            "popularity": "count canonical events once per event key",
        },
        "boundary_sensitivity": {
            "decision": "preserve previously frozen raw-count cutoffs",
            "reason": (
                "protocol-v2 aggregate final statistics were disclosed before "
                "protocol-v2.1; moving the cutoff would silently redefine holdout"
            ),
            "frozen_raw_count_boundaries": {
                "train_end_exclusive": float(
                    raw_boundaries["train_end_exclusive"]
                ),
                "validation_end_exclusive": float(
                    raw_boundaries["validation_end_exclusive"]
                ),
            },
            "counterfactual_canonical_count_boundaries": {
                "train_end_exclusive": float(
                    canonical_boundaries["train_end_exclusive"]
                ),
                "validation_end_exclusive": float(
                    canonical_boundaries["validation_end_exclusive"]
                ),
            },
            "frozen_boundary_canonical_event_counts": {
                name: int(frozen_assignment_counts[name])
                for name in ("train", "validation", "temporal_final")
            },
            "counterfactual_boundary_canonical_event_counts": {
                name: int(counterfactual_assignment_counts[name])
                for name in ("train", "validation", "temporal_final")
            },
            "canonical_events_with_changed_assignment": changed,
            "canonical_event_changed_fraction": (
                changed / canonical_count if canonical_count else 0.0
            ),
            "assignment_transition_counts": dict(sorted(transitions.items())),
        },
        "per_frozen_split": split_payload,
        "_canonical_count_boundaries": canonical_boundaries,
    }


def audit_candidate_sizes_and_hard_negatives(
    path: Path,
    canonical_targets: dict[str, pd.DataFrame],
    catalog_policy: dict[str, Any],
    boundaries: dict[str, float | int],
    *,
    session_gap_minutes: float,
    hard_window_minutes: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay candidate sizes and audit train-only future quick-skip pools.

    This computes no scores and runs no baseline.  Future positive labels are
    consulted only for aggregate false-negative diagnostics; they never alter
    the pool or any future sampler.
    """

    query_map: dict[int, dict[float, dict[str, Any]]] = {}
    for split_name, targets in canonical_targets.items():
        for (user_id, timestamp), group in targets.groupby(
            ["user_id", "timestamp"], sort=False
        ):
            query_map.setdefault(int(user_id), {})[float(timestamp)] = {
                "split": split_name,
                "targets": set(group["video_id"].astype(int).tolist()),
            }

    sizes: dict[str, dict[str, list[int]]] = {
        split: {"available": [], "unseen": [], "uniform_pool": []}
        for split in ("train", "validation", "temporal_final")
    }
    target_missing_from_candidate: Counter[str] = Counter()
    hard_query_count = 0
    hard_queries_with_pool = 0
    hard_event_count = 0
    hard_unique_pair_count = 0
    hard_pool_sizes: list[int] = []
    false_negative_pair_counts: Counter[str] = Counter()
    false_negative_query_counts: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    gap_seconds = float(session_gap_minutes) * 60.0
    hard_window_seconds = float(hard_window_minutes) * 60.0
    train_cutoff = float(boundaries["train_end_exclusive"])
    item_index = catalog_policy["video_index"]
    catalog_video_ids = list(
        catalog_policy.get(
            "video_ids", sorted(item_index, key=lambda value: item_index[value])
        )
    )
    if catalog_video_ids != sorted(catalog_video_ids):
        raise RuntimeError("Candidate catalog video_ids must be in stable sorted order")
    membership_width_bytes = (len(catalog_video_ids) + 7) // 8
    membership_digests = {
        split: hashlib.sha256()
        for split in ("train", "validation", "temporal_final")
    }
    membership_query_counts: Counter[str] = Counter()
    risk_names = (
        "remaining_session",
        "within_1d",
        "within_7d",
        "before_fit_cutoff",
    )

    for user_id, raw_frame in iter_user_frames(path, EVENT_COLUMNS):
        user_queries = query_map.get(user_id)
        if not user_queries:
            continue
        frame, _, _, _ = canonicalize_behavior_events(raw_frame)
        video_values = pd.to_numeric(
            frame["video_id"], errors="raise"
        ).to_numpy(dtype=np.int64)
        timestamps = pd.to_numeric(
            frame["timestamp"], errors="raise"
        ).to_numpy(dtype=np.float64)
        strong_values = frame["_is_strong_positive"].to_numpy(dtype=bool)
        quick_skip_values = frame["_is_quick_skip"].to_numpy(dtype=bool)
        quick_strong_conflicts = frame[
            "_quick_skip_strong_conflict"
        ].to_numpy(dtype=bool)
        if timestamps.size > 1 and (timestamps[1:] < timestamps[:-1]).any():
            raise RuntimeError("User timestamps must remain nondecreasing")

        # Compute sessions at array speed. Equal timestamps and a gap of exactly
        # 30 minutes remain in the same session; only a strictly larger gap cuts.
        starts_new_session = np.zeros(timestamps.size, dtype=np.int64)
        if timestamps.size > 1:
            starts_new_session[1:] = (
                np.diff(timestamps) > gap_seconds
            ).astype(np.int64)
        session_ids = np.cumsum(starts_new_session)
        session_end_by_id: dict[int, float] = {}
        for session_id, event_time in zip(session_ids, timestamps, strict=True):
            session_end_by_id[int(session_id)] = float(event_time)

        first_time_by_item: dict[int, float] = {}
        for video_id, event_time in zip(video_values, timestamps, strict=True):
            first_time_by_item.setdefault(int(video_id), float(event_time))
        strong_times_by_item: dict[int, list[float]] = {}
        for video_id, event_time in zip(
            video_values[strong_values & (timestamps < train_cutoff)],
            timestamps[strong_values & (timestamps < train_cutoff)],
            strict=True,
        ):
            strong_times_by_item.setdefault(int(video_id), []).append(
                float(event_time)
            )

        position_values = np.fromiter(
            (item_index.get(int(video_id), -1) for video_id in video_values),
            dtype=np.int64,
            count=len(video_values),
        )
        seen_mask = 0
        history_cursor = 0
        for query_time, query in sorted(user_queries.items()):
            query_time = float(query_time)
            query_left = int(np.searchsorted(timestamps, query_time, side="left"))
            if query_left >= len(timestamps) or timestamps[query_left] != query_time:
                raise RuntimeError(
                    "Canonical query timestamp is absent from canonical history"
                )

            # Consume every event strictly before the query. The current
            # timestamp group remains unseen until a later query.
            if query_left > history_cursor:
                prior_positions = position_values[history_cursor:query_left]
                for position in np.unique(prior_positions[prior_positions >= 0]):
                    seen_mask |= 1 << int(position)
                history_cursor = query_left

            query_date = timestamp_to_local_date(query_time)
            state = catalog_policy["state_by_date"].get(query_date)
            candidate_mask = int(state["eligible_mask"]) if state else 0
            available_count = candidate_mask.bit_count()
            unseen_mask = candidate_mask & ~seen_mask
            unseen_count = unseen_mask.bit_count()
            target_group = sorted(int(target) for target in query["targets"])
            identity = (
                f"membership-bitset-v1|{query['split']}|{user_id}|"
                f"{query_time:.6f}|{','.join(map(str, target_group))}|"
            ).encode()
            membership_digest = membership_digests[query["split"]]
            membership_digest.update(identity)
            membership_digest.update(
                unseen_mask.to_bytes(
                    membership_width_bytes, byteorder="little", signed=False
                )
            )
            membership_digest.update(b"\n")
            membership_query_counts[query["split"]] += 1
            target_count_in_pool = 0
            for target in query["targets"]:
                position = item_index.get(int(target))
                if position is not None and ((unseen_mask >> position) & 1):
                    target_count_in_pool += 1
                else:
                    target_missing_from_candidate[query["split"]] += 1
            uniform_pool_count = unseen_count - target_count_in_pool
            if uniform_pool_count < 0:
                raise RuntimeError("Uniform negative pool size became negative")
            split_sizes = sizes[query["split"]]
            split_sizes["available"].append(available_count)
            split_sizes["unseen"].append(unseen_count)
            split_sizes["uniform_pool"].append(uniform_pool_count)

            if query["split"] != "train":
                continue
            hard_query_count += 1
            left = int(np.searchsorted(timestamps, query_time, side="right"))
            right = min(
                int(
                    np.searchsorted(
                        timestamps,
                        query_time + hard_window_seconds,
                        side="right",
                    )
                ),
                int(np.searchsorted(timestamps, train_cutoff, side="left")),
            )
            query_session = int(session_ids[query_left])
            window = np.arange(left, right, dtype=np.int64)
            if window.size:
                same_session = session_ids[window] == query_session
                skipped_reasons["outside_same_session"] += int(
                    window.size - same_session.sum()
                )
                window = window[same_session]
            if window.size:
                conflict = quick_strong_conflicts[window]
                skipped_reasons["quick_skip_strong_positive_conflict"] += int(
                    conflict.sum()
                )
                window = window[~conflict]
            if window.size:
                quick_skip = quick_skip_values[window]
                skipped_reasons["not_quick_skip"] += int(
                    window.size - quick_skip.sum()
                )
                window = window[quick_skip]
            hard_event_count += int(window.size)

            first_event_by_item: dict[int, float] = {}
            for row_position in window:
                video_id = int(video_values[row_position])
                event_time = float(timestamps[row_position])
                catalog_position = int(position_values[row_position])
                if catalog_position < 0 or not (
                    (candidate_mask >> catalog_position) & 1
                ):
                    skipped_reasons["not_candidate_at_query"] += 1
                    continue
                if first_time_by_item[video_id] < query_time:
                    skipped_reasons["seen_at_query"] += 1
                    continue
                if video_id in query["targets"]:
                    skipped_reasons["current_target"] += 1
                    continue
                first_event_by_item.setdefault(video_id, event_time)

            hard_unique_pair_count += len(first_event_by_item)
            hard_pool_sizes.append(len(first_event_by_item))
            if first_event_by_item:
                hard_queries_with_pool += 1
            query_risks = {name: False for name in risk_names}
            session_end = min(
                session_end_by_id.get(query_session, query_time), train_cutoff
            )
            for video_id, hard_time in first_event_by_item.items():
                later = strong_times_by_item.get(video_id, [])
                bounds = {
                    "remaining_session": session_end,
                    "within_1d": min(query_time + 86400.0, train_cutoff),
                    "within_7d": min(query_time + 7 * 86400.0, train_cutoff),
                    "before_fit_cutoff": train_cutoff,
                }
                start = bisect.bisect_right(later, hard_time)
                for name, bound in bounds.items():
                    later_timestamp = later[start] if start < len(later) else None
                    within_bound = (
                        later_timestamp is not None
                        and later_timestamp < train_cutoff
                        and (
                            later_timestamp < bound
                            if bound == train_cutoff
                            else later_timestamp <= bound
                        )
                    )
                    if within_bound:
                        false_negative_pair_counts[name] += 1
                        query_risks[name] = True
            for name, has_risk in query_risks.items():
                if has_risk:
                    false_negative_query_counts[name] += 1

    candidate_output = {
        "rule": (
            "per-query causal NORMAL/public/uploaded catalog; remove all items "
            "observed strictly before query; score before consuming equal timestamp"
        ),
        "per_split": {
            split: {
                "available_candidate_size": _candidate_distribution(
                    values["available"]
                ),
                "available_unseen_candidate_size": _candidate_distribution(
                    values["unseen"]
                ),
                "uniform_pool_excluding_targets_size": _candidate_distribution(
                    values["uniform_pool"]
                ),
                "target_missing_from_candidate_count": int(
                    target_missing_from_candidate[split]
                ),
                "candidate_membership_sha256": membership_digests[
                    split
                ].hexdigest(),
                "candidate_membership_query_count": int(
                    membership_query_counts[split]
                ),
            }
            for split, values in sizes.items()
        },
        "membership_hash_format": {
            "algorithm": "sha256",
            "version": "membership-bitset-v1",
            "query_order": "ascending (user_id, timestamp) over canonical target groups",
            "query_identity": (
                "split|user_id|timestamp formatted to 6 decimals|sorted target IDs"
            ),
            "candidate_membership": (
                "fixed-width little-endian bitset over globally ascending video_ids; "
                "causal NORMAL/public/uploaded and unseen strictly before query; "
                "current target group remains included"
            ),
            "catalog_video_id_order_sha256": _stable_int_set_hash(
                set(catalog_video_ids)
            ),
            "bitset_width_bytes": membership_width_bytes,
        },
    }
    pair_denominator = hard_unique_pair_count
    hard_output = {
        "status": "aggregate_pool_audit_only_no_sampler_or_model_executed",
        "fit_context": "selection_train_only",
        "future_window": (
            "query_time < event_time <= query_time+30m and event_time < "
            "train_end; same session; never reads validation for a train query"
        ),
        "later_positive_diagnostic_cutoff": (
            "every later positive must satisfy later_timestamp < "
            "train_end_exclusive, including remaining_session"
        ),
        "seen_definition": "first canonical event timestamp < query_time",
        "quick_skip_conflict_policy": (
            "exclude a (user,item,timestamp) key if any row is quick-skip and "
            "any row is strong-positive"
        ),
        "query_count": hard_query_count,
        "queries_with_nonempty_pool": hard_queries_with_pool,
        "query_pool_coverage": (
            hard_queries_with_pool / hard_query_count if hard_query_count else 0.0
        ),
        "same_session_window_quick_skip_event_count_before_candidate_filters": hard_event_count,
        "unique_hard_query_item_pairs": hard_unique_pair_count,
        "pool_size_per_query": _candidate_distribution(hard_pool_sizes),
        "filter_counts": dict(sorted(skipped_reasons.items())),
        "false_negative_risk": {
            "diagnostic_only_not_used_to_filter_or_tune": True,
            "pair_denominator": pair_denominator,
            "query_denominator": hard_queries_with_pool,
            "pair_counts": {
                name: int(false_negative_pair_counts[name]) for name in risk_names
            },
            "pair_fractions": {
                name: false_negative_pair_counts[name] / pair_denominator
                if pair_denominator
                else 0.0
                for name in risk_names
            },
            "query_any_risk_counts": {
                name: int(false_negative_query_counts[name]) for name in risk_names
            },
            "query_any_risk_fractions": {
                name: false_negative_query_counts[name] / hard_queries_with_pool
                if hard_queries_with_pool
                else 0.0
                for name in risk_names
            },
        },
    }
    return candidate_output, hard_output


def metadata_coverage(
    files: dict[str, Path], catalog_items: set[int]
) -> dict[str, Any]:
    categories = pd.read_csv(files["item_categories.csv"])
    captions = pd.read_csv(
        files["kuairec_caption_category.csv"], lineterminator="\n"
    )
    daily = pd.read_csv(
        files["item_daily_features.csv"],
        usecols=["video_id", "upload_dt", "author_id", "video_duration"],
        low_memory=False,
    )

    def covered_ids(frame: pd.DataFrame, column: str) -> set[int]:
        value = frame[column]
        present = value.notna() & value.astype(str).str.strip().ne("")
        return set(frame.loc[present, "video_id"].astype(int).unique().tolist())

    tag_present = categories["feat"].notna() & ~categories["feat"].astype(str).isin(
        ["", "[]"]
    )
    tag_ids = set(categories.loc[tag_present, "video_id"].astype(int).tolist())
    caption_ids = covered_ids(captions, "caption")
    cover_text_ids = covered_ids(captions, "manual_cover_text")
    topic_tag_ids = covered_ids(captions, "topic_tag")
    first_category_ids = covered_ids(captions, "first_level_category_id")
    second_category_ids = covered_ids(captions, "second_level_category_id")
    third_category_ids = covered_ids(captions, "third_level_category_id")
    upload_ids = covered_ids(daily, "upload_dt")

    def row(ids: set[int]) -> dict[str, Any]:
        covered = len(ids & catalog_items)
        return {
            "covered_videos": covered,
            "catalog_videos": len(catalog_items),
            "coverage": covered / len(catalog_items) if catalog_items else 0.0,
        }

    upload_values = pd.to_datetime(daily["upload_dt"], errors="coerce")
    return {
        "catalog_definition": "union of video_id in big_matrix and small_matrix",
        "item_categories_feat": row(tag_ids),
        "caption": row(caption_ids),
        "manual_cover_text": row(cover_text_ids),
        "topic_tag": row(topic_tag_ids),
        "first_level_category": row(first_category_ids),
        "second_level_category": row(second_category_ids),
        "third_level_category": row(third_category_ids),
        "upload_time": row(upload_ids),
        "upload_time_min": (
            upload_values.min().isoformat() if upload_values.notna().any() else None
        ),
        "upload_time_max": (
            upload_values.max().isoformat() if upload_values.notna().any() else None
        ),
    }


def history_distributions(
    split_stats: dict[str, Any], small_users: set[int]
) -> dict[str, Any]:
    train_counts = Counter(
        {int(key): int(value) for key, value in split_stats["train"]["per_user_events"].items()}
    )
    validation_counts = Counter(
        {
            int(key): int(value)
            for key, value in split_stats["validation"]["per_user_events"].items()
        }
    )
    train_validation_counts = train_counts + validation_counts
    validation_users = set(split_stats["validation"]["user_ids"])
    final_users = set(split_stats["temporal_final"]["user_ids"])
    return {
        "validation_history_from_train": distribution(
            [train_counts.get(user, 0) for user in validation_users]
        ),
        "temporal_final_history_from_train_and_validation": distribution(
            [train_validation_counts.get(user, 0) for user in final_users]
        ),
        "small_audit_history_from_big_first_85_percent": distribution(
            [train_validation_counts.get(user, 0) for user in small_users]
        ),
    }


def cold_start_context_audit(
    split_stats: dict[str, Any],
    canonical_targets: dict[str, pd.DataFrame],
    small_catalog_items: set[int],
) -> dict[str, Any]:
    train_items = set(int(value) for value in split_stats["train"]["video_ids"])
    through_validation_items = train_items | set(
        int(value) for value in split_stats["validation"]["video_ids"]
    )

    def context_row(
        reference_splits: list[str], reference_items: set[int], target_items: set[int]
    ) -> dict[str, Any]:
        cold = target_items - reference_items
        warm = target_items & reference_items
        return {
            "reference_splits": reference_splits,
            "reference_item_count": len(reference_items),
            "reference_membership_sha256": _stable_int_set_hash(reference_items),
            "target_item_count": len(target_items),
            "warm_target_item_count": len(warm),
            "cold_target_item_count": len(cold),
            "cold_target_membership_sha256": _stable_int_set_hash(cold),
            "data_warm_definition": (
                "at least one Big Matrix interaction in the context reference"
            ),
            "id_embedding_actually_trained_is_separate": True,
        }

    validation_targets = set(
        canonical_targets["validation"]["video_id"].astype(int).tolist()
    )
    final_targets = set(
        canonical_targets["temporal_final"]["video_id"].astype(int).tolist()
    )
    return {
        "validation": context_row(["train"], train_items, validation_targets),
        "temporal_final": context_row(
            ["train", "validation"], through_validation_items, final_targets
        ),
        "small_matrix": context_row(
            ["train", "validation"],
            through_validation_items,
            small_catalog_items,
        ),
        "untrained_id_policy": (
            "an ID embedding may be used only if optimization actually touched it; "
            "otherwise use the content-only/UNK fallback"
        ),
    }


def baseline_cost_estimates(
    split_stats: dict[str, Any],
    small_primary_pairs: int,
    small_secondary_pairs: int,
    big_video_count: int,
) -> dict[str, Any]:
    train_positive = split_stats["train"]["unique_eligible_target_count"]
    validation_queries = split_stats["validation"]["temporal_query_count"]
    dense_validation_upper = validation_queries * big_video_count
    bpr_epochs = 10
    return {
        "disclaimer": (
            "Planning estimates only; no baseline was executed in Phase 0. "
            "Temporal query count uses exact (user_id, next-positive timestamp) groups."
        ),
        "validation_temporal_queries": validation_queries,
        "big_candidate_catalog_upper_bound": big_video_count,
        "validation_dense_score_pair_upper_bound": dense_validation_upper,
        "small_matrix_primary_observed_pairs": small_primary_pairs,
        "small_matrix_secondary_full_ranking_pairs": small_secondary_pairs,
        "canonical_train_targets": int(train_positive),
        "bpr_reference_epochs": bpr_epochs,
        "bpr_positive_updates_before_batching": int(train_positive * bpr_epochs),
        "baselines": {
            "random": {
                "fit_scale": "none",
                "evaluation_scale": f"up to {dense_validation_upper:,} candidate pairs",
                "expected_compute": "under 5 CPU minutes with direct seeded top-K sampling",
            },
            "global_popularity": {
                "fit_scale": (
                    f"one pass over {train_positive:,} canonical train targets"
                ),
                "evaluation_scale": "one shared ranking plus per-user seen filtering",
                "expected_compute": "roughly 1-5 CPU minutes",
            },
            "time_decayed_popularity": {
                "fit_scale": (
                    f"one causal chronological stream over {train_positive:,} "
                    "canonical train targets"
                ),
                "evaluation_scale": "state update plus shared ranking at query times",
                "expected_compute": "roughly 3-15 CPU minutes",
            },
            "itemcf": {
                "fit_scale": (
                    f"sparse co-occurrence from {train_positive:,} strong positives "
                    f"over at most {big_video_count:,} videos"
                ),
                "evaluation_scale": "history-neighbor aggregation for every temporal query",
                "expected_compute": "roughly 10-60 CPU minutes and 1-4 GB working memory",
            },
            "bpr_mf": {
                "fit_scale": (
                    f"pre-registered 10 epochs = about {train_positive * bpr_epochs:,} "
                    "positive-pair updates before batching"
                ),
                "evaluation_scale": f"up to {dense_validation_upper:,} dot products",
                "expected_compute": "roughly 30-120 CPU minutes or 5-20 GPU minutes",
            },
        },
    }


def strip_heavy_internal_fields(data: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(data))
    for file_stats in copied.get("files", {}).values():
        interaction = file_stats.get("interaction")
        if interaction:
            for field in (
                "user_ids",
                "video_ids",
                "per_user_events",
                "per_user_positives",
            ):
                interaction.pop(field, None)
    for split in copied.get("splits", {}).values():
        for field in ("user_ids", "video_ids", "per_user_events", "per_user_positives"):
            split.pop(field, None)
    behavior = copied.get("big_matrix_behavior_event_audit")
    if isinstance(behavior, dict):
        behavior.pop("_canonical_count_boundaries", None)
        for split in behavior.get("per_frozen_split", {}).values():
            for field in (
                "user_ids",
                "video_ids",
                "per_user_events",
                "per_user_strong_positives",
            ):
                split.pop(field, None)
    return copied


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite immutable manifest {path}. "
            "Delete it only as an explicitly reviewed protocol change."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def ensure_phase0_outputs_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Phase 0 outputs are immutable as one protocol bundle. Refusing to "
            f"scan or overwrite existing outputs: {existing}"
        )


def archive_protocol_v1_bundle(
    paths: list[Path], archive_root: Path
) -> dict[str, Any]:
    """Archive a complete schema-v1 bundle before the reviewed v2 migration."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Protocol-v1 supersession requires a complete existing bundle; "
            f"missing: {missing}"
        )
    manifest_path = next(path for path in paths if path.name == "split_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Only the reviewed schema-v1 bundle may be superseded")
    archive_root.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    for path in paths:
        destination = archive_root / path.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise RuntimeError(
                    f"Existing protocol-v1 archive differs from {path}: {destination}"
                )
        else:
            shutil.copy2(path, destination)
        archived[str(path)] = sha256_file(path)
    for path in paths:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.unlink()
    return {
        "parent_manifest_sha256": archived[str(manifest_path)],
        "archive_directory": str(archive_root),
        "archived_file_sha256": archived,
    }


def archive_protocol_v2_bundle(
    paths: list[Path], archive_root: Path
) -> dict[str, Any]:
    """Archive the complete immutable protocol-v2 bundle before v2.1."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Protocol-v2 supersession requires a complete existing bundle; "
            f"missing: {missing}"
        )
    manifest_path = next(path for path in paths if path.name == "split_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_revision") != "protocol-v2":
        raise RuntimeError("Only the reviewed protocol-v2 bundle may be superseded")
    archive_root.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    for path in paths:
        destination = archive_root / path.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise RuntimeError(
                    f"Existing protocol-v2 archive differs from {path}: {destination}"
                )
        else:
            shutil.copy2(path, destination)
        archived[str(path)] = sha256_file(path)
    for path in paths:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.unlink()
    return {
        "parent_manifest_sha256": archived[str(manifest_path)],
        "archive_directory": str(archive_root),
        "archived_file_sha256": archived,
    }


def markdown_report(audit: dict[str, Any]) -> str:
    lines = [
        "# KuaiRec Phase 0 Audit",
        "",
        f"Generated: `{audit['generated_at_utc']}`",
        "",
        "> No model or baseline was trained or evaluated in Phase 0.",
        "",
        "## Locked label",
        "",
        "```text",
        "watch_ratio > 2.0",
        "```",
        "",
        "The threshold was not changed in response to these statistics.",
        "",
        "## Interaction and label summary",
        "",
        "| source/split | rows | users | videos | positives | positive rate | users with zero positives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    big = audit["files"]["big_matrix.csv"]["interaction"]
    small = audit["files"]["small_matrix.csv"]["interaction"]
    for name, stats in audit["splits"].items():
        zero = stats["positives_per_active_user"]["zero_fraction"]
        lines.append(
            f"| `{name}` | {stats['rows']:,} | {stats['users']:,} | "
            f"{stats['videos']:,} | {stats['positive_count']:,} | "
            f"{stats['positive_fraction']:.4%} | {zero:.4%} |"
        )
    small_zero = small["positives_per_user"]["zero_fraction"]
    lines.append(
        f"| `small_matrix_audit` | {audit['files']['small_matrix.csv']['rows']:,} | "
        f"{small['users']:,} | {small['videos']:,} | {small['positive_count']:,} | "
        f"{small['positive_fraction']:.4%} | {small_zero:.4%} |"
    )
    lines.extend(
        [
            "",
            "### Time ranges",
            "",
            f"- Big Matrix (Asia/Shanghai): `{big['timestamp_min_asia_shanghai']}` "
            f"to `{big['timestamp_max_asia_shanghai']}`",
            f"- Small Matrix (Asia/Shanghai): `{small['timestamp_min_asia_shanghai']}` "
            f"to `{small['timestamp_max_asia_shanghai']}`",
            "",
            "The raw `date` column is not used for splitting because it disagrees",
            f"with the localized timestamp on {big['date_field_consistency']['mismatch_count']:,} "
            "Big Matrix rows.",
            "",
            "### Watch-ratio buckets",
            "",
            "| source | bucket | count |",
            "|---|---|---:|",
        ]
    )
    for source_name, source in (("big_matrix", big), ("small_matrix", small)):
        for bucket, count in source["watch_ratio"]["buckets"].items():
            lines.append(f"| `{source_name}` | `{bucket}` | {count:,} |")

    behavior = audit["big_matrix_behavior_event_audit"]
    sensitivity = behavior["boundary_sensitivity"]
    lines.extend(
        [
            "",
            "## Canonical Big Matrix behavior events",
            "",
            f"- Raw rows: {behavior['raw_rows']:,}; canonical events: "
            f"{behavior['canonical_event_count']:,}.",
            f"- Full eight-field exact duplicate extras removed: "
            f"{behavior['exact_duplicate_rows_removed']:,} across "
            f"{behavior['exact_duplicate_full_row_group_count']:,} groups.",
            f"- Non-identical rows sharing `(user,item,timestamp)` removed: "
            f"{behavior['same_key_nonexact_rows_removed']:,} across "
            f"{behavior['same_key_nonexact_key_count']:,} keys.",
            f"- Quick-skip/strong-positive conflict keys excluded from hard negatives: "
            f"{behavior['quick_skip_strong_positive_conflict_key_count']:,}.",
            "- Split cutoffs remain the frozen protocol-v2 raw-count cutoffs, while "
            "history, last-50, seen filtering, quick-skip pools, and popularity use "
            "canonical events.",
            f"- Counterfactual canonical-count cutoffs would reassign "
            f"{sensitivity['canonical_events_with_changed_assignment']:,} canonical "
            f"events ({sensitivity['canonical_event_changed_fraction']:.6%}); the "
            "holdout was not silently redefined.",
            "- Exact and non-exact duplicate extras are reported by affected user and "
            "Asia/Shanghai date in `audit.json`.",
        ]
    )

    lines.extend(["", "## Per-user distributions", ""])
    for name, stats in audit["splits"].items():
        pos = stats["positives_per_active_user"]
        event = stats["events_per_active_user"]
        lines.append(
            f"- **{name}**: events p50/p90/p99 = "
            f"{event['quantiles']['p50']:.0f}/{event['quantiles']['p90']:.0f}/"
            f"{event['quantiles']['p99']:.0f}; positives p50/p90/p99 = "
            f"{pos['quantiles']['p50']:.0f}/{pos['quantiles']['p90']:.0f}/"
            f"{pos['quantiles']['p99']:.0f}."
        )
    small_pos = small["positives_per_user"]
    lines.append(
        "- **small_matrix_audit**: positives p50/p90/p99 = "
        f"{small_pos['quantiles']['p50']:.0f}/"
        f"{small_pos['quantiles']['p90']:.0f}/"
        f"{small_pos['quantiles']['p99']:.0f}."
    )
    lines.extend(["", "### History available at evaluation time", ""])
    for name, stats in audit["history_distributions"].items():
        q = stats["quantiles"]
        lines.append(
            f"- `{name}`: p50/p90/p99 = {q['p50']:.0f}/{q['p90']:.0f}/"
            f"{q['p99']:.0f}; zero-history users = {stats['zero_fraction']:.4%}."
        )

    lines.extend(
        [
            "",
            "## Metadata coverage",
            "",
            "| feature | covered videos | catalog videos | coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    coverage = audit["metadata_coverage"]
    for feature, stats in coverage.items():
        if not isinstance(stats, dict) or "coverage" not in stats:
            continue
        lines.append(
            f"| `{feature}` | {stats['covered_videos']:,} | "
            f"{stats['catalog_videos']:,} | {stats['coverage']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## New users and videos by temporal split",
            "",
            "| split | new users | fraction | new videos | fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("validation", "temporal_final"):
        stats = audit["splits"][name]
        lines.append(
            f"| `{name}` | {stats['new_user_count']:,} | "
            f"{stats['new_user_fraction_of_split_users']:.4%} | "
            f"{stats['new_video_count']:,} | "
            f"{stats['new_video_fraction_of_split_videos']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## Evaluation contracts",
            "",
            "- Event canonicalization: `contracts/event_canonicalization_v1.yaml`",
            "- Temporal: `contracts/temporal_evaluation_v2.yaml`",
            "- Small Matrix: `contracts/fully_observed_audit_v2.yaml`",
            "- Fit contexts: `contracts/fit_contexts_v1.yaml`",
            "- Candidate catalog: `contracts/candidate_catalog_v1.yaml`",
            "- Target deduplication: `contracts/target_deduplication_v1.yaml`",
            "- Metrics: `contracts/metrics_v1.yaml`",
            "- Baselines: `contracts/baselines_v1.yaml`",
            "- Negative sampling: `contracts/negative_sampling_v2.yaml`",
            "- Cold-item fallback: `contracts/two_tower_cold_start_v2.yaml`",
            "",
            "The temporal final split is not claimed to be untouched. It is frozen",
            "after this Phase 0 aggregate audit; no ranking metric was computed.",
            "",
            "Equal-timestamp strong positives are one multi-target query with shared",
            "history ending strictly before that timestamp.",
            "",
            "A target must also be unseen before its query timestamp and certainly uploaded.",
            "Because `upload_dt` has date precision only, an item becomes eligible at the",
            "next Asia/Shanghai midnight; same-day events are excluded as unverifiable.",
            "The following exclusion counts are event-level. Formal targets then use",
            "the locked 8-field duplicate/conflict rule shown below.",
            "",
            "| split | raw positives | eligible rows | canonical targets | before date | same-day unknown | no prior status | AD | not public | previously seen |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in audit["splits"].items():
        lines.append(
            f"| `{name}` | {stats['positive_count']:,} | "
            f"{stats['eligible_positive_count']:,} | "
            f"{stats['unique_eligible_target_count']:,} | "
            f"{stats['positive_before_declared_upload_date_count']:,} | "
            f"{stats['positive_same_day_upload_time_unverifiable_count']:,} | "
            f"{stats['positive_without_prior_status_count']:,} | "
            f"{stats['positive_ad_video_count']:,} | "
            f"{stats['positive_not_public_count']:,} | "
            f"{stats['positive_previously_seen_count']:,} |"
        )
    lines.extend(
        [
            "",
            "### Pre-catalog duplicate anomaly reconciliation",
            "",
            "This reproduces the original upload/unseen eligible-event stage before",
            "the new NORMAL/public catalog filter, including the reported final gap.",
            "",
            "| split | eligible positive rows | exact duplicate extras | same-key nonexact extras | binary-conflict keys | canonical keys |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in audit["splits"].items():
        dedup = stats["pre_catalog_target_deduplication_audit"]
        lines.append(
            f"| `{name}` | {dedup['raw_eligible_positive_rows']:,} | "
            f"{dedup['exact_duplicate_positive_rows_removed']:,} | "
            f"{dedup['same_key_nonexact_positive_rows_removed']:,} | "
            f"{dedup['binary_label_conflict_keys_excluded']:,} | "
            f"{dedup['canonical_target_count']:,} |"
        )
        difference = dedup["difference_distribution"]
        if difference["affected_users"]:
            user_q = difference["extras_per_affected_user"]["quantiles"]
            top_date, top_date_count = max(
                difference["by_asia_shanghai_date"].items(),
                key=lambda item: (item[1], item[0]),
            )
            lines.append(
                f"  - `{name}`: {difference['affected_users']:,} affected users; "
                f"extras/user p50/p90/p99/max={user_q['p50']:.0f}/"
                f"{user_q['p90']:.0f}/{user_q['p99']:.0f}/{user_q['p100']:.0f}; "
                f"largest date `{top_date}`={top_date_count:,}."
            )
    lines.extend(
        [
            "",
            "### Formal catalog-eligible target reconciliation",
            "",
            "| split | eligible positive rows | exact duplicate extras | same-key nonexact extras | binary-conflict keys | positives excluded by conflict | canonical targets |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in audit["splits"].items():
        dedup = stats["target_deduplication_audit"]
        lines.append(
            f"| `{name}` | {dedup['raw_eligible_positive_rows']:,} | "
            f"{dedup['exact_duplicate_positive_rows_removed']:,} | "
            f"{dedup['same_key_nonexact_positive_rows_removed']:,} | "
            f"{dedup['binary_label_conflict_keys_excluded']:,} | "
            f"{dedup['positive_rows_in_conflict_keys_excluded']:,} | "
            f"{dedup['canonical_target_count']:,} |"
        )
        difference = dedup["difference_distribution"]
        if difference["affected_users"]:
            user_q = difference["extras_per_affected_user"]["quantiles"]
            top_date, top_date_count = max(
                difference["by_asia_shanghai_date"].items(),
                key=lambda item: (item[1], item[0]),
            )
            lines.append(
                f"  - `{name}` duplicate/conflict extras affect "
                f"{difference['affected_users']:,} users; extras/user "
                f"p50/p90/p99/max={user_q['p50']:.0f}/{user_q['p90']:.0f}/"
                f"{user_q['p99']:.0f}/{user_q['p100']:.0f}; largest date is "
                f"`{top_date}` with {top_date_count:,}. Full user/date counts are in audit.json."
            )
    lines.extend(
        [
            "",
            "| split | temporal queries | multi-target queries | fraction | max targets |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, stats in audit["splits"].items():
        lines.append(
            f"| `{name}` | {stats['temporal_query_count']:,} | "
            f"{stats['multi_target_query_count']:,} | "
            f"{stats['multi_target_query_fraction']:.6%} | "
            f"{stats['targets_per_query']['quantiles']['p100']:.0f} |"
        )
    small_coverage = audit["small_matrix_observation_coverage"]
    missing_quantiles = small_coverage["missing_pairs_per_user"]["quantiles"]
    lines.extend(
        [
            "",
            "### Small Matrix observation coverage",
            "",
            f"- Full ranking catalog: {small_coverage['catalog_videos']:,} videos for "
            f"each of {small_coverage['users']:,} users "
            f"({small_coverage['expected_complete_pairs']:,} scored pairs).",
            f"- Observed feedback pairs: {small_coverage['observed_unique_pairs']:,} "
            f"({small_coverage['observed_pair_fraction']:.4%}); blocked/missing pairs: "
            f"{small_coverage['missing_pairs']:,}.",
            f"- Missing pairs per user p50/p90/p99/max: "
            f"{missing_quantiles['p50']:.0f}/{missing_quantiles['p90']:.0f}/"
            f"{missing_quantiles['p99']:.0f}/{missing_quantiles['p100']:.0f}.",
            "- Officially, missing pairs represent videos/authors blocked by that user.",
            "- Primary audit removes each user's blocked/missing pairs. Full 3,327-item "
            "ranking is secondary only and must report `Blocked@K` and user hit rate.",
            "- Blocked information never enters training, history, features, negative "
            "sampling, or hyperparameter selection.",
        ]
    )
    catalog = audit["candidate_catalog_audit"]
    lines.extend(
        [
            "",
            "## Causal candidate catalog audit",
            "",
            f"- Primary type: `NORMAL`; excluded `AD` videos: "
            f"{catalog['video_type_video_counts'].get('AD', 0):,}.",
            "- Visibility at query date D uses the latest snapshot with `date < D`; "
            "same-day or missing status is not treated as visible.",
            f"- Per-video inconsistent `upload_dt`: "
            f"{catalog['per_video_inconsistent_fields']['upload_dt']:,}; inconsistent "
            f"`video_type`: {catalog['per_video_inconsistent_fields']['video_type']:,}.",
            f"- Visible-status transitions: `{json.dumps(catalog['visible_status_transition_counts'], sort_keys=True)}`.",
            "",
            "### Per-query candidate-size distributions",
            "",
            "| split | queries | available p50/p90/p99 | unseen p50/p90/p99 | uniform pool p50/p90/p99 | missing targets |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, values in audit["candidate_size_audit"]["per_split"].items():
        available = values["available_candidate_size"]
        unseen = values["available_unseen_candidate_size"]
        uniform = values["uniform_pool_excluding_targets_size"]
        lines.append(
            f"| `{split}` | {available['count']:,} | "
            f"{available['quantiles']['p50']:.0f}/{available['quantiles']['p90']:.0f}/{available['quantiles']['p99']:.0f} | "
            f"{unseen['quantiles']['p50']:.0f}/{unseen['quantiles']['p90']:.0f}/{unseen['quantiles']['p99']:.0f} | "
            f"{uniform['quantiles']['p50']:.0f}/{uniform['quantiles']['p90']:.0f}/{uniform['quantiles']['p99']:.0f} | "
            f"{values['target_missing_from_candidate_count']:,} |"
        )
    hard = audit["hard_negative_pool_audit"]
    hard_sizes = hard["pool_size_per_query"]
    lines.extend(
        [
            "",
            "### Train hard-negative pool audit",
            "",
            f"- Nonempty-pool query coverage: {hard['query_pool_coverage']:.4%} "
            f"({hard['queries_with_nonempty_pool']:,}/{hard['query_count']:,}).",
            f"- Deduplicated hard `(query,item)` pairs: "
            f"{hard['unique_hard_query_item_pairs']:,}; pool p50/p90/p99: "
            f"{hard_sizes['quantiles']['p50']:.0f}/{hard_sizes['quantiles']['p90']:.0f}/"
            f"{hard_sizes['quantiles']['p99']:.0f}.",
            "- Future positive labels are used only for aggregate false-negative risk "
            "diagnostics; they do not filter samples or tune the sampler.",
            f"- Pair-level risk fractions: `{json.dumps(hard['false_negative_risk']['pair_fractions'], sort_keys=True)}`.",
        ]
    )
    caption_note = audit["files"]["kuairec_caption_category.csv"]["parser_note"]
    small_time_field = next(
        field
        for field in audit["files"]["small_matrix.csv"]["fields"]
        if field["name"] == "time"
    )
    lines.extend(
        [
            "",
            "## Data quality findings",
            "",
            f"- Big Matrix raw `date` disagrees with localized `timestamp` on "
            f"{big['date_field_consistency']['mismatch_count']:,} rows "
            f"({big['date_field_consistency']['mismatch_fraction']:.6%}); splitting uses timestamp.",
            f"- Small Matrix has {small_time_field['missing_count']:,} missing "
            "`time/date/timestamp` rows; it is therefore used only as a static audit.",
            f"- Caption CSV contains {caption_note['bare_carriage_return_bytes']} bare carriage "
            "returns. LF-only record parsing preserves all 10,728 video rows.",
            f"- Non-empty caption coverage is {coverage['caption']['coverage']:.4%}; cold items "
            "must also fall back to category/topic content.",
            "- Big Matrix is verified user-major and timestamp-monotonic within each user; "
            "this permits exact first-view target filtering without reordering equal timestamps.",
            "- `upload_dt` has day precision only. Candidate availability is conservatively "
            "the following local midnight, so same-day targets with unverifiable upload "
            "times are excluded: "
            f"train={audit['splits']['train']['positive_same_day_upload_time_unverifiable_count']:,}, "
            f"validation={audit['splits']['validation']['positive_same_day_upload_time_unverifiable_count']:,}, "
            f"temporal_final={audit['splits']['temporal_final']['positive_same_day_upload_time_unverifiable_count']:,}.",
            "- Strong positives timestamped before even the declared upload date are also "
            "excluded as metadata inconsistencies: "
            f"train={audit['splits']['train']['positive_before_declared_upload_date_count']:,}, "
            f"validation={audit['splits']['validation']['positive_before_declared_upload_date_count']:,}, "
            f"temporal_final={audit['splits']['temporal_final']['positive_before_declared_upload_date_count']:,}.",
            f"- Small Matrix is {small_coverage['observed_pair_fraction']:.4%} observed, "
            f"not literally complete; {small_coverage['missing_pairs']:,} pairs are "
            "treated as blocked/missing for the primary audit.",
            "",
            "## Baseline scale and estimated cost",
            "",
        ]
    )
    cost = audit["baseline_cost_estimates"]
    lines.append(f"> {cost['disclaimer']}")
    lines.extend(
        [
            "",
            "| baseline | fit scale | evaluation scale | planning estimate |",
            "|---|---|---|---|",
        ]
    )
    for name, stats in cost["baselines"].items():
        lines.append(
            f"| `{name}` | {stats['fit_scale']} | {stats['evaluation_scale']} | "
            f"{stats['expected_compute']} |"
        )

    lines.extend(["", "## Complete schema and missingness", ""])
    for filename, file_stats in audit["files"].items():
        lines.extend(
            [
                f"### `{filename}`",
                "",
                f"Rows: {file_stats['rows']:,}",
                "",
                "| field | observed dtype(s) | missing | missing rate |",
                "|---|---|---:|---:|",
            ]
        )
        for field in file_stats["fields"]:
            lines.append(
                f"| `{field['name']}` | `{', '.join(field['observed_dtypes'])}` | "
                f"{field['missing_count']:,} | {field['missing_fraction']:.6%} |"
            )
        lines.append("")
    return "\n".join(lines)


def generate_bundle(
    args: argparse.Namespace,
    *,
    generated_at_override: str | None = None,
    lineage_override: dict[str, Any] | None = None,
) -> list[Path]:
    lock_path = args.manifest.parent / "FINAL_HOLDOUT_LOCKED.json"
    report_json_path = args.report_dir / "audit.json"
    report_markdown_path = args.report_dir / "audit.md"
    ensure_phase0_outputs_absent(
        [args.manifest, lock_path, report_json_path, report_markdown_path]
    )
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    protocol_revision = str(config.get("protocol", {}).get("revision", ""))
    if protocol_revision != "protocol-v2.1":
        raise RuntimeError(
            "This generator requires the reviewed protocol-v2.1 config"
        )
    if config["label"] != {
        "field": "watch_ratio",
        "operator": ">",
        "threshold": 2.0,
        "quick_skip_ms": 3000,
    }:
        raise RuntimeError("The Phase 0 label contract must remain watch_ratio > 2.0")

    archive = args.data_root / "KuaiRec.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    archive_md5 = md5_file(archive)
    if archive_md5 != config["dataset"]["archive_md5"]:
        raise RuntimeError(
            f"Archive MD5 mismatch: expected {config['dataset']['archive_md5']}, got {archive_md5}"
        )

    files = find_dataset_files(args.data_root, config["dataset"]["expected_files"])
    file_audits: dict[str, Any] = {}
    for filename, path in files.items():
        print(f"Scanning {filename}...", flush=True)
        file_audits[filename] = scan_csv(
            path, interaction=filename in {"big_matrix.csv", "small_matrix.csv"}
        )

    big_internal = file_audits["big_matrix.csv"]["interaction"]
    small_internal = file_audits["small_matrix.csv"]["interaction"]
    train_fraction, validation_fraction, temporal_final_fraction = (
        validate_split_fractions(config["split"])
    )
    boundaries = choose_timestamp_boundaries(
        files["big_matrix.csv"],
        train_fraction,
        validation_fraction,
    )
    print("Auditing canonical Big Matrix behavior events...", flush=True)
    behavior_event_audit = audit_canonical_behavior_events(
        files["big_matrix.csv"],
        boundaries,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print("Computing temporal split statistics...", flush=True)
    catalog_policy = load_candidate_catalog_policy(
        files["item_daily_features.csv"],
        float(big_internal["timestamp_min"]),
        float(big_internal["timestamp_max"]),
    )
    upload_epoch_by_video, available_epoch_by_video = load_upload_availability_epochs(
        files["item_daily_features.csv"]
    )
    split_stats = scan_big_splits(
        files["big_matrix.csv"],
        boundaries,
        upload_epoch_by_video,
        available_epoch_by_video,
        catalog_policy,
        behavior_event_audit,
    )
    canonical_targets = {
        split_name: stats.pop("_canonical_targets")
        for split_name, stats in split_stats.items()
    }
    print("Replaying causal candidate sizes and hard-negative pools...", flush=True)
    candidate_sizes, hard_negative_audit = audit_candidate_sizes_and_hard_negatives(
        files["big_matrix.csv"],
        canonical_targets,
        catalog_policy,
        boundaries,
        session_gap_minutes=float(config["negative_sampling"]["session_gap_minutes"]),
        hard_window_minutes=float(config["negative_sampling"]["local_window_minutes"]),
    )
    small_coverage = small_matrix_observation_coverage(
        files["small_matrix.csv"]
    )

    catalog_items = set(big_internal["video_ids"]) | set(small_internal["video_ids"])
    coverage = metadata_coverage(files, catalog_items)
    histories = history_distributions(split_stats, set(small_internal["user_ids"]))
    cold_start = cold_start_context_audit(
        split_stats,
        canonical_targets,
        set(int(value) for value in small_internal["video_ids"]),
    )
    costs = baseline_cost_estimates(
        split_stats,
        small_coverage["observed_unique_pairs"],
        small_coverage["expected_complete_pairs"],
        len(big_internal["video_ids"]),
    )

    print("Hashing source files and contracts...", flush=True)
    source_files = {
        filename: {
            "relative_path": str(path.relative_to(args.data_root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for filename, path in files.items()
    }
    project_root = Path(__file__).resolve().parent.parent
    active_contracts: dict[str, dict[str, str]] = {}
    for name, relative in config["protocol"]["active_contracts"].items():
        path = project_root / str(relative)
        if not path.exists():
            raise FileNotFoundError(f"Active contract is missing: {path}")
        active_contracts[name] = {
            "path": str(relative),
            "sha256": sha256_file(path),
        }

    generated_at = generated_at_override or datetime.now(tz=timezone.utc).isoformat()
    lineage = lineage_override or dict(config["protocol"]["lineage"])
    audit = {
        "schema_version": 2,
        "protocol_revision": protocol_revision,
        "generated_at_utc": generated_at,
        "phase": "phase_0_data_and_evaluation_audit_only",
        "model_or_baseline_executed": False,
        "final_ranking_evaluation_executed": False,
        "config_sha256": sha256_bytes(config_bytes),
        "files": file_audits,
        "big_matrix_behavior_event_audit": behavior_event_audit,
        "splits": split_stats,
        "small_matrix_observation_coverage": small_coverage,
        "candidate_catalog_audit": catalog_policy["audit"],
        "candidate_size_audit": candidate_sizes,
        "hard_negative_pool_audit": hard_negative_audit,
        "cold_start_context_audit": cold_start,
        "metadata_coverage": coverage,
        "history_distributions": histories,
        "baseline_cost_estimates": costs,
    }

    report_payload = strip_heavy_internal_fields(audit)

    manifest_splits: dict[str, Any] = {}
    for name, stats in split_stats.items():
        manifest_splits[name] = {
            key: value
            for key, value in stats.items()
            if key not in {"user_ids", "video_ids", "per_user_events", "per_user_positives"}
        }
        manifest_splits[name]["canonical_target_sha256"] = _stable_target_hash(
            canonical_targets[name]
        )
        manifest_splits[name]["candidate_membership_sha256"] = candidate_sizes[
            "per_split"
        ][name]["candidate_membership_sha256"]
        manifest_splits[name]["candidate_membership_query_count"] = candidate_sizes[
            "per_split"
        ][name]["candidate_membership_query_count"]
    manifest = {
        "schema_version": 2,
        "protocol_revision": protocol_revision,
        "created_at_utc": generated_at,
        "immutable": True,
        "lineage": lineage,
        "dataset": {
            "name": config["dataset"]["name"],
            "zenodo_record": config["dataset"]["zenodo_record"],
            "source": config["dataset"]["source"],
            "archive_md5": archive_md5,
            "archive_sha256": sha256_file(archive),
            "source_files": source_files,
        },
        "config_sha256": sha256_bytes(config_bytes),
        "active_contracts": active_contracts,
        "generation_code": {
            "path": "scripts/audit_phase0.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "label": {
            "expression": "watch_ratio > 2.0",
            "field": "watch_ratio",
            "operator": ">",
            "threshold": 2.0,
        },
        "split_algorithm": {
            "source": "big_matrix.csv",
            "time_field": "timestamp",
            "timezone_for_reporting": "Asia/Shanghai",
            "sort_unit": "global raw interaction timestamp for frozen cutoff selection",
            "event_assignment_unit": "canonical (user_id,video_id,timestamp) event",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "temporal_final_fraction": temporal_final_fraction,
            "fraction_validation": dict(config["split"]["fraction_validation"]),
            "boundary_method": (
                "floor interaction-count quantile; equal boundary timestamps "
                "remain atomic in the later split"
            ),
            "train_end_exclusive": boundaries["train_end_exclusive"],
            "train_end_exclusive_asia_shanghai": local_timestamp(
                float(boundaries["train_end_exclusive"])
            ),
            "validation_end_exclusive": boundaries["validation_end_exclusive"],
            "validation_end_exclusive_asia_shanghai": local_timestamp(
                float(boundaries["validation_end_exclusive"])
            ),
            "raw_date_field_used": False,
            "new_entity_fraction_denominator": "unique active entities in the split",
            "canonical_event_boundary_sensitivity": behavior_event_audit[
                "boundary_sensitivity"
            ],
        },
        "canonical_behavior_events": {
            key: value
            for key, value in behavior_event_audit.items()
            if key not in {"per_frozen_split", "_canonical_count_boundaries"}
        },
        "splits": manifest_splits,
        "fit_contexts": config["fit_contexts"],
        "candidate_catalog": {
            "policy": catalog_policy["audit"],
            "per_query_sizes": candidate_sizes,
            "membership_hash_format": candidate_sizes[
                "membership_hash_format"
            ],
        },
        "cold_start_contexts": cold_start,
        "hard_negative_pool_audit": hard_negative_audit,
        "small_matrix_audit": {
            "rows": file_audits["small_matrix.csv"]["rows"],
            "users": small_internal["users"],
            "videos": small_internal["videos"],
            "positive_count": small_internal["positive_count"],
            "positive_fraction": small_internal["positive_fraction"],
            "users_with_zero_positives": small_internal["positives_per_user"]["zero_count"],
            "users_with_zero_positives_fraction": small_internal["positives_per_user"]["zero_fraction"],
            "ground_truth_only": True,
            "enters_training_or_history": False,
            "blocked_information_enters_negative_sampling": False,
            "primary_candidate_policy": (
                "rank physically observed pairs only (therefore blocked/missing "
                "pairs are excluded per user), intersected with the NORMAL-only "
                "primary task catalog"
            ),
            "primary_catalog_video_type": "NORMAL",
            "advertisement_quality_policy": (
                "AD items are excluded from primary quality metrics and reported "
                "only as a separate diagnostic"
            ),
            "secondary_candidate_policy": (
                "rank all 3327 only for Blocked@K safety audit"
            ),
            "observation_coverage": small_coverage,
        },
        "holdout_disclosure": {
            "aggregate_label_query_and_quality_statistics_already_published": True,
            "untouched_claim": False,
            "claim": "frozen after Phase 0 aggregate audit",
        },
        "locks": {
            "temporal_final_locked": True,
            "small_matrix_audit_locked": True,
            "ordinary_baseline_scripts_may_run_final": False,
            "unlock_requires_separate_explicit_final_command": True,
        },
    }
    write_immutable_json(args.manifest, manifest)
    lock_payload = {
        "locked": True,
        "protocol_revision": protocol_revision,
        "manifest": "manifests/split_manifest.json",
        "manifest_sha256": sha256_file(args.manifest),
        "protected": ["temporal_final", "small_matrix_audit"],
        "ordinary_baseline_access": False,
        "final_entrypoint": "scripts/final_evaluation.py",
        "final_receipt_overwrite": False,
    }
    write_immutable_json(lock_path, lock_payload)
    write_json(report_json_path, report_payload)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path.write_text(markdown_report(report_payload))
    print(f"Wrote {report_markdown_path}")
    print(f"Wrote immutable {args.manifest}")
    print(f"Wrote immutable {lock_path}")
    return [args.manifest, lock_path, report_json_path, report_markdown_path]


def recompute_protocol_derived_hashes(
    *,
    config_path: Path,
    data_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Recompute target and candidate hashes without writing or evaluating.

    This is the deliberately expensive, fail-closed hook used by the bundle
    verifier before baseline data preparation. It performs no scoring, model
    fitting, hyperparameter selection, or final evaluation.
    """

    config = yaml.safe_load(config_path.read_text())
    if config.get("protocol", {}).get("revision") != "protocol-v2.1":
        raise RuntimeError("Derived-hash recomputation requires protocol-v2.1")
    train_fraction, validation_fraction, _ = validate_split_fractions(
        config["split"]
    )
    files = find_dataset_files(data_root, config["dataset"]["expected_files"])
    boundaries = choose_timestamp_boundaries(
        files["big_matrix.csv"], train_fraction, validation_fraction
    )
    locked_algorithm = manifest.get("split_algorithm", {})
    for field in ("train_end_exclusive", "validation_end_exclusive"):
        if float(boundaries[field]) != float(locked_algorithm.get(field, math.nan)):
            raise RuntimeError(
                f"Recomputed {field} does not match the locked manifest"
            )

    manifest_splits = manifest.get("splits", {})
    timestamp_min = min(
        float(manifest_splits[name]["timestamp_start_inclusive"])
        for name in ("train", "validation", "temporal_final")
    )
    timestamp_max = max(
        float(manifest_splits[name]["timestamp_end_inclusive"])
        for name in ("train", "validation", "temporal_final")
    )
    catalog_policy = load_candidate_catalog_policy(
        files["item_daily_features.csv"], timestamp_min, timestamp_max
    )
    upload_epoch_by_video, available_epoch_by_video = load_upload_availability_epochs(
        files["item_daily_features.csv"]
    )
    split_stats = scan_big_splits(
        files["big_matrix.csv"],
        boundaries,
        upload_epoch_by_video,
        available_epoch_by_video,
        catalog_policy,
    )
    canonical_targets = {
        split_name: stats["_canonical_targets"]
        for split_name, stats in split_stats.items()
    }
    candidate_sizes, _ = audit_candidate_sizes_and_hard_negatives(
        files["big_matrix.csv"],
        canonical_targets,
        catalog_policy,
        boundaries,
        session_gap_minutes=float(config["negative_sampling"]["session_gap_minutes"]),
        hard_window_minutes=float(config["negative_sampling"]["local_window_minutes"]),
    )
    return {
        "canonical_targets": {
            name: _stable_target_hash(targets)
            for name, targets in canonical_targets.items()
        },
        "candidate_membership": {
            name: candidate_sizes["per_split"][name][
                "candidate_membership_sha256"
            ]
            for name in ("train", "validation", "temporal_final")
        },
    }


def verify_bundle(args: argparse.Namespace) -> None:
    reference_manifest = json.loads(args.reference_manifest.read_text())
    config = yaml.safe_load(args.config.read_text())
    expected_revision = config.get("protocol", {}).get("revision")
    if expected_revision != "protocol-v2.1":
        raise RuntimeError("Verify mode requires the protocol-v2.1 config")
    if reference_manifest.get("protocol_revision") != expected_revision:
        raise RuntimeError(
            "Verify mode requires a committed manifest matching the config revision"
        )
    reference_lock = args.reference_manifest.parent / "FINAL_HOLDOUT_LOCKED.json"
    lock = json.loads(reference_lock.read_text())
    if lock.get("manifest_sha256") != sha256_file(args.reference_manifest):
        raise RuntimeError("Committed manifest does not match the holdout lock")
    reference_paths = {
        "manifest": args.reference_manifest,
        "lock": reference_lock,
        "json": args.reference_report_dir / "audit.json",
        "markdown": args.reference_report_dir / "audit.md",
    }
    for path in reference_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="kuairec-protocol-v2.1-verify-") as temp:
        root = Path(temp)
        generated_args = argparse.Namespace(
            config=args.config,
            data_root=args.data_root,
            report_dir=root / "reports" / "phase0",
            manifest=root / "manifests" / "split_manifest.json",
        )
        generated = generate_bundle(
            generated_args,
            generated_at_override=str(reference_manifest["created_at_utc"]),
            lineage_override=dict(reference_manifest["lineage"]),
        )
        generated_paths = {
            "manifest": generated_args.manifest,
            "lock": generated_args.manifest.parent / "FINAL_HOLDOUT_LOCKED.json",
            "json": generated_args.report_dir / "audit.json",
            "markdown": generated_args.report_dir / "audit.md",
        }
        mismatches = []
        for name, reference in reference_paths.items():
            candidate = generated_paths[name]
            if reference.read_bytes() != candidate.read_bytes():
                mismatches.append(
                    {
                        "artifact": name,
                        "reference_sha256": sha256_file(reference),
                        "recomputed_sha256": sha256_file(candidate),
                    }
                )
        if mismatches:
            raise RuntimeError(
                "Protocol-v2.1 verification failed: "
                + json.dumps(mismatches, sort_keys=True)
            )
    print("Verified protocol-v2.1 bundle: all four artifacts match byte-for-byte")


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        if args.supersede_protocol_v1 or args.supersede_protocol_v2:
            raise RuntimeError("Verify mode cannot supersede an existing bundle")
        verify_bundle(args)
        return

    if args.supersede_protocol_v1 and args.supersede_protocol_v2:
        raise RuntimeError("Select at most one protocol supersession source")

    lock_path = args.manifest.parent / "FINAL_HOLDOUT_LOCKED.json"
    output_paths = [
        args.manifest,
        lock_path,
        args.report_dir / "audit.json",
        args.report_dir / "audit.md",
    ]
    if args.supersede_protocol_v1:
        archive = archive_protocol_v1_bundle(
            output_paths, Path("archive/protocol-v1")
        )
        config = yaml.safe_load(args.config.read_text())
        expected_parent = config["protocol"]["lineage"][
            "parent_manifest_sha256"
        ]
        if archive["parent_manifest_sha256"] != expected_parent:
            raise RuntimeError(
                "Archived v1 manifest hash does not match configured lineage"
            )
    elif args.supersede_protocol_v2:
        config = yaml.safe_load(args.config.read_text())
        expected_parent = config["protocol"]["lineage"][
            "parent_manifest_sha256"
        ]
        # Fail before moving the immutable bundle. A lineage mismatch must not
        # leave the working tree with only an archive and no active manifest.
        if sha256_file(args.manifest) != expected_parent:
            raise RuntimeError(
                "Existing protocol-v2 manifest hash does not match configured "
                "protocol-v2.1 lineage"
            )
        project_root = Path(__file__).resolve().parent.parent
        archive = archive_protocol_v2_bundle(
            output_paths, project_root / "archive" / "protocol-v2"
        )
        if archive["parent_manifest_sha256"] != expected_parent:
            raise RuntimeError(
                "Archived v2 manifest hash does not match configured lineage"
            )
    generate_bundle(args)


if __name__ == "__main__":
    main()
