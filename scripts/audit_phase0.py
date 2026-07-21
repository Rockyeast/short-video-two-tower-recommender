#!/usr/bin/env python3
"""Run the KuaiRec Phase 0 data and evaluation audit.

This script intentionally contains no model training or baseline execution.  It
creates a data inventory, a globally chronological split manifest, evaluation
cost estimates, and final-holdout lock files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


CHUNK_SIZE = 250_000
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase0"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/split_manifest.json")
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
    """Measure how close Small Matrix is to a complete user-item matrix."""

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
        "observed_pair_fraction": (
            observed_unique_pairs / expected_pairs if expected_pairs else 0.0
        ),
        "missing_pairs_per_user": distribution(per_user_missing),
        "primary_ranking_catalog_size": len(catalog),
        "missing_feedback_policy": (
            "retain all catalog items; an unobserved pair is unjudged and never a "
            "training negative; primary binary metrics use only observed strong-positive "
            "items as the relevance set"
        ),
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


def scan_big_splits(
    path: Path,
    boundaries: dict[str, float | int],
    upload_epoch_by_video: dict[int, float],
    available_epoch_by_video: dict[int, float],
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
            "users": set(),
            "videos": set(),
            "per_user_events": Counter(),
            "per_user_positives": Counter(),
            "positive_group_parts": [],
            "timestamp_min": math.inf,
            "timestamp_max": -math.inf,
        }
        for name in split_order
    }
    carry_user: int | None = None
    carry_first_timestamp_by_video: dict[int, float] = {}
    previous_user: int | None = None
    previous_timestamp: float | None = None
    columns = ["user_id", "video_id", "timestamp", "watch_ratio"]
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
            positive = frame["watch_ratio"] > 2.0
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
            eligible_positive = (
                positive
                & ~missing_upload
                & ~before_declared_upload_date
                & ~same_day_upload_time_unverifiable
                & ~previously_seen
            )
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
                (
                    positive
                    & ~missing_upload
                    & ~before_declared_upload_date
                    & ~same_day_upload_time_unverifiable
                    & previously_seen
                ).sum()
            )
            current["users"].update(split_user.unique().tolist())
            current["videos"].update(split_video.unique().tolist())
            add_counts(current["per_user_events"], split_user.value_counts())
            add_counts(current["per_user_positives"], positive_user.value_counts())
            if eligible_positive.any():
                current["positive_group_parts"].append(
                    frame.loc[
                        eligible_positive, ["user_id", "video_id", "timestamp"]
                    ].copy()
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
        if current["positive_group_parts"]:
            positive_events = pd.concat(
                current["positive_group_parts"], ignore_index=True
            ).drop_duplicates(["user_id", "timestamp", "video_id"])
            target_group_sizes = (
                positive_events.groupby(["user_id", "timestamp"], sort=False)
                .size()
                .to_numpy()
            )
        else:
            target_group_sizes = np.asarray([], dtype=np.int64)
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
            "unique_eligible_target_count": int(target_group_sizes.sum()),
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
        }
    return output


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


def baseline_cost_estimates(
    split_stats: dict[str, Any], small_complete_pairs: int, big_video_count: int
) -> dict[str, Any]:
    train_rows = split_stats["train"]["rows"]
    train_positive = split_stats["train"]["positive_count"]
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
        "small_matrix_full_ranking_pairs": small_complete_pairs,
        "baselines": {
            "random": {
                "fit_scale": "none",
                "evaluation_scale": f"up to {dense_validation_upper:,} candidate pairs",
                "expected_compute": "under 5 CPU minutes with direct seeded top-K sampling",
            },
            "global_popularity": {
                "fit_scale": f"one pass over {train_rows:,} train interactions",
                "evaluation_scale": "one shared ranking plus per-user seen filtering",
                "expected_compute": "roughly 1-5 CPU minutes",
            },
            "time_decayed_popularity": {
                "fit_scale": f"one chronological pass over {train_rows:,} interactions",
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
            "- Temporal: `contracts/temporal_evaluation_v1.yaml`",
            "- Fully observed: `contracts/fully_observed_audit_v1.yaml`",
            "- Negative sampling: `contracts/negative_sampling_v1.yaml`",
            "- Cold-item fallback: `contracts/two_tower_cold_start_v1.yaml`",
            "",
            "Equal-timestamp strong positives are one multi-target query with shared",
            "history ending strictly before that timestamp.",
            "",
            "A target must also be unseen before its query timestamp and certainly uploaded.",
            "Because `upload_dt` has date precision only, an item becomes eligible at the",
            "next Asia/Shanghai midnight; same-day events are excluded as unverifiable.",
            "The following exclusion counts are event-level; unique target videos are",
            "deduplicated inside each `(user_id, timestamp)` query group.",
            "",
            "| split | raw positives | eligible events | unique targets | before declared date | same-day time unknown | missing upload | previously seen |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in audit["splits"].items():
        lines.append(
            f"| `{name}` | {stats['positive_count']:,} | "
            f"{stats['eligible_positive_count']:,} | "
            f"{stats['unique_eligible_target_count']:,} | "
            f"{stats['positive_before_declared_upload_date_count']:,} | "
            f"{stats['positive_same_day_upload_time_unverifiable_count']:,} | "
            f"{stats['positive_missing_upload_count']:,} | "
            f"{stats['positive_previously_seen_count']:,} |"
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
            f"({small_coverage['observed_pair_fraction']:.4%}); missing/unjudged pairs: "
            f"{small_coverage['missing_pairs']:,}.",
            f"- Missing pairs per user p50/p90/p99/max: "
            f"{missing_quantiles['p50']:.0f}/{missing_quantiles['p90']:.0f}/"
            f"{missing_quantiles['p99']:.0f}/{missing_quantiles['p100']:.0f}.",
            "- Missing pairs remain in the 3,327-item ranking catalog but are unjudged, "
            "never sampled as training negatives, and never added to the positive set.",
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
            f"not literally complete; {small_coverage['missing_pairs']:,} pairs are unjudged.",
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


def main() -> None:
    args = parse_args()
    lock_path = args.manifest.parent / "FINAL_HOLDOUT_LOCKED.json"
    report_json_path = args.report_dir / "audit.json"
    report_markdown_path = args.report_dir / "audit.md"
    ensure_phase0_outputs_absent(
        [args.manifest, lock_path, report_json_path, report_markdown_path]
    )
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["label"] != {"field": "watch_ratio", "operator": ">", "threshold": 2.0, "quick_skip_ms": 3000}:
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
    boundaries = choose_timestamp_boundaries(
        files["big_matrix.csv"],
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
    )
    print("Computing temporal split statistics...", flush=True)
    upload_epoch_by_video, available_epoch_by_video = load_upload_availability_epochs(
        files["item_daily_features.csv"]
    )
    split_stats = scan_big_splits(
        files["big_matrix.csv"],
        boundaries,
        upload_epoch_by_video,
        available_epoch_by_video,
    )
    small_coverage = small_matrix_observation_coverage(
        files["small_matrix.csv"]
    )

    catalog_items = set(big_internal["video_ids"]) | set(small_internal["video_ids"])
    coverage = metadata_coverage(files, catalog_items)
    histories = history_distributions(split_stats, set(small_internal["user_ids"]))
    costs = baseline_cost_estimates(
        split_stats,
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
    contract_paths = sorted(Path("contracts").glob("*.yaml"))
    contract_hashes = {str(path): sha256_file(path) for path in contract_paths}

    generated_at = datetime.now(tz=timezone.utc).isoformat()
    audit = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "phase": "phase_0_data_and_evaluation_audit_only",
        "model_or_baseline_executed": False,
        "config_sha256": sha256_bytes(config_bytes),
        "files": file_audits,
        "splits": split_stats,
        "small_matrix_observation_coverage": small_coverage,
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
    manifest = {
        "schema_version": 1,
        "created_at_utc": generated_at,
        "immutable": True,
        "dataset": {
            "name": config["dataset"]["name"],
            "zenodo_record": config["dataset"]["zenodo_record"],
            "source": config["dataset"]["source"],
            "archive_md5": archive_md5,
            "archive_sha256": sha256_file(archive),
            "source_files": source_files,
        },
        "config_sha256": sha256_bytes(config_bytes),
        "contract_sha256": contract_hashes,
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
            "sort_unit": "global interaction timestamp",
            "train_fraction": config["split"]["train_fraction"],
            "validation_fraction": config["split"]["validation_fraction"],
            "temporal_final_fraction": config["split"]["temporal_final_fraction"],
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
        },
        "splits": manifest_splits,
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
            "observation_coverage": small_coverage,
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
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "protected": ["temporal_final", "small_matrix_audit"],
        "ordinary_baseline_access": False,
    }
    write_immutable_json(lock_path, lock_payload)
    write_json(report_json_path, report_payload)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path.write_text(markdown_report(report_payload))
    print(f"Wrote {report_markdown_path}")
    print(f"Wrote immutable {args.manifest}")
    print(f"Wrote immutable {lock_path}")


if __name__ == "__main__":
    main()
