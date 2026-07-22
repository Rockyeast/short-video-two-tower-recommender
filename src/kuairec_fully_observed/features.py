"""Fail-closed V1 item-feature contract.

Daily engagement aggregates are intentionally excluded because a latest-row
lookup could expose validation or Small-Matrix information.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ITEM_FEATURE_COLUMNS = frozenset(
    {
        "video_id",
        "caption_embedding",
        "category_ids",
        "video_duration",
        "video_width",
        "video_height",
        "upload_type",
        "upload_dt",
    }
)
DAILY_STATIC_SOURCE_COLUMNS = (
    "video_id",
    "date",
    "video_type",
    "upload_dt",
    "upload_type",
    "video_duration",
    "video_width",
    "video_height",
)
CAPTION_SOURCE_COLUMNS = (
    "video_id",
    "manual_cover_text",
    "caption",
    "topic_tag",
    "first_level_category_id",
    "second_level_category_id",
    "third_level_category_id",
)
STATIC_ITEM_FRAME_COLUMNS = (
    "video_id",
    "caption_text",
    "category_ids",
    "video_duration",
    "video_width",
    "video_height",
    "upload_type",
    "upload_dt",
)


@dataclass(frozen=True)
class StaticItemFeatures:
    frame: pd.DataFrame
    normal_item_ids: np.ndarray
    variant_static_item_ids: np.ndarray


def validate_model_item_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Accept only the frozen static/content model inputs."""

    requested = tuple(columns)
    unknown = sorted(set(requested) - MODEL_ITEM_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Disallowed or unknown model item features: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("Model item feature columns must be unique")
    return requested


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    return text.mask(text.str.upper().eq("UNKNOWN"), "")


def load_static_item_features(
    data_dir: str | Path,
    *,
    item_ids: np.ndarray | None = None,
    chunksize: int = 100_000,
) -> StaticItemFeatures:
    """Load only static/content columns from real KuaiRec files.

    The daily CSV is read with an explicit ``usecols`` list. Engagement columns
    are never materialized. Every selected static field must be constant for a
    video across daily rows. If a source correction exists, the loader records
    that video and deterministically uses its earliest available row rather
    than leaking a later snapshot.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    root = Path(data_dir).expanduser().resolve()
    daily_path = root / "item_daily_features.csv"
    caption_path = root / "kuairec_caption_category.csv"
    if not daily_path.is_file() or not caption_path.is_file():
        raise ValueError("Static feature source files are missing")
    requested = (
        None
        if item_ids is None
        else set(int(item) for item in np.asarray(item_ids, dtype=np.int64))
    )
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        daily_path, usecols=list(DAILY_STATIC_SOURCE_COLUMNS), chunksize=chunksize
    ):
        if requested is not None:
            chunk = chunk[chunk["video_id"].isin(requested)]
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No requested videos were found in item_daily_features")
    daily = pd.concat(chunks, ignore_index=True)
    static_columns = [
        "video_type",
        "upload_dt",
        "upload_type",
        "video_duration",
        "video_width",
        "video_height",
    ]
    inconsistent = daily.groupby("video_id", sort=False)[static_columns].nunique(
        dropna=False
    )
    bad = inconsistent.index[(inconsistent > 1).any(axis=1)].to_numpy(np.int64)
    daily = (
        daily.sort_values(["video_id", "date"], kind="mergesort")
        .drop_duplicates("video_id", keep="first")
        .reset_index(drop=True)
    )
    captions = pd.read_csv(
        caption_path,
        usecols=list(CAPTION_SOURCE_COLUMNS),
        lineterminator="\n",
    )
    if requested is not None:
        captions = captions[captions["video_id"].isin(requested)]
    if captions["video_id"].duplicated().any():
        raise ValueError("Caption/category source contains duplicate video_id")
    caption = _clean_text(captions["caption"])
    cover = _clean_text(captions["manual_cover_text"])
    topic = _clean_text(captions["topic_tag"])
    captions = captions.assign(
        caption_text=caption.mask(caption.eq(""), cover).mask(
            caption.eq("") & cover.eq(""), topic
        ),
        category_ids=list(
            zip(
                captions["first_level_category_id"].fillna(-1).astype(np.int64),
                captions["second_level_category_id"].fillna(-1).astype(np.int64),
                captions["third_level_category_id"].fillna(-1).astype(np.int64),
                strict=True,
            )
        ),
    )[["video_id", "caption_text", "category_ids"]]
    merged = daily.merge(captions, on="video_id", how="left", validate="one_to_one")
    merged["caption_text"] = merged["caption_text"].fillna("")
    merged["category_ids"] = merged["category_ids"].map(
        lambda value: (-1, -1, -1) if not isinstance(value, tuple) else value
    )
    normal = merged.loc[merged["video_type"].eq("NORMAL"), "video_id"].to_numpy(
        np.int64
    )
    frame = merged[list(STATIC_ITEM_FRAME_COLUMNS)].sort_values(
        "video_id", kind="mergesort"
    )
    return StaticItemFeatures(
        frame=frame.reset_index(drop=True),
        normal_item_ids=normal,
        variant_static_item_ids=bad,
    )
