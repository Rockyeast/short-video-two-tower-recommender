"""Fail-closed V1 item-feature contract.

Daily engagement aggregates are intentionally excluded because a latest-row
lookup could expose validation or Small-Matrix information.
"""

from __future__ import annotations

from collections.abc import Iterable


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


def validate_model_item_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Accept only the frozen static/content model inputs."""

    requested = tuple(columns)
    unknown = sorted(set(requested) - MODEL_ITEM_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Disallowed or unknown model item features: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("Model item feature columns must be unique")
    return requested
