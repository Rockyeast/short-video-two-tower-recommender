"""Fail-closed frozen caption-embedding cache for Phase B2A.

The optional sentence-transformers dependency is imported only by the two
runtime helper functions at the bottom of this module. Unit tests inject a toy
encoder and never access the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


CAPTION_MODEL_ID = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CAPTION_DIM = 384
PREPROCESSING_VERSION = "phase-b2a-caption-fallback-v1"


class CaptionEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_membership_sha256(item_ids: np.ndarray) -> str:
    values = np.asarray(item_ids, dtype=np.int64)
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError("Caption-cache item IDs must be unique rank-1 values")
    digest = hashlib.sha256(b"caption-item-membership-v1\n")
    for item_id in values:
        digest.update(f"{int(item_id)}\n".encode())
    return digest.hexdigest()


def cleaned_text_sha256(item_ids: np.ndarray, texts: list[str]) -> str:
    if len(item_ids) != len(texts):
        raise ValueError("Caption text and item IDs differ in length")
    digest = hashlib.sha256(b"caption-cleaned-text-v1\n")
    for item_id, text in zip(item_ids, texts, strict=True):
        encoded = text.encode("utf-8")
        digest.update(f"{int(item_id)}:{len(encoded)}:".encode())
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def embedding_payload_sha256(
    item_ids: np.ndarray, embeddings: np.ndarray
) -> str:
    ids = np.ascontiguousarray(item_ids, dtype=np.int64)
    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    digest = hashlib.sha256(b"caption-embedding-payload-v1\n")
    digest.update(str(ids.shape).encode())
    digest.update(ids.tobytes())
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CaptionCache:
    item_ids: np.ndarray
    embeddings: np.ndarray
    metadata: dict[str, Any]


def build_caption_cache(
    *,
    item_ids: np.ndarray,
    cleaned_texts: list[str],
    encoder: CaptionEncoder,
    cache_path: Path,
    metadata_path: Path,
    model_id: str,
    resolved_revision: str,
    source_actual_sha256: str,
    source_expected_sha256: str,
    versions: dict[str, str | None],
    batch_size: int = 128,
) -> CaptionCache:
    """Encode all nonempty texts once and atomically publish cache + metadata."""

    ids = np.asarray(item_ids, dtype=np.int64)
    if not np.array_equal(ids, np.unique(ids)):
        raise ValueError("Caption-cache item IDs must be sorted and unique")
    if len(ids) != len(cleaned_texts):
        raise ValueError("Caption-cache item/text counts differ")
    if source_actual_sha256 != source_expected_sha256:
        raise RuntimeError(
            "Caption source SHA256 mismatch: "
            f"actual={source_actual_sha256} expected={source_expected_sha256}"
        )
    if not resolved_revision or len(resolved_revision) != 40:
        raise ValueError("Caption model revision must be a resolved commit SHA")
    embeddings = np.zeros((len(ids), CAPTION_DIM), dtype=np.float32)
    nonempty = np.asarray([bool(value) for value in cleaned_texts], dtype=bool)
    if np.any(nonempty):
        encoded = np.asarray(
            encoder.encode(
                [text for text in cleaned_texts if text],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        if encoded.shape != (int(nonempty.sum()), CAPTION_DIM):
            raise RuntimeError(
                f"Caption encoder returned {encoded.shape}, expected "
                f"({int(nonempty.sum())}, {CAPTION_DIM})"
            )
        if not np.isfinite(encoded).all():
            raise FloatingPointError("Caption encoder returned NaN or Inf")
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError("Nonempty caption produced a zero embedding")
        embeddings[nonempty] = encoded / norms
    if np.any(embeddings[~nonempty] != 0.0):
        raise RuntimeError("Empty caption embeddings must be exactly zero")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, item_ids=ids, embeddings=embeddings)
    os.replace(temporary, cache_path)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "cache_locator": "artifacts/phase_b2a/caption_embeddings.npz",
        "source_locator": "KUAIREC_DATA_DIR/kuairec_caption_category.csv",
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "torch_version": versions.get("torch"),
        "sentence_transformers_version": versions.get("sentence_transformers"),
        "cuda_version": versions.get("cuda"),
        "source_actual_sha256": source_actual_sha256,
        "source_expected_sha256": source_expected_sha256,
        "ordered_item_membership_sha256": ordered_membership_sha256(ids),
        "cleaned_text_sha256": cleaned_text_sha256(ids, cleaned_texts),
        "embedding_payload_sha256": embedding_payload_sha256(ids, embeddings),
        "cache_file_sha256": sha256_file(cache_path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "empty_text_count": int((~nonempty).sum()),
        "nonempty_text_count": int(nonempty.sum()),
        "nonempty_coverage": float(nonempty.mean()) if len(nonempty) else 0.0,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return CaptionCache(ids, embeddings, metadata)


def load_caption_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
    expected_item_ids: np.ndarray,
    expected_model_id: str,
    expected_revision: str,
    expected_source_sha256: str,
    expected_cleaned_text_sha256: str,
) -> CaptionCache:
    """Load only after every identity and payload check succeeds."""

    metadata = json.loads(metadata_path.read_text())
    expected = {
        "model_id": expected_model_id,
        "resolved_revision": expected_revision,
        "source_actual_sha256": expected_source_sha256,
        "source_expected_sha256": expected_source_sha256,
        "ordered_item_membership_sha256": ordered_membership_sha256(
            expected_item_ids
        ),
        "cleaned_text_sha256": expected_cleaned_text_sha256,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Caption cache identity mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    if metadata.get("cache_file_sha256") != sha256_file(cache_path):
        raise RuntimeError("Caption cache file SHA256 mismatch")
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"item_ids", "embeddings"}:
            raise RuntimeError("Caption cache payload fields changed")
        ids = payload["item_ids"]
        embeddings = payload["embeddings"]
    expected_ids = np.asarray(expected_item_ids, dtype=np.int64)
    if ids.dtype != np.int64 or not np.array_equal(ids, expected_ids):
        raise RuntimeError("Caption cache item IDs or order changed")
    if embeddings.dtype != np.float32 or embeddings.shape != (
        len(ids),
        CAPTION_DIM,
    ):
        raise RuntimeError("Caption cache shape or dtype changed")
    if metadata.get("shape") != [len(ids), CAPTION_DIM]:
        raise RuntimeError("Caption cache metadata shape changed")
    if metadata.get("dtype") != "float32":
        raise RuntimeError("Caption cache metadata dtype changed")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Caption cache contains NaN or Inf")
    if metadata.get("embedding_payload_sha256") != embedding_payload_sha256(
        ids, embeddings
    ):
        raise RuntimeError("Caption cache embedding payload SHA256 mismatch")
    return CaptionCache(ids, embeddings, metadata)


def resolve_model_revision(model_id: str = CAPTION_MODEL_ID) -> str:
    raise RuntimeError(
        "Floating caption revision resolution is forbidden; use the pinned "
        "40-character revision from the Phase B2A config"
    )


def validate_pinned_revision(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Caption revision must be a pinned 40-character lowercase git SHA"
        )
    return revision


def load_sentence_transformer(model_id: str, revision: str) -> CaptionEncoder:
    """Load the required encoder at an immutable revision, lazily."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_id, revision=validate_pinned_revision(revision)
    )
