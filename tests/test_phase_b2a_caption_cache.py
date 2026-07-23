from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from kuairec_fully_observed.caption_embeddings import (
    CAPTION_MODEL_ID,
    PREPROCESSING_VERSION,
    build_caption_cache,
    cleaned_text_sha256,
    load_sentence_transformer,
    load_caption_cache,
    resolve_model_revision,
    validate_pinned_revision,
)


REVISION = "1" * 40
SOURCE_SHA = "2" * 64


class ToyEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, sentences, **kwargs):
        del kwargs
        self.calls.append(list(sentences))
        output = np.zeros((len(sentences), 384), dtype=np.float32)
        for row, text in enumerate(sentences):
            output[row, 0] = len(text)
            output[row, 1] = 1.0
        return output


def _build(tmp_path):
    ids = np.asarray([3, 7, 9], dtype=np.int64)
    texts = ["caption", "", "topic"]
    cache = tmp_path / "captions.npz"
    metadata = tmp_path / "captions.json"
    encoder = ToyEncoder()
    value = build_caption_cache(
        item_ids=ids,
        cleaned_texts=texts,
        encoder=encoder,
        cache_path=cache,
        metadata_path=metadata,
        model_id=CAPTION_MODEL_ID,
        resolved_revision=REVISION,
        source_actual_sha256=SOURCE_SHA,
        source_expected_sha256=SOURCE_SHA,
        versions={
            "torch": "toy",
            "sentence_transformers": "toy",
            "cuda": None,
        },
        batch_size=2,
    )
    return ids, texts, cache, metadata, encoder, value


def _load(ids, texts, cache, metadata, **overrides):
    arguments = {
        "cache_path": cache,
        "metadata_path": metadata,
        "expected_item_ids": ids,
        "expected_model_id": CAPTION_MODEL_ID,
        "expected_revision": REVISION,
        "expected_source_sha256": SOURCE_SHA,
        "expected_cleaned_text_sha256": cleaned_text_sha256(ids, texts),
    }
    arguments.update(overrides)
    return load_caption_cache(**arguments)


def test_caption_cache_round_trip_and_empty_text_is_exact_zero(tmp_path):
    ids, texts, cache, metadata, encoder, built = _build(tmp_path)
    assert encoder.calls == [["caption", "topic"]]
    assert np.array_equal(built.item_ids, ids)
    assert np.array_equal(built.embeddings[1], np.zeros(384, dtype=np.float32))
    assert built.metadata["empty_text_count"] == 1
    assert built.metadata["preprocessing_version"] == PREPROCESSING_VERSION
    assert str(tmp_path) not in json.dumps(built.metadata)
    assert "/home/" not in metadata.read_text()
    loaded = _load(ids, texts, cache, metadata)
    np.testing.assert_array_equal(loaded.embeddings, built.embeddings)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_id", "wrong/model"),
        ("resolved_revision", "3" * 40),
        ("source_actual_sha256", "4" * 64),
        ("source_expected_sha256", "4" * 64),
        ("ordered_item_membership_sha256", "5" * 64),
        ("cleaned_text_sha256", "6" * 64),
        ("preprocessing_version", "wrong"),
    ],
)
def test_caption_cache_rejects_every_metadata_identity_mismatch(
    tmp_path, field, replacement
):
    ids, texts, cache, metadata, _, _ = _build(tmp_path)
    value = json.loads(metadata.read_text())
    value[field] = replacement
    metadata.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load(ids, texts, cache, metadata)


def test_caption_cache_rejects_wrong_item_order_and_corrupted_payload(tmp_path):
    ids, texts, cache, metadata, _, _ = _build(tmp_path)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load(ids[::-1], texts[::-1], cache, metadata)

    with cache.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="file SHA256 mismatch"):
        _load(ids, texts, cache, metadata)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("shape", [3, 1], "metadata shape changed"),
        ("dtype", "float64", "metadata dtype changed"),
    ],
)
def test_caption_cache_rejects_metadata_payload_contract_changes(
    tmp_path, field, replacement, message
):
    ids, texts, cache, metadata, _, _ = _build(tmp_path)
    value = json.loads(metadata.read_text())
    value[field] = replacement
    metadata.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match=message):
        _load(ids, texts, cache, metadata)


def test_caption_cache_rejects_source_sha_before_encoding(tmp_path):
    encoder = ToyEncoder()
    with pytest.raises(RuntimeError, match="Caption source SHA256 mismatch"):
        build_caption_cache(
            item_ids=np.asarray([1], dtype=np.int64),
            cleaned_texts=["text"],
            encoder=encoder,
            cache_path=tmp_path / "cache.npz",
            metadata_path=tmp_path / "metadata.json",
            model_id=CAPTION_MODEL_ID,
            resolved_revision=REVISION,
            source_actual_sha256="a" * 64,
            source_expected_sha256="b" * 64,
            versions={},
        )
    assert encoder.calls == []


def test_sentence_transformer_loads_the_pinned_revision_without_resolving_main(
    monkeypatch,
):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_id, *, revision):
            calls.append((model_id, revision))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    pinned = "a" * 40
    loaded = load_sentence_transformer(CAPTION_MODEL_ID, pinned)
    assert isinstance(loaded, FakeSentenceTransformer)
    assert calls == [(CAPTION_MODEL_ID, pinned)]
    assert validate_pinned_revision(pinned) == pinned
    with pytest.raises(ValueError, match="pinned 40-character"):
        load_sentence_transformer(CAPTION_MODEL_ID, "main")
    with pytest.raises(RuntimeError, match="Floating caption revision"):
        resolve_model_revision(CAPTION_MODEL_ID)
