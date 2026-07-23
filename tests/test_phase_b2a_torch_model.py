from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kuairec_fully_observed.torch_models import (  # noqa: E402
    TorchItemFeatures,
    TwoTowerV1,
    masked_in_batch_cross_entropy,
)
from kuairec_fully_observed.torch_training import (  # noqa: E402
    PreparedItemFeatureStore,
    TrainingBatch,
    encode_item_ids,
    encode_query_users_from_precomputed,
    load_checkpoint,
    preencode_item_universe,
    record_successful_step_membership,
    save_checkpoint,
)
from kuairec_fully_observed.training import build_in_batch_logit_mask  # noqa: E402


def _features(count: int = 8) -> TorchItemFeatures:
    generator = torch.Generator().manual_seed(4)
    return TorchItemFeatures(
        item_indices=torch.arange(1, count + 1),
        category_indices=torch.tensor(
            [[1 + row % 3, 4 + row % 2, 6] for row in range(count)]
        ),
        caption_embeddings=torch.randn(count, 384, generator=generator),
        caption_present=torch.ones(count, dtype=torch.bool),
        numeric_features=torch.randn(count, 4, generator=generator),
        upload_type_indices=torch.tensor([1 + row % 2 for row in range(count)]),
    )


def _model() -> TwoTowerV1:
    torch.manual_seed(7)
    return TwoTowerV1(
        num_items=8,
        num_users=4,
        num_category_tokens=8,
        num_upload_types=2,
    )


def test_tower_shapes_norms_weighted_mean_and_empty_history_are_finite():
    model = _model()
    items = model.encode_items(
        _features(), use_id_embedding=torch.ones(8, dtype=torch.bool)
    )
    assert items.shape == (8, 128)
    torch.testing.assert_close(items.norm(dim=1), torch.ones(8), atol=1e-5, rtol=0)

    history = torch.stack((items[:2], items[2:4]))
    weights = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [False, False]])
    pooled = model.weighted_history_mean(history, weights, mask)
    torch.testing.assert_close(pooled[0], (items[0] + 3 * items[1]) / 4)
    torch.testing.assert_close(pooled[1], torch.zeros(128))
    users = model.encode_users(
        user_indices=torch.tensor([1, 2]),
        history_vectors=history,
        history_weights=weights,
        padding_mask=mask,
        use_id_embedding=torch.tensor([True, False]),
    )
    assert users.shape == (2, 128)
    assert torch.isfinite(users).all()
    torch.testing.assert_close(users.norm(dim=1), torch.ones(2), atol=1e-5, rtol=0)


def test_cold_item_and_history_only_user_ignore_untrained_id_rows():
    model = _model().eval()
    features = _features(2)
    with torch.no_grad():
        before_item = model.encode_items(
            features, use_id_embedding=torch.tensor([False, False])
        )
        model.item_id_embedding.weight[1:].add_(1000)
        after_item = model.encode_items(
            features, use_id_embedding=torch.tensor([False, False])
        )
    torch.testing.assert_close(before_item, after_item)

    history = before_item.unsqueeze(0)
    with torch.no_grad():
        before_user = model.encode_users(
            user_indices=torch.tensor([1]),
            history_vectors=history,
            history_weights=torch.ones(1, 2),
            padding_mask=torch.ones(1, 2, dtype=torch.bool),
            use_id_embedding=torch.tensor([False]),
        )
        model.user_id_embedding.weight[1].add_(1000)
        after_user = model.encode_users(
            user_indices=torch.tensor([1]),
            history_vectors=history,
            history_weights=torch.ones(1, 2),
            padding_mask=torch.ones(1, 2, dtype=torch.bool),
            use_id_embedding=torch.tensor([False]),
        )
    torch.testing.assert_close(before_user, after_user)


def test_target_history_and_candidate_share_one_item_tower_and_mask_contract():
    model = _model().eval()
    features = _features(3)
    use_id = torch.ones(3, dtype=torch.bool)
    with torch.no_grad():
        target = model.encode_items(features, use_id_embedding=use_id)
        history = model.encode_items(features, use_id_embedding=use_id)
        candidate = model.encode_items(features, use_id_embedding=use_id)
    torch.testing.assert_close(target, history)
    torch.testing.assert_close(target, candidate)

    mask = build_in_batch_logit_mask(
        np.asarray([1, 2, 1]),
        np.asarray([10, 10, 30]),
        {1: frozenset({10, 30}), 2: frozenset({10})},
    )
    assert np.all(np.diag(mask))
    assert mask[0, 1] == 0  # repeated target
    assert mask[0, 2] == 0  # same user's other known positive


def test_all_covered_paths_have_gradients_and_false_negative_mask_is_respected():
    model = _model()
    features = _features()
    targets = model.encode_items(
        features.select(torch.tensor([0, 1, 2, 3])),
        use_id_embedding=torch.ones(4, dtype=torch.bool),
    )
    histories = model.encode_items(
        features.select(torch.tensor([4, 5, 6, 7])),
        use_id_embedding=torch.ones(4, dtype=torch.bool),
    ).reshape(4, 1, 128)
    histories.retain_grad()
    users = model.encode_users(
        user_indices=torch.arange(1, 5),
        history_vectors=histories,
        history_weights=torch.ones(4, 1),
        padding_mask=torch.ones(4, 1, dtype=torch.bool),
        use_id_embedding=torch.ones(4, dtype=torch.bool),
    )
    allowed = torch.ones(4, 4, dtype=torch.bool)
    allowed[0, 1] = False
    loss, stats = masked_in_batch_cross_entropy(
        users, targets, allowed, temperature=0.07
    )
    loss.backward()
    for module in (
        model.item_id_embedding,
        model.category_embedding,
        model.caption_projection,
        model.static_projection,
        model.upload_type_embedding,
        model.user_id_embedding,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.norm() > 0
            for parameter in module.parameters()
        )
    assert histories.grad is not None and histories.grad.norm() > 0
    assert stats["off_diagonal_masked_fraction"] == pytest.approx(1 / 12)
    bad = allowed.clone()
    bad[0] = False
    bad[0, 0] = True
    with pytest.raises(ValueError, match="at least one valid negative"):
        masked_in_batch_cross_entropy(users, targets, bad, temperature=0.07)


def _checkpoint_identity() -> dict:
    return {
        "schema_version": 2,
        "config": {"locator": "config", "sha256": "a" * 64},
        "processed_manifest_sha256": "b" * 64,
        "raw_inputs": {
            name: {
                "source_locator": f"KUAIREC_DATA_DIR/{name}",
                "actual_sha256": character * 64,
                "expected_sha256": character * 64,
                "sha256_match": True,
            }
            for name, character in (
                ("big_matrix.csv", "c"),
                ("item_daily_features.csv", "d"),
                ("kuairec_caption_category.csv", "e"),
            )
        },
        "code_commit": "f" * 40,
        "ordered_item_store": {"count": 8, "sha256": "1" * 64},
        "ordered_user_position_mapping": {
            "count": 4,
            "sha256": (
                "3a1cdc165e726031712a6b9f9d331f931965eb4c8408fd84da45dd13015b6837"
            ),
            "hash_scheme": (
                "sha256(phase-b2a-ordered-user-position-mapping-v1\\n + "
                "sorted-unique-decimal-id\\n)"
            ),
        },
        "memberships": {
            "normal": {"count": 8, "sha256": "2" * 64},
            "fixed_retrieval_catalog": {"count": 8, "sha256": "3" * 64},
            "model_item_universe": {"count": 8, "sha256": "4" * 64},
        },
        "feature_identity": {
            "category_vocab_sha256": "5" * 64,
            "upload_type_vocab_sha256": "6" * 64,
            "numeric_preprocessing_sha256": "7" * 64,
        },
        "caption_identity": {
            "model_id": "toy",
            "resolved_revision": "8" * 40,
            "item_membership_sha256": "9" * 64,
            "embedding_payload_sha256": "a" * 64,
        },
        "actual_touched_membership": {
            "users": {
                "count": 4,
                "sha256": (
                    "ef9fefe661841a7ee4cfe88838d19ef5a3da0aa5c7ad8251cc1ec2ed "
                ).replace(" ", "0")[:64],
            },
            "items": {"count": 8, "sha256": "b" * 64},
        },
    }


def _identity_for_payload(
    ordered_users: np.ndarray,
    touched_users: np.ndarray,
    touched_items: np.ndarray,
) -> dict:
    from kuairec_fully_observed.provenance import membership_record
    from kuairec_fully_observed.torch_training import (
        stable_int_membership_sha256,
    )

    identity = _checkpoint_identity()
    identity["ordered_user_position_mapping"] = membership_record(
        ordered_users,
        label="phase-b2a-ordered-user-position-mapping-v1",
    )
    identity["actual_touched_membership"] = {
        "users": {
            "count": len(touched_users),
            "sha256": stable_int_membership_sha256(
                "phase-b2a-touched-users-v1", touched_users
            ),
        },
        "items": {
            "count": len(touched_items),
            "sha256": stable_int_membership_sha256(
                "phase-b2a-touched-items-v1", touched_items
            ),
        },
    }
    return identity


def test_toy_batch_overfits_and_cpu_checkpoint_round_trip(tmp_path):
    model = _model()
    features = _features(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    allowed = torch.ones(4, 4, dtype=torch.bool)

    def forward():
        items = model.encode_items(
            features, use_id_embedding=torch.ones(4, dtype=torch.bool)
        )
        users = model.encode_users(
            user_indices=torch.arange(1, 5),
            history_vectors=items.roll(1, 0).reshape(4, 1, 128),
            history_weights=torch.ones(4, 1),
            padding_mask=torch.ones(4, 1, dtype=torch.bool),
            use_id_embedding=torch.ones(4, dtype=torch.bool),
        )
        return masked_in_batch_cross_entropy(
            users, items, allowed, temperature=0.07
        )

    initial, _ = forward()
    for _ in range(80):
        optimizer.zero_grad()
        loss, _ = forward()
        loss.backward()
        optimizer.step()
        model._zero_padding_rows()
    final, stats = forward()
    assert final < initial * 0.2
    assert stats["diagonal_top1_rate"] == 1.0

    path = tmp_path / "model.pt"
    dimensions = {
        "num_items": 8,
        "num_users": 4,
        "num_category_tokens": 8,
        "num_upload_types": 2,
    }
    ordered_users = np.arange(4, dtype=np.int64)
    touched_items = np.arange(8, dtype=np.int64)
    identity = _identity_for_payload(
        ordered_users, ordered_users, touched_items
    )
    save_checkpoint(
        path,
        model=model,
        model_dimensions=dimensions,
        ordered_user_ids=ordered_users,
        touched_user_ids=ordered_users,
        touched_item_ids=touched_items,
        identity=identity,
    )
    restored, payload = load_checkpoint(
        path, device="cpu", expected_identity=identity
    )
    assert payload["model_dimensions"] == dimensions
    assert {parameter.device.type for parameter in restored.parameters()} == {
        "cpu"
    }
    with torch.no_grad():
        original = model.encode_items(
            features, use_id_embedding=torch.ones(4, dtype=torch.bool)
        )
        loaded = restored.encode_items(
            features, use_id_embedding=torch.ones(4, dtype=torch.bool)
        )
    torch.testing.assert_close(original, loaded)

    changed = dict(identity)
    changed["code_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_checkpoint(path, device="cpu", expected_identity=changed)

    for section, key in (
        ("config", "sha256"),
        ("ordered_item_store", "sha256"),
        ("ordered_user_position_mapping", "sha256"),
        ("feature_identity", "category_vocab_sha256"),
        ("caption_identity", "embedding_payload_sha256"),
    ):
        changed = copy.deepcopy(identity)
        changed[section][key] = "0" * 64
        with pytest.raises(RuntimeError, match="identity mismatch"):
            load_checkpoint(path, device="cpu", expected_identity=changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_checkpoint_round_trip_places_model_and_inputs_on_cuda(tmp_path):
    model = _model().cuda()
    dimensions = {
        "num_items": 8,
        "num_users": 4,
        "num_category_tokens": 8,
        "num_upload_types": 2,
    }
    ordered_users = np.arange(4, dtype=np.int64)
    touched_items = np.arange(8, dtype=np.int64)
    identity = _identity_for_payload(
        ordered_users, ordered_users, touched_items
    )
    path = tmp_path / "cuda.pt"
    save_checkpoint(
        path,
        model=model,
        model_dimensions=dimensions,
        ordered_user_ids=ordered_users,
        touched_user_ids=ordered_users,
        touched_item_ids=touched_items,
        identity=identity,
    )
    restored, _ = load_checkpoint(
        path, device="cuda", expected_identity=identity
    )
    features = TorchItemFeatures(
        item_indices=_features(2).item_indices.cuda(),
        category_indices=_features(2).category_indices.cuda(),
        caption_embeddings=_features(2).caption_embeddings.cuda(),
        caption_present=_features(2).caption_present.cuda(),
        numeric_features=_features(2).numeric_features.cuda(),
        upload_type_indices=_features(2).upload_type_indices.cuda(),
    )
    with torch.inference_mode():
        output = restored.encode_items(
            features, use_id_embedding=torch.ones(2, dtype=torch.bool, device="cuda")
        )
    assert output.device.type == "cuda"


def test_skipped_batch_ids_are_not_recorded_as_touched():
    successful = TrainingBatch(
        user_ids=np.asarray([1, 2]),
        target_item_ids=np.asarray([10, 20]),
        history_item_ids=np.asarray([[30], [40]]),
        history_weights=torch.ones(2, 1),
        history_mask=torch.ones(2, 1, dtype=torch.bool),
        allowed_logits=torch.ones(2, 2, dtype=torch.bool),
    )
    skipped = TrainingBatch(
        user_ids=np.asarray([99]),
        target_item_ids=np.asarray([999]),
        history_item_ids=np.asarray([[998]]),
        history_weights=torch.ones(1, 1),
        history_mask=torch.ones(1, 1, dtype=torch.bool),
        allowed_logits=torch.ones(1, 1, dtype=torch.bool),
    )
    users: set[int] = set()
    items: set[int] = set()
    record_successful_step_membership(
        successful, touched_users=users, touched_items=items
    )
    assert users == {1, 2}
    assert items == {10, 20, 30, 40}
    assert 99 not in users
    assert {998, 999}.isdisjoint(items)

    model = _model().eval()
    cold = _features(1)
    with torch.inference_mode():
        before = model.encode_items(
            cold, use_id_embedding=torch.tensor([999 in items])
        )
        model.item_id_embedding.weight[1].add_(500)
        after = model.encode_items(
            cold, use_id_embedding=torch.tensor([999 in items])
        )
    torch.testing.assert_close(before, after)


def test_precomputed_query_encoding_matches_naive_reference():
    model = _model().eval()
    features = _features(4)
    store = PreparedItemFeatureStore(
        item_ids=np.arange(1, 5, dtype=np.int64),
        category_indices=features.category_indices.numpy(),
        caption_embeddings=features.caption_embeddings.numpy(),
        caption_present=features.caption_present.numpy(),
        numeric_features=features.numeric_features.numpy(),
        upload_type_indices=features.upload_type_indices.numpy(),
        category_vocab={},
        upload_type_vocab={},
        preprocessing={},
    )
    touched_items = {1, 2, 3}
    touched_users = {10, 11}
    user_positions = {10: 1, 11: 2}
    histories = (
        np.asarray([1, 4], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
    )
    weights = (
        np.asarray([1.0, 0.5], dtype=np.float32),
        np.asarray([0.25, 1.0], dtype=np.float32),
    )
    precomputed = preencode_item_universe(
        model=model,
        store=store,
        touched_item_ids=touched_items,
        device="cpu",
        batch_size=2,
    )
    optimized = encode_query_users_from_precomputed(
        model=model,
        store=store,
        precomputed_item_vectors=precomputed,
        user_ids=np.asarray([10, 11]),
        histories=histories,
        history_weights=weights,
        user_positions=user_positions,
        touched_user_ids=touched_users,
        device="cpu",
        batch_size=1,
    )
    torch_store = store.torch_features(torch.device("cpu"))
    with torch.inference_mode():
        naive_histories = torch.stack(
            [
                encode_item_ids(
                    model,
                    store,
                    torch_store,
                    history,
                    touched_item_ids=touched_items,
                    device=torch.device("cpu"),
                )
                for history in histories
            ]
        )
        naive = model.encode_users(
            user_indices=torch.tensor([1, 2]),
            history_vectors=naive_histories,
            history_weights=torch.tensor(np.asarray(weights)),
            padding_mask=torch.ones(2, 2, dtype=torch.bool),
            use_id_embedding=torch.ones(2, dtype=torch.bool),
        )
    torch.testing.assert_close(optimized, naive)
    assert optimized.grad_fn is None
    assert precomputed.grad_fn is None
