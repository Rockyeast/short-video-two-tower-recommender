from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kuairec_fully_observed.torch_models import (  # noqa: E402
    TorchItemFeatures,
    TwoTowerV1,
    masked_in_batch_cross_entropy,
)
from kuairec_fully_observed.torch_training import (  # noqa: E402
    load_checkpoint,
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


def test_toy_batch_overfits_and_checkpoint_round_trip(tmp_path):
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
    save_checkpoint(
        path,
        model=model,
        model_dimensions=dimensions,
        touched_user_ids=np.arange(4),
        touched_item_ids=np.arange(8),
    )
    restored, payload = load_checkpoint(path)
    assert payload["model_dimensions"] == dimensions
    with torch.no_grad():
        original = model.encode_items(
            features, use_id_embedding=torch.ones(4, dtype=torch.bool)
        )
        loaded = restored.encode_items(
            features, use_id_embedding=torch.ones(4, dtype=torch.bool)
        )
    torch.testing.assert_close(original, loaded)
