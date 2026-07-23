from __future__ import annotations

import copy
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from kuairec_fully_observed.caption_embeddings import CaptionCache  # noqa: E402
from kuairec_fully_observed.data import RetrievalQueries  # noqa: E402
from kuairec_fully_observed.full_training import (  # noqa: E402
    build_checkpoint_identity,
    evaluate_frozen_gates,
    load_full_epoch_checkpoint,
    planned_training_membership,
    save_full_epoch_checkpoint,
    select_checkpoint_epoch,
    train_full_two_tower,
    validation_query_contract_sha256,
    verify_validation_contract,
)
from kuairec_fully_observed.provenance import membership_record  # noqa: E402
from kuairec_fully_observed.torch_models import TwoTowerV1  # noqa: E402
from kuairec_fully_observed.torch_training import (  # noqa: E402
    prepare_item_feature_store,
)
from kuairec_fully_observed.training import (  # noqa: E402
    build_two_tower_training_dataset,
)
from scripts.run_phase_b2b_full_two_tower import (  # noqa: E402
    validate_config,
)


def _toy_setup():
    rows = []
    item_ids = np.arange(10, 26, dtype=np.int64)
    for user in range(1, 5):
        for offset in range(4):
            rows.append(
                {
                    "user_id": user,
                    "video_id": int(item_ids[(user - 1) * 4 + offset]),
                    "timestamp": float(offset),
                    "play_duration": 4000.0 + offset,
                    "video_duration": 1000.0,
                    "watch_ratio": 3.0,
                }
            )
    events = pd.DataFrame(rows)
    dataset = build_two_tower_training_dataset(
        events, max_history=3, normal_item_ids=item_ids
    )
    static = pd.DataFrame(
        {
            "video_id": item_ids,
            "caption_text": [f"item {value}" for value in item_ids],
            "category_ids": [
                (1 + index % 2, 3 + index % 3, 7)
                for index in range(len(item_ids))
            ],
            "video_duration": np.full(len(item_ids), 1000.0),
            "video_width": np.full(len(item_ids), 720.0),
            "video_height": np.full(len(item_ids), 1280.0),
            "upload_type": ["A" if index % 2 else "B" for index in range(len(item_ids))],
            "upload_dt": ["2020-01-01"] * len(item_ids),
        }
    )
    generator = np.random.default_rng(7)
    caption = CaptionCache(
        item_ids=item_ids,
        embeddings=generator.normal(
            size=(len(item_ids), 384)
        ).astype(np.float32),
        metadata={},
    )
    store = prepare_item_feature_store(
        static_frame=static,
        caption_cache=caption,
        item_universe=item_ids,
        train_observed_item_ids=item_ids,
        train_observed_normal_item_ids=item_ids,
    )
    example_indices = np.arange(len(dataset), dtype=np.int64)
    ordered_users, planned_items = planned_training_membership(
        dataset, example_indices
    )
    dimensions = {
        "num_items": len(item_ids),
        "num_users": len(ordered_users),
        "num_category_tokens": len(store.category_vocab),
        "num_upload_types": len(store.upload_type_vocab),
    }
    base_identity = {
        "config": {"locator": "toy", "sha256": "a" * 64},
        "processed_manifest_sha256": "b" * 64,
        "raw_inputs": {},
        "code_commit": "c" * 40,
        "memberships": {
            "normal": membership_record(item_ids, label="toy-normal"),
            "fixed_retrieval_catalog": membership_record(
                item_ids, label="toy-catalog"
            ),
            "model_item_universe": membership_record(
                item_ids, label="toy-universe"
            ),
        },
        "feature_identity": {
            "category_vocab_count": len(store.category_vocab),
            "category_vocab_sha256": "d" * 64,
            "upload_type_vocab_count": len(store.upload_type_vocab),
            "upload_type_vocab_sha256": "e" * 64,
            "numeric_preprocessing_sha256": "f" * 64,
        },
        "caption_identity": {
            "model_id": "toy",
            "resolved_revision": "1" * 40,
            "item_membership_sha256": "2" * 64,
            "embedding_payload_sha256": "3" * 64,
        },
    }
    return (
        dataset,
        store,
        example_indices,
        ordered_users,
        planned_items,
        dimensions,
        base_identity,
    )


def _new_model_optimizer(dimensions):
    torch.manual_seed(20260722)
    model = TwoTowerV1(**dimensions)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.00001
    )
    return model, optimizer


def _train_kwargs(setup):
    dataset, store, indices, users, items, _, _ = setup
    return {
        "dataset": dataset,
        "example_indices": indices,
        "store": store,
        "ordered_user_ids": users,
        "planned_item_ids": items,
        "device": "cpu",
        "seed": 20260722,
        "diagnostic_seed": 20260723,
        "batch_size": 8,
        "temperature": 0.07,
        "gradient_clip_norm": 5.0,
        "log_every_steps": 100,
    }


def _optimizer_tensors(state):
    values = []
    for key in sorted(state["state"]):
        for name in sorted(state["state"][key]):
            value = state["state"][key][name]
            if torch.is_tensor(value):
                values.append((key, name, value.detach().cpu()))
            else:
                values.append((key, name, value))
    return values


def _resume_worker(checkpoint: str, output: str) -> None:
    setup = _toy_setup()
    _, _, _, users, _, dimensions, base = setup
    inspection = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = build_checkpoint_identity(
        base_identity=base,
        model_dimensions=dimensions,
        ordered_item_ids=setup[1].item_ids,
        ordered_user_ids=users,
        touched_user_ids=inspection["touched_user_ids"],
        touched_item_ids=inspection["touched_item_ids"],
        training_seed=20260722,
    )
    model, optimizer, restored = load_full_epoch_checkpoint(
        Path(checkpoint),
        device="cpu",
        expected_identity=identity,
        learning_rate=0.001,
        weight_decay=0.00001,
    )
    result = train_full_two_tower(
        model=model,
        optimizer=optimizer,
        start_epoch=2,
        end_epoch=3,
        prior_epoch_losses=restored["epoch_losses"],
        touched_user_ids=restored["touched_user_ids"],
        touched_item_ids=restored["touched_item_ids"],
        **_train_kwargs(setup),
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_losses": result["epoch_losses"],
            "touched_user_ids": result["touched_user_ids"],
            "touched_item_ids": result["touched_item_ids"],
            "item_vectors": model.encode_items(
                setup[1].torch_features(torch.device("cpu")),
                use_id_embedding=torch.ones(
                    len(setup[1].item_ids), dtype=torch.bool
                ),
            ).detach(),
        },
        output,
    )


def test_frozen_config_and_selection_gate_contracts():
    config = yaml.safe_load(
        Path("configs/phase_b2b_full_two_tower.yaml").read_text()
    )
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(RuntimeError, match="training configuration"):
        validate_config(changed)
    records = [
        {
            "epoch": 1,
            "metrics": {"Recall@100": 0.04, "NDCG@20": 0.02},
        },
        {
            "epoch": 2,
            "metrics": {"Recall@100": 0.05, "NDCG@20": 0.01},
        },
        {
            "epoch": 3,
            "metrics": {"Recall@100": 0.05, "NDCG@20": 0.01},
        },
    ]
    assert select_checkpoint_epoch(records) == 2
    nested = [
        {"epoch": row["epoch"], "validation": {"metrics": row["metrics"]}}
        for row in records
    ]
    assert select_checkpoint_epoch(nested) == 2
    gates = evaluate_frozen_gates(
        {
            "Recall@100": 0.051,
            "NDCG@20": 0.003,
            "Coverage@100": 0.39,
            "Data-Cold Recall@100": 0.051,
        },
        {"data_cold_target_count": 100},
        config["gate"],
    )
    assert gates == {"A": True, "B": True, "C": True}


def test_validation_contract_hash_and_fail_closed_counts():
    queries = RetrievalQueries(
        user_ids=np.asarray([1, 2]),
        histories=(np.asarray([10]), np.asarray([], dtype=np.int64)),
        history_weights=(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        ),
        candidates=(np.asarray([20, 30]), np.asarray([10, 30])),
        relevant=(np.asarray([30]), np.asarray([10])),
        catalog=np.asarray([10, 20, 30]),
        warm_user_mask=np.asarray([True, False]),
    )
    digest = validation_query_contract_sha256(queries)
    counts = {
        "query_count": 2,
        "warm_query_count": 1,
        "target_count": 2,
        "warm_target_count": 1,
        "data_cold_item_count": 1,
        "query_contract_sha256": digest,
    }
    expected = {
        "fixed_catalog_count": 3,
        "fixed_catalog_sha256": membership_record(
            queries.catalog,
            label="phase-b2a-fixed-retrieval-catalog-v1",
        )["sha256"],
        **counts,
    }
    verify_validation_contract(
        queries=queries, counts=counts, expected=expected
    )
    changed = dict(expected)
    changed["query_contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="query_contract_sha256"):
        verify_validation_contract(
            queries=queries, counts=counts, expected=changed
        )


def test_canonical_labels_override_representative_row_labels():
    events = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 20],
            "timestamp": [1.0, 2.0],
            "play_duration": [100.0, 4000.0],
            "video_duration": [1000.0, 1000.0],
            "watch_ratio": [3.0, 3.0],
            "_is_strong_positive": [False, True],
            "_is_quick_skip": [False, False],
        }
    )
    dataset = build_two_tower_training_dataset(
        events, max_history=5, normal_item_ids=np.asarray([10, 20])
    )
    assert len(dataset) == 1
    example = dataset[0]
    assert example.target_item_id == 20
    np.testing.assert_array_equal(example.history, [10])
    # Canonical quick-skip=false wins over the representative row's duration.
    np.testing.assert_allclose(example.history_weights, [3.0])


def test_three_epochs_equal_one_epoch_then_new_process_resume(tmp_path):
    continuous_setup = _toy_setup()
    continuous_model, continuous_optimizer = _new_model_optimizer(
        continuous_setup[5]
    )
    continuous = train_full_two_tower(
        model=continuous_model,
        optimizer=continuous_optimizer,
        start_epoch=1,
        end_epoch=3,
        **_train_kwargs(continuous_setup),
    )

    resumed_setup = _toy_setup()
    resumed_model, resumed_optimizer = _new_model_optimizer(resumed_setup[5])
    checkpoint = tmp_path / "epoch_001.pt"

    def save_epoch(
        epoch, model, optimizer, losses, touched_users, touched_items
    ):
        identity = build_checkpoint_identity(
            base_identity=resumed_setup[6],
            model_dimensions=resumed_setup[5],
            ordered_item_ids=resumed_setup[1].item_ids,
            ordered_user_ids=resumed_setup[3],
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            training_seed=20260722,
        )
        save_full_epoch_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch,
            epoch_losses=losses,
            order_seed=20260722,
            model_dimensions=resumed_setup[5],
            ordered_item_ids=resumed_setup[1].item_ids,
            ordered_user_ids=resumed_setup[3],
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            identity=identity,
        )

    first = train_full_two_tower(
        model=resumed_model,
        optimizer=resumed_optimizer,
        start_epoch=1,
        end_epoch=1,
        checkpoint_callback=save_epoch,
        **_train_kwargs(resumed_setup),
    )
    assert first["completed_epoch"] == 1
    output = tmp_path / "resumed.pt"
    process = multiprocessing.get_context("spawn").Process(
        target=_resume_worker, args=(str(checkpoint), str(output))
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    resumed = torch.load(output, map_location="cpu", weights_only=False)
    np.testing.assert_allclose(
        continuous["epoch_losses"], resumed["epoch_losses"], rtol=0, atol=0
    )
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(value, resumed["state_dict"][name], rtol=0, atol=0)
    continuous_optimizer_values = _optimizer_tensors(
        continuous_optimizer.state_dict()
    )
    resumed_optimizer_values = _optimizer_tensors(resumed["optimizer"])
    assert len(continuous_optimizer_values) == len(resumed_optimizer_values)
    for expected, actual in zip(
        continuous_optimizer_values, resumed_optimizer_values, strict=True
    ):
        assert expected[:2] == actual[:2]
        if torch.is_tensor(expected[2]):
            torch.testing.assert_close(
                expected[2], actual[2], rtol=0, atol=0
            )
        else:
            assert expected[2] == actual[2]
    np.testing.assert_array_equal(
        continuous["touched_user_ids"], resumed["touched_user_ids"]
    )
    np.testing.assert_array_equal(
        continuous["touched_item_ids"], resumed["touched_item_ids"]
    )
    with torch.inference_mode():
        continuous_vectors = continuous_model.encode_items(
            continuous_setup[1].torch_features(torch.device("cpu")),
            use_id_embedding=torch.ones(
                len(continuous_setup[1].item_ids), dtype=torch.bool
            ),
        )
    torch.testing.assert_close(
        continuous_vectors, resumed["item_vectors"], rtol=0, atol=0
    )


def test_incomplete_or_identity_mismatched_checkpoint_is_rejected(tmp_path):
    setup = _toy_setup()
    model, optimizer = _new_model_optimizer(setup[5])
    path = tmp_path / "epoch.pt"

    def save_epoch(epoch, current, opt, losses, touched_users, touched_items):
        identity = build_checkpoint_identity(
            base_identity=setup[6],
            model_dimensions=setup[5],
            ordered_item_ids=setup[1].item_ids,
            ordered_user_ids=setup[3],
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            training_seed=20260722,
        )
        save_full_epoch_checkpoint(
            path,
            model=current,
            optimizer=opt,
            completed_epoch=epoch,
            epoch_losses=losses,
            order_seed=20260722,
            model_dimensions=setup[5],
            ordered_item_ids=setup[1].item_ids,
            ordered_user_ids=setup[3],
            touched_user_ids=touched_users,
            touched_item_ids=touched_items,
            identity=identity,
        )

    result = train_full_two_tower(
        model=model,
        optimizer=optimizer,
        start_epoch=1,
        end_epoch=1,
        checkpoint_callback=save_epoch,
        **_train_kwargs(setup),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = build_checkpoint_identity(
        base_identity=setup[6],
        model_dimensions=setup[5],
        ordered_item_ids=setup[1].item_ids,
        ordered_user_ids=setup[3],
        touched_user_ids=result["touched_user_ids"],
        touched_item_ids=result["touched_item_ids"],
        training_seed=20260722,
    )
    payload["completed_epoch"] = 2
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="epoch/order contract"):
        load_full_epoch_checkpoint(
            path,
            device="cpu",
            expected_identity=identity,
            learning_rate=0.001,
            weight_decay=0.00001,
        )
    changed = copy.deepcopy(identity)
    changed["code_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_full_epoch_checkpoint(
            path,
            device="cpu",
            expected_identity=changed,
            learning_rate=0.001,
            weight_decay=0.00001,
        )
