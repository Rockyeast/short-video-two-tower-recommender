from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from kuairec_fully_observed.caption_embeddings import CaptionCache  # noqa: E402
from kuairec_fully_observed.data import RetrievalQueries  # noqa: E402
from kuairec_fully_observed.models import PopularityBaseline  # noqa: E402
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
    _completed_checkpoint_result,
    _evaluate_model,
    _report_mode,
    _write_markdown,
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


FROZEN_MUTATION_PATHS = [
    ("training_contract", "full_example_count"),
    ("training_contract", "training_user_count"),
    ("checkpoint", "epochs"),
    ("checkpoint", "directory"),
    ("validation", "k"),
    ("validation", "item_encoding_batch_size"),
    ("validation", "user_encoding_batch_size"),
    ("validation", "score_block_size"),
    ("validation", "expected", "fixed_catalog_count"),
    ("validation", "expected", "fixed_catalog_sha256"),
    ("validation", "expected", "query_count"),
    ("validation", "expected", "warm_query_count"),
    ("validation", "expected", "target_count"),
    ("validation", "expected", "warm_target_count"),
    ("validation", "expected", "data_cold_item_count"),
    ("validation", "expected", "query_contract_sha256"),
    ("selection", "primary"),
    ("selection", "tie_break"),
    ("selection", "final_tie_break"),
    ("frozen_bpr_epoch_20", "Recall@100"),
    ("frozen_bpr_epoch_20", "NDCG@20"),
    ("frozen_bpr_epoch_20", "Coverage@100"),
    ("frozen_bpr_epoch_20", "Data-Cold Recall@100"),
    ("gate", "common_ndcg_minimum"),
    ("gate", "A", "Recall@100_minimum"),
    ("gate", "B", "Recall@100_minimum"),
    ("gate", "B", "Coverage@100_minimum"),
    ("gate", "C", "Recall@100_minimum"),
    ("gate", "C", "data_cold_target_denominator_minimum"),
    ("gate", "C", "Data-Cold_Recall@100_minimum"),
    ("preflight", "claims", "formal_gate_executed"),
    ("preflight", "claims", "effectiveness_claim"),
    ("preflight", "claims", "full_big_train"),
    ("preflight", "claims", "full_big_validation"),
]


def _mutate_path(config, path):
    changed = copy.deepcopy(config)
    cursor = changed
    for name in path[:-1]:
        cursor = cursor[name]
    value = cursor[path[-1]]
    if isinstance(value, bool):
        cursor[path[-1]] = not value
    elif isinstance(value, (int, float)):
        cursor[path[-1]] = value + 1
    elif isinstance(value, str):
        cursor[path[-1]] = value + "-changed"
    elif isinstance(value, list):
        cursor[path[-1]] = list(reversed(value))
    else:
        raise AssertionError(f"Unsupported test mutation: {path}")
    return changed


def _minimal_report(preflight):
    mode = _report_mode(preflight)
    return {
        "phase": mode["phase"],
        "claim_boundary": mode["claim_boundary"],
        "environment": {"device": "cpu"},
        "runtime_s": 1.0,
        "peak_rss_mb": 10.0,
        "resume": {"verified": True},
        "training": {
            "example_count": 12,
            "process_statistics": {
                "optimizer_steps": 0 if not preflight else 2,
                "skipped_batches": 0,
                "completed_examples": 0 if not preflight else 12,
            },
            "cumulative_statistics": {
                "optimizer_steps": 6,
                "skipped_batches": 1,
                "completed_examples": 36,
            },
        },
        "validation": {"evaluated_queries": 4},
        "estimated_full_run_minutes": {"low": 1.0, "high": 2.0},
        "checkpoints": [
            {
                "epoch": epoch,
                "epoch_loss": 1.0 / epoch,
                "validation": {
                    "metrics": {
                        "Recall@100": 0.01 * epoch,
                        "NDCG@20": 0.001 * epoch,
                        "Coverage@100": 0.1 * epoch,
                    }
                },
            }
            for epoch in (1, 2, 3)
        ],
    }


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
        prior_optimizer_steps=restored["cumulative_training_statistics"][
            "optimizer_steps"
        ],
        prior_skipped_batches=restored["cumulative_training_statistics"][
            "skipped_batches"
        ],
        prior_completed_examples=restored[
            "cumulative_training_statistics"
        ]["completed_examples"],
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
            "process_statistics": result["process_statistics"],
            "cumulative_statistics": result["cumulative_statistics"],
            "item_vectors": model.encode_items(
                setup[1].torch_features(torch.device("cpu")),
                use_id_embedding=torch.ones(
                    len(setup[1].item_ids), dtype=torch.bool
                ),
            ).detach(),
        },
        output,
    )


def _epoch3_finalize_worker(checkpoint_dir: str, output: str) -> None:
    setup = _toy_setup()
    _, store, _, users, _, dimensions, base = setup
    final_path = Path(checkpoint_dir) / "epoch_003.pt"
    final_inspection = torch.load(
        final_path, map_location="cpu", weights_only=False
    )
    final_identity = build_checkpoint_identity(
        base_identity=base,
        model_dimensions=dimensions,
        ordered_item_ids=store.item_ids,
        ordered_user_ids=users,
        touched_user_ids=final_inspection["touched_user_ids"],
        touched_item_ids=final_inspection["touched_item_ids"],
        training_seed=20260722,
    )
    _, _, restored = load_full_epoch_checkpoint(
        final_path,
        device="cpu",
        expected_identity=final_identity,
        learning_rate=0.001,
        weight_decay=0.00001,
    )
    completed = _completed_checkpoint_result(restored)
    histories = tuple(
        np.asarray([10 + 4 * row], dtype=np.int64)
        for row in range(len(users))
    )
    queries = RetrievalQueries(
        user_ids=users.copy(),
        histories=histories,
        history_weights=tuple(
            np.ones(1, dtype=np.float32) for _ in users
        ),
        candidates=tuple(store.item_ids.copy() for _ in users),
        relevant=tuple(
            np.asarray([11 + 4 * row], dtype=np.int64)
            for row in range(len(users))
        ),
        catalog=store.item_ids.copy(),
        warm_user_mask=np.ones(len(users), dtype=bool),
    )
    popularity = PopularityBaseline(
        {int(item): float(index) for index, item in enumerate(store.item_ids)}
    )
    evaluation_config = {
        "validation": {
            "k": 10,
            "item_encoding_batch_size": 8,
            "user_encoding_batch_size": 4,
            "score_block_size": 4,
        }
    }
    records = []
    for epoch in (1, 2, 3):
        path = Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        inspection = torch.load(path, map_location="cpu", weights_only=False)
        identity = build_checkpoint_identity(
            base_identity=base,
            model_dimensions=dimensions,
            ordered_item_ids=store.item_ids,
            ordered_user_ids=users,
            touched_user_ids=inspection["touched_user_ids"],
            touched_item_ids=inspection["touched_item_ids"],
            training_seed=20260722,
        )
        model, _, payload = load_full_epoch_checkpoint(
            path,
            device="cpu",
            expected_identity=identity,
            learning_rate=0.001,
            weight_decay=0.00001,
        )
        validation, timings = _evaluate_model(
            model=model,
            store=store,
            queries=queries,
            data_cold_items=np.asarray([], dtype=np.int64),
            ordered_user_ids=users,
            touched_user_ids=payload["touched_user_ids"],
            touched_item_ids=payload["touched_item_ids"],
            popularity=popularity,
            device=torch.device("cpu"),
            config=evaluation_config,
        )
        records.append(
            {
                "epoch": epoch,
                "epoch_loss": payload["epoch_losses"][-1],
                "validation": validation,
                "timings_s": timings,
            }
        )
    selected = select_checkpoint_epoch(records)
    report = _minimal_report(False)
    report["checkpoints"] = records
    report["training"]["process_statistics"] = completed[
        "process_statistics"
    ]
    report["training"]["cumulative_statistics"] = completed[
        "cumulative_statistics"
    ]
    markdown_path = Path(output).with_suffix(".md")
    _write_markdown(report, markdown_path)
    Path(output).write_text(
        json.dumps(
            {
                "process_statistics": completed["process_statistics"],
                "cumulative_statistics": completed["cumulative_statistics"],
                "reevaluated_epochs": [row["epoch"] for row in records],
                "selected_epoch": selected,
                "markdown": markdown_path.read_text(),
            },
            sort_keys=True,
        )
    )


def test_frozen_config_and_selection_gate_contracts():
    config = yaml.safe_load(
        Path("configs/phase_b2b_full_two_tower.yaml").read_text()
    )
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(RuntimeError, match="training is not frozen"):
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


@pytest.mark.parametrize("path", FROZEN_MUTATION_PATHS)
def test_each_frozen_execution_contract_value_fails_closed(path):
    config = yaml.safe_load(
        Path("configs/phase_b2b_full_two_tower.yaml").read_text()
    )
    with pytest.raises(RuntimeError):
        validate_config(_mutate_path(config, path))


@pytest.mark.parametrize(
    ("preflight", "expected_phase"),
    [
        (True, "phase-b2b0-full-runner-preflight"),
        (False, "phase-b2b-full-two-tower"),
    ],
)
def test_report_json_and_markdown_match_execution_mode(
    tmp_path, preflight, expected_phase
):
    report = _minimal_report(preflight)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, markdown_path)
    loaded = json.loads(json_path.read_text())
    markdown = markdown_path.read_text()
    assert loaded["phase"] == expected_phase
    if preflight:
        assert loaded["claim_boundary"] == {
            "formal_gate_executed": False,
            "effectiveness_claim": False,
            "full_big_train": False,
            "full_big_validation": False,
        }
        assert "bounded engineering preflight" in markdown
        assert "full_big_train=false" in markdown
        assert "full_big_validation=false" in markdown
    else:
        assert loaded["claim_boundary"] == {
            "formal_gate_executed": True,
            "effectiveness_claim": False,
            "full_big_train": True,
            "full_big_validation": True,
        }
        for forbidden in (
            "bounded preflight",
            "not a formal effectiveness experiment",
            "full_big_train=false",
            "full_big_validation=false",
        ):
            assert forbidden not in markdown
        assert "full_big_train=true" in markdown
        assert "full_big_validation=true" in markdown


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
        epoch,
        model,
        optimizer,
        losses,
        touched_users,
        touched_items,
        cumulative_statistics,
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
            cumulative_statistics=cumulative_statistics,
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
    assert (
        continuous["cumulative_statistics"]
        == resumed["cumulative_statistics"]
    )
    assert resumed["process_statistics"] == {
        "optimizer_steps": continuous["optimizer_steps"]
        - first["optimizer_steps"],
        "skipped_batches": continuous["skipped_batches"]
        - first["skipped_batches"],
        "completed_examples": continuous["completed_examples"]
        - first["completed_examples"],
    }
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


def test_epoch_three_resume_skips_training_and_reevaluates_all_checkpoints(
    tmp_path,
):
    setup = _toy_setup()
    model, optimizer = _new_model_optimizer(setup[5])
    checkpoint_dir = tmp_path / "checkpoints"

    def save_each_epoch(
        epoch,
        current,
        opt,
        losses,
        touched_users,
        touched_items,
        cumulative_statistics,
    ):
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
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
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
            cumulative_statistics=cumulative_statistics,
            identity=identity,
        )

    trained = train_full_two_tower(
        model=model,
        optimizer=optimizer,
        start_epoch=1,
        end_epoch=3,
        checkpoint_callback=save_each_epoch,
        **_train_kwargs(setup),
    )
    assert trained["completed_epoch"] == 3
    output = tmp_path / "epoch3-finalized.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_epoch3_finalize_worker,
        args=(str(checkpoint_dir), str(output)),
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    finalized = json.loads(output.read_text())
    assert finalized["process_statistics"] == {
        "optimizer_steps": 0,
        "skipped_batches": 0,
        "completed_examples": 0,
    }
    assert finalized["cumulative_statistics"] == trained[
        "cumulative_statistics"
    ]
    assert finalized["reevaluated_epochs"] == [1, 2, 3]
    assert finalized["selected_epoch"] in (1, 2, 3)
    assert "# Phase B2B Full Two-Tower Results" in finalized["markdown"]
    assert "full_big_train=true" in finalized["markdown"]
    assert "full_big_validation=true" in finalized["markdown"]


def test_incomplete_or_identity_mismatched_checkpoint_is_rejected(tmp_path):
    setup = _toy_setup()
    model, optimizer = _new_model_optimizer(setup[5])
    path = tmp_path / "epoch.pt"

    def save_epoch(
        epoch,
        current,
        opt,
        losses,
        touched_users,
        touched_items,
        cumulative_statistics,
    ):
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
            cumulative_statistics=cumulative_statistics,
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
