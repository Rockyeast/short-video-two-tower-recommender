"""Build and verify reusable Phase 1 processed artifacts exactly once."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from kuairec_protocol.access import ProtocolBundleVerification, sha256_file


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FILES = (
    "catalog.npz",
    "events_train_validation.npz",
    "targets_train.npz",
    "queries_validation.npz",
    "candidate_bits_validation.npy",
    "itemcf_user_item.npz",
    "itemcf_cooccurrence.npz",
    "bpr_negative_indices.npz",
)


class ArtifactError(RuntimeError):
    """Raised when a processed cache is incomplete, stale, or tampered."""


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ArtifactError("Cannot determine artifact generator commit")
    return completed.stdout.strip()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(root: Path, verification: ProtocolBundleVerification) -> dict[str, Any]:
    source_hashes = dict(verification.source_file_sha256)
    contract_hashes = dict(verification.contract_sha256)
    generator_paths = (
        "src/kuairec_phase1/artifacts.py",
        "scripts/audit_phase0.py",
        "configs/phase1_selection_plan.yaml",
    )
    return {
        "protocol_revision": verification.protocol_revision,
        "split_manifest_sha256": verification.manifest_sha256,
        "phase0_config_sha256": verification.config_sha256,
        "phase0_generation_code_sha256": verification.generation_code_sha256,
        "source_file_sha256": source_hashes,
        "contract_sha256": contract_hashes,
        "generator_file_sha256": {
            path: sha256_file(root / path) for path in generator_paths
        },
        "selection_code_commit": _git_head(root),
    }


def _artifact_root(root: Path, manifest_sha: str) -> Path:
    return root / "artifacts" / "phase1" / manifest_sha


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_phase0_module(root: Path):
    import importlib.util
    import sys
    import uuid

    path = root / "scripts/audit_phase0.py"
    name = f"_phase1_phase0_helpers_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArtifactError("Cannot load frozen Phase 0 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _data_file(root: Path, name: str) -> Path:
    matches = list((root / "data/raw").rglob(name))
    if len(matches) != 1:
        raise ArtifactError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]


def _pack_mask(mask: int, width: int) -> np.ndarray:
    return np.frombuffer(mask.to_bytes(width, "little"), dtype=np.uint8)


def _unpack_positions(row: np.ndarray, item_count: int) -> np.ndarray:
    return np.flatnonzero(np.unpackbits(row, bitorder="little")[:item_count])


def _canonical_inputs(
    root: Path,
    work: Path,
    verification: ProtocolBundleVerification,
) -> dict[str, Any]:
    phase0 = _load_phase0_module(root)
    manifest = json.loads((root / "manifests/split_manifest.json").read_text())
    boundaries = manifest["split_boundaries"]
    train_end = float(boundaries["train_end_exclusive"])
    validation_end = float(boundaries["validation_end_exclusive"])
    big_path = _data_file(root, "big_matrix.csv")
    daily_path = _data_file(root, "item_daily_features.csv")
    audit = json.loads((root / "reports/phase0/audit.json").read_text())
    timestamp_min = float(audit["files"]["big_matrix.csv"]["interaction"]["timestamp_min"])
    timestamp_max = float(audit["files"]["big_matrix.csv"]["interaction"]["timestamp_max"])
    catalog = phase0.load_candidate_catalog_policy(
        daily_path, timestamp_min, timestamp_max
    )
    video_ids = np.asarray(catalog["video_ids"], dtype=np.int32)
    video_index = catalog["video_index"]
    user_ids: list[int] = []
    event_users: list[np.ndarray] = []
    event_items: list[np.ndarray] = []
    event_times: list[np.ndarray] = []
    event_strong: list[np.ndarray] = []
    event_lengths: list[int] = []
    train_target_rows: list[tuple[int, float, int]] = []
    validation_target_rows: list[tuple[int, float, int]] = []

    for user_id, raw in phase0.iter_user_frames(big_path, phase0.EVENT_COLUMNS):
        timestamp = pd.to_numeric(raw["timestamp"], errors="raise")
        # Temporal-final rows are rejected before label canonicalization and are
        # never persisted in Phase 1 processed artifacts.
        raw = raw.loc[timestamp < validation_end].copy()
        if raw.empty:
            continue
        canonical, _, _, _ = phase0.canonicalize_behavior_events(raw)
        canonical = canonical.sort_values(["timestamp", "video_id"], kind="mergesort")
        items = canonical["video_id"].map(video_index)
        if items.isna().any():
            raise ArtifactError("Canonical event video is absent from catalog metadata")
        times = canonical["timestamp"].to_numpy(dtype=np.float64)
        item_positions = items.to_numpy(dtype=np.int32)
        strong = canonical["_is_strong_positive"].to_numpy(dtype=bool)
        event_users.append(np.full(len(canonical), int(user_id), dtype=np.int32))
        event_items.append(item_positions)
        event_times.append(times)
        event_strong.append(strong)
        event_lengths.append(len(canonical))
        user_ids.append(int(user_id))

        first_time: dict[int, float] = {}
        for video_id, event_time in zip(
            canonical["video_id"].astype(int), times, strict=True
        ):
            first_time.setdefault(int(video_id), float(event_time))
        for video_id, item_position, event_time, is_strong in zip(
            canonical["video_id"].astype(int),
            item_positions,
            times,
            strong,
            strict=True,
        ):
            if not is_strong or first_time[int(video_id)] != float(event_time):
                continue
            query_date = phase0.timestamp_to_local_date(float(event_time))
            state = catalog["state_by_date"].get(query_date)
            if state is None or not bool(state["eligible"][int(item_position)]):
                continue
            row = (int(user_id), float(event_time), int(item_position))
            if event_time < train_end:
                train_target_rows.append(row)
            else:
                validation_target_rows.append(row)

    users = np.asarray(sorted(set(user_ids)), dtype=np.int32)
    user_index = {int(value): index for index, value in enumerate(users)}
    events_user_id = np.concatenate(event_users)
    events_user = np.fromiter(
        (user_index[int(value)] for value in events_user_id),
        dtype=np.int32,
        count=len(events_user_id),
    )
    events_item = np.concatenate(event_items)
    events_time = np.concatenate(event_times)
    events_strong_array = np.concatenate(event_strong)
    event_indptr = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(event_lengths, dtype=np.int64)]
    )
    np.savez_compressed(
        work / "events_train_validation.npz",
        user_ids=users,
        user=events_user,
        item=events_item,
        timestamp=events_time,
        strong=events_strong_array,
        user_indptr=event_indptr,
    )

    train_targets = np.asarray(train_target_rows, dtype=np.float64)
    validation_targets = np.asarray(validation_target_rows, dtype=np.float64)
    if train_targets.size == 0 or validation_targets.size == 0:
        raise ArtifactError("Processed target table unexpectedly empty")
    train_user_id = train_targets[:, 0].astype(np.int32)
    train_user = np.fromiter(
        (user_index[int(value)] for value in train_user_id),
        dtype=np.int32,
        count=len(train_user_id),
    )
    train_time = train_targets[:, 1].astype(np.float64)
    train_item = train_targets[:, 2].astype(np.int32)
    np.savez_compressed(
        work / "targets_train.npz",
        user=train_user,
        item=train_item,
        timestamp=train_time,
    )

    validation_order = np.lexsort(
        (validation_targets[:, 2], validation_targets[:, 1], validation_targets[:, 0])
    )
    validation_targets = validation_targets[validation_order]
    grouped_targets: dict[tuple[int, float], list[int]] = defaultdict(list)
    for user_id, event_time, item in validation_targets:
        grouped_targets[(int(user_id), float(event_time))].append(int(item))
    query_keys = sorted(grouped_targets)
    query_users = np.asarray([user_index[key[0]] for key in query_keys], dtype=np.int32)
    query_user_ids = np.asarray([key[0] for key in query_keys], dtype=np.int32)
    query_times = np.asarray([key[1] for key in query_keys], dtype=np.float64)
    target_indptr = np.zeros(len(query_keys) + 1, dtype=np.int64)
    target_values: list[int] = []
    for index, key in enumerate(query_keys):
        target_values.extend(sorted(set(grouped_targets[key])))
        target_indptr[index + 1] = len(target_values)
    target_indices = np.asarray(target_values, dtype=np.int32)

    width = (len(video_ids) + 7) // 8
    bits_path = work / "candidate_bits_validation.npy"
    candidate_bits = np.lib.format.open_memmap(
        bits_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(query_keys), width),
    )
    membership_digest = hashlib.sha256()
    candidate_union_mask = 0
    candidate_counts = np.zeros(len(query_keys), dtype=np.int32)
    query_lookup: dict[int, list[int]] = defaultdict(list)
    for index, user_id in enumerate(query_user_ids):
        query_lookup[int(user_id)].append(index)
    for user_id, query_positions in query_lookup.items():
        user_position = user_index[user_id]
        start = int(event_indptr[user_position])
        end = int(event_indptr[user_position + 1])
        times = events_time[start:end]
        items = events_item[start:end]
        seen_mask = 0
        cursor = 0
        for query_position in query_positions:
            query_time = float(query_times[query_position])
            left = int(np.searchsorted(times, query_time, side="left"))
            for item in np.unique(items[cursor:left]):
                seen_mask |= 1 << int(item)
            cursor = left
            query_date = phase0.timestamp_to_local_date(query_time)
            state = catalog["state_by_date"].get(query_date)
            eligible = int(state["eligible_mask"]) if state else 0
            unseen = eligible & ~seen_mask
            candidate_union_mask |= unseen
            packed = _pack_mask(unseen, width)
            candidate_bits[query_position] = packed
            candidate_counts[query_position] = unseen.bit_count()
            target_slice = target_indices[
                target_indptr[query_position] : target_indptr[query_position + 1]
            ]
            target_video_ids = video_ids[target_slice]
            identity = (
                "membership-bitset-v1|validation|"
                f"{user_id}|{query_time:.6f}|"
                f"{','.join(str(int(value)) for value in target_video_ids)}|"
            ).encode()
            membership_digest.update(identity)
            membership_digest.update(packed.tobytes())
            membership_digest.update(b"\n")
            if any(((unseen >> int(item)) & 1) == 0 for item in target_slice):
                raise ArtifactError("Validation target is absent from its candidate set")
    candidate_bits.flush()
    del candidate_bits

    expected_target_hash = dict(verification.canonical_target_sha256)["validation"]
    hash_frame = pd.DataFrame(
        {
            "user_id": np.repeat(
                query_user_ids,
                np.diff(target_indptr),
            ),
            "timestamp": np.repeat(query_times, np.diff(target_indptr)),
            "video_id": video_ids[target_indices],
        }
    )
    target_hash = phase0._stable_target_hash(hash_frame)
    if target_hash != expected_target_hash:
        raise ArtifactError(
            f"Validation target hash mismatch: {target_hash} != {expected_target_hash}"
        )
    expected_membership = dict(verification.candidate_membership_sha256)["validation"]
    if membership_digest.hexdigest() != expected_membership:
        raise ArtifactError("Validation candidate membership hash mismatch")
    train_frame = pd.DataFrame(
        {
            "user_id": users[train_user],
            "timestamp": train_time,
            "video_id": video_ids[train_item],
        }
    )
    expected_train_hash = dict(verification.canonical_target_sha256)["train"]
    if phase0._stable_target_hash(train_frame) != expected_train_hash:
        raise ArtifactError("Train target hash mismatch")

    train_counts = np.bincount(train_item, minlength=len(video_ids)).astype(np.int64)
    order = np.lexsort((video_ids, -train_counts))
    total = int(train_counts.sum())
    cutoff_index = int(np.searchsorted(np.cumsum(train_counts[order]), 0.8 * total))
    head = np.zeros(len(video_ids), dtype=bool)
    if total:
        head[order[: cutoff_index + 1]] = True
    warm = train_counts > 0
    tail = warm & ~head
    cold = ~warm
    dates = np.asarray(sorted(catalog["state_by_date"]), dtype=np.int32)
    catalog_bits = np.stack(
        [
            _pack_mask(int(catalog["state_by_date"][int(date)]["eligible_mask"]), width)
            for date in dates
        ]
    )
    np.savez_compressed(
        work / "catalog.npz",
        video_ids=video_ids,
        dates=dates,
        eligible_bits=catalog_bits,
        train_counts=train_counts,
        warm=warm,
        head=head,
        tail=tail,
        cold=cold,
        train_end=np.asarray([train_end]),
        validation_end=np.asarray([validation_end]),
    )
    np.savez_compressed(
        work / "queries_validation.npz",
        user=query_users,
        user_ids=query_user_ids,
        timestamp=query_times,
        target_indptr=target_indptr,
        target_indices=target_indices,
        candidate_count=candidate_counts,
        candidate_union_count=np.asarray([candidate_union_mask.bit_count()], dtype=np.int32),
    )

    unique_pairs = np.unique(
        np.stack([train_user, train_item], axis=1), axis=0
    )
    user_item = sparse.csr_matrix(
        (
            np.ones(len(unique_pairs), dtype=np.float32),
            (unique_pairs[:, 0], unique_pairs[:, 1]),
        ),
        shape=(len(users), len(video_ids)),
        dtype=np.float32,
    )
    sparse.save_npz(work / "itemcf_user_item.npz", user_item, compressed=True)
    cooccurrence = (user_item.T @ user_item).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    sparse.save_npz(
        work / "itemcf_cooccurrence.npz", cooccurrence, compressed=True
    )

    negative_arrays: dict[str, np.ndarray] = {}
    target_group_map: dict[tuple[int, float], list[int]] = defaultdict(list)
    target_keys_by_user: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for user, event_time, item in zip(train_user, train_time, train_item, strict=True):
        target_group_map[(int(user), float(event_time))].append(int(item))
    for key in target_group_map:
        target_keys_by_user[key[0]].append(key)
    plan = json.loads(json.dumps([20260721, 20260722, 20260723]))
    target_row_lookup = {
        (int(user), float(event_time), int(item)): index
        for index, (user, event_time, item) in enumerate(
            zip(train_user, train_time, train_item, strict=True)
        )
    }
    for seed in plan:
        rng = np.random.default_rng(seed)
        negatives = np.full(len(train_item), -1, dtype=np.int32)
        for user in sorted(target_keys_by_user):
            start = int(event_indptr[user])
            end = int(event_indptr[user + 1])
            times = events_time[start:end]
            items = events_item[start:end]
            seen_mask = 0
            cursor = 0
            keys = sorted(target_keys_by_user[user])
            for key in keys:
                query_time = key[1]
                left = int(np.searchsorted(times, query_time, side="left"))
                for item in np.unique(items[cursor:left]):
                    seen_mask |= 1 << int(item)
                cursor = left
                query_date = phase0.timestamp_to_local_date(query_time)
                state = catalog["state_by_date"].get(query_date)
                pool = (int(state["eligible_mask"]) if state else 0) & ~seen_mask
                for target in target_group_map[key]:
                    pool &= ~(1 << int(target))
                pool_count = pool.bit_count()
                if not pool_count:
                    raise ArtifactError("BPR negative pool is empty")
                for target in target_group_map[key]:
                    while True:
                        candidate = int(rng.integers(len(video_ids)))
                        if (pool >> candidate) & 1:
                            break
                    row_index = target_row_lookup[(user, query_time, target)]
                    negatives[row_index] = candidate
        if (negatives < 0).any():
            raise ArtifactError("BPR negative index is incomplete")
        negative_arrays[f"seed_{seed}"] = negatives
    np.savez_compressed(work / "bpr_negative_indices.npz", **negative_arrays)

    return {
        "users": int(len(users)),
        "videos": int(len(video_ids)),
        "canonical_train_targets": int(len(train_item)),
        "validation_queries": int(len(query_times)),
        "validation_targets": int(len(target_indices)),
        "validation_candidate_score_count": int(candidate_counts.sum()),
        "validation_target_sha256": target_hash,
        "validation_candidate_membership_sha256": membership_digest.hexdigest(),
        "warm_items": int(warm.sum()),
        "tail_items": int(tail.sum()),
        "cold_items": int(cold.sum()),
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "temporal_final_rows_persisted": 0,
        "small_matrix_rows_read": 0,
    }


def verify_processed_artifacts(
    repo_root: str | Path,
    verification: ProtocolBundleVerification,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    directory = _artifact_root(root, verification.manifest_sha256)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"Missing processed artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("Processed artifact schema version mismatch")
    expected_fingerprint = _fingerprint(root, verification)
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ArtifactError("Processed artifact fingerprint mismatch; fail closed")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_FILES):
        raise ArtifactError("Processed artifact file table is incomplete")
    for name in ARTIFACT_FILES:
        path = directory / name
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ArtifactError(f"Processed artifact changed or is missing: {name}")
    return manifest


def fast_verify_processed_artifacts(repo_root: str | Path) -> dict[str, Any]:
    """Verify an existing cache without rerunning Phase 0 derived-hash replay."""

    root = Path(repo_root).resolve()
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    directory = _artifact_root(root, manifest_sha)
    artifact_manifest = directory / "manifest.json"
    if not artifact_manifest.is_file():
        raise ArtifactError("No completed processed artifact cache exists")
    value = json.loads(artifact_manifest.read_text())
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ArtifactError("Processed artifact fingerprint is missing")
    if fingerprint.get("split_manifest_sha256") != manifest_sha:
        raise ArtifactError("Processed artifact manifest binding changed")
    if fingerprint.get("selection_code_commit") != _git_head(root):
        raise ArtifactError("Processed artifacts belong to a different code commit")
    if fingerprint.get("phase0_config_sha256") != sha256_file(
        root / "configs/phase0.yaml"
    ):
        raise ArtifactError("Phase 0 config changed after artifact verification")
    for path, digest in fingerprint.get("generator_file_sha256", {}).items():
        if sha256_file(root / path) != digest:
            raise ArtifactError(f"Artifact generator input changed: {path}")
    phase0_manifest = json.loads((root / "manifests/split_manifest.json").read_text())
    active_contracts = phase0_manifest.get("active_contracts", {})
    current_contracts = {
        name: sha256_file(root / entry["path"])
        for name, entry in active_contracts.items()
    }
    if current_contracts != fingerprint.get("contract_sha256"):
        raise ArtifactError("Active contracts changed after full verification")
    source_hashes = fingerprint.get("source_file_sha256")
    if not isinstance(source_hashes, dict):
        raise ArtifactError("Artifact source hashes are missing")
    dataset_sources = phase0_manifest.get("dataset", {}).get("source_files", {})
    for name, digest in source_hashes.items():
        entry = dataset_sources.get(name)
        if not isinstance(entry, dict):
            raise ArtifactError(f"Raw source disappeared from manifest: {name}")
        source = root / "data/raw" / entry["relative_path"]
        if sha256_file(source) != digest:
            raise ArtifactError(f"Raw source changed after full verification: {name}")
    hashes = value.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_FILES):
        raise ArtifactError("Processed artifact file table is incomplete")
    for name, digest in hashes.items():
        if sha256_file(directory / name) != digest:
            raise ArtifactError(f"Processed artifact changed: {name}")
    return value


def build_processed_artifacts(
    repo_root: str | Path,
    verification: ProtocolBundleVerification,
) -> tuple[dict[str, Any], bool]:
    """Build once with resumable staging; reuse only an exactly matching cache."""

    root = Path(repo_root).resolve()
    destination = _artifact_root(root, verification.manifest_sha256)
    if destination.exists():
        return verify_processed_artifacts(root, verification), True
    staging = destination.with_name(destination.name + ".building")
    fingerprint = _fingerprint(root, verification)
    if staging.exists():
        state_path = staging / "BUILD_STATE.json"
        if not state_path.is_file():
            raise ArtifactError("Incomplete artifact staging lacks BUILD_STATE.json")
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise ArtifactError("Incomplete artifact staging fingerprint mismatch")
        # The current builder has one atomic heavy step. A matching interrupted
        # stage is safe to discard and rebuild; a mismatched one always fails.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _write_json(
        staging / "BUILD_STATE.json",
        {"fingerprint": fingerprint, "status": "building"},
    )
    try:
        statistics = _canonical_inputs(root, staging, verification)
        file_hashes = {name: sha256_file(staging / name) for name in ARTIFACT_FILES}
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_scope": "train_and_validation_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "files": file_hashes,
            "statistics": statistics,
            "cache_policy": "reuse only after complete fingerprint and file-hash verification",
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "BUILD_STATE.json").unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, destination)
    except Exception:
        raise
    return verify_processed_artifacts(root, verification), False
