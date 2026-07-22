"""Auditable Phase 1 segment-membership correction; never opens holdouts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import (
    ARTIFACT_FILES,
    ArtifactError,
    build_selection_segment_membership,
)
from .baselines import (
    load_artifacts,
    rank_bpr,
    rank_causal_streaming_decayed,
    rank_fit_frozen_decayed,
    rank_global_popularity,
    rank_itemcf,
    rank_random,
)
from .gates import (
    CI_METRICS,
    ERRATUM_SELECTION_EVALUATOR,
    GateError,
    METHODS,
    SEGMENT_METRICS,
    canonical_json,
    derive_final_method_bundle,
    load_and_validate_selection_plan,
    sha256_file,
    validate_selection_erratum_invariants,
    validate_selection_result,
)
from .metrics import common_bootstrap_indices, evaluate_topk
from .publish import publish_files_transactionally, recover_publish_transaction
from .runner import _write_markdown
from .watchdog import ProcessLivenessPulse, WatchdogTimeout, run_supervised


ERRATUM_ID = "ERRATUM-001"
ERRATUM_REASON = (
    "Phase 1 incorrectly derived data-warm membership from eligible strong-positive "
    "train targets instead of all canonical train-window interactions"
)
ORIGINAL_MERGE_COMMIT = "4fab970fe36685f0c23aef49ac713dc100570502"
ORIGINAL_RESULT_SHA256 = (
    "f3dbbba9de5552d8d6bb34ae0fbe58dc50b57726a180b61bd9d216f31927857f"
)
ORIGINAL_BUNDLE_SHA256 = (
    "c56c83bc96486fb87d4650be59321aeb3dfdf11421d181a50baf1f23448119ac"
)
ORIGINAL_RECEIPT_SHA256 = (
    "7acbb6ea4dd9bd88374b479b6aa54f1c97779d027c527f328e9db0835842d57a"
)
MAX_WALL_SECONDS = 3 * 60 * 60
ROW_TIMEOUT_SECONDS = 10 * 60
LIVENESS_INTERVAL_SECONDS = 30


def _git_head_clean(root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise GateError("Worktree, including untracked files, must be clean")
    return head


def _git_json(root: Path, commit: str, path: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GateError(f"Cannot load historical artifact {commit}:{path}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise GateError(f"Historical artifact is not an object: {path}")
    return value


def _git_file_sha256(root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GateError(f"Cannot load historical source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_original_state(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    result_path = root / "reports/phase1/validation_baselines.json"
    bundle_path = root / "reports/phase1/final_method_bundle.json"
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    receipt_path = root / "receipts" / manifest_sha / "SELECTION_RECEIPT.json"
    expected = (
        (result_path, ORIGINAL_RESULT_SHA256),
        (bundle_path, ORIGINAL_BUNDLE_SHA256),
        (receipt_path, ORIGINAL_RECEIPT_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise GateError(f"Known original Phase 1 artifact changed: {path}")
    historical_result = _git_json(
        root, ORIGINAL_MERGE_COMMIT, "reports/phase1/validation_baselines.json"
    )
    historical_bundle = _git_json(
        root, ORIGINAL_MERGE_COMMIT, "reports/phase1/final_method_bundle.json"
    )
    return historical_result, historical_bundle, receipt_path


def _verify_erratum_source_artifacts(
    root: Path, historical_result: Mapping[str, Any]
) -> Path:
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    directory = root / "artifacts" / "phase1" / manifest_sha
    artifact_manifest_path = directory / "manifest.json"
    if (
        not artifact_manifest_path.is_file()
        or sha256_file(artifact_manifest_path)
        != historical_result["hashes"]["processed_artifact_manifest_sha256"]
    ):
        raise ArtifactError("Historical processed-artifact manifest binding changed")
    manifest = json.loads(artifact_manifest_path.read_text())
    if manifest.get("artifact_scope") != "train_and_validation_only":
        raise ArtifactError("Erratum source cache is not train/validation only")
    statistics = manifest.get("statistics", {})
    if statistics.get("temporal_final_rows_persisted") != 0:
        raise ArtifactError("Erratum source cache contains temporal-final rows")
    if statistics.get("small_matrix_rows_read") != 0:
        raise ArtifactError("Erratum source cache records Small Matrix access")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise ArtifactError("Erratum source artifact file table is incomplete")
    for name, digest in files.items():
        if sha256_file(directory / name) != digest:
            raise ArtifactError(f"Erratum source artifact changed: {name}")
    fingerprint = manifest.get("fingerprint", {})
    if fingerprint.get("split_manifest_sha256") != manifest_sha:
        raise ArtifactError("Erratum source split-manifest binding changed")
    generation_commit = fingerprint.get("selection_code_commit")
    generators = fingerprint.get("generator_file_sha256", {})
    for path, digest in generators.items():
        if _git_file_sha256(root, generation_commit, path) != digest:
            raise ArtifactError(f"Historical artifact generator mismatch: {path}")
    return directory


def _corrected_artifacts(
    root: Path, directory: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = load_artifacts(directory)
    catalog = {name: artifacts["catalog"][name] for name in artifacts["catalog"].files}
    reference = json.loads((root / "manifests/split_manifest.json").read_text())[
        "cold_start_contexts"
    ]["validation"]
    segments = build_selection_segment_membership(
        video_ids=catalog["video_ids"],
        event_items=artifacts["events"]["item"],
        event_timestamps=artifacts["events"]["timestamp"],
        positive_target_items=artifacts["train"]["item"],
        train_end_exclusive=float(catalog["train_end"][0]),
        expected_data_warm_count=int(reference["reference_item_count"]),
        expected_data_warm_sha256=reference["reference_membership_sha256"],
    )
    if not np.array_equal(catalog["train_counts"], segments["positive_target_count"]):
        raise ArtifactError("Positive-target counts changed during membership correction")
    catalog.update(
        {
            "interaction_count": segments["interaction_count"],
            "positive_target_count": segments["positive_target_count"],
            "data_warm": segments["data_warm"],
            "data_cold": segments["data_cold"],
            "warm": segments["data_warm"],
            "cold": segments["data_cold"],
            "head": segments["head"],
            "tail": segments["tail"],
        }
    )
    corrected = dict(artifacts)
    corrected["catalog"] = catalog
    return corrected, segments


def _prepare_execution(
    root: Path,
    *,
    verify_selected_fallback: bool,
    expected_cache_key: str | None = None,
    row_index: int | None = None,
) -> dict[str, Any]:
    code_commit = _git_head_clean(root)
    original_result, original_bundle, receipt_path = _verify_original_state(root)
    artifact_dir = _verify_erratum_source_artifacts(root, original_result)
    artifacts, segments = _corrected_artifacts(root, artifact_dir)
    plan = load_and_validate_selection_plan(root)
    original_rows = {
        (row["method"], row["config_id"], row["seed"]): row
        for row in original_result["rows"]
    }
    fallback_config = next(
        method
        for method in original_bundle["methods"]
        if method["name"] == "time_decayed_popularity"
    )
    fallback_file = artifact_dir / "topk" / f"{fallback_config['config_id']}.npz"
    if not fallback_file.is_file():
        raise ArtifactError("Historical selected fallback Top-K is missing")
    fallback_topk = np.load(fallback_file)["topk"]
    _validate_topk_candidates(fallback_topk, artifacts)
    if verify_selected_fallback:
        fallback_hp = fallback_config["hyperparameters"]
        recomputed = rank_causal_streaming_decayed(
            artifacts, fallback_hp["half_life_days"]
        )
        if not np.array_equal(recomputed, fallback_topk):
            raise GateError("Recomputed selected fallback ranking differs from cache")
    if expected_cache_key is None:
        ranking_input_manifest = _build_ranking_input_manifest(
            root=root,
            artifact_dir=artifact_dir,
            plan_rows=plan.rows,
            fallback_file=fallback_file,
        )
        _validate_ranking_manifest_structure(ranking_input_manifest, plan.rows)
        _verify_ranking_manifest_files(root, ranking_input_manifest, row_index=None)
        ranking_manifest_sha = _json_file_sha256(ranking_input_manifest)
    else:
        cache_dir = _cache_directory(artifact_dir, expected_cache_key)
        ranking_manifest_path = cache_dir / "RANKING_INPUT_MANIFEST.json"
        try:
            ranking_input_manifest = json.loads(ranking_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError("Ranking-input manifest is unavailable to row worker") from exc
        _validate_ranking_manifest_structure(ranking_input_manifest, plan.rows)
        if row_index is None or row_index < 0 or row_index >= len(plan.rows):
            raise GateError("Row-specific ranking-input verification requires a valid row")
        _verify_ranking_manifest_files(
            root, ranking_input_manifest, row_index=row_index
        )
        ranking_manifest_sha = sha256_file(ranking_manifest_path)
    binding = _cache_binding(
        root=root,
        code_commit=code_commit,
        artifact_dir=artifact_dir,
        plan_sha256=plan.sha256,
        segments=segments,
        fallback_file=fallback_file,
        ranking_input_manifest_sha256=ranking_manifest_sha,
    )
    cache_dir = _cache_directory(artifact_dir, binding["cache_key"])
    if expected_cache_key is not None and binding["cache_key"] != expected_cache_key:
        raise GateError("Row worker cache key differs from its ranking inputs")
    _initialize_cache(cache_dir, binding, ranking_input_manifest)
    return {
        "code_commit": code_commit,
        "original_result": original_result,
        "original_bundle": original_bundle,
        "receipt_path": receipt_path,
        "artifact_dir": artifact_dir,
        "artifacts": artifacts,
        "segments": segments,
        "plan": plan,
        "original_rows": original_rows,
        "fallback_topk": fallback_topk,
        "fallback_file": fallback_file,
        "bpr_prefix": sha256_file(fallback_file)[:8],
        "binding": binding,
        "ranking_input_manifest": ranking_input_manifest,
        "cache_dir": cache_dir,
    }


def _peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _emit_progress(stage: str, done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (total - done) if done else None
    print(
        json.dumps(
            {
                "stage": stage,
                "done": done,
                "total": total,
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": round(eta, 2) if eta is not None else None,
                "rss_mb": round(_peak_memory_mb(), 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_file_sha256(value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_topk(
    path: Path, topk: np.ndarray, metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                topk=topk,
                metadata=np.asarray(canonical_json(metadata)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bpr_model_path(
    artifact_dir: Path, planned: Mapping[str, Any], bpr_prefix: str
) -> Path:
    hp = planned["hyperparameters"]
    group = "_".join(
        map(
            str,
            (
                hp["embedding_dim"],
                hp["learning_rate"],
                hp["l2"],
                planned["seed"],
            ),
        )
    )
    return (
        artifact_dir
        / "bpr_models"
        / f"{bpr_prefix}_{group}"
        / f"epoch_{hp['epoch']}.npz"
    )


def _relative_hashed_file(root: Path, path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ArtifactError(f"Ranking input is missing: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
    }


def _build_ranking_input_manifest(
    *,
    root: Path,
    artifact_dir: Path,
    plan_rows: tuple[dict[str, Any], ...],
    fallback_file: Path,
) -> dict[str, Any]:
    bpr_prefix = sha256_file(fallback_file)[:8]
    entries: list[dict[str, Any]] = []
    for index, planned in enumerate(plan_rows):
        files: list[dict[str, str]] = []
        if planned["method"] == "time_decayed_popularity":
            files.append(
                _relative_hashed_file(
                    root,
                    artifact_dir / "topk" / f"{planned['config_id']}.npz",
                )
            )
        elif planned["method"] == "bpr_mf":
            files.extend(
                (
                    _relative_hashed_file(
                        root, _bpr_model_path(artifact_dir, planned, bpr_prefix)
                    ),
                    _relative_hashed_file(root, fallback_file),
                )
            )
        entries.append(
            {
                "row_index": index,
                "planned": dict(planned),
                "files": files,
            }
        )
    return {
        "schema_version": 1,
        "artifact_scope": "train_and_validation_only",
        "processed_artifact_manifest": _relative_hashed_file(
            root, artifact_dir / "manifest.json"
        ),
        "candidate_membership": _relative_hashed_file(
            root, artifact_dir / "candidate_bits_validation.npy"
        ),
        "rows": entries,
    }


def _validate_ranking_manifest_structure(
    manifest: Mapping[str, Any], plan_rows: tuple[dict[str, Any], ...]
) -> None:
    if manifest.get("schema_version") != 1:
        raise GateError("Ranking-input manifest schema changed")
    if manifest.get("artifact_scope") != "train_and_validation_only":
        raise GateError("Ranking-input manifest scope changed")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(plan_rows):
        raise GateError("Ranking-input manifest row coverage changed")
    for index, (entry, planned) in enumerate(zip(rows, plan_rows, strict=True)):
        if not isinstance(entry, Mapping) or entry.get("row_index") != index:
            raise GateError("Ranking-input manifest row order changed")
        if canonical_json(entry.get("planned")) != canonical_json(planned):
            raise GateError("Ranking-input manifest differs from selection plan")
        files = entry.get("files")
        if not isinstance(files, list):
            raise GateError("Ranking-input manifest files must be a list")
        expected_count = 1 if planned["method"] == "time_decayed_popularity" else 0
        if planned["method"] == "bpr_mf":
            expected_count = 2
        if len(files) != expected_count:
            raise GateError("Ranking-input manifest file coverage changed")


def _verify_hashed_input(root: Path, entry: Mapping[str, Any]) -> None:
    path = root / str(entry.get("path"))
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ArtifactError(f"Ranking input hash mismatch: {path}")


def _verify_ranking_manifest_files(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    row_index: int | None,
) -> None:
    _verify_hashed_input(root, manifest["processed_artifact_manifest"])
    _verify_hashed_input(root, manifest["candidate_membership"])
    entries = manifest["rows"] if row_index is None else [manifest["rows"][row_index]]
    for entry in entries:
        for file_entry in entry["files"]:
            _verify_hashed_input(root, file_entry)


def _cache_binding(
    *,
    root: Path,
    code_commit: str,
    artifact_dir: Path,
    plan_sha256: str,
    segments: Mapping[str, Any],
    fallback_file: Path,
    ranking_input_manifest_sha256: str,
) -> dict[str, Any]:
    binding = {
        "schema_version": 1,
        "erratum_id": ERRATUM_ID,
        "original_selection_result_sha256": ORIGINAL_RESULT_SHA256,
        "original_final_method_bundle_sha256": ORIGINAL_BUNDLE_SHA256,
        "original_selection_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "code_commit": code_commit,
        "selection_plan_sha256": plan_sha256,
        "processed_artifact_manifest_sha256": sha256_file(
            artifact_dir / "manifest.json"
        ),
        "evaluator_sha256": sha256_file(root / ERRATUM_SELECTION_EVALUATOR),
        "data_warm_membership_sha256": segments["data_warm_sha256"],
        "selected_fallback_topk_sha256": sha256_file(fallback_file),
        "ranking_input_manifest_sha256": ranking_input_manifest_sha256,
    }
    binding["cache_key"] = hashlib.sha256(
        canonical_json(binding).encode()
    ).hexdigest()
    return binding


def _cache_directory(artifact_dir: Path, cache_key: str) -> Path:
    return artifact_dir / "errata" / ERRATUM_ID / cache_key


def _initialize_cache(
    cache_dir: Path,
    binding: Mapping[str, Any],
    ranking_input_manifest: Mapping[str, Any],
) -> None:
    manifest = cache_dir / "CACHE_MANIFEST.json"
    ranking_manifest_path = cache_dir / "RANKING_INPUT_MANIFEST.json"
    if manifest.is_file():
        try:
            existing = json.loads(manifest.read_text())
            existing_ranking = json.loads(ranking_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError("Erratum cache manifest is unreadable") from exc
        if canonical_json(existing) != canonical_json(binding):
            raise GateError("Erratum cache manifest binding changed")
        if canonical_json(existing_ranking) != canonical_json(ranking_input_manifest):
            raise GateError("Ranking-input manifest changed")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(ranking_manifest_path, ranking_input_manifest)
    _atomic_write_json(manifest, binding)


def _row_stem(index: int, planned: Mapping[str, Any]) -> str:
    seed = "deterministic" if planned["seed"] is None else str(planned["seed"])
    return f"{index + 1:03d}_{planned['method']}__{planned['config_id']}__{seed}"


def _row_paths(
    cache_dir: Path, index: int, planned: Mapping[str, Any]
) -> tuple[Path, Path]:
    stem = _row_stem(index, planned)
    return cache_dir / "topk" / f"{stem}.npz", cache_dir / "rows" / f"{stem}.json"


def _topk_metadata(
    *,
    row_index: int,
    planned: Mapping[str, Any],
    binding: Mapping[str, Any],
    ranking_input_manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cache_key": binding["cache_key"],
        "row_index": row_index,
        "planned": dict(planned),
        "ranking_input_manifest_sha256": binding[
            "ranking_input_manifest_sha256"
        ],
        "processed_artifact_manifest_sha256": binding[
            "processed_artifact_manifest_sha256"
        ],
        "candidate_membership_sha256": ranking_input_manifest[
            "candidate_membership"
        ]["sha256"],
        "shape": [len(artifacts["queries"]["user"]), 100],
        "dtype": "int32",
        "item_count": len(artifacts["catalog"]["video_ids"]),
    }


def _validate_topk_candidates(topk: np.ndarray, artifacts: Mapping[str, Any]) -> None:
    query_count = len(artifacts["queries"]["user"])
    item_count = len(artifacts["catalog"]["video_ids"])
    if topk.shape != (query_count, 100):
        raise GateError("Top-K checkpoint shape differs from the evaluation contract")
    if topk.dtype != np.dtype(np.int32):
        raise GateError("Top-K checkpoint dtype must be int32")
    if np.any(topk < -1) or np.any(topk >= item_count):
        raise GateError("Top-K checkpoint contains an out-of-range item")
    candidate_bits = artifacts["candidate_bits"]
    for begin in range(0, query_count, 2048):
        block = topk[begin : begin + 2048]
        valid = block >= 0
        padding_seen = np.maximum.accumulate(~valid, axis=1)
        if np.any(padding_seen & valid):
            raise GateError("Top-K checkpoint has a non-padding item after -1")
        comparable = np.where(valid, block, item_count)
        ordered = np.sort(comparable, axis=1)
        if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, :-1] < item_count)):
            raise GateError("Top-K checkpoint contains a duplicate item")
        if not valid.any():
            continue
        safe_items = np.where(valid, block, 0)
        rows = np.arange(begin, begin + len(block), dtype=np.int64)[:, None]
        byte_positions = safe_items >> 3
        bit_masks = np.left_shift(1, safe_items & 7)
        allowed = (
            np.bitwise_and(candidate_bits[rows, byte_positions], bit_masks) != 0
        )
        if np.any(valid & ~allowed):
            raise GateError("Top-K checkpoint contains an item outside query candidates")


def _load_topk_checkpoint(
    *,
    path: Path,
    expected_metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    validate_candidates: bool,
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            if set(checkpoint.files) != {"topk", "metadata"}:
                raise GateError("Top-K checkpoint fields are incomplete")
            topk = np.asarray(checkpoint["topk"])
            metadata = json.loads(str(checkpoint["metadata"].item()))
    except GateError:
        raise
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise GateError("Top-K checkpoint is unreadable") from exc
    if canonical_json(metadata) != canonical_json(expected_metadata):
        raise GateError("Top-K checkpoint identity or input binding changed")
    if list(topk.shape) != metadata.get("shape") or str(topk.dtype) != metadata.get(
        "dtype"
    ):
        raise GateError("Top-K checkpoint array metadata changed")
    if validate_candidates:
        _validate_topk_candidates(topk, artifacts)
    return topk


def _validate_cached_corrected_row(
    *,
    payload: Mapping[str, Any],
    planned: Mapping[str, Any],
    original: Mapping[str, Any],
    cache_key: str,
    topk_path: Path,
    topk_metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("cache_key") != cache_key:
        raise GateError("Corrected-row cache binding changed")
    if canonical_json(payload.get("planned")) != canonical_json(planned):
        raise GateError("Corrected-row cache differs from the selection plan")
    if not topk_path.is_file():
        raise GateError("Corrected-row cache is missing its Top-K file")
    if payload.get("topk_sha256") != sha256_file(topk_path):
        raise GateError("Corrected-row Top-K hash mismatch")
    _load_topk_checkpoint(
        path=topk_path,
        expected_metadata=topk_metadata,
        artifacts=artifacts,
        validate_candidates=False,
    )
    if payload.get("candidate_validation") != {
        "status": "passed",
        "candidate_membership_sha256": topk_metadata[
            "candidate_membership_sha256"
        ],
    }:
        raise GateError("Corrected-row candidate validation evidence changed")
    corrected = payload.get("corrected_row")
    if not isinstance(corrected, dict):
        raise GateError("Corrected-row cache payload is missing")
    for key in ("method", "config_id", "hyperparameters", "seed"):
        if canonical_json(corrected.get(key)) != canonical_json(planned[key]):
            raise GateError(f"Corrected-row identity changed: {key}")
    for metric, value in original["metrics"].items():
        if metric not in SEGMENT_METRICS and corrected["metrics"].get(metric) != value:
            raise GateError(f"Cached corrected row changed protected metric: {metric}")
    if corrected.get("coverage") != original.get("coverage"):
        raise GateError("Cached corrected row changed coverage evidence")
    for metric in SEGMENT_METRICS:
        value = corrected["metrics"].get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise GateError(f"Cached corrected row has invalid metric: {metric}")
    return corrected


def _load_cached_row(
    *,
    cache_dir: Path,
    index: int,
    planned: Mapping[str, Any],
    original: Mapping[str, Any],
    cache_key: str,
    topk_metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    topk_path, row_path = _row_paths(cache_dir, index, planned)
    if row_path.is_file():
        try:
            payload = json.loads(row_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError("Corrected-row cache JSON is unreadable") from exc
        return (
            _validate_cached_corrected_row(
                payload=payload,
                planned=planned,
                original=original,
                cache_key=cache_key,
                topk_path=topk_path,
                topk_metadata=topk_metadata,
                artifacts=artifacts,
            ),
            None,
        )
    if not topk_path.is_file():
        return None, None
    topk = _load_topk_checkpoint(
        path=topk_path,
        expected_metadata=topk_metadata,
        artifacts=artifacts,
        validate_candidates=True,
    )
    return None, topk


def _write_row_checkpoint(
    *,
    cache_dir: Path,
    index: int,
    planned: Mapping[str, Any],
    corrected: Mapping[str, Any],
    topk: np.ndarray,
    cache_key: str,
    topk_metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    topk_path, row_path = _row_paths(cache_dir, index, planned)
    if not topk_path.is_file():
        _atomic_write_topk(topk_path, topk, topk_metadata)
    payload = {
        "schema_version": 1,
        "cache_key": cache_key,
        "planned": dict(planned),
        "topk_sha256": sha256_file(topk_path),
        "candidate_validation": {
            "status": "passed",
            "candidate_membership_sha256": topk_metadata[
                "candidate_membership_sha256"
            ],
        },
        "corrected_row": dict(corrected),
    }
    _atomic_write_json(row_path, payload)
    return topk_path, row_path


def _topk_for_row(
    *,
    planned: Mapping[str, Any],
    artifacts: dict[str, Any],
    artifact_dir: Path,
    fallback_topk: np.ndarray,
    bpr_prefix: str,
) -> np.ndarray:
    method = planned["method"]
    hp = planned["hyperparameters"]
    if method == "random":
        return rank_random(artifacts, int(planned["seed"]))
    if method == "global_popularity":
        return rank_global_popularity(artifacts)
    if method == "time_decayed_popularity":
        if hp["variant"] == "fit_frozen":
            return rank_fit_frozen_decayed(artifacts, hp["half_life_days"])
        return rank_causal_streaming_decayed(artifacts, hp["half_life_days"])
    if method == "itemcf":
        return rank_itemcf(
            artifacts,
            neighbor_count=hp["neighbor_count"],
            shrinkage=hp["shrinkage"],
        )
    if method == "bpr_mf":
        model_path = _bpr_model_path(artifact_dir, planned, bpr_prefix)
        if not model_path.is_file():
            raise ArtifactError(f"Frozen BPR checkpoint is missing: {model_path}")
        model = np.load(model_path)
        topk, _ = rank_bpr(
            artifacts,
            model["users"],
            model["items"],
            fallback_topk=fallback_topk,
        )
        return topk
    raise GateError(f"Unsupported erratum method: {method}")


def _correct_row(
    *,
    original: Mapping[str, Any],
    topk: np.ndarray,
    artifacts: dict[str, Any],
    bootstrap_users: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    catalog = artifacts["catalog"]
    queries = artifacts["queries"]
    evaluated = evaluate_topk(
        topk=topk,
        query_users=queries["user"],
        target_indptr=queries["target_indptr"],
        target_indices=queries["target_indices"],
        candidate_union_count=int(queries["candidate_union_count"][0]),
        candidate_score_count=int(queries["candidate_count"].sum()),
        warm_mask=catalog["warm"],
        tail_mask=catalog["tail"],
        cold_mask=catalog["cold"],
        bootstrap_users=bootstrap_users,
        bootstrap_indices=bootstrap_indices,
    )
    corrected = copy.deepcopy(dict(original))
    for metric, value in evaluated["metrics"].items():
        if not math.isfinite(value):
            raise GateError(f"NaN/Inf metric in corrected row: {metric}")
        if metric in SEGMENT_METRICS:
            corrected["metrics"][metric] = value
        elif value != original["metrics"][metric]:
            raise GateError(
                f"Protected overall metric changed for {original['method']} "
                f"{original['config_id']} seed={original['seed']}: {metric}"
            )
    for metric in CI_METRICS:
        if metric.startswith(("TailRecall@", "ColdRecall@")):
            corrected["bootstrap_95_percent_intervals"][metric] = evaluated[
                "bootstrap_95_percent_intervals"
            ][metric]
    for prefix in ("warm", "tail", "cold"):
        for suffix in ("query_count", "user_count", "target_count"):
            key = f"{prefix}_{suffix}"
            corrected["denominators"][key] = evaluated["denominators"][key]
    for metric in SEGMENT_METRICS:
        corrected["secondary_user_macro"][metric] = evaluated[
            "secondary_user_macro"
        ][metric]
    if evaluated["coverage"] != original["coverage"]:
        raise GateError("Coverage evidence changed during segment-only correction")
    return corrected


def run_segment_membership_erratum_row(
    repo_root: str | Path, *, row_index: int, expected_cache_key: str
) -> dict[str, Any]:
    """Compute or resume one row; never writes any formal report or receipt."""

    root = Path(repo_root).resolve()
    context = _prepare_execution(
        root,
        verify_selected_fallback=False,
        expected_cache_key=expected_cache_key,
        row_index=row_index,
    )
    binding = context["binding"]
    if binding["cache_key"] != expected_cache_key:
        raise GateError("Row worker cache key differs from its supervisor")
    plan = context["plan"]
    if row_index < 0 or row_index >= len(plan.rows):
        raise GateError("Row worker index is outside the frozen selection plan")
    planned = plan.rows[row_index]
    topk_metadata = _topk_metadata(
        row_index=row_index,
        planned=planned,
        binding=binding,
        ranking_input_manifest=context["ranking_input_manifest"],
        artifacts=context["artifacts"],
    )
    key = (planned["method"], planned["config_id"], planned["seed"])
    original = context["original_rows"][key]
    cached, partial_topk = _load_cached_row(
        cache_dir=context["cache_dir"],
        index=row_index,
        planned=planned,
        original=original,
        cache_key=expected_cache_key,
        topk_metadata=topk_metadata,
        artifacts=context["artifacts"],
    )
    if cached is not None:
        return {
            "row": row_index + 1,
            "method": planned["method"],
            "config_id": planned["config_id"],
            "status": "cache_hit",
        }
    topk = partial_topk
    topk_source = "partial_erratum_checkpoint" if topk is not None else "recomputed"
    if topk is None and planned["method"] == "time_decayed_popularity":
        historical = context["artifact_dir"] / "topk" / f"{planned['config_id']}.npz"
        if historical.is_file():
            topk = np.load(historical)["topk"]
            topk_source = "historical_phase1_cache"
    if topk is None:
        topk = _topk_for_row(
            planned=planned,
            artifacts=context["artifacts"],
            artifact_dir=context["artifact_dir"],
            fallback_topk=context["fallback_topk"],
            bpr_prefix=context["bpr_prefix"],
        )
    # Persist the expensive ranking before metric evaluation. If the worker is
    # interrupted during bootstrap, the next run resumes from this Top-K rather
    # than repeating model scoring.
    topk_path, _ = _row_paths(context["cache_dir"], row_index, planned)
    if not topk_path.is_file():
        _atomic_write_topk(topk_path, topk, topk_metadata)
    _validate_topk_candidates(topk, context["artifacts"])
    bootstrap_users, bootstrap_indices = common_bootstrap_indices(
        context["artifacts"]["queries"]["user"]
    )
    corrected = _correct_row(
        original=original,
        topk=topk,
        artifacts=context["artifacts"],
        bootstrap_users=bootstrap_users,
        bootstrap_indices=bootstrap_indices,
    )
    topk_path, row_path = _write_row_checkpoint(
        cache_dir=context["cache_dir"],
        index=row_index,
        planned=planned,
        corrected=corrected,
        topk=topk,
        cache_key=expected_cache_key,
        topk_metadata=topk_metadata,
    )
    return {
        "row": row_index + 1,
        "method": planned["method"],
        "config_id": planned["config_id"],
        "status": "completed",
        "topk_source": topk_source,
        "topk_sha256": sha256_file(topk_path),
        "row_sha256": sha256_file(row_path),
    }


def _active_row_liveness_path(artifact_dir: Path) -> Path:
    return artifact_dir / "errata" / ERRATUM_ID / "ACTIVE_ROW_LIVENESS.json"


def _publish_journal_path(root: Path) -> Path:
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    return (
        root
        / "artifacts"
        / "phase1"
        / manifest_sha
        / "errata"
        / ERRATUM_ID
        / "PUBLISH_JOURNAL.json"
    )


def _liveness_callback(
    *,
    cache_dir: Path,
    artifact_dir: Path,
    planned: Mapping[str, Any],
    row_index: int,
    total: int,
    overall_started: float,
) -> Callable[[ProcessLivenessPulse], None]:
    def emit(sample: ProcessLivenessPulse) -> None:
        payload = {
            "schema_version": 1,
            "pulse_type": "process_liveness_not_progress",
            "status": "running",
            "stage": "exact_segment_metric_replay",
            "row": row_index + 1,
            "total": total,
            "method": planned["method"],
            "config_id": planned["config_id"],
            "seed": planned["seed"],
            "worker_pid": sample.pid,
            "worker_process_group_id": sample.process_group_id,
            "orchestrator_pid": os.getpid(),
            "worker_elapsed_seconds": round(sample.elapsed_seconds, 2),
            "overall_elapsed_seconds": round(
                time.perf_counter() - overall_started, 2
            ),
            "cpu_percent": (
                round(sample.cpu_percent, 2)
                if sample.cpu_percent is not None
                else None
            ),
            "rss_mb": round(sample.rss_mb, 2) if sample.rss_mb is not None else None,
            "process_state": sample.state,
            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(cache_dir / "LIVENESS_PULSE.json", payload)
        _atomic_write_json(_active_row_liveness_path(artifact_dir), payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    return emit


def _mark_row_inactive(artifact_dir: Path) -> None:
    _atomic_write_json(
        _active_row_liveness_path(artifact_dir),
        {
            "schema_version": 1,
            "pulse_type": "process_liveness_not_progress",
            "status": "inactive",
            "orchestrator_pid": os.getpid(),
            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _build_replacement_receipt(
    *,
    root: Path,
    result_path: Path,
    bundle_path: Path,
    code_commit: str,
    plan: Any,
    selected_at_utc: str,
) -> tuple[Path, Path, bytes, dict[str, Any]]:
    manifest_sha = sha256_file(root / "manifests/split_manifest.json")
    receipt_path = root / "receipts" / manifest_sha / "SELECTION_RECEIPT.json"
    if not receipt_path.is_file() or sha256_file(receipt_path) != ORIGINAL_RECEIPT_SHA256:
        raise GateError("Known original selection receipt changed before publication")
    old_bytes = receipt_path.read_bytes()
    old_payload = json.loads(old_bytes)
    if (
        old_payload.get("selection_result_sha256") != ORIGINAL_RESULT_SHA256
        or old_payload.get("final_method_bundle_sha256") != ORIGINAL_BUNDLE_SHA256
    ):
        raise GateError("Original receipt no longer binds the known formal artifacts")
    payload = {
        "schema_version": 1,
        "receipt_type": "selection",
        "status": "completed",
        "protocol_revision": plan.protocol_revision,
        "split_manifest_sha256": manifest_sha,
        "selection_plan_path": str(Path(plan.path).relative_to(root)),
        "selection_plan_sha256": plan.sha256,
        "selection_result_path": "reports/phase1/validation_baselines.json",
        "selection_result_sha256": sha256_file(result_path),
        "final_method_bundle_path": "reports/phase1/final_method_bundle.json",
        "final_method_bundle_sha256": sha256_file(bundle_path),
        "code_commit": code_commit,
        "methods": list(METHODS),
        "selected_at_utc": selected_at_utc,
        "erratum_id": ERRATUM_ID,
        "erratum_reason": ERRATUM_REASON,
        "supersedes_selection_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "supersedes_selection_result_sha256": ORIGINAL_RESULT_SHA256,
        "supersedes_final_method_bundle_sha256": ORIGINAL_BUNDLE_SHA256,
    }
    archive_path = (
        receipt_path.parent
        / "superseded"
        / ORIGINAL_RESULT_SHA256
        / receipt_path.name
    )
    return receipt_path, archive_path, old_bytes, payload


def _write_erratum(
    *,
    repo_root: Path,
    path: Path,
    result: Mapping[str, Any],
    bundle: Mapping[str, Any],
    archive_path: Path,
    segments: Mapping[str, Any],
    runtime_seconds: float,
    corrected_result_sha256: str,
    corrected_bundle_sha256: str,
    corrected_receipt_sha256: str,
) -> None:
    selected = {method["name"]: method for method in bundle["methods"]}
    lines = [
        "# ERRATUM-001: Phase 1 segment membership",
        "",
        "## Error and contract",
        "",
        "The active cold-start contract defines a data-warm item as any video with at least one canonical Big Matrix interaction in the train reference window, independent of label. The original Phase 1 implementation instead used eligible strong-positive train-target counts. It therefore mislabeled data-warm-but-positive-untouched items as Cold.",
        "",
        f"- Original merge commit: `{ORIGINAL_MERGE_COMMIT}`",
        f"- Fix code commit: `{result['code_commit']}`",
        f"- Corrected run time: `{runtime_seconds:.2f}` seconds",
        "- Temporal final accessed: **no**",
        "- Small Matrix accessed: **no**",
        "",
        "## Scope of correction",
        "",
        "Changed: Warm/Tail/Cold Recall, their denominators and bootstrap intervals, and segment membership counts/hashes.",
        "",
        "Unchanged and fail-closed checked: candidate membership, every Top-K-derived overall Recall/NDCG/Coverage value, all 97 configurations/seeds, and every selected configuration.",
        "",
        "## Membership",
        "",
        f"- data-warm items: `{int(np.asarray(segments['data_warm']).sum())}`",
        f"- data-cold items: `{int(np.asarray(segments['data_cold']).sum())}`",
        f"- head items: `{int(np.asarray(segments['head']).sum())}`",
        f"- tail items: `{int(np.asarray(segments['tail']).sum())}`",
        f"- data-warm SHA256: `{segments['data_warm_sha256']}`",
        "- model-ID-touched membership: not inferred; Phase 2 must record actual optimizer updates.",
        "",
        "## Artifact lineage",
        "",
        f"- Original selection result SHA256: `{ORIGINAL_RESULT_SHA256}`",
        f"- Original final bundle SHA256: `{ORIGINAL_BUNDLE_SHA256}`",
        f"- Original selection receipt SHA256: `{ORIGINAL_RECEIPT_SHA256}`",
        f"- Corrected selection result SHA256: `{corrected_result_sha256}`",
        f"- Corrected final bundle SHA256: `{corrected_bundle_sha256}`",
        f"- Corrected selection receipt SHA256: `{corrected_receipt_sha256}`",
        f"- Archived old receipt: `{archive_path.relative_to(repo_root)}`",
        "",
        "## Selected configurations",
        "",
        "Original and corrected selected configurations are identical:",
        "",
    ]
    for name, method in selected.items():
        lines.append(f"- `{name}`: `{method['config_id']}`")
    path.write_text("\n".join(lines) + "\n")


def run_segment_membership_erratum(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    recover_publish_transaction(_publish_journal_path(root))
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    context = _prepare_execution(root, verify_selected_fallback=True)
    code_commit = context["code_commit"]
    original_result = context["original_result"]
    original_bundle = context["original_bundle"]
    artifacts = context["artifacts"]
    segments = context["segments"]
    plan = context["plan"]
    original_rows = context["original_rows"]
    cache_dir = context["cache_dir"]
    cache_key = context["binding"]["cache_key"]

    corrected_rows: list[dict[str, Any]] = []
    total = len(plan.rows)
    reused_rows = 0
    computed_rows = 0
    _emit_progress("exact_segment_metric_replay", 0, total, started)
    for row_index, planned in enumerate(plan.rows):
        key = (planned["method"], planned["config_id"], planned["seed"])
        topk_metadata = _topk_metadata(
            row_index=row_index,
            planned=planned,
            binding=context["binding"],
            ranking_input_manifest=context["ranking_input_manifest"],
            artifacts=artifacts,
        )
        cached, _ = _load_cached_row(
            cache_dir=cache_dir,
            index=row_index,
            planned=planned,
            original=original_rows[key],
            cache_key=cache_key,
            topk_metadata=topk_metadata,
            artifacts=artifacts,
        )
        if cached is None:
            elapsed = time.perf_counter() - started
            remaining = MAX_WALL_SECONDS - elapsed
            if remaining <= 0:
                raise GateError("Erratum run exceeded the external three-hour deadline")
            command = [
                sys.executable,
                str(root / "scripts/correct_phase1_segment_row.py"),
                "--row-index",
                str(row_index),
                "--cache-key",
                cache_key,
            ]
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                returncode = run_supervised(
                    command,
                    cwd=root,
                    timeout_seconds=min(float(ROW_TIMEOUT_SECONDS), remaining),
                    liveness_interval_seconds=float(LIVENESS_INTERVAL_SECONDS),
                    liveness_callback=_liveness_callback(
                        cache_dir=cache_dir,
                        artifact_dir=context["artifact_dir"],
                        planned=planned,
                        row_index=row_index,
                        total=total,
                        overall_started=started,
                    ),
                    environment=environment,
                )
            except WatchdogTimeout as exc:
                raise GateError(
                    f"Erratum row {row_index + 1} exceeded its supervised deadline"
                ) from exc
            finally:
                _mark_row_inactive(context["artifact_dir"])
            if returncode != 0:
                raise GateError(
                    f"Erratum row {row_index + 1} worker failed with exit {returncode}"
                )
            cached, _ = _load_cached_row(
                cache_dir=cache_dir,
                index=row_index,
                planned=planned,
                original=original_rows[key],
                cache_key=cache_key,
                topk_metadata=topk_metadata,
                artifacts=artifacts,
            )
            if cached is None:
                raise GateError(
                    f"Erratum row {row_index + 1} exited without an atomic checkpoint"
                )
            computed_rows += 1
        else:
            reused_rows += 1
        corrected_rows.append(cached)
        _emit_progress(
            "exact_segment_metric_replay", row_index + 1, total, started
        )

    if time.perf_counter() - started > MAX_WALL_SECONDS:
        raise GateError("Erratum run exceeded the external three-hour deadline")

    result = copy.deepcopy(original_result)
    result["code_commit"] = code_commit
    result["rows"] = corrected_rows
    result["erratum"] = {
        "id": ERRATUM_ID,
        "reason": ERRATUM_REASON,
        "original_merge_commit": ORIGINAL_MERGE_COMMIT,
        "original_selection_result_sha256": ORIGINAL_RESULT_SHA256,
        "original_final_method_bundle_sha256": ORIGINAL_BUNDLE_SHA256,
        "original_selection_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
    }
    result["hashes"]["evaluator"] = {
        "path": ERRATUM_SELECTION_EVALUATOR,
        "sha256": sha256_file(root / ERRATUM_SELECTION_EVALUATOR),
    }
    result["hashes"]["segment_membership"] = {
        "context": "selection_train_only",
        "data_warm_item_count": int(np.asarray(segments["data_warm"]).sum()),
        "data_cold_item_count": int(np.asarray(segments["data_cold"]).sum()),
        "positive_target_item_count": int(
            (np.asarray(segments["positive_target_count"]) > 0).sum()
        ),
        "head_item_count": int(np.asarray(segments["head"]).sum()),
        "tail_item_count": int(np.asarray(segments["tail"]).sum()),
        "data_warm_membership_sha256": segments["data_warm_sha256"],
        "model_id_trained": "not_inferred",
    }
    result["run"] = {
        "seconds": float(time.perf_counter() - started),
        "peak_memory_mb": float(_peak_memory_mb()),
        "processed_cache_hit": True,
        "full_protocol_verification_count_this_run": 0,
    }
    result["scope_access"] = {"temporal_final": False, "small_matrix": False}

    report_dir = root / "reports/phase1"
    result_path = report_dir / "validation_baselines.json"
    bundle_path = report_dir / "final_method_bundle.json"
    staging = cache_dir / "formal_staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    staged_result = staging / "validation_baselines.json"
    staged_bundle = staging / "final_method_bundle.json"
    staged_markdown = staging / "validation_baselines.md"
    staged_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    validate_selection_result(root, staged_result, plan=plan)
    corrected_bundle = derive_final_method_bundle(
        root, staged_result, staged_bundle
    )
    validate_selection_erratum_invariants(
        original_result, result, original_bundle, corrected_bundle
    )
    _write_markdown(staged_markdown, result, corrected_bundle)
    selected_at_utc = datetime.now(timezone.utc).isoformat()
    receipt_path, archive_path, old_receipt_bytes, receipt_payload = (
        _build_replacement_receipt(
            root=root,
            result_path=staged_result,
            bundle_path=staged_bundle,
            code_commit=code_commit,
            plan=plan,
            selected_at_utc=selected_at_utc,
        )
    )
    staged_receipt = staging / "SELECTION_RECEIPT.json"
    staged_archive = staging / "SUPERSEDED_SELECTION_RECEIPT.json"
    _atomic_write_json(staged_receipt, receipt_payload)
    staged_archive.write_bytes(old_receipt_bytes)
    os.chmod(staged_receipt, 0o444)
    os.chmod(staged_archive, 0o444)
    runtime_seconds = time.perf_counter() - started
    run_record = {
        "schema_version": 1,
        "erratum_id": ERRATUM_ID,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "code_commit": code_commit,
        "selection_rows": 97,
        "resumed_rows": reused_rows,
        "computed_rows": computed_rows,
        "row_timeout_seconds": ROW_TIMEOUT_SECONDS,
        "liveness_interval_seconds": LIVENESS_INTERVAL_SECONDS,
        "data_warm_item_count": int(np.asarray(segments["data_warm"]).sum()),
        "data_cold_item_count": int(np.asarray(segments["data_cold"]).sum()),
        "data_warm_membership_sha256": segments["data_warm_sha256"],
        "original_selection_result_sha256": ORIGINAL_RESULT_SHA256,
        "corrected_selection_result_sha256": sha256_file(staged_result),
        "original_final_method_bundle_sha256": ORIGINAL_BUNDLE_SHA256,
        "corrected_final_method_bundle_sha256": sha256_file(staged_bundle),
        "original_selection_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "corrected_selection_receipt_sha256": sha256_file(staged_receipt),
        "archived_receipt_path": str(archive_path.relative_to(root)),
        "temporal_final_accessed": False,
        "small_matrix_accessed": False,
    }
    staged_run_record = staging / "erratum_001_run.json"
    _atomic_write_json(staged_run_record, run_record)
    staged_erratum = staging / "ERRATUM-001.md"
    _write_erratum(
        repo_root=root,
        path=staged_erratum,
        result=result,
        bundle=corrected_bundle,
        archive_path=archive_path,
        segments=segments,
        runtime_seconds=runtime_seconds,
        corrected_result_sha256=sha256_file(staged_result),
        corrected_bundle_sha256=sha256_file(staged_bundle),
        corrected_receipt_sha256=sha256_file(staged_receipt),
    )
    publish_files_transactionally(
        journal_path=_publish_journal_path(root),
        replacements={
            result_path: staged_result,
            bundle_path: staged_bundle,
            report_dir / "validation_baselines.md": staged_markdown,
            archive_path: staged_archive,
            receipt_path: staged_receipt,
            report_dir / "erratum_001_run.json": staged_run_record,
            report_dir / "ERRATUM-001.md": staged_erratum,
        },
    )
    if (
        sha256_file(result_path) != run_record["corrected_selection_result_sha256"]
        or sha256_file(bundle_path)
        != run_record["corrected_final_method_bundle_sha256"]
        or sha256_file(receipt_path)
        != run_record["corrected_selection_receipt_sha256"]
        or sha256_file(archive_path) != ORIGINAL_RECEIPT_SHA256
    ):
        raise GateError("Committed formal publication differs from its staging hashes")
    return run_record
