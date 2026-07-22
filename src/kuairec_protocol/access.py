"""Fail-closed protocol verification and one-shot evaluation receipts.

This module intentionally contains no model, baseline, metric, or final-label
loader.  It verifies the frozen protocol inputs and provides the only receipt
primitives that later experiment entrypoints may use.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SELECTION_FIT_SPLITS = ("train",)
SELECTION_EVALUATION_SPLIT = "validation"
FINAL_FIT_SPLITS = ("train", "validation")
FINAL_EVALUATION_SPLIT = "temporal_final"
DERIVED_SPLITS = ("train", "validation", "temporal_final")
PROTECTED_EVALUATION_SCOPES = frozenset(
    {FINAL_EVALUATION_SPLIT, "small_matrix_audit"}
)
FINAL_CONFIRMATION = "RUN_FROZEN_TEMPORAL_FINAL_ONCE"
PROTOCOL_REVISION = "protocol-v2.1.1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_UNTRACKED_OUTPUT_ROOTS = frozenset(
    {"artifacts", "experiments", "receipts", "reports"}
)
_FORBIDDEN_UNTRACKED_SOURCE_SUFFIXES = frozenset(
    {
        ".bash",
        ".dll",
        ".dylib",
        ".fish",
        ".pth",
        ".py",
        ".pyi",
        ".pyx",
        ".sh",
        ".so",
        ".zsh",
    }
)

DerivedHashes = Mapping[str, Mapping[str, str]]
DerivedHashRebuilder = Callable[[Path, Mapping[str, Any]], DerivedHashes]


class ProtocolAccessError(RuntimeError):
    """Raised before an experiment is allowed to open evaluation data."""


class ReceiptAlreadyExistsError(ProtocolAccessError):
    """Raised when an immutable one-shot receipt already exists."""


@dataclass(frozen=True)
class LockVerification:
    manifest_path: str
    manifest_sha256: str
    lock_path: str
    lock_sha256: str
    protocol_revision: str
    protected_scopes: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolBundleVerification(LockVerification):
    repo_root: str
    config_path: str
    config_sha256: str
    contract_sha256: tuple[tuple[str, str], ...]
    generation_code_sha256: str
    source_file_sha256: tuple[tuple[str, str], ...]
    canonical_target_sha256: tuple[tuple[str, str], ...]
    candidate_membership_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ExperimentBundleVerification:
    path: str
    sha256: str
    code_commit: str
    method_names: tuple[str, ...]
    method_table_sha256: str
    methods_canonical_json: str


@dataclass(frozen=True)
class SelectionReceiptVerification:
    path: str
    sha256: str
    experiment_bundle_sha256: str
    selection_result_sha256: str


@dataclass(frozen=True)
class FinalRequest:
    """One frozen, complete experiment table authorized for final once."""

    repo_root: str
    protocol_revision: str
    split_manifest_path: str
    split_manifest_sha256: str
    holdout_lock_path: str
    holdout_lock_sha256: str
    experiment_bundle_path: str
    experiment_bundle_sha256: str
    method_table_sha256: str
    selection_receipt_path: str
    selection_receipt_sha256: str
    code_commit: str
    method_names: tuple[str, ...]
    fit_splits: tuple[str, ...]
    evaluation_split: str
    claimed_at_utc: str


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolAccessError(f"Cannot read valid {label} JSON: {file_path}") from exc
    if not isinstance(value, dict):
        raise ProtocolAccessError(f"{label} must be a JSON object: {file_path}")
    return value


def _read_yaml_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolAccessError(f"Cannot read valid {label} YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolAccessError(f"{label} must be a YAML mapping: {path}")
    return value, raw


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolAccessError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _repo_file(repo_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProtocolAccessError(f"{label} path must be a nonempty relative string")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ProtocolAccessError(f"{label} path must be relative to repository root")
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ProtocolAccessError(f"{label} path escapes repository root: {relative}")
    if not candidate.is_file():
        raise ProtocolAccessError(f"Missing {label}: {candidate}")
    return candidate


def _relative_repo_file(repo_root: Path, path: str | Path, label: str) -> tuple[Path, str]:
    root = repo_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ProtocolAccessError(f"{label} must be stored under repository root")
    if not candidate.is_file():
        raise ProtocolAccessError(f"Missing {label}: {candidate}")
    return candidate, candidate.relative_to(root).as_posix()


def verify_manifest_lock(
    manifest_path: str | Path,
    lock_path: str | Path,
) -> LockVerification:
    """Verify that the holdout lock binds the exact manifest bytes."""

    manifest_path = Path(manifest_path).resolve()
    lock_path = Path(lock_path).resolve()
    lock = _read_json_object(lock_path, "holdout lock")
    if lock.get("manifest") != "manifests/split_manifest.json":
        raise ProtocolAccessError(
            "Holdout lock must point to manifests/split_manifest.json"
        )
    manifest = _read_json_object(manifest_path, "split manifest")
    manifest_sha = sha256_file(manifest_path)
    expected_sha = lock.get("manifest_sha256")
    if expected_sha != manifest_sha:
        raise ProtocolAccessError(
            "Holdout lock does not match split manifest: "
            f"expected {expected_sha!r}, actual {manifest_sha}"
        )
    revision = manifest.get("protocol_revision")
    if not isinstance(revision, str) or not revision:
        raise ProtocolAccessError("Split manifest protocol_revision is missing")
    if lock.get("protocol_revision") != revision:
        raise ProtocolAccessError("Holdout lock protocol_revision does not match manifest")
    if manifest.get("immutable") is not True:
        raise ProtocolAccessError("Split manifest must have immutable=true")
    if lock.get("locked") is not True:
        raise ProtocolAccessError("Holdout lock must have locked=true")
    if lock.get("ordinary_baseline_access") is not False:
        raise ProtocolAccessError("ordinary_baseline_access must be false")

    protected = lock.get("protected")
    if not isinstance(protected, list) or not all(
        isinstance(value, str) for value in protected
    ):
        raise ProtocolAccessError("Holdout lock protected scopes must be a string list")
    missing = PROTECTED_EVALUATION_SCOPES - set(protected)
    if missing:
        raise ProtocolAccessError(
            f"Holdout lock is missing protected scopes: {sorted(missing)}"
        )

    manifest_locks = manifest.get("locks")
    if not isinstance(manifest_locks, dict):
        raise ProtocolAccessError("Split manifest is missing its locks object")
    required_manifest_locks = {
        "temporal_final_locked": True,
        "small_matrix_audit_locked": True,
        "ordinary_baseline_scripts_may_run_final": False,
        "unlock_requires_separate_explicit_final_command": True,
    }
    for key, expected in required_manifest_locks.items():
        if manifest_locks.get(key) is not expected:
            raise ProtocolAccessError(
                f"Split manifest lock {key!r} must be {expected!r}"
            )

    return LockVerification(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        lock_path=str(lock_path),
        lock_sha256=sha256_file(lock_path),
        protocol_revision=revision,
        protected_scopes=tuple(sorted(protected)),
    )


def _validate_split_fractions(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    split_config = config.get("split")
    split_manifest = manifest.get("split_algorithm")
    if not isinstance(split_config, Mapping) or not isinstance(split_manifest, Mapping):
        raise ProtocolAccessError("Config and manifest must define split settings")

    fraction_contract = split_config.get("fraction_validation")
    manifest_contract = split_manifest.get("fraction_validation")
    if not isinstance(fraction_contract, Mapping) or not isinstance(
        manifest_contract, Mapping
    ):
        raise ProtocolAccessError(
            "Config and manifest must define split fraction_validation"
        )
    if dict(fraction_contract) != dict(manifest_contract):
        raise ProtocolAccessError(
            "Config and manifest split fraction_validation contracts differ"
        )
    names_value = fraction_contract.get("keys")
    expected_names = (
        "train_fraction",
        "validation_fraction",
        "temporal_final_fraction",
    )
    if not isinstance(names_value, list) or tuple(names_value) != expected_names:
        raise ProtocolAccessError(
            "split.fraction_validation.keys must list train, validation, final"
        )
    required_sum = fraction_contract.get("required_sum")
    tolerance = fraction_contract.get("absolute_tolerance")
    if (
        isinstance(required_sum, bool)
        or not isinstance(required_sum, (int, float))
        or not math.isfinite(float(required_sum))
        or not math.isclose(float(required_sum), 1.0, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ProtocolAccessError("split fraction required_sum must be exactly 1.0")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ProtocolAccessError(
            "split fraction absolute_tolerance must be a positive finite number"
        )

    values: list[float] = []
    for name in expected_names:
        value = split_config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolAccessError(f"split.{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ProtocolAccessError(f"split.{name} must be strictly between 0 and 1")
        manifest_value = split_manifest.get(name)
        if isinstance(manifest_value, bool) or not isinstance(manifest_value, (int, float)):
            raise ProtocolAccessError(f"manifest split_algorithm.{name} must be numeric")
        if not math.isclose(
            value,
            float(manifest_value),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        ):
            raise ProtocolAccessError(f"Config and manifest disagree on split.{name}")
        values.append(value)
    if not math.isclose(
        sum(values),
        float(required_sum),
        rel_tol=0.0,
        abs_tol=float(tolerance),
    ):
        raise ProtocolAccessError(
            "train/validation/final split fractions violate configured required_sum"
        )


def _normalize_derived_hashes(value: DerivedHashes, label: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ProtocolAccessError(f"{label} must be a mapping")
    normalized: dict[str, dict[str, str]] = {}
    for category in ("canonical_targets", "candidate_membership"):
        category_value = value.get(category)
        if not isinstance(category_value, Mapping):
            raise ProtocolAccessError(f"{label}.{category} must be a mapping")
        split_hashes: dict[str, str] = {}
        for split in DERIVED_SPLITS:
            split_hashes[split] = _require_sha256(
                category_value.get(split), f"{label}.{category}.{split}"
            )
        normalized[category] = split_hashes
    return normalized


def verify_protocol_bundle(
    repo_root: str | Path,
    *,
    derived_hash_rebuilder: DerivedHashRebuilder | None = None,
) -> ProtocolBundleVerification:
    """Verify every frozen protocol input plus independently rebuilt hashes.

    Merely finding derived hashes in the locked manifest is not sufficient.
    After all static generator inputs are verified, this function dynamically
    loads the reviewed generator's independent hash rebuilder (or uses an
    explicitly injected synthetic rebuilder in tests) and compares all three
    split hashes.
    """

    root = Path(repo_root).resolve()
    manifest_path = root / "manifests/split_manifest.json"
    lock_path = root / "manifests/FINAL_HOLDOUT_LOCKED.json"
    verification = verify_manifest_lock(manifest_path, lock_path)
    manifest = _read_json_object(manifest_path, "split manifest")
    if verification.protocol_revision != PROTOCOL_REVISION:
        raise ProtocolAccessError(
            f"Protocol revision must be exactly {PROTOCOL_REVISION!r}"
        )

    config_path = root / "configs/phase0.yaml"
    config, config_bytes = _read_yaml_object(config_path, "phase0 config")
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    if _require_sha256(manifest.get("config_sha256"), "manifest config_sha256") != config_sha:
        raise ProtocolAccessError("Phase0 config hash does not match manifest")
    config_protocol = config.get("protocol")
    if not isinstance(config_protocol, Mapping):
        raise ProtocolAccessError("Phase0 config is missing protocol mapping")
    if config_protocol.get("revision") != verification.protocol_revision:
        raise ProtocolAccessError("Config protocol revision does not match manifest")
    _validate_split_fractions(config, manifest)

    config_contracts = config_protocol.get("active_contracts")
    manifest_contracts = manifest.get("active_contracts")
    if not isinstance(config_contracts, Mapping) or not isinstance(
        manifest_contracts, Mapping
    ):
        raise ProtocolAccessError("Config and manifest must define active contracts")
    if set(config_contracts) != set(manifest_contracts):
        raise ProtocolAccessError("Config and manifest active contract sets differ")
    verified_contracts: list[tuple[str, str]] = []
    for name in sorted(config_contracts):
        entry = manifest_contracts.get(name)
        if not isinstance(entry, Mapping):
            raise ProtocolAccessError(f"Manifest active contract {name!r} is invalid")
        relative = config_contracts[name]
        if entry.get("path") != relative:
            raise ProtocolAccessError(f"Active contract path mismatch for {name}")
        contract_path = _repo_file(root, relative, f"active contract {name}")
        expected = _require_sha256(entry.get("sha256"), f"contract {name} sha256")
        actual = sha256_file(contract_path)
        if actual != expected:
            raise ProtocolAccessError(f"Active contract hash mismatch for {name}")
        verified_contracts.append((str(name), actual))

    code_entry = manifest.get("generation_code")
    if not isinstance(code_entry, Mapping):
        raise ProtocolAccessError("Manifest generation_code entry is missing")
    code_path = _repo_file(root, code_entry.get("path"), "generation code")
    expected_code = _require_sha256(
        code_entry.get("sha256"), "generation code sha256"
    )
    actual_code = sha256_file(code_path)
    if actual_code != expected_code:
        raise ProtocolAccessError("Generation code hash does not match manifest")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ProtocolAccessError("Manifest dataset entry is missing")
    data_root = root / "data/raw"
    archive = data_root / "KuaiRec.zip"
    if not archive.is_file():
        raise ProtocolAccessError(f"Missing KuaiRec archive: {archive}")
    expected_archive = _require_sha256(
        dataset.get("archive_sha256"), "dataset archive_sha256"
    )
    if sha256_file(archive) != expected_archive:
        raise ProtocolAccessError("KuaiRec archive hash does not match manifest")
    source_files = dataset.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ProtocolAccessError("Manifest source_files mapping is missing")
    configured_dataset = config.get("dataset")
    if not isinstance(configured_dataset, Mapping):
        raise ProtocolAccessError("Config dataset must be a mapping")
    configured_source_files = configured_dataset.get("expected_files")
    if not isinstance(configured_source_files, list) or not all(
        isinstance(name, str) and name for name in configured_source_files
    ):
        raise ProtocolAccessError("Config dataset.expected_files must be a string list")
    if set(source_files) != set(configured_source_files):
        raise ProtocolAccessError(
            "Manifest source files differ from config dataset.expected_files"
        )
    verified_sources: list[tuple[str, str]] = []
    for name in sorted(source_files):
        entry = source_files[name]
        if not isinstance(entry, Mapping):
            raise ProtocolAccessError(f"Source file entry is invalid: {name}")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ProtocolAccessError(f"Source file path must be relative: {name}")
        source = (data_root / relative).resolve()
        if not source.is_relative_to(data_root.resolve()) or not source.is_file():
            raise ProtocolAccessError(f"Missing or unsafe source file: {name}")
        expected_size = entry.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ProtocolAccessError(f"Source file size is invalid: {name}")
        if source.stat().st_size != expected_size:
            raise ProtocolAccessError(f"Source file size mismatch: {name}")
        expected = _require_sha256(entry.get("sha256"), f"source file {name} sha256")
        actual = sha256_file(source)
        if actual != expected:
            raise ProtocolAccessError(f"Source file hash mismatch: {name}")
        verified_sources.append((str(name), actual))

    catalog = manifest.get("candidate_catalog")
    hash_format = catalog.get("membership_hash_format") if isinstance(catalog, Mapping) else None
    if not isinstance(hash_format, Mapping):
        raise ProtocolAccessError(
            "Manifest candidate_catalog.membership_hash_format is required"
        )
    if hash_format.get("algorithm") != "sha256" or not isinstance(
        hash_format.get("version"), str
    ) or not hash_format.get("version"):
        raise ProtocolAccessError(
            "Candidate membership hash format requires version and algorithm=sha256"
        )

    split_entries = manifest.get("splits")
    if not isinstance(split_entries, Mapping):
        raise ProtocolAccessError("Manifest splits mapping is missing")
    expected_derived = {"canonical_targets": {}, "candidate_membership": {}}
    for split in DERIVED_SPLITS:
        entry = split_entries.get(split)
        if not isinstance(entry, Mapping):
            raise ProtocolAccessError(f"Manifest split entry is missing: {split}")
        expected_derived["canonical_targets"][split] = _require_sha256(
            entry.get("canonical_target_sha256"),
            f"splits.{split}.canonical_target_sha256",
        )
        expected_derived["candidate_membership"][split] = _require_sha256(
            entry.get("candidate_membership_sha256"),
            f"splits.{split}.candidate_membership_sha256",
        )

    if derived_hash_rebuilder is None:
        derived_hash_rebuilder = load_audit_derived_hash_rebuilder(root)
    try:
        rebuilt = derived_hash_rebuilder(root, manifest)
    except ProtocolAccessError:
        raise
    except Exception as exc:
        raise ProtocolAccessError(
            f"Independent derived hash rebuild failed: {exc}"
        ) from exc
    actual_derived = _normalize_derived_hashes(rebuilt, "actual_derived_hashes")
    for category in expected_derived:
        for split in DERIVED_SPLITS:
            if actual_derived[category][split] != expected_derived[category][split]:
                raise ProtocolAccessError(
                    f"Rebuilt {category}.{split} hash does not match manifest"
                )

    return ProtocolBundleVerification(
        **asdict(verification),
        repo_root=str(root),
        config_path=str(config_path),
        config_sha256=config_sha,
        contract_sha256=tuple(verified_contracts),
        generation_code_sha256=actual_code,
        source_file_sha256=tuple(verified_sources),
        canonical_target_sha256=tuple(
            (split, actual_derived["canonical_targets"][split])
            for split in DERIVED_SPLITS
        ),
        candidate_membership_sha256=tuple(
            (split, actual_derived["candidate_membership"][split])
            for split in DERIVED_SPLITS
        ),
    )


def load_audit_derived_hash_rebuilder(repo_root: str | Path) -> DerivedHashRebuilder:
    """Load the reviewed Phase0 recomputation helper without importing a baseline."""

    root = Path(repo_root).resolve()
    module_path = root / "scripts/audit_phase0.py"
    module_name = f"_kuairec_phase0_audit_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ProtocolAccessError(f"Cannot load Phase0 audit module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ProtocolAccessError(f"Cannot import Phase0 audit module: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
    helper = getattr(module, "recompute_protocol_derived_hashes", None)
    if not callable(helper):
        raise ProtocolAccessError(
            "Generation code must export recompute_protocol_derived_hashes"
        )

    def rebuild(repo: Path, manifest: Mapping[str, Any]) -> DerivedHashes:
        return helper(
            config_path=repo / "configs/phase0.yaml",
            data_root=repo / "data/raw",
            manifest=manifest,
        )

    return rebuild


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    if isinstance(splits, (str, bytes)):
        raise ProtocolAccessError("fit_splits must be a sequence, not one string")
    normalized = tuple(splits)
    if not all(isinstance(value, str) for value in normalized):
        raise ProtocolAccessError("fit_splits must contain only strings")
    return normalized


def authorize_baseline_selection(
    *,
    fit_splits: Sequence[str],
    evaluation_split: str,
    repo_root: str | Path,
) -> ProtocolBundleVerification:
    """Authorize train-fit/validation only after full bundle verification."""

    verification = verify_protocol_bundle(repo_root)
    normalized_fit = _normalize_splits(fit_splits)
    requested = set(normalized_fit) | {evaluation_split}
    protected = requested & PROTECTED_EVALUATION_SCOPES
    if protected:
        raise ProtocolAccessError(
            "Ordinary baseline entrypoints cannot access protected scope(s): "
            + ", ".join(sorted(protected))
        )
    if normalized_fit != SELECTION_FIT_SPLITS:
        raise ProtocolAccessError(
            f"Selection fit_splits must be exactly {SELECTION_FIT_SPLITS}"
        )
    if evaluation_split != SELECTION_EVALUATION_SPLIT:
        raise ProtocolAccessError(
            "Selection evaluation_split must be exactly "
            f"{SELECTION_EVALUATION_SPLIT!r}"
        )
    return verification


def authorize_explicit_final(
    *,
    fit_splits: Sequence[str],
    evaluation_split: str,
    verification: ProtocolBundleVerification,
) -> ProtocolBundleVerification:
    """Authorize the dedicated train+validation -> temporal-final context."""

    normalized_fit = _normalize_splits(fit_splits)
    if normalized_fit != FINAL_FIT_SPLITS:
        raise ProtocolAccessError(
            f"Final fit_splits must be exactly {FINAL_FIT_SPLITS}"
        )
    if evaluation_split != FINAL_EVALUATION_SPLIT:
        raise ProtocolAccessError(
            f"Final evaluation_split must be exactly {FINAL_EVALUATION_SPLIT!r}"
        )
    return verification


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _git_output(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolAccessError(f"Cannot verify experiment code commit: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ProtocolAccessError(
            "Cannot verify experiment code commit with git"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _verify_code_commit(repo_root: Path, code_commit: str) -> None:
    """Bind an experiment bundle to the exact clean tracked checkout."""

    top_level = Path(_git_output(repo_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repo_root.resolve():
        raise ProtocolAccessError(
            "Experiment repository root must equal the Git worktree root"
        )
    _git_output(repo_root, "cat-file", "-e", f"{code_commit}^{{commit}}")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if head != code_commit:
        raise ProtocolAccessError(
            f"Experiment bundle code_commit {code_commit} does not match HEAD {head}"
        )
    tracked_status = _git_output(
        repo_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status:
        raise ProtocolAccessError(
            "Tracked worktree/index must be clean for the frozen code_commit"
        )
    untracked = _git_output(
        repo_root, "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    unsafe_untracked = [
        path
        for path in untracked
        if not path
        or Path(path).parts[0] not in _ALLOWED_UNTRACKED_OUTPUT_ROOTS
        or Path(path).suffix.lower() in _FORBIDDEN_UNTRACKED_SOURCE_SUFFIXES
    ]
    if unsafe_untracked:
        preview = ", ".join(sorted(unsafe_untracked)[:5])
        raise ProtocolAccessError(
            "Untracked files outside the approved experiment-output roots "
            f"could bypass code_commit: {preview}"
        )


def validate_experiment_bundle(
    repo_root: str | Path,
    path: str | Path,
    verification: ProtocolBundleVerification,
) -> ExperimentBundleVerification:
    """Validate the complete final-table method/hyperparameter/seed bundle."""

    root = Path(repo_root).resolve()
    bundle_path, _ = _relative_repo_file(root, path, "experiment bundle")
    bundle = _read_json_object(bundle_path, "experiment bundle")
    if bundle.get("schema_version") != 1:
        raise ProtocolAccessError("Experiment bundle schema_version must be 1")
    if bundle.get("bundle_scope") != "complete_final_method_table":
        raise ProtocolAccessError(
            "Experiment bundle must declare bundle_scope=complete_final_method_table"
        )
    if bundle.get("protocol_revision") != verification.protocol_revision:
        raise ProtocolAccessError("Experiment bundle protocol_revision mismatch")
    if bundle.get("split_manifest_sha256") != verification.manifest_sha256:
        raise ProtocolAccessError("Experiment bundle manifest hash mismatch")
    code_commit = bundle.get("code_commit")
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise ProtocolAccessError("Experiment bundle code_commit must be full 40-char SHA1")
    _verify_code_commit(root, code_commit)
    methods = bundle.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ProtocolAccessError("Experiment bundle methods must be a nonempty list")
    names: list[str] = []
    for index, method in enumerate(methods):
        if not isinstance(method, Mapping):
            raise ProtocolAccessError(f"Experiment bundle method {index} must be a mapping")
        name = method.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolAccessError(f"Experiment bundle method {index} needs a name")
        if name in names:
            raise ProtocolAccessError(f"Duplicate experiment bundle method: {name}")
        if not isinstance(method.get("hyperparameters"), Mapping):
            raise ProtocolAccessError(f"Method {name} hyperparameters must be a mapping")
        seeds = method.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ProtocolAccessError(f"Method {name} seeds must be a nonempty list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ProtocolAccessError(f"Method {name} seeds must be non-negative integers")
        if len(seeds) != len(set(seeds)):
            raise ProtocolAccessError(f"Method {name} seeds must be unique")
        names.append(name)
    return ExperimentBundleVerification(
        path=str(bundle_path),
        sha256=sha256_file(bundle_path),
        code_commit=code_commit,
        method_names=tuple(names),
        method_table_sha256=hashlib.sha256(
            _canonical_json(methods).encode("utf-8")
        ).hexdigest(),
        methods_canonical_json=_canonical_json(methods),
    )


def receipt_root(repo_root: str | Path, manifest_sha256: str) -> Path:
    _require_sha256(manifest_sha256, "manifest receipt key")
    return Path(repo_root).resolve() / "receipts" / manifest_sha256


def selection_receipt_path(repo_root: str | Path, manifest_sha256: str) -> Path:
    return receipt_root(repo_root, manifest_sha256) / "SELECTION_RECEIPT.json"


def final_receipt_dir(repo_root: str | Path, manifest_sha256: str) -> Path:
    return receipt_root(repo_root, manifest_sha256)


def write_exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a read-only JSON receipt without overwriting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReceiptAlreadyExistsError(
                f"Refusing to overwrite one-shot receipt: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_selection_result(
    *,
    result_path: Path,
    verification: ProtocolBundleVerification,
    experiment_bundle: ExperimentBundleVerification,
) -> str:
    result = _read_json_object(result_path, "selection result")
    required = {
        "schema_version": 1,
        "result_scope": "selection_validation",
        "protocol_revision": verification.protocol_revision,
        "split_manifest_sha256": verification.manifest_sha256,
        "experiment_bundle_sha256": experiment_bundle.sha256,
        "code_commit": experiment_bundle.code_commit,
        "fit_splits": list(SELECTION_FIT_SPLITS),
        "evaluation_split": SELECTION_EVALUATION_SPLIT,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ProtocolAccessError(
                f"Selection result field {key!r} is inconsistent"
            )
    methods = result.get("methods")
    if not isinstance(methods, list):
        raise ProtocolAccessError("Selection result methods must be a list")
    if _canonical_json(methods) != experiment_bundle.methods_canonical_json:
        raise ProtocolAccessError(
            "Selection result methods/hyperparameters/seeds differ from experiment bundle"
        )
    return sha256_file(result_path)


def write_selection_receipt(
    *,
    verification: ProtocolBundleVerification,
    experiment_bundle_path: str | Path,
    selection_result_path: str | Path,
    selected_at_utc: str | None = None,
) -> SelectionReceiptVerification:
    """Bind validation selection results to one complete experiment bundle."""

    root = Path(verification.repo_root)
    bundle = validate_experiment_bundle(root, experiment_bundle_path, verification)
    result_path, result_relative = _relative_repo_file(
        root, selection_result_path, "selection result"
    )
    result_sha = _validate_selection_result(
        result_path=result_path,
        verification=verification,
        experiment_bundle=bundle,
    )
    receipt_path = selection_receipt_path(root, verification.manifest_sha256)
    payload = {
        "schema_version": 1,
        "receipt_type": "selection",
        "status": "completed",
        "protocol_revision": verification.protocol_revision,
        "split_manifest_sha256": verification.manifest_sha256,
        "holdout_lock_sha256": verification.lock_sha256,
        "experiment_bundle_path": Path(bundle.path).relative_to(root).as_posix(),
        "experiment_bundle_sha256": bundle.sha256,
        "method_table_sha256": bundle.method_table_sha256,
        "fit_splits": list(SELECTION_FIT_SPLITS),
        "evaluation_split": SELECTION_EVALUATION_SPLIT,
        "selection_result_path": result_relative,
        "selection_result_sha256": result_sha,
        "code_commit": bundle.code_commit,
        "methods": list(bundle.method_names),
        "selected_at_utc": selected_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    write_exclusive_json(receipt_path, payload)
    return validate_selection_receipt(
        verification=verification,
        experiment_bundle=bundle,
    )


def validate_selection_receipt(
    *,
    verification: ProtocolBundleVerification,
    experiment_bundle: ExperimentBundleVerification,
) -> SelectionReceiptVerification:
    """Recheck the fixed selection receipt and every file it binds."""

    root = Path(verification.repo_root)
    path = selection_receipt_path(root, verification.manifest_sha256)
    receipt = _read_json_object(path, "selection receipt")
    if path.stat().st_mode & 0o222:
        raise ProtocolAccessError("Selection receipt must be immutable/read-only")
    required = {
        "schema_version": 1,
        "receipt_type": "selection",
        "status": "completed",
        "protocol_revision": verification.protocol_revision,
        "split_manifest_sha256": verification.manifest_sha256,
        "holdout_lock_sha256": verification.lock_sha256,
        "experiment_bundle_sha256": experiment_bundle.sha256,
        "method_table_sha256": experiment_bundle.method_table_sha256,
        "fit_splits": list(SELECTION_FIT_SPLITS),
        "evaluation_split": SELECTION_EVALUATION_SPLIT,
        "code_commit": experiment_bundle.code_commit,
        "methods": list(experiment_bundle.method_names),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ProtocolAccessError(f"Selection receipt field {key!r} is inconsistent")
    bundle_file = _repo_file(
        root, receipt.get("experiment_bundle_path"), "selection experiment bundle"
    )
    if bundle_file.resolve() != Path(experiment_bundle.path).resolve():
        raise ProtocolAccessError("Selection receipt points to a different experiment bundle")
    if sha256_file(bundle_file) != experiment_bundle.sha256:
        raise ProtocolAccessError("Selection experiment bundle hash changed")
    result_file = _repo_file(
        root, receipt.get("selection_result_path"), "selection result"
    )
    expected_result_sha = _require_sha256(
        receipt.get("selection_result_sha256"), "selection result sha256"
    )
    result_sha = _validate_selection_result(
        result_path=result_file,
        verification=verification,
        experiment_bundle=experiment_bundle,
    )
    if result_sha != expected_result_sha:
        raise ProtocolAccessError("Selection result hash changed")
    return SelectionReceiptVerification(
        path=str(path),
        sha256=sha256_file(path),
        experiment_bundle_sha256=experiment_bundle.sha256,
        selection_result_sha256=expected_result_sha,
    )


def validate_final_request(
    *,
    confirmation: str,
    repo_root: str | Path,
    experiment_bundle_path: str | Path,
    now_utc: str | None = None,
) -> FinalRequest:
    """Freeze a complete experiment table without opening final labels."""

    if confirmation != FINAL_CONFIRMATION:
        raise ProtocolAccessError(
            f"Explicit final confirmation must equal {FINAL_CONFIRMATION!r}"
        )
    verification = verify_protocol_bundle(repo_root)
    authorize_explicit_final(
        fit_splits=FINAL_FIT_SPLITS,
        evaluation_split=FINAL_EVALUATION_SPLIT,
        verification=verification,
    )
    bundle = validate_experiment_bundle(repo_root, experiment_bundle_path, verification)
    selection = validate_selection_receipt(
        verification=verification,
        experiment_bundle=bundle,
    )
    return FinalRequest(
        repo_root=verification.repo_root,
        protocol_revision=verification.protocol_revision,
        split_manifest_path=verification.manifest_path,
        split_manifest_sha256=verification.manifest_sha256,
        holdout_lock_path=verification.lock_path,
        holdout_lock_sha256=verification.lock_sha256,
        experiment_bundle_path=bundle.path,
        experiment_bundle_sha256=bundle.sha256,
        method_table_sha256=bundle.method_table_sha256,
        selection_receipt_path=selection.path,
        selection_receipt_sha256=selection.sha256,
        code_commit=bundle.code_commit,
        method_names=bundle.method_names,
        fit_splits=FINAL_FIT_SPLITS,
        evaluation_split=FINAL_EVALUATION_SPLIT,
        claimed_at_utc=now_utc or datetime.now(timezone.utc).isoformat(),
    )


def verify_final_request_consistency(request: FinalRequest) -> None:
    """Rebuild the protocol and recheck frozen files immediately before claim."""

    verification = verify_protocol_bundle(request.repo_root)
    if request.fit_splits != FINAL_FIT_SPLITS:
        raise ProtocolAccessError(f"Frozen final fit_splits must be {FINAL_FIT_SPLITS}")
    if request.evaluation_split != FINAL_EVALUATION_SPLIT:
        raise ProtocolAccessError(
            f"Frozen final evaluation_split must be {FINAL_EVALUATION_SPLIT!r}"
        )
    expected = {
        "protocol revision": (request.protocol_revision, verification.protocol_revision),
        "manifest hash": (request.split_manifest_sha256, verification.manifest_sha256),
        "lock hash": (request.holdout_lock_sha256, verification.lock_sha256),
    }
    for label, (frozen, current) in expected.items():
        if frozen != current:
            raise ProtocolAccessError(f"Final request {label} changed")
    bundle = validate_experiment_bundle(
        request.repo_root, request.experiment_bundle_path, verification
    )
    if bundle.sha256 != request.experiment_bundle_sha256:
        raise ProtocolAccessError("Frozen experiment bundle hash changed")
    if bundle.method_table_sha256 != request.method_table_sha256:
        raise ProtocolAccessError("Frozen experiment method table hash changed")
    if bundle.code_commit != request.code_commit:
        raise ProtocolAccessError("Frozen experiment code_commit changed")
    if bundle.method_names != request.method_names:
        raise ProtocolAccessError("Frozen experiment method names changed")
    selection = validate_selection_receipt(
        verification=verification,
        experiment_bundle=bundle,
    )
    if selection.path != request.selection_receipt_path:
        raise ProtocolAccessError("Frozen selection receipt path changed")
    if selection.sha256 != request.selection_receipt_sha256:
        raise ProtocolAccessError("Frozen selection receipt hash changed")


def run_final_once(
    *,
    request: FinalRequest,
    evaluator: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Open final once for the whole table and atomically persist claim/result."""

    verify_final_request_consistency(request)
    receipt_dir = final_receipt_dir(request.repo_root, request.split_manifest_sha256)
    claim_path = receipt_dir / "FINAL_ATTEMPT_CLAIM.json"
    result_path = receipt_dir / "FINAL_RESULT_RECEIPT.json"
    if result_path.exists():
        raise ReceiptAlreadyExistsError(
            f"Final result receipt already exists: {result_path}"
        )
    claim = {
        "schema_version": 1,
        "receipt_type": "final_attempt_claim",
        "status": "claimed",
        **asdict(request),
    }
    write_exclusive_json(claim_path, claim)

    # A failed evaluator deliberately leaves its claim and therefore consumes
    # the one final opening for the entire experiment bundle.
    result = evaluator()
    if not isinstance(result, Mapping):
        raise ProtocolAccessError("Final evaluator result must be a mapping")
    method_results = result.get("methods")
    if not isinstance(method_results, Mapping):
        raise ProtocolAccessError("Final evaluator result.methods must be a mapping")
    if set(method_results) != set(request.method_names):
        raise ProtocolAccessError(
            "Final evaluator result.methods must exactly cover the frozen method table"
        )
    if not all(isinstance(value, Mapping) for value in method_results.values()):
        raise ProtocolAccessError(
            "Every final evaluator method result must be a mapping"
        )
    result_payload = {
        "schema_version": 1,
        "receipt_type": "final_result",
        "status": "completed",
        "claim_sha256": sha256_file(claim_path),
        "experiment_bundle_sha256": request.experiment_bundle_sha256,
        "methods": list(request.method_names),
        "result": dict(result),
    }
    write_exclusive_json(result_path, result_payload)
    return result
