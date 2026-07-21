"""Fail-closed access guards for selection and one-shot final evaluation.

This module deliberately contains no recommender, baseline, data loader, or
metric implementation.  It only verifies the committed protocol bundle before
an experiment entrypoint is allowed to choose an evaluation scope.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SELECTION_FIT_SPLITS = ("train",)
SELECTION_EVALUATION_SPLIT = "validation"
FINAL_FIT_SPLITS = ("train", "validation")
FINAL_EVALUATION_SPLIT = "temporal_final"
PROTECTED_EVALUATION_SCOPES = frozenset(
    {FINAL_EVALUATION_SPLIT, "small_matrix_audit"}
)
FINAL_CONFIRMATION = "RUN_FROZEN_TEMPORAL_FINAL_ONCE"


class ProtocolAccessError(RuntimeError):
    """Raised before protected data is opened when a protocol check fails."""


class ReceiptAlreadyExistsError(ProtocolAccessError):
    """Raised when a one-shot final-evaluation receipt already exists."""


@dataclass(frozen=True)
class LockVerification:
    manifest_path: str
    manifest_sha256: str
    lock_path: str
    lock_sha256: str
    protected_scopes: tuple[str, ...]


@dataclass(frozen=True)
class FinalRequest:
    """Frozen inputs required before a final evaluator may be attached."""

    split_manifest_path: str
    split_manifest_sha256: str
    holdout_lock_path: str
    holdout_lock_sha256: str
    method_config_path: str
    method_config_sha256: str
    artifact_manifest_path: str
    artifact_manifest_sha256: str
    fit_splits: tuple[str, ...]
    evaluation_split: str
    seed: int
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


def verify_manifest_lock(
    manifest_path: str | Path,
    lock_path: str | Path,
) -> LockVerification:
    """Verify the lock against the exact committed manifest bytes.

    File permissions are intentionally not treated as a security boundary:
    Git does not preserve read-only mode bits.  The content hash and explicit
    access policy are the portable checks.
    """

    manifest_path = Path(manifest_path)
    lock_path = Path(lock_path)
    manifest = _read_json_object(manifest_path, "split manifest")
    lock = _read_json_object(lock_path, "holdout lock")
    manifest_sha = sha256_file(manifest_path)
    expected_sha = lock.get("manifest_sha256")
    if expected_sha != manifest_sha:
        raise ProtocolAccessError(
            "Holdout lock does not match split manifest: "
            f"expected {expected_sha!r}, actual {manifest_sha}"
        )
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
        protected_scopes=tuple(sorted(protected)),
    )


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
    manifest_path: str | Path,
    lock_path: str | Path,
) -> LockVerification:
    """Authorize only train-fit/validation-evaluation baseline selection."""

    verification = verify_manifest_lock(manifest_path, lock_path)
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
    manifest_path: str | Path,
    lock_path: str | Path,
) -> LockVerification:
    """Authorize the one dedicated train+validation -> final context."""

    verification = verify_manifest_lock(manifest_path, lock_path)
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


def validate_final_request(
    *,
    confirmation: str,
    method_config_path: str | Path,
    artifact_manifest_path: str | Path,
    manifest_path: str | Path,
    lock_path: str | Path,
    seed: int,
    now_utc: str | None = None,
) -> FinalRequest:
    """Freeze and hash final-run inputs without opening final labels."""

    if confirmation != FINAL_CONFIRMATION:
        raise ProtocolAccessError(
            f"Explicit final confirmation must equal {FINAL_CONFIRMATION!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ProtocolAccessError("Final seed must be a non-negative integer")
    method_config_path = Path(method_config_path)
    artifact_manifest_path = Path(artifact_manifest_path)
    for path, label in (
        (method_config_path, "frozen method config"),
        (artifact_manifest_path, "artifact manifest"),
    ):
        if not path.is_file():
            raise ProtocolAccessError(f"Missing {label}: {path}")

    verification = authorize_explicit_final(
        fit_splits=FINAL_FIT_SPLITS,
        evaluation_split=FINAL_EVALUATION_SPLIT,
        manifest_path=manifest_path,
        lock_path=lock_path,
    )
    claimed_at = now_utc or datetime.now(timezone.utc).isoformat()
    return FinalRequest(
        split_manifest_path=verification.manifest_path,
        split_manifest_sha256=verification.manifest_sha256,
        holdout_lock_path=verification.lock_path,
        holdout_lock_sha256=verification.lock_sha256,
        method_config_path=str(method_config_path),
        method_config_sha256=sha256_file(method_config_path),
        artifact_manifest_path=str(artifact_manifest_path),
        artifact_manifest_sha256=sha256_file(artifact_manifest_path),
        fit_splits=FINAL_FIT_SPLITS,
        evaluation_split=FINAL_EVALUATION_SPLIT,
        seed=seed,
        claimed_at_utc=claimed_at,
    )


def write_exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Create a JSON receipt exactly once using the OS O_EXCL primitive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as exc:
        raise ReceiptAlreadyExistsError(
            f"Refusing to overwrite one-shot receipt: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve even a partial claim.  Once a final attempt starts, a crash
        # must not silently make the holdout reusable.
        raise


def verify_final_request_consistency(request: FinalRequest) -> None:
    """Recheck every frozen input immediately before claiming the holdout."""

    if request.fit_splits != FINAL_FIT_SPLITS:
        raise ProtocolAccessError(
            f"Frozen final fit_splits must be exactly {FINAL_FIT_SPLITS}"
        )
    if request.evaluation_split != FINAL_EVALUATION_SPLIT:
        raise ProtocolAccessError(
            "Frozen final evaluation_split must be exactly "
            f"{FINAL_EVALUATION_SPLIT!r}"
        )
    verification = authorize_explicit_final(
        fit_splits=request.fit_splits,
        evaluation_split=request.evaluation_split,
        manifest_path=request.split_manifest_path,
        lock_path=request.holdout_lock_path,
    )
    expected_files = (
        (
            request.split_manifest_path,
            request.split_manifest_sha256,
            "split manifest",
        ),
        (request.holdout_lock_path, request.holdout_lock_sha256, "holdout lock"),
        (request.method_config_path, request.method_config_sha256, "method config"),
        (
            request.artifact_manifest_path,
            request.artifact_manifest_sha256,
            "artifact manifest",
        ),
    )
    for path, expected_sha, label in expected_files:
        try:
            actual_sha = sha256_file(path)
        except OSError as exc:
            raise ProtocolAccessError(f"Cannot re-read frozen {label}: {path}") from exc
        if actual_sha != expected_sha:
            raise ProtocolAccessError(
                f"Frozen {label} hash changed: expected {expected_sha}, got {actual_sha}"
            )
    if verification.manifest_sha256 != request.split_manifest_sha256:
        raise ProtocolAccessError("Final request manifest verification drifted")
    if verification.lock_sha256 != request.holdout_lock_sha256:
        raise ProtocolAccessError("Final request lock verification drifted")


def run_final_once(
    *,
    request: FinalRequest,
    receipt_dir: str | Path,
    evaluator: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run an injected evaluator once and leave an immutable attempt claim.

    The repository CLI does not attach a real evaluator in protocol-v2.  This
    function exists so the one-shot behavior can be tested synthetically now
    and used without redesign after a final evaluator is separately reviewed.
    """

    verify_final_request_consistency(request)
    receipt_dir = Path(receipt_dir)
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

    # If evaluator raises, the claim deliberately remains and blocks retries.
    result = evaluator()
    if not isinstance(result, Mapping):
        raise ProtocolAccessError("Final evaluator result must be a mapping")
    result_payload = {
        "schema_version": 1,
        "receipt_type": "final_result",
        "status": "completed",
        "claim_sha256": sha256_file(claim_path),
        "result": dict(result),
    }
    write_exclusive_json(result_path, result_payload)
    return result
