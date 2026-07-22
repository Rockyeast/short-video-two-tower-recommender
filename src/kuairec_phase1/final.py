"""Exactly-once launcher for the registered temporal-final evaluator."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .gates import (
    FIXED_FINAL_EVALUATOR,
    GateError,
    sha256_file,
    validate_final_method_bundle,
    validate_final_result_coverage,
)


FINAL_CONFIRMATION = "RUN_FROZEN_TEMPORAL_FINAL_ONCE"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GateError(f"Exactly-once receipt already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_registered_evaluator(root: Path, expected_sha256: str):
    path = (root / FIXED_FINAL_EVALUATOR).resolve()
    if sha256_file(path) != expected_sha256:
        raise GateError("Registered final evaluator hash changed before execution")
    name = f"_kuairec_final_evaluator_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError("Cannot load the registered final evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise GateError(f"Cannot import registered final evaluator: {exc}") from exc
    finally:
        sys.modules.pop(name, None)
    evaluator = getattr(module, "evaluate_frozen_final", None)
    if not callable(evaluator):
        raise GateError("Registered evaluator lacks evaluate_frozen_final()")
    return evaluator


def run_registered_final_once(
    *, confirmation: str, repo_root: str | Path, bundle_path: str | Path
) -> Mapping[str, Any]:
    """Claim once, load only the bound tracked evaluator, and validate all seeds."""

    if confirmation != FINAL_CONFIRMATION:
        raise GateError(f"Confirmation must equal {FINAL_CONFIRMATION!r}")
    root = Path(repo_root).resolve()
    bundle = validate_final_method_bundle(root, bundle_path)
    manifest_sha = bundle["split_manifest_sha256"]
    receipt_root = root / "receipts" / manifest_sha
    selection_receipt = receipt_root / "SELECTION_RECEIPT.json"
    if not selection_receipt.is_file():
        raise GateError("A valid selection receipt is required before final")
    selection = _read_json(selection_receipt, "selection receipt")
    bundle_file = Path(bundle_path)
    if not bundle_file.is_absolute():
        bundle_file = root / bundle_file
    if selection.get("final_method_bundle_sha256") != sha256_file(bundle_file):
        raise GateError("Selection receipt does not bind this final method bundle")

    claim_path = receipt_root / "FINAL_ATTEMPT_CLAIM.json"
    result_path = receipt_root / "FINAL_RESULT_RECEIPT.json"
    if claim_path.exists() or result_path.exists():
        raise GateError("Temporal final has already been claimed or completed")
    claim = {
        "schema_version": 1,
        "receipt_type": "final_attempt_claim",
        "status": "claimed",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_revision": bundle["protocol_revision"],
        "split_manifest_sha256": manifest_sha,
        "final_method_bundle_sha256": sha256_file(bundle_file),
        "evaluator": bundle["evaluator"],
        "methods": bundle["methods"],
    }
    _exclusive_json(claim_path, claim)

    evaluator = _load_registered_evaluator(root, bundle["evaluator"]["sha256"])
    result = evaluator(repo_root=root, bundle=bundle)
    if not isinstance(result, Mapping):
        raise GateError("Registered final evaluator must return a mapping")
    validate_final_result_coverage(bundle, result)
    receipt = {
        "schema_version": 1,
        "receipt_type": "final_result",
        "status": "completed",
        "claim_sha256": sha256_file(claim_path),
        "final_method_bundle_sha256": sha256_file(bundle_file),
        "result": dict(result),
    }
    _exclusive_json(result_path, receipt)
    return result
