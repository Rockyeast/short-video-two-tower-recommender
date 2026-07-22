"""Crash-recoverable multi-file publication for formal experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping


class PublishTransactionError(RuntimeError):
    """Raised when a formal artifact transaction cannot be verified or recovered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.chmod(temporary, source.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rollback(journal_path: Path, journal: dict[str, object]) -> None:
    journal["state"] = "rolling_back"
    _atomic_json(journal_path, journal)
    entries = journal.get("entries")
    if not isinstance(entries, list):
        raise PublishTransactionError("Publish journal entries are invalid")
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            raise PublishTransactionError("Publish journal entry is invalid")
        destination = Path(str(entry["destination"]))
        if bool(entry["original_exists"]):
            backup = Path(str(entry["backup"]))
            if not backup.is_file() or _sha256(backup) != entry["original_sha256"]:
                raise PublishTransactionError("Publish rollback backup is missing or changed")
            _atomic_copy(backup, destination)
        else:
            destination.unlink(missing_ok=True)
    journal["state"] = "rolled_back"
    _atomic_json(journal_path, journal)


def recover_publish_transaction(journal_path: str | Path) -> str:
    """Rollback an interrupted transaction before checking Git cleanliness."""

    path = Path(journal_path)
    if not path.is_file():
        return "none"
    try:
        journal = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishTransactionError("Publish recovery journal is unreadable") from exc
    state = journal.get("state")
    if state in {"prepared", "publishing", "rolling_back"}:
        _rollback(path, journal)
        return "rolled_back"
    if state in {"committed", "rolled_back"}:
        return str(state)
    raise PublishTransactionError(f"Unknown publish journal state: {state!r}")


def publish_files_transactionally(
    *,
    journal_path: str | Path,
    replacements: Mapping[str | Path, str | Path],
    fault_after_replacements: int | None = None,
    leave_for_recovery_on_fault: bool = False,
) -> None:
    """Stage, publish and journal a set of files as one recoverable operation."""

    journal_file = Path(journal_path)
    recover_publish_transaction(journal_file)
    transaction_id = uuid.uuid4().hex
    transaction_dir = journal_file.parent / "publish_transactions" / transaction_id
    staged_dir = transaction_dir / "staged"
    backup_dir = transaction_dir / "backups"
    entries: list[dict[str, object]] = []
    for index, (destination_value, source_value) in enumerate(replacements.items()):
        destination = Path(destination_value).resolve()
        source = Path(source_value).resolve()
        if not source.is_file():
            raise PublishTransactionError(f"Staged publication file is missing: {source}")
        staged = staged_dir / f"{index:03d}.bin"
        _atomic_copy(source, staged)
        original_exists = destination.is_file()
        backup = backup_dir / f"{index:03d}.bin"
        original_sha: str | None = None
        if original_exists:
            _atomic_copy(destination, backup)
            original_sha = _sha256(backup)
        entries.append(
            {
                "destination": str(destination),
                "staged": str(staged),
                "staged_sha256": _sha256(staged),
                "original_exists": original_exists,
                "backup": str(backup),
                "original_sha256": original_sha,
                "applied": False,
            }
        )
    journal: dict[str, object] = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "state": "prepared",
        "entries": entries,
    }
    _atomic_json(journal_file, journal)
    journal["state"] = "publishing"
    _atomic_json(journal_file, journal)
    try:
        for count, entry in enumerate(entries, start=1):
            staged = Path(str(entry["staged"]))
            if _sha256(staged) != entry["staged_sha256"]:
                raise PublishTransactionError("Staged publication file changed")
            destination = Path(str(entry["destination"]))
            _atomic_copy(staged, destination)
            entry["applied"] = True
            _atomic_json(journal_file, journal)
            if fault_after_replacements == count:
                raise PublishTransactionError("Injected mid-publication failure")
        journal["state"] = "committed"
        _atomic_json(journal_file, journal)
    except BaseException:
        if not leave_for_recovery_on_fault:
            _rollback(journal_file, journal)
        raise
