from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuairec_phase1.publish import (
    PublishTransactionError,
    publish_files_transactionally,
    recover_publish_transaction,
)


def _files(tmp_path: Path):
    destination_a = tmp_path / "formal" / "result.json"
    destination_b = tmp_path / "formal" / "receipt.json"
    destination_a.parent.mkdir(parents=True)
    destination_a.write_text("old-result")
    destination_b.write_text("old-receipt")
    staged_a = tmp_path / "stage" / "result.json"
    staged_b = tmp_path / "stage" / "receipt.json"
    staged_a.parent.mkdir(parents=True)
    staged_a.write_text("new-result")
    staged_b.write_text("new-receipt")
    return destination_a, destination_b, staged_a, staged_b


def test_multi_file_publication_commits_all_files(tmp_path: Path):
    destination_a, destination_b, staged_a, staged_b = _files(tmp_path)
    journal = tmp_path / "ignored" / "PUBLISH_JOURNAL.json"

    publish_files_transactionally(
        journal_path=journal,
        replacements={destination_a: staged_a, destination_b: staged_b},
    )

    assert destination_a.read_text() == "new-result"
    assert destination_b.read_text() == "new-receipt"
    assert json.loads(journal.read_text())["state"] == "committed"


def test_mid_publication_exception_rolls_back_immediately(tmp_path: Path):
    destination_a, destination_b, staged_a, staged_b = _files(tmp_path)
    journal = tmp_path / "ignored" / "PUBLISH_JOURNAL.json"

    with pytest.raises(PublishTransactionError, match="Injected"):
        publish_files_transactionally(
            journal_path=journal,
            replacements={destination_a: staged_a, destination_b: staged_b},
            fault_after_replacements=1,
        )

    assert destination_a.read_text() == "old-result"
    assert destination_b.read_text() == "old-receipt"
    assert json.loads(journal.read_text())["state"] == "rolled_back"


def test_recovery_journal_repairs_hard_crash_before_git_clean_check(tmp_path: Path):
    destination_a, destination_b, staged_a, staged_b = _files(tmp_path)
    journal = tmp_path / "ignored" / "PUBLISH_JOURNAL.json"

    with pytest.raises(PublishTransactionError, match="Injected"):
        publish_files_transactionally(
            journal_path=journal,
            replacements={destination_a: staged_a, destination_b: staged_b},
            fault_after_replacements=1,
            leave_for_recovery_on_fault=True,
        )
    assert destination_a.read_text() == "new-result"
    assert destination_b.read_text() == "old-receipt"
    assert json.loads(journal.read_text())["state"] == "publishing"

    assert recover_publish_transaction(journal) == "rolled_back"
    assert destination_a.read_text() == "old-result"
    assert destination_b.read_text() == "old-receipt"
    assert json.loads(journal.read_text())["state"] == "rolled_back"
