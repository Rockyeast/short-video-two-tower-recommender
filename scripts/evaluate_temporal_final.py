#!/usr/bin/env python3
"""Registered temporal-final evaluator; never invoke during Phase 1 selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def evaluate_frozen_final(
    *, repo_root: Path, bundle: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Delegate to the shared evaluator only after the exactly-once claim exists."""

    from kuairec_phase1.runner import evaluate_frozen_final

    return evaluate_frozen_final(repo_root=repo_root, bundle=bundle)
