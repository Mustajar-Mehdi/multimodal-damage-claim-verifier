#!/usr/bin/env python3
"""
checkpoint.py
-------------
Lightweight JSONL checkpoint for the claims pipeline.

Each successfully processed row is appended as one JSON line immediately
after the API call returns — before the next row starts.  This means that
if the pipeline is killed (quota, crash, Ctrl-C), every successfully
processed row is already persisted on disk.

On resume, load_checkpoint() returns a dict keyed by claim_id so the
caller can skip rows that were already processed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(checkpoint_path: Path, row: Dict[str, Any]) -> None:
    """Append one successfully-processed row to the checkpoint file.

    Args:
        checkpoint_path: Path to the .jsonl checkpoint file.
        row: The full output row dict (must include 'claim_id').
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all checkpointed rows, keyed by claim_id.

    Returns an empty dict if the checkpoint file does not exist.

    Args:
        checkpoint_path: Path to the .jsonl checkpoint file.

    Returns:
        Mapping of claim_id → row dict for every successfully saved row.
    """
    if not checkpoint_path.is_file():
        return {}

    rows: Dict[str, Dict[str, Any]] = {}
    with open(checkpoint_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = str(obj.get("claim_id", ""))
                if cid:
                    rows[cid] = obj
                else:
                    # Fallback: use line number as key so we don't silently
                    # drop data even if claim_id is missing
                    rows[f"__line_{line_no}"] = obj
            except json.JSONDecodeError:
                pass  # Skip malformed lines, don't crash resume

    return rows


def delete_checkpoint(checkpoint_path: Path) -> None:
    """Remove the checkpoint file after a successful full run."""
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


def checkpoint_count(checkpoint_path: Path) -> int:
    """Return the number of rows saved in the checkpoint file."""
    return len(load_checkpoint(checkpoint_path))
