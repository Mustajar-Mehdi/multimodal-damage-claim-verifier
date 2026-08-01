#!/usr/bin/env python3
"""
quota_utils.py
--------------
Helpers for classifying Gemini API quota errors.

Gemini returns a structured error body that includes a `quotaId` field.
We use that field to distinguish:
  - Hard daily cap  (quotaId contains "PerDay")  → cannot retry, must stop
  - Transient limit (quotaId contains "PerMinute") → exponential backoff ok
"""

from __future__ import annotations

import json
import re


class HardQuotaError(RuntimeError):
    """Raised when the Gemini hard daily request limit has been hit.

    Retrying is pointless — the quota resets at midnight PST.
    The pipeline should stop gracefully, save progress, and tell the user
    to resume the next day (or use a different API key).
    """
    def __init__(self, row_index: int, total_rows: int, quota_id: str = ""):
        self.row_index = row_index
        self.total_rows = total_rows
        self.quota_id = quota_id
        super().__init__(
            f"Daily quota exhausted at row {row_index}/{total_rows}. "
            f"Resume tomorrow or with a new key using --resume flag. "
            f"(quotaId={quota_id!r})"
        )


def _extract_quota_id(error_text: str) -> str:
    """Try to pull quotaId value out of an error string / JSON blob."""
    # Fast regex path first
    m = re.search(r'"quotaId"\s*:\s*"([^"]+)"', error_text)
    if m:
        return m.group(1)
    # Try JSON parse if error_text is a JSON object
    try:
        obj = json.loads(error_text)
        # Google API error format: {"error": {"details": [{"quotaId": "..."}]}}
        details = obj.get("error", obj).get("details", [])
        for d in details:
            qid = d.get("quotaId", "")
            if qid:
                return qid
    except (json.JSONDecodeError, AttributeError):
        pass
    return ""


def classify_quota_error(exc: Exception) -> str:
    """
    Given an exception from a Gemini API call, return:
      "hard_daily"  — PerDay quota, stop immediately
      "transient"   — PerMinute / per-second quota, retry with backoff
      "other"       — non-quota error, re-raise normally
    """
    msg = str(exc)
    quota_id = _extract_quota_id(msg)

    # Direct quotaId classification
    if "PerDay" in quota_id:
        return "hard_daily"
    if "PerMinute" in quota_id or "PerSecond" in quota_id:
        return "transient"

    # Fallback: text-based heuristics for cases where JSON isn't surfaced
    msg_lower = msg.lower()
    is_quota = (
        "429" in msg
        or "resource_exhausted" in msg_lower
        or "quota" in msg_lower
    )
    if not is_quota:
        return "other"

    # If we can see "per day" or "daily" in the raw text, treat as hard
    if re.search(r"per\s*day|daily|perday", msg_lower):
        return "hard_daily"

    # Anything else quota-related: assume transient (safe default — will retry)
    return "transient"


def get_quota_id_from_exc(exc: Exception) -> str:
    """Extract the raw quotaId string from an exception, or '' if not found."""
    return _extract_quota_id(str(exc))
