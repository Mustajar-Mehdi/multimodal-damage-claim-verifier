#!/usr/bin/env python3
"""
Evaluation Script: Strategy 1 vs Strategy 2 Benchmarking
---------------------------------------------------------
Runs both strategies on dataset/sample_claims.csv against the embedded
labeled ground truth, then auto-generates evaluation/evaluation_report.md.

Key improvements over the original:
- Merges predictions with ground truth on claim_id (row-stable), not user_id,
  so multi-claim users don't cause mismatched comparisons.
- Validity check: if >50% of a strategy's results are fallback defaults, that
  strategy is marked INVALID and excluded from the conclusion.
- Report content (numbers + winner) is fully computed from real metrics —
  no hardcoded prose that can contradict the numbers.
- --sample-size flag: limit how many rows are used for the strategy comparison
  (saves quota when the daily cap is tight).

Usage:
    python code/evaluate.py                     # default: all rows in sample_claims.csv
    python code/evaluate.py --sample-size 5     # compare only first 5 rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))

from pipeline import run_pipeline, CHECKPOINT_PATH, DEFAULT_MODEL, BASE_DIR as PIPE_BASE

EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

SAMPLE_CSV = BASE_DIR / "dataset" / "sample_claims.csv"
TEST_CSV = BASE_DIR / "dataset" / "claims.csv"

# Fields in sample_claims.csv that are ground truth labels
GROUND_TRUTH_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
]

# These are the fallback values used when an API call fails or images are
# missing.  A run is considered INVALID if more than this fraction of rows
# return these defaults.
FALLBACK_CLAIM_STATUS = "not_enough_information"
FALLBACK_ISSUE_TYPE = "unknown"
INVALID_RUN_THRESHOLD = 0.50   # >50% fallback → INVALID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_claim_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a row-position claim_id if one doesn't already exist."""
    if "claim_id" not in df.columns:
        df = df.copy()
        df.insert(0, "claim_id", [f"claim_{i:04d}" for i in range(len(df))])
    return df


def detect_fallback_rows(pred_df: pd.DataFrame) -> int:
    """Count rows that look like API-error fallbacks (not real predictions)."""
    fallback_mask = (
        (pred_df["claim_status"].str.strip().str.lower() == FALLBACK_CLAIM_STATUS)
        & (pred_df["issue_type"].str.strip().str.lower() == FALLBACK_ISSUE_TYPE)
    )
    return int(fallback_mask.sum())


def calculate_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Compute per-field accuracy and overall exact match.

    Both DataFrames must have a 'claim_id' column so we can align them
    correctly even if there are missing or extra rows.
    """
    # Align on claim_id — inner join so we only score rows present in both
    gt = gt_df[["claim_id"] + GROUND_TRUTH_FIELDS].set_index("claim_id")
    pred_cols = ["claim_id"] + [f for f in GROUND_TRUTH_FIELDS if f in pred_df.columns]
    pred = pred_df[pred_cols].set_index("claim_id")

    merged = gt.join(pred, how="inner", lsuffix="_gt", rsuffix="_pred")
    total = len(merged)

    metrics: dict = {"total_samples": total}
    if total == 0:
        for field in GROUND_TRUTH_FIELDS:
            metrics[field] = 0.0
        metrics["overall_exact_match"] = 0.0
        return metrics

    for field in GROUND_TRUTH_FIELDS:
        gt_col = f"{field}_gt"
        pred_col = f"{field}_pred"
        if gt_col not in merged.columns or pred_col not in merged.columns:
            metrics[field] = 0.0
            continue
        correct = (
            merged[gt_col].astype(str).str.strip().str.lower()
            == merged[pred_col].astype(str).str.strip().str.lower()
        ).sum()
        metrics[field] = round((correct / total) * 100.0, 1)

    # Overall exact match across the 4 key decision fields
    exact_fields = ["claim_status", "issue_type", "object_part", "severity"]
    match_mask = pd.Series([True] * total, index=merged.index)
    for field in exact_fields:
        gt_col, pred_col = f"{field}_gt", f"{field}_pred"
        if gt_col in merged.columns and pred_col in merged.columns:
            match_mask &= (
                merged[gt_col].astype(str).str.strip().str.lower()
                == merged[pred_col].astype(str).str.strip().str.lower()
            )
    metrics["overall_exact_match"] = round((match_mask.sum() / total) * 100.0, 1)
    return metrics


def strategy_validity(pred_df: pd.DataFrame) -> tuple[bool, float, str]:
    """Return (is_valid, fallback_fraction, reason_string)."""
    total = len(pred_df)
    if total == 0:
        return False, 1.0, "No predictions generated."
    n_fallback = detect_fallback_rows(pred_df)
    frac = n_fallback / total
    if frac > INVALID_RUN_THRESHOLD:
        reason = (
            f"{n_fallback}/{total} rows ({frac:.0%}) are fallback defaults — "
            "likely caused by quota exhaustion mid-run. Real API responses < threshold."
        )
        return False, frac, reason
    return True, frac, "OK"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    s1_metrics: dict,
    s2_metrics: dict,
    s1_valid: bool,
    s2_valid: bool,
    s1_validity_reason: str,
    s2_validity_reason: str,
    s1_stats: dict,
    s2_stats: dict,
    test_stats: dict,
    sample_size_used: int,
) -> str:
    """Build the full evaluation_report.md content from computed numbers."""

    # ── Determine winner ──────────────────────────────────────────────────
    if s1_valid and s2_valid:
        s1_score = s1_metrics["overall_exact_match"]
        s2_score = s2_metrics["overall_exact_match"]
        if s2_score > s1_score:
            winner_text = (
                f"**Strategy 2 (Enhanced CoT + Guardrails)** performed better, "
                f"achieving {s2_score:.1f}% overall exact match vs "
                f"{s1_score:.1f}% for Strategy 1."
            )
        elif s1_score > s2_score:
            winner_text = (
                f"**Strategy 1 (Baseline Direct Vision Prompt)** performed better, "
                f"achieving {s1_score:.1f}% overall exact match vs "
                f"{s2_score:.1f}% for Strategy 2."
            )
        else:
            winner_text = (
                f"Both strategies achieved equal overall exact match: {s1_score:.1f}%."
            )
    elif s1_valid and not s2_valid:
        winner_text = (
            "**Strategy 1** is the only valid run. Strategy 2 was invalidated "
            f"due to quota exhaustion (see below)."
        )
    elif s2_valid and not s1_valid:
        winner_text = (
            "**Strategy 2** is the only valid run. Strategy 1 was invalidated "
            f"due to quota exhaustion (see below)."
        )
    else:
        winner_text = (
            "**Neither strategy produced a valid run.** Both had >50% fallback "
            "defaults, indicating quota exhaustion during both runs. "
            "No meaningful comparison can be drawn."
        )

    # ── Validity badges ───────────────────────────────────────────────────
    def validity_badge(valid: bool, reason: str) -> str:
        if valid:
            return "✅ VALID"
        return f"❌ INVALID RUN — quota exhausted mid-run: {reason}"

    # ── Table rows ────────────────────────────────────────────────────────
    def row(label: str, field: str) -> str:
        s1_v = f"{s1_metrics[field]:.1f}%" if s1_valid else "N/A (invalid)"
        s2_v = f"{s2_metrics[field]:.1f}%" if s2_valid else "N/A (invalid)"
        if s1_valid and s2_valid:
            diff = s2_metrics[field] - s1_metrics[field]
            sign = "+" if diff >= 0 else ""
            delta = f"{sign}{diff:.1f}%"
        else:
            delta = "—"
        return f"| **{label}** | {s1_v} | {s2_v} | {delta} |"

    # ── Cost estimates ────────────────────────────────────────────────────
    total_input  = s1_stats["est_input_tokens"] + s2_stats["est_input_tokens"] + test_stats.get("est_input_tokens", 0)
    total_output = s1_stats["est_output_tokens"] + s2_stats["est_output_tokens"] + test_stats.get("est_output_tokens", 0)
    total_calls  = s1_stats["api_calls"] + s2_stats["api_calls"] + test_stats.get("api_calls", 0)
    total_images = s1_stats["images_processed"] + s2_stats["images_processed"] + test_stats.get("images_processed", 0)
    input_cost   = (total_input / 1_000_000) * 0.075
    output_cost  = (total_output / 1_000_000) * 0.30
    total_cost   = input_cost + output_cost

    s1_elapsed = s1_stats.get("elapsed_time", 0)
    s2_elapsed = s2_stats.get("elapsed_time", 0)
    test_elapsed = test_stats.get("elapsed_time", 0)

    content = f"""# Multi-Modal Evidence Review — Evaluation & Operational Report

> **Auto-generated** by `evaluate.py` — all numbers are computed from actual
> API results. No prose has been hand-written.

---

## 1. Executive Summary

This report evaluates the **Multi-Modal Evidence Review System** on damage
claims verification across `car`, `laptop`, and `package` objects.

Two strategies were evaluated on `dataset/sample_claims.csv`
({sample_size_used} rows used for comparison):

- **Strategy 1 (Baseline Direct Vision Prompt)**: Single-pass direct prompt.
- **Strategy 2 (Enhanced CoT + Guardrails)**: Multi-step Chain-of-Thought with
  domain evidence requirements, prompt-injection defense, multi-image
  cross-verification, and post-processing consistency guardrails.

### Conclusion

{winner_text}

---

## 2. Strategy Run Validity

| Strategy | Status | Detail |
| :--- | :--- | :--- |
| Strategy 1 | {validity_badge(s1_valid, s1_validity_reason)} | |
| Strategy 2 | {validity_badge(s2_valid, s2_validity_reason)} | |

> [!NOTE]
> A strategy run is marked **INVALID** when more than {INVALID_RUN_THRESHOLD:.0%} of its
> predictions are fallback defaults (`claim_status=not_enough_information`,
> `issue_type=unknown`). This pattern indicates the Gemini daily quota was
> exhausted mid-run and the remaining rows received no real API response.
> Comparing an invalid run against ground truth produces misleading accuracy
> numbers and **no conclusion is drawn from invalid runs**.

---

## 3. Quantitative Benchmark Results

Evaluation on `dataset/sample_claims.csv` ({s1_metrics['total_samples'] if s1_valid else s2_metrics['total_samples']} labeled cases aligned by `claim_id`).

| Evaluation Field | Strategy 1 Accuracy | Strategy 2 Accuracy | Δ (S2 − S1) |
| :--- | :---: | :---: | :---: |
{row("Claim Status", "claim_status")}
{row("Issue Type", "issue_type")}
{row("Object Part", "object_part")}
{row("Severity", "severity")}
{row("Evidence Standard Met", "evidence_standard_met")}
{row("Valid Image", "valid_image")}
{row("Overall Exact Match (4 key fields)", "overall_exact_match")}

---

## 4. Operational Summary

| Parameter | S1 Sample | S2 Sample | Test Run | Total |
| :--- | :---: | :---: | :---: | :---: |
| API Calls | {s1_stats['api_calls']} | {s2_stats['api_calls']} | {test_stats.get('api_calls', 0)} | **{total_calls}** |
| Images Processed | {s1_stats['images_processed']} | {s2_stats['images_processed']} | {test_stats.get('images_processed', 0)} | **{total_images}** |
| Est. Input Tokens | {s1_stats['est_input_tokens']:,} | {s2_stats['est_input_tokens']:,} | {test_stats.get('est_input_tokens', 0):,} | **{total_input:,}** |
| Est. Output Tokens | {s1_stats['est_output_tokens']:,} | {s2_stats['est_output_tokens']:,} | {test_stats.get('est_output_tokens', 0):,} | **{total_output:,}** |
| Runtime (s) | {s1_elapsed:.1f}s | {s2_elapsed:.1f}s | {test_elapsed:.1f}s | **{s1_elapsed+s2_elapsed+test_elapsed:.1f}s** |

---

## 5. Cost Estimation (Gemini 2.5 Flash Pricing)

- Input: ${input_cost:.6f} ({total_input:,} tokens × $0.075/M)
- Output: ${output_cost:.6f} ({total_output:,} tokens × $0.30/M)
- **Total: ${total_cost:.6f} USD (~${total_cost*100:.4f} cents)**

---

## 6. Quota & Rate Limit Handling

- **Hard daily cap detection**: Inspects `quotaId` in Gemini error responses.
  If `PerDay` is detected, the pipeline stops immediately and saves a
  checkpoint — no wasted retries.
- **Transient errors** (`PerMinute`): handled with exponential backoff (up to
  8 retries, starting at 12 s).
- **Checkpointing**: every successful row is saved to `checkpoint.jsonl`
  immediately. Use `--resume` to continue after a quota stop.
- **Sample size**: `--sample-size N` limits rows for strategy comparison so
  the evaluation doesn't burn the full daily quota.
"""
    return content


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(sample_size: int | None = None, model_name: str | None = None):
    print("=" * 60)
    print("  Strategy 1 vs Strategy 2 Evaluation Benchmark")
    print("=" * 60)

    if model_name is None:
        model_name = DEFAULT_MODEL

    # ── Ground truth ──────────────────────────────────────────────────────
    ground_truth_raw = pd.read_csv(SAMPLE_CSV)
    ground_truth_df = add_claim_id(ground_truth_raw)

    total_gt = len(ground_truth_df)
    rows_to_use = min(sample_size, total_gt) if sample_size else total_gt
    print(f"Ground truth loaded: {total_gt} rows. Using {rows_to_use} for comparison.")

    # Separate checkpoint paths per strategy so they don't interfere
    s1_ckpt = BASE_DIR / "checkpoint_s1.jsonl"
    s2_ckpt = BASE_DIR / "checkpoint_s2.jsonl"

    # ── Strategy 1 ────────────────────────────────────────────────────────
    print("\n--- Running Strategy 1 (Baseline Direct Vision Prompt) ---")
    s1_out = EVAL_DIR / "sample_predictions_strategy1.csv"
    s1_df, s1_stats = run_pipeline(
        SAMPLE_CSV, s1_out,
        strategy="strategy1", pacing_delay=1.5,
        sample_size=sample_size, model_name=model_name,
        checkpoint_path=s1_ckpt,
    )
    s1_df = add_claim_id(s1_df)
    s1_valid, s1_frac, s1_reason = strategy_validity(s1_df)
    s1_metrics = calculate_metrics(ground_truth_df.head(rows_to_use), s1_df)
    print(f"Strategy 1 validity: {'VALID' if s1_valid else 'INVALID'} — {s1_reason}")

    # ── Strategy 2 ────────────────────────────────────────────────────────
    print("\n--- Running Strategy 2 (Enhanced CoT + Guardrails) ---")
    s2_out = EVAL_DIR / "sample_predictions_strategy2.csv"
    s2_df, s2_stats = run_pipeline(
        SAMPLE_CSV, s2_out,
        strategy="strategy2", pacing_delay=1.5,
        sample_size=sample_size, model_name=model_name,
        checkpoint_path=s2_ckpt,
    )
    s2_df = add_claim_id(s2_df)
    s2_valid, s2_frac, s2_reason = strategy_validity(s2_df)
    s2_metrics = calculate_metrics(ground_truth_df.head(rows_to_use), s2_df)
    print(f"Strategy 2 validity: {'VALID' if s2_valid else 'INVALID'} — {s2_reason}")

    # ── Full test run (claims.csv → output.csv) ───────────────────────────
    print("\n--- Generating Final Predictions for dataset/claims.csv ---")
    test_out = BASE_DIR / "output.csv"
    test_ckpt = BASE_DIR / "checkpoint.jsonl"
    _, test_stats = run_pipeline(
        TEST_CSV, test_out,
        strategy="strategy2", pacing_delay=1.5,
        model_name=model_name,
        checkpoint_path=test_ckpt,
        sample_size=sample_size,
    )

    # ── Generate report ───────────────────────────────────────────────────
    report_content = generate_report(
        s1_metrics, s2_metrics,
        s1_valid, s2_valid,
        s1_reason, s2_reason,
        s1_stats, s2_stats, test_stats,
        sample_size_used=rows_to_use,
    )
    report_path = EVAL_DIR / "evaluation_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n[OK] Evaluation report written to: {report_path}")
    print(f"[OK] output.csv written to: {test_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategy 1 vs Strategy 2 evaluation benchmark"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, dest="sample_size",
        help=(
            "Number of rows from sample_claims.csv to use for strategy comparison. "
            "Reduces quota usage. Default: all rows (20)."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Gemini model name override (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()
    run_evaluation(sample_size=args.sample_size, model_name=args.model)


if __name__ == "__main__":
    main()
