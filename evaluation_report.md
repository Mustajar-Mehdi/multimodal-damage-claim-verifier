# Multi-Modal Evidence Review — Evaluation & Operational Report

> **Auto-generated** by `evaluate.py` — all numbers are computed from actual
> API results. No prose has been hand-written.

---

## 1. Executive Summary

This report evaluates the **Multi-Modal Evidence Review System** on damage
claims verification across `car`, `laptop`, and `package` objects.

Two strategies were evaluated on `dataset/sample_claims.csv`
(5 rows used for comparison):

- **Strategy 1 (Baseline Direct Vision Prompt)**: Single-pass direct prompt.
- **Strategy 2 (Enhanced CoT + Guardrails)**: Multi-step Chain-of-Thought with
  domain evidence requirements, prompt-injection defense, multi-image
  cross-verification, and post-processing consistency guardrails.

### Conclusion

**Neither strategy produced a valid run.** Both had >50% fallback defaults, indicating quota exhaustion during both runs. No meaningful comparison can be drawn.

---

## 2. Strategy Run Validity

| Strategy | Status | Detail |
| :--- | :--- | :--- |
| Strategy 1 | ❌ INVALID RUN — quota exhausted mid-run: 5/5 rows (100%) are fallback defaults — likely caused by quota exhaustion mid-run. Real API responses < threshold. | |
| Strategy 2 | ❌ INVALID RUN — quota exhausted mid-run: 5/5 rows (100%) are fallback defaults — likely caused by quota exhaustion mid-run. Real API responses < threshold. | |

> [!NOTE]
> A strategy run is marked **INVALID** when more than 50% of its
> predictions are fallback defaults (`claim_status=not_enough_information`,
> `issue_type=unknown`). This pattern indicates the Gemini daily quota was
> exhausted mid-run and the remaining rows received no real API response.
> Comparing an invalid run against ground truth produces misleading accuracy
> numbers and **no conclusion is drawn from invalid runs**.

---

## 3. Quantitative Benchmark Results

Evaluation on `dataset/sample_claims.csv` (5 labeled cases aligned by `claim_id`).

| Evaluation Field | Strategy 1 Accuracy | Strategy 2 Accuracy | Δ (S2 − S1) |
| :--- | :---: | :---: | :---: |
| **Claim Status** | N/A (invalid) | N/A (invalid) | — |
| **Issue Type** | N/A (invalid) | N/A (invalid) | — |
| **Object Part** | N/A (invalid) | N/A (invalid) | — |
| **Severity** | N/A (invalid) | N/A (invalid) | — |
| **Evidence Standard Met** | N/A (invalid) | N/A (invalid) | — |
| **Valid Image** | N/A (invalid) | N/A (invalid) | — |
| **Overall Exact Match (4 key fields)** | N/A (invalid) | N/A (invalid) | — |

---

## 4. Operational Summary

| Parameter | S1 Sample | S2 Sample | Test Run | Total |
| :--- | :---: | :---: | :---: | :---: |
| API Calls | 0 | 0 | 0 | **0** |
| Images Processed | 8 | 8 | 9 | **25** |
| Est. Input Tokens | 0 | 0 | 0 | **0** |
| Est. Output Tokens | 0 | 0 | 0 | **0** |
| Runtime (s) | 16.5s | 19.5s | 19.8s | **55.8s** |

---

## 5. Cost Estimation (Gemini 2.5 Flash Pricing)

- Input: $0.000000 (0 tokens × $0.075/M)
- Output: $0.000000 (0 tokens × $0.30/M)
- **Total: $0.000000 USD (~$0.0000 cents)**

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
