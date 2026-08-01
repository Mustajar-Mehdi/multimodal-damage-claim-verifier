# Multi-Modal Evidence Review System

Damage claims verification pipeline powered by **Gemini 2.5 Flash Vision**.

This system verifies insurance damage claims across three object categories (`car`, `laptop`, `package`) by analyzing user chat transcripts, multi-image evidence, minimum evidence requirements, user risk history, and potential security prompt injections.

---

## ⚡ Quick Start

### 1. Set up your API key

```bash
# Copy the example file and fill in your key
cp code/.env.example code/.env
# Then edit code/.env and set GEMINI_API_KEY=<your_key>
```

> Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 2. Install Dependencies

```bash
pip install -r code/requirements.txt
```

### 3. Run the Processing Pipeline

```bash
python code/pipeline.py
```

This processes `dataset/claims.csv` and writes `output.csv`.

### 4. Launch the Dashboard

```bash
streamlit run code/dashboard.py
```

---

## 🔑 API Key & Free-Tier Quota

> [!IMPORTANT]
> **Free-tier daily limit is ~20 requests/day.**
> The full `claims.csv` (44 rows) **cannot be processed in a single day** on
> the free tier. Use the checkpoint/resume mechanism to spread processing
> across multiple days, or switch to a paid API key.

The pipeline automatically detects when the daily quota is exhausted:
- It **stops immediately** (no wasted retries).
- It prints a clear message with the row count completed.
- Every successfully processed row is saved to `checkpoint.jsonl` before the
  pipeline exits.

---

## 🔄 Checkpoint & Resume

Each successfully processed row is **immediately** appended to `checkpoint.jsonl`. If the pipeline stops (quota, crash, Ctrl-C), all completed work is preserved.

**To resume where you left off:**

```bash
python code/pipeline.py --resume
```

The pipeline will skip already-processed rows and only call the API for the remaining ones. Once all rows are done, the checkpoint file is automatically deleted.

---

## ⚙️ Pipeline CLI Options

```
python code/pipeline.py [options]

Options:
  --claims PATH        Input claims CSV (default: dataset/claims.csv)
  --output PATH        Output CSV path (default: output.csv)
  --strategy {strategy1,strategy2}
                       Prompt strategy (default: strategy2)
  --pacing SECONDS     Delay between API calls (default: 2.0s)
  --resume             Resume from checkpoint — skip already-processed rows
  --sample-size N      Process only the first N rows (useful when quota is tight)
  --model MODEL        Gemini model string (default: gemini-2.5-flash)
  --checkpoint PATH    Path to checkpoint file (default: checkpoint.jsonl)
```

### Examples

```bash
# Normal run
python code/pipeline.py

# Resume after a quota stop
python code/pipeline.py --resume

# Process only 5 rows (quota-safe testing)
python code/pipeline.py --sample-size 5

# Use a specific model
python code/pipeline.py --model gemini-2.5-flash
```

---

## 📊 Evaluation (Strategy 1 vs Strategy 2)

```bash
# Full evaluation on all 20 sample rows (uses ~40 API calls)
python code/evaluate.py

# Quota-safe evaluation using only 5 rows per strategy
python code/evaluate.py --sample-size 5
```

The evaluation script:
- Runs both strategies on `dataset/sample_claims.csv`
- Merges predictions with ground truth using a stable `claim_id` (row-position based), not `user_id`, to handle users with multiple claims correctly
- Detects if a strategy produced >50% fallback results (quota exhausted mid-run) and marks it **INVALID**
- Auto-generates `evaluation/evaluation_report.md` with computed numbers — the "better strategy" conclusion is determined by actual accuracy, never hardcoded

---

## 🛠️ Key Features

- **Grounded Vision Verification**: Analyzes image pixel evidence directly.
- **Strict Schema Enforcement**: 14-column output matching the problem statement.
- **Adversarial Defense**: Scans user claims for prompt injection attempts.
- **User History Integration**: Incorporates past claim risk context.
- **Quota-Safe Operation**:
  - Hard daily cap → stop immediately, print message, save checkpoint
  - Transient per-minute limit → exponential backoff retry (up to 8×)
- **Checkpoint & Resume**: Never lose completed work to a quota stop.
- **Model Name Consistency**: Single `DEFAULT_MODEL` constant (`gemini-2.5-flash`) used everywhere; override via `GEMINI_MODEL` env var or `--model` flag.
- **Interactive Streamlit Dashboard**: Visual interface for claim inspection.

---

## 📁 Repository Structure

```
Hackathon_files/
├── code/
│   ├── pipeline.py           # Main verification engine
│   ├── evaluate.py           # Strategy comparison & report generation
│   ├── dashboard.py          # Streamlit dashboard
│   ├── quota_utils.py        # Quota error detection helpers
│   ├── checkpoint.py         # JSONL checkpointing utilities
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Your API key (DO NOT COMMIT)
│   └── .env.example          # Template — copy and fill in your key
├── dataset/
│   ├── claims.csv            # Test dataset (44 rows)
│   ├── sample_claims.csv     # Labeled sample dataset (20 rows, ground truth)
│   ├── user_history.csv      # User risk history
│   ├── evidence_requirements.csv
│   └── images/               # Image assets
├── evaluation/
│   ├── sample_predictions_strategy1.csv
│   ├── sample_predictions_strategy2.csv
│   └── evaluation_report.md  # Auto-generated benchmark report
├── output.csv                # Final predictions for claims.csv
├── checkpoint.jsonl          # Resume checkpoint (auto-deleted on full success)
└── README.md
```

---

## 📋 Output Schema (14 columns)

| # | Column | Values |
|---|--------|--------|
| 1 | `user_id` | — |
| 2 | `image_paths` | semicolon-separated |
| 3 | `user_claim` | raw text |
| 4 | `claim_object` | `car` / `laptop` / `package` |
| 5 | `evidence_standard_met` | `True` / `False` |
| 6 | `evidence_standard_met_reason` | text |
| 7 | `risk_flags` | semicolon-separated or `none` |
| 8 | `issue_type` | `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown` |
| 9 | `object_part` | domain-specific enum |
| 10 | `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| 11 | `claim_status_justification` | text |
| 12 | `supporting_image_ids` | semicolon-separated or `none` |
| 13 | `valid_image` | `True` / `False` |
| 14 | `severity` | `low` / `medium` / `high` / `none` / `unknown` |
