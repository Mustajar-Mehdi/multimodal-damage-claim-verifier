import pandas as pd
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    
    # Typically pipeline saves predictions here
    predictions_path = base_dir / "dataset" / "test_predictions.csv"
    
    # We score it against sample_claims.csv which likely acts as ground truth
    expected_path = base_dir / "dataset" / "sample_claims.csv"
    
    if not expected_path.exists():
        print(f"Error: Expected outputs file not found at {expected_path}")
        sys.exit(1)
        
    if not predictions_path.exists():
        print(f"Error: Predictions file not found at {predictions_path}")
        sys.exit(1)
        
    try:
        preds = pd.read_csv(predictions_path)
        expected = pd.read_csv(expected_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)

    if 'user_id' not in preds.columns or 'user_id' not in expected.columns:
        print("Both CSVs must contain 'user_id' column for matching.")
        sys.exit(1)

    # Merge on user_id to align rows
    merged = pd.merge(expected, preds, on='user_id', suffixes=('_expected', '_pred'))
    
    if merged.empty:
        print("No matching user_ids found between predictions and expected outputs.")
        sys.exit(1)

    # Ignore inputs or metadata
    ignore_cols = ['user_id', 'image_paths', 'user_claim', 'claim_object']
    
    # Dynamically find columns that exist in both DataFrames
    eval_cols = [c for c in expected.columns if c not in ignore_cols and f"{c}_pred" in merged.columns]
    
    total_matches = 0
    total_evals = 0
    
    results = {}
    
    for col in eval_cols:
        expected_col = merged[f"{col}_expected"].astype(str).str.lower().str.strip()
        pred_col = merged[f"{col}_pred"].astype(str).str.lower().str.strip()
        
        matches = (expected_col == pred_col).sum()
        total_matches += matches
        total_evals += len(merged)
        results[col] = matches / len(merged)
        
    print(f"Evaluated {len(merged)} rows.")
    for col, score in results.items():
        print(f"Accuracy for {col}: {score:.2%}")
        
    if total_evals > 0:
        overall = total_matches / total_evals
        print(f"\nOverall Score: {overall:.2%}")
    else:
        print("No columns to evaluate.")

if __name__ == "__main__":
    main()
