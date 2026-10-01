#!/usr/bin/env python3
"""
Regenerate the internal condition key for Aya seed 42 confirmatory run
by scanning all dev generation CSVs and mapping example_id -> condition.
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEV_DIR = BASE / "artifacts" / "confirmatory_aya_2026_09_30" / "primary_seed42" / "dev"
HUMAN_DIR = BASE / "results" / "human_evaluation" / "aya_seed42_confirmatory"

def main():
    rows = []

    for csv_path in DEV_DIR.glob("dev_en_transfer_l*_a*.csv"):
        if "_metrics" in csv_path.name:
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append({
                "annotation_item_id": row["example_id"],
                "condition_name": row["condition"],
                "layer": int(row["layer"]),
                "alpha": float(row["alpha"]),
                "language": row["language"],
                "language_name": row["language_name"],
                "split": "dev",
            })

    key_df = pd.DataFrame(rows)
    out_path = HUMAN_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv"
    key_df.to_csv(out_path, index=False)
    print(f"Condition key written to: {out_path}")
    print(f"Total items: {len(key_df)}")
    print(key_df.head(10))

if __name__ == "__main__":
    main()
