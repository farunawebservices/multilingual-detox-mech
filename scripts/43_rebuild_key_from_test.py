#!/usr/bin/env python3
"""
Rebuild the internal condition key from test-set generations so that
annotation_item_id (DET-...) matches the blinded annotation sheets.
"""

import hashlib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TEST_DIR = BASE / "artifacts" / "confirmatory_aya_2026_09_30" / "primary_seed42" / "test"
HUMAN_DIR = BASE / "results" / "human_evaluation" / "aya_seed42_confirmatory"

def blind_id(example_id: str, condition: str) -> str:
    digest = hashlib.sha256(
        f"aya-seed42-confirmatory|{example_id}|{condition}".encode("utf-8")
    ).hexdigest()[:12]
    return f"DET-{digest.upper()}"


def main():
    rows = []

    for csv_path in TEST_DIR.glob("*_generations.csv"):
        df = pd.read_csv(csv_path, keep_default_na=False)
        for _, row in df.iterrows():
            example_id = row["example_id"]
            condition = row["condition"]
            det_id = blind_id(example_id, condition)
            rows.append({
                "annotation_item_id": det_id,
                "example_id": example_id,
                "condition_name": condition,
                "layer": int(row["layer"]),
                "alpha": float(row["alpha"]),
                "language": row["language"],
                "language_name": row["language_name"],
                "split": "test",
            })

    key_df = pd.DataFrame(rows)
    out_path = HUMAN_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv"
    key_df.to_csv(out_path, index=False)
    print(f"Key written to: {out_path}")
    print(f"Total items: {len(key_df)}")
    print(key_df.head(10))

if __name__ == "__main__":
    main()
