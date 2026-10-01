#!/usr/bin/env python3
"""
Recompute annotation_item_id (DET-...) for the internal condition key
using the same hashing logic as 37_make_blinded_annotation_sheet.py.
"""

import hashlib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
HUMAN_DIR = BASE / "results" / "human_evaluation" / "aya_seed42_confirmatory"

def blind_id(example_id: str, condition: str) -> str:
    digest = hashlib.sha256(
        f"aya-seed42-confirmatory|{example_id}|{condition}".encode("utf-8")
    ).hexdigest()[:12]
    return f"DET-{digest.upper()}"


def main():
    key = pd.read_csv(HUMAN_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv")

    # The current key has:
    # annotation_item_id = example_id (dev-...)
    # condition_name = condition string like dev_en_transfer_l16_a0.25
    # We need to recompute annotation_item_id using blind_id(example_id, condition_name)

    rows = []
    for _, row in key.iterrows():
        example_id = row["annotation_item_id"]  # currently dev-...
        condition = row["condition_name"]
        det_id = blind_id(example_id, condition)
        rows.append({
            "annotation_item_id": det_id,
            "example_id": example_id,
            "condition_name": condition,
            "layer": row["layer"],
            "alpha": row["alpha"],
            "language": row["language"],
            "language_name": row["language_name"],
            "split": row["split"],
        })

    fixed_key = pd.DataFrame(rows)
    out_path = HUMAN_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv"
    fixed_key.to_csv(out_path, index=False)
    print(f"Fixed key written to: {out_path}")
    print(f"Total items: {len(fixed_key)}")
    print(fixed_key.head(10))

if __name__ == "__main__":
    main()
