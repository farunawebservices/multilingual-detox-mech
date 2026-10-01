#!/usr/bin/env python3
"""
Merge automatic metrics with human evaluation for the Aya confirmatory run.
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
AUTO_DEV_METRICS = BASE / "artifacts" / "confirmatory_aya_2026_09_30" / "primary_seed42" / "dev" / "all_dev_metrics.csv"
HUMAN_OUTCOMES = BASE / "tables" / "human_outcomes_by_condition_aya_seed42.csv"
TABLES_DIR = BASE / "tables"


def main():
    auto_df = pd.read_csv(AUTO_DEV_METRICS)
    human_df = pd.read_csv(HUMAN_OUTCOMES)

    # Rename condition -> condition_name in auto_df to match human_df
    auto_df = auto_df.rename(columns={"condition": "condition_name"})

    key_cols = ["condition_name", "layer", "alpha", "language"]

    merged = pd.merge(
        auto_df,
        human_df,
        on=key_cols,
        how="inner",
        suffixes=("_auto", "_human"),
    )

    merged.to_csv(TABLES_DIR / "combined_auto_human_aya_seed42.csv", index=False)
    print("Combined table written to:", TABLES_DIR / "combined_auto_human_aya_seed42.csv")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    main()
