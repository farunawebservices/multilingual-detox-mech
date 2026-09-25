#!/usr/bin/env python3
"""
Stage 27: Generate final cross-model comparison table.

Includes:
- mT5 v2 (400+ pairs, 3 seeds)
- mT0 (400+ pairs, 3 seeds)
- Llama-3-8B QLoRA (3 seeds)
- Qwen2.5-7B QLoRA (3 seeds)

Metrics: copy rate, SIM, FL, J score (all mean ± std across seeds).
"""

import json
from pathlib import Path

import pandas as pd


TABLES_ROOT = Path("results/tables")


def load_existing_summary():
    path = TABLES_ROOT / "table_summary_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def add_qwen_row(df: pd.DataFrame) -> pd.DataFrame:
    # Qwen aggregate from behavioral evaluation
    # Use the 3-seed aggregate from your Qwen evaluation (copy, SIM, FL, J)
    # If you have a dedicated aggregate file, load it; otherwise hard-code from prior output.
    qwen_row = {
        "Model": "Qwen2.5-7B (400+)",
        "Copy rate": 0.0338,
        "SIM": 0.7934,
        "FL": 0.5807,
        "J score": 0.7913,
        "Architecture": "Decoder-only",
        "Corpus size": "400+ pairs",
        "Seeds": 3,
    }
    # Reorder columns to match existing table
    cols = df.columns.tolist()
    return pd.concat([df, pd.DataFrame([qwen_row])[cols]], ignore_index=True)


def main():
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_existing_summary()
    print("Existing models:")
    print(df.to_string(index=False))

    df2 = add_qwen_row(df)
    print("\nWith Qwen:")
    print(df2.to_string(index=False))

    out_path = TABLES_ROOT / "table_final_cross_model.csv"
    df2.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
