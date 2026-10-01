#!/usr/bin/env python3
"""
Generate human-evaluation tables and figures for the Aya confirmatory run.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
HUMAN_DIR = BASE / "results" / "human_evaluation" / "aya_seed42_confirmatory"
FIGURES_DIR = BASE / "figures"
TABLES_DIR = BASE / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

HUMAN_CSV = HUMAN_DIR / "human_ratings_unblinded.csv"
SUMMARY_JSON = HUMAN_DIR / "human_evaluation_summary.json"


def main():
    df = pd.read_csv(HUMAN_CSV)
    with open(SUMMARY_JSON, "r") as f:
        summary = json.load(f)

    agreement = summary["agreement"]

    # Table 1: Agreement
    agreement_rows = []
    for metric, value in agreement.items():
        agreement_rows.append({"metric": metric, "value": round(value, 3)})
    agreement_table = pd.DataFrame(agreement_rows)
    agreement_table.to_csv(TABLES_DIR / "human_agreement_aya_seed42.csv", index=False)

    # Table 2: Human outcomes by condition
    # Use language_x as the language column (from rater1 side)
    cond_cols = ["condition_name", "layer", "alpha", "language_x"]
    out_cols = [
        "non_toxic_r1",
        "non_toxic_r2",
        "meaning_preservation_1_to_5_r1",
        "meaning_preservation_1_to_5_r2",
        "fluency_1_to_5_r1",
        "fluency_1_to_5_r2",
        "acceptable_detoxification_r1",
        "acceptable_detoxification_r2",
        "language_preserved_r1",
        "language_preserved_r2",
    ]

    grouped = df.groupby(cond_cols, dropna=False)
    human_summary_rows = []

    for name, group in grouped:
        row = {
            "condition_name": name[0],
            "layer": name[1],
            "alpha": name[2],
            "language": name[3],
        }
        row["n"] = len(group)
        for col in out_cols:
            if col.startswith("non_toxic") or col.startswith("acceptable_detoxification") or col.startswith("language_preserved"):
                vals = group[col].astype(str).str.lower() == "yes"
                row[f"{col}_mean"] = float(vals.mean())
            else:
                row[f"{col}_mean"] = float(group[col].mean())
        human_summary_rows.append(row)

    human_summary = pd.DataFrame(human_summary_rows)
    human_summary.to_csv(TABLES_DIR / "human_outcomes_by_condition_aya_seed42.csv", index=False)

    # Figure: non_toxic by condition
    plt.figure(figsize=(8, 4))
    pivot = human_summary.pivot_table(
        index="condition_name",
        columns="language",
        values="non_toxic_r1_mean",
    )
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Non-toxic rate (Rater 1)")
    plt.title("Human non-toxicity by condition (Aya seed 42)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "human_non_toxic_by_condition_aya_seed42.png", dpi=200)
    plt.close()

    # Figure: meaning preservation by condition
    plt.figure(figsize=(8, 4))
    pivot_mp = human_summary.pivot_table(
        index="condition_name",
        columns="language",
        values="meaning_preservation_1_to_5_r1_mean",
    )
    pivot_mp.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Meaning preservation (1–5, Rater 1)")
    plt.title("Human meaning preservation by condition (Aya seed 42)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "human_meaning_preservation_by_condition_aya_seed42.png", dpi=200)
    plt.close()

    print("Tables written to:", TABLES_DIR)
    print("Figures written to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
