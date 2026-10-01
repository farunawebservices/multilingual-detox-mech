#!/usr/bin/env python3
"""
Validate returned annotation CSVs, compute inter-annotator agreement,
merge with the internal condition key, and produce human-outcome tables.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).resolve().parents[1]
HUMAN_DIR = BASE / "results" / "human_evaluation" / "aya_seed42_confirmatory"
RETURNED_DIR = HUMAN_DIR / "returned"

RATER1 = RETURNED_DIR / "rater_1_blinded_annotation_sheet_annotated.csv"
RATER2 = RETURNED_DIR / "rater_2_blinded_annotation_sheet_annotated.csv"
CONDITION_KEY = HUMAN_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv"

REQUIRED_COLUMNS = [
    "annotation_order",
    "annotation_item_id",
    "language",
    "language_name",
    "source_text",
    "candidate_rewrite",
    "annotator_id",
    "non_toxic",
    "meaning_preservation_1_to_5",
    "fluency_1_to_5",
    "acceptable_detoxification",
    "language_preserved",
    "optional_comment",
]

NUMERIC_RATINGS = [
    "meaning_preservation_1_to_5",
    "fluency_1_to_5",
]

BINARY_RATINGS = [
    "non_toxic",
    "acceptable_detoxification",
    "language_preserved",
]


def validate_sheet(path: Path, expected_annotator: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert set(df.columns) == set(REQUIRED_COLUMNS), f"Column mismatch in {path.name}"
    assert df["annotator_id"].iloc[0] == expected_annotator, f"Annotator ID mismatch in {path.name}"
    assert df["annotation_item_id"].is_unique, f"Non-unique annotation_item_id in {path.name}"
    assert df["annotation_order"].is_monotonic_increasing, f"annotation_order not monotonic in {path.name}"
    for col in NUMERIC_RATINGS:
        vals = df[col]
        assert vals.between(1, 5).all(), f"{col} out of range in {path.name}"
        assert not vals.isna().any(), f"{col} has missing values in {path.name}"
    for col in BINARY_RATINGS:
        vals = df[col].astype(str).str.lower()
        assert vals.isin(["yes", "no"]).all(), f"{col} not yes/no in {path.name}"
        assert not vals.isna().any(), f"{col} has missing values in {path.name}"
    return df


def main():
    rater1 = validate_sheet(RATER1, "rater_1")
    rater2 = validate_sheet(RATER2, "rater_2")

    merged = pd.merge(
        rater1,
        rater2,
        on=["annotation_item_id", "language", "source_text", "candidate_rewrite"],
        suffixes=("_r1", "_r2"),
        how="inner",
    )

    agreement = {}
    for col in BINARY_RATINGS:
        c1 = merged[f"{col}_r1"].astype(str).str.lower()
        c2 = merged[f"{col}_r2"].astype(str).str.lower()
        agreement[col] = float(cohen_kappa_score(c1, c2))

    for col in NUMERIC_RATINGS:
        c1 = merged[f"{col}_r1"]
        c2 = merged[f"{col}_r2"]
        agreement[f"{col}_pearson"] = float(c1.corr(c2))

    key = pd.read_csv(CONDITION_KEY)
    human_with_conditions = pd.merge(
        merged,
        key,
        on="annotation_item_id",
        how="left",
    )

    human_with_conditions.to_csv(
        HUMAN_DIR / "human_ratings_unblinded.csv",
        index=False,
    )

    summary = {
        "n_items": len(merged),
        "agreement": agreement,
    }

    out_path = HUMAN_DIR / "human_evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Items validated: {len(merged)}")
    print(f"Agreement: {agreement}")
    print(f"Unblinded ratings written to: {HUMAN_DIR / 'human_ratings_unblinded.csv'}")
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()
