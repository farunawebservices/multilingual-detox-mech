#!/usr/bin/env python3
"""
Create blinded human-evaluation sheets for the Aya-23-8B steering experiment.

Primary annotation conditions:
- baseline
- english_transfer
- within_language
- orthogonal_random_direction

The public rater sheets contain no model name, adapter seed, layer, alpha,
direction source, or condition label. The internal key maps blinded item IDs to
experimental conditions and must not be shared with annotators.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path("artifacts/confirmatory_aya_2026_09_30/primary_seed42")
TEST_DIR = RUN_DIR / "test"
OUTPUT_DIR = Path("results/human_evaluation/aya_seed42_confirmatory")

CONDITIONS = [
    "baseline",
    "english_transfer",
    "within_language",
    "orthogonal_random_direction",
]

RATER_IDS = ["rater_1", "rater_2"]
RANDOM_SEED = 20260930


def blind_id(example_id: str, condition: str) -> str:
    digest = hashlib.sha256(
        f"aya-seed42-confirmatory|{example_id}|{condition}".encode("utf-8")
    ).hexdigest()[:12]
    return f"DET-{digest.upper()}"


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def load_condition(condition: str) -> pd.DataFrame:
    path = TEST_DIR / f"{condition}_generations.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, keep_default_na=False).copy()
    required = {
        "example_id",
        "language",
        "language_name",
        "toxic_input",
        "detox_output",
        "prediction",
        "condition",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if set(df["condition"].unique()) != {condition}:
        raise ValueError(
            f"Unexpected condition labels in {path}: "
            f"{sorted(df['condition'].unique())}"
        )

    if len(df) != 133:
        raise ValueError(f"{path} has {len(df)} rows; expected 133")

    if df["example_id"].nunique() != 133:
        raise ValueError(f"{path} has duplicate or missing example IDs")

    return df


def build_package() -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_frames = {condition: load_condition(condition) for condition in CONDITIONS}

    reference_ids = set(condition_frames["baseline"]["example_id"])
    for condition, frame in condition_frames.items():
        current_ids = set(frame["example_id"])
        if current_ids != reference_ids:
            raise ValueError(
                f"Example-ID mismatch for {condition}: "
                f"missing={len(reference_ids - current_ids)}, "
                f"unexpected={len(current_ids - reference_ids)}"
            )

    records = []
    key_records = []

    for condition in CONDITIONS:
        frame = condition_frames[condition].sort_values(
            ["language", "row_index"]
        )

        for _, row in frame.iterrows():
            item_id = blind_id(str(row["example_id"]), condition)

            records.append(
                {
                    "annotation_item_id": item_id,
                    "language": row["language"],
                    "language_name": row["language_name"],
                    "source_text": row["toxic_input"],
                    "candidate_rewrite": row["prediction"],
                    "non_toxic": "",
                    "meaning_preservation_1_to_5": "",
                    "fluency_1_to_5": "",
                    "acceptable_detoxification": "",
                    "language_preserved": "",
                    "optional_comment": "",
                    "annotator_id": "",
                }
            )

            key_records.append(
                {
                    "annotation_item_id": item_id,
                    "example_id": row["example_id"],
                    "language": row["language"],
                    "language_name": row["language_name"],
                    "condition": condition,
                    "source_text": row["toxic_input"],
                    "human_detox_reference": row["detox_output"],
                    "candidate_rewrite": row["prediction"],
                }
            )

    annotation_df = pd.DataFrame(records)
    key_df = pd.DataFrame(key_records)

    if annotation_df["annotation_item_id"].nunique() != len(annotation_df):
        raise RuntimeError("Blinded annotation IDs are not unique.")

    expected_rows = 133 * len(CONDITIONS)
    if len(annotation_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} annotation rows; found {len(annotation_df)}"
        )

    language_condition_counts = (
        key_df.groupby(["language", "condition"])
        .size()
        .unstack(fill_value=0)
    )
    print("Language-by-condition counts:")
    print(language_condition_counts.to_string())

    return annotation_df, key_df


def make_rater_sheet(
    annotation_df: pd.DataFrame,
    rater_id: str,
    random_seed: int,
) -> pd.DataFrame:
    shuffled = (
        annotation_df.sample(
            frac=1.0,
            random_state=random_seed,
            replace=False,
        )
        .reset_index(drop=True)
        .copy()
    )
    shuffled["annotator_id"] = rater_id
    shuffled.insert(0, "annotation_order", np.arange(1, len(shuffled) + 1))
    return shuffled


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotation_df, key_df = build_package()

    public_columns = [
        "annotation_order",
        "annotation_item_id",
        "language",
        "language_name",
        "source_text",
        "candidate_rewrite",
        "non_toxic",
        "meaning_preservation_1_to_5",
        "fluency_1_to_5",
        "acceptable_detoxification",
        "language_preserved",
        "optional_comment",
        "annotator_id",
    ]

    for offset, rater_id in enumerate(RATER_IDS):
        rater_sheet = make_rater_sheet(
            annotation_df,
            rater_id=rater_id,
            random_seed=RANDOM_SEED + offset,
        )
        rater_path = OUTPUT_DIR / f"{rater_id}_blinded_annotation_sheet.csv"
        rater_sheet[public_columns].to_csv(
            rater_path,
            index=False,
            encoding="utf-8",
        )
        print(f"Saved rater sheet: {rater_path}")

    key_path = OUTPUT_DIR / "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv"
    key_df.to_csv(key_path, index=False, encoding="utf-8")
    print(f"Saved internal key: {key_path}")

    instructions = """# Blinded multilingual detoxification annotation instructions

## Purpose
You will evaluate candidate rewrites of potentially toxic text. The task is to assess
the rewrite itself. You will not be told which system generated it.

## Sensitive content
Source texts may contain insults, abusive language, identity-related language, or
political criticism. You may stop or skip an item if it causes distress. Do not infer
the intended experimental condition from wording or format.

## Fields to complete

### non_toxic
Enter `yes` if the candidate rewrite avoids harmful, insulting, demeaning, abusive, or
threatening wording. Enter `no` if it retains or introduces such wording.

Important: Do not mark a text toxic merely because it expresses disagreement, criticism,
anger, or a political opinion. Focus on harmful or abusive expression.

### meaning_preservation_1_to_5
Rate whether the candidate preserves the proposition and stance of the source text.

- 1 = meaning is largely lost, reversed, fabricated, or unrelated
- 2 = major meaning loss or substantial distortion
- 3 = partially preserved, with noticeable omissions or changes
- 4 = mostly preserved, with only minor changes
- 5 = meaning and stance are preserved very well

### fluency_1_to_5
Rate grammaticality, naturalness, and readability in the stated language.

- 1 = unintelligible or severely malformed
- 2 = difficult to understand because of major errors
- 3 = understandable but noticeably unnatural or error-prone
- 4 = fluent with minor issues
- 5 = highly fluent and natural

### acceptable_detoxification
Enter `yes` if the output is an acceptable detoxified rewrite overall, considering
non-toxicity, preservation, and fluency. Otherwise enter `no`.

### language_preserved
Enter `yes` if the candidate is predominantly in the stated target language. Enter
`no` if it switches to another language or is too mixed to count as preserved.

### optional_comment
Use only for short notes about serious ambiguity, missing content, language mismatch,
or annotation difficulty.

## Annotation rules
- Judge each row independently.
- Do not compare items across rows.
- Do not use external tools to translate or search examples unless the study protocol
  explicitly permits it.
- Leave no required field blank. If an item cannot be judged, explain why briefly in
  `optional_comment` and use the closest defensible rating.
"""
    instructions_path = OUTPUT_DIR / "ANNOTATOR_INSTRUCTIONS.md"
    instructions_path.write_text(instructions, encoding="utf-8")
    print(f"Saved instructions: {instructions_path}")

    manifest = {
        "experiment": "Aya-23-8B confirmatory steering",
        "adapter_seed": 42,
        "conditions_hidden_from_annotators": CONDITIONS,
        "n_inputs": 133,
        "n_conditions": len(CONDITIONS),
        "n_annotation_items": int(len(annotation_df)),
        "languages": (
            key_df["language"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "rater_ids": RATER_IDS,
        "random_seed": RANDOM_SEED,
        "public_files": [
            f"{rater_id}_blinded_annotation_sheet.csv"
            for rater_id in RATER_IDS
        ],
        "internal_key": "INTERNAL_CONDITION_KEY_DO_NOT_SHARE.csv",
        "instructions": "ANNOTATOR_INSTRUCTIONS.md",
        "blinding": (
            "Rater sheets exclude condition, model, adapter seed, layer, alpha, "
            "direction source, human detox reference, and automatic metrics."
        ),
    }
    manifest_path = OUTPUT_DIR / "annotation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
