#!/usr/bin/env python3
"""
Stage 26: Sample outputs for human evaluation.

Protocol:
- For each model family (mT5, mT0, Llama, Qwen), sample 50 examples from test predictions.
- Stratify by language (≈5–6 per language).
- For each example, store: toxic input, reference detox, model prediction, model name, seed.
- Output a single CSV for annotation and a compact JSON manifest.
"""

import json
import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 0
SAMPLE_PER_MODEL = 50
TEST_ROOT = Path("results/training_v2_400plus/seed_2024")
OUTPUT_DIR = Path("results/human_evaluation")


def load_test():
    return pd.read_csv(TEST_ROOT / "test.csv", keep_default_na=False)


def load_predictions(model_name, seed, path: Path):
    df = pd.read_csv(path, keep_default_na=False)
    df["model"] = model_name
    df["seed"] = seed
    return df


def stratified_sample(df: pd.DataFrame, n: int, rng: random.Random):
    # Sample roughly equally per language
    langs = df["language"].unique()
    per_lang = max(1, n // len(langs))
    rows = []
    for lang in langs:
        subset = df[df["language"] == lang]
        sample_n = min(per_lang, len(subset))
        sampled = subset.sample(n=sample_n, random_state=rng.randint(0, 2**31 - 1))
        rows.append(sampled)
    sampled_df = pd.concat(rows, ignore_index=True)
    if len(sampled_df) < n:
        # Fill remainder randomly
        remainder = n - len(sampled_df)
        extra = df.sample(n=remainder, random_state=rng.randint(0, 2**31 - 1))
        sampled_df = pd.concat([sampled_df, extra], ignore_index=True)
    return sampled_df.head(n)


def main():
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = load_test()

    # Define prediction files to sample from
    sources = [
        # mT5 v2
        ("mt5_v2", 2024, Path("results/evaluation/mt5_v2_400plus/predictions_seed_2024.csv")),
        ("mt5_v2", 42, Path("results/evaluation/mt5_v2_400plus/predictions_seed_42.csv")),
        ("mt5_v2", 1337, Path("results/evaluation/mt5_v2_400plus/predictions_seed_1337.csv")),
        # mT0
        ("mt0", 2024, Path("results/evaluation/mt0/predictions_seed_2024.csv")),
        ("mt0", 42, Path("results/evaluation/mt0/predictions_seed_42.csv")),
        ("mt0", 1337, Path("results/evaluation/mt0/predictions_seed_1337.csv")),
        # Llama
        ("llama3_8b", 2024, Path("results/evaluation/llama8b_qlora/seed_2024_predictions.csv")),
        ("llama3_8b", 42, Path("results/evaluation/llama8b_qlora/seed_42_predictions.csv")),
        ("llama3_8b", 1337, Path("results/evaluation/llama8b_qlora/seed_1337_predictions.csv")),
        # Qwen
        ("qwen2_5_7b", 42, Path("results/evaluation/qwen2_5_7b_qlora/seed_42/predictions_raw.csv")),
    ]

    all_rows = []
    manifest = {"models": {}, "n_examples_per_model": SAMPLE_PER_MODEL, "random_seed": RANDOM_SEED}

    for model_name, seed, path in sources:
        if not path.exists():
            print("Skipping missing file:", path)
            continue

        preds = load_predictions(model_name, seed, path)
        # Ensure columns match
        if "prediction" not in preds.columns:
            # Qwen uses 'prediction', others may use different names; unify here if needed
            raise ValueError(f"Missing 'prediction' column in {path}")

        sampled = stratified_sample(preds, SAMPLE_PER_MODEL, rng)
        for _, row in sampled.iterrows():
            all_rows.append(
                {
                    "sample_id": len(all_rows),
                    "language": row["language"],
                    "toxic_input": row["toxic_input"],
                    "reference_detox": row["detox_output"],
                    "model_prediction": row["prediction"],
                    "model": model_name,
                    "seed": seed,
                }
            )

        key = f"{model_name}_seed{seed}"
        manifest["models"][key] = {
            "model": model_name,
            "seed": seed,
            "n_sampled": len(sampled),
            "languages": sorted(sampled["language"].unique().tolist()),
        }

    samples_df = pd.DataFrame(all_rows)
    samples_path = OUTPUT_DIR / "human_eval_samples.csv"
    samples_df.to_csv(samples_path, index=False, encoding="utf-8")

    manifest_path = OUTPUT_DIR / "human_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Sampled", len(samples_df), "examples for human evaluation")
    print("Saved:", samples_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
