#!/usr/bin/env python3
"""
Paired bootstrap analysis for the locked Aya-23-8B confirmatory steering run.

Computes paired condition differences with percentile 95% confidence intervals
over aligned frozen-test examples. The unit resampled is an example ID, preserving
within-example pairing across conditions.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


RUN_DIR = Path("artifacts/confirmatory_aya_2026_09_30/primary_seed42")
TEST_DIR = RUN_DIR / "test"
METRICS_DIR = RUN_DIR / "metrics"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 20260930

METRICS = [
    "chrf_to_reference",
    "labse_to_reference",
    "labse_to_input",
    "copy_input",
    "copy_reference",
    "empty_output",
    "output_chars",
    "output_words",
]

COMPARISONS = [
    ("english_transfer", "baseline"),
    ("within_language", "baseline"),
    ("random_direction", "baseline"),
    ("orthogonal_random_direction", "baseline"),
    ("negative_english_transfer", "baseline"),
    ("english_transfer", "random_direction"),
    ("english_transfer", "orthogonal_random_direction"),
    ("english_transfer", "negative_english_transfer"),
    ("english_transfer", "within_language"),
]


def load_condition_metrics() -> pd.DataFrame:
    frames = []
    for path in sorted(TEST_DIR.glob("*_metrics.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        if "condition" not in df.columns:
            raise ValueError(f"Missing condition column: {path}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No metric files found in {TEST_DIR}")

    all_metrics = pd.concat(frames, ignore_index=True)

    expected = {
        "baseline",
        "english_transfer",
        "within_language",
        "random_direction",
        "shuffled_pair",
        "negative_english_transfer",
        "orthogonal_random_direction",
    }
    observed = set(all_metrics["condition"].unique())
    missing = expected.difference(observed)
    if missing:
        raise ValueError(f"Missing conditions: {sorted(missing)}")

    return all_metrics


def paired_frame(
    dataframe: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    language: str | None,
) -> pd.DataFrame:
    df = dataframe.copy()
    if language is not None:
        df = df[df["language"] == language].copy()

    left = (
        df[df["condition"] == left_condition]
        .set_index("example_id")
        .sort_index()
    )
    right = (
        df[df["condition"] == right_condition]
        .set_index("example_id")
        .sort_index()
    )

    if set(left.index) != set(right.index):
        missing_left = sorted(set(right.index).difference(left.index))
        missing_right = sorted(set(left.index).difference(right.index))
        raise ValueError(
            f"Unaligned IDs for {left_condition} vs {right_condition}, "
            f"language={language}: "
            f"missing_left={len(missing_left)}, missing_right={len(missing_right)}"
        )

    return left.join(
        right[METRICS],
        how="inner",
        lsuffix="_left",
        rsuffix="_right",
    )


def bootstrap_difference(
    differences: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(differences)
    observed = float(np.mean(differences))

    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    samples = differences[indices].mean(axis=1)

    low, high = np.quantile(samples, [0.025, 0.975])
    return observed, float(low), float(high)


def analyze_comparison(
    dataframe: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    language: str | None,
    rng: np.random.Generator,
) -> list[dict]:
    paired = paired_frame(
        dataframe,
        left_condition=left_condition,
        right_condition=right_condition,
        language=language,
    )

    records = []
    for metric in METRICS:
        values = (
            paired[f"{metric}_left"].astype(float).to_numpy()
            - paired[f"{metric}_right"].astype(float).to_numpy()
        )
        effect, ci_low, ci_high = bootstrap_difference(values, rng)
        records.append(
            {
                "language": language if language is not None else "pooled",
                "comparison": f"{left_condition} - {right_condition}",
                "left_condition": left_condition,
                "right_condition": right_condition,
                "metric": metric,
                "n_pairs": len(values),
                "mean_paired_difference": effect,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            }
        )
    return records


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    metrics = load_condition_metrics()

    records = []
    for language in ["xh", "yo", None]:
        for left, right in COMPARISONS:
            records.extend(
                analyze_comparison(
                    metrics,
                    left_condition=left,
                    right_condition=right,
                    language=language,
                    rng=rng,
                )
            )

    result = pd.DataFrame(records)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = METRICS_DIR / "paired_bootstrap_results.csv"
    result.to_csv(output_csv, index=False, encoding="utf-8")

    manifest = {
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "unit": "aligned frozen-test example_id",
        "languages": ["xh", "yo", "pooled"],
        "conditions": sorted(metrics["condition"].unique().tolist()),
        "metrics": METRICS,
        "comparisons": [
            f"{left} - {right}" for left, right in COMPARISONS
        ],
        "output_csv": str(output_csv),
    }
    manifest_path = METRICS_DIR / "paired_bootstrap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output_csv}")
    print(f"Saved: {manifest_path}")

    focus = result[
        result["comparison"].isin(
            [
                "english_transfer - baseline",
                "english_transfer - orthogonal_random_direction",
                "english_transfer - negative_english_transfer",
            ]
        )
        & result["metric"].isin(
            [
                "chrf_to_reference",
                "labse_to_reference",
                "labse_to_input",
                "copy_input",
            ]
        )
    ]
    print("\nKey paired results:")
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
