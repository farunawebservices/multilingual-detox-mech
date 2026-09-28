from pathlib import Path
import pandas as pd
import numpy as np

pred_path = Path("results/mechanistic/steering/steering_sweep_results.csv")
test_path = Path("results/training_v2_400plus/seed_2024/test.csv")
out_dir = Path("paper_audit/generated_tables")
out_dir.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(pred_path, keep_default_na=False)
test = pd.read_csv(test_path, keep_default_na=False)

print("prediction rows:", len(pred))
print("test rows:", len(test))
print("test columns:", test.columns.tolist())

key = ["toxic_input", "detox_output"]
missing_pred = set(key) - set(pred.columns)
missing_test = set(key + ["language"]) - set(test.columns)
if missing_pred:
    raise ValueError(f"Sweep file lacks join columns: {sorted(missing_pred)}")
if missing_test:
    raise ValueError(f"Test file lacks needed columns: {sorted(missing_test)}")

test_lookup = test[key + ["language"]].copy()
duplicated_pairs = test_lookup.duplicated(key, keep=False)
if duplicated_pairs.any():
    n = int(duplicated_pairs.sum())
    raise ValueError(
        f"Test split has {n} rows with non-unique toxic/detox text-pair keys; "
        "do not join by text without an explicit disambiguation audit."
    )

pred = pred.merge(
    test_lookup,
    on=key,
    how="left",
    validate="many_to_one",
)

if pred["language"].isna().any():
    unmatched = pred.loc[pred["language"].isna(), key].drop_duplicates()
    unmatched.to_csv(out_dir / "mt5_steering_unmatched_pairs.csv", index=False)
    raise ValueError(
        f"{len(unmatched)} unique sweep text pairs did not match the frozen test CSV. "
        "See paper_audit/generated_tables/mt5_steering_unmatched_pairs.csv"
    )

if pred.duplicated(["pair_id", "steering_strength"]).any():
    raise ValueError("Duplicate pair_id/strength rows in sweep.")

pred.to_csv(out_dir / "mt5_steering_joined_predictions.csv", index=False)

print("\nLanguage counts among 50 sweep pairs:")
print(
    pred.drop_duplicates("pair_id")["language"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nStrength counts:")
print(pred["steering_strength"].value_counts().sort_index().to_string())

try:
    import sacrebleu
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("\nMetric dependencies unavailable:", repr(e))
    print("Joined prediction artifact was still saved successfully.")
    raise SystemExit(0)

embedder = SentenceTransformer("sentence-transformers/LaBSE")
src_emb = embedder.encode(
    pred["toxic_input"].tolist(),
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True,
)
out_emb = embedder.encode(
    pred["prediction"].tolist(),
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True,
)

pred["sim"] = np.sum(src_emb * out_emb, axis=1)
pred["chrf_reference_overlap"] = [
    sacrebleu.corpus_chrf([str(h)], [[str(r)]]).score / 100.0
    for h, r in zip(pred["prediction"], pred["detox_output"])
]
pred["is_copy"] = (
    pred["prediction"].str.strip() == pred["toxic_input"].str.strip()
)
pred["is_empty"] = pred["prediction"].str.strip().eq("")
pred["output_chars"] = pred["prediction"].str.len()
pred["output_words"] = pred["prediction"].str.split().str.len()

summary = (
    pred.groupby("steering_strength", as_index=False)
    .agg(
        n_rows=("pair_id", "size"),
        n_pairs=("pair_id", "nunique"),
        copy_rate=("is_copy", "mean"),
        sim=("sim", "mean"),
        chrf_reference_overlap=("chrf_reference_overlap", "mean"),
        empty_rate=("is_empty", "mean"),
        mean_output_chars=("output_chars", "mean"),
        mean_output_words=("output_words", "mean"),
        unique_outputs=("prediction", "nunique"),
    )
    .sort_values("steering_strength")
)

summary["unique_output_rate"] = (
    summary["unique_outputs"] / summary["n_pairs"]
)

language_summary = (
    pred.groupby(["steering_strength", "language"], as_index=False)
    .agg(
        n=("pair_id", "size"),
        copy_rate=("is_copy", "mean"),
        sim=("sim", "mean"),
        chrf_reference_overlap=("chrf_reference_overlap", "mean"),
        empty_rate=("is_empty", "mean"),
        unique_outputs=("prediction", "nunique"),
    )
    .sort_values(["steering_strength", "language"])
)

pred.to_csv(out_dir / "mt5_steering_recomputed_per_example.csv", index=False)
summary.to_csv(out_dir / "mt5_steering_recomputed_summary.csv", index=False)
language_summary.to_csv(
    out_dir / "mt5_steering_recomputed_by_language.csv",
    index=False,
)

print("\n=== Overall strength summary ===")
print(summary.to_string(index=False))

print("\n=== Per-language summary ===")
print(language_summary.to_string(index=False))
