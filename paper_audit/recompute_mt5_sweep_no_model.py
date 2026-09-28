from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import importlib.util
import pandas as pd

pred_path = Path("paper_audit/generated_tables/mt5_steering_joined_predictions.csv")
out_dir = Path("paper_audit/generated_tables")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(pred_path, keep_default_na=False)

required = {
    "pair_id", "language", "toxic_input", "detox_output",
    "prediction", "steering_strength"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

def norm(s):
    return " ".join(str(s).strip().split())

def seq_ratio(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

df["is_copy"] = [
    norm(p) == norm(t) for p, t in zip(df["prediction"], df["toxic_input"])
]
df["is_empty"] = df["prediction"].map(norm).eq("")
df["output_chars"] = df["prediction"].str.len()
df["output_words"] = df["prediction"].str.split().str.len()
df["edit_similarity_to_input"] = [
    seq_ratio(p, t) for p, t in zip(df["prediction"], df["toxic_input"])
]
df["edit_similarity_to_reference"] = [
    seq_ratio(p, r) for p, r in zip(df["prediction"], df["detox_output"])
]

has_sacrebleu = importlib.util.find_spec("sacrebleu") is not None
if has_sacrebleu:
    import sacrebleu
    df["chrf_reference_overlap"] = [
        sacrebleu.corpus_chrf([str(p)], [[str(r)]]).score / 100.0
        for p, r in zip(df["prediction"], df["detox_output"])
    ]

rows = []
for strength, g in df.groupby("steering_strength", sort=True):
    preds = g["prediction"].map(norm).tolist()
    counts = Counter(preds)
    top_count = max(counts.values()) if counts else 0

    row = {
        "steering_strength": strength,
        "n_rows": len(g),
        "n_pairs": g["pair_id"].nunique(),
        "language_set": ";".join(sorted(g["language"].unique())),
        "copy_rate": g["is_copy"].mean(),
        "empty_rate": g["is_empty"].mean(),
        "mean_output_chars": g["output_chars"].mean(),
        "mean_output_words": g["output_words"].mean(),
        "unique_outputs": len(counts),
        "unique_output_rate": len(counts) / len(g),
        "top_output_count": top_count,
        "top_output_rate": top_count / len(g),
        "mean_edit_similarity_to_input": g["edit_similarity_to_input"].mean(),
        "mean_edit_similarity_to_reference": g["edit_similarity_to_reference"].mean(),
    }
    if has_sacrebleu:
        row["mean_chrf_reference_overlap"] = g["chrf_reference_overlap"].mean()
    rows.append(row)

summary = pd.DataFrame(rows).sort_values("steering_strength")
df.to_csv(out_dir / "mt5_steering_no_model_per_example.csv", index=False)
summary.to_csv(out_dir / "mt5_steering_no_model_summary.csv", index=False)

print("sacrebleu_available:", has_sacrebleu)
print("\n=== Amharic steering strength summary ===")
print(summary.to_string(index=False))
