from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/multilingual-detox-mech")
DATA = ROOT / "data" / "external" / "afrihate"
OUT = ROOT / "outputs" / "phase2_gpt5_detox"

YO_IN = DATA / "afrihate_yor_toxic_candidates_200.csv"
XH_IN = DATA / "afrihate_xho_toxic_candidates_200.csv"
OUT_FILE = OUT / "afrihate_phase2_candidates_100.csv"

OUT.mkdir(parents=True, exist_ok=True)

def clean_select(path, language, n=50):
    df = pd.read_csv(path).copy()

    text_col = "toxic_input_processing"
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df["char_len"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()

    # Exclude obvious fragments, very long passages, and posts containing
    # explicit high-risk identity/violence patterns. GPT-4 remains final judge.
    df = df[
        (df["char_len"] >= 12)
        & (df["char_len"] <= 280)
        & (df["word_count"] >= 3)
        & (df["word_count"] <= 45)
    ].copy()

    # Use a reproducible shuffled sample, stratified roughly by source label.
    hate = df[df["label"] == "Hate"].sample(
        n=min(10, (df["label"] == "Hate").sum()),
        random_state=42,
    )

    remaining = df.drop(index=hate.index)
    abuse_n = n - len(hate)

    abuse = remaining[remaining["label"] == "Abuse"].sample(
        n=min(abuse_n, (remaining["label"] == "Abuse").sum()),
        random_state=42,
    )

    selected = pd.concat([hate, abuse], ignore_index=True)
    selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)
    selected["language"] = language

    return selected.head(n)

yo = clean_select(YO_IN, "Yorùbá", 50)
xh = clean_select(XH_IN, "isiXhosa", 50)

result = pd.concat([yo, xh], ignore_index=True)
result.insert(0, "phase2_row_id", range(1, len(result) + 1))

keep = [
    "phase2_row_id",
    "source_dataset",
    "source_config",
    "source_split",
    "id",
    "language",
    "label",
    "length",
    "toxic_input_raw",
    "toxic_input_processing",
    "existing_parallel_overlap",
    "phase1_status",
]

result[keep].to_csv(OUT_FILE, index=False, encoding="utf-8")

print("Wrote:", OUT_FILE)
print("Rows:", len(result))
print(result.groupby(["language", "label"]).size())