from datasets import load_dataset
import pandas as pd
import os

# Load dataset from Hugging Face Hub
ds = load_dataset("textdetox/multilingual_paradetox")

print(ds)

# Each key in ds is already a language split: 'en', 'ru', 'uk', 'de', 'es', 'am', 'zh', 'ar', 'hi'
# Map those split names to ISO codes and choose which ones we care about
SPLIT_TO_ISO = {
    "en": "en",
    "de": "de",
    "es": "es",
    "uk": "uk",
    "hi": "hi",
    "ar": "ar",
    "am": "am",
    # Add "zh": "zh" or other splits if you want them too
}

os.makedirs("data/processed", exist_ok=True)

for split_name, iso_code in SPLIT_TO_ISO.items():
    if split_name not in ds:
        print(f"[WARN] split '{split_name}' not found in dataset, skipping.")
        continue

    dsplit = ds[split_name]
    print(f"Processing split '{split_name}' with {dsplit.num_rows} rows")

    df = dsplit.to_pandas()
    print("Columns for", split_name, ":", df.columns)

    # According to your log, these are the columns: 'toxic_sentence', 'neutral_sentence'
    TOXIC_COL = "toxic_sentence"
    DETOX_COL = "neutral_sentence"

    df["language"] = iso_code
    df["toxic_input"] = df[TOXIC_COL]
    df["detox_output"] = df[DETOX_COL]
    df["source"] = "MultiParaDetox"
    df["split"] = split_name  # here split_name is actually the language; we'll refine train/dev later

    out_path = f"data/processed/{iso_code}_detox.csv"
    df[["language", "toxic_input", "detox_output", "source", "split"]].to_csv(
        out_path, index=False
    )
    print(f"Saved {len(df)} rows for language={iso_code} -> {out_path}")
