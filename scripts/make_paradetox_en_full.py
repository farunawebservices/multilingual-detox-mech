from datasets import load_dataset
import pandas as pd
import os

# Load ParaDetox EN from HF
ds = load_dataset("s-nlp/paradetox")  # ACL 2022 ParaDetox English[web:6]

print(ds)

# Commonly the split is "train" or "all"; inspect keys:
split_name = "train" if "train" in ds else list(ds.keys())[0]
dsplit = ds[split_name]
print("Split:", split_name, "rows:", dsplit.num_rows)

df = dsplit.to_pandas()
print("Columns:", df.columns)

# Use the actual column names from ParaDetox EN[web:6]
TOXIC_COL = "en_toxic_comment"
DETOX_COL = "en_neutral_comment"

df["language"] = "en"
df["toxic_input"] = df[TOXIC_COL]
df["detox_output"] = df[DETOX_COL]
df["source"] = "ParaDetoxEN"
df["split"] = split_name

os.makedirs("data/processed", exist_ok=True)
out_path = "data/processed/en_detox_full.csv"
df[["language", "toxic_input", "detox_output", "source", "split"]].to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
