import pandas as pd
import os

in_path = "data/raw/naijaoffens/NaijaOffens/multilingual_hate_speech_dataset.csv"

if not os.path.exists(in_path):
    raise FileNotFoundError(f"{in_path} does not exist. Make sure the CSV is uploaded there.")

df = pd.read_csv(in_path)

print("Columns:", df.columns)
print("Sample rows:")
print(df.head())

# Based on your screenshot, columns are: 'class', 'text', 'language'[file:151]
CLASS_COL = "class"
TEXT_COL = "text"
LANG_COL = "language"

for col in [CLASS_COL, TEXT_COL, LANG_COL]:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Update CLASS_COL/TEXT_COL/LANG_COL in make_multilingual_hate_ha_yo.py."
        )

# Toxic classes: 1 (offensive) and 2 (hate). 0 is neutral.
TOXIC_CLASSES = [1, 2]

toxic_mask = df[CLASS_COL].isin(TOXIC_CLASSES)

# Filter by language (lowercase to be safe)
lang_series = df[LANG_COL].astype(str).str.lower()

df_ha = df[toxic_mask & (lang_series == "hausa")].copy()
df_yo = df[toxic_mask & (lang_series == "yoruba")].copy()

print("Hausa toxic rows:", len(df_ha))
print("Yoruba toxic rows:", len(df_yo))

os.makedirs("data/processed", exist_ok=True)

# Hausa toxic-only
df_ha_out = pd.DataFrame()
df_ha_out["language"] = "ha"
df_ha_out["toxic_input"] = df_ha[TEXT_COL]
df_ha_out["detox_output"] = ""        # to be filled by mt0-xl-detox-orpo later
df_ha_out["source"] = "MultilingualHateSpeechCSV"
df_ha_out["split"] = "ha_toxic_only"

# Yoruba toxic-only
df_yo_out = pd.DataFrame()
df_yo_out["language"] = "yo"
df_yo_out["toxic_input"] = df_yo[TEXT_COL]
df_yo_out["detox_output"] = ""
df_yo_out["source"] = "MultilingualHateSpeechCSV"
df_yo_out["split"] = "yo_toxic_only"

df_ha_out.to_csv("data/processed/ha_toxic_only.csv", index=False)
df_yo_out.to_csv("data/processed/yo_toxic_only.csv", index=False)

print(f"Saved {len(df_ha_out)} Hausa toxic-only rows to data/processed/ha_toxic_only.csv")
print(f"Saved {len(df_yo_out)} Yoruba toxic-only rows to data/processed/yo_toxic_only.csv")
