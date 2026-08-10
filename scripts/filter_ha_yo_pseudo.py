import os
import re
import pandas as pd

IN_HA = "data/processed/ha_detox_pseudo_llama.csv"
IN_YO = "data/processed/yo_detox_pseudo_llama.csv"
OUT_HA = "data/processed/ha_detox_pseudo_llama_filtered.csv"
OUT_YO = "data/processed/yo_detox_pseudo_llama_filtered.csv"

for path in [IN_HA, IN_YO]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; generate pseudo data first.")

# Common English function/meta words that indicate leaked English,
# not just Latin-script Hausa/Yoruba text.
ENGLISH_MARKERS = re.compile(
    r"\b(the|and|is|are|here|please|sorry|i'm|i am|happy|help|rewrite|"
    r"rewritten|version|sentence|neutral|assistant|note|explanation)\b",
    re.IGNORECASE
)

def tok_len(s):
    return len(str(s).strip().split())

def filter_df(df, lang_name):
    print(f"=== {lang_name}: starting {len(df)} rows ===")

    df = df.dropna(subset=["toxic_input", "detox_output"]).copy()

    df["len_toxic"] = df["toxic_input"].apply(tok_len)
    df["len_detox"] = df["detox_output"].apply(tok_len)

    mask_len = (df["len_detox"] >= 3) & (df["len_detox"] <= 50)
    mask_diff = df["toxic_input"].astype(str).str.strip() != df["detox_output"].astype(str).str.strip()
    mask_no_english = ~df["detox_output"].astype(str).str.contains(ENGLISH_MARKERS)

    mask = mask_len & mask_diff & mask_no_english
    df_f = df[mask].copy()
    print(f"{lang_name}: after length + diff + no-English-markers -> {len(df_f)} rows")

    before = len(df_f)
    df_f = df_f.drop_duplicates(subset=["detox_output"])
    print(f"{lang_name}: after dedup detox_output -> {len(df_f)} rows (dropped {before - len(df_f)})")

    df_f = df_f.drop(columns=["len_toxic", "len_detox"])
    return df_f

df_ha = pd.read_csv(IN_HA)
df_yo = pd.read_csv(IN_YO)

df_ha_f = filter_df(df_ha, "ha")
df_yo_f = filter_df(df_yo, "yo")

os.makedirs("data/processed", exist_ok=True)
df_ha_f.to_csv(OUT_HA, index=False)
df_yo_f.to_csv(OUT_YO, index=False)

print("Wrote filtered Hausa to", OUT_HA)
print("Wrote filtered Yoruba to", OUT_YO)
