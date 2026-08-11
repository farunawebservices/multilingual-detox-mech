import os
import pandas as pd

STA_THRESHOLD = 0.50
SIM_THRESHOLD = 0.30

def garbage_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 1.0
    alpha = sum(c.isalpha() for c in text)
    return 1.0 - (alpha / len(text))

def final_filter(in_path, out_path, lang_name):
    if not os.path.exists(in_path):
        print("Skipping " + lang_name + ": file " + in_path + " does not exist.")
        return None

    df = pd.read_csv(in_path)
    print("=== " + lang_name.upper() + ": starting " + str(len(df)) + " rows ===")

    df["garbage_ratio"] = df["toxic_input"].apply(garbage_ratio)
    mask_not_garbage = df["garbage_ratio"] < 0.5

    mask_sta = df["sta_score"] >= STA_THRESHOLD
    mask_sim = df["sim_score"] >= SIM_THRESHOLD

    mask = mask_not_garbage & mask_sta & mask_sim
    df_f = df[mask].copy().drop(columns=["garbage_ratio"])

    print(lang_name.upper() + " stats: "
          + "Garbage pass -> " + str(mask_not_garbage.sum()) + ", "
          + "STA pass (>= " + str(STA_THRESHOLD) + ") -> " + str(mask_sta.sum()) + ", "
          + "SIM pass (>= " + str(SIM_THRESHOLD) + ") -> " + str(mask_sim.sum()) + ", "
          + "Final retained -> " + str(len(df_f)))

    df_f.to_csv(out_path, index=False)
    print("Saved final cleaned dataset to " + out_path + "\n")
    return df_f

df_ha = final_filter(
    "data/processed/ha_detox_pseudo_llama_scored.csv",
    "data/processed/ha_detox_final_clean.csv",
    "ha"
)

df_yo = final_filter(
    "data/processed/yo_detox_pseudo_llama_scored.csv",
    "data/processed/yo_detox_final_clean.csv",
    "yo"
)
