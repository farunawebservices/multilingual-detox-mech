from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/multilingual-detox-mech")
INFILE = ROOT / "outputs" / "phase2_gpt5_detox" / "afrihate_phase2_candidates_filtered.csv"
OUTFILE = ROOT / "outputs" / "phase2_gpt5_detox" / "afrihate_phase2_manual_shortlist_30.csv"

YORUBA_IDS = {
    "train_yoruba_01754",
    "train_yoruba_03122",
    "train_yoruba_03325",
    "train_yoruba_03069",
    "test_yoruba_00244",
    "train_yoruba_00191",
    "test_yoruba_00230",
    "train_yoruba_02374",
    "train_yoruba_00838",
    "train_yoruba_01135",
    "dev_yoruba_00156",
}

XHOSA_IDS = {
    "train_isixhosa_02411",
    "train_isixhosa_02047",
    "dev_isixhosa_00313",
    "test_isixhosa_00301",
    "train_isixhosa_02225",
    "train_isixhosa_01206",
    "train_isixhosa_00712",
    "train_isixhosa_01028",
    "train_isixhosa_01082",
    "train_isixhosa_00278",
}

df = pd.read_csv(INFILE)

selected = df[
    df["id"].isin(YORUBA_IDS | XHOSA_IDS)
].copy()

selected = selected.sort_values(
    ["language", "id"]
).reset_index(drop=True)

selected.insert(0, "manual_shortlist_id", range(1, len(selected) + 1))
selected["manual_selection_reason"] = (
    "Clear candidate insult/profanity; excluded obvious identity, "
    "sexual, violent, long, and English-dominant cases"
)
selected["native_precheck_decision"] = ""
selected["native_precheck_comments"] = ""

selected.to_csv(OUTFILE, index=False, encoding="utf-8")

print("Rows selected:", len(selected))
print(selected.groupby("language").size())
print("Output:", OUTFILE)
print()
print(selected[
    [
        "manual_shortlist_id",
        "language",
        "id",
        "label",
        "toxic_input_processing",
    ]
].to_string(index=False))