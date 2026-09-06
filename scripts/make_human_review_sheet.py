import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/detox_cot_pilot")
OUT = ROOT / "outputs"

GENS = OUT / "cot_pilot_v2_generations.csv"
REVIEW = OUT / "human_review_sheet.csv"

if not GENS.exists():
    raise SystemExit("Missing generations file: " + str(GENS))

df = pd.read_csv(GENS)

keep_cols = [
    "language", "toxic_input", "generated_output",
    "source_status", "detox_strategy", "meaning_risk", "analysis_reason"
]
out = df[keep_cols].copy()
out["decision"] = ""
out["comments"] = ""

out.to_csv(REVIEW, index=False, encoding="utf-8")
print("Human review sheet:", REVIEW)