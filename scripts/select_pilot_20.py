import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/detox_cot_pilot")
OUT = ROOT / "outputs"

GENS = OUT / "cot_pilot_generations.csv"
YO_OUT = OUT / "yo_pilot_20.csv"
XH_OUT = OUT / "xh_pilot_20.csv"

if not GENS.exists():
    raise SystemExit(f"Missing generations file: {GENS}")

df = pd.read_csv(GENS)

# Keep only rows with toxic_input
df = df.dropna(subset=["toxic_input"])
df = df[df["toxic_input"].str.strip().ne("")]

# Heuristic scoring for "clearly toxic"
def score_toxicity(row):
    t = (row["toxic_input"] or "").lower()
    score = 0
    if 5 <= len(t.split()) <= 15:
        score += 1
    if any(k in t for k in ["jẹ́", "aláì", "burú", "asán", "aláìmọ̀", "stupid", "idiot", "useless", "crazy"]):
        score += 2
    if any(k in t for k in ["banujẹ", "jìyà", "bínú", "run", "pa", "bá ẹ jẹ́", "kò yẹ", "àjálù", "isihogo", "yahla", "umsindo"]):
        score += 2
    return score

df["score"] = df.apply(score_toxicity, axis=1)

yo = df[df["language"] == "Yorùbá"].copy()
xh = df[df["language"] == "isiXhosa"].copy()

yo_sel = yo.sort_values("score", ascending=False).head(10).copy()
xh_sel = xh.sort_values("score", ascending=False).head(10).copy()

yo_sel["split"] = "pilot_20"
xh_sel["split"] = "pilot_20"

yo_sel.to_csv(YO_OUT, index=False, encoding="utf-8")
xh_sel.to_csv(XH_OUT, index=False, encoding="utf-8")

print("Selected pilot inputs:")
print(YO_OUT, len(yo_sel))
print(XH_OUT, len(xh_sel))
