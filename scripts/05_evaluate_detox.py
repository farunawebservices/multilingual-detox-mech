#!/usr/bin/env python
"""Stage 5 -- Automatic evaluation of the baseline generations.

Computes STA, SIM, FL and the joint score J for every generation produced in Stage 4,
separately for dev and test and separately for each of the three seeds, and reports
them alongside the Stage 4 degeneracy diagnostics so a joint score can never be read
as a single quality axis.

Everything is read-only: datasets, splits, checkpoints and generation files are opened
for reading and their checksums are verified unchanged at the end of the run.

Metric definitions (following the official TextDetox 2024 protocol)
-------------------------------------------------------------------
STA  Style transfer accuracy. P(neutral) from textdetox/xlmr-large-toxicity-classifier,
     whose label mapping was verified as {0: neutral, 1: toxic}.
SIM  Content preservation. Cosine similarity between LaBSE embeddings of the original
     toxic input and the generated output.
FL   Fluency. chrF1 of the generation against the human reference. The 2024 shared task
     used ChrF rather than a classifier precisely because no CoLA-style fluency data
     exists for most of these languages; beta=1 gives the "chrF1" variant they specify.
J    Mean over samples of STA * SIM * FL.

Language coverage, checked rather than assumed
----------------------------------------------
The STA classifier's model card enumerates nine fine-tuning languages: en, ru, uk, de,
es, ar, am, hi, zh. Seven of this study's languages are covered. **Yoruba and isiXhosa
are not.** XLM-R's pretraining includes xh but not yo, and neither has toxicity-labelled
fine-tuning data, so the classifier will still emit a confident-looking probability for
both while having no validated basis for it.

Accordingly STA is reported for yo/xh only in columns explicitly suffixed `_unvalidated`,
and J -- which depends on STA -- is NaN for those two languages. A parallel
`J_unvalidated` column is provided so the number exists for inspection, but it is never
presented as a validated result. This follows the brief's instruction to mark STA
unavailable rather than silently reporting a number as if it were reliable.

LaBSE declares 110 languages including yo, xh and am, so SIM is valid throughout.
chrF is language-agnostic, so FL is valid throughout.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Input toxicity is also scored, for the covered languages, and reported as
  `sta_input`. Without it there is no way to tell a genuinely detoxifying model from
  one handed inputs the classifier already considers neutral, and it doubles as a
  sanity check that the classifier behaves on this corpus.
* Bootstrap resampling is over pair_ids within a language/split/seed cell, which is the
  brief's "paired by sentence pair" unit: a resample draws whole pairs, so STA, SIM and
  FL for a given sentence always move together. 1000 resamples, percentile 95% CI, with
  a fixed seed so intervals are reproducible.
* Cross-seed aggregation reports the mean and standard deviation of the three per-seed
  means. With only three seeds a standard deviation is descriptive, not inferential, so
  no CI is attached to it.
* Flag thresholds are fixed constants declared below rather than tuned, and the
  underlying values are always reported next to the flag so a reader can apply their
  own threshold.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sacrebleu.metrics import CHRF
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
GEN_DIR = REPO_ROOT / "results" / "generations"
EVAL_DIR = REPO_ROOT / "results" / "evaluation"

# STA classifier fine-tuning languages, from its model card.
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
CI = 0.95
BATCH_SIZE = 32

FLAG_THRESHOLDS = {
    "template_collapse_unique_output_rate": 0.60,
    "template_collapse_top_output_share": 0.25,
    "high_copy_rate": 0.20,
    "sta_high": 0.70,
    "sim_low": 0.50,
    "sim_high": 0.70,
    "sta_low": 0.50,
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def file_digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@torch.no_grad()
def score_neutrality(texts: list[str], model, tokenizer, device: str) -> np.ndarray:
    """P(neutral) for each text. Label mapping verified as {0: neutral, 1: toxic}."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t if str(t).strip() else " " for t in texts[i:i + BATCH_SIZE]]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)
        out.extend(probs[:, 0].cpu().numpy())
    return np.asarray(out)


@torch.no_grad()
def labse_similarity(a: list[str], b: list[str], model, tokenizer, device: str) -> np.ndarray:
    def embed(texts: list[str]) -> torch.Tensor:
        vecs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t if str(t).strip() else " " for t in texts[i:i + BATCH_SIZE]]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to(device)
            out = model(**enc)
            # LaBSE sentence embedding is the pooler output, L2-normalised.
            emb = out.pooler_output
            vecs.append(torch.nn.functional.normalize(emb, p=2, dim=1).cpu())
        return torch.cat(vecs)

    return (embed(a) * embed(b)).sum(dim=1).numpy()


def chrf1_scores(hyps: list[str], refs: list[str]) -> np.ndarray:
    metric = CHRF(beta=1)  # chrF1, the TextDetox 2024 FL variant
    return np.asarray([
        metric.sentence_score(str(h) if str(h).strip() else " ", [str(r)]).score / 100.0
        for h, r in zip(hyps, refs)
    ])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI over resampled sentence pairs."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def aggregate(scores: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Per language/split/seed metrics with bootstrap CIs, joined to Stage 4 diagnostics."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (split, lang, seed), g in scores.groupby(["split", "language", "seed"]):
        supported = lang in STA_SUPPORTED
        row = {
            "split": split, "language": lang, "seed": seed,
            "n_examples": int(len(g)),
            "sta_supported": supported,
            "sta_input": float(g["sta_input"].mean()) if supported else float("nan"),
        }
        for metric in ("sta", "sim", "fl", "j"):
            # Where the classifier has no coverage the values live in the
            # *_unvalidated column, so read from there rather than the NaN'd one.
            key = metric if (supported or metric in ("sim", "fl")) else f"{metric}_unvalidated"
            vals = g[key].to_numpy(dtype=float)
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            lo, hi = bootstrap_ci(vals, rng)
            row[key] = mean
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
            if not supported and metric in ("sta", "j"):
                row[metric] = float("nan")

        # Detoxification actually achieved: how much more neutral the output is than
        # the input. A high STA on inputs the classifier already calls neutral is not
        # evidence of detoxification, so the raw STA alone can be misleading.
        row["delta_sta"] = (row["sta"] - row["sta_input"]) if supported else float("nan")
        rows.append(row)

    agg = pd.DataFrame(rows)
    keep = ["seed", "split", "language", "unique_output_rate", "top_output_share",
            "prefix_diversity", "template_rate", "matches_train_output_fuzzy_rate",
            "empty_rate", "sentinel_rate", "copy_input_rate", "copy_input_rate_loose"]
    return agg.merge(diagnostics[keep], on=["seed", "split", "language"], how="left")


def across_seeds(agg: pd.DataFrame) -> pd.DataFrame:
    metrics = ["sta", "sim", "fl", "j", "sta_unvalidated", "j_unvalidated", "sta_input",
               "delta_sta",
               "unique_output_rate", "top_output_share", "prefix_diversity",
               "template_rate", "matches_train_output_fuzzy_rate", "empty_rate",
               "sentinel_rate", "copy_input_rate", "copy_input_rate_loose"]
    metrics = [m for m in metrics if m in agg.columns]
    out = agg.groupby(["split", "language"]).agg(
        n_examples=("n_examples", "first"),
        n_seeds=("seed", "nunique"),
        sta_supported=("sta_supported", "first"),
        **{f"{m}_{stat}": (m, stat) for m in metrics for stat in ("mean", "std")},
    ).reset_index()
    return out


def build_flags(seedmean: pd.DataFrame) -> dict:
    t = FLAG_THRESHOLDS
    flags: dict = {"thresholds": t, "by_split": {}}
    for split, g in seedmean.groupby("split"):
        entries = []
        for _, r in g.iterrows():
            lang = r["language"]
            reasons = []
            if (r["unique_output_rate_mean"] < t["template_collapse_unique_output_rate"]
                    or r["top_output_share_mean"] > t["template_collapse_top_output_share"]):
                reasons.append(
                    f"template collapse/repetition: unique_output_rate="
                    f"{r['unique_output_rate_mean']:.3f}, "
                    f"top_output_share={r['top_output_share_mean']:.3f}"
                )
            if r["copy_input_rate_loose_mean"] > t["high_copy_rate"]:
                reasons.append(
                    f"high copy-input rate: {r['copy_input_rate_loose_mean']:.3f} "
                    f"(exact {r['copy_input_rate_mean']:.3f})"
                )
            sta = r["sta_mean"] if r["sta_supported"] else r.get("sta_unvalidated_mean", np.nan)
            sim = r["sim_mean"]
            label = "" if r["sta_supported"] else " [STA UNVALIDATED]"
            if np.isfinite(sta) and sta > t["sta_high"] and sim < t["sim_low"]:
                reasons.append(
                    f"STA high but SIM low{label}: STA={sta:.3f}, SIM={sim:.3f} "
                    f"-- output may be discarding input content"
                )
            if np.isfinite(sta) and sim > t["sim_high"] and sta < t["sta_low"]:
                reasons.append(
                    f"SIM high but STA low{label}: SIM={sim:.3f}, STA={sta:.3f} "
                    f"-- output may be echoing the toxic input"
                )
            if not r["sta_supported"]:
                reasons.append("STA unavailable/unvalidated: language outside the "
                               "classifier's nine fine-tuning languages")
            if reasons:
                entries.append({"language": lang, "flags": reasons})
        flags["by_split"][split] = entries
    return flags


def main() -> int:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    inputs = sorted(GEN_DIR.glob("seed*_*.csv")) + [
        GEN_DIR / "template_diagnostics.csv",
        REPO_ROOT / "data" / "splits" / "dev.csv",
        REPO_ROOT / "data" / "splits" / "test.csv",
    ]
    digests_before = {p.name: file_digest(p) for p in inputs}

    frames = []
    for seed in cfg["seeds_multi"]:
        for split in ("dev", "test"):
            df = pd.read_csv(GEN_DIR / f"seed{seed}_{split}.csv")
            frames.append(df)
    gen = pd.concat(frames, ignore_index=True)
    diagnostics = pd.read_csv(GEN_DIR / "template_diagnostics.csv")

    print(f"loaded {len(gen)} generations "
          f"({gen['split'].value_counts().to_dict()}), {gen['seed'].nunique()} seeds")
    covered = sorted(set(cfg["languages"]) & STA_SUPPORTED)
    uncovered = sorted(set(cfg["languages"]) - STA_SUPPORTED)
    print(f"STA coverage : supported {covered}")
    print(f"               UNSUPPORTED {uncovered} -> STA/J reported as unvalidated only")

    gen["generated"] = gen["generated"].fillna("").astype(str)

    # ---- STA ----
    print("\nscoring STA ...")
    sta_id = cfg["models"]["sta"]
    tok = AutoTokenizer.from_pretrained(sta_id)
    clf = AutoModelForSequenceClassification.from_pretrained(sta_id).to(device).eval()
    assert clf.config.id2label[0].lower() == "neutral", clf.config.id2label
    gen["sta_raw"] = score_neutrality(gen["generated"].tolist(), clf, tok, device)
    gen["sta_input"] = score_neutrality(gen["toxic_input"].astype(str).tolist(),
                                        clf, tok, device)
    del clf
    torch.cuda.empty_cache()

    # ---- SIM ----
    print("scoring SIM (LaBSE) ...")
    sim_id = cfg["models"]["sim"]
    ltok = AutoTokenizer.from_pretrained(sim_id)
    lmodel = AutoModel.from_pretrained(sim_id).to(device).eval()
    gen["sim"] = labse_similarity(gen["toxic_input"].astype(str).tolist(),
                                  gen["generated"].tolist(), lmodel, ltok, device)
    del lmodel
    torch.cuda.empty_cache()

    # ---- FL ----
    print("scoring FL (chrF1) ...")
    gen["fl"] = chrf1_scores(gen["generated"].tolist(),
                             gen["detox_reference"].astype(str).tolist())

    # ---- STA / J with coverage handling ----
    supported_mask = gen["language"].isin(STA_SUPPORTED)
    gen["sta_supported"] = supported_mask
    gen["sta"] = np.where(supported_mask, gen["sta_raw"], np.nan)
    gen["sta_unvalidated"] = np.where(supported_mask, np.nan, gen["sta_raw"])
    gen["j"] = gen["sta"] * gen["sim"] * gen["fl"]
    gen["j_unvalidated"] = gen["sta_unvalidated"] * gen["sim"] * gen["fl"]
    gen.loc[~supported_mask, "sta_input"] = np.nan

    for split in ("dev", "test"):
        gen[gen["split"] == split].to_csv(
            EVAL_DIR / f"{split}_scores.csv", index=False, encoding="utf-8")

    agg = aggregate(gen, diagnostics)
    seedmean = across_seeds(agg)

    for split in ("dev", "test"):
        agg[agg["split"] == split].to_csv(
            EVAL_DIR / f"{split}_metrics.csv", index=False, encoding="utf-8")
        seedmean[seedmean["split"] == split].to_csv(
            EVAL_DIR / f"{split}_metrics_across_seeds.csv", index=False, encoding="utf-8")
    agg.to_csv(REPO_ROOT / "results" / "baseline_metrics.csv", index=False, encoding="utf-8")

    flags = build_flags(seedmean)
    (EVAL_DIR / "flags.json").write_text(
        json.dumps({
            "run_id": cfg["run_id"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sta_model": sta_id,
            "sta_supported_languages": sorted(STA_SUPPORTED),
            "sta_unsupported_study_languages": uncovered,
            "sim_model": sim_id,
            "fl_metric": "chrF1 (sacrebleu CHRF beta=1) against the human reference",
            "bootstrap": {"resamples": BOOTSTRAP_N, "ci": CI, "unit": "sentence_pair",
                          "seed": BOOTSTRAP_SEED},
            **flags,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- console report ----------------
    for split in ("dev", "test"):
        s = seedmean[seedmean["split"] == split].set_index("language")
        print(f"\n=== {split.upper()}: core metrics, mean +/- sd over 3 seeds ===")
        disp = pd.DataFrame({
            "n": s["n_examples"].astype(int),
            "STA": s.apply(lambda r: (f"{r['sta_mean']:.3f}+-{r['sta_std']:.3f}"
                                      if r["sta_supported"]
                                      else f"({r['sta_unvalidated_mean']:.3f})*"), axis=1),
            "SIM": s.apply(lambda r: f"{r['sim_mean']:.3f}+-{r['sim_std']:.3f}", axis=1),
            "FL": s.apply(lambda r: f"{r['fl_mean']:.3f}+-{r['fl_std']:.3f}", axis=1),
            "J": s.apply(lambda r: (f"{r['j_mean']:.3f}+-{r['j_std']:.3f}"
                                    if r["sta_supported"]
                                    else f"({r['j_unvalidated_mean']:.3f})*"), axis=1),
            "STA_input": s["sta_input_mean"].round(3),
            "dSTA": s["delta_sta_mean"].round(3),
        })
        print(disp.to_string())
        print("  * parenthesised = UNVALIDATED, language outside the STA classifier's coverage")

        print(f"\n=== {split.upper()}: degeneracy diagnostics (mean over seeds) ===")
        print(s[["unique_output_rate_mean", "top_output_share_mean", "prefix_diversity_mean",
                 "template_rate_mean", "matches_train_output_fuzzy_rate_mean",
                 "empty_rate_mean", "sentinel_rate_mean", "copy_input_rate_mean",
                 "copy_input_rate_loose_mean"]].round(3).to_string())

    print("\n=== 95% bootstrap CIs, test split, seed 42 (1000 resamples, paired by pair) ===")
    t42 = agg[(agg["split"] == "test") & (agg["seed"] == 42)].set_index("language")
    for lang, r in t42.iterrows():
        sta_s = (f"STA {r['sta']:.3f} [{r['sta_ci_lo']:.3f},{r['sta_ci_hi']:.3f}]"
                 if r["sta_supported"]
                 else f"STA ({r['sta_unvalidated']:.3f})* "
                      f"[{r['sta_unvalidated_ci_lo']:.3f},{r['sta_unvalidated_ci_hi']:.3f}]")
        print(f"  {lang}: n={int(r['n_examples'])}  {sta_s}  "
              f"SIM {r['sim']:.3f} [{r['sim_ci_lo']:.3f},{r['sim_ci_hi']:.3f}]  "
              f"FL {r['fl']:.3f} [{r['fl_ci_lo']:.3f},{r['fl_ci_hi']:.3f}]")

    print("\n=== FLAGS ===")
    for split, entries in flags["by_split"].items():
        print(f"  [{split}]")
        for e in entries:
            print(f"    {e['language']}:")
            for f in e["flags"]:
                print(f"      - {f}")

    digests_after = {p.name: file_digest(p) for p in inputs}
    changed = [k for k in digests_before if digests_before[k] != digests_after[k]]
    print(f"\ninput files unchanged: {'YES' if not changed else 'NO -> ' + str(changed)}")
    print(f"wrote {EVAL_DIR.relative_to(REPO_ROOT)}/"
          "{dev,test}_{scores,metrics,metrics_across_seeds}.csv, flags.json, "
          "and results/baseline_metrics.csv")
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
