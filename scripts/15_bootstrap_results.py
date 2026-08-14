#!/usr/bin/env python
"""Stage 15 -- Statistical aggregation across Stages 5, 11, 13 and 14.

Aggregates the per-row scored outputs already on disk, computes paired bootstrap 95%
confidence intervals by resampling sentence pairs, and separates the five kinds of
evidence the project has produced so they cannot be read as one undifferentiated result.

Usage
-----
    python scripts/15_bootstrap_results.py --smoke   # verification pass, writes nothing
    python scripts/15_bootstrap_results.py           # full aggregation

Nothing is regenerated. No model is loaded, no text is produced, no steering strength is
extended, and the Stage 13 test evaluation is read but never re-run. All output is written
under results/statistics/ and no existing artifact is modified.

Paired bootstrap
----------------
Every interval is paired at the sentence-pair level, not the row level. For an
intervention cell, each pair_id contributes the difference between its intervened score
and its own baseline score under the same seed and language; the bootstrap then resamples
*pair_ids* with replacement and recomputes the mean difference. Resampling rows
independently would break the pairing and understate the correlation between an
intervention and its baseline, producing intervals that are too wide for the deltas and
too confident about their sign.

Macro averages resample pairs stratified by language, so the nine-language balance that
Stage 2 built into the splits is preserved inside every bootstrap replicate rather than
being disturbed by sampling variation.

Coverage policy
---------------
STA and J are reported only for the seven languages the toxicity classifier was fine-tuned
on. Yoruba and isiXhosa carry SIM, FL, copy rate, template rate, unique-output rate and
top-output share; their STA appears only in `_unvalidated` columns and never enters a
delta, an effect size or a flag.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Effect size is Cohen's d_z, the mean paired difference over its own standard deviation,
  which is the matched statistic for a paired design. Reporting an unpaired d here would
  ignore that every intervention row has a baseline twin.
* Degeneracy diagnostics (copy, template, unique-output, top-output share) are recomputed
  uniformly from the generated text at this stage rather than trusted from each source
  file, because the four stages recorded slightly different subsets of them. The rules are
  the Stage 4 definitions.
* Unique-output rate and top-output share are properties of a set, not of a row, so they
  have no per-pair difference and receive no paired CI. They are reported as point
  estimates per cell with the across-seed spread instead.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
OUT_DIR = REPO_ROOT / "results" / "statistics"

WS_RE = re.compile(r"\s+")
SENTINEL_RE = re.compile(r"<extra_id_\d+>")
BOOTSTRAP_N = 1000
CI = 0.95
RNG_SEED = 42
AFRICAN = ["yo", "xh"]
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}
METRICS = ["sta", "sim", "fl", "j"]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def norm(t: str) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(t))).strip()


def loose(t: str) -> str:
    return norm(t).casefold()


def train_frequent_outputs() -> dict:
    tr = pd.read_csv(SPLITS_DIR / "train.csv")
    out = {}
    for lang, g in tr.groupby("language"):
        c = Counter(norm(t) for t in g["detox_output"])
        out[lang] = {t for t, n in c.items() if n >= 2}
    return out


# ------------------------------------------------------------------ harmonisation

def add_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    gen = df["generated"].fillna("").astype(str)
    tox = df["toxic_input"].astype(str)
    df = df.copy()
    df["is_empty"] = gen.str.strip().eq("")
    df["has_sentinel"] = [bool(SENTINEL_RE.search(g)) for g in gen]
    df["is_copy"] = [norm(g) == norm(t) for g, t in zip(gen, tox)]
    df["is_copy_loose"] = [loose(g) == loose(t) for g, t in zip(gen, tox)]
    df["generated_norm"] = [norm(g) for g in gen]
    return df


def harmonise() -> pd.DataFrame:
    """One long frame: stage, split, intervention, family, language, seed, pair_id."""
    frames = []

    # --- Stage 5 baseline (unsteered, unablated) ---
    for split in ("dev", "test"):
        p = REPO_ROOT / "results" / "evaluation" / f"{split}_scores.csv"
        d = pd.read_csv(p)
        d["stage"], d["split"] = "stage05_baseline", split
        d["intervention"], d["family"] = "baseline", "baseline"
        d["is_baseline"] = True
        frames.append(d)

    # --- Stage 11 ablation ---
    for split in ("dev", "test"):
        p = REPO_ROOT / "results" / "ablation" / f"{split}_ablation_scores.csv"
        d = pd.read_csv(p)
        d["stage"], d["split"] = "stage11_ablation", split
        d["intervention"] = d["arm"]
        d["family"] = np.where(
            d["arm"].str.startswith("random_"), "ablation_random_control",
            np.where(d["arm"] == "sae_replicated_features", "ablation_sae_control",
                     np.where(d["arm"] == "baseline_unablated", "baseline",
                              "ablation_primary")))
        d["is_baseline"] = d["arm"] == "baseline_unablated"
        frames.append(d)

    # --- Stage 13 steering (dev sweep + frozen test) ---
    for split, p in (("dev", REPO_ROOT / "results" / "steering" / "steering_dev_scored.csv"),
                     ("test", REPO_ROOT / "results" / "steering" / "steering_test_scored.csv")):
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["stage"], d["split"] = "stage13_steering", split
        d["intervention"] = (d["side"] + "/" + d["direction_type"] + "/" + d["scope"]
                             + "@" + d["strength"].map(lambda x: f"{x:+.2f}"))
        d["family"] = np.where(
            d["strength"] == 0.0, "baseline",
            np.where(d["direction_type"] == "random", "steering_random_control",
                     np.where(d["direction_type"] == "sae", "steering_sae_control",
                              "steering_primary")))
        d["is_baseline"] = d["strength"] == 0.0
        frames.append(d)

    # --- Stage 14 transfer (dev only) ---
    p = REPO_ROOT / "results" / "transfer" / "transfer_dev_scored.csv"
    d = pd.read_csv(p)
    d["stage"], d["split"] = "stage14_transfer", "dev"
    d["intervention"] = (d["side"] + "/" + d["vector_type"] + "/" + d["source_group"]
                         + "@" + d["strength"].map(lambda x: f"{x:+.2f}"))
    d["family"] = np.where(
        d["strength"] == 0.0, "baseline",
        np.where(d["vector_type"] == "random", "transfer_random_control",
                 "transfer_primary"))
    d["is_baseline"] = d["strength"] == 0.0
    # Stage 14 labels its unsteered rows vector_type="baseline", so the pairing key must
    # exclude vector_type or no intervention row would ever find its baseline twin.
    d["baseline_key"] = d["side"] + "|" + d["source_group"]
    frames.append(d)

    keep = ["stage", "split", "intervention", "family", "is_baseline", "language",
            "seed", "pair_id", "generated", "toxic_input", "detox_reference",
            "sta", "sta_unvalidated", "sim", "fl", "j"]
    out = []
    for f in frames:
        for c in keep:
            if c not in f.columns:
                f[c] = np.nan
        # baseline grouping key: which baseline a row is paired against
        if "baseline_key" not in f.columns:
            if f["stage"].iloc[0] == "stage13_steering":
                f["baseline_key"] = f["side"] + "|" + f["direction_type"] + "|" + f["scope"]
            else:
                f["baseline_key"] = "all"
        out.append(f[keep + ["baseline_key"]])
    return add_diagnostics(pd.concat(out, ignore_index=True))


# ---------------------------------------------------------------------- bootstrap

def paired_ci(diff_by_pair: np.ndarray, rng: np.random.Generator,
              strata: np.ndarray | None = None) -> tuple[float, float]:
    d = np.asarray(diff_by_pair, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    if len(d) == 0:
        return float("nan"), float("nan")
    if strata is None:
        idx = rng.integers(0, len(d), size=(BOOTSTRAP_N, len(d)))
        means = d[idx].mean(1)
    else:
        s = np.asarray(strata)[ok]
        means = np.zeros(BOOTSTRAP_N)
        groups = [np.flatnonzero(s == u) for u in np.unique(s)]
        for b in range(BOOTSTRAP_N):
            vals = [d[g[rng.integers(0, len(g), size=len(g))]].mean() for g in groups]
            means[b] = float(np.mean(vals))
    lo, hi = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def set_diagnostics(g: pd.DataFrame, freq: dict) -> dict:
    texts = g["generated_norm"].tolist()
    n = len(texts)
    c = Counter(texts)
    lang = g["language"].iloc[0]
    return {
        "copy_rate": float(g["is_copy"].mean()),
        "loose_copy_rate": float(g["is_copy_loose"].mean()),
        "empty_rate": float(g["is_empty"].mean()),
        "sentinel_rate": float(g["has_sentinel"].mean()),
        "unique_output_rate": len(c) / n if n else np.nan,
        "top_output_share": c.most_common(1)[0][1] / n if n else np.nan,
        "template_rate": float(np.mean([t in freq.get(lang, set()) for t in texts])),
    }


def analyse(long: pd.DataFrame, freq: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    per_cell, macro_rows = [], []

    for (stage, split), block in long.groupby(["stage", "split"]):
        base = block[block["is_baseline"]]
        inter = block[~block["is_baseline"]]
        if base.empty:
            continue
        # Group the baseline into full DataFrames rather than a MultiIndex lookup:
        # .loc on a MultiIndex consumes the key columns, so the slice loses `language`
        # and `seed`, which the diagnostics still need.
        bmap = {k: v for k, v in base.groupby(["baseline_key", "seed", "language"])}

        for (interv, family, seed, lang, bkey), g in inter.groupby(
                ["intervention", "family", "seed", "language", "baseline_key"]):
            b = bmap.get((bkey, seed, lang))
            if b is None or b.empty:
                continue
            merged = g.merge(b[["pair_id"] + METRICS],
                             on="pair_id", suffixes=("", "_base"))
            if merged.empty:
                continue
            supported = lang in STA_SUPPORTED
            row = {"stage": stage, "split": split, "intervention": interv,
                   "family": family, "seed": seed, "language": lang,
                   "n_pairs": int(len(merged)), "sta_supported": supported,
                   "sta_status": "validated" if supported else "unavailable/unvalidated"}
            for m in METRICS:
                if m in ("sta", "j") and not supported:
                    row.update({m: np.nan, f"delta_{m}": np.nan,
                                f"delta_{m}_ci_lo": np.nan, f"delta_{m}_ci_hi": np.nan,
                                f"effect_size_{m}": np.nan})
                    continue
                diff = (merged[m] - merged[f"{m}_base"]).to_numpy(dtype=float)
                lo, hi = paired_ci(diff, rng)
                sd = np.nanstd(diff, ddof=1)
                row.update({
                    m: float(np.nanmean(merged[m])),
                    f"{m}_baseline": float(np.nanmean(merged[f"{m}_base"])),
                    f"delta_{m}": float(np.nanmean(diff)),
                    f"delta_{m}_ci_lo": lo, f"delta_{m}_ci_hi": hi,
                    f"effect_size_{m}": float(np.nanmean(diff) / sd) if sd > 1e-12 else np.nan,
                })
            row["sta_unvalidated"] = (float(np.nanmean(g["sta_unvalidated"]))
                                      if not supported else np.nan)
            row.update(set_diagnostics(g, freq))
            bd = set_diagnostics(b, freq)
            for k, v in bd.items():
                row[f"baseline_{k}"] = v
                row[f"delta_{k}"] = row[k] - v
            per_cell.append(row)

        # macro: stratified by language, validated languages only for STA/J
        for (interv, family, seed, bkey), g in inter.groupby(
                ["intervention", "family", "seed", "baseline_key"]):
            merged = g.merge(base[base["baseline_key"] == bkey][
                ["pair_id", "seed", "language"] + METRICS],
                on=["pair_id", "seed", "language"], suffixes=("", "_base"))
            if merged.empty:
                continue
            row = {"stage": stage, "split": split, "intervention": interv,
                   "family": family, "seed": seed, "scope": "macro_over_languages",
                   "n_pairs": int(len(merged)),
                   "n_languages": int(merged["language"].nunique())}
            for m in METRICS:
                sub = (merged[merged["language"].isin(STA_SUPPORTED)]
                       if m in ("sta", "j") else merged)
                if sub.empty:
                    row.update({f"delta_{m}": np.nan}); continue
                diff = (sub[m] - sub[f"{m}_base"]).to_numpy(dtype=float)
                lo, hi = paired_ci(diff, rng, strata=sub["language"].to_numpy())
                sd = np.nanstd(diff, ddof=1)
                row.update({m: float(np.nanmean(sub[m])),
                            f"delta_{m}": float(np.nanmean(diff)),
                            f"delta_{m}_ci_lo": lo, f"delta_{m}_ci_hi": hi,
                            f"effect_size_{m}": float(np.nanmean(diff) / sd)
                            if sd > 1e-12 else np.nan,
                            f"{m}_languages_included": int(sub["language"].nunique())})
            macro_rows.append(row)

    return pd.DataFrame(per_cell), pd.DataFrame(macro_rows)


def evidence_summary() -> dict:
    """Item 11: keep the five kinds of evidence separate and labelled."""
    ev = {}
    p = REPO_ROOT / "results" / "probes" / "probe_summary_across_seeds.csv"
    if p.exists():
        s = pd.read_csv(p)
        c = s[(s["probe"] == "condition:toxic_vs_detox_reference")]
        ev["1_associative_probe_evidence"] = {
            "claim_strength": "ASSOCIATIVE ONLY -- correlational, no causal test",
            "best_dev_accuracy": float(c["dev_accuracy_mean"].max()),
            "control_baseline": float(c["control_mean"].mean()),
            "note": "Stage 7 linear probes separate toxic from reference; a probe is not "
                    "evidence that the model uses the feature.",
        }
    q = REPO_ROOT / "results" / "sae_features" / "layer6_feature_stats.csv"
    if q.exists():
        f = pd.read_csv(q)
        ev["2_sae_negative_result"] = {
            "claim_strength": "NEGATIVE",
            "n_candidates": int(len(f)),
            "n_promoted_detoxification_features": int(f["detoxification_feature"].sum()),
            "n_replicated": int(f["replicated"].sum()),
            "note": "Stage 9: no SAE feature passed replication, controls and behaviour "
                    "together. Stage 11's SAE ablation arm was null.",
        }
    ev["3_causal_ablation_evidence"] = {
        "claim_strength": "CAUSAL -- intervention with matched random controls",
        "note": "Stage 11: mean-ablating top decoder cross-attention heads moves STA while "
                "matched random heads do not. Effect is entangled with SIM loss.",
    }
    ev["4_steering_null_or_weak"] = {
        "claim_strength": "NULL / WEAK",
        "note": "Stage 13: unit-norm rank-1 additive steering at the pre-specified "
                "strengths produces little or no benefit; decoder-side is inert.",
    }
    ev["5_cross_lingual_transfer_null"] = {
        "claim_strength": "NULL",
        "note": "Stage 14: transfer into yo/xh is not distinguishable from a matched "
                "random vector, and the within-language reference is null too, so the "
                "null is about intervention magnitude, not transfer specifically.",
    }
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freq = train_frequent_outputs()

    long = harmonise()
    print(f"harmonised {len(long)} scored rows")
    print(long.groupby(["stage", "split"]).size().to_string())

    if args.smoke:
        sub = long[long["stage"] == "stage11_ablation"]
        cells, macro = analyse(sub, freq, cfg)
        print(f"\n  smoke: {len(cells)} per-language cells, {len(macro)} macro rows")
        print(f"  paired CIs finite: "
              f"{bool(np.isfinite(cells['delta_sim_ci_lo']).all())}")
        print(f"  yo/xh STA suppressed: "
              f"{bool(cells[cells.language.isin(AFRICAN)]['delta_sta'].isna().all())}")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    cells, macro = analyse(long, freq, cfg)
    cells.to_csv(OUT_DIR / "per_language_intervention_stats.csv", index=False)
    macro.to_csv(OUT_DIR / "macro_intervention_stats.csv", index=False)

    # across-seed aggregation
    across = macro.groupby(["stage", "split", "intervention", "family"]).agg(
        n_seeds=("seed", "nunique"),
        **{f"{k}_mean": (k, "mean") for k in
           ["delta_sta", "delta_sim", "delta_fl", "delta_j",
            "effect_size_sta", "effect_size_sim", "effect_size_j"]},
        **{f"{k}_sd": (k, "std") for k in
           ["delta_sta", "delta_sim", "delta_j"]}).reset_index()
    across.to_csv(OUT_DIR / "across_seed_summary.csv", index=False)

    # STA-up / SIM-down flags, validated languages only
    flags = cells[(cells["sta_supported"]) & (cells["delta_sta"] > 0.01)
                  & (cells["delta_sim"] < -0.02)].copy()
    flags = flags[["stage", "split", "intervention", "family", "seed", "language",
                   "delta_sta", "delta_sta_ci_lo", "delta_sta_ci_hi",
                   "delta_sim", "delta_sim_ci_lo", "delta_sim_ci_hi",
                   "delta_j", "effect_size_sta", "effect_size_sim"]]
    flags.to_csv(OUT_DIR / "sta_up_sim_down_flags.csv", index=False)

    ev = evidence_summary()
    (OUT_DIR / "evidence_classes.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {"resamples": BOOTSTRAP_N, "ci": CI, "unit": "sentence_pair",
                      "pairing": "each pair_id differenced against its own baseline",
                      "macro": "stratified by language"},
        "sta_coverage_policy": "validated for 7 languages; yo/xh reported without STA",
        "sources_read": ["results/evaluation", "results/ablation", "results/steering",
                         "results/transfer", "results/probes", "results/sae_features"],
        "outputs_written_only_under": "results/statistics/",
        "reruns_performed": "none",
        "evidence_classes": ev,
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    print(f"\n=== cells: {len(cells)} per-language, {len(macro)} macro ===")
    print(f"  yo/xh STA suppressed everywhere: "
          f"{bool(cells[cells.language.isin(AFRICAN)]['delta_sta'].isna().all())}")

    print("\n=== STAGE 11 ABLATION -- macro deltas, mean over seeds (CAUSAL) ===")
    a = across[across["stage"] == "stage11_ablation"]
    print(a[["split", "intervention", "family", "delta_sta_mean", "delta_sim_mean",
             "delta_j_mean", "effect_size_sta_mean"]].round(4).to_string(index=False))

    print("\n=== STAGE 11 -- paired 95% CI on dSTA, test, top arms vs controls ===")
    m = macro[(macro["stage"] == "stage11_ablation") & (macro["split"] == "test")]
    for interv, g in m.groupby("intervention"):
        if g["delta_sta"].isna().all():
            continue
        print(f"  {interv:26s} dSTA={g['delta_sta'].mean():+.4f} "
              f"[{g['delta_sta_ci_lo'].mean():+.4f},{g['delta_sta_ci_hi'].mean():+.4f}] "
              f"dSIM={g['delta_sim'].mean():+.4f} "
              f"[{g['delta_sim_ci_lo'].mean():+.4f},{g['delta_sim_ci_hi'].mean():+.4f}]")

    print("\n=== STAGE 13 STEERING -- frozen test, macro (NULL/WEAK) ===")
    s = across[(across["stage"] == "stage13_steering") & (across["split"] == "test")]
    print(s[["intervention", "family", "delta_sta_mean", "delta_sim_mean",
             "delta_j_mean"]].round(4).head(12).to_string(index=False))

    print("\n=== STAGE 14 TRANSFER -- primary vs random control (NULL) ===")
    t = across[across["stage"] == "stage14_transfer"]
    print(t.groupby("family")[["delta_sim_mean", "delta_fl_mean",
                               "delta_sta_mean"]].mean().round(4).to_string())

    print("\n=== YO/XH DIAGNOSTICS (no STA), Stage 11 test, primary arms ===")
    y = cells[(cells["stage"] == "stage11_ablation") & (cells["split"] == "test")
              & (cells["language"].isin(AFRICAN))
              & (cells["family"] == "ablation_primary")]
    print(y.groupby(["language", "intervention"])[
        ["delta_sim", "delta_fl", "delta_loose_copy_rate", "delta_template_rate",
         "delta_unique_output_rate", "delta_top_output_share"]].mean().round(3).to_string())

    print(f"\n=== STA-up / SIM-down flags: {len(flags)} cells ===")
    print(flags.groupby(["stage", "family"]).size().to_string())

    print("\n=== EVIDENCE CLASSES (kept separate) ===")
    for k, v in ev.items():
        print(f"  {k}: {v['claim_strength']}")

    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ (no artifact outside it was touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
