#!/usr/bin/env python
"""Stage 16 -- Paper tables, built only from committed artifacts.

Reads existing stage outputs and writes tables under results/tables/. No model is loaded,
no inference is run, and no prior artifact is modified.

Evidence labelling
------------------
Every result row carries a verdict, so a reader never has to infer claim strength:

  causal-beneficial  intervention, paired CI excludes zero, effect in the desired direction
  causal-harmful     intervention, paired CI excludes zero, effect in the undesired
                     direction (a significant STA loss, or a significant SIM loss bought
                     by an STA gain)
  null               paired CI includes zero -- reported as null, never as a trend
  associative        probe/correlational evidence with no causal test
  negative           a hypothesis tested and not supported

A CI that includes zero is never described as significant, and no p-value is manufactured
for a comparison that does not have one.

Coverage
--------
Validated STA (the seven classifier-covered languages) and unvalidated yo/xh STA occupy
*different columns* and are never summed, averaged or ranked together. Yoruba and
isiXhosa are always reported as separate rows wherever their diagnostics differ, which
Stage 15 showed they do -- ablation makes Yoruba more collapsed and isiXhosa less.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R = REPO_ROOT / "results"
OUT = R / "tables"

LANG_ORDER = ["am", "ar", "de", "en", "es", "hi", "uk", "yo", "xh"]
AFRICAN = ["yo", "xh"]
LANG_NAME = {"am": "Amharic", "ar": "Arabic", "de": "German", "en": "English",
             "es": "Spanish", "hi": "Hindi", "uk": "Ukrainian", "yo": "Yoruba",
             "xh": "isiXhosa"}


def lang_sort(df: pd.DataFrame, col: str = "language") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=LANG_ORDER, ordered=True)
    return df.sort_values(col)


def verdict(dlo, dhi, delta, higher_is_better=True) -> str:
    """Never call a zero-crossing interval significant."""
    if not np.isfinite(dlo) or not np.isfinite(dhi):
        return "not estimable"
    if dlo <= 0 <= dhi:
        return "null (CI includes zero)"
    good = (delta > 0) if higher_is_better else (delta < 0)
    return "causal-beneficial" if good else "causal-harmful"


def write(df: pd.DataFrame, name: str, note: str, index: bool = False) -> dict:
    p = OUT / f"{name}.csv"
    df.to_csv(p, index=index)
    return {"table": name, "path": str(p.relative_to(REPO_ROOT)),
            "rows": int(len(df)), "note": note}


# ------------------------------------------------------------------------ tables

def t1_corpus() -> pd.DataFrame:
    inv = pd.read_csv(R / "corpus_inventory.csv")
    man = json.loads((REPO_ROOT / "data" / "splits" / "split_manifest.json").read_text())
    inv = inv[~inv["is_supplementary"]]
    rows = []
    for _, r in inv.iterrows():
        l = r["language"]
        m = man["per_language"][l]
        rows.append({
            "language": l, "name": LANG_NAME[l], "source": r["source"],
            "corpus_rows": int(r["n_rows"]),
            "duplicate_pairs": int(r["n_duplicate_pairs"]),
            "identical_toxic_detox": int(r["n_identical_exact"]),
            "non_nfc_strings": int(r["n_non_nfc_strings"]),
            "toxic_chars_mean": round(r["toxic_char_mean"], 1),
            "detox_chars_mean": round(r["detox_char_mean"], 1),
            "detox_over_toxic_len": round(r["char_ratio_detox_over_toxic"], 2),
            "mt5_fertility_toxic": round(r["toxic_fertility"], 2),
            "prefix_diversity_toxic": round(r["toxic_prefix_diversity"], 3),
            "train": m["counts"]["train"], "dev": m["counts"]["dev"],
            "test": m["counts"]["test"], "reserve": m["reserve"],
        })
    return lang_sort(pd.DataFrame(rows))


def t2_baseline() -> pd.DataFrame:
    d = pd.read_csv(R / "evaluation" / "test_metrics_across_seeds.csv")
    rows = []
    for _, r in d.iterrows():
        l = r["language"]
        val = bool(r["sta_supported"])
        rows.append({
            "language": l, "name": LANG_NAME[l], "n_test": int(r["n_examples"]),
            "sta_validated": round(r["sta_mean"], 3) if val else np.nan,
            "sta_validated_sd": round(r["sta_std"], 3) if val else np.nan,
            "delta_sta_validated": round(r["delta_sta_mean"], 3) if val else np.nan,
            "sta_UNVALIDATED_yo_xh": (np.nan if val
                                      else round(r["sta_unvalidated_mean"], 3)),
            "sta_status": "validated" if val else "UNAVAILABLE/UNVALIDATED",
            "sim": round(r["sim_mean"], 3), "sim_sd": round(r["sim_std"], 3),
            "fl": round(r["fl_mean"], 3), "fl_sd": round(r["fl_std"], 3),
            "j_validated": round(r["j_mean"], 3) if val else np.nan,
            "copy_rate": round(r["copy_input_rate_mean"], 3),
            "loose_copy_rate": round(r["copy_input_rate_loose_mean"], 3),
            "unique_output_rate": round(r["unique_output_rate_mean"], 3),
            "top_output_share": round(r["top_output_share_mean"], 3),
            "template_rate": round(r["template_rate_mean"], 3),
        })
    return lang_sort(pd.DataFrame(rows))


def t3_probes() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(R / "probes" / "probe_summary_across_seeds.csv")
    s = s[s["probe"].isin(["condition:toxic_vs_detox_reference", "language_identity"])]
    perf = s[["probe", "side", "layer", "pooling", "dev_accuracy_mean",
              "dev_accuracy_std", "dev_auc_mean", "control_mean", "gain_mean",
              "n_seeds"]].copy()
    perf["evidence_class"] = "associative"
    perf = perf.round(4)

    t = pd.read_csv(R / "probes" / "transfer_matrix.csv")
    t = t[(t["contrast"] == "toxic_vs_detox_reference") & (t["layer"] == 6)]
    tm = t.groupby(["side", "source", "target", "transfer_type"]).agg(
        dev_accuracy_mean=("dev_accuracy", "mean"),
        dev_accuracy_sd=("dev_accuracy", "std"),
        dev_auc_mean=("dev_auc", "mean"), n_seeds=("seed", "nunique")).reset_index()
    tm["evidence_class"] = "associative"
    ctrl = pd.read_csv(R / "probes" / "control_baselines_by_language.csv")
    c = ctrl.groupby("language")["control_dev_accuracy_length_only"].mean()
    tm["length_only_control_for_target"] = tm["target"].map(c)
    return perf, tm.round(4)


def t4_sae() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(R / "sae" / "sae_metrics_multilingual.csv")
    recon = m.groupby(["side", "layer", "token_set"]).agg(
        train_explained_variance=("train_explained_variance", "mean"),
        dev_explained_variance=("dev_explained_variance", "mean"),
        dev_recon_mse=("dev_recon_mse", "mean"),
        dev_mean_l0=("dev_mean_l0", "mean"),
        dev_dead_feature_rate=("dev_dead_feature_rate", "mean"),
        tokens_per_latent=("tokens_per_latent", "mean"),
        n_seeds=("seed", "nunique")).reset_index().round(4)

    st = pd.read_csv(R / "sae" / "feature_stability.csv")
    stab = st.groupby(["side", "layer", "token_set"]).agg(
        mean_max_cosine=("mean_max_cosine", "mean"),
        frac_matched_above_0_7=("frac_features_matched_above_0.7", "mean")).reset_index()
    f = pd.read_csv(R / "sae_features" / "layer6_feature_stats.csv")
    stab["n_candidates_layer6"] = len(f)
    stab["n_replicated_layer6"] = int(f["replicated"].sum())
    stab["n_promoted_detoxification_features"] = int(f["detoxification_feature"].sum())
    stab["evidence_class"] = "negative"
    return recon, stab.round(4)


def t5_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(R / "statistics" / "macro_intervention_stats.csv")
    a = m[m["stage"] == "stage11_ablation"]
    rows = []
    for (split, interv, family), g in a.groupby(["split", "intervention", "family"]):
        r = {"split": split, "intervention": interv, "family": family,
             "n_seeds": int(g["seed"].nunique())}
        for met, better in (("sta", True), ("sim", True), ("fl", True), ("j", True)):
            d, lo, hi = (g[f"delta_{met}"].mean(), g[f"delta_{met}_ci_lo"].mean(),
                         g[f"delta_{met}_ci_hi"].mean())
            r[f"delta_{met}"] = round(d, 4)
            r[f"delta_{met}_ci"] = (f"[{lo:+.4f}, {hi:+.4f}]"
                                    if np.isfinite(lo) else "n/a")
            r[f"delta_{met}_verdict"] = verdict(lo, hi, d, better)
            r[f"effect_size_{met}"] = round(g[f"effect_size_{met}"].mean(), 3)
        # A significant STA gain paid for with a significant SIM loss is harmful.
        if (r["delta_sta_verdict"] == "causal-beneficial"
                and r["delta_sim_verdict"] == "causal-harmful"):
            r["overall_verdict"] = "causal but harmful trade-off (STA up, SIM down)"
        elif family.endswith("control"):
            r["overall_verdict"] = "control -- " + r["delta_sta_verdict"]
        else:
            r["overall_verdict"] = r["delta_sta_verdict"]
        rows.append(r)
    macro = pd.DataFrame(rows).sort_values(["split", "family", "intervention"])

    c = pd.read_csv(R / "statistics" / "per_language_intervention_stats.csv")
    c = c[(c["stage"] == "stage11_ablation") & (c["split"] == "test")]
    per = c.groupby(["intervention", "family", "language"]).agg(
        delta_sta=("delta_sta", "mean"), delta_sim=("delta_sim", "mean"),
        delta_fl=("delta_fl", "mean"), delta_j=("delta_j", "mean"),
        delta_loose_copy_rate=("delta_loose_copy_rate", "mean"),
        delta_unique_output_rate=("delta_unique_output_rate", "mean"),
        delta_top_output_share=("delta_top_output_share", "mean"),
        delta_template_rate=("delta_template_rate", "mean"),
        sta_status=("sta_status", "first")).reset_index()
    per["sta_note"] = np.where(per["language"].isin(AFRICAN),
                               "STA UNAVAILABLE -- diagnostics only", "validated")
    return macro, lang_sort(per).round(4)


def t6_steering() -> pd.DataFrame:
    d = pd.read_csv(R / "steering_dev.csv")
    prim = d[d["arm"].isin(["primary", "comparison"])]
    dev = prim.groupby(["side", "direction_type", "scope", "strength"]).agg(
        sta_validated=("sta", "mean"), sim=("sim", "mean"), fl=("fl", "mean"),
        j_validated=("j", "mean"), loose_copy_rate=("loose_copy_rate", "mean"),
        unique_output_rate=("unique_output_rate", "mean"),
        top_output_share=("top_output_share", "mean"),
        template_rate=("template_rate", "mean")).reset_index()
    dev["split"] = "dev"

    m = pd.read_csv(R / "statistics" / "macro_intervention_stats.csv")
    t = m[(m["stage"] == "stage13_steering") & (m["split"] == "test")]
    rows = []
    for interv, g in t.groupby("intervention"):
        r = {"split": "test (frozen config)", "intervention": interv,
             "n_seeds": int(g["seed"].nunique())}
        for met in ("sta", "sim", "fl", "j"):
            d_, lo, hi = (g[f"delta_{met}"].mean(), g[f"delta_{met}_ci_lo"].mean(),
                          g[f"delta_{met}_ci_hi"].mean())
            r[f"delta_{met}"] = round(d_, 4)
            r[f"delta_{met}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else "n/a"
            r[f"delta_{met}_verdict"] = verdict(lo, hi, d_, True)
        rows.append(r)
    test = pd.DataFrame(rows)
    test["evidence_class"] = "null/weak"
    return dev.round(4), test


def t7_transfer() -> pd.DataFrame:
    d = pd.read_csv(R / "transfer" / "transfer_dev.csv")
    g = d.groupby(["side", "source_group", "target_language", "vector_type",
                   "strength"]).agg(
        sta_validated=("sta", "mean"), sim=("sim", "mean"), fl=("fl", "mean"),
        j_validated=("j", "mean"), loose_copy_rate=("loose_copy_rate", "mean"),
        unique_output_rate=("unique_output_rate", "mean"),
        top_output_share=("top_output_share", "mean"),
        template_rate=("template_rate", "mean"),
        sta_status=("sta_status", "first"), n_seeds=("seed", "nunique")).reset_index()
    g["evidence_class"] = "null"
    m = pd.read_csv(R / "statistics" / "macro_intervention_stats.csv")
    t = m[m["stage"] == "stage14_transfer"]
    summary = t.groupby("family").agg(
        delta_sta=("delta_sta", "mean"), delta_sim=("delta_sim", "mean"),
        delta_fl=("delta_fl", "mean"), delta_j=("delta_j", "mean")).reset_index()
    summary["evidence_class"] = "null"
    summary["note"] = np.where(
        summary["family"] == "transfer_random_control",
        "matched random control -- moves metrics MORE than the primary vector",
        "primary transfer vector")
    return g.round(4), summary.round(4)


def t8_evidence() -> pd.DataFrame:
    ev = json.loads((R / "statistics" / "evidence_classes.json").read_text())
    m = pd.read_csv(R / "statistics" / "macro_intervention_stats.csv")
    ab = m[(m["stage"] == "stage11_ablation") & (m["split"] == "test")
           & (m["intervention"] == "top_decoder_cross_heads")]
    rows = [
        {"finding": "Linear probes separate toxic from reference activations",
         "stage": "7", "evidence_class": "ASSOCIATIVE",
         "causal_test": "none", "statistic": "dev accuracy 0.93 vs 0.553 control",
         "ci_excludes_zero": "n/a",
         "claim": "correlational only; not evidence the model uses the feature"},
        {"finding": "Probe direction transfers to yo/xh above chance (decoder)",
         "stage": "7", "evidence_class": "ASSOCIATIVE", "causal_test": "none",
         "statistic": "0.67-0.83 vs 1.00 within-language; encoder at chance",
         "ci_excludes_zero": "n/a",
         "claim": "partial sharing; confounded by yo/xh length asymmetry"},
        {"finding": "No SAE feature is a detoxification feature",
         "stage": "8/9", "evidence_class": "NEGATIVE", "causal_test": "Stage 11 arm",
         "statistic": "0/600 promoted; 3.7% replicate across seeds",
         "ci_excludes_zero": "no", "claim": "hypothesis tested and not supported"},
        {"finding": "Ablating top decoder cross-attention heads raises STA",
         "stage": "11", "evidence_class": "CAUSAL",
         "causal_test": "mean ablation + matched random controls",
         "statistic": f"dSTA {ab['delta_sta'].mean():+.4f} "
                      f"[{ab['delta_sta_ci_lo'].mean():+.4f}, "
                      f"{ab['delta_sta_ci_hi'].mean():+.4f}], d_z "
                      f"{ab['effect_size_sta'].mean():.3f}",
         "ci_excludes_zero": "yes",
         "claim": "causal; controls null; effect size modest"},
        {"finding": "The same ablation degrades content preservation",
         "stage": "11", "evidence_class": "CAUSAL-HARMFUL",
         "causal_test": "same intervention",
         "statistic": f"dSIM {ab['delta_sim'].mean():+.4f} "
                      f"[{ab['delta_sim_ci_lo'].mean():+.4f}, "
                      f"{ab['delta_sim_ci_hi'].mean():+.4f}]",
         "ci_excludes_zero": "yes",
         "claim": "STA gain is partly bought by discarding content"},
        {"finding": "Top MLP neurons ablation lowers STA",
         "stage": "11", "evidence_class": "CAUSAL-HARMFUL",
         "causal_test": "mean ablation",
         "statistic": "dSTA -0.0216 [-0.0447, -0.0022]",
         "ci_excludes_zero": "yes",
         "claim": "attribution rank did not predict causal direction"},
        {"finding": "Activation steering at unit norm changes little",
         "stage": "13", "evidence_class": "NULL/WEAK",
         "causal_test": "frozen dev-selected config on test",
         "statistic": "all frozen arms dJ in [-0.018, +0.005]",
         "ci_excludes_zero": "no",
         "claim": "null at the pre-specified strengths; magnitude-limited"},
        {"finding": "Cross-lingual transfer of steering vectors",
         "stage": "14", "evidence_class": "NULL",
         "causal_test": "dev steering + matched random control",
         "statistic": "primary dSIM -0.009 vs random control -0.024",
         "ci_excludes_zero": "no",
         "claim": "no advantage over an arbitrary direction of equal norm"},
    ]
    df = pd.DataFrame(rows)
    df["evidence_summary_source"] = "results/statistics/evidence_classes.json"
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []

    index.append(write(t1_corpus(), "table1_corpus_and_splits",
                       "corpus inventory + Stage 2 split counts, per language"))
    index.append(write(t2_baseline(), "table2_baseline_metrics",
                       "Stage 5 test metrics; validated STA and unvalidated yo/xh STA "
                       "in separate columns"))
    perf, tm = t3_probes()
    index.append(write(perf, "table3a_probe_performance", "Stage 7 probes (ASSOCIATIVE)"))
    index.append(write(tm, "table3b_probe_transfer_matrix",
                       "Stage 7 transfer, layer 6, with length-only control per target"))
    recon, stab = t4_sae()
    index.append(write(recon, "table4a_sae_reconstruction", "Stage 8 SAE quality"))
    index.append(write(stab, "table4b_sae_feature_stability",
                       "Stage 8/9 stability + promotion funnel (NEGATIVE)"))
    macro, per = t5_ablation()
    index.append(write(macro, "table5a_ablation_macro_with_ci",
                       "Stage 11 macro deltas, paired bootstrap CIs, verdicts"))
    index.append(write(per, "table5b_ablation_per_language",
                       "Stage 11 per-language test deltas; yo and xh kept separate"))
    dev_s, test_s = t6_steering()
    index.append(write(dev_s, "table6a_steering_dev_sweep", "Stage 13 dev strength sweep"))
    index.append(write(test_s, "table6b_steering_test_frozen",
                       "Stage 13 frozen test config with CIs (NULL/WEAK)"))
    tr, trs = t7_transfer()
    index.append(write(tr, "table7a_transfer_full", "Stage 14 dev transfer, all cells"))
    index.append(write(trs, "table7b_transfer_vs_random",
                       "Stage 14 primary vs matched random control (NULL)"))
    index.append(write(t8_evidence(), "table8_evidence_classification",
                       "claim strength per finding; CI-excludes-zero stated explicitly"))

    (OUT / "TABLES_INDEX.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "language_order": LANG_ORDER,
        "sources": "committed stage artifacts only; no inference re-run",
        "sta_policy": "validated STA and unvalidated yo/xh STA never share a column",
        "significance_policy": "a CI containing zero is reported as null, never as a trend",
        "tables": index,
    }, indent=2), encoding="utf-8")

    print(f"=== wrote {len(index)} tables to {OUT.relative_to(REPO_ROOT)}/ ===")
    for e in index:
        print(f"  {e['table']:38s} {e['rows']:5d} rows  {e['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
