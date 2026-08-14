#!/usr/bin/env python
"""Stage 17 -- Paper figures, built only from committed artifacts.

Writes PNG + PDF under results/figures/. No model is loaded, no inference is run, and no
prior artifact is modified.

Visual encoding contract
------------------------
The encoding carries claim strength, so a reader cannot mistake one kind of evidence for
another:

  validated STA        solid fill, categorical slot 1 (blue)
  UNVALIDATED yo/xh    never drawn on a STA axis. Where a STA panel would show them, a
                       hatched neutral placeholder is drawn and annotated
                       "STA unavailable" instead of a number. Their SIM/FL/diagnostics
                       are plotted normally, since those metrics *are* valid.
  associative          open marker, dashed edge
  causal               solid marker, solid edge, error bars
  null / negative      neutral grey, reduced emphasis
  random control       hatched, always drawn beside the arm it controls

Palette is the validated reference instance (slots 1-4: blue, orange, aqua, yellow),
checked with the skill's validator: lightness band, chroma floor, adjacent CVD separation
and normal-vision floor all PASS. The contrast WARN on the lighter slots is discharged the
way the rule requires -- every figure carries direct labels or a companion table under
results/tables/.

Language order is fixed at am, ar, de, en, es, hi, uk, yo, xh everywhere, so a language
occupies the same position in every figure.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R = REPO_ROOT / "results"
OUT = R / "figures"

LANG_ORDER = ["am", "ar", "de", "en", "es", "hi", "uk", "yo", "xh"]
AFRICAN = {"yo", "xh"}

# Validated categorical slots (light mode) from the skill's reference palette.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_GREY, C_INK, C_MUTED = "#8a8a85", "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.labelcolor": C_MUTED, "text.color": C_INK, "legend.frameon": False,
})

MANIFEST: list[dict] = []


def save(fig, name: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    for p in paths:
        MANIFEST.append({
            "figure": name, "file": str(p.relative_to(REPO_ROOT)),
            "bytes": p.stat().st_size,
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            "caption": caption,
        })


def order_langs(df, col="language"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=LANG_ORDER, ordered=True)
    return df.sort_values(col)


# ------------------------------------------------------------------------ figures

def fig1_overview():
    t = pd.read_csv(R / "tables" / "table1_corpus_and_splits.csv")
    t = order_langs(t)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(t))
    cols = [C_ORANGE if l in AFRICAN else C_BLUE for l in t["language"]]

    ax[0].bar(x, t["corpus_rows"], color=cols, width=0.68)
    for i, v in enumerate(t["corpus_rows"]):
        ax[0].text(i, v + 8, str(int(v)), ha="center", fontsize=7, color=C_MUTED)
    ax[0].set_title("Corpus size (rows)", loc="left", fontsize=10)
    ax[0].set_ylim(0, max(t["corpus_rows"]) * 1.18)

    ax[1].bar(x, t["mt5_fertility_toxic"], color=cols, width=0.68)
    for i, v in enumerate(t["mt5_fertility_toxic"]):
        ax[1].text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=7, color=C_MUTED)
    ax[1].set_title("mT5 fertility (subwords / word)", loc="left", fontsize=10)
    ax[1].set_ylim(0, max(t["mt5_fertility_toxic"]) * 1.2)

    ax[2].bar(x, t["detox_over_toxic_len"], color=cols, width=0.68)
    ax[2].axhline(1.0, color=C_INK, lw=1, ls="--")
    ax[2].text(0.1, 1.02, "equal length", fontsize=7, color=C_MUTED)
    for i, v in enumerate(t["detox_over_toxic_len"]):
        ax[2].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=C_MUTED)
    ax[2].set_title("Reference / input length ratio", loc="left", fontsize=10)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(t["language"])
    fig.legend(handles=[mpatches.Patch(color=C_BLUE, label="comparison languages"),
                        mpatches.Patch(color=C_ORANGE, label="African languages (yo, xh)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Fig 1 — Corpus, tokenisation and annotation-style overview",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    save(fig, "fig1_language_data_overview",
         "Corpus size, mT5 fertility, and reference/input length ratio. yo/xh invert the "
         "length ratio, an annotation-style difference that confounds later contrasts.")


def fig2_probe_transfer():
    t = pd.read_csv(R / "tables" / "table3b_probe_transfer_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, side in zip(axes, ["decoder", "encoder"]):
        s = t[t["side"] == side]
        srcs = ["within_language", "EN_DE_ES", "all_non_african", "YO_XH"]
        piv = s.pivot_table(index="source", columns="target",
                            values="dev_accuracy_mean", aggfunc="mean")
        piv = piv.reindex(index=[i for i in srcs if i in piv.index],
                          columns=[c for c in LANG_ORDER if c in piv.columns])
        im = ax.imshow(piv.values, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.75 else C_INK)
        ax.set_title(f"{side} · layer 6", loc="left", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="dev accuracy")
    fig.suptitle("Fig 2 — Probe transfer matrix  [ASSOCIATIVE evidence: no causal test]",
                 x=0.02, ha="left", fontsize=11)
    save(fig, "fig2_probe_transfer_matrix",
         "Toxic-vs-reference probe accuracy, source group to target language. Encoder "
         "transfer into yo/xh sits at chance. Associative only.")


def fig3_sae_stability():
    s = pd.read_csv(R / "tables" / "table4b_sae_feature_stability.csv")
    rec = pd.read_csv(R / "tables" / "table4a_sae_reconstruction.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    s["cfg"] = s["side"].str[:3] + " L" + s["layer"].astype(str) + " " + s["token_set"]
    x = np.arange(len(s))
    ax[0].bar(x, s["mean_max_cosine"], color=C_GREY, width=0.68)
    ax[0].bar(x, s["frac_matched_above_0_7"], color=C_ORANGE, width=0.36,
              label="fraction matched > 0.7")
    ax[0].axhline(0.7, color=C_INK, ls="--", lw=1)
    ax[0].text(0.05, 0.72, "replication threshold 0.7", fontsize=7, color=C_MUTED)
    for i, v in enumerate(s["frac_matched_above_0_7"]):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.5, color=C_MUTED)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(s["cfg"], rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("cosine / fraction")
    ax[0].set_title("Cross-seed feature stability  [NEGATIVE]", loc="left", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper right")

    rec["cfg"] = rec["side"].str[:3] + " L" + rec["layer"].astype(str)
    for side, c in (("encoder", C_BLUE), ("decoder", C_AQUA)):
        d = rec[rec["side"] == side]
        ax[1].scatter(d["dev_explained_variance"], d["dev_dead_feature_rate"],
                      s=70, color=c, label=side, edgecolor="white", linewidth=1.5)
        for _, r in d.iterrows():
            ax[1].annotate(f"L{int(r['layer'])}", (r["dev_explained_variance"],
                                                   r["dev_dead_feature_rate"]),
                           fontsize=6.5, color=C_MUTED,
                           xytext=(4, 3), textcoords="offset points")
    ax[1].set_xlabel("dev explained variance")
    ax[1].set_ylabel("dead-feature rate")
    ax[1].set_title("SAE reconstruction vs dead features", loc="left", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("Fig 3 — SAE feature stability and reconstruction  "
                 "[NEGATIVE: 0 of 600 candidates promoted]",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig3_sae_feature_stability",
         "Under 1.5% of SAE features find a cross-seed match above cosine 0.7. Negative "
         "result: no feature survived replication, controls and behaviour together.")


def fig4_attribution():
    t = pd.read_csv(R / "tables" if False else R / "attribution" / "top_components.csv")
    rc = pd.read_csv(R / "attribution" / "random_controls.csv")
    t = t[(t["direction"] == "probe") & (t["contrast"] == "toxic_vs_reference")]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for a, kind, col in ((ax[0], "head", C_BLUE), (ax[1], "neuron", C_AQUA)):
        d = t[t["kind"] == kind].head(12).iloc[::-1]
        y = np.arange(len(d))
        a.barh(y, d["dev_d_tox_ref"].abs(), color=col, height=0.62)
        p95 = rc[(rc["direction"] == "probe") & (rc["kind"] == kind)]["random_p95_abs_d"].mean()
        a.axvline(p95, color=C_ORANGE, ls="--", lw=1.4)
        a.text(p95, len(d) - 0.4, " random p95", fontsize=7, color=C_ORANGE)
        a.set_yticks(y)
        a.set_yticklabels(d["component"], fontsize=7)
        for i, v in enumerate(d["dev_d_tox_ref"].abs()):
            a.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=6.5, color=C_MUTED)
        a.set_xlabel("|Cohen's d| (dev, toxic vs reference)")
        a.set_title(f"top {kind}s", loc="left", fontsize=10)
    fig.suptitle("Fig 4 — Component attribution ranking  "
                 "[ASSOCIATIVE: attribution is not causal necessity]",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig4_attribution_rankings",
         "Top components by gradient x activation attribution, against the matched "
         "random 95th percentile. Ranking only; Stage 11 supplies the causal test.")


def fig5_ablation():
    m = pd.read_csv(R / "statistics" / "macro_intervention_stats.csv")
    a = m[(m["stage"] == "stage11_ablation") & (m["split"] == "test")]
    g = a.groupby(["intervention", "family"]).agg(
        d=("delta_sta", "mean"), lo=("delta_sta_ci_lo", "mean"),
        hi=("delta_sta_ci_hi", "mean"), dsim=("delta_sim", "mean")).reset_index()
    g = g.sort_values("d")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(g))
    for i, (_, r) in enumerate(g.iterrows()):
        sig = not (r["lo"] <= 0 <= r["hi"])
        is_ctrl = r["family"].endswith("control")
        col = C_GREY if not sig else (C_ORANGE if r["d"] < 0 else C_BLUE)
        ax.barh(i, r["d"], color=col, height=0.6,
                hatch="///" if is_ctrl else None, edgecolor="white", linewidth=1.2)
        ax.plot([r["lo"], r["hi"]], [i, i], color=C_INK, lw=1.4)
        ax.plot([r["lo"], r["lo"]], [i - .12, i + .12], color=C_INK, lw=1.4)
        ax.plot([r["hi"], r["hi"]], [i - .12, i + .12], color=C_INK, lw=1.4)
        lbl = "significant" if sig else "null"
        ax.text(r["hi"] + 0.006, i, f"{r['d']:+.3f}  ({lbl})", va="center",
                fontsize=7, color=C_INK if sig else C_MUTED)
    ax.axvline(0, color=C_INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(g["intervention"], fontsize=8)
    ax.set_xlabel("Δ STA vs unablated baseline (validated languages, paired 95% CI)")
    ax.set_xlim(-0.09, 0.22)
    handles = [mpatches.Patch(color=C_BLUE, label="causal, CI excludes zero"),
               mpatches.Patch(color=C_ORANGE, label="causal-harmful (significant, wrong sign)"),
               mpatches.Patch(color=C_GREY, label="null (CI includes zero)"),
               mpatches.Patch(facecolor="white", edgecolor=C_MUTED, hatch="///",
                              label="matched control")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    ax.set_title("Fig 5 — Ablation effect sizes with matched random controls  [CAUSAL]",
                 loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_ablation_effects",
         "Test-split ΔSTA per ablation arm with paired bootstrap CIs. All four controls "
         "are null; three primary arms exclude zero.")


def fig6_steering_curves():
    d = pd.read_csv(R / "steering_dev.csv")
    d = d[d["arm"].isin(["primary", "comparison"])
          & (d["scope"] == "pooled_all_balanced")
          & (d["direction_type"] == "mean_diff")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    val = d[~d["language"].isin(AFRICAN)]
    afr = d[d["language"].isin(AFRICAN)]

    def across_seed(frame, metric, side):
        """Macro-average over languages per seed, then spread ACROSS SEEDS.

        Error bars must express uncertainty, not between-language heterogeneity. Taking
        the standard deviation over languages directly would draw bars spanning most of
        the axis purely because Amharic and German differ, which reads as though the
        curve were unmeasurable when the seed-to-seed uncertainty is in fact small.
        """
        s = frame[frame["side"] == side]
        per_seed = s.groupby(["strength", "seed"])[metric].mean().reset_index()
        return per_seed.groupby("strength")[metric].agg(["mean", "std"])

    for ax, metric, title in zip(axes, ["sta", "sim", "fl"],
                                 ["STA (validated languages only)", "SIM", "FL (chrF1)"]):
        src = val if metric == "sta" else d[~d["language"].isin(AFRICAN)]
        for side, c in (("decoder", C_BLUE), ("encoder", C_ORANGE)):
            s = across_seed(src, metric, side)
            ax.errorbar(s.index, s["mean"], yerr=s["std"], color=c, marker="o",
                        ms=5, lw=2, capsize=3, label=f"{side} (validated langs)")
        if metric != "sta":
            for side, c in (("decoder", C_AQUA), ("encoder", C_YELLOW)):
                s = across_seed(afr, metric, side)
                ax.errorbar(s.index, s["mean"], yerr=s["std"], color=c, ls="--",
                            marker="s", ms=4, lw=1.6, capsize=3,
                            label=f"{side} (yo/xh)")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("steering strength α")
        ax.margins(y=0.18)
    axes[0].text(0.03, 0.06, "yo/xh excluded:\nSTA unavailable", transform=axes[0].transAxes,
                 fontsize=7, color=C_MUTED,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f2f1ec", ec="#d9d8d3"))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Fig 6 — Steering strength curves, dev (error bars = across-seed sd)  "
                 "[NULL/WEAK]", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    save(fig, "fig6_steering_strength_curves",
         "Dev strength sweep for the pooled all-language mean-difference vector. Decoder "
         "is flat; encoder degrades SIM/FL at larger |α|. yo/xh omitted from the STA panel.")


def fig7_tradeoff():
    t = pd.read_csv(R / "tables" / "table2_baseline_metrics.csv")
    t = order_langs(t)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    v = t[t["sta_status"] == "validated"]
    sc = ax[0].scatter(v["sim"], v["sta_validated"], s=110, c=v["fl"], cmap="Blues",
                       vmin=0.4, vmax=0.85, edgecolor=C_INK, linewidth=0.8, zorder=3)
    for _, r in v.iterrows():
        ax[0].annotate(r["language"], (r["sim"], r["sta_validated"]), fontsize=8,
                       xytext=(6, -3), textcoords="offset points")
    # Margins so the point labels are not clipped at the axis edges.
    ax[0].margins(x=0.16, y=0.14)
    fig.colorbar(sc, ax=ax[0], label="FL (chrF1)", shrink=0.85)
    ax[0].set_xlabel("SIM (content preservation)")
    ax[0].set_ylabel("STA (validated)")
    ax[0].set_title("Validated languages: STA vs SIM", loc="left", fontsize=10)

    a = t[t["sta_status"] != "validated"]
    x = np.arange(len(a))
    w = 0.26
    ax[1].bar(x - w, a["sim"], w, color=C_BLUE, label="SIM")
    ax[1].bar(x, a["fl"], w, color=C_AQUA, label="FL")
    ax[1].bar(x + w, a["top_output_share"], w, color=C_ORANGE, label="top-output share")
    for i, r in enumerate(a.itertuples()):
        for off, val in ((-w, r.sim), (0, r.fl), (w, r.top_output_share)):
            ax[1].text(i + off, val + 0.012, f"{val:.2f}", ha="center", fontsize=7,
                       color=C_MUTED)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{l}\nSTA unavailable" for l in a["language"]], fontsize=8)
    ax[1].set_ylim(0, 0.75)
    ax[1].legend(fontsize=7.5)
    ax[1].set_title("yo / xh: STA not plotted (unvalidated)", loc="left", fontsize=10)
    fig.suptitle("Fig 7 — STA / SIM / FL trade-off  "
                 "[validated and unvalidated shown on separate axes]",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "fig7_sta_sim_fl_tradeoff",
         "Left: validated languages only. Right: yo/xh diagnostics with STA deliberately "
         "omitted, since the classifier has no validated coverage for them.")


def fig8_transfer_vs_random():
    d = pd.read_csv(R / "transfer" / "transfer_dev.csv")
    d = d[d["target_language"].isin(AFRICAN)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, lang in zip(axes, ["yo", "xh"]):
        s = d[d["target_language"] == lang]
        for vt, c, ls, lbl in (("mean_diff", C_BLUE, "-", "transfer vector"),
                               ("random", C_GREY, "--", "matched random control")):
            g = s[s["vector_type"] == vt].groupby("strength")["sim"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=c, ls=ls, marker="o",
                        ms=5, lw=2, capsize=3, label=lbl)
        ax.set_title(f"{lang} — SIM (STA unavailable)", loc="left", fontsize=10)
        ax.set_xlabel("steering strength α")
    axes[0].set_ylabel("SIM")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 8 — Cross-lingual transfer into yo/xh vs matched random  [NULL]",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "fig8_transfer_vs_random",
         "Transfer vectors give no SIM advantage over a matched random vector of equal "
         "norm at any strength. Null result; STA omitted for both languages.")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(REPO_ROOT)): hashlib.md5(p.read_bytes()).hexdigest()
              for p in sorted(R.rglob("*.csv")) if "figures" not in str(p)}

    fig1_overview(); fig2_probe_transfer(); fig3_sae_stability(); fig4_attribution()
    fig5_ablation(); fig6_steering_curves(); fig7_tradeoff(); fig8_transfer_vs_random()

    after = {str(p.relative_to(REPO_ROOT)): hashlib.md5(p.read_bytes()).hexdigest()
             for p in sorted(R.rglob("*.csv")) if "figures" not in str(p)}
    changed = [k for k in before if before.get(k) != after.get(k)]

    (OUT / "FIGURES_INDEX.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "language_order": LANG_ORDER,
        "palette": {"source": "dataviz reference palette (validated)",
                    "slots": [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]},
        "encoding_contract": {
            "validated_sta": "solid fill, blue",
            "unvalidated_yo_xh_sta": "never drawn on a STA axis; annotated as unavailable",
            "associative": "heatmap/ranking panels labelled ASSOCIATIVE",
            "causal": "solid fill with paired 95% CI error bars",
            "null_negative": "neutral grey, reduced emphasis",
            "random_control": "hatched, drawn beside the arm it controls"},
        "prior_artifacts_modified": changed,
        "figures": MANIFEST,
    }, indent=2), encoding="utf-8")

    print(f"=== wrote {len(MANIFEST)//2} figures ({len(MANIFEST)} files) ===")
    for e in MANIFEST:
        if e["file"].endswith(".png"):
            print(f"  {e['figure']:34s} {e['bytes']:8d} B  {e['md5'][:12]}")
    print(f"\nfile-integrity check: prior CSV artifacts modified = "
          f"{len(changed)} {changed if changed else '(none)'}")
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
