#!/usr/bin/env python
"""Stage 9 -- Stability-aware SAE feature analysis, against Stage 7 probe directions.

Discovers candidate features on train, requires them to replicate across at least two of
three SAE seeds, confirms them on dev, and compares every candidate direction against the
linear-probe direction, the toxic-minus-reference mean difference, and language-only and
length-only control directions.

Usage
-----
    python scripts/09_analyze_sae_features.py --smoke   # verification pass, writes nothing
    python scripts/09_analyze_sae_features.py           # full run

Split roles
-----------
train  candidate discovery, all reference directions, and the SAE weights themselves.
dev    confirmation and every behavioural test.
test   NOT READ; a path guard raises, and the files opened are recorded in the output.

This tightens Stage 8 in one important way. Stage 8 ranked features by their *dev*
toxic-vs-reference AUC, which is selection on the confirmation split. Here ranking is
done on train and dev is only ever used to confirm, so a candidate that looks good purely
by chance on dev cannot be promoted by the ranking step.

Comparing SAE features to probe directions
------------------------------------------
The Stage 7 probes were fitted on *pooled* vectors standardised with pooled statistics,
while the SAEs consume *token* activations standardised with token statistics. Those are
different spaces, so a raw cosine between a Stage 7 coefficient vector and an SAE decoder
row would not mean anything. The probe is therefore refitted here in the SAE's own input
space -- same contrast, same train rows, token-level standardisation -- so the comparison
is like for like. The Stage 7 result stands as reported; this is a re-fit for geometry,
not a re-evaluation of that result.

Control directions
------------------
language-only  the one-vs-rest logistic direction for the language a feature most prefers.
length-only    the correlation between each standardised dimension and token count,
               normalised. A feature aligned with this is a length detector.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Replication requires a decoder-direction cosine of at least 0.7 against a feature in
  another seed. Counts at 0.5, 0.6, 0.7 and 0.8 are reported so the conclusion can be
  read independently of that choice.
* Matching uses decoder directions rather than activation correlations because the
  decoder row is what a later steering or ablation step would actually manipulate.
* The full behavioural battery runs at layer 6 for both sides and both token sets;
  replication counts are computed for all three layers. Layer 6 was the strongest probe
  site in Stage 7, and running the whole battery everywhere would multiply cost without
  changing the replication conclusion.
* Discovery keeps the top 50 features per seed and configuration by train AUC. A wider
  net mostly adds features whose train separation is indistinguishable from noise.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
ACT_DIR = REPO_ROOT / "results" / "activations"
SAE_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "results" / "sae_features"

SIDES = ["encoder", "decoder"]
LAYERS = [4, 6, 8]
TOKEN_SETS = ["all", "content"]
BATTERY_LAYER = 6
CONDITIONS = ["toxic", "detox_reference"]
TOP_N = 50
MATCH_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
REPLICATION_THRESHOLD = 0.7
MIN_SEEDS_FOR_REPLICATION = 2
PREFIX_LEN = 4

# Promotion criteria (item 10). All must hold.
CTRL_MAX_LANGUAGE_AUC = 0.90
CTRL_MAX_ABS_LENGTH_CORR = 0.30
CTRL_MAX_ABS_COS_CONTROL = 0.50
MIN_BEHAVIOUR_EFFECT = 0.15
MIN_CONTRAST_EFFECT = 0.15

_FILES_READ: list[str] = []


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard(split: str) -> None:
    if split == "test":
        raise RuntimeError("Stage 9 must not read test activations.")


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k, self.d_sae = k, d_sae
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))


def load_sae(side: str, layer: int, token_set: str, seed: int, d_model: int = 768):
    path = SAE_DIR / f"sae_{side}_layer{layer}" / f"sae_{token_set}_seed{seed}.pt"
    d = torch.load(path, weights_only=False)
    sae = TopKSAE(d_model, d["d_sae"], d["k"])
    sae.load_state_dict(d["state_dict"])
    return sae.eval()


def load_tokens(split: str, seed: int, side: str, layer: int, condition: str,
                token_set: str):
    guard(split)
    path = ACT_DIR / side / f"layer_{layer:02d}" / f"{split}_seed{seed}_{condition}.pt"
    if not path.exists():
        return None, None
    _FILES_READ.append(str(path.relative_to(REPO_ROOT)))
    d = torch.load(path, weights_only=False)
    toks, offs = d["tokens"], d["offsets"]
    skip = (PREFIX_LEN if side == "encoder" else 1) if token_set == "content" else 0
    idx, meta = [], []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        s2 = min(s + skip, e)
        if e - s2 <= 0:
            continue
        idx.append(torch.arange(s2, e))
        meta.append({"row": i, "pair_id": d["pair_id"][i], "language": d["language"][i],
                     "condition": condition, "n_tokens": e - s2, "row_n_tokens": e - s})
    sel = torch.cat(idx)
    m = pd.DataFrame(meta)
    m = m.loc[m.index.repeat(m["n_tokens"])].reset_index(drop=True)
    return toks[sel].to(torch.float32), m


def token_standardizer(seed: int, side: str, layer: int):
    means, varis, counts = [], [], []
    for cond in CONDITIONS:
        st = torch.load(ACT_DIR / "norm_stats" /
                        f"seed{seed}_{side}_layer{layer:02d}_{cond}.pt", weights_only=False)
        means.append(st["token_mean"].double())
        varis.append(st["token_std"].double() ** 2)
        counts.append(float(st["n_tokens"]))
    total = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / total
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / total
    return mu.float(), (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def build_matrix(split: str, seed: int, side: str, layer: int, token_set: str,
                 conditions: list[str]):
    mu, sd = token_standardizer(seed, side, layer)
    Xs, ms = [], []
    for cond in conditions:
        X, m = load_tokens(split, seed, side, layer, cond, token_set)
        if X is None:
            continue
        Xs.append((X - mu) / sd.clamp_min(1e-6))
        ms.append(m)
    if not Xs:
        return None, None, mu, sd
    return torch.cat(Xs), pd.concat(ms, ignore_index=True), mu, sd


@torch.no_grad()
def example_features(sae: TopKSAE, X: torch.Tensor, meta: pd.DataFrame, device: str):
    f = sae.encode(X.to(device)).cpu().numpy()
    key = meta[["row", "condition"]].copy()
    key["_i"] = np.arange(len(meta))
    grp = key.groupby(["row", "condition"], sort=True)["_i"].apply(list)
    mat = np.stack([f[i].mean(0) for i in grp.values])
    idx = pd.DataFrame(list(grp.index), columns=["row", "condition"]).merge(
        meta.groupby(["row", "condition"]).agg(
            language=("language", "first"), pair_id=("pair_id", "first"),
            n_tokens=("n_tokens", "first"), row_n_tokens=("row_n_tokens", "first")
        ).reset_index(), on=["row", "condition"], how="left")
    return mat, idx


def auc(y, s) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def balanced_index(idx: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    per = idx.groupby(["language", "condition"]).size()
    n = int(per.min())
    picks = []
    for (lang, cond), _ in per.items():
        sel = idx.index[(idx["language"] == lang) & (idx["condition"] == cond)].to_numpy()
        picks.append(rng.choice(sel, size=n, replace=False))
    return np.sort(np.concatenate(picks))


# ------------------------------------------------------------- reference directions

def reference_directions(X: torch.Tensor, meta: pd.DataFrame, languages: list[str]) -> dict:
    """Probe, mean-difference, language-only and length-only directions, from train."""
    Xn = X.numpy()
    y = (meta["condition"] == "detox_reference").to_numpy().astype(int)

    probe = LogisticRegression(C=0.1, max_iter=3000).fit(Xn, y)
    probe_dir = probe.coef_[0]

    mean_diff = Xn[y == 0].mean(0) - Xn[y == 1].mean(0)  # toxic minus reference

    lang_dirs = {}
    for lang in languages:
        yl = (meta["language"] == lang).to_numpy().astype(int)
        if yl.sum() == 0 or yl.sum() == len(yl):
            continue
        lang_dirs[lang] = LogisticRegression(
            C=0.1, max_iter=2000).fit(Xn, yl).coef_[0]

    length = meta["row_n_tokens"].to_numpy(dtype=float)
    length = (length - length.mean()) / (length.std() + 1e-8)
    length_dir = (Xn * length[:, None]).mean(0)

    def unit(v):
        v = np.asarray(v, dtype=float)
        return v / (np.linalg.norm(v) + 1e-12)

    return {"probe": unit(probe_dir), "mean_diff": unit(mean_diff),
            "language": {k: unit(v) for k, v in lang_dirs.items()},
            "length": unit(length_dir)}


# ------------------------------------------------------------------------ analysis

def analyse_config(side: str, layer: int, token_set: str, seeds: list[int],
                   cfg: dict, device: str, battery: bool) -> tuple[list, list]:
    languages = cfg["languages"]
    rng = np.random.default_rng(42)

    saes, train_feats, dirs = {}, {}, {}
    for seed in seeds:
        sae = load_sae(side, layer, token_set, seed).to(device)
        saes[seed] = sae
        Xtr, mtr, _, _ = build_matrix("train", seed, side, layer, token_set, CONDITIONS)
        mat, idx = example_features(sae, Xtr, mtr, device)
        y = (idx["condition"] == "detox_reference").to_numpy().astype(int)
        # DISCOVERY: train only.
        train_auc = np.array([auc(y, mat[:, j]) for j in range(mat.shape[1])])
        train_feats[seed] = {"train_auc": train_auc,
                             "top": np.argsort(-np.nan_to_num(np.abs(train_auc - 0.5)))[:TOP_N]}
        dirs[seed] = reference_directions(Xtr, mtr, languages)

    # ---- cross-seed replication on decoder directions ----
    W = {s: torch.nn.functional.normalize(saes[s].W_dec.detach().cpu(), dim=1)
         for s in seeds}
    repl_rows, battery_rows = [], []

    for seed in seeds:
        other = [s for s in seeds if s != seed]
        for j in train_feats[seed]["top"]:
            v = W[seed][j:j + 1]
            best = {o: float((v @ W[o].T).max()) for o in other}
            n_match = {t: sum(1 for o in other if best[o] >= t) for t in MATCH_THRESHOLDS}
            replicated = (n_match[REPLICATION_THRESHOLD] >= MIN_SEEDS_FOR_REPLICATION - 1)
            row = {
                "side": side, "layer": layer, "token_set": token_set, "seed": seed,
                "feature": int(j),
                "train_auc_toxic_vs_reference": float(train_feats[seed]["train_auc"][j]),
                **{f"best_cos_seed{o}": best[o] for o in other},
                **{f"n_seeds_matched_at_{t}": n_match[t] + 1 for t in MATCH_THRESHOLDS},
                "replicated": bool(replicated),
                "status": "replicated" if replicated else "seed_specific",
            }
            repl_rows.append(row)

    if not battery:
        return repl_rows, battery_rows

    # ---- confirmation battery on dev, for discovered candidates ----
    for seed in seeds:
        sae = saes[seed]
        Xdv, mdv, mu, sd = build_matrix("dev", seed, side, layer, token_set, CONDITIONS)
        dmat, didx = example_features(sae, Xdv, mdv, device)
        Xg, mg = load_tokens("dev", seed, side, layer, "generated", token_set)
        gmat = None
        if Xg is not None:
            gz = (Xg - mu) / sd.clamp_min(1e-6)
            gmat, gidx = example_features(sae, gz, mg, device)

        dev_rows = pd.read_csv(ACT_DIR / "index" / f"dev_seed{seed}_rows.csv")
        behaviour = dev_rows.set_index("pair_id")["behaviour_label"]
        tox_mask = (didx["condition"] == "toxic").to_numpy()
        lab = didx.loc[tox_mask, "pair_id"].map(behaviour).to_numpy()

        y_dev = (didx["condition"] == "detox_reference").to_numpy().astype(int)
        bal = balanced_index(didx, rng)
        Wn = W[seed]

        for j in train_feats[seed]["top"]:
            col = dmat[:, j]
            tox_col = col[tox_mask]
            copy_sel = np.isin(lab, ["copy_input", "ordinary_rewrite"])
            tmpl_sel = np.isin(lab, ["generic_template", "ordinary_rewrite"])

            lang_aucs = {l: auc((didx["language"] == l).to_numpy().astype(int), col)
                         for l in languages}
            per_lang_contrast = {}
            for l in languages:
                sel = (didx["language"] == l).to_numpy()
                per_lang_contrast[l] = auc(y_dev[sel], col[sel])

            fv = Wn[j].numpy()
            cos_lang = max(abs(float(fv @ dirs[seed]["language"][l]))
                           for l in dirs[seed]["language"]) if dirs[seed]["language"] else np.nan

            rec = {
                "side": side, "layer": layer, "token_set": token_set, "seed": seed,
                "feature": int(j),
                "train_auc_toxic_vs_reference": float(train_feats[seed]["train_auc"][j]),
                "dev_auc_toxic_vs_reference": auc(y_dev, col),
                "dev_auc_toxic_vs_reference_balanced": auc(y_dev[bal], col[bal]),
                "dev_auc_toxic_vs_generated": (
                    auc(np.r_[np.zeros(tox_mask.sum()), np.ones(gmat.shape[0])],
                        np.r_[tox_col, gmat[:, j]]) if gmat is not None else np.nan),
                "dev_auc_copy_vs_ordinary": auc(
                    (lab[copy_sel] == "copy_input").astype(int), tox_col[copy_sel]),
                "dev_auc_template_vs_ordinary": auc(
                    (lab[tmpl_sel] == "generic_template").astype(int), tox_col[tmpl_sel]),
                "dev_auc_language_max": float(max(lang_aucs.values())),
                "dev_language_argmax": max(lang_aucs, key=lang_aucs.get),
                "dev_corr_token_count": float(np.corrcoef(col, didx["n_tokens"])[0, 1]),
                "dev_corr_sequence_length": float(np.corrcoef(col, didx["row_n_tokens"])[0, 1]),
                "cos_with_probe_direction": float(fv @ dirs[seed]["probe"]),
                "cos_with_mean_diff_direction": float(fv @ dirs[seed]["mean_diff"]),
                "cos_with_language_control": float(cos_lang),
                "cos_with_length_control": float(fv @ dirs[seed]["length"]),
                **{f"dev_auc_contrast_{l}": per_lang_contrast[l] for l in languages},
            }
            battery_rows.append(rec)

    return repl_rows, battery_rows


def promote(df: pd.DataFrame) -> pd.DataFrame:
    """Item 10: replication AND control survival AND a real behavioural effect."""
    contrast = (df["dev_auc_toxic_vs_reference"] - 0.5).abs() >= MIN_CONTRAST_EFFECT
    controls = (
        (df["dev_auc_language_max"] < CTRL_MAX_LANGUAGE_AUC)
        & (df["dev_corr_token_count"].abs() < CTRL_MAX_ABS_LENGTH_CORR)
        & (df["dev_corr_sequence_length"].abs() < CTRL_MAX_ABS_LENGTH_CORR)
        & (df["cos_with_language_control"].abs() < CTRL_MAX_ABS_COS_CONTROL)
        & (df["cos_with_length_control"].abs() < CTRL_MAX_ABS_COS_CONTROL)
    )
    behaviour = (
        ((df["dev_auc_copy_vs_ordinary"] - 0.5).abs() >= MIN_BEHAVIOUR_EFFECT)
        | ((df["dev_auc_template_vs_ordinary"] - 0.5).abs() >= MIN_BEHAVIOUR_EFFECT)
        | ((df["dev_auc_toxic_vs_generated"] - 0.5).abs() >= MIN_BEHAVIOUR_EFFECT)
    )
    df = df.copy()
    df["passes_contrast"] = contrast
    df["passes_controls"] = controls
    df["passes_behaviour"] = behaviour
    df["passes_replication"] = df["replicated"]
    df["detoxification_feature"] = contrast & controls & behaviour & df["replicated"]
    df["label"] = np.where(
        df["detoxification_feature"],
        "detoxification-feature candidate (replicated + controlled + behavioural; "
        "causal status untested)",
        "not promoted")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seeds = cfg["seeds_multi"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("=== SMOKE TEST: encoder layer 6 content, writes nothing ===")
        repl, batt = analyse_config("encoder", 6, "content", seeds, cfg, device, battery=True)
        r, b = pd.DataFrame(repl), pd.DataFrame(batt)
        print(f"  candidates discovered on TRAIN: {len(r)} ({TOP_N} per seed x {len(seeds)})")
        print(f"  replicated at cos>={REPLICATION_THRESHOLD}: {int(r['replicated'].sum())}")
        print(f"  battery rows: {len(b)}; all finite AUCs: "
              f"{bool(np.isfinite(b['dev_auc_toxic_vs_reference']).all())}")
        print(f"  files read: {len(set(_FILES_READ))}, any test: "
              f"{any('test_' in f for f in _FILES_READ)}")
        try:
            load_tokens("test", 42, "encoder", 6, "toxic", "content")
            print("  TEST GUARD FAILED"); return 1
        except RuntimeError:
            print("  test guard active")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    all_repl, all_batt = [], []
    for side in SIDES:
        for layer in LAYERS:
            for token_set in TOKEN_SETS:
                battery = (layer == BATTERY_LAYER)
                repl, batt = analyse_config(side, layer, token_set, seeds, cfg,
                                            device, battery)
                all_repl += repl
                all_batt += batt
                n_rep = sum(1 for r in repl if r["replicated"])
                print(f"  {side} L{layer} {token_set}: {len(repl)} candidates, "
                      f"{n_rep} replicated")

    rdf = pd.DataFrame(all_repl)
    bdf = pd.DataFrame(all_batt)
    merged = bdf.merge(
        rdf[["side", "layer", "token_set", "seed", "feature", "replicated", "status"]
            + [f"n_seeds_matched_at_{t}" for t in MATCH_THRESHOLDS]],
        on=["side", "layer", "token_set", "seed", "feature"], how="left")
    promoted = promote(merged)

    rdf.to_csv(OUT_DIR / "replication.csv", index=False)
    promoted.to_csv(OUT_DIR / f"layer{BATTERY_LAYER}_feature_stats.csv", index=False)
    promoted[promoted["replicated"]].to_csv(OUT_DIR / "replicated_features.csv", index=False)
    promoted[~promoted["replicated"]].to_csv(OUT_DIR / "seed_specific_features.csv", index=False)

    (OUT_DIR / "stage9_meta.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_split": "train", "confirmation_split": "dev", "test_split": "NOT READ",
        "top_n_per_seed": TOP_N,
        "replication_threshold_cosine": REPLICATION_THRESHOLD,
        "min_seeds_for_replication": MIN_SEEDS_FOR_REPLICATION,
        "promotion_criteria": {
            "contrast": f"|dev AUC toxic-vs-reference - 0.5| >= {MIN_CONTRAST_EFFECT}",
            "controls": f"language AUC < {CTRL_MAX_LANGUAGE_AUC}, |length corr| < "
                        f"{CTRL_MAX_ABS_LENGTH_CORR}, |cos with control dirs| < "
                        f"{CTRL_MAX_ABS_COS_CONTROL}",
            "behaviour": f"|AUC - 0.5| >= {MIN_BEHAVIOUR_EFFECT} on at least one of "
                         "copy-vs-ordinary, template-vs-ordinary, toxic-vs-generated",
            "replication": f">= {MIN_SEEDS_FOR_REPLICATION} of 3 seeds",
        },
        "files_read": sorted(set(_FILES_READ)),
        "any_test_file_read": any("test_" in f for f in _FILES_READ),
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    print("\n=== replication across seeds (candidates discovered on TRAIN) ===")
    print(rdf.groupby(["side", "layer", "token_set"]).agg(
        candidates=("feature", "size"),
        replicated=("replicated", "sum")).to_string())

    print("\n=== replication count sensitivity to the matching threshold ===")
    sens = {f"cos>={t}": int((rdf[f"n_seeds_matched_at_{t}"] >= MIN_SEEDS_FOR_REPLICATION).sum())
            for t in MATCH_THRESHOLDS}
    print(f"  candidates replicating in >=2 of 3 seeds: {sens} (of {len(rdf)} candidates)")

    print(f"\n=== promotion funnel (layer {BATTERY_LAYER} battery, n={len(promoted)}) ===")
    for stage, col in [("separates toxic vs reference on dev", "passes_contrast"),
                       ("survives language/length controls", "passes_controls"),
                       ("predicts a behaviour distinction", "passes_behaviour"),
                       ("replicates in >=2 of 3 seeds", "passes_replication"),
                       ("ALL FOUR -> detoxification feature", "detoxification_feature")]:
        print(f"  {stage:42s}: {int(promoted[col].sum()):4d}")

    n_final = int(promoted["detoxification_feature"].sum())
    if n_final == 0:
        print("\n  *** NEGATIVE RESULT ***")
        print("  No SAE feature satisfies all four criteria. No feature is called a")
        print("  detoxification feature. The binding constraint is reported below.")
        blockers = {
            "replication": int((~promoted["passes_replication"]).sum()),
            "controls": int((~promoted["passes_controls"]).sum()),
            "behaviour": int((~promoted["passes_behaviour"]).sum()),
            "contrast": int((~promoted["passes_contrast"]).sum()),
        }
        print(f"  candidates failing each criterion: {blockers}")
    else:
        print(f"\n  {n_final} features promoted; see replicated_features.csv")

    print("\n=== direction comparison (mean |cosine| over layer-6 candidates) ===")
    print(promoted.groupby(["side", "token_set"])[
        ["cos_with_probe_direction", "cos_with_mean_diff_direction",
         "cos_with_language_control", "cos_with_length_control"]
    ].apply(lambda g: g.abs().mean()).round(3).to_string())

    print("\n=== behavioural discrimination, mean |AUC-0.5| (layer 6) ===")
    print(promoted.groupby("side")[
        ["dev_auc_toxic_vs_reference", "dev_auc_toxic_vs_generated",
         "dev_auc_copy_vs_ordinary", "dev_auc_template_vs_ordinary"]
    ].apply(lambda g: (g - 0.5).abs().mean()).round(3).to_string())

    print("\n=== per-language toxic-vs-reference AUC (mean over candidates, layer 6) ===")
    lang_cols = [c for c in promoted.columns if c.startswith("dev_auc_contrast_")]
    per_lang = promoted.groupby("side")[lang_cols].apply(lambda g: (g - 0.5).abs().mean())
    per_lang.columns = [c.replace("dev_auc_contrast_", "") for c in per_lang.columns]
    print(per_lang.round(3).to_string())

    print(f"\n  any test file read: {any('test_' in f for f in _FILES_READ)}")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
