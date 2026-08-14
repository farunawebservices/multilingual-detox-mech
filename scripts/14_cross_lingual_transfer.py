#!/usr/bin/env python
"""Stage 14 -- Cross-lingual transfer of steering vectors. Dev only.

Tests H2 directly: does a steering direction built from higher-resource languages work on
Yoruba and isiXhosa, and does a direction built from those two work on the comparison
languages? Every vector is built from train, every evaluation runs on dev, and no Stage 13
test artifact is read or written.

Usage
-----
    python scripts/14_cross_lingual_transfer.py --smoke   # verification pass, writes nothing
    python scripts/14_cross_lingual_transfer.py           # full dev transfer sweep

Relationship to earlier stages
------------------------------
Injection and scoring are imported from scripts/13_generate_steered.py rather than
reimplemented, so a transfer number here is directly comparable to a Stage 13 number
instead of differing by some quiet detail of hook placement or metric definition.

The {EN, DE, ES} source group does not exist in the Stage 12 vector library, which built
per-language, pooled_non_african, pooled_all_balanced and pooled_african scopes only. It
is therefore constructed here from train activations by the same procedure. The two
groups that *do* overlap Stage 12 -- all non-African, and {YO, XH} -- are rebuilt too and
checked against the stored vectors, so any drift in the construction path would surface
as a cosine below 1 rather than passing unnoticed.

Split discipline
----------------
train  source-group vector construction only.
dev    every evaluation reported here.
test   NOT READ and NOT WRITTEN. A path guard raises on any test path, and the Stage 13
       test outputs are never opened; Stage 14 writes only under results/transfer/.

Strength is swept and reported in full rather than selected. Nothing in this stage picks a
strength, so no target-language result -- dev or test -- can leak into a choice.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Both encoder and decoder layer 6 are swept. Stage 11 found the causal ablation effect
  in the decoder, while Stage 13's dev selection preferred encoder vectors for eight of
  nine languages; reporting one side only would presuppose the answer.
* Language-specific vectors applied to their own language serve as the within-language
  reference row. H2 is a claim about transfer being *worse* than within-language steering,
  which is unfalsifiable without that reference measured the same way.
* Matched random controls use the same norm and dimensionality and are swept at every
  strength, so a transfer effect can be read against what an arbitrary direction of the
  same magnitude does at that strength.
* STA is reported for the seven covered languages only. For yo/xh the report carries SIM,
  FL, copy rate, template rate, unique-output rate and top-output share, and the STA
  column stays empty rather than being filled with an unvalidated number.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, MT5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
ACT_DIR = REPO_ROOT / "results" / "activations"
VEC_DIR = REPO_ROOT / "results" / "steering_vectors"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
OUT_DIR = REPO_ROOT / "results" / "transfer"

LAYER = 6
POOLING = "pooled_mean_content"
SIDES = ["encoder", "decoder"]
VECTOR_TYPES = ["mean_diff", "probe", "random"]
STRENGTHS = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
AFRICAN = ["yo", "xh"]
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}
PROBE_C = 0.1


def _load_stage13():
    spec = importlib.util.spec_from_file_location(
        "stage13", REPO_ROOT / "scripts" / "13_generate_steered.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S13 = _load_stage13()


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard(path_or_split: str) -> None:
    if "test" in str(path_or_split).lower():
        raise RuntimeError(
            f"Stage 14 is dev-only and must not touch test ('{path_or_split}'). "
            "The Stage 13 test artifacts are frozen and are neither read nor written.")


def source_groups(cfg: dict) -> dict[str, list[str]]:
    non_afr = [l for l in cfg["languages"] if l not in AFRICAN]
    return {
        "EN_DE_ES": ["en", "de", "es"],
        "all_non_african": non_afr,
        "YO_XH": list(AFRICAN),
    }


def target_for(source_name: str, cfg: dict) -> list[str]:
    non_afr = [l for l in cfg["languages"] if l not in AFRICAN]
    return non_afr if source_name == "YO_XH" else list(AFRICAN)


# ------------------------------------------------------------ vector construction

def pooled_train(seed: int, side: str):
    """Standardised pooled train activations plus language and condition labels."""
    guard("train")  # no-op, documents intent
    mu, sd = S13_pooled_stats(seed, side)
    mats, langs, labels = [], [], []
    for label, cond in enumerate(("toxic", "detox_reference")):
        p = ACT_DIR / side / f"layer_{LAYER:02d}" / f"train_seed{seed}_{cond}.pt"
        d = torch.load(p, weights_only=False)
        z = (d[POOLING] - mu) / sd.clamp_min(1e-6)
        mats.append(z.numpy())
        langs.append(np.array(d["language"]))
        labels.append(np.full(len(z), label))
    return np.vstack(mats), np.concatenate(langs), np.concatenate(labels)


def S13_pooled_stats(seed: int, side: str):
    means, varis, counts = [], [], []
    for cond in ("toxic", "detox_reference"):
        st = torch.load(ACT_DIR / "norm_stats" /
                        f"seed{seed}_{side}_layer{LAYER:02d}_{cond}.pt", weights_only=False)
        means.append(st[f"{POOLING}_mean"].double())
        varis.append(st[f"{POOLING}_std"].double() ** 2)
        counts.append(float(st["n_rows"]))
    tot = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / tot
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / tot
    return mu.float(), (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def build_source_vectors(seed: int, side: str, cfg: dict,
                         rng: np.random.Generator) -> tuple[dict, list[dict]]:
    X, langs, y = pooled_train(seed, side)
    vecs, recs = {}, []

    scopes = dict(source_groups(cfg))
    scopes.update({f"lang_{l}": [l] for l in cfg["languages"]})

    for scope, members in scopes.items():
        sel = np.isin(langs, members)
        Xs, ys = X[sel], y[sel]
        n_tox, n_ref = int((ys == 0).sum()), int((ys == 1).sum())
        for vtype in ("mean_diff", "probe"):
            if vtype == "mean_diff":
                raw = Xs[ys == 0].mean(0) - Xs[ys == 1].mean(0)
            else:
                raw = -LogisticRegression(C=PROBE_C, max_iter=3000).fit(Xs, ys).coef_[0]
            vecs[(scope, vtype)] = unit(raw)
            recs.append({"scope": scope, "vector_type": vtype, "side": side, "seed": seed,
                         "layer": LAYER, "pooling": POOLING,
                         "source_languages": "|".join(members),
                         "n_rows_toxic": n_tox, "n_rows_reference": n_ref,
                         "underdetermined": bool(n_tox + n_ref < Xs.shape[1]),
                         "built_from": "train"})
        vecs[(scope, "random")] = unit(rng.normal(size=X.shape[1]))
        recs.append({"scope": scope, "vector_type": "random", "side": side, "seed": seed,
                     "layer": LAYER, "pooling": POOLING,
                     "source_languages": "|".join(members),
                     "n_rows_toxic": -1, "n_rows_reference": -1,
                     "underdetermined": False, "built_from": "matched random control"})
    return vecs, recs


def consistency_check(vecs: dict, seed: int, side: str) -> list[dict]:
    """Rebuilt groups must reproduce the Stage 12 stored vectors."""
    stored = dict(np.load(VEC_DIR / "vectors.npz"))
    out = []
    for scope, s12 in (("all_non_african", "pooled_non_african"),
                       ("YO_XH", "pooled_african")):
        for vtype in ("mean_diff", "probe"):
            key = f"{vtype}|{s12}|{side}|L{LAYER}|{POOLING}|seed{seed}"
            if key in stored and (scope, vtype) in vecs:
                out.append({"seed": seed, "side": side, "scope": scope,
                            "vector_type": vtype, "stage12_scope": s12,
                            "cosine_vs_stage12": float(np.dot(vecs[(scope, vtype)],
                                                              stored[key]))})
    return out


# ---------------------------------------------------------------------- sweeping

def run(cfg: dict, device: str, smoke: bool = False) -> tuple[pd.DataFrame, list, list]:
    dev = pd.read_csv(REPO_ROOT / "data" / "splits" / "dev.csv").rename(
        columns={"detox_output": "detox_reference"})
    seeds = cfg["seeds_multi"][:1] if smoke else cfg["seeds_multi"]
    sides = ["decoder"] if smoke else SIDES
    strengths = [0.5] if smoke else [s for s in STRENGTHS if s != 0.0]

    frames, vec_records, checks = [], [], []
    rng = np.random.default_rng(42)

    for seed in seeds:
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / f"seed{seed}"))
        model = MT5ForConditionalGeneration.from_pretrained(
            str(MODEL_DIR / f"seed{seed}")).to(device).eval()

        for side in sides:
            vecs, recs = build_source_vectors(seed, side, cfg, rng)
            vec_records += recs
            checks += consistency_check(vecs, seed, side)
            _, sd = S13_pooled_stats(seed, side)

            # unsteered baseline for every target language
            base = S13.generate_by_language(model, tok, dev, device, side,
                                            {}, None, sd, 0.0)
            for src in list(source_groups(cfg)) + ["within_language"]:
                tgts = (cfg["languages"] if src == "within_language"
                        else target_for(src, cfg))
                b = base[base["language"].isin(tgts)].copy()
                b["source_group"], b["vector_type"] = src, "baseline"
                b["side"], b["strength"], b["seed"] = side, 0.0, seed
                frames.append(b)

            for src in list(source_groups(cfg)) + ["within_language"]:
                tgts = (cfg["languages"] if src == "within_language"
                        else target_for(src, cfg))
                sub = dev[dev["language"].isin(tgts)]
                for vtype in (["mean_diff"] if smoke else VECTOR_TYPES):
                    if src == "within_language":
                        vfl = (lambda l, vt=vtype: (f"lang_{l}", vt))
                    else:
                        vfl = (lambda l, s=src, vt=vtype: (s, vt))
                    for alpha in strengths:
                        rec = S13.generate_by_language(
                            model, tok, sub, device, side,
                            {k: v for k, v in vecs.items()}, vfl, sd, alpha)
                        rec["source_group"], rec["vector_type"] = src, vtype
                        rec["side"], rec["strength"], rec["seed"] = side, alpha, seed
                        frames.append(rec)
                    print(f"  seed{seed} {side} {src}->{'/'.join(tgts[:3])}... "
                          f"{vtype}: {len(strengths)} strengths")
        del model
        torch.cuda.empty_cache()

    return pd.concat(frames, ignore_index=True), vec_records, checks


def summarise(scored: pd.DataFrame, train_freq: dict) -> pd.DataFrame:
    agg = S13.aggregate(
        scored.assign(arm="transfer", direction_type=scored["vector_type"],
                      scope=scored["source_group"]), train_freq)
    agg = agg.rename(columns={"scope": "source_group", "direction_type": "vector_type",
                              "language": "target_language"})
    agg["layer"], agg["pooling"], agg["split"] = LAYER, POOLING, "dev"
    agg["sta_status"] = np.where(agg["target_language"].isin(AFRICAN),
                                 "unavailable/unvalidated", "validated")
    # STA is reported only where coverage is validated.
    for col in ("sta", "delta_sta", "j"):
        agg.loc[agg["target_language"].isin(AFRICAN), col] = np.nan
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_freq = S13.train_frequent_outputs()

    if args.smoke:
        print("=== SMOKE TEST: seed 42, decoder, alpha=0.5, mean_diff, writes nothing ===")
        raw, recs, checks = run(cfg, device, smoke=True)
        print(f"  generations: {len(raw)}")
        print(f"  source groups: {sorted(raw.source_group.unique())}")
        print(f"  targets per source:")
        for s, g in raw.groupby("source_group"):
            print(f"    {s:18s} -> {sorted(g.language.unique())}")
        print("  Stage 12 consistency (rebuilt vs stored, cosine should be ~1.0):")
        for c in checks:
            print(f"    {c['scope']:16s} {c['vector_type']:10s} cos={c['cosine_vs_stage12']:.6f}")
        try:
            guard("data/splits/test.csv")
            print("  TEST GUARD FAILED"); return 1
        except RuntimeError:
            print("  test guard active")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    raw, vec_records, checks = run(cfg, device)
    print(f"scoring {len(raw)} dev generations ...")
    scored = S13.score_frame(raw, cfg, device)
    scored.to_csv(OUT_DIR / "transfer_dev_scored.csv", index=False)

    agg = summarise(scored, train_freq)
    agg.to_csv(OUT_DIR / "transfer_dev.csv", index=False)
    pd.DataFrame(vec_records).to_csv(OUT_DIR / "source_vector_manifest.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT_DIR / "stage12_consistency.csv", index=False)

    (OUT_DIR / "stage14_meta.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split_roles": {"train": "source-vector construction",
                        "dev": "all evaluation reported here",
                        "test": "NOT READ, NOT WRITTEN"},
        "stage13_test_artifacts": "untouched; Stage 14 writes only under results/transfer/",
        "source_groups": source_groups(cfg),
        "vector_types": VECTOR_TYPES, "strengths": STRENGTHS,
        "layer": LAYER, "pooling": POOLING, "sides": SIDES,
        "strength_selection": "none; full curves reported so no result can drive a choice",
        "sta_policy": "reported only for the seven covered languages; yo/xh use "
                      "SIM, FL, copy, template, unique-output and top-output share",
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    print("\n=== Stage 12 consistency (rebuilt source vectors vs stored) ===")
    c = pd.DataFrame(checks)
    print(c.groupby(["scope", "vector_type"])["cosine_vs_stage12"]
          .agg(["mean", "min"]).round(6).to_string())

    prim = agg[agg["vector_type"].isin(["mean_diff", "random"])]

    print("\n=== TRANSFER INTO YO/XH (mean over seeds), mean_diff vectors ===")
    t = prim[(prim["target_language"].isin(AFRICAN))
             & (prim["vector_type"] == "mean_diff")]
    print(t.pivot_table(index=["side", "source_group"], columns=["target_language", "strength"],
                        values="sim").round(3).to_string())

    print("\n=== YO/XH degeneracy diagnostics at each strength (mean_diff, decoder) ===")
    d = t[t["side"] == "decoder"]
    print(d.groupby(["source_group", "target_language", "strength"])[
        ["sim", "fl", "loose_copy_rate", "template_rate",
         "unique_output_rate", "top_output_share"]].mean().round(3).to_string())

    print("\n=== TRANSFER FROM YO/XH INTO COMPARISON LANGUAGES (validated STA) ===")
    f = prim[(prim["source_group"] == "YO_XH") & (prim["sta_status"] == "validated")]
    print(f.groupby(["side", "vector_type", "strength"])[
        ["delta_sta", "sim", "fl", "j"]].mean().round(4).to_string())

    print("\n=== WITHIN-LANGUAGE REFERENCE (validated languages) ===")
    w = prim[(prim["source_group"] == "within_language")
             & (prim["sta_status"] == "validated")]
    print(w.groupby(["side", "vector_type", "strength"])[
        ["delta_sta", "sim", "fl", "j"]].mean().round(4).to_string())

    print("\n=== STA-up / SIM-down flags (validated languages only) ===")
    base = agg[agg["strength"] == 0.0].groupby(
        ["side", "source_group", "target_language"])[["sta", "sim"]].mean()
    flagged = 0
    for k, g in agg[agg["strength"] != 0.0].groupby(
            ["side", "source_group", "target_language", "vector_type", "strength"]):
        if k[2] in AFRICAN:
            continue
        b = base.loc[(k[0], k[1], k[2])]
        dsta, dsim = g["sta"].mean() - b["sta"], g["sim"].mean() - b["sim"]
        if dsta > 0.01 and dsim < -0.02:
            flagged += 1
            if flagged <= 10:
                print(f"  FLAG {k}: dSTA={dsta:+.3f} dSIM={dsim:+.3f}")
    print(f"  total flagged cells: {flagged}")

    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ (dev only; Stage 13 test untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
