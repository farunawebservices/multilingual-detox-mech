#!/usr/bin/env python
"""Stage 12 -- Steering-vector construction and stability analysis.

Builds candidate steering directions from train activations only, records every vector
in a manifest, and reports how stable they are across model seeds and how similar they
are across languages. No text is generated here; that is Stage 13.

Usage
-----
    python scripts/12_build_steering_vectors.py --smoke   # verification pass, writes nothing
    python scripts/12_build_steering_vectors.py           # full run

Direction types
---------------
mean_diff  mean(toxic) - mean(reference) over standardised pooled activations. Primary.
probe      logistic-regression weight direction for the same contrast. Primary.
sae        SAE decoder rows for the Stage 9 replicated features. **Control arm only.**
           Stage 9 found no SAE feature passing replication, controls and behaviour
           together, and Stage 11's SAE ablation arm was null (+0.003 test STA). These
           are carried so the negative arm stays visible, never as a primary candidate.
random     matched random control, drawn Gaussian and rescaled to the exact norm and
           dimensionality of the vector it is matched to.

Spaces, and why the manifest records one
----------------------------------------
mean_diff, probe and random vectors live in the *pooled-standardised* space of a given
(side, layer, pooling). SAE decoder rows live in the *token-standardised* space, which is
a different transform. Cosines are therefore only computed within a space, never across,
and each vector records `space` plus the norm_stats file needed to map it back to raw
activations at injection time. A steering step of strength alpha adds
`alpha * vector * sd`, where sd is the standard deviation vector from that file.

Test discipline
---------------
Only train activations are read. A path guard raises on dev or test, since even dev is
not needed to *construct* a vector -- dev's role begins with the Stage 13 strength sweep.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Normalisation is L2 to unit norm, with the pre-normalisation norm recorded separately.
  Unit norm makes the Stage 13 strength sweep comparable across vector types, and keeping
  the original magnitude means the natural scale of each direction is not lost.
* Per-language probe directions are fitted on 244 samples in 768 dimensions, so they are
  underdetermined and can separate the training data perfectly without the direction
  being well determined. Every vector records `underdetermined` so this cannot be
  overlooked downstream; the per-language mean-difference vectors do not have this
  problem and are the safer per-language choice.
* The pooled all-language vector uses equal rows per language. Train is already 122 rows
  per language, so balance is automatic, but it is asserted rather than assumed.
* Random controls are matched per vector rather than once per configuration, so each real
  direction has its own null of identical norm and dimensionality.
* Cross-seed cosine is reported with the caveat that the three seeds are different
  fine-tunes with their own bases, so cosine measures whether the same direction is
  recovered in comparable coordinates, not an identity of the underlying subspace.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
ACT_DIR = REPO_ROOT / "results" / "activations"
SAE_FEAT_DIR = REPO_ROOT / "results" / "sae_features"
OUT_DIR = REPO_ROOT / "results" / "steering_vectors"

SIDES = ["encoder", "decoder"]
LAYERS = [4, 6, 8]
PRIMARY_LAYER = 6
PRIMARY_SIDE = "decoder"
POOLINGS = ["pooled_mean_all", "pooled_mean_content"]
CONDITIONS = ["toxic", "detox_reference"]
AFRICAN = ["yo", "xh"]
PROBE_C = 0.1
NORMALIZATION = "L2 unit norm; pre-normalisation norm recorded as raw_norm"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard(split: str) -> None:
    if split != "train":
        raise RuntimeError(
            f"Stage 12 builds vectors from train only; refusing to read '{split}'.")


def norm_stats_path(seed: int, side: str, layer: int, cond: str) -> Path:
    return ACT_DIR / "norm_stats" / f"seed{seed}_{side}_layer{layer:02d}_{cond}.pt"


def pooled_standardizer(seed: int, side: str, layer: int, pooling: str):
    means, varis, counts = [], [], []
    for cond in CONDITIONS:
        st = torch.load(norm_stats_path(seed, side, layer, cond), weights_only=False)
        means.append(st[f"{pooling}_mean"].double())
        varis.append(st[f"{pooling}_std"].double() ** 2)
        counts.append(float(st["n_rows"]))
    total = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / total
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / total
    return mu.float(), (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def load_pooled_train(seed: int, side: str, layer: int, pooling: str):
    guard("train")
    mu, sd = pooled_standardizer(seed, side, layer, pooling)
    mats, langs, labels = [], [], []
    for label, cond in enumerate(CONDITIONS):
        p = ACT_DIR / side / f"layer_{layer:02d}" / f"train_seed{seed}_{cond}.pt"
        d = torch.load(p, weights_only=False)
        z = (d[pooling] - mu) / sd.clamp_min(1e-6)
        mats.append(z.numpy())
        langs.append(np.array(d["language"]))
        labels.append(np.full(len(z), label))
    return np.vstack(mats), np.concatenate(langs), np.concatenate(labels), sd


def unit(v: np.ndarray) -> tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v), n


def scopes_for(cfg: dict) -> dict[str, list[str]]:
    langs = cfg["languages"]
    non_afr = [l for l in langs if l not in AFRICAN]
    s = {f"lang_{l}": [l] for l in langs}
    s["pooled_non_african"] = non_afr
    s["pooled_all_balanced"] = list(langs)
    s["pooled_african"] = list(AFRICAN)
    return s


def build_for_config(seed: int, side: str, layer: int, pooling: str,
                     cfg: dict, rng: np.random.Generator) -> tuple[list[dict], dict]:
    X, langs, y, sd = load_pooled_train(seed, side, layer, pooling)
    records, vectors = [], {}

    for scope, members in scopes_for(cfg).items():
        sel = np.isin(langs, members)
        Xs, ys, ls = X[sel], y[sel], langs[sel]

        # language balance: train is 122 rows per language per condition
        counts = pd.Series(ls).groupby([pd.Series(ls), pd.Series(ys)]).size()
        balanced = bool(pd.Series(ls).value_counts().nunique() == 1)
        if scope == "pooled_all_balanced":
            assert balanced, "pooled_all_balanced requires equal rows per language"

        n_tox, n_ref = int((ys == 0).sum()), int((ys == 1).sum())
        d_model = Xs.shape[1]
        underdetermined = (n_tox + n_ref) < d_model

        for dtype in ("mean_diff", "probe"):
            if dtype == "mean_diff":
                raw = Xs[ys == 0].mean(0) - Xs[ys == 1].mean(0)
            else:
                raw = LogisticRegression(C=PROBE_C, max_iter=3000).fit(Xs, ys).coef_[0]
                raw = -raw  # orient toxic-positive, matching mean_diff
            v, rawnorm = unit(raw)
            key = f"{dtype}|{scope}|{side}|L{layer}|{pooling}|seed{seed}"
            vectors[key] = v
            records.append({
                "key": key, "direction_type": dtype, "scope": scope,
                "source_languages": "|".join(members), "n_source_languages": len(members),
                "n_rows_toxic": n_tox, "n_rows_reference": n_ref,
                "n_rows_total": n_tox + n_ref,
                "seed": seed, "side": side, "layer": layer, "pooling": pooling,
                "space": "pooled_standardized", "d_model": d_model,
                "raw_norm": rawnorm, "normalization": NORMALIZATION,
                "language_balanced": balanced,
                "underdetermined": bool(underdetermined and dtype == "probe"),
                "is_primary": bool(dtype in ("mean_diff", "probe")),
                "arm": "primary",
                "norm_stats_file": str(norm_stats_path(seed, side, layer, "toxic")
                                       .relative_to(REPO_ROOT)),
                "injection_note": "add alpha * vector * sd to the raw activation",
            })

            # matched random control: identical norm and dimensionality
            r = rng.normal(size=d_model)
            r, _ = unit(r)
            rkey = f"random|{scope}|{side}|L{layer}|{pooling}|seed{seed}|match_{dtype}"
            vectors[rkey] = r
            records.append({**records[-1], "key": rkey, "direction_type": "random",
                            "arm": "random_control", "is_primary": False,
                            "matched_to": key, "raw_norm": 1.0,
                            "underdetermined": False})
    return records, vectors


def build_sae_arm(cfg: dict) -> tuple[list[dict], dict]:
    """SAE decoder rows for Stage 9 replicated features. Control arm only."""
    path = SAE_FEAT_DIR / "replicated_features.csv"
    records, vectors = [], {}
    if not path.exists():
        return records, vectors
    rep = pd.read_csv(path)
    rep = rep[(rep["layer"] == PRIMARY_LAYER) & (rep["token_set"] == "content")]
    for _, r in rep[["side", "seed", "feature"]].drop_duplicates().iterrows():
        side, seed, feat = r["side"], int(r["seed"]), int(r["feature"])
        p = REPO_ROOT / "models" / f"sae_{side}_layer{PRIMARY_LAYER}" / f"sae_content_seed{seed}.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        # Stage 8 saved these from a CUDA model, so move to host before converting.
        w = d["state_dict"]["W_dec"][feat].detach().cpu().numpy()
        v, rawnorm = unit(w)
        key = f"sae|feature{feat}|{side}|L{PRIMARY_LAYER}|token_content|seed{seed}"
        vectors[key] = v
        records.append({
            "key": key, "direction_type": "sae", "scope": f"sae_feature_{feat}",
            "source_languages": "|".join(cfg["languages"]),
            "n_source_languages": len(cfg["languages"]),
            "n_rows_toxic": -1, "n_rows_reference": -1, "n_rows_total": -1,
            "seed": seed, "side": side, "layer": PRIMARY_LAYER,
            "pooling": "token_content", "space": "token_standardized",
            "d_model": len(v), "raw_norm": rawnorm, "normalization": NORMALIZATION,
            "language_balanced": True, "underdetermined": False,
            "is_primary": False, "arm": "sae_control_only",
            "norm_stats_file": str(norm_stats_path(seed, side, PRIMARY_LAYER, "toxic")
                                   .relative_to(REPO_ROOT)),
            "injection_note": ("token-standardised space; NOT comparable to pooled "
                               "vectors and never a primary candidate"),
        })
    return records, vectors


def stability(man: pd.DataFrame, vecs: dict, seeds: list[int]) -> pd.DataFrame:
    rows = []
    grp = man[man["arm"].isin(["primary", "random_control"])].groupby(
        ["direction_type", "scope", "side", "layer", "pooling"])
    for (dtype, scope, side, layer, pooling), g in grp:
        present = {int(r["seed"]): r["key"] for _, r in g.iterrows()}
        cos = []
        for a, b in combinations(sorted(present), 2):
            va, vb = vecs[present[a]], vecs[present[b]]
            cos.append(float(np.dot(va, vb)))
        if cos:
            rows.append({"direction_type": dtype, "scope": scope, "side": side,
                         "layer": layer, "pooling": pooling, "n_seeds": len(present),
                         "mean_cosine_across_seeds": float(np.mean(cos)),
                         "min_cosine_across_seeds": float(np.min(cos)),
                         "max_cosine_across_seeds": float(np.max(cos))})
    return pd.DataFrame(rows)


def cross_language(man: pd.DataFrame, vecs: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    langs = cfg["languages"]
    sub = man[(man["arm"] == "primary") & (man["scope"].str.startswith("lang_"))]
    for (dtype, side, layer, pooling, seed), g in sub.groupby(
            ["direction_type", "side", "layer", "pooling", "seed"]):
        by_lang = {r["scope"].replace("lang_", ""): r["key"] for _, r in g.iterrows()}
        for a, b in combinations([l for l in langs if l in by_lang], 2):
            rows.append({"direction_type": dtype, "side": side, "layer": layer,
                         "pooling": pooling, "seed": seed, "lang_a": a, "lang_b": b,
                         "cosine": float(np.dot(vecs[by_lang[a]], vecs[by_lang[b]])),
                         "pair_type": ("african-african" if a in AFRICAN and b in AFRICAN
                                       else "cross-group" if (a in AFRICAN) != (b in AFRICAN)
                                       else "nonafrican-nonafrican")})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seeds = cfg["seeds_multi"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("=== SMOKE TEST: decoder L6 pooled_mean_content, seed 42, writes nothing ===")
        rng = np.random.default_rng(0)
        recs, vecs = build_for_config(42, "decoder", 6, "pooled_mean_content", cfg, rng)
        m = pd.DataFrame(recs)
        print(f"  vectors built: {len(m)} ({int((m.arm=='primary').sum())} primary, "
              f"{int((m.arm=='random_control').sum())} random controls)")
        print(f"  scopes: {sorted(m.scope.unique())}")
        norms = [float(np.linalg.norm(v)) for v in vecs.values()]
        print(f"  all unit norm: {np.allclose(norms, 1.0)} "
              f"(min={min(norms):.6f}, max={max(norms):.6f})")
        print(f"  all finite: {all(np.isfinite(v).all() for v in vecs.values())}")
        print(f"  underdetermined probe vectors: {int(m.underdetermined.sum())} of "
              f"{int((m.direction_type=='probe').sum())}")
        md = vecs['mean_diff|pooled_all_balanced|decoder|L6|pooled_mean_content|seed42']
        pr = vecs['probe|pooled_all_balanced|decoder|L6|pooled_mean_content|seed42']
        rd = vecs['random|pooled_all_balanced|decoder|L6|pooled_mean_content|seed42|match_probe']
        print(f"  cos(mean_diff, probe) = {np.dot(md,pr):+.4f}")
        print(f"  cos(mean_diff, random) = {np.dot(md,rd):+.4f}")
        try:
            guard("test")
            print("  TEST GUARD FAILED"); return 1
        except RuntimeError as e:
            print(f"  split guard active: {str(e)[:60]}...")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    rng = np.random.default_rng(42)
    all_recs, all_vecs = [], {}
    for seed in seeds:
        for side in SIDES:
            for layer in LAYERS:
                for pooling in POOLINGS:
                    r, v = build_for_config(seed, side, layer, pooling, cfg, rng)
                    all_recs += r
                    all_vecs.update(v)
        print(f"  seed {seed}: {len(all_recs)} records so far")

    sr, sv = build_sae_arm(cfg)
    all_recs += sr
    all_vecs.update(sv)

    man = pd.DataFrame(all_recs)
    man.to_csv(OUT_DIR / "manifest.csv", index=False)
    np.savez_compressed(OUT_DIR / "vectors.npz", **all_vecs)

    stab = stability(man, all_vecs, seeds)
    stab.to_csv(OUT_DIR / "stability_across_seeds.csv", index=False)
    xl = cross_language(man, all_vecs, cfg)
    xl.to_csv(OUT_DIR / "cosine_across_languages.csv", index=False)

    # cosine between the two primary types, within configuration
    pair_rows = []
    prim = man[man["arm"] == "primary"]
    for (scope, side, layer, pooling, seed), g in prim.groupby(
            ["scope", "side", "layer", "pooling", "seed"]):
        keys = {r["direction_type"]: r["key"] for _, r in g.iterrows()}
        if "mean_diff" in keys and "probe" in keys:
            pair_rows.append({"scope": scope, "side": side, "layer": layer,
                              "pooling": pooling, "seed": seed,
                              "cos_meandiff_probe": float(
                                  np.dot(all_vecs[keys["mean_diff"]],
                                         all_vecs[keys["probe"]]))})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_DIR / "cosine_meandiff_vs_probe.csv", index=False)

    (OUT_DIR / "stage12_meta.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "construction_split": "train only",
        "dev_role": "reserved for the Stage 13 strength sweep",
        "test_role": "NOT READ",
        "normalization": NORMALIZATION,
        "primary_configuration": {"side": PRIMARY_SIDE, "layer": PRIMARY_LAYER,
                                  "pooling": "pooled_mean_content"},
        "retained_layers": LAYERS, "sides": SIDES, "poolings": POOLINGS,
        "direction_types": {
            "mean_diff": "primary", "probe": "primary",
            "sae": "control arm only; never a primary candidate",
            "random": "matched control, identical norm and dimensionality"},
        "orientation": "all primary vectors oriented toxic-positive",
        "n_vectors": len(all_vecs),
        "spaces": sorted(man["space"].unique().tolist()),
        "cross_space_cosine_policy": "never computed; pooled and token spaces differ",
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    print(f"\n=== manifest: {len(man)} vectors ===")
    print(man.groupby(["arm", "direction_type"]).size().to_string())
    norms = np.array([np.linalg.norm(v) for v in all_vecs.values()])
    print(f"  all unit norm: {np.allclose(norms, 1.0)} | all finite: "
          f"{all(np.isfinite(v).all() for v in all_vecs.values())}")
    print(f"  underdetermined probe vectors: {int(man.underdetermined.sum())}")

    print("\n=== cross-seed stability (mean cosine), primary configuration ===")
    p = stab[(stab.side == PRIMARY_SIDE) & (stab.layer == PRIMARY_LAYER)
             & (stab.pooling == "pooled_mean_content")]
    print(p.pivot_table(index="scope", columns="direction_type",
                        values="mean_cosine_across_seeds").round(3).to_string())

    print("\n=== cross-seed stability by side/layer (mean over scopes) ===")
    print(stab[stab.direction_type.isin(["mean_diff", "probe", "random"])]
          .pivot_table(index=["side", "layer"], columns="direction_type",
                       values="mean_cosine_across_seeds").round(3).to_string())

    print("\n=== cosine between the two primary types (decoder L6 content) ===")
    pp = pairs[(pairs.side == PRIMARY_SIDE) & (pairs.layer == PRIMARY_LAYER)
               & (pairs.pooling == "pooled_mean_content")]
    print(pp.groupby("scope")["cos_meandiff_probe"].mean().round(3).to_string())

    print("\n=== cross-language cosine (decoder L6 content, mean over seeds) ===")
    x = xl[(xl.side == PRIMARY_SIDE) & (xl.layer == PRIMARY_LAYER)
           & (xl.pooling == "pooled_mean_content")]
    print(x.groupby(["direction_type", "pair_type"])["cosine"]
          .agg(["mean", "min", "max", "size"]).round(3).to_string())

    print("\n  mean cosine among per-language vectors, by language pair type:")
    md = x[x.direction_type == "mean_diff"]
    piv = md.pivot_table(index="lang_a", columns="lang_b", values="cosine", aggfunc="mean")
    print(piv.round(2).to_string())

    print("\nNo steered text generated. Stage 13 performs the strength sweep on dev.")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
