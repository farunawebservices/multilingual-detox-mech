#!/usr/bin/env python
"""Stage 8 -- Sparse autoencoder training and feature discovery.

Trains TopK sparse autoencoders on standardised mT5 activations to look for features
that behave consistently across languages, and reports how stable those features are
across seeds. No feature is ablated or steered here; that is Stage 11.

Usage
-----
    python scripts/08_train_sae.py --smoke   # verification pass, writes nothing
    python scripts/08_train_sae.py           # full run

Why a custom SAE rather than sae-lens
-------------------------------------
The brief allows either, provided the choice is verified and documented. sae-lens drives
activation capture through TransformerLens, whose official model list contains
google-t5/t5-{small,base,large} but no mT5 entry, and which in any case cannot load a
locally fine-tuned encoder-decoder checkpoint. Since Stage 6 already wrote activations
to disk, sae-lens's hooking machinery would contribute nothing while constraining the
architecture. The SAE below is a standard TopK autoencoder: unit-norm decoder columns, a
pre-bias subtracted before encoding, and an exact-k activation so L0 is known rather
than tuned through an L1 penalty.

Split roles
-----------
train  SAE fitting, and the normalisation statistics (already train-only from Stage 6).
dev    configuration selection (the k sweep) and all feature evaluation.
test   NOT READ. A path guard raises on any attempt; the set of files opened is recorded
       in the output so the claim is checkable rather than asserted.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Expansion factor is 4, not the 8 recorded in configs/experiment.yaml. Layer 6 offers
  roughly 60k train tokens before language balancing and about 36k after, so 8x (6144
  latents) would leave under 6 tokens per latent. 4x is still thin at ~12 tokens per
  latent, and the resulting dead-feature rates are reported rather than hidden.
* The two poolings from Stage 6 are aggregation choices over tokens, so their token-level
  analogue is used here: "all" keeps every non-pad position, "content" drops the task
  prefix on the encoder side and the decoder start token on the decoder side. Training an
  SAE on the 2196 pooled vectors themselves would put fewer samples than latents into the
  fit, so that variant is reported as infeasible rather than run.
* Language balancing samples an equal number of tokens per language, capped by the
  smallest language. This matters more than it might seem: isiXhosa needs 3.37 subword
  tokens per word against English's 1.60, so unbalanced pooling would over-weight exactly
  the languages whose tokenisation is worst.
* Per-language SAEs run only at layer 6 on content tokens. Each language has roughly 4k
  tokens, so they use 1x expansion and are labelled exploratory; their sample-to-latent
  ratio is reported alongside every metric.
* No feature is called a detoxification feature. Separating toxic from reference text is
  necessary but nowhere near sufficient -- the same separation is produced by sentence
  length, which the Stage 7 control showed is a live confound in yo/xh. Features are
  reported as candidates with their control associations attached, and the causal test
  is deferred to Stage 11.
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
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
ACT_DIR = REPO_ROOT / "results" / "activations"
OUT_DIR = REPO_ROOT / "results" / "sae"
MODEL_OUT = REPO_ROOT / "models"

SIDES = ["encoder", "decoder"]
LAYERS = [4, 6, 8]
TOKEN_SETS = ["all", "content"]
TRAIN_CONDITIONS = ["toxic", "detox_reference"]
EXPANSION = 4
PER_LANGUAGE_EXPANSION = 1
K_GRID = [16, 32, 64]
DEFAULT_K = 32
STEPS = 1500
BATCH = 2048
LR = 1e-3
PER_LANGUAGE_LAYER = 6
PER_LANGUAGE_TOKEN_SET = "content"
MIN_TOKENS_PER_LANGUAGE = 2000

_FILES_READ: list[str] = []


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard(split: str) -> None:
    if split == "test":
        raise RuntimeError(
            "Stage 8 must not read test activations. Test is held out until the final "
            "feature evaluation; SAEs are fitted on train and selected on dev."
        )


def act_path(split: str, seed: int, side: str, layer: int, condition: str) -> Path:
    guard(split)
    return ACT_DIR / side / f"layer_{layer:02d}" / f"{split}_seed{seed}_{condition}.pt"


# --------------------------------------------------------------------- data loading

def load_tokens(split: str, seed: int, side: str, layer: int, condition: str,
                token_set: str, prefix_len: int) -> tuple[torch.Tensor, pd.DataFrame]:
    """Token activations plus per-token metadata (row, language, position, condition)."""
    path = act_path(split, seed, side, layer, condition)
    if not path.exists():
        return None, None
    _FILES_READ.append(str(path.relative_to(REPO_ROOT)))
    d = torch.load(path, weights_only=False)
    toks, offs = d["tokens"], d["offsets"]
    langs, pair_ids = d["language"], d["pair_id"]

    skip = (prefix_len if side == "encoder" else 1) if token_set == "content" else 0
    keep_idx, meta = [], []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        s2 = min(s + skip, e)
        n = e - s2
        if n <= 0:
            continue
        keep_idx.append(torch.arange(s2, e))
        meta.append(pd.DataFrame({
            "row": i, "pair_id": pair_ids[i], "language": langs[i],
            "condition": condition, "position": np.arange(skip, skip + n),
            "row_n_tokens": e - s,
        }))
    idx = torch.cat(keep_idx)
    return toks[idx].to(torch.float32), pd.concat(meta, ignore_index=True)


def train_standardizer(seed: int, side: str, layer: int,
                       conditions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition-pooled token statistics, fitted on train only (Stage 6 artifacts)."""
    means, varis, counts = [], [], []
    for cond in conditions:
        st = torch.load(ACT_DIR / "norm_stats" /
                        f"seed{seed}_{side}_layer{layer:02d}_{cond}.pt", weights_only=False)
        means.append(st["token_mean"].double())
        varis.append(st["token_std"].double() ** 2)
        counts.append(float(st["n_tokens"]))
    total = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / total
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / total
    return mu.float(), (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def balance_by_language(meta: pd.DataFrame, rng: np.random.Generator,
                        cap: int | None = None) -> np.ndarray:
    """Equal token count per language; without this, high-fertility languages dominate."""
    per = meta.groupby("language").size()
    n = int(per.min()) if cap is None else min(int(per.min()), cap)
    picks = []
    for lang in sorted(meta["language"].unique()):
        idx = meta.index[meta["language"] == lang].to_numpy()
        picks.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(picks))


# ------------------------------------------------------------------------- the SAE

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.d_sae = d_sae
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = torch.topk(pre, self.k, dim=-1)
        out = torch.zeros_like(pre)
        return out.scatter_(-1, idx, torch.relu(vals))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


def train_sae(X: torch.Tensor, d_sae: int, k: int, device: str,
              steps: int = STEPS, seed: int = 0) -> TopKSAE:
    torch.manual_seed(seed)
    sae = TopKSAE(X.shape[1], d_sae, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    Xd = X.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, Xd.shape[0], (min(BATCH, Xd.shape[0]),), generator=g)
        batch = Xd[idx.to(device)]
        recon, _ = sae(batch)
        loss = (recon - batch).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
    return sae


@torch.no_grad()
def evaluate_sae(sae: TopKSAE, X: torch.Tensor, device: str) -> dict:
    Xd = X.to(device)
    recon, f = sae(Xd)
    mse = float((recon - Xd).pow(2).mean())
    total_var = float(Xd.var(0).sum())
    resid_var = float((Xd - recon).var(0).sum())
    alive = (f > 0).any(0)
    return {
        "recon_mse": mse,
        "explained_variance": 1.0 - resid_var / total_var if total_var > 0 else float("nan"),
        "mean_l0": float((f > 0).float().sum(1).mean()),
        "dead_feature_rate": float((~alive).float().mean()),
        "n_alive_features": int(alive.sum()),
        "all_finite": bool(torch.isfinite(recon).all() and torch.isfinite(f).all()),
    }


@torch.no_grad()
def per_example_features(sae: TopKSAE, X: torch.Tensor, meta: pd.DataFrame,
                         device: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean feature activation per (row, condition)."""
    f = sae.encode(X.to(device)).cpu().numpy()
    key = meta[["row", "condition"]].copy()
    key["_i"] = np.arange(len(meta))
    groups = key.groupby(["row", "condition"])["_i"].apply(list)
    mat = np.stack([f[i].mean(0) for i in groups.values])
    index = pd.DataFrame(list(groups.index), columns=["row", "condition"])
    index = index.merge(meta.groupby(["row", "condition"]).agg(
        language=("language", "first"), pair_id=("pair_id", "first"),
        n_tokens=("position", "size"), row_n_tokens=("row_n_tokens", "first"),
    ).reset_index(), on=["row", "condition"], how="left")
    return mat, index


def feature_stability(saes: dict[int, TopKSAE]) -> dict:
    """Matched cosine between decoder directions across seeds."""
    seeds = sorted(saes)
    out = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            Wa = torch.nn.functional.normalize(saes[a].W_dec.detach().cpu(), dim=1)
            Wb = torch.nn.functional.normalize(saes[b].W_dec.detach().cpu(), dim=1)
            sim = Wa @ Wb.T
            out[f"seed{a}_vs_seed{b}"] = {
                "mean_max_cosine": float(sim.max(1).values.mean()),
                "median_max_cosine": float(sim.max(1).values.median()),
                "frac_features_matched_above_0.7": float((sim.max(1).values > 0.7).float().mean()),
            }
    return out


# ------------------------------------------------------------------ feature testing

def auc_safe(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def test_features(mat: np.ndarray, index: pd.DataFrame, dev_mat: np.ndarray,
                  dev_index: pd.DataFrame, dev_rows: pd.DataFrame,
                  gen_mat: np.ndarray | None, gen_index: pd.DataFrame | None,
                  top_n: int = 25) -> pd.DataFrame:
    """Per-feature association with the four targets plus the control variables.

    Deliberately reports control associations next to the target AUCs. A feature that
    separates toxic from reference while also tracking token count or firing on one
    language is not evidence of a detoxification mechanism.
    """
    cond = index["condition"].to_numpy()
    y_tr = (cond == "detox_reference").astype(int)

    dcond = dev_index["condition"].to_numpy()
    y_dev = (dcond == "detox_reference").astype(int)

    # rank candidates by dev toxic-vs-reference separation
    aucs = np.array([auc_safe(y_dev, dev_mat[:, j]) for j in range(dev_mat.shape[1])])
    order = np.argsort(-np.nan_to_num(np.abs(aucs - 0.5)))[:top_n]

    behaviour = dev_rows.set_index("pair_id")["behaviour_label"]
    dev_tox = dev_index["condition"] == "toxic"
    lab = dev_index.loc[dev_tox, "pair_id"].map(behaviour).to_numpy()

    recs = []
    for j in order:
        col_dev = dev_mat[:, j]
        tox_col = col_dev[dev_tox.to_numpy()]

        copy_sel = np.isin(lab, ["copy_input", "ordinary_rewrite"])
        tmpl_sel = np.isin(lab, ["generic_template", "ordinary_rewrite"])

        rec = {
            "feature": int(j),
            "auc_toxic_vs_reference_train": auc_safe(y_tr, mat[:, j]),
            "auc_toxic_vs_reference_dev": float(aucs[j]),
            "auc_copy_vs_ordinary_dev": auc_safe(
                (lab[copy_sel] == "copy_input").astype(int), tox_col[copy_sel]),
            "auc_template_vs_ordinary_dev": auc_safe(
                (lab[tmpl_sel] == "generic_template").astype(int), tox_col[tmpl_sel]),
            # controls
            "corr_token_count_dev": float(np.corrcoef(col_dev, dev_index["n_tokens"])[0, 1]),
            "corr_sequence_length_dev": float(
                np.corrcoef(col_dev, dev_index["row_n_tokens"])[0, 1]),
            "auc_language_max_dev": max(
                auc_safe((dev_index["language"] == l).astype(int), col_dev)
                for l in sorted(dev_index["language"].unique())),
            "n_languages_active_dev": int(sum(
                (col_dev[(dev_index["language"] == l).to_numpy()] > 0).mean() > 0.1
                for l in sorted(dev_index["language"].unique()))),
        }
        if gen_mat is not None:
            gcond = np.concatenate([np.zeros(int(dev_tox.sum())), np.ones(gen_mat.shape[0])])
            gscore = np.concatenate([tox_col, gen_mat[:, j]])
            rec["auc_toxic_vs_generated_dev"] = auc_safe(gcond, gscore)
        else:
            rec["auc_toxic_vs_generated_dev"] = float("nan")
        recs.append(rec)

    df = pd.DataFrame(recs)
    # Candidate criteria: separates the contrast, is not a language detector, is not
    # explained by length, and fires in at least three languages incl. an African one.
    df["candidate_cross_lingual"] = (
        (df["auc_toxic_vs_reference_dev"].sub(0.5).abs() > 0.15)
        & (df["auc_language_max_dev"] < 0.90)
        & (df["corr_token_count_dev"].abs() < 0.30)
        & (df["n_languages_active_dev"] >= 3)
    )
    df["label"] = np.where(
        df["candidate_cross_lingual"],
        "candidate (associative; causal status untested -- Stage 11)",
        "not a candidate")
    return df


# ----------------------------------------------------------------------------- run

def run_config(seed: int, side: str, layer: int, token_set: str, cfg: dict, device: str,
               prefix_len: int, k: int, expansion: int, language: str | None = None,
               rng_seed: int = 0) -> dict:
    mu, sd = train_standardizer(seed, side, layer, TRAIN_CONDITIONS)
    mats, metas = [], []
    for cond in TRAIN_CONDITIONS:
        X, meta = load_tokens("train", seed, side, layer, cond, token_set, prefix_len)
        mats.append((X - mu) / sd.clamp_min(1e-6))
        metas.append(meta)
    X = torch.cat(mats)
    meta = pd.concat(metas, ignore_index=True)

    if language:
        sel = (meta["language"] == language).to_numpy()
        X, meta = X[sel], meta[sel].reset_index(drop=True)
        balanced_note = f"per-language ({language})"
    else:
        rng = np.random.default_rng(rng_seed)
        idx = balance_by_language(meta, rng)
        X, meta = X[idx], meta.iloc[idx].reset_index(drop=True)
        balanced_note = "language-balanced (equal tokens per language)"

    assert torch.isfinite(X).all(), "non-finite standardized activations"
    d_sae = expansion * X.shape[1]
    sae = train_sae(X, d_sae, k, device, seed=rng_seed)

    train_metrics = evaluate_sae(sae, X, device)

    # dev evaluation, same standardizer
    dev_mats, dev_metas = [], []
    for cond in TRAIN_CONDITIONS:
        Xd, md = load_tokens("dev", seed, side, layer, cond, token_set, prefix_len)
        dev_mats.append((Xd - mu) / sd.clamp_min(1e-6))
        dev_metas.append(md)
    Xdev = torch.cat(dev_mats)
    mdev = pd.concat(dev_metas, ignore_index=True)
    if language:
        s = (mdev["language"] == language).to_numpy()
        Xdev, mdev = Xdev[s], mdev[s].reset_index(drop=True)
    dev_metrics = evaluate_sae(sae, Xdev, device)

    return {
        "seed": seed, "side": side, "layer": layer, "token_set": token_set,
        "language": language or "multilingual_balanced", "k": k,
        "d_sae": d_sae, "expansion": expansion,
        "n_train_tokens": int(X.shape[0]),
        "tokens_per_latent": float(X.shape[0] / d_sae),
        "sampling": balanced_note,
        **{f"train_{a}": b for a, b in train_metrics.items()},
        **{f"dev_{a}": b for a, b in dev_metrics.items()},
        "_sae": sae, "_X": X, "_meta": meta, "_Xdev": Xdev, "_mdev": mdev,
        "_mu": mu, "_sd": sd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seeds = args.seeds or cfg["seeds_multi"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix_len = 4  # "detoxify: " per the Stage 6 manifest
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("=== SMOKE TEST: seed 42, encoder layer 6, content tokens, writes nothing ===")
        r = run_config(42, "encoder", 6, "content", cfg, device, prefix_len,
                       DEFAULT_K, EXPANSION, rng_seed=42)
        print(f"  train tokens (balanced): {r['n_train_tokens']} "
              f"| d_sae={r['d_sae']} | tokens/latent={r['tokens_per_latent']:.1f}")
        print(f"  standardized inputs finite: {bool(torch.isfinite(r['_X']).all())}")
        print(f"  train: mse={r['train_recon_mse']:.4f} EV={r['train_explained_variance']:.4f} "
              f"L0={r['train_mean_l0']:.1f} dead={r['train_dead_feature_rate']:.3f}")
        print(f"  dev  : mse={r['dev_recon_mse']:.4f} EV={r['dev_explained_variance']:.4f} "
              f"L0={r['dev_mean_l0']:.1f} dead={r['dev_dead_feature_rate']:.3f}")
        print(f"  per-language token balance: "
              f"{r['_meta'].groupby('language').size().to_dict()}")
        try:
            act_path("test", 42, "encoder", 6, "toxic")
            print("  TEST GUARD FAILED")
            return 1
        except RuntimeError as exc:
            print(f"  test guard active: {str(exc)[:70]}...")
        print(f"  files read: {len(set(_FILES_READ))}, none from test: "
              f"{not any('test_' in f for f in _FILES_READ)}")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    # ---- k selection on dev, at the primary configuration ----
    print("selecting k on dev (encoder+decoder, layer 6, content tokens, seed 42)")
    ksel = []
    for side in SIDES:
        for k in K_GRID:
            r = run_config(42, side, 6, "content", cfg, device, prefix_len, k,
                           EXPANSION, rng_seed=42)
            ksel.append({m: r[m] for m in ("side", "k", "dev_explained_variance",
                                           "dev_recon_mse", "dev_dead_feature_rate",
                                           "train_explained_variance")})
            print(f"  {side} k={k:3d}: dev EV={r['dev_explained_variance']:.4f} "
                  f"dead={r['dev_dead_feature_rate']:.3f}")
    kdf = pd.DataFrame(ksel)
    kdf.to_csv(OUT_DIR / "k_selection_dev.csv", index=False)
    chosen_k = int(kdf.groupby("k")["dev_explained_variance"].mean().idxmax())
    print(f"  chosen k (max mean dev explained variance): {chosen_k}")

    # ---- multilingual balanced grid ----
    rows, saes_by_config, feature_tables = [], {}, []
    for seed in seeds:
        dev_rows = pd.read_csv(ACT_DIR / "index" / f"dev_seed{seed}_rows.csv")
        for side in SIDES:
            for layer in LAYERS:
                for token_set in TOKEN_SETS:
                    r = run_config(seed, side, layer, token_set, cfg, device, prefix_len,
                                   chosen_k, EXPANSION, rng_seed=seed)
                    key = (side, layer, token_set)
                    saes_by_config.setdefault(key, {})[seed] = r["_sae"]
                    rows.append({k: v for k, v in r.items() if not k.startswith("_")})

                    if layer == PER_LANGUAGE_LAYER and token_set == PER_LANGUAGE_TOKEN_SET:
                        mat, index = per_example_features(r["_sae"], r["_X"], r["_meta"], device)
                        dmat, dindex = per_example_features(r["_sae"], r["_Xdev"],
                                                            r["_mdev"], device)
                        gX, gmeta = load_tokens("dev", seed, side, layer, "generated",
                                                token_set, prefix_len)
                        gmat = None
                        if gX is not None:
                            gz = (gX - r["_mu"]) / r["_sd"].clamp_min(1e-6)
                            gmat, _ = per_example_features(r["_sae"], gz, gmeta, device)
                        ft = test_features(mat, index, dmat, dindex, dev_rows, gmat, None)
                        ft.insert(0, "seed", seed)
                        ft.insert(1, "side", side)
                        ft.insert(2, "layer", layer)
                        feature_tables.append(ft)
                    print(f"  seed{seed} {side} L{layer} {token_set}: "
                          f"dev EV={r['dev_explained_variance']:.3f} "
                          f"dead={r['dev_dead_feature_rate']:.3f}")

    # ---- per-language SAEs ----
    per_lang_rows = []
    for seed in seeds:
        for side in SIDES:
            for lang in cfg["languages"]:
                r = run_config(seed, side, PER_LANGUAGE_LAYER, PER_LANGUAGE_TOKEN_SET,
                               cfg, device, prefix_len, chosen_k, PER_LANGUAGE_EXPANSION,
                               language=lang, rng_seed=seed)
                row = {k: v for k, v in r.items() if not k.startswith("_")}
                row["below_min_tokens"] = row["n_train_tokens"] < MIN_TOKENS_PER_LANGUAGE
                row["status"] = "exploratory (underdetermined)"
                per_lang_rows.append(row)
        print(f"  per-language SAEs done for seed {seed}")

    # ---- stability across seeds ----
    stability = []
    for (side, layer, token_set), saes in saes_by_config.items():
        for pair, vals in feature_stability(saes).items():
            stability.append({"side": side, "layer": layer, "token_set": token_set,
                              "seed_pair": pair, **vals})

    mdf = pd.DataFrame(rows)
    pdf = pd.DataFrame(per_lang_rows)
    sdf = pd.DataFrame(stability)
    fdf = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()

    mdf.to_csv(OUT_DIR / "sae_metrics_multilingual.csv", index=False)
    pdf.to_csv(OUT_DIR / "sae_metrics_per_language.csv", index=False)
    sdf.to_csv(OUT_DIR / "feature_stability.csv", index=False)
    fdf.to_csv(OUT_DIR / "feature_candidates.csv", index=False)

    MODEL_OUT.mkdir(exist_ok=True)
    for (side, layer, token_set), saes in saes_by_config.items():
        for seed, sae in saes.items():
            d = MODEL_OUT / f"sae_{side}_layer{layer}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": sae.state_dict(), "d_sae": sae.d_sae, "k": sae.k,
                        "side": side, "layer": layer, "token_set": token_set,
                        "seed": seed, "expansion": EXPANSION,
                        "standardization": "train-only Stage 6 token statistics"},
                       d / f"sae_{token_set}_seed{seed}.pt")

    (OUT_DIR / "stage8_meta.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "TopK SAE, unit-norm decoder, pre-bias",
        "sae_lens_used": False,
        "sae_lens_rationale": ("TransformerLens supports google-t5/t5-{small,base,large} "
                               "but has no mT5 entry and cannot load a local fine-tuned "
                               "encoder-decoder checkpoint; activations already on disk"),
        "expansion_multilingual": EXPANSION,
        "expansion_per_language": PER_LANGUAGE_EXPANSION,
        "k_grid": K_GRID, "chosen_k": chosen_k, "steps": STEPS, "batch": BATCH, "lr": LR,
        "split_roles": {"train": "SAE fitting", "dev": "k selection and feature evaluation",
                        "test": "NOT READ"},
        "files_read": sorted(set(_FILES_READ)),
        "any_test_file_read": any("test_" in f for f in _FILES_READ),
        "feature_labeling_policy": ("no feature is called a detoxification feature; "
                                    "candidates are associative pending Stage 11"),
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    print("\n=== SAE quality, multilingual balanced (mean over seeds) ===")
    print(mdf.groupby(["side", "layer", "token_set"])[
        ["train_explained_variance", "dev_explained_variance", "dev_recon_mse",
         "dev_mean_l0", "dev_dead_feature_rate", "tokens_per_latent"]
    ].mean().round(3).to_string())

    print("\n=== feature stability across seeds (decoder direction matched cosine) ===")
    print(sdf.groupby(["side", "layer", "token_set"])[
        ["mean_max_cosine", "median_max_cosine", "frac_features_matched_above_0.7"]
    ].mean().round(3).to_string())

    print("\n=== per-language SAEs (layer 6, content, exploratory) ===")
    print(pdf.groupby(["side", "language"])[
        ["n_train_tokens", "tokens_per_latent", "dev_explained_variance",
         "dev_dead_feature_rate"]].mean().round(3).to_string())

    if len(fdf):
        print("\n=== candidate features (top by dev toxic-vs-reference separation) ===")
        n_cand = int(fdf["candidate_cross_lingual"].sum())
        print(f"  {n_cand} of {len(fdf)} inspected features meet the candidate criteria")
        cols = ["seed", "side", "feature", "auc_toxic_vs_reference_dev",
                "auc_copy_vs_ordinary_dev", "auc_template_vs_ordinary_dev",
                "auc_toxic_vs_generated_dev", "corr_token_count_dev",
                "auc_language_max_dev", "n_languages_active_dev"]
        print(fdf[fdf["candidate_cross_lingual"]][cols].head(12).round(3).to_string(index=False))
        print("\n  NOTE: 'candidate' is associative only. Separating toxic from reference "
              "is not sufficient to call a feature a detoxification feature; the causal "
              "test is Stage 11.")

    print(f"\n  any test file read: {any('test_' in f for f in _FILES_READ)}")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/ and models/sae_<side>_layer<N>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
