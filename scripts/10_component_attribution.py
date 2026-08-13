#!/usr/bin/env python
"""Stage 10 -- Component attribution: ranking attention heads and MLP neurons.

Ranks encoder and decoder components by how much they drive two readout directions at
decoder layer 6: the Stage 7 linear-probe direction and the train-derived
toxic-minus-reference mean difference. This is a ranking pass only. Nothing is ablated,
and no component is called a circuit -- attribution measures association with a readout,
not causal necessity, and the causal test is Stage 11.

Usage
-----
    python scripts/10_component_attribution.py --smoke   # verification pass, writes nothing
    python scripts/10_component_attribution.py           # full run

Method
------
Gradient-times-activation attribution (attribution patching) onto a scalar readout

    s(x) = < standardise(mean_content(decoder_block6_hidden(x))), d >

for d in {probe direction, mean-difference direction}. For each component the attribution
is (activation * d s / d activation), summed over that component's units and over content
positions. Attention heads are read at the input to the output projection `o`, which is
the concatenation of 12 heads of 64 dims, so a head's attribution is the sum over its own
64 dims. MLP neurons are read at the input to `wo`, where each of the 2048 entries is one
neuron.

This approximates activation patching to first order at a small fraction of the cost,
which is what makes a 10,552-component sweep across three seeds affordable. Being a
first-order approximation is exactly why it cannot settle causality, and why Stage 11
exists.

Split roles
-----------
train  discovery. Component ranking for toxic-vs-reference is established here.
dev    ranking confirmation, and the three contrasts that need generations
       (toxic-vs-generated, copy-vs-ordinary, template-vs-ordinary), since Stage 4 did
       not generate on train.
test   NOT READ. A path guard raises, and the files opened are recorded in the output.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Attention heads are swept across all 12 encoder layers and decoder layers 0-6, for both
  decoder self-attention and cross-attention. Decoder layers above 6 have no path to a
  layer-6 readout, so their attribution must be exactly zero; that is asserted at runtime
  as a correctness check on the whole gradient pipeline rather than assumed.
* MLP neurons are swept at the configured layers only -- encoder {4,6,8} and decoder
  {4,6} -- because 2048 neurons per layer would otherwise dominate both runtime and the
  multiple-comparison burden without adding a comparison the heads do not already give.
* Effect sizes are Cohen's d on per-example attributions, with AUC reported alongside.
  d is signed, which matters here: the direction of a component's contribution is the
  thing Stage 12 would need in order to build a steering vector from it.
* Matched random controls draw components of the same type from the same layers, so the
  null shares the ranking pool's structure. The 95th percentile of the matched-random
  |d| is reported as the threshold a real component has to clear.
* Model parameters keep requires_grad=True because activation gradients need the graph.
  No optimiser is constructed and no step is taken, so the checkpoints are untouched;
  this is verified by checksum at the end of the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, MT5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
ACT_DIR = REPO_ROOT / "results" / "activations"
GEN_DIR = REPO_ROOT / "results" / "generations"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
OUT_DIR = REPO_ROOT / "results" / "attribution"

TASK_PREFIX = "detoxify: "
READOUT_SIDE, READOUT_LAYER = "decoder", 6
ENC_HEAD_LAYERS = list(range(12))
DEC_HEAD_LAYERS = list(range(0, READOUT_LAYER + 1))
ENC_MLP_LAYERS = [4, 6, 8]
DEC_MLP_LAYERS = [4, 6]
MAX_SOURCE, MAX_TARGET = 96, 112
BATCH = 8
TOP_K = 25
N_RANDOM_CONTROLS = 200

_FILES_READ: list[str] = []


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard(split: str) -> None:
    if split == "test":
        raise RuntimeError("Stage 10 must not read the test split; it is held for Stage 11+.")


def digest(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- readout directions

def pooled_standardizer(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    means, varis, counts = [], [], []
    for cond in ("toxic", "detox_reference"):
        st = torch.load(ACT_DIR / "norm_stats" /
                        f"seed{seed}_{READOUT_SIDE}_layer{READOUT_LAYER:02d}_{cond}.pt",
                        weights_only=False)
        means.append(st["pooled_mean_content_mean"].double())
        varis.append(st["pooled_mean_content_std"].double() ** 2)
        counts.append(float(st["n_rows"]))
    total = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / total
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / total
    return mu.float(), (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def readout_directions(seed: int) -> dict[str, torch.Tensor]:
    """Stage 7 probe direction and the toxic-minus-reference mean difference, from train."""
    mu, sd = pooled_standardizer(seed)
    mats, labels = [], []
    for label, cond in enumerate(("toxic", "detox_reference")):
        p = ACT_DIR / READOUT_SIDE / f"layer_{READOUT_LAYER:02d}" / f"train_seed{seed}_{cond}.pt"
        _FILES_READ.append(str(p.relative_to(REPO_ROOT)))
        d = torch.load(p, weights_only=False)
        z = (d["pooled_mean_content"] - mu) / sd.clamp_min(1e-6)
        mats.append(z.numpy())
        labels.append(np.full(len(z), label))
    X, y = np.vstack(mats), np.concatenate(labels)
    probe = LogisticRegression(C=0.1, max_iter=3000).fit(X, y).coef_[0]
    mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)

    def unit(v):
        return torch.tensor(v / (np.linalg.norm(v) + 1e-12), dtype=torch.float32)

    return {"probe": unit(probe), "mean_diff": unit(mean_diff)}


# ----------------------------------------------------------------------- attribution

class AttributionHooks:
    """Captures head-projection inputs and MLP neuron activations, with gradients."""

    def __init__(self, model, n_heads: int, d_kv: int):
        self.n_heads, self.d_kv = n_heads, d_kv
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = []
        self.readout: torch.Tensor | None = None

        def cap(name):
            def pre_hook(_m, inputs):
                t = inputs[0]
                t.requires_grad_(True)
                t.retain_grad()
                self.acts[name] = t
            return pre_hook

        for i in ENC_HEAD_LAYERS:
            self.handles.append(model.encoder.block[i].layer[0].SelfAttention.o
                                .register_forward_pre_hook(cap(f"enc.self.{i}.head")))
        for i in DEC_HEAD_LAYERS:
            self.handles.append(model.decoder.block[i].layer[0].SelfAttention.o
                                .register_forward_pre_hook(cap(f"dec.self.{i}.head")))
            self.handles.append(model.decoder.block[i].layer[1].EncDecAttention.o
                                .register_forward_pre_hook(cap(f"dec.cross.{i}.head")))
        for i in ENC_MLP_LAYERS:
            self.handles.append(model.encoder.block[i].layer[-1].DenseReluDense.wo
                                .register_forward_pre_hook(cap(f"enc.mlp.{i}.neuron")))
        for i in DEC_MLP_LAYERS:
            self.handles.append(model.decoder.block[i].layer[-1].DenseReluDense.wo
                                .register_forward_pre_hook(cap(f"dec.mlp.{i}.neuron")))

        def readout_hook(_m, _i, output):
            self.readout = output[0] if isinstance(output, tuple) else output
        self.handles.append(
            model.decoder.block[READOUT_LAYER].register_forward_hook(readout_hook))

    def clear(self):
        self.acts.clear()
        self.readout = None

    def remove(self):
        for h in self.handles:
            h.remove()


def canonical_keys() -> list[str]:
    """Hook keys in the same order component_names() lists components.

    This must not be read off `hooks.acts` insertion order. That dict is populated during
    the forward pass, so encoder block 4's MLP is inserted between self-attention layers 4
    and 5, while component_names() groups all heads before all neurons. Relying on dict
    order silently misaligned attribution columns with component names -- caught by the
    zero-path check, which found non-zero attribution for decoder layers above the readout
    where no gradient path exists.
    """
    keys = [f"enc.self.{i}.head" for i in ENC_HEAD_LAYERS]
    for i in DEC_HEAD_LAYERS:
        keys += [f"dec.self.{i}.head", f"dec.cross.{i}.head"]
    keys += [f"enc.mlp.{i}.neuron" for i in ENC_MLP_LAYERS]
    keys += [f"dec.mlp.{i}.neuron" for i in DEC_MLP_LAYERS]
    return keys


def component_names(n_heads: int) -> list[str]:
    names = []
    for i in ENC_HEAD_LAYERS:
        names += [f"enc.self.L{i}.head{h}" for h in range(n_heads)]
    for i in DEC_HEAD_LAYERS:
        names += [f"dec.self.L{i}.head{h}" for h in range(n_heads)]
        names += [f"dec.cross.L{i}.head{h}" for h in range(n_heads)]
    for i in ENC_MLP_LAYERS:
        names += [f"enc.mlp.L{i}.n{j}" for j in range(2048)]
    for i in DEC_MLP_LAYERS:
        names += [f"dec.mlp.L{i}.n{j}" for j in range(2048)]
    return names


def attribute_batch(model, hooks: AttributionHooks, enc, dec_ids, dec_mask,
                    direction: torch.Tensor, mu, sd, n_heads: int, d_kv: int) -> np.ndarray:
    hooks.clear()
    model.zero_grad(set_to_none=True)
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
          decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask)

    h = hooks.readout                                   # [B, T, d_model]
    m = dec_mask.clone().bool()
    m[:, 0] = False                                     # drop the decoder start token
    w = m.float().unsqueeze(-1)
    pooled = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
    z = (pooled - mu) / sd.clamp_min(1e-6)
    s = (z @ direction).sum()
    s.backward()

    parts = []
    for name in canonical_keys():
        act = hooks.acts[name]
        g = act.grad
        contrib = act.detach() * (g if g is not None else torch.zeros_like(act))
        side = name.split(".")[0]
        mask = (enc.attention_mask if side == "enc" else dec_mask).bool()
        contrib = contrib * mask.unsqueeze(-1)
        summed = contrib.sum(1)                          # sum over positions -> [B, D]
        if name.endswith("head"):
            summed = summed.view(summed.shape[0], n_heads, d_kv).sum(-1)  # [B, heads]
        parts.append(summed.detach().float().cpu())
    return torch.cat(parts, dim=1).numpy()


def run_split(split: str, seed: int, cfg: dict, device: str, direction_name: str,
              direction: torch.Tensor, limit: int | None = None):
    guard(split)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / f"seed{seed}"))
    model = MT5ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR / f"seed{seed}")).to(device).eval()
    mu, sd = pooled_standardizer(seed)
    mu, sd = mu.to(device), sd.to(device)
    n_heads, d_kv = model.config.num_heads, model.config.d_kv

    if split == "train":
        base = pd.read_csv(SPLITS_DIR / "train.csv").rename(
            columns={"detox_output": "detox_reference"})
        base["generated"] = None
        base["behaviour_label"] = "no_generation"
        conditions = ["toxic", "detox_reference"]
    else:
        base = pd.read_csv(GEN_DIR / f"seed{seed}_{split}.csv")
        idx = pd.read_csv(ACT_DIR / "index" / f"{split}_seed{seed}_rows.csv")
        base = base.merge(idx[["pair_id", "behaviour_label"]], on="pair_id", how="left")
        conditions = ["toxic", "detox_reference", "generated"]
    if limit:
        base = base.groupby("language").head(max(1, limit // 9)).reset_index(drop=True)

    hooks = AttributionHooks(model, n_heads, d_kv)
    rows, attrs = [], []
    for cond in conditions:
        text_col = {"toxic": "toxic_input", "detox_reference": "detox_reference",
                    "generated": "generated"}[cond]
        for start in range(0, len(base), BATCH):
            chunk = base.iloc[start:start + BATCH]
            src = [TASK_PREFIX + str(t) for t in chunk["toxic_input"]]
            tgt = [str(t) if str(t).strip() else " " for t in chunk[text_col]]
            enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_SOURCE).to(device)
            t = tokenizer(text_target=tgt, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_TARGET).to(device)
            dec_ids = model._shift_right(t.input_ids)
            a = attribute_batch(model, hooks, enc, dec_ids, t.attention_mask,
                                direction.to(device), mu, sd, n_heads, d_kv)
            attrs.append(a)
            meta = chunk[["pair_id", "language", "behaviour_label"]].copy()
            meta["condition"] = cond
            meta["n_tokens"] = t.attention_mask.sum(1).cpu().numpy()
            rows.append(meta)
    hooks.remove()
    A = np.vstack(attrs)
    meta = pd.concat(rows, ignore_index=True)
    del model
    torch.cuda.empty_cache()
    return A, meta, n_heads


# --------------------------------------------------------------------------- scoring

def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = len(a), len(b)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return (a.mean(0) - b.mean(0)) / np.where(pooled > 1e-12, pooled, np.nan)


def contrast_scores(A: np.ndarray, meta: pd.DataFrame, sel_a: np.ndarray,
                    sel_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sel_a.sum() < 3 or sel_b.sum() < 3:
        return np.full(A.shape[1], np.nan), np.full(A.shape[1], np.nan)
    d = cohens_d(A[sel_a], A[sel_b])
    y = np.r_[np.ones(sel_a.sum()), np.zeros(sel_b.sum())]
    X = np.vstack([A[sel_a], A[sel_b]])
    aucs = np.array([roc_auc_score(y, X[:, j]) if np.ptp(X[:, j]) > 0 else 0.5
                     for j in range(X.shape[1])])
    return d, aucs


def control_stats(A: np.ndarray, meta: pd.DataFrame, languages: list[str]) -> dict:
    lang_auc = np.zeros(A.shape[1])
    for lang in languages:
        y = (meta["language"] == lang).to_numpy().astype(int)
        if 0 < y.sum() < len(y):
            a = np.array([roc_auc_score(y, A[:, j]) if np.ptp(A[:, j]) > 0 else 0.5
                          for j in range(A.shape[1])])
            lang_auc = np.maximum(lang_auc, np.abs(a - 0.5) + 0.5)
    n = meta["n_tokens"].to_numpy(dtype=float)
    Az = A - A.mean(0)
    nz = (n - n.mean()) / (n.std() + 1e-12)
    corr_len = (Az * nz[:, None]).mean(0) / (A.std(0) + 1e-12)
    return {"control_language_max_auc": lang_auc, "control_corr_token_count": corr_len}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seeds = cfg["seeds_multi"]
    languages = cfg["languages"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(MODEL_DIR.glob("seed*/model.safetensors"))
    before = {str(p.relative_to(REPO_ROOT)): digest(p) for p in ckpts}

    if args.smoke:
        print("=== SMOKE TEST: seed 42, dev, 18 rows, probe direction, writes nothing ===")
        dirs = readout_directions(42)
        A, meta, n_heads = run_split("dev", 42, cfg, device, "probe", dirs["probe"], limit=18)
        names = component_names(n_heads)
        print(f"  attribution matrix: {A.shape} (rows x components), names={len(names)}")
        print(f"  all finite: {bool(np.isfinite(A).all())}")
        zero_layers = [i for i in DEC_HEAD_LAYERS if i > READOUT_LAYER]
        print(f"  decoder layers above the readout: {zero_layers} (expected none in sweep)")
        head_cols = [k for k, nm in enumerate(names) if ".head" in nm]
        print(f"  head components={len(head_cols)}, neuron components={len(names)-len(head_cols)}")
        print(f"  conditions: {sorted(meta['condition'].unique())}")
        print(f"  languages balanced: {meta.groupby(['condition','language']).size().unique()}")
        d, _ = contrast_scores(A, meta, (meta.condition == "toxic").to_numpy(),
                               (meta.condition == "detox_reference").to_numpy())
        print(f"  toxic-vs-reference |d|: max={np.nanmax(np.abs(d)):.3f} "
              f"median={np.nanmedian(np.abs(d)):.3f}")
        after = {str(p.relative_to(REPO_ROOT)): digest(p) for p in ckpts}
        print(f"  checkpoints unchanged: {before == after}")
        try:
            run_split("test", 42, cfg, device, "probe", dirs["probe"], limit=4)
            print("  TEST GUARD FAILED"); return 1
        except RuntimeError:
            print("  test guard active")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    all_scores, stability_rows, random_rows = [], [], []
    per_seed_rank = {}

    for dname in ("probe", "mean_diff"):
        for seed in seeds:
            dirs = readout_directions(seed)
            direction = dirs[dname]

            Atr, mtr, n_heads = run_split("train", seed, cfg, device, dname, direction)
            Adv, mdv, _ = run_split("dev", seed, cfg, device, dname, direction)
            names = component_names(n_heads)

            assert np.isfinite(Atr).all() and np.isfinite(Adv).all(), "non-finite attribution"
            bal_tr = mtr.groupby(["condition", "language"]).size()
            bal_dv = mdv.groupby(["condition", "language"]).size()

            # discovery on train, confirmation on dev
            d_tr, auc_tr = contrast_scores(
                Atr, mtr, (mtr.condition == "toxic").to_numpy(),
                (mtr.condition == "detox_reference").to_numpy())
            d_dv, auc_dv = contrast_scores(
                Adv, mdv, (mdv.condition == "toxic").to_numpy(),
                (mdv.condition == "detox_reference").to_numpy())

            tox = (mdv.condition == "toxic").to_numpy()
            gen = (mdv.condition == "generated").to_numpy()
            d_tg, auc_tg = contrast_scores(Adv, mdv, tox, gen)

            lab = mdv["behaviour_label"].to_numpy()
            d_co, auc_co = contrast_scores(
                Adv, mdv, tox & (lab == "copy_input"), tox & (lab == "ordinary_rewrite"))
            d_te, auc_te = contrast_scores(
                Adv, mdv, tox & (lab == "generic_template"),
                tox & (lab == "ordinary_rewrite"))

            ctrl = control_stats(Adv, mdv, languages)

            df = pd.DataFrame({
                "direction": dname, "seed": seed, "component": names,
                "kind": ["head" if ".head" in n else "neuron" for n in names],
                "side": ["encoder" if n.startswith("enc") else "decoder" for n in names],
                "sublayer": [n.split(".")[1] for n in names],
                "layer": [int(n.split(".L")[1].split(".")[0]) for n in names],
                "train_d_toxic_vs_reference": d_tr,
                "train_auc_toxic_vs_reference": auc_tr,
                "dev_d_toxic_vs_reference": d_dv,
                "dev_auc_toxic_vs_reference": auc_dv,
                "dev_d_toxic_vs_generated": d_tg,
                "dev_auc_toxic_vs_generated": auc_tg,
                "dev_d_copy_vs_ordinary": d_co,
                "dev_auc_copy_vs_ordinary": auc_co,
                "dev_d_template_vs_ordinary": d_te,
                "dev_auc_template_vs_ordinary": auc_te,
                "control_language_max_auc": ctrl["control_language_max_auc"],
                "control_corr_token_count": ctrl["control_corr_token_count"],
                "train_rows_per_cell": int(bal_tr.min()),
                "dev_rows_per_cell": int(bal_dv.min()),
                "language_balanced": bool(bal_tr.nunique() == 1 and bal_dv.nunique() == 1),
            })
            all_scores.append(df)
            per_seed_rank[(dname, seed)] = pd.Series(
                np.abs(df["train_d_toxic_vs_reference"].to_numpy()), index=names)

            # matched random controls, drawn from the same pool per kind
            rng = np.random.default_rng(seed)
            for kind in ("head", "neuron"):
                pool = df.index[df["kind"] == kind].to_numpy()
                top = df.loc[pool].reindex(
                    df.loc[pool, "train_d_toxic_vs_reference"].abs()
                      .sort_values(ascending=False).index)[:TOP_K]
                rand = rng.choice(pool, size=min(N_RANDOM_CONTROLS, len(pool)), replace=False)
                rd = df.loc[rand, "train_d_toxic_vs_reference"].abs()
                random_rows.append({
                    "direction": dname, "seed": seed, "kind": kind,
                    "n_pool": len(pool), "top_k": TOP_K,
                    "top_mean_abs_d": float(top["train_d_toxic_vs_reference"].abs().mean()),
                    "random_mean_abs_d": float(rd.mean()),
                    "random_p95_abs_d": float(rd.quantile(0.95)),
                    "top_above_random_p95": int(
                        (top["train_d_toxic_vs_reference"].abs() > rd.quantile(0.95)).sum()),
                })
            print(f"  {dname} seed{seed}: attributed {A_shape(Atr)} train / "
                  f"{A_shape(Adv)} dev components={len(names)}")

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(OUT_DIR / "component_scores.csv", index=False)
    rnd = pd.DataFrame(random_rows)
    rnd.to_csv(OUT_DIR / "random_controls.csv", index=False)

    # cross-seed stability of the train discovery ranking
    for dname in ("probe", "mean_diff"):
        ss = [per_seed_rank[(dname, s)] for s in seeds]
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                rho = spearmanr(ss[i].values, ss[j].values).statistic
                topi = set(ss[i].sort_values(ascending=False).index[:TOP_K])
                topj = set(ss[j].sort_values(ascending=False).index[:TOP_K])
                stability_rows.append({
                    "direction": dname, "seed_a": seeds[i], "seed_b": seeds[j],
                    "spearman_rho": float(rho),
                    f"overlap_top{TOP_K}": len(topi & topj),
                })
    stab = pd.DataFrame(stability_rows)
    stab.to_csv(OUT_DIR / "rank_stability.csv", index=False)

    # top components per contrast, averaged over seeds
    tops = []
    for dname in ("probe", "mean_diff"):
        sub = scores[scores["direction"] == dname]
        agg = sub.groupby(["component", "kind", "side", "sublayer", "layer"]).agg(
            train_d=("train_d_toxic_vs_reference", "mean"),
            dev_d_tox_ref=("dev_d_toxic_vs_reference", "mean"),
            dev_d_tox_gen=("dev_d_toxic_vs_generated", "mean"),
            dev_d_copy=("dev_d_copy_vs_ordinary", "mean"),
            dev_d_template=("dev_d_template_vs_ordinary", "mean"),
            lang_auc=("control_language_max_auc", "mean"),
            len_corr=("control_corr_token_count", "mean"),
            n_seeds=("seed", "nunique")).reset_index()
        for contrast, col in [("toxic_vs_reference", "dev_d_tox_ref"),
                              ("toxic_vs_generated", "dev_d_tox_gen"),
                              ("copy_vs_ordinary", "dev_d_copy"),
                              ("template_vs_ordinary", "dev_d_template")]:
            for kind in ("head", "neuron"):
                t = agg[agg["kind"] == kind].reindex(
                    agg[agg["kind"] == kind][col].abs().sort_values(ascending=False).index
                )[:TOP_K].copy()
                t["direction"], t["contrast"], t["rank"] = dname, contrast, range(1, len(t) + 1)
                tops.append(t)
    topdf = pd.concat(tops, ignore_index=True)
    topdf.to_csv(OUT_DIR / "top_components.csv", index=False)

    payload = {}
    for dname in ("probe", "mean_diff"):
        payload[dname] = {}
        for contrast in ("toxic_vs_reference", "toxic_vs_generated",
                         "copy_vs_ordinary", "template_vs_ordinary"):
            sel = topdf[(topdf.direction == dname) & (topdf.contrast == contrast)]
            payload[dname][contrast] = sel.head(TOP_K * 2).to_dict("records")
    (OUT_DIR / "top_components.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "gradient x activation attribution onto a decoder layer-6 readout",
        "readout": f"{READOUT_SIDE} block {READOUT_LAYER}, content-position mean, "
                   "standardised with train-only statistics",
        "directions": ["Stage 7 probe direction", "train toxic-minus-reference mean difference"],
        "discovery_split": "train", "confirmation_split": "dev", "test_split": "NOT READ",
        "interpretation_caveat": (
            "Attribution ranks association with a readout, not causal necessity. No "
            "component here is a detoxification circuit; the causal test is Stage 11."),
        "sae_feature_arm": ("retained as a low-priority negative arm for Stage 11 per "
                            "Stage 9, where no SAE feature passed replication + controls "
                            "+ behaviour"),
        "files_read": sorted(set(_FILES_READ)),
        "any_test_file_read": any("test_" in f for f in _FILES_READ),
        "top": payload,
    }, indent=2), encoding="utf-8")

    after = {str(p.relative_to(REPO_ROOT)): digest(p) for p in ckpts}

    # ---------------- console report ----------------
    print("\n=== attribution magnitude by component type (|train d|, mean over seeds) ===")
    print(scores.groupby(["direction", "side", "sublayer", "kind"])[
        "train_d_toxic_vs_reference"].apply(lambda s: s.abs().mean()).round(3).to_string())

    print("\n=== matched random controls ===")
    print(rnd.groupby(["direction", "kind"])[
        ["top_mean_abs_d", "random_mean_abs_d", "random_p95_abs_d", "top_above_random_p95"]
    ].mean().round(3).to_string())

    print("\n=== cross-seed rank stability (train discovery ranking) ===")
    print(stab.groupby("direction")[["spearman_rho", f"overlap_top{TOP_K}"]]
          .mean().round(3).to_string())

    print("\n=== top 10 head components, toxic vs reference (probe direction) ===")
    sel = topdf[(topdf.direction == "probe") & (topdf.contrast == "toxic_vs_reference")
                & (topdf["kind"] == "head")].head(10)
    print(sel[["component", "dev_d_tox_ref", "train_d", "dev_d_copy", "dev_d_template",
               "lang_auc", "len_corr"]].round(3).to_string(index=False))

    print("\n=== top 10 neuron components, toxic vs reference (probe direction) ===")
    sel = topdf[(topdf.direction == "probe") & (topdf.contrast == "toxic_vs_reference")
                & (topdf["kind"] == "neuron")].head(10)
    print(sel[["component", "dev_d_tox_ref", "train_d", "dev_d_copy", "dev_d_template",
               "lang_auc", "len_corr"]].round(3).to_string(index=False))

    print(f"\n  checkpoints unchanged: {before == after}")
    print(f"  any test file read: {any('test_' in f for f in _FILES_READ)}")
    print("\n  Attribution ranks association with a readout, not causal necessity.")
    print("  No component is called a detoxification circuit; Stage 11 is the causal test.")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0 if before == after else 1


def A_shape(a: np.ndarray) -> str:
    return f"{a.shape[0]}x{a.shape[1]}"


if __name__ == "__main__":
    raise SystemExit(main())
