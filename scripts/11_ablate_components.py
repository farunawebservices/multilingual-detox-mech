#!/usr/bin/env python
"""Stage 11 -- Causal ablation of the Stage 10 components.

Ablates each component group during generation and measures the change in STA, SIM, FL
and J against the unablated baseline. The design is frozen to disk before any evaluation
runs, dev is evaluated first, and test is evaluated exactly once afterwards using that
same frozen design.

Usage
-----
    python scripts/11_ablate_components.py --freeze        # write the frozen design only
    python scripts/11_ablate_components.py --phase dev     # dev evaluation
    python scripts/11_ablate_components.py --phase test    # single test evaluation

Ablation method: mean ablation, with means from train
-----------------------------------------------------
A component is replaced by its train-set mean activation rather than by zero. Zero
ablation would be the bigger intervention here, not the smaller one: Stage 6 measured
mT5 activations peaking around 2e5 with per-dimension standard deviations spanning 16.8
to 62,838, so forcing a component to zero injects a large off-distribution shift and
conflates "this component's information was removed" with "the residual stream was
knocked far off its manifold". Mean ablation removes the component's *variation* while
leaving the stream where the model expects it. Means are computed on train only.

Frozen design
-------------
Component membership for every arm is derived from Stage 10 scores (train discovery, dev
confirmation) and written to results/ablation/frozen_design.json with a SHA-256 digest
before any generation happens. The test phase reloads that file and verifies the digest,
so the test evaluation cannot be influenced by anything seen on dev.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Random controls are matched on component type, sublayer and layer, and count. For each
  ablated head the control draws a different head from the same (side, sublayer, layer);
  for each neuron, a different neuron from the same layer. This makes the control share
  the ablation's structural footprint rather than merely its size.
* Arm sizes are 10 heads, 50 neurons, and 25 for mixed arms. Heads are far coarser than
  neurons -- one head is 64 dimensions against a neuron's one -- so equal counts would
  make the head arms a much larger intervention.
* SAE features are ablated by subtracting the feature's own contribution,
  x <- x - f_j * W_dec[j] * sd, rather than by replacing the activation with an SAE
  reconstruction. Stage 8 measured decoder reconstruction at 0.75 explained variance, so
  round-tripping through the SAE would perturb the model far more than removing one
  feature does, and any resulting metric change would be reconstruction error rather than
  the feature's causal role.
* Per-language macro-averaging is used for headline numbers. Dev and test are already
  27 and 26 rows per language, so micro and macro agree; macro is reported so the
  language balance is explicit rather than incidental.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sacrebleu.metrics import CHRF
from sklearn.metrics import roc_auc_score  # noqa: F401  (kept for parity of imports)
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          MT5ForConditionalGeneration)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
ACT_DIR = REPO_ROOT / "results" / "activations"
ATTR_DIR = REPO_ROOT / "results" / "attribution"
SAE_FEAT_DIR = REPO_ROOT / "results" / "sae_features"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
OUT_DIR = REPO_ROOT / "results" / "ablation"

TASK_PREFIX = "detoxify: "
WS_RE = re.compile(r"\s+")
MAX_SOURCE, MAX_TARGET = 96, 112
GEN_BATCH = 32
DECODING = dict(num_beams=4, do_sample=False, length_penalty=1.0,
                early_stopping=True, max_new_tokens=112)
K_HEADS, K_NEURONS, K_MIXED = 10, 50, 25
LANG_DETECTOR_AUC = 0.85
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def norm(t: str) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(t))).strip()


def parse_component(name: str) -> dict:
    """'dec.cross.L5.head7' / 'enc.mlp.L8.n1167' -> structured spec."""
    side, sublayer, rest = name.split(".", 2)
    layer = int(rest.split(".")[0][1:])
    unit = rest.split(".")[1]
    kind = "head" if unit.startswith("head") else "neuron"
    idx = int(unit[4:]) if kind == "head" else int(unit[1:])
    return {"name": name, "side": "encoder" if side == "enc" else "decoder",
            "sublayer": sublayer, "layer": layer, "kind": kind, "index": idx}


# ------------------------------------------------------------------- frozen design

def build_design(cfg: dict) -> dict:
    scores = pd.read_csv(ATTR_DIR / "component_scores.csv")
    agg = scores.groupby(["direction", "component", "kind", "side", "sublayer", "layer"]).agg(
        dev_d=("dev_d_toxic_vs_reference", "mean"),
        train_d=("train_d_toxic_vs_reference", "mean"),
        lang_auc=("control_language_max_auc", "mean")).reset_index()

    def top(df: pd.DataFrame, k: int) -> list[str]:
        return df.reindex(df["dev_d"].abs().sort_values(ascending=False).index)["component"].head(k).tolist()

    probe = agg[agg["direction"] == "probe"]
    mdiff = agg[agg["direction"] == "mean_diff"]

    arms: dict[str, dict] = {}
    arms["top_decoder_cross_heads"] = {"components": top(
        probe[(probe.side == "decoder") & (probe.sublayer == "cross") & (probe["kind"] == "head")],
        K_HEADS)}
    arms["top_decoder_self_heads"] = {"components": top(
        probe[(probe.side == "decoder") & (probe.sublayer == "self") & (probe["kind"] == "head")],
        K_HEADS)}
    arms["top_decoder_mlp_neurons"] = {"components": top(
        probe[(probe.side == "decoder") & (probe["kind"] == "neuron")], K_NEURONS)}
    arms["top_encoder_components"] = {"components": top(
        probe[probe.side == "encoder"], K_MIXED)}
    arms["top_probe_direction"] = {"components": top(probe, K_MIXED)}
    arms["top_mean_diff_direction"] = {"components": top(mdiff, K_MIXED)}
    arms["language_detectors"] = {"components": top(
        probe[probe["lang_auc"] > LANG_DETECTOR_AUC], K_MIXED)}

    # matched random controls: same (side, sublayer, layer) per head, same layer per neuron
    rng = np.random.default_rng(42)
    pool = agg[agg["direction"] == "probe"]

    def matched_random(targets: list[str]) -> list[str]:
        picks = []
        for name in targets:
            spec = parse_component(name)
            if spec["kind"] == "head":
                cand = pool[(pool.side == spec["side"]) & (pool.sublayer == spec["sublayer"])
                            & (pool.layer == spec["layer"]) & (pool["kind"] == "head")]
            else:
                cand = pool[(pool.side == spec["side"]) & (pool["kind"] == "neuron")
                            & (pool.layer == spec["layer"])]
            cand = cand[~cand["component"].isin(targets + picks)]
            if len(cand):
                picks.append(str(rng.choice(cand["component"].to_numpy())))
        return picks

    head_targets = arms["top_decoder_cross_heads"]["components"] + \
        arms["top_decoder_self_heads"]["components"]
    arms["random_heads_matched"] = {"components": matched_random(head_targets),
                                    "matched_to": "top_decoder_cross_heads + top_decoder_self_heads"}
    arms["random_neurons_matched"] = {"components": matched_random(
        arms["top_decoder_mlp_neurons"]["components"]),
        "matched_to": "top_decoder_mlp_neurons"}

    # SAE arm
    sae_path = SAE_FEAT_DIR / "replicated_features.csv"
    if sae_path.exists():
        rep = pd.read_csv(sae_path)
        rep = rep[(rep["layer"] == 6) & (rep["token_set"] == "content")]
        feats = (rep[["side", "seed", "feature"]].drop_duplicates()
                 .to_dict("records"))
        arms["sae_replicated_features"] = {
            "components": [], "sae_features": feats,
            "status": "available" if feats else "unavailable",
            "note": ("Stage 9 replicated features; none passed the full promotion "
                     "criteria, so this is retained as a negative arm"),
        }
    else:
        arms["sae_replicated_features"] = {
            "components": [], "sae_features": [], "status": "unavailable",
            "note": "no Stage 9 replicated-feature table found"}

    design = {
        "run_id": cfg["run_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ablation": "mean ablation with train-derived component means",
        "arm_sizes": {"heads": K_HEADS, "neurons": K_NEURONS, "mixed": K_MIXED},
        "ranking_source": "results/attribution/component_scores.csv (train discovery, dev confirm)",
        "arms": arms,
    }
    design["digest"] = hashlib.sha256(
        json.dumps(design["arms"], sort_keys=True).encode()).hexdigest()
    return design


# ---------------------------------------------------------------- ablation hooks

class Ablator:
    """Mean-ablates the listed components, and optionally subtracts SAE features."""

    def __init__(self, model, specs: list[dict], means: dict[str, torch.Tensor],
                 sae_specs: list[dict] | None = None, sae_bundle: dict | None = None):
        self.handles = []
        self.by_module: dict[tuple, list[dict]] = {}
        for s in specs:
            key = (s["side"], s["sublayer"], s["layer"], s["kind"])
            self.by_module.setdefault(key, []).append(s)

        for (side, sublayer, layer, kind), items in self.by_module.items():
            stack = model.encoder if side == "encoder" else model.decoder
            if kind == "head":
                attn = (stack.block[layer].layer[0].SelfAttention if sublayer == "self"
                        else stack.block[layer].layer[1].EncDecAttention)
                mod, idxs = attn.o, [i["index"] for i in items]
                mkey = f"{side}.{sublayer}.{layer}.head"
                self.handles.append(mod.register_forward_pre_hook(
                    self._head_hook(idxs, means.get(mkey), model.config.num_heads,
                                    model.config.d_kv)))
            else:
                mod = stack.block[layer].layer[-1].DenseReluDense.wo
                idxs = [i["index"] for i in items]
                mkey = f"{side}.mlp.{layer}.neuron"
                self.handles.append(mod.register_forward_pre_hook(
                    self._neuron_hook(idxs, means.get(mkey))))

        if sae_specs and sae_bundle:
            for spec in sae_specs:
                side, feat = spec["side"], int(spec["feature"])
                b = sae_bundle[side]
                stack = model.encoder if side == "encoder" else model.decoder
                self.handles.append(stack.block[6].register_forward_hook(
                    self._sae_hook(b["sae"], feat, b["mu"], b["sd"])))

    @staticmethod
    def _head_hook(idxs, mean, n_heads, d_kv):
        def hook(_m, inputs):
            x = inputs[0].clone()
            B, T, D = x.shape
            v = x.view(B, T, n_heads, d_kv)
            for i in idxs:
                v[:, :, i, :] = (mean[i].to(x.device, x.dtype) if mean is not None else 0.0)
            return (v.view(B, T, D),)
        return hook

    @staticmethod
    def _neuron_hook(idxs, mean):
        def hook(_m, inputs):
            x = inputs[0].clone()
            for i in idxs:
                x[:, :, i] = (mean[i].to(x.device, x.dtype) if mean is not None else 0.0)
            return (x,)
        return hook

    @staticmethod
    def _sae_hook(sae, feat, mu, sd):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            z = (h - mu.to(h.device)) / sd.to(h.device).clamp_min(1e-6)
            f = sae.encode(z.reshape(-1, z.shape[-1]).float())
            corr = (f[:, feat:feat + 1] * sae.W_dec[feat:feat + 1]).reshape(h.shape)
            new = h - corr.to(h.dtype) * sd.to(h.device)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def component_means(model, tokenizer, train_df: pd.DataFrame, device: str,
                    specs: list[dict]) -> dict[str, torch.Tensor]:
    """Train-set mean activation per component, averaged over valid positions."""
    needed = {(s["side"], s["sublayer"], s["layer"], s["kind"]) for s in specs}
    sums, counts, store = {}, {}, {}
    handles = []

    def cap(key):
        def hook(_m, inputs):
            store[key] = inputs[0].detach()
        return hook

    for side, sublayer, layer, kind in needed:
        stack = model.encoder if side == "encoder" else model.decoder
        if kind == "head":
            attn = (stack.block[layer].layer[0].SelfAttention if sublayer == "self"
                    else stack.block[layer].layer[1].EncDecAttention)
            key = f"{side}.{sublayer}.{layer}.head"
            handles.append(attn.o.register_forward_pre_hook(cap(key)))
        else:
            key = f"{side}.mlp.{layer}.neuron"
            handles.append(stack.block[layer].layer[-1].DenseReluDense.wo
                           .register_forward_pre_hook(cap(key)))

    n_heads, d_kv = model.config.num_heads, model.config.d_kv
    for start in range(0, len(train_df), GEN_BATCH):
        chunk = train_df.iloc[start:start + GEN_BATCH]
        enc = tokenizer([TASK_PREFIX + str(t) for t in chunk["toxic_input"]],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_SOURCE).to(device)
        tgt = tokenizer(text_target=[str(t) for t in chunk["detox_output"]],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_TARGET).to(device)
        store.clear()
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
              decoder_input_ids=model._shift_right(tgt.input_ids),
              decoder_attention_mask=tgt.attention_mask)
        for key, act in store.items():
            side = key.split(".")[0]
            mask = (enc.attention_mask if side == "encoder" else tgt.attention_mask).bool()
            sel = act[mask]                                    # [N, D]
            if key.endswith("head"):
                sel = sel.view(sel.shape[0], n_heads, d_kv)
            sums[key] = sums.get(key, 0) + sel.float().sum(0)
            counts[key] = counts.get(key, 0) + sel.shape[0]
    for h in handles:
        h.remove()
    return {k: (sums[k] / max(counts[k], 1)) for k in sums}


@torch.no_grad()
def generate(model, tokenizer, df: pd.DataFrame, device: str) -> list[str]:
    outs = []
    for start in range(0, len(df), GEN_BATCH):
        chunk = df.iloc[start:start + GEN_BATCH]
        enc = tokenizer([TASK_PREFIX + str(t) for t in chunk["toxic_input"]],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_SOURCE).to(device)
        g = model.generate(**enc, **DECODING)
        outs.extend(tokenizer.batch_decode(g, skip_special_tokens=True))
    return [norm(o) for o in outs]


# ---------------------------------------------------------------------- metrics

@torch.no_grad()
def score_all(frame: pd.DataFrame, cfg: dict, device: str) -> pd.DataFrame:
    sta_id, sim_id = cfg["models"]["sta"], cfg["models"]["sim"]
    tok = AutoTokenizer.from_pretrained(sta_id)
    clf = AutoModelForSequenceClassification.from_pretrained(sta_id).to(device).eval()
    texts = frame["generated"].fillna("").astype(str).tolist()
    sta = []
    for i in range(0, len(texts), 64):
        b = [t if t.strip() else " " for t in texts[i:i + 64]]
        e = tok(b, return_tensors="pt", padding=True, truncation=True,
                max_length=256).to(device)
        sta.extend(torch.softmax(clf(**e).logits.float(), -1)[:, 0].cpu().numpy())
    del clf
    torch.cuda.empty_cache()

    ltok = AutoTokenizer.from_pretrained(sim_id)
    lm = AutoModel.from_pretrained(sim_id).to(device).eval()

    def embed(ts):
        vs = []
        for i in range(0, len(ts), 64):
            b = [t if str(t).strip() else " " for t in ts[i:i + 64]]
            e = ltok(b, return_tensors="pt", padding=True, truncation=True,
                     max_length=256).to(device)
            vs.append(torch.nn.functional.normalize(lm(**e).pooler_output, dim=1).cpu())
        return torch.cat(vs)

    sim = (embed(frame["toxic_input"].astype(str).tolist()) * embed(texts)).sum(1).numpy()
    del lm
    torch.cuda.empty_cache()

    chrf = CHRF(beta=1)
    fl = np.array([chrf.sentence_score(t if t.strip() else " ",
                                       [str(r)]).score / 100.0
                   for t, r in zip(texts, frame["detox_reference"].astype(str))])

    out = frame.copy()
    supported = out["language"].isin(STA_SUPPORTED)
    out["sta_raw"] = sta
    out["sta"] = np.where(supported, sta, np.nan)
    out["sta_unvalidated"] = np.where(supported, np.nan, sta)
    out["sim"] = sim
    out["fl"] = fl
    out["j"] = out["sta"] * out["sim"] * out["fl"]
    out["j_unvalidated"] = out["sta_unvalidated"] * out["sim"] * out["fl"]
    return out


def load_sae_bundle(seed: int, device: str) -> dict:
    import torch.nn as nn

    class TopKSAE(nn.Module):
        def __init__(self, d_model, d_sae, k):
            super().__init__()
            self.k = k
            self.b_dec = nn.Parameter(torch.zeros(d_model))
            self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))

        def encode(self, x):
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            v, i = torch.topk(pre, self.k, -1)
            return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))

    bundle = {}
    for side in ("encoder", "decoder"):
        p = REPO_ROOT / "models" / f"sae_{side}_layer6" / f"sae_content_seed{seed}.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        sae = TopKSAE(768, d["d_sae"], d["k"])
        sae.load_state_dict(d["state_dict"])
        means, varis, counts = [], [], []
        for cond in ("toxic", "detox_reference"):
            st = torch.load(ACT_DIR / "norm_stats" /
                            f"seed{seed}_{side}_layer06_{cond}.pt", weights_only=False)
            means.append(st["token_mean"].double())
            varis.append(st["token_std"].double() ** 2)
            counts.append(float(st["n_tokens"]))
        tot = sum(counts)
        mu = sum(n * m for n, m in zip(counts, means)) / tot
        ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / tot
        bundle[side] = {"sae": sae.to(device).eval(), "mu": mu.float(),
                        "sd": (ex2 - mu ** 2).clamp_min(0).sqrt().float()}
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--phase", choices=["dev", "test"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design_path = OUT_DIR / "frozen_design.json"

    if args.freeze or not design_path.exists():
        design = build_design(cfg)
        design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
        print(f"froze ablation design -> {design_path.relative_to(REPO_ROOT)}")
        print(f"  digest: {design['digest'][:16]}...")
        for name, arm in design["arms"].items():
            n = len(arm["components"]) or len(arm.get("sae_features", []))
            print(f"  {name:28s} {n:3d} components  {arm.get('status','')}")
        if args.freeze:
            return 0

    design = json.loads(design_path.read_text())
    expect = hashlib.sha256(json.dumps(design["arms"], sort_keys=True).encode()).hexdigest()
    assert expect == design["digest"], "frozen design digest mismatch -- design was edited"
    print(f"loaded frozen design, digest verified: {design['digest'][:16]}...")

    split = args.phase
    if split is None:
        print("no --phase given; nothing evaluated")
        return 0
    if split == "test":
        stamp = OUT_DIR / "TEST_EVALUATED.txt"
        if stamp.exists():
            print(f"REFUSING: test was already evaluated ({stamp.read_text().strip()}). "
                  "The brief allows exactly one test evaluation.")
            return 1

    eval_df = pd.read_csv(SPLITS_DIR / f"{split}.csv").rename(
        columns={"detox_output": "detox_reference"})
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    print(f"{split}: {len(eval_df)} rows, "
          f"{eval_df.groupby('language').size().nunique() == 1 and 'language-balanced' or 'UNBALANCED'}"
          f" ({eval_df.groupby('language').size().iloc[0]} per language)")

    frames = []
    for seed in cfg["seeds_multi"]:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / f"seed{seed}"))
        model = MT5ForConditionalGeneration.from_pretrained(
            str(MODEL_DIR / f"seed{seed}")).to(device).eval()

        all_specs = [parse_component(c) for arm in design["arms"].values()
                     for c in arm["components"]]
        means = component_means(model, tokenizer, train_df, device, all_specs) if all_specs else {}
        sae_bundle = load_sae_bundle(seed, device)

        # baseline, preserved unablated
        base = eval_df.copy()
        base["generated"] = generate(model, tokenizer, eval_df, device)
        base["arm"] = "baseline_unablated"
        base["seed"] = seed
        base["n_components"] = 0
        frames.append(base)
        print(f"  seed{seed} baseline_unablated: {len(base)} generations")

        for arm_name, arm in design["arms"].items():
            specs = [parse_component(c) for c in arm["components"]]
            sae_specs = [s for s in arm.get("sae_features", []) if s.get("seed") == seed]
            if not specs and not sae_specs:
                rec = eval_df.copy()
                rec["generated"] = ""
                rec["arm"] = arm_name
                rec["seed"] = seed
                rec["n_components"] = 0
                rec["status"] = arm.get("status", "unavailable")
                frames.append(rec)
                print(f"  seed{seed} {arm_name}: UNAVAILABLE ({arm.get('note','')[:50]})")
                continue
            abl = Ablator(model, specs, means, sae_specs, sae_bundle)
            rec = eval_df.copy()
            rec["generated"] = generate(model, tokenizer, eval_df, device)
            abl.remove()
            rec["arm"] = arm_name
            rec["seed"] = seed
            rec["n_components"] = len(specs) + len(sae_specs)
            frames.append(rec)
            print(f"  seed{seed} {arm_name}: {len(specs) + len(sae_specs)} components ablated")

        del model
        torch.cuda.empty_cache()

    allgen = pd.concat(frames, ignore_index=True)
    print(f"scoring {len(allgen)} generations ...")
    scored = score_all(allgen, cfg, device)
    scored.to_csv(OUT_DIR / f"{split}_ablation_scores.csv", index=False)

    per_lang = scored.groupby(["arm", "seed", "language"])[
        ["sta", "sim", "fl", "j", "sta_unvalidated", "j_unvalidated"]].mean().reset_index()
    macro = per_lang.groupby(["arm", "seed"])[
        ["sta", "sim", "fl", "j", "sta_unvalidated", "j_unvalidated"]].mean().reset_index()

    base_macro = macro[macro["arm"] == "baseline_unablated"].set_index("seed")
    for m in ("sta", "sim", "fl", "j"):
        macro[f"delta_{m}"] = macro.apply(
            lambda r: r[m] - base_macro.loc[r["seed"], m], axis=1)
    macro.to_csv(OUT_DIR / f"{split}_macro_by_seed.csv", index=False)

    base_lang = per_lang[per_lang["arm"] == "baseline_unablated"].set_index(
        ["seed", "language"])
    for m in ("sta", "sim", "fl", "j"):
        per_lang[f"delta_{m}"] = per_lang.apply(
            lambda r: r[m] - base_lang.loc[(r["seed"], r["language"]), m], axis=1)
    per_lang.to_csv(OUT_DIR / f"{split}_per_language.csv", index=False)

    summary = macro.groupby("arm").agg(
        n_seeds=("seed", "nunique"),
        sta=("sta", "mean"), sim=("sim", "mean"), fl=("fl", "mean"), j=("j", "mean"),
        d_sta=("delta_sta", "mean"), d_sta_sd=("delta_sta", "std"),
        d_sim=("delta_sim", "mean"), d_sim_sd=("delta_sim", "std"),
        d_fl=("delta_fl", "mean"), d_j=("delta_j", "mean"), d_j_sd=("delta_j", "std"),
    ).reset_index()
    summary.to_csv(OUT_DIR / f"{split}_summary.csv", index=False)

    if split == "test":
        (OUT_DIR / "TEST_EVALUATED.txt").write_text(
            f"test evaluated once at {datetime.now(timezone.utc).isoformat()} "
            f"under frozen design {design['digest']}\n", encoding="utf-8")

    print(f"\n=== {split.upper()}: macro-averaged over languages, mean over 3 seeds ===")
    print(summary[["arm", "n_seeds", "sta", "sim", "fl", "j",
                   "d_sta", "d_sim", "d_fl", "d_j"]].round(4).to_string(index=False))

    print("\n=== SIM collapse check (does any arm raise STA while SIM falls?) ===")
    for _, r in summary.iterrows():
        if r["arm"] == "baseline_unablated":
            continue
        if r["d_sta"] > 0.01 and r["d_sim"] < -0.02:
            print(f"  FLAG {r['arm']}: dSTA={r['d_sta']:+.4f} with dSIM={r['d_sim']:+.4f} "
                  "-- higher STA bought by discarding content")
    if not ((summary["d_sta"] > 0.01) & (summary["d_sim"] < -0.02)).any():
        print("  none flagged")

    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/{split}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
