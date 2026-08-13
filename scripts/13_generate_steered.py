#!/usr/bin/env python
"""Stage 13 -- Inference-time activation steering.

Sweeps steering strength on dev across every arm and scope, selects a configuration
using dev only, freezes it, and then evaluates test exactly once.

Usage
-----
    python scripts/13_generate_steered.py --phase dev     # full dev sweep
    python scripts/13_generate_steered.py --select        # freeze the test configuration
    python scripts/13_generate_steered.py --phase test    # single test evaluation

Injection
---------
A forward hook on {encoder,decoder}.block[6] adds `alpha * v * sd` to the block output at
every non-pad position, where v is the unit vector from Stage 12 and sd is the
standard-deviation vector from the train-only statistics that vector was built in. This
is exactly the mapping recorded as `injection_note` in the Stage 12 manifest.

Rows are batched by language even for pooled scopes, so a batch always shares one vector.
That keeps the hook independent of beam expansion, which otherwise reorders and repeats
the batch dimension underneath a per-row vector.

Selection rule, and why it differs for Yoruba and isiXhosa
----------------------------------------------------------
For the seven languages the toxicity classifier covers, selection maximises dev J.
For Yoruba and isiXhosa the classifier has no validated coverage, so STA and therefore J
cannot be used. Selection there maximises SIM * FL subject to degeneracy not worsening
against baseline: unique-output rate must not fall by more than 0.05, and neither the
loose-copy rate nor the template rate may rise by more than 0.05. STA is reported for
those two languages only in the `_unvalidated` columns and never enters the choice.

Test discipline
---------------
The dev sweep writes a selection; --select freezes it to frozen_test_config.json with a
SHA-256 digest; --phase test reloads that file, verifies the digest, and refuses to run
if the stamp file shows test was already evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sacrebleu.metrics import CHRF
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          MT5ForConditionalGeneration)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
ACT_DIR = REPO_ROOT / "results" / "activations"
VEC_DIR = REPO_ROOT / "results" / "steering_vectors"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
OUT_DIR = REPO_ROOT / "results" / "steering"

TASK_PREFIX = "detoxify: "
WS_RE = re.compile(r"\s+")
SENTINEL_RE = re.compile(r"<extra_id_\d+>")
MAX_SOURCE = 96
DECODING = dict(num_beams=4, do_sample=False, length_penalty=1.0,
                early_stopping=True, max_new_tokens=112)
STRENGTHS = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
LAYER = 6
POOLING = "pooled_mean_content"
SCOPES = ["per_language", "pooled_non_african", "pooled_all_balanced", "pooled_african"]
ARMS = [
    ("decoder", "mean_diff", "primary"),
    ("decoder", "probe", "primary"),
    ("encoder", "mean_diff", "comparison"),
    ("encoder", "probe", "comparison"),
    ("decoder", "random", "random_control"),
    ("decoder", "sae", "sae_control_only"),
]
AFRICAN = ["yo", "xh"]
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}
# Guardrails for the yo/xh selection, which cannot use STA.
MAX_UNIQUE_DROP, MAX_COPY_RISE, MAX_TEMPLATE_RISE = 0.05, 0.05, 0.05


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def norm(t: str) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(t))).strip()


def loose(t: str) -> str:
    return norm(t).casefold()


def sd_vector(seed: int, side: str) -> torch.Tensor:
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
    return (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def sd_vector_token(seed: int, side: str) -> torch.Tensor:
    means, varis, counts = [], [], []
    for cond in ("toxic", "detox_reference"):
        st = torch.load(ACT_DIR / "norm_stats" /
                        f"seed{seed}_{side}_layer{LAYER:02d}_{cond}.pt", weights_only=False)
        means.append(st["token_mean"].double())
        varis.append(st["token_std"].double() ** 2)
        counts.append(float(st["n_tokens"]))
    tot = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / tot
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / tot
    return (ex2 - mu ** 2).clamp_min(0.0).sqrt().float()


def vector_key(dtype: str, scope: str, side: str, seed: int, lang: str | None) -> str:
    s = f"lang_{lang}" if scope == "per_language" else scope
    if dtype == "random":
        return f"random|{s}|{side}|L{LAYER}|{POOLING}|seed{seed}|match_mean_diff"
    return f"{dtype}|{s}|{side}|L{LAYER}|{POOLING}|seed{seed}"


class Steerer:
    def __init__(self, model, side: str, vec: torch.Tensor, sd: torch.Tensor, alpha: float):
        stack = model.encoder if side == "encoder" else model.decoder
        delta = (alpha * vec * sd)

        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            new = h + delta.to(h.device, h.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

        self.handle = stack.block[LAYER].register_forward_hook(hook)

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def generate_by_language(model, tok, df: pd.DataFrame, device: str, side: str,
                         vectors: dict, vec_for_lang, sd: torch.Tensor,
                         alpha: float) -> pd.DataFrame:
    """One pass over df, batching by language so a batch shares one steering vector."""
    outs = []
    for lang, chunk in df.groupby("language", sort=True):
        steer = None
        if alpha != 0.0 and vec_for_lang is not None:
            key = vec_for_lang(lang)
            if key is not None and key in vectors:
                v = torch.tensor(vectors[key], dtype=torch.float32)
                steer = Steerer(model, side, v, sd, alpha)
        enc = tok([TASK_PREFIX + str(t) for t in chunk["toxic_input"]],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_SOURCE).to(device)
        g = model.generate(**enc, **DECODING)
        text = tok.batch_decode(g, skip_special_tokens=True)
        if steer is not None:
            steer.remove()
        rec = chunk.copy()
        rec["generated"] = [norm(t) for t in text]
        outs.append(rec)
    return pd.concat(outs, ignore_index=True)


# ---------------------------------------------------------------------- metrics

@torch.no_grad()
def score_frame(frame: pd.DataFrame, cfg: dict, device: str) -> pd.DataFrame:
    sta_id, sim_id = cfg["models"]["sta"], cfg["models"]["sim"]
    tok = AutoTokenizer.from_pretrained(sta_id)
    clf = AutoModelForSequenceClassification.from_pretrained(sta_id).to(device).eval()

    def neutrality(texts):
        out = []
        for i in range(0, len(texts), 128):
            b = [t if str(t).strip() else " " for t in texts[i:i + 128]]
            e = tok(b, return_tensors="pt", padding=True, truncation=True,
                    max_length=256).to(device)
            out.extend(torch.softmax(clf(**e).logits.float(), -1)[:, 0].cpu().numpy())
        return np.asarray(out)

    gen = frame["generated"].fillna("").astype(str).tolist()
    sta_out = neutrality(gen)
    sta_in = neutrality(frame["toxic_input"].astype(str).tolist())
    del clf
    torch.cuda.empty_cache()

    ltok = AutoTokenizer.from_pretrained(sim_id)
    lm = AutoModel.from_pretrained(sim_id).to(device).eval()

    def embed(ts):
        vs = []
        for i in range(0, len(ts), 128):
            b = [t if str(t).strip() else " " for t in ts[i:i + 128]]
            e = ltok(b, return_tensors="pt", padding=True, truncation=True,
                     max_length=256).to(device)
            vs.append(torch.nn.functional.normalize(lm(**e).pooler_output, dim=1).cpu())
        return torch.cat(vs)

    sim = (embed(frame["toxic_input"].astype(str).tolist()) * embed(gen)).sum(1).numpy()
    del lm
    torch.cuda.empty_cache()

    chrf = CHRF(beta=1)
    fl = np.array([chrf.sentence_score(t if t.strip() else " ", [str(r)]).score / 100.0
                   for t, r in zip(gen, frame["detox_reference"].astype(str))])

    out = frame.copy()
    sup = out["language"].isin(STA_SUPPORTED)
    out["sta"] = np.where(sup, sta_out, np.nan)
    out["sta_unvalidated"] = np.where(sup, np.nan, sta_out)
    out["sta_input"] = np.where(sup, sta_in, np.nan)
    out["delta_sta"] = out["sta"] - out["sta_input"]
    out["sim"] = sim
    out["fl"] = fl
    out["j"] = out["sta"] * out["sim"] * out["fl"]
    out["sim_fl"] = out["sim"] * out["fl"]
    out["is_empty"] = out["generated"].str.len() == 0
    out["has_sentinel"] = [bool(SENTINEL_RE.search(t)) for t in gen]
    out["is_copy"] = [norm(g) == norm(t) for g, t in zip(out["generated"], out["toxic_input"])]
    out["is_copy_loose"] = [loose(g) == loose(t)
                            for g, t in zip(out["generated"], out["toxic_input"])]
    return out


def aggregate(scored: pd.DataFrame, train_freq: dict) -> pd.DataFrame:
    rows = []
    keys = ["arm", "side", "direction_type", "scope", "strength", "seed", "language"]
    for k, g in scored.groupby(keys):
        texts = [norm(t) for t in g["generated"]]
        counts = Counter(texts)
        n = len(texts)
        lang = k[-1]
        rows.append({
            **dict(zip(keys, k)), "n": n,
            "sta": g["sta"].mean(), "sta_unvalidated": g["sta_unvalidated"].mean(),
            "delta_sta": g["delta_sta"].mean(),
            "sim": g["sim"].mean(), "fl": g["fl"].mean(), "j": g["j"].mean(),
            "sim_fl": g["sim_fl"].mean(),
            "copy_rate": g["is_copy"].mean(),
            "loose_copy_rate": g["is_copy_loose"].mean(),
            "unique_output_rate": len(counts) / n if n else np.nan,
            "top_output_share": counts.most_common(1)[0][1] / n if n else np.nan,
            "template_rate": float(np.mean([t in train_freq.get(lang, set()) for t in texts])),
            "empty_rate": g["is_empty"].mean(),
            "sentinel_rate": g["has_sentinel"].mean(),
            "sta_supported": lang in STA_SUPPORTED,
        })
    return pd.DataFrame(rows)


def train_frequent_outputs() -> dict:
    tr = pd.read_csv(SPLITS_DIR / "train.csv")
    out = {}
    for lang, g in tr.groupby("language"):
        c = Counter(norm(t) for t in g["detox_output"])
        out[lang] = {t for t, n in c.items() if n >= 2}
    return out


# ------------------------------------------------------------------------ phases

def run_phase(split: str, cfg: dict, device: str, config_filter=None) -> pd.DataFrame:
    vectors = dict(np.load(VEC_DIR / "vectors.npz"))
    man = pd.read_csv(VEC_DIR / "manifest.csv")
    sae_keys = man[man["arm"] == "sae_control_only"].set_index("seed")["key"].to_dict() \
        if (man["arm"] == "sae_control_only").any() else {}
    sae_by_seed = {}
    for _, r in man[man["arm"] == "sae_control_only"].iterrows():
        sae_by_seed.setdefault(int(r["seed"]), []).append(r["key"])

    df = pd.read_csv(SPLITS_DIR / f"{split}.csv").rename(
        columns={"detox_output": "detox_reference"})
    frames = []
    out_path = OUT_DIR / f"{split}_raw_generations.csv"
    first = True

    for seed in cfg["seeds_multi"]:
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / f"seed{seed}"))
        model = MT5ForConditionalGeneration.from_pretrained(
            str(MODEL_DIR / f"seed{seed}")).to(device).eval()
        sd_pool = {s: sd_vector(seed, s) for s in ("encoder", "decoder")}
        sd_tok = {s: sd_vector_token(seed, s) for s in ("encoder", "decoder")}

        # unsteered baseline, preserved exactly, reused for every strength-0 cell
        base = generate_by_language(model, tok, df, device, "decoder", vectors, None,
                                    sd_pool["decoder"], 0.0)
        for side, dtype, arm in ARMS:
            for scope in (["sae_features"] if dtype == "sae" else SCOPES):
                b = base.copy()
                b["arm"], b["side"], b["direction_type"] = arm, side, dtype
                b["scope"], b["strength"], b["seed"] = scope, 0.0, seed
                frames.append(b)

        for side, dtype, arm in ARMS:
            scopes = ["sae_features"] if dtype == "sae" else SCOPES
            for scope in scopes:
                for alpha in [s for s in STRENGTHS if s != 0.0]:
                    if config_filter and not config_filter(side, dtype, scope, alpha, seed):
                        continue
                    if dtype == "sae":
                        keys = sae_by_seed.get(seed, [])
                        if not keys:
                            continue
                        key = keys[0]
                        vfl = (lambda _l, k=key: k)
                        sdv = sd_tok[side]
                    else:
                        vfl = (lambda l, dt=dtype, sc=scope, sd_=side, se=seed:
                               vector_key(dt, sc, sd_, se, l))
                        sdv = sd_pool[side]
                    rec = generate_by_language(model, tok, df, device, side, vectors,
                                               vfl, sdv, alpha)
                    rec["arm"], rec["side"], rec["direction_type"] = arm, side, dtype
                    rec["scope"], rec["strength"], rec["seed"] = scope, alpha, seed
                    frames.append(rec)
                    rec.to_csv(out_path, mode="w" if first else "a",
                               header=first, index=False)
                    first = False
                print(f"  seed{seed} {arm}/{side}/{dtype}/{scope}: swept "
                      f"{len([s for s in STRENGTHS if s != 0])} strengths")
        del model
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def select_config(agg: pd.DataFrame, cfg: dict) -> dict:
    """Dev-only selection. J for covered languages; SIM*FL plus guardrails for yo/xh."""
    primary = agg[agg["arm"].isin(["primary", "comparison"])]
    base = agg[(agg["strength"] == 0.0)].groupby("language")[
        ["unique_output_rate", "loose_copy_rate", "template_rate"]].mean()

    chosen = {}
    for lang, g in primary.groupby("language"):
        m = g.groupby(["side", "direction_type", "scope", "strength"]).agg(
            j=("j", "mean"), sim=("sim", "mean"), fl=("fl", "mean"),
            sim_fl=("sim_fl", "mean"), unique=("unique_output_rate", "mean"),
            copy=("loose_copy_rate", "mean"), template=("template_rate", "mean"),
            delta_sta=("delta_sta", "mean")).reset_index()
        if lang in AFRICAN:
            ok = m[(m["unique"] >= base.loc[lang, "unique_output_rate"] - MAX_UNIQUE_DROP)
                   & (m["copy"] <= base.loc[lang, "loose_copy_rate"] + MAX_COPY_RISE)
                   & (m["template"] <= base.loc[lang, "template_rate"] + MAX_TEMPLATE_RISE)]
            pool = ok if len(ok) else m
            best = pool.loc[pool["sim_fl"].idxmax()]
            rule = "max dev SIM*FL subject to degeneracy guardrails; STA UNAVAILABLE"
        else:
            best = m.loc[m["j"].idxmax()]
            rule = "max dev J"
        chosen[lang] = {
            "side": best["side"], "direction_type": best["direction_type"],
            "scope": best["scope"], "strength": float(best["strength"]),
            "selection_rule": rule,
            "dev_j": (None if lang in AFRICAN else float(best["j"])),
            "dev_sim": float(best["sim"]), "dev_fl": float(best["fl"]),
            "sta_status": "unavailable/unvalidated" if lang in AFRICAN else "validated",
        }
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test"])
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT_DIR / "frozen_test_config.json"
    train_freq = train_frequent_outputs()

    if args.phase == "dev":
        raw = run_phase("dev", cfg, device)
        print(f"scoring {len(raw)} dev generations ...")
        scored = score_frame(raw, cfg, device)
        scored.to_csv(OUT_DIR / "steering_dev_scored.csv", index=False)
        agg = aggregate(scored, train_freq)
        agg.to_csv(REPO_ROOT / "results" / "steering_dev.csv", index=False)
        print(f"wrote results/steering_dev.csv ({len(agg)} cells)")
        return 0

    if args.select:
        agg = pd.read_csv(REPO_ROOT / "results" / "steering_dev.csv")
        chosen = select_config(agg, cfg)
        payload = {"run_id": cfg["run_id"],
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "selection_split": "dev only",
                   "strengths_swept": STRENGTHS, "layer": LAYER, "pooling": POOLING,
                   "per_language_choice": chosen}
        payload["digest"] = hashlib.sha256(
            json.dumps(chosen, sort_keys=True).encode()).hexdigest()
        frozen_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"froze test configuration -> {frozen_path.relative_to(REPO_ROOT)}")
        print(f"  digest {payload['digest'][:16]}...")
        for lang, c in chosen.items():
            print(f"  {lang}: {c['side']}/{c['direction_type']}/{c['scope']} "
                  f"alpha={c['strength']:+.2f}  [{c['sta_status']}]")
        return 0

    if args.phase == "test":
        stamp = OUT_DIR / "TEST_EVALUATED.txt"
        if stamp.exists():
            print(f"REFUSING: test already evaluated ({stamp.read_text().strip()})")
            return 1
        payload = json.loads(frozen_path.read_text())
        assert payload["digest"] == hashlib.sha256(
            json.dumps(payload["per_language_choice"], sort_keys=True).encode()).hexdigest()
        print(f"loaded frozen config, digest verified {payload['digest'][:16]}...")
        chosen = payload["per_language_choice"]

        def keep(side, dtype, scope, alpha, seed):
            return any(c["side"] == side and c["direction_type"] == dtype
                       and c["scope"] == scope and abs(c["strength"] - alpha) < 1e-9
                       for c in chosen.values())

        raw = run_phase("test", cfg, device, config_filter=keep)
        scored = score_frame(raw, cfg, device)
        scored.to_csv(OUT_DIR / "steering_test_scored.csv", index=False)
        agg = aggregate(scored, train_freq)

        sel_rows = []
        for lang, c in chosen.items():
            m = agg[(agg["language"] == lang) & (agg["side"] == c["side"])
                    & (agg["direction_type"] == c["direction_type"])
                    & (agg["scope"] == c["scope"])
                    & (np.isclose(agg["strength"], c["strength"]))]
            b = agg[(agg["language"] == lang) & (agg["strength"] == 0.0)]
            if len(m):
                r = m.mean(numeric_only=True).to_dict()
                r.update({"language": lang, "selected_side": c["side"],
                          "selected_direction": c["direction_type"],
                          "selected_scope": c["scope"], "selected_strength": c["strength"],
                          "sta_status": c["sta_status"]})
                for col in ("sta", "sim", "fl", "j", "delta_sta", "loose_copy_rate",
                            "unique_output_rate", "template_rate"):
                    r[f"baseline_{col}"] = b[col].mean()
                    r[f"delta_vs_baseline_{col}"] = r[col] - b[col].mean()
                sel_rows.append(r)
        sel = pd.DataFrame(sel_rows)
        agg.to_csv(REPO_ROOT / "results" / "steering_test.csv", index=False)
        sel.to_csv(OUT_DIR / "test_selected_summary.csv", index=False)
        stamp.write_text(f"test evaluated once at {datetime.now(timezone.utc).isoformat()} "
                         f"under frozen config {payload['digest']}\n", encoding="utf-8")
        print(f"wrote results/steering_test.csv and test_selected_summary.csv")
        return 0

    print("nothing to do; pass --phase dev, --select, or --phase test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
