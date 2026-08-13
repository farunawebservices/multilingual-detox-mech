#!/usr/bin/env python
"""Stage 7 -- Linear probes and representation separation.

Fits probes on standardised mT5 activations to ask which internal features are shared
across languages and which are language-specific (RQ1), and how well a probe trained on
higher-resource languages transfers to Yoruba and isiXhosa (RQ2).

Usage
-----
    python scripts/07_train_probes.py --smoke   # verification pass, writes nothing
    python scripts/07_train_probes.py           # full run

Split roles, enforced not just intended
---------------------------------------
train  fits normalisation and every probe.
dev    probe evaluation, hyperparameter selection, exploratory validation.
test   NOT READ. A path guard raises on any attempt. Stage 7 is not the final
       evaluation, and the brief allows test exactly one visit, which is spent later.

Standardisation
---------------
Every analysis below -- cosine similarity, representation distance, probe fitting --
runs on standardised activations. The transform comes exclusively from the Stage 6
train-only statistics.

One subtlety governs how those statistics are combined. Standardising each condition by
its *own* mean would subtract away the very toxic-versus-reference difference the probes
are meant to detect. So for each contrast the per-condition train statistics are merged
into a single condition-pooled standardiser using the exact pooled-variance identity

    mu   = sum_i n_i mu_i / sum_i n_i
    var  = sum_i n_i (var_i + mu_i^2) / sum_i n_i  -  mu^2

which reproduces what fitting on the concatenated train activations would have given,
while still touching only train-derived numbers. The same standardiser is then applied
unchanged to dev.

Why standardisation is not optional here
----------------------------------------
mT5 carries massive outlier dimensions: per-dimension standard deviations span 16.8 to
62,838, a factor of 3,700. Raw cosine similarity between decoder states is ~0.999 for
every pair of conditions purely because a few dimensions dominate the dot product. Any
"shared circuit" claim resting on raw correlation would measure that scale artefact
rather than shared computation. Per the brief, correlation evidence alone is labelled
associative regardless; the causal test belongs to Stage 11.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Probes are logistic regression on standardised pooled activations, with the inverse
  regularisation strength C chosen on dev from a small fixed grid. Selection is by dev
  accuracy, never by test.
* The toxic-vs-reference and toxic-vs-generated contrasts are fitted as two separate
  probes and never merged into one "detoxified" class. They are different objects: the
  reference is a human rewrite, the generation is frequently a copy of the input or a
  generic template, so pooling them would define a class the model does not have.
* Control baselines fit the same targets from control features alone -- character
  length, token count, language one-hot -- so every activation probe can be read against
  what is achievable without looking at the model at all. A condition probe that merely
  reads sentence length would otherwise look like a detoxification feature; the Stage 1
  inventory found yo/xh references are *longer* than their inputs while the other seven
  languages' are shorter, so this confound is live.
* Condition-probe accuracy is additionally broken down by Stage 6 behaviour label, since
  a probe scoring well only on copy_input rows is detecting copying, not detoxification.
* The detox-success probe can only be fitted where generations exist, which excludes
  train. It is therefore fitted on dev with 5-fold cross-validation grouped by group_id,
  and is reported as dev-only exploratory rather than as a train-fitted result.
* Cross-language transfer uses the pooled_mean_content pooling at all three layers for
  both sides. Running the full transfer grid at both poolings would double the fit count
  for a comparison the global probes already cover.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
ACT_DIR = REPO_ROOT / "results" / "activations"
OUT_DIR = REPO_ROOT / "results" / "probes"

SIDES = ["encoder", "decoder"]
LAYERS = [4, 6, 8]
POOLINGS = ["pooled_mean_all", "pooled_mean_content"]
C_GRID = [0.01, 0.1, 1.0]
TRANSFER_POOLING = "pooled_mean_content"
AFRICAN = ["yo", "xh"]
STA_SUPPORTED = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def guard_split(split: str) -> None:
    if split == "test":
        raise RuntimeError(
            "Stage 7 must not read test activations. Test is reserved for the single "
            "final evaluation; probes are fitted on train and evaluated on dev."
        )


# ------------------------------------------------------------------ normalisation

def load_norm_stats(seed: int, side: str, layer: int, condition: str) -> dict:
    path = ACT_DIR / "norm_stats" / f"seed{seed}_{side}_layer{layer:02d}_{condition}.pt"
    return torch.load(path, weights_only=False)


def pooled_standardizer(seed: int, side: str, layer: int, conditions: list[str],
                        pooling: str, n_rows_train: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition-pooled mean/std from train-only statistics.

    Merging the per-condition statistics rather than standardising each condition
    separately is what preserves the between-condition difference the probe targets.
    """
    means, varis, counts = [], [], []
    for cond in conditions:
        st = load_norm_stats(seed, side, layer, cond)
        if pooling.startswith("pooled"):
            # Statistics are pooling-specific. pooled_mean_all includes the task prefix
            # and decoder start token, whose states are far from the content positions,
            # so the two poolings need their own means and standard deviations.
            mu, sd, n = st[f"{pooling}_mean"], st[f"{pooling}_std"], st["n_rows"]
        else:
            mu, sd, n = st["token_mean"], st["token_std"], st["n_tokens"]
        means.append(mu.double())
        varis.append((sd.double() ** 2))
        counts.append(float(n))

    total = sum(counts)
    mu = sum(n * m for n, m in zip(counts, means)) / total
    ex2 = sum(n * (v + m ** 2) for n, v, m in zip(counts, varis, means)) / total
    var = (ex2 - mu ** 2).clamp_min(0.0)
    return mu.float(), var.sqrt().float()


def standardize(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor,
                eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Returns standardized matrix and the boolean mask of usable (non-constant) dims."""
    usable = (sd > eps)
    z = (x - mu) / sd.clamp_min(eps)
    return z.numpy(), usable.numpy()


# --------------------------------------------------------------------- activations

def load_pooled(split: str, seed: int, side: str, layer: int, condition: str,
                pooling: str) -> tuple[torch.Tensor, list[str]]:
    guard_split(split)
    path = ACT_DIR / side / f"layer_{layer:02d}" / f"{split}_seed{seed}_{condition}.pt"
    if not path.exists():
        return None, None
    d = torch.load(path, weights_only=False)
    return d[pooling], d["pair_id"]


def load_rows(split: str, seed: int) -> pd.DataFrame:
    guard_split(split)
    return pd.read_csv(ACT_DIR / "index" / f"{split}_seed{seed}_rows.csv")


CONDITION_TEXT_COLUMN = {
    "toxic": "toxic_input",
    "detox_reference": "detox_reference",
    "generated": "generated",
}


def condition_text(rows: pd.DataFrame) -> pd.Series:
    """The text each row actually represents, chosen by its condition.

    This must vary with the condition. An earlier version measured control features from
    toxic_input for both classes, which made them identical within a pair and forced the
    control baseline to exactly 0.500 by construction -- an uninformative control dressed
    up as a clean one. The length confound is real here: Stage 1 found yo/xh references
    run longer than their inputs while the other seven languages' run shorter.
    """
    if "condition" not in rows.columns:
        return rows["toxic_input"].fillna("").astype(str)
    out = pd.Series(index=rows.index, dtype=object)
    for cond, col in CONDITION_TEXT_COLUMN.items():
        sel = rows["condition"] == cond
        if sel.any():
            out.loc[sel] = rows.loc[sel, col].fillna("").astype(str)
    return out.fillna("").astype(str)


def control_features(rows: pd.DataFrame, text_col: str, languages: list[str]) -> np.ndarray:
    """Language identity, character length and token count -- the confound baseline."""
    txt = condition_text(rows) if text_col == "__condition__" else \
        rows[text_col].fillna("").astype(str)
    feats = [
        txt.str.len().to_numpy(dtype=float)[:, None],
        txt.str.split().apply(len).to_numpy(dtype=float)[:, None],
    ]
    onehot = np.zeros((len(rows), len(languages)))
    for i, lang in enumerate(languages):
        onehot[:, i] = (rows["language"] == lang).to_numpy(dtype=float)
    feats.append(onehot)
    X = np.hstack(feats)
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)


# --------------------------------------------------------------------------- probes

def fit_binary(Xtr, ytr, Xdv, ydv) -> dict:
    """Logistic probe; C selected on dev."""
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xdv)
        acc = accuracy_score(ydv, pred)
        try:
            auc = roc_auc_score(ydv, clf.predict_proba(Xdv)[:, 1])
        except ValueError:
            auc = float("nan")
        if best is None or acc > best["dev_accuracy"]:
            best = {"C": C, "dev_accuracy": float(acc), "dev_auc": float(auc),
                    "train_accuracy": float(clf.score(Xtr, ytr)), "model": clf}
    return best


def fit_multiclass(Xtr, ytr, Xdv, ydv) -> dict:
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(ydv, clf.predict(Xdv))
        if best is None or acc > best["dev_accuracy"]:
            best = {"C": C, "dev_accuracy": float(acc),
                    "train_accuracy": float(clf.score(Xtr, ytr)), "model": clf}
    return best


def condition_matrices(seed: int, side: str, layer: int, pooling: str,
                       contrast: tuple[str, str], cfg: dict):
    """Standardised train/dev matrices for a two-condition contrast, plus row metadata."""
    cond_a, cond_b = contrast

    # Availability must be checked before building the standardiser: train has no
    # `generated` condition, so neither its activations nor its train statistics exist.
    for cond in (cond_a, cond_b):
        if not (ACT_DIR / side / f"layer_{layer:02d}" / f"train_seed{seed}_{cond}.pt").exists():
            return None

    tr_rows = load_rows("train", seed)
    dv_rows = load_rows("dev", seed)

    mu, sd = pooled_standardizer(seed, side, layer, [cond_a, cond_b], pooling, len(tr_rows))

    parts = {}
    for split, rows in (("train", tr_rows), ("dev", dv_rows)):
        mats, labels, meta = [], [], []
        for label, cond in enumerate((cond_a, cond_b)):
            x, pair_ids = load_pooled(split, seed, side, layer, cond, pooling)
            if x is None:
                return None  # condition unavailable for this split (train/generated)
            z, usable = standardize(x, mu, sd)
            mats.append(z)
            labels.append(np.full(len(z), label))
            m = rows.copy()
            m["condition"] = cond
            meta.append(m)
        parts[split] = (np.vstack(mats), np.concatenate(labels),
                        pd.concat(meta, ignore_index=True), usable)
    return parts, mu, sd


# ----------------------------------------------------------------------- analyses

def representation_separation(seed: int, cfg: dict) -> list[dict]:
    """Standardised cosine between conditions, per language. Associative evidence only."""
    out = []
    for side in SIDES:
        for layer in LAYERS:
            for contrast in [("toxic", "detox_reference"), ("toxic", "generated")]:
                built = condition_matrices(seed, side, layer, TRANSFER_POOLING, contrast, cfg)
                if built is None:
                    continue
                parts, _, _ = built
                Z, y, meta, usable = parts["dev"]
                a, b = Z[y == 0][:, usable], Z[y == 1][:, usable]
                langs = meta[y == 0]["language"].to_numpy()
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                bn = b / np.linalg.norm(b, axis=1, keepdims=True)
                cos = (an * bn).sum(1)
                for lang in sorted(set(langs)):
                    sel = langs == lang
                    out.append({
                        "seed": seed, "side": side, "layer": layer,
                        "contrast": f"{contrast[0]}_vs_{contrast[1]}",
                        "language": lang, "split": "dev",
                        "standardized_cosine_mean": float(cos[sel].mean()),
                        "standardized_cosine_std": float(cos[sel].std()),
                        "evidence_type": "associative (correlational; no causal claim)",
                    })
    return out


def run_global_probes(seed: int, cfg: dict) -> list[dict]:
    """Condition, language-identity and control-baseline probes over the full grid."""
    languages = cfg["languages"]
    results = []
    for side in SIDES:
        for layer in LAYERS:
            for pooling in POOLINGS:
                for contrast in [("toxic", "detox_reference"), ("toxic", "generated")]:
                    built = condition_matrices(seed, side, layer, pooling, contrast, cfg)
                    if built is None:
                        results.append({
                            "seed": seed, "side": side, "layer": layer, "pooling": pooling,
                            "probe": f"condition:{contrast[0]}_vs_{contrast[1]}",
                            "scope": "multilingual_pooled", "status": "unavailable",
                            "note": "generated condition absent for train (Stage 4 did not "
                                    "generate on train)",
                        })
                        continue
                    parts, _, _ = built
                    Xtr, ytr, mtr, usable = parts["train"]
                    Xdv, ydv, mdv, _ = parts["dev"]
                    r = fit_binary(Xtr[:, usable], ytr, Xdv[:, usable], ydv)

                    # control baseline: no activations, only length/token/language
                    ctr = control_features(mtr, "__condition__", languages)
                    cdv = control_features(mdv, "__condition__", languages)
                    rc = fit_binary(ctr, ytr, cdv, ydv)

                    # per-behaviour breakdown on dev
                    per_behaviour = {}
                    pred = r["model"].predict(Xdv[:, usable])
                    for lab, g in mdv.groupby("behaviour_label"):
                        idx = g.index.to_numpy()
                        per_behaviour[lab] = float(accuracy_score(ydv[idx], pred[idx]))

                    results.append({
                        "seed": seed, "side": side, "layer": layer, "pooling": pooling,
                        "probe": f"condition:{contrast[0]}_vs_{contrast[1]}",
                        "scope": "multilingual_pooled", "status": "ok",
                        "n_train": int(len(ytr)), "n_dev": int(len(ydv)),
                        "n_dims_used": int(usable.sum()),
                        "C": r["C"], "train_accuracy": r["train_accuracy"],
                        "dev_accuracy": r["dev_accuracy"], "dev_auc": r["dev_auc"],
                        "control_dev_accuracy": rc["dev_accuracy"],
                        "control_dev_auc": rc["dev_auc"],
                        "gain_over_control": r["dev_accuracy"] - rc["dev_accuracy"],
                        "accuracy_by_behaviour": json.dumps(per_behaviour),
                    })

                # language identity, from the toxic condition
                xtr, _ = load_pooled("train", seed, side, layer, "toxic", pooling)
                xdv, _ = load_pooled("dev", seed, side, layer, "toxic", pooling)
                mu, sd = pooled_standardizer(seed, side, layer, ["toxic"], pooling, len(xtr))
                ztr, usable = standardize(xtr, mu, sd)
                zdv, _ = standardize(xdv, mu, sd)
                mtr, mdv = load_rows("train", seed), load_rows("dev", seed)
                rl = fit_multiclass(ztr[:, usable], mtr["language"].to_numpy(),
                                    zdv[:, usable], mdv["language"].to_numpy())
                rlc = fit_multiclass(
                    control_features(mtr, "toxic_input", [])[:, :2],
                    mtr["language"].to_numpy(),
                    control_features(mdv, "toxic_input", [])[:, :2],
                    mdv["language"].to_numpy())
                results.append({
                    "seed": seed, "side": side, "layer": layer, "pooling": pooling,
                    "probe": "language_identity", "scope": "multilingual_pooled",
                    "status": "ok", "n_train": int(len(ztr)), "n_dev": int(len(zdv)),
                    "n_dims_used": int(usable.sum()), "C": rl["C"],
                    "train_accuracy": rl["train_accuracy"],
                    "dev_accuracy": rl["dev_accuracy"], "dev_auc": float("nan"),
                    "control_dev_accuracy": rlc["dev_accuracy"],
                    "control_dev_auc": float("nan"),
                    "gain_over_control": rl["dev_accuracy"] - rlc["dev_accuracy"],
                    "accuracy_by_behaviour": "",
                })
    return results


def run_transfer(seed: int, cfg: dict) -> list[dict]:
    """Within-language and cross-language transfer for the condition contrasts."""
    languages = cfg["languages"]
    non_african = [l for l in languages if l not in AFRICAN]
    source_sets = {
        "within_language": None,  # handled specially
        "EN_DE_ES": ["en", "de", "es"],
        "all_non_african": non_african,
        "YO_XH": AFRICAN,
    }
    out = []
    for side in SIDES:
        for layer in LAYERS:
            for contrast in [("toxic", "detox_reference"), ("toxic", "generated")]:
                built = condition_matrices(seed, side, layer, TRANSFER_POOLING, contrast, cfg)
                if built is None:
                    continue
                parts, _, _ = built
                Xtr, ytr, mtr, usable = parts["train"]
                Xdv, ydv, mdv, _ = parts["dev"]
                Xtr, Xdv = Xtr[:, usable], Xdv[:, usable]
                tr_lang, dv_lang = mtr["language"].to_numpy(), mdv["language"].to_numpy()
                cname = f"{contrast[0]}_vs_{contrast[1]}"

                for lang in languages:  # within-language
                    tsel, dsel = tr_lang == lang, dv_lang == lang
                    r = fit_binary(Xtr[tsel], ytr[tsel], Xdv[dsel], ydv[dsel])
                    out.append({"seed": seed, "side": side, "layer": layer,
                                "contrast": cname, "source": lang, "target": lang,
                                "transfer_type": "within_language",
                                "n_train": int(tsel.sum()), "n_dev": int(dsel.sum()),
                                "dev_accuracy": r["dev_accuracy"], "dev_auc": r["dev_auc"]})

                for name, srcs in source_sets.items():
                    if srcs is None:
                        continue
                    tsel = np.isin(tr_lang, srcs)
                    targets = AFRICAN if name != "YO_XH" else [l for l in languages
                                                               if l not in AFRICAN]
                    for tgt in targets:
                        dsel = dv_lang == tgt
                        r = fit_binary(Xtr[tsel], ytr[tsel], Xdv[dsel], ydv[dsel])
                        out.append({"seed": seed, "side": side, "layer": layer,
                                    "contrast": cname, "source": name, "target": tgt,
                                    "transfer_type": "cross_language",
                                    "n_train": int(tsel.sum()), "n_dev": int(dsel.sum()),
                                    "dev_accuracy": r["dev_accuracy"],
                                    "dev_auc": r["dev_auc"]})
    return out


def run_dev_only_contrast(seed: int, cfg: dict) -> list[dict]:
    """toxic vs generated, which train cannot support.

    Stage 4 generated only on dev and test, so this contrast has no train activations and
    cannot be train-fitted. Rather than drop it -- the brief asks for the two contrasts to
    be compared separately -- it is fitted on dev under 5-fold cross-validation grouped by
    group_id, so a pair's toxic and generated rows never straddle a fold. The standardiser
    still comes from train (the toxic condition's statistics), so no transform sees dev.
    Cross-language transfer needs no CV, since source and target languages are disjoint.
    """
    languages = cfg["languages"]
    non_african = [l for l in languages if l not in AFRICAN]
    out = []
    rows = load_rows("dev", seed)
    n_train = len(load_rows("train", seed))

    for side in SIDES:
        for layer in LAYERS:
            mu, sd = pooled_standardizer(seed, side, layer, ["toxic"],
                                         TRANSFER_POOLING, n_train)
            xa, _ = load_pooled("dev", seed, side, layer, "toxic", TRANSFER_POOLING)
            xb, _ = load_pooled("dev", seed, side, layer, "generated", TRANSFER_POOLING)
            za, usable = standardize(xa, mu, sd)
            zb, _ = standardize(xb, mu, sd)
            Z = np.vstack([za[:, usable], zb[:, usable]])
            y = np.concatenate([np.zeros(len(za)), np.ones(len(zb))])
            groups = np.concatenate([rows["group_id"].to_numpy()] * 2)
            langs = np.concatenate([rows["language"].to_numpy()] * 2)

            accs, aucs = [], []
            for tr_i, te_i in GroupKFold(n_splits=5).split(Z, y, groups):
                clf = LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1)
                clf.fit(Z[tr_i], y[tr_i])
                accs.append(accuracy_score(y[te_i], clf.predict(Z[te_i])))
                aucs.append(roc_auc_score(y[te_i], clf.predict_proba(Z[te_i])[:, 1]))
            out.append({"seed": seed, "side": side, "layer": layer,
                        "contrast": "toxic_vs_generated", "source": "dev_cv",
                        "target": "multilingual_pooled", "transfer_type": "dev_cv",
                        "n_train": 0, "n_dev": int(len(y)),
                        "dev_accuracy": float(np.mean(accs)),
                        "dev_auc": float(np.mean(aucs)),
                        "fit_split": "dev (grouped 5-fold CV)"})

            for name, srcs in (("EN_DE_ES", ["en", "de", "es"]),
                               ("all_non_african", non_african),
                               ("YO_XH", AFRICAN)):
                tsel = np.isin(langs, srcs)
                targets = AFRICAN if name != "YO_XH" else non_african
                for tgt in targets:
                    dsel = langs == tgt
                    clf = LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1)
                    clf.fit(Z[tsel], y[tsel])
                    try:
                        auc = roc_auc_score(y[dsel], clf.predict_proba(Z[dsel])[:, 1])
                    except ValueError:
                        auc = float("nan")
                    out.append({"seed": seed, "side": side, "layer": layer,
                                "contrast": "toxic_vs_generated", "source": name,
                                "target": tgt, "transfer_type": "cross_language",
                                "n_train": int(tsel.sum()), "n_dev": int(dsel.sum()),
                                "dev_accuracy": float(accuracy_score(
                                    y[dsel], clf.predict(Z[dsel]))),
                                "dev_auc": float(auc),
                                "fit_split": "dev (disjoint source languages)"})
    return out


def run_detox_success(seed: int, cfg: dict) -> list[dict]:
    """Dev-only exploratory probe: generations exist only for dev/test, so train is out."""
    out = []
    rows = load_rows("dev", seed)
    y = rows["behaviour_label"].to_numpy()
    if len(set(y)) < 2:
        return out
    groups = rows["group_id"].to_numpy()
    for side in SIDES:
        for layer in LAYERS:
            x, _ = load_pooled("dev", seed, side, layer, "generated", TRANSFER_POOLING)
            mu, sd = pooled_standardizer(seed, side, layer, ["toxic"], TRANSFER_POOLING,
                                         len(load_rows("train", seed)))
            z, usable = standardize(x, mu, sd)
            z = z[:, usable]
            accs = []
            for tr_i, te_i in GroupKFold(n_splits=5).split(z, y, groups):
                clf = LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1)
                clf.fit(z[tr_i], y[tr_i])
                accs.append(accuracy_score(y[te_i], clf.predict(z[te_i])))
            majority = pd.Series(y).value_counts(normalize=True).iloc[0]
            out.append({"seed": seed, "side": side, "layer": layer,
                        "probe": "detox_success_behaviour", "n_dev": int(len(y)),
                        "cv_accuracy_mean": float(np.mean(accs)),
                        "cv_accuracy_std": float(np.std(accs)),
                        "majority_baseline": float(majority),
                        "note": "dev-only, 5-fold grouped CV; train has no generations"})
    return out


# ------------------------------------------------------------------ verification

def control_baselines_by_language(seed: int, cfg: dict) -> list[dict]:
    """Per-language length-only baseline for the toxic-vs-reference contrast.

    The pooled control understates the problem. Stage 1 found yo/xh references run longer
    than their inputs while the other seven languages' run shorter, so length alone
    separates the conditions far better for yo/xh than elsewhere. Any within-language
    probe result for those two has to be read against this number, not against chance.
    """
    tr, dv = load_rows("train", seed), load_rows("dev", seed)
    out = []
    for lang in cfg["languages"]:
        a, b = tr[tr.language == lang], dv[dv.language == lang]

        def feats(d: pd.DataFrame) -> np.ndarray:
            return np.vstack([
                np.c_[d.toxic_input.str.len(), d.toxic_input.str.split().apply(len)],
                np.c_[d.detox_reference.str.len(), d.detox_reference.str.split().apply(len)],
            ]).astype(float)

        ytr = np.r_[np.zeros(len(a)), np.ones(len(a))]
        ydv = np.r_[np.zeros(len(b)), np.ones(len(b))]
        clf = LogisticRegression(max_iter=3000).fit(feats(a), ytr)
        out.append({
            "seed": seed, "language": lang,
            "control_dev_accuracy_length_only": float(accuracy_score(ydv, clf.predict(feats(b)))),
            "mean_toxic_chars": float(a.toxic_input.str.len().mean()),
            "mean_reference_chars": float(a.detox_reference.str.len().mean()),
            "length_ratio_reference_over_toxic": float(
                a.detox_reference.str.len().mean() / a.toxic_input.str.len().mean()),
        })
    return out


def verify_standardization(seed: int, cfg: dict) -> dict:
    """Item 3: standardized TRAIN dims must be ~mean 0, ~sd 1, excluding constant dims."""
    report = []
    for side in SIDES:
        for layer in LAYERS:
            for pooling in POOLINGS:
                conds = ["toxic", "detox_reference"]
                tr_rows = load_rows("train", seed)
                mu, sd = pooled_standardizer(seed, side, layer, conds, pooling, len(tr_rows))
                mats = []
                for cond in conds:
                    x, _ = load_pooled("train", seed, side, layer, cond, pooling)
                    mats.append(x)
                X = torch.cat(mats)
                z, usable = standardize(X, mu, sd)
                zu = z[:, usable]
                report.append({
                    "seed": seed, "side": side, "layer": layer, "pooling": pooling,
                    "n_dims": int(z.shape[1]),
                    "n_constant_dims_excluded": int((~usable).sum()),
                    "mean_of_dim_means": float(zu.mean(0).mean()),
                    "max_abs_dim_mean": float(np.abs(zu.mean(0)).max()),
                    "mean_of_dim_stds": float(zu.std(0).mean()),
                    "min_dim_std": float(zu.std(0).min()),
                    "max_dim_std": float(zu.std(0).max()),
                    "all_finite": bool(np.isfinite(z).all()),
                })
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--controls-only", action="store_true",
                    help="recompute only the per-language control baselines")
    args = ap.parse_args()

    cfg = load_config()
    seeds = args.seeds or cfg["seeds_multi"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("=== SMOKE TEST: standardisation verification, seed 42, writes nothing ===")
        rep = verify_standardization(42, cfg)
        df = pd.DataFrame(rep)
        print(df[["side", "layer", "pooling", "n_constant_dims_excluded",
                  "max_abs_dim_mean", "mean_of_dim_stds", "min_dim_std",
                  "max_dim_std", "all_finite"]].round(4).to_string(index=False))
        # Tolerances: the saved standard deviations use the unbiased (n-1) estimator
        # while verification measures the population sd, so a ratio of
        # sqrt((n-1)/n) ~ 0.9998 at n=2196 is expected and is not a defect.
        ok_mean = df["max_abs_dim_mean"].max() < 1e-3
        ok_std = (df["mean_of_dim_stds"] - 1).abs().max() < 0.01
        print(f"\n  all standardized dims mean ~0 (max |mean| < 1e-3): {ok_mean}")
        print(f"  all standardized dims sd ~1 (|mean sd - 1| < 0.01): {ok_std}")
        print(f"  all finite: {bool(df['all_finite'].all())}")
        print("\n  transform provenance: fitted from results/activations/norm_stats/, "
              "which Stage 6 computed on the train split only")
        try:
            load_rows("test", 42)
            print("  TEST GUARD FAILED: test rows were readable")
            return 1
        except RuntimeError as exc:
            print(f"  test guard active: {exc}")
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return 0

    if args.controls_only:
        ctrl = [r for seed in seeds for r in control_baselines_by_language(seed, cfg)]
        cdf = pd.DataFrame(ctrl)
        cdf.to_csv(OUT_DIR / "control_baselines_by_language.csv", index=False)
        print(cdf.groupby("language")[[
            "control_dev_accuracy_length_only", "length_ratio_reference_over_toxic"
        ]].mean().round(3).to_string())
        return 0

    verif, seps, globals_, transfers, successes = [], [], [], [], []
    for seed in seeds:
        print(f"seed {seed}: verifying standardisation ...")
        verif += verify_standardization(seed, cfg)
        print(f"seed {seed}: representation separation ...")
        seps += representation_separation(seed, cfg)
        print(f"seed {seed}: global probes ...")
        globals_ += run_global_probes(seed, cfg)
        print(f"seed {seed}: transfer matrices ...")
        transfers += run_transfer(seed, cfg)
        print(f"seed {seed}: dev-only toxic-vs-generated contrast ...")
        transfers += run_dev_only_contrast(seed, cfg)
        print(f"seed {seed}: detox-success probe ...")
        successes += run_detox_success(seed, cfg)

    pd.DataFrame(verif).to_csv(OUT_DIR / "standardization_verification.csv", index=False)
    pd.DataFrame(seps).to_csv(OUT_DIR / "representation_separation.csv", index=False)
    gdf = pd.DataFrame(globals_)
    gdf.to_csv(OUT_DIR / "probe_results.csv", index=False)
    tdf = pd.DataFrame(transfers)
    tdf.to_csv(OUT_DIR / "transfer_matrix.csv", index=False)
    pd.DataFrame(successes).to_csv(OUT_DIR / "detox_success_probe.csv", index=False)
    pd.DataFrame([r for seed in seeds for r in control_baselines_by_language(seed, cfg)]
                 ).to_csv(OUT_DIR / "control_baselines_by_language.csv", index=False)

    ok = gdf[gdf["status"] == "ok"]
    summary = (ok.groupby(["probe", "side", "layer", "pooling"])
                 .agg(dev_accuracy_mean=("dev_accuracy", "mean"),
                      dev_accuracy_std=("dev_accuracy", "std"),
                      dev_auc_mean=("dev_auc", "mean"),
                      control_mean=("control_dev_accuracy", "mean"),
                      gain_mean=("gain_over_control", "mean"),
                      n_seeds=("seed", "nunique")).reset_index())
    summary.to_csv(OUT_DIR / "probe_summary_across_seeds.csv", index=False)

    (OUT_DIR / "stage7_meta.json").write_text(json.dumps({
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split_roles": {"train": "fit normalisation and probes",
                        "dev": "probe evaluation and selection",
                        "test": "NOT READ in Stage 7"},
        "standardization": "condition-pooled merge of Stage 6 train-only statistics",
        "evidence_caveat": ("probe and cosine results are associative; a shared-circuit "
                            "claim additionally requires the Stage 11 causal test"),
        "C_grid": C_GRID, "transfer_pooling": TRANSFER_POOLING,
    }, indent=2), encoding="utf-8")

    # ---------------- console report ----------------
    vdf = pd.DataFrame(verif)
    print("\n=== standardisation verification (train) ===")
    print(f"  constant dims excluded (total): {int(vdf['n_constant_dims_excluded'].sum())}")
    print(f"  max |dim mean| across all configs: {vdf['max_abs_dim_mean'].max():.2e}")
    print(f"  dim sd range: [{vdf['min_dim_std'].min():.4f}, {vdf['max_dim_std'].max():.4f}]"
          f", mean {vdf['mean_of_dim_stds'].mean():.4f}")
    print(f"  all finite: {bool(vdf['all_finite'].all())}")

    print("\n=== condition probes, dev accuracy (mean +/- sd over seeds) ===")
    for probe in sorted(ok["probe"].unique()):
        p = summary[summary["probe"] == probe]
        print(f"\n  [{probe}]")
        piv = p.pivot_table(index=["side", "layer"], columns="pooling",
                            values="dev_accuracy_mean").round(3)
        print(piv.to_string())
        ctrl = p["control_mean"].mean()
        print(f"    control baseline (length/tokens/language only): {ctrl:.3f}")

    print("\n=== unavailable configurations ===")
    un = gdf[gdf["status"] == "unavailable"]
    print(f"  {len(un)} entries marked unavailable "
          f"({un['probe'].nunique()} probe types) -- generated condition absent for train")

    print("\n=== cross-lingual transfer (dev accuracy, mean over seeds) ===")
    for contrast in sorted(tdf["contrast"].unique()):
        t = tdf[(tdf["contrast"] == contrast) & (tdf["layer"] == 6)]
        print(f"\n  [{contrast}] layer 6")
        piv = t.pivot_table(index=["side", "source"], columns="target",
                            values="dev_accuracy", aggfunc="mean").round(3)
        print(piv.to_string())

    print("\n=== detox-success probe (dev-only, grouped 5-fold CV) ===")
    sdf = pd.DataFrame(successes)
    if len(sdf):
        print(sdf.groupby(["side", "layer"])[["cv_accuracy_mean", "majority_baseline"]]
              .mean().round(3).to_string())

    print("\n=== representation separation, standardized cosine (dev, layer 6) ===")
    rdf = pd.DataFrame(seps)
    print(rdf[rdf["layer"] == 6].pivot_table(
        index="language", columns=["side", "contrast"],
        values="standardized_cosine_mean", aggfunc="mean").round(3).to_string())
    print("\n  NOTE: cosine values are associative evidence only. Per the brief, a shared "
          "circuit claim requires a causal effect (Stage 11), not correlation alone.")

    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ "
          "(probe_results, transfer_matrix, representation_separation, "
          "standardization_verification, detox_success_probe, summary, meta)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
