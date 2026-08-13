#!/usr/bin/env python
"""Stage 4 -- Baseline generation with template-collapse diagnostics.

Generates detoxifications for the dev and test splits with each of the three Stage 3
checkpoints (seeds 42, 1337, 2024) under deterministic decoding, and audits the
outputs for the template-collapse failure mode observed in the Stage 3 smoke test.

Outputs
-------
results/generations/seed<N>_dev.csv        per-seed dev generations
results/generations/seed<N>_test.csv       per-seed test generations, kept separate
results/baseline_generations.csv           all seeds/splits combined (Stage 4 of the brief)
results/generations/template_diagnostics.csv   per seed/split/language diagnostics
results/generations/top_repeated_outputs.json  the actual repeated strings, per language

Test-split discipline
---------------------
Test outputs are generated and written, but nothing in this stage reads them back to
make a choice. The decoding strategy is fixed a priori rather than tuned, no seed or
checkpoint is ranked, and the diagnostics are computed independently per split so a
test number can never influence a dev-side decision. data/splits and the checkpoints
are opened read-only.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Decoding is beam search with 4 beams, no sampling, length penalty 1.0. It is fixed
  in advance rather than chosen by comparing candidates, because any comparison would
  need generation metrics, and the only splits available to score against are dev
  (reserved for Stage 13's strength sweep) and test (off limits). Beam-4 is the
  conventional default for seq2seq detoxification baselines, so it is the defensible
  a-priori choice. Greedy output is not generated here; if a decoding comparison is
  wanted later it belongs in a dev-only experiment with its own stage.
* "Frequent training output" means a detox_output occurring at least twice within that
  language's training split. With 122 training rows per language, a string appearing
  twice or more is already a template rather than a coincidence. Matches against *any*
  training output are also recorded separately, since a generation reproducing a unique
  training target is memorisation even though it is not template collapse.
* Copied-input detection is reported two ways: exact (after NFC and whitespace strip)
  and relaxed (additionally casefolded). The relaxed figure catches near-copies that
  differ only in capitalisation, which are still failures to detoxify.
* Generation length cap is 112 new tokens, matching the Stage 3 target length, so the
  model is never truncated below what it was trained to produce.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, MT5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
GEN_DIR = REPO_ROOT / "results" / "generations"

TASK_PREFIX = "detoxify: "
WS_RE = re.compile(r"\s+")
SENTINEL_RE = re.compile(r"<extra_id_\d+>")

DECODING = {
    "strategy": "beam",
    "num_beams": 4,
    "do_sample": False,
    "length_penalty": 1.0,
    "early_stopping": True,
    "max_new_tokens": 112,
}
BATCH_SIZE = 32
FREQUENT_TRAIN_MIN_COUNT = 2
TOP_REPEATED_K = 5


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def norm(text: str) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def norm_loose(text: str) -> str:
    return norm(text).casefold()


def norm_fuzzy(text: str) -> str:
    """Casefold and strip combining marks.

    Exact matching against training outputs proved close to vacuous: the model
    reproduces a training target's wording while dropping or altering a tone mark,
    so an exact comparison scores it as novel. This looser key catches those.
    """
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )
    return WS_RE.sub(" ", stripped).strip().casefold()


def bigram_prefix(text: str) -> str:
    return " ".join(norm_loose(text).split()[:2])


@torch.no_grad()
def generate_split(model, tokenizer, df: pd.DataFrame, device: str) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        enc = tokenizer(
            [TASK_PREFIX + str(t) for t in batch["toxic_input"]],
            return_tensors="pt", padding=True, truncation=True, max_length=96,
        ).to(device)
        gen = model.generate(
            **enc,
            num_beams=DECODING["num_beams"],
            do_sample=DECODING["do_sample"],
            length_penalty=DECODING["length_penalty"],
            early_stopping=DECODING["early_stopping"],
            max_new_tokens=DECODING["max_new_tokens"],
        )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=False))
    # Strip special tokens manually so sentinel tokens stay visible for the audit.
    cleaned = []
    for text in outputs:
        text = text.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
        cleaned.append(text.strip())
    return cleaned


def train_output_frequencies(train_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-language sets of training outputs: all, frequent (>=2), and fuzzy-keyed."""
    all_outputs, frequent, fuzzy = {}, {}, {}
    for lang, g in train_df.groupby("language"):
        counts = Counter(norm(t) for t in g["detox_output"])
        all_outputs[lang] = set(counts)
        frequent[lang] = {t for t, c in counts.items() if c >= FREQUENT_TRAIN_MIN_COUNT}
        fuzzy[lang] = {norm_fuzzy(t) for t in g["detox_output"]}
    return all_outputs, frequent, fuzzy


def annotate(df: pd.DataFrame, generations: list[str], seed: int, checkpoint: Path,
             model_id: str, split: str, all_train: dict, freq_train: dict,
             fuzzy_train: dict) -> pd.DataFrame:
    out = df[["pair_id", "group_id", "language", "toxic_input", "detox_output"]].copy()
    out = out.rename(columns={"detox_output": "detox_reference"})
    out["split"] = split
    out["seed"] = seed
    out["model_id"] = model_id
    out["checkpoint_path"] = str(checkpoint.relative_to(REPO_ROOT))
    out["decoding"] = f"{DECODING['strategy']}{DECODING['num_beams']}"
    out["generated_raw"] = generations
    out["generated"] = [norm(SENTINEL_RE.sub("", g)) for g in generations]

    out["is_empty"] = out["generated"].str.len() == 0
    out["has_sentinel"] = [bool(SENTINEL_RE.search(g)) for g in generations]
    out["is_copy_of_input"] = [
        norm(g) == norm(t) for g, t in zip(out["generated"], out["toxic_input"])
    ]
    out["is_copy_of_input_loose"] = [
        norm_loose(g) == norm_loose(t) for g, t in zip(out["generated"], out["toxic_input"])
    ]
    out["matches_train_output"] = [
        norm(g) in all_train.get(l, set()) for g, l in zip(out["generated"], out["language"])
    ]
    out["matches_frequent_train_output"] = [
        norm(g) in freq_train.get(l, set()) for g, l in zip(out["generated"], out["language"])
    ]
    out["matches_train_output_fuzzy"] = [
        norm_fuzzy(g) in fuzzy_train.get(l, set())
        for g, l in zip(out["generated"], out["language"])
    ]
    out["gen_char_len"] = out["generated"].str.len()
    return out


def diagnose(gen_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per seed/split/language template-collapse diagnostics."""
    rows, repeated = [], {}
    for (seed, split, lang), g in gen_df.groupby(["seed", "split", "language"]):
        texts = [norm(t) for t in g["generated"]]
        counts = Counter(texts)
        prefixes = Counter(bigram_prefix(t) for t in texts)
        n = len(texts)
        top = counts.most_common(TOP_REPEATED_K)

        rows.append({
            "seed": seed, "split": split, "language": lang, "n": n,
            "n_unique_outputs": len(counts),
            "unique_output_rate": len(counts) / n if n else float("nan"),
            "n_unique_prefixes": len(prefixes),
            "prefix_diversity": len(prefixes) / n if n else float("nan"),
            "top_output_count": top[0][1] if top else 0,
            "top_output_share": (top[0][1] / n) if top and n else float("nan"),
            "top_prefix_share": (prefixes.most_common(1)[0][1] / n) if n else float("nan"),
            "empty_rate": float(g["is_empty"].mean()),
            "sentinel_rate": float(g["has_sentinel"].mean()),
            "copy_input_rate": float(g["is_copy_of_input"].mean()),
            "copy_input_rate_loose": float(g["is_copy_of_input_loose"].mean()),
            "matches_train_output_rate": float(g["matches_train_output"].mean()),
            "matches_train_output_fuzzy_rate": float(g["matches_train_output_fuzzy"].mean()),
            "template_rate": float(g["matches_frequent_train_output"].mean()),
            "repetition_rate": float(1 - len(counts) / n) if n else float("nan"),
        })
        repeated.setdefault(str(seed), {}).setdefault(split, {})[lang] = [
            {"output": t, "count": c, "share": round(c / n, 4)} for t, c in top if c > 1
        ]
    return pd.DataFrame(rows), repeated


def main() -> int:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    all_train, freq_train, fuzzy_train = train_output_frequencies(train_df)
    n_freq = {l: len(s) for l, s in freq_train.items()}
    print(f"training outputs occurring >={FREQUENT_TRAIN_MIN_COUNT}x per language: {n_freq}")

    splits = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("dev", "test")}
    print(f"dev rows: {len(splits['dev'])} | test rows: {len(splits['test'])}")
    print(f"decoding (fixed a priori, not tuned): {DECODING}\n")

    combined = []
    for seed in cfg["seeds_multi"]:
        ckpt = MODEL_DIR / f"seed{seed}"
        print(f"seed {seed}: loading {ckpt.relative_to(REPO_ROOT)}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = MT5ForConditionalGeneration.from_pretrained(str(ckpt)).to(device).eval()

        for split, df in splits.items():
            gens = generate_split(model, tokenizer, df, device)
            annotated = annotate(df, gens, seed, ckpt, cfg["models"]["primary"],
                                 split, all_train, freq_train, fuzzy_train)
            path = GEN_DIR / f"seed{seed}_{split}.csv"
            annotated.to_csv(path, index=False, encoding="utf-8")
            combined.append(annotated)
            print(f"  {split:4s}: {len(annotated)} rows -> {path.relative_to(REPO_ROOT)}")

        del model
        torch.cuda.empty_cache()

    all_gen = pd.concat(combined, ignore_index=True)
    all_gen.to_csv(REPO_ROOT / "results" / "baseline_generations.csv",
                   index=False, encoding="utf-8")

    diag, repeated = diagnose(all_gen)
    diag.to_csv(GEN_DIR / "template_diagnostics.csv", index=False)
    (GEN_DIR / "top_repeated_outputs.json").write_text(
        json.dumps({
            "run_id": cfg["run_id"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "decoding": DECODING,
            "frequent_train_output_min_count": FREQUENT_TRAIN_MIN_COUNT,
            "top_repeated": repeated,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- smoke test / console report ----------------
    print("\n=== smoke test: degenerate output audit (all seeds pooled) ===")
    for split in ("dev", "test"):
        s = all_gen[all_gen["split"] == split]
        print(f"  {split}: empty={int(s['is_empty'].sum())} "
              f"sentinel={int(s['has_sentinel'].sum())} "
              f"copied_input={int(s['is_copy_of_input'].sum())} "
              f"copied_input_loose={int(s['is_copy_of_input_loose'].sum())} "
              f"of {len(s)} rows")

    for split in ("dev", "test"):
        print(f"\n=== template-collapse diagnostics: {split} "
              f"(mean over 3 seeds) ===")
        d = diag[diag["split"] == split].groupby("language")[[
            "unique_output_rate", "prefix_diversity", "top_output_share",
            "template_rate", "matches_train_output_rate",
            "matches_train_output_fuzzy_rate", "copy_input_rate_loose",
        ]].mean().round(3)
        print(d.to_string())

    print("\n=== top repeated outputs (seed 42, dev) ===")
    for lang, items in repeated.get("42", {}).get("dev", {}).items():
        if items:
            print(f"  {lang}:")
            for it in items:
                print(f"    x{it['count']:2d} ({it['share']:.0%})  {it['output'][:70]!r}")

    print("\n=== seed agreement (identical generations across seeds) ===")
    for split in ("dev", "test"):
        s = all_gen[all_gen["split"] == split]
        piv = s.pivot_table(index="pair_id", columns="seed", values="generated",
                            aggfunc="first")
        agree = (piv.nunique(axis=1) == 1).mean()
        print(f"  {split}: {agree:.1%} of pairs produce the identical string in all 3 seeds")

    print(f"\nwrote per-seed files under {GEN_DIR.relative_to(REPO_ROOT)}/, "
          f"combined results/baseline_generations.csv, template_diagnostics.csv, "
          f"top_repeated_outputs.json")
    print("test generations were written but never read back for any decision in this stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
