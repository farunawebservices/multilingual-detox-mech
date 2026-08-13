#!/usr/bin/env python
"""Stage 1 -- Corpus inventory.

Read-only audit of data/processed/. Nothing under data/ is written or modified.

Per language reports: row counts, missing/empty values, duplicate pairs, rows where
toxic_input == detox_output, sentence length statistics (characters, whitespace words,
and mT5 subword tokens), and tokenizer-health statistics that matter specifically for
Yoruba and isiXhosa.

Outputs
-------
results/corpus_inventory.csv        : one row per language (+ supplementary corpus).
results/corpus_inventory_flags.json : the actual offending rows behind each flag,
                                      so any anomaly can be inspected rather than
                                      just counted.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* "Duplicate pair" is an exact repeat of the (toxic_input, detox_output) tuple after
  stripping surrounding whitespace. Duplicate *toxic inputs* with differing references
  are counted separately, because those are legitimate multi-reference paraphrases in
  ParaDetox-style corpora, not corpus errors -- but they DO matter for split hygiene,
  since the same toxic input landing in both train and test would leak. Stage 2 must
  group by toxic_input when splitting; this stage quantifies how often that arises.
* "Identical" is reported twice: exact string equality, and equality after casefolding
  and whitespace collapsing. The normalised count catches trivially-nonsubstantive
  pairs that exact matching misses.
* Token statistics use google/mt5-base, the primary mechanistic model, so the numbers
  describe the representation the circuit experiments actually operate on. Fertility
  (subword tokens per whitespace word) is included because it is the standard measure
  of tokenizer disadvantage for low-resource languages and is direct evidence for the
  vocabulary component of H2.
* Round-trip fidelity (decode(encode(s)) == s) is measured because mT5's SentencePiece
  vocabulary can silently drop or recompose combining diacritics. Yoruba is written with
  tone marks and sub-dot characters, so a low round-trip rate would place a hard ceiling
  on achievable detox quality for reasons unrelated to the circuits under study. This is
  a diagnostic the brief's hypotheses depend on, so it is measured rather than assumed.
* Lexical diversity (type-token ratio) and template concentration (share of distinct
  leading bigrams) are reported because a corpus assembled from templates carries far
  less independent signal than its row count suggests. With only 178 pairs for the two
  African languages, that distinction changes how much any yo/xh result can bear.
* Unicode normalisation consistency is checked because mixed NFC/NFD encoding makes
  visually identical words tokenise differently, silently splitting the model's evidence
  for a word across two token sequences. This stage only reports it; the data is not
  modified here.
* The supplementary English corpus (en_detox_full.csv) is inventoried for completeness
  but flagged is_supplementary=True and given the language code "en_full" so it can
  never be confused with a tenth study language.
"""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
RESULTS_DIR = REPO_ROOT / "results"

WS_RE = re.compile(r"\s+")
MAX_FLAG_EXAMPLES = 20


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def norm(text: str) -> str:
    """Casefold + collapse whitespace, for the 'normalised identical' comparison."""
    return WS_RE.sub(" ", str(text)).strip().casefold()


def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def pct(values: list[int] | np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def diversity_stats(texts: list[str], prefix_n: int = 2) -> dict:
    """Lexical diversity and template concentration."""
    words = [w for t in texts for w in WS_RE.sub(" ", t).strip().split()]
    prefixes = collections.Counter(
        " ".join(WS_RE.sub(" ", t).strip().split()[:prefix_n]) for t in texts
    )
    top_share = (prefixes.most_common(1)[0][1] / len(texts)) if texts and prefixes else float("nan")
    return {
        "type_token_ratio": (len(set(words)) / len(words)) if words else float("nan"),
        "n_unique_prefixes": len(prefixes),
        "prefix_diversity": (len(prefixes) / len(texts)) if texts else float("nan"),
        "top_prefix_share": top_share,
    }


def normalization_stats(texts: list[str]) -> dict:
    """Count strings not already in NFC. Mixed encodings tokenise inconsistently."""
    not_nfc = sum(1 for t in texts if t != unicodedata.normalize("NFC", t))
    return {
        "n_non_nfc_strings": not_nfc,
        "non_nfc_rate": (not_nfc / len(texts)) if texts else float("nan"),
    }


def token_stats(texts: list[str], tokenizer) -> dict:
    """mT5 subword statistics: length, fertility, UNK rate, round-trip fidelity."""
    unk_id = tokenizer.unk_token_id
    lengths, unk_counts, roundtrip_ok, fertilities = [], [], [], []

    encoded = tokenizer(texts, add_special_tokens=True)["input_ids"]
    for text, ids in zip(texts, encoded):
        lengths.append(len(ids))
        unk_counts.append(sum(1 for i in ids if i == unk_id))
        n_words = max(len(WS_RE.sub(" ", text).strip().split()), 1)
        fertilities.append(len(ids) / n_words)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        roundtrip_ok.append(WS_RE.sub(" ", decoded).strip() == WS_RE.sub(" ", text).strip())

    total_tokens = int(sum(lengths))
    return {
        "tok_mean": float(np.mean(lengths)) if lengths else float("nan"),
        "tok_median": float(np.median(lengths)) if lengths else float("nan"),
        "tok_p95": pct(lengths, 95),
        "tok_max": int(max(lengths)) if lengths else 0,
        "fertility_mean": float(np.mean(fertilities)) if fertilities else float("nan"),
        "unk_token_rate": (sum(unk_counts) / total_tokens) if total_tokens else float("nan"),
        "rows_with_unk": int(sum(1 for c in unk_counts if c > 0)),
        "roundtrip_exact_rate": float(np.mean(roundtrip_ok)) if roundtrip_ok else float("nan"),
    }


def inventory_one(path: Path, code: str, name: str, supplementary: bool,
                  tokenizer, encoding: str) -> tuple[dict, dict]:
    df = pd.read_csv(path, encoding=encoding)
    n_rows = len(df)

    # ---- missing / blank values ----
    blank_toxic = is_blank(df["toxic_input"])
    blank_detox = is_blank(df["detox_output"])
    blank_any = blank_toxic | blank_detox

    # ---- language column consistency ----
    lang_values = sorted(df["language"].astype(str).unique().tolist())

    # ---- duplicates ----
    stripped = pd.DataFrame({
        "toxic": df["toxic_input"].astype(str).str.strip(),
        "detox": df["detox_output"].astype(str).str.strip(),
    })
    dup_pair_mask = stripped.duplicated(keep="first")
    toxic_counts = stripped["toxic"].value_counts()
    repeated_toxic = toxic_counts[toxic_counts > 1]
    # Toxic inputs reused with a *different* reference (multi-reference paraphrases).
    multi_ref = (
        stripped[stripped["toxic"].isin(repeated_toxic.index)]
        .groupby("toxic")["detox"].nunique()
    )
    n_multi_ref_inputs = int((multi_ref > 1).sum())

    # ---- identical toxic == detox ----
    identical_exact = stripped["toxic"] == stripped["detox"]
    identical_norm = df["toxic_input"].map(norm) == df["detox_output"].map(norm)

    # ---- lengths (computed on non-blank rows only) ----
    valid = df.loc[~blank_any]
    toxic_texts = valid["toxic_input"].astype(str).tolist()
    detox_texts = valid["detox_output"].astype(str).tolist()

    toxic_chars = [len(t) for t in toxic_texts]
    detox_chars = [len(t) for t in detox_texts]
    toxic_words = [len(WS_RE.sub(" ", t).strip().split()) for t in toxic_texts]
    detox_words = [len(WS_RE.sub(" ", t).strip().split()) for t in detox_texts]

    tox_tok = token_stats(toxic_texts, tokenizer)
    det_tok = token_stats(detox_texts, tokenizer)
    tox_div = diversity_stats(toxic_texts)
    det_div = diversity_stats(detox_texts)
    nfc = normalization_stats(toxic_texts + detox_texts)

    row = {
        "language": code,
        "language_name": name,
        "is_supplementary": supplementary,
        "source": "|".join(sorted(df["source"].astype(str).unique().tolist())),
        "n_rows": n_rows,
        "language_values": "|".join(lang_values),
        "n_language_values": len(lang_values),
        # missing
        "n_missing_toxic": int(blank_toxic.sum()),
        "n_missing_detox": int(blank_detox.sum()),
        "n_missing_any": int(blank_any.sum()),
        # duplicates
        "n_unique_pairs": int(n_rows - dup_pair_mask.sum()),
        "n_duplicate_pairs": int(dup_pair_mask.sum()),
        "n_repeated_toxic_inputs": int(len(repeated_toxic)),
        "n_multiref_toxic_inputs": n_multi_ref_inputs,
        # degenerate pairs
        "n_identical_exact": int(identical_exact.sum()),
        "n_identical_normalized": int(identical_norm.sum()),
        # char / word lengths
        "toxic_char_mean": float(np.mean(toxic_chars)) if toxic_chars else float("nan"),
        "toxic_char_median": float(np.median(toxic_chars)) if toxic_chars else float("nan"),
        "toxic_char_p95": pct(toxic_chars, 95),
        "toxic_char_max": int(max(toxic_chars)) if toxic_chars else 0,
        "detox_char_mean": float(np.mean(detox_chars)) if detox_chars else float("nan"),
        "detox_char_median": float(np.median(detox_chars)) if detox_chars else float("nan"),
        "detox_char_p95": pct(detox_chars, 95),
        "detox_char_max": int(max(detox_chars)) if detox_chars else 0,
        "toxic_word_mean": float(np.mean(toxic_words)) if toxic_words else float("nan"),
        "detox_word_mean": float(np.mean(detox_words)) if detox_words else float("nan"),
        "char_ratio_detox_over_toxic": (
            float(np.mean(detox_chars) / np.mean(toxic_chars)) if toxic_chars else float("nan")
        ),
        # mT5 token stats -- toxic side
        "toxic_tok_mean": tox_tok["tok_mean"],
        "toxic_tok_median": tox_tok["tok_median"],
        "toxic_tok_p95": tox_tok["tok_p95"],
        "toxic_tok_max": tox_tok["tok_max"],
        "toxic_fertility": tox_tok["fertility_mean"],
        "toxic_unk_token_rate": tox_tok["unk_token_rate"],
        "toxic_rows_with_unk": tox_tok["rows_with_unk"],
        "toxic_roundtrip_exact_rate": tox_tok["roundtrip_exact_rate"],
        # mT5 token stats -- detox side
        "detox_tok_mean": det_tok["tok_mean"],
        "detox_tok_median": det_tok["tok_median"],
        "detox_tok_p95": det_tok["tok_p95"],
        "detox_tok_max": det_tok["tok_max"],
        "detox_fertility": det_tok["fertility_mean"],
        "detox_unk_token_rate": det_tok["unk_token_rate"],
        "detox_rows_with_unk": det_tok["rows_with_unk"],
        "detox_roundtrip_exact_rate": det_tok["roundtrip_exact_rate"],
        # diversity / template concentration
        "toxic_type_token_ratio": tox_div["type_token_ratio"],
        "toxic_prefix_diversity": tox_div["prefix_diversity"],
        "toxic_top_prefix_share": tox_div["top_prefix_share"],
        "detox_prefix_diversity": det_div["prefix_diversity"],
        "detox_top_prefix_share": det_div["top_prefix_share"],
        # unicode hygiene
        "n_non_nfc_strings": nfc["n_non_nfc_strings"],
        "non_nfc_rate": nfc["non_nfc_rate"],
    }

    def _examples(mask: pd.Series) -> list[dict]:
        sel = df.loc[mask, ["toxic_input", "detox_output"]].head(MAX_FLAG_EXAMPLES)
        return [
            {"row_index": int(i), "toxic_input": str(r.toxic_input), "detox_output": str(r.detox_output)}
            for i, r in sel.iterrows()
        ]

    flags = {
        "missing_any": _examples(blank_any),
        "duplicate_pairs": _examples(dup_pair_mask),
        "identical_exact": _examples(identical_exact),
        "identical_normalized_only": _examples(identical_norm & ~identical_exact),
        "rows_with_unk_tokens": [],
    }
    # UNK examples are worth seeing verbatim for yo/xh.
    unk_id = tokenizer.unk_token_id
    for i, text in zip(valid.index, toxic_texts):
        if unk_id in tokenizer(text).input_ids:
            flags["rows_with_unk_tokens"].append({"row_index": int(i), "toxic_input": text})
        if len(flags["rows_with_unk_tokens"]) >= MAX_FLAG_EXAMPLES:
            break

    return row, flags


def main() -> int:
    cfg = load_config()
    encoding = cfg["io"]["csv_encoding"]
    data_dir = REPO_ROOT / cfg["io"]["data_processed"]
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"loading tokenizer: {cfg['models']['primary']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["primary"])

    targets: list[tuple[Path, str, str, bool]] = [
        (data_dir / f"{code}_detox.csv", code, cfg["language_names"][code], False)
        for code in cfg["languages"]
    ]
    targets.append(
        (data_dir / "en_detox_full.csv", "en_full", "English (supplementary)", True)
    )

    rows, all_flags = [], {}
    for path, code, name, supp in targets:
        row, flags = inventory_one(path, code, name, supp, tokenizer, encoding)
        rows.append(row)
        all_flags[code] = flags
        print(f"  inventoried {code:8s} rows={row['n_rows']:6d}")

    inv = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "corpus_inventory.csv"
    inv.to_csv(out_csv, index=False)
    (RESULTS_DIR / "corpus_inventory_flags.json").write_text(
        json.dumps(all_flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- console summary ----------------
    study = inv[~inv["is_supplementary"]]

    pd.set_option("display.width", 200)
    print("\n=== counts / integrity (study languages) ===")
    print(study[[
        "language", "n_rows", "n_missing_any", "n_duplicate_pairs",
        "n_repeated_toxic_inputs", "n_multiref_toxic_inputs",
        "n_identical_exact", "n_identical_normalized",
    ]].to_string(index=False))

    print("\n=== length statistics ===")
    print(study[[
        "language", "toxic_char_mean", "detox_char_mean", "char_ratio_detox_over_toxic",
        "toxic_word_mean", "toxic_tok_mean", "toxic_tok_p95", "toxic_tok_max",
    ]].round(2).to_string(index=False))

    print("\n=== mT5 tokenizer health (fertility = subword tokens per whitespace word) ===")
    print(study[[
        "language", "toxic_fertility", "detox_fertility", "toxic_unk_token_rate",
        "toxic_rows_with_unk", "toxic_roundtrip_exact_rate", "detox_roundtrip_exact_rate",
    ]].round(4).to_string(index=False))

    print("\n=== diversity / template concentration / unicode hygiene ===")
    print(study[[
        "language", "toxic_type_token_ratio", "toxic_prefix_diversity",
        "toxic_top_prefix_share", "detox_prefix_diversity", "detox_top_prefix_share",
        "n_non_nfc_strings",
    ]].round(3).to_string(index=False))

    afr = cfg["african_languages"]
    non_afr = cfg["non_african_languages"]
    f_afr = study.loc[study["language"].isin(afr), "toxic_fertility"].mean()
    f_non = study.loc[study["language"].isin(non_afr), "toxic_fertility"].mean()
    print(f"\nmean toxic fertility: African ({'/'.join(afr)}) = {f_afr:.2f} | "
          f"non-African = {f_non:.2f} | ratio = {f_afr / f_non:.2f}x")

    supp = inv[inv["is_supplementary"]]
    print("\n=== supplementary (NOT a study language) ===")
    print(supp[["language", "n_rows", "n_duplicate_pairs", "n_identical_exact",
                "toxic_tok_mean"]].round(2).to_string(index=False))

    # ---------------- sanity checks ----------------
    print("\n=== sanity checks ===")
    problems = []
    if study["n_rows"].isna().any() or (study["n_rows"] == 0).any():
        problems.append("a study language has zero rows")
    if (study["n_language_values"] != 1).any():
        problems.append("a language file mixes multiple language codes")
    if study[[c for c in study.columns if c.endswith(("_mean", "_median", "_p95"))]].isna().any().any():
        problems.append("NaN in a length statistic")
    for code in afr:
        if code not in set(study["language"]):
            problems.append(f"African language {code} missing from inventory")
    print("  no NaNs in length stats     :", not study[[
        "toxic_char_mean", "detox_char_mean", "toxic_tok_mean", "detox_tok_mean"
    ]].isna().any().any())
    print("  one language code per file  :", bool((study["n_language_values"] == 1).all()))
    print("  all 9 study languages       :", len(study) == len(cfg["languages"]))
    print("  status                      :", "PASS" if not problems else "FAIL")
    for p in problems:
        print("   PROBLEM:", p)

    print(f"\nwrote {out_csv.relative_to(REPO_ROOT)} and results/corpus_inventory_flags.json")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
