#!/usr/bin/env python
"""Stage 6 -- Activation collection from the frozen Stage 3 checkpoints.

Registers forward hooks on mT5 encoder and decoder blocks {4, 6, 8} and records
hidden states for three conditions per example, for every seed, language and split.
Models are loaded in eval mode under torch.no_grad and are never modified.

Usage
-----
    python scripts/06_collect_activations.py --smoke   # small verification pass, writes nothing
    python scripts/06_collect_activations.py           # full collection

Conditions
----------
toxic             the toxic input, as the model actually receives it (with task prefix)
detox_reference   the human reference detoxification
generated         the model's own Stage 4 output (dev/test only -- Stage 4 did not
                  generate on train, so this condition is absent there and is recorded
                  as such rather than silently skipped)

Encoder activations come from encoding the condition text directly. Decoder activations
come from a teacher-forced pass whose *encoder* input is always the toxic input -- the
real task input -- with the decoder forced onto the condition text. So the decoder
states answer "how does the model represent this target while solving this example",
which is the representation the detoxification circuit actually produces.

Layer indexing
--------------
Layer N means `model.{encoder,decoder}.block[N]`, zero-indexed, so layer 4 is the fifth
of twelve blocks. Hooks capture each block's output hidden state, which in T5 is taken
*before* the final layer norm applied after the whole stack. Module names are recorded
verbatim in the manifest so the extraction point is unambiguous.

Test-split discipline
---------------------
Test activations are written to their own artifacts and nothing in this stage inspects
them. No layer, position, feature or example is chosen using test data: the layer set
comes from configs/experiment.yaml, positions are all valid (non-pad) tokens, and every
row is collected. Evaluation metrics are attached to the row index as metadata for later
stages, but no selection here is a function of them.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Train activations are collected too. The project brief designates train+dev as the
  probe/SAE training data, and dev alone is 243 rows across nine languages -- 27 per
  language, far too few to fit a probe per language, let alone an SAE. Train carries no
  `generated` condition because Stage 4 did not generate on it.
* The task prefix "detoxify: " is applied to all three encoder conditions, not just the
  toxic input. The reference and generated texts are not natural model inputs, so the
  choice is arbitrary either way; applying it uniformly keeps the three conditions
  directly comparable, which is what the toxic-vs-detox contrast in Stage 7 requires.
  The prefix span is recorded per row so it can be excluded at analysis time.
* Two poolings are stored: the mean over all valid positions, and the mean over content
  positions only (excluding the task prefix on the encoder side and the decoder start
  token on the decoder side). Per-token states are also stored, since Stage 8's SAE
  needs them; pooled vectors alone would foreclose that.
* Per-token states are stored as float16 and pooled vectors as float32. The pooled
  vectors are what probes consume, so they keep full precision; the token tensors are
  bulk data where halving the footprint matters more than the last few bits.
* Behaviour labels are derived only from structural properties of the generations --
  copy flags and output repetition counts -- never from STA/SIM/FL. Deriving them from
  metrics would make the labels a function of test scores, which the brief forbids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, MT5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
GEN_DIR = REPO_ROOT / "results" / "generations"
EVAL_DIR = REPO_ROOT / "results" / "evaluation"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
ACT_DIR = REPO_ROOT / "results" / "activations"

TASK_PREFIX = "detoxify: "
CONDITIONS = ["toxic", "detox_reference", "generated"]
MAX_SOURCE_LENGTH = 96
MAX_TARGET_LENGTH = 112
BATCH_SIZE = 16
# mT5 hidden states reach ~2e5 in these blocks, well past float16's 65504 ceiling, so a
# float16 store silently turns the largest activations into inf. bfloat16 has float32's
# exponent range at the same 2 bytes, which is the tradeoff that fits here: the outliers
# are the interesting part, their precision is not.
TOKEN_DTYPE = torch.bfloat16
POOLED_DTYPE = torch.float32

BEHAVIOUR_LABELS = ["copy_input", "generic_template", "ordinary_rewrite",
                    "other_uncertain", "no_generation"]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def file_digest(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_targets() -> list[Path]:
    paths = [SPLITS_DIR / f"{s}.csv" for s in ("train", "dev", "test")]
    paths += sorted(GEN_DIR.glob("seed*_*.csv"))
    paths += sorted(MODEL_DIR.glob("seed*/model.safetensors"))
    return [p for p in paths if p.exists()]


# --------------------------------------------------------------------------- rows

def behaviour_label(row: pd.Series, template_outputs: set[str]) -> str:
    if not isinstance(row.get("generated"), str) or row.get("generated") is None:
        return "no_generation"
    if row.get("is_empty") or row.get("has_sentinel"):
        return "other_uncertain"
    if row.get("is_copy_of_input_loose"):
        return "copy_input"
    if str(row["generated"]).strip() in template_outputs:
        return "generic_template"
    if str(row["generated"]).strip() and str(row["generated"]).strip() != str(row["toxic_input"]).strip():
        return "ordinary_rewrite"
    return "other_uncertain"


def build_rows(split: str, seed: int, cfg: dict) -> pd.DataFrame:
    """Row metadata for one (split, seed), including behaviour labels and metric metadata."""
    base = pd.read_csv(SPLITS_DIR / f"{split}.csv")

    if split == "train":
        rows = base[["pair_id", "group_id", "language", "toxic_input", "detox_output"]].copy()
        rows = rows.rename(columns={"detox_output": "detox_reference"})
        rows["generated"] = None
        for col in ("is_empty", "has_sentinel", "is_copy_of_input", "is_copy_of_input_loose"):
            rows[col] = False
        rows["is_generic_template"] = False
        rows["behaviour_label"] = "no_generation"
    else:
        gen = pd.read_csv(GEN_DIR / f"seed{seed}_{split}.csv")
        rows = gen[["pair_id", "group_id", "language", "toxic_input", "detox_reference",
                    "generated", "is_empty", "has_sentinel", "is_copy_of_input",
                    "is_copy_of_input_loose"]].copy()
        rows["generated"] = rows["generated"].fillna("").astype(str)
        # Structural template detection: an output repeated within its language cell.
        template_by_lang = {}
        for lang, g in rows.groupby("language"):
            counts = Counter(g["generated"].str.strip())
            template_by_lang[lang] = {t for t, c in counts.items() if c >= 2 and t}
        rows["is_generic_template"] = [
            r["generated"].strip() in template_by_lang.get(r["language"], set())
            for _, r in rows.iterrows()
        ]
        rows["behaviour_label"] = [
            behaviour_label(r, template_by_lang.get(r["language"], set()))
            for _, r in rows.iterrows()
        ]

    rows["split"] = split
    rows["seed"] = seed

    # Metric metadata only -- never used to select anything in this stage.
    rows["sta_supported"] = rows["language"].isin(
        {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh"})
    metric_path = EVAL_DIR / f"{split}_scores.csv"
    for col in ("sta", "sta_unvalidated", "sim", "fl", "j", "j_unvalidated"):
        rows[col] = np.nan
    if metric_path.exists():
        sc = pd.read_csv(metric_path)
        sc = sc[sc["seed"] == seed][["pair_id", "sta", "sta_unvalidated", "sim", "fl",
                                     "j", "j_unvalidated"]]
        rows = rows.drop(columns=["sta", "sta_unvalidated", "sim", "fl", "j",
                                  "j_unvalidated"]).merge(sc, on="pair_id", how="left")
    return rows.reset_index(drop=True)


# --------------------------------------------------------------- hooks / extraction

class BlockCapture:
    """Captures the hidden-state output of selected encoder/decoder blocks."""

    def __init__(self, model, enc_layers: list[int], dec_layers: list[int]):
        self.store: dict[str, torch.Tensor] = {}
        self.handles = []
        self.module_names: dict[str, str] = {}
        for side, layers, stack in (("encoder", enc_layers, model.encoder),
                                    ("decoder", dec_layers, model.decoder)):
            for layer in layers:
                key = f"{side}.{layer}"
                name = f"{side}.block.{layer}"
                self.module_names[key] = name
                self.handles.append(
                    stack.block[layer].register_forward_hook(self._make_hook(key))
                )

    def _make_hook(self, key: str):
        def hook(_module, _inputs, output):
            # MT5Block returns a tuple; element 0 is the hidden state.
            hidden = output[0] if isinstance(output, tuple) else output
            self.store[key] = hidden.detach()
        return hook

    def clear(self) -> None:
        self.store.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def pack(per_row: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate variable-length per-row token states plus an offsets index."""
    lengths = torch.tensor([t.shape[0] for t in per_row], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    return (torch.cat(per_row, dim=0) if per_row else torch.empty(0), offsets)


@torch.no_grad()
def collect_for_condition(model, tokenizer, rows: pd.DataFrame, condition: str,
                          capture: BlockCapture, enc_layers: list[int],
                          dec_layers: list[int], device: str, prefix_len: int) -> dict:
    """One condition: encoder states over the condition text, decoder states teacher-forced."""
    texts = {
        "toxic": rows["toxic_input"].astype(str).tolist(),
        "detox_reference": rows["detox_reference"].astype(str).tolist(),
        "generated": rows["generated"].astype(str).tolist() if "generated" in rows else [],
    }[condition]
    texts = [t if t.strip() else " " for t in texts]
    toxic_texts = [t if str(t).strip() else " " for t in rows["toxic_input"].astype(str)]

    keys = [f"encoder.{l}" for l in enc_layers] + [f"decoder.{l}" for l in dec_layers]
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    pooled_all_acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    pooled_content_acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    max_abs: dict[str, float] = {k: 0.0 for k in keys}
    tok_ids: dict[str, list[torch.Tensor]] = {"encoder": [], "decoder": []}

    def stash(key: str, hidden: torch.Tensor, mask: torch.Tensor, skip: int) -> None:
        """Pool in float32 from the raw states, then store tokens at reduced precision."""
        for i in range(hidden.shape[0]):
            row = hidden[i][mask[i]].to(torch.float32)
            max_abs[key] = max(max_abs[key], float(row.abs().max()))
            pooled_all_acc[key].append(row.mean(0).cpu())
            body = row[skip:] if row.shape[0] > skip else row
            pooled_content_acc[key].append(body.mean(0).cpu())
            acc[key].append(row.to(TOKEN_DTYPE).cpu())

    for start in range(0, len(rows), BATCH_SIZE):
        cond_batch = texts[start:start + BATCH_SIZE]
        tox_batch = toxic_texts[start:start + BATCH_SIZE]

        # --- encoder pass over the condition text ---
        enc_in = tokenizer([TASK_PREFIX + t for t in cond_batch], return_tensors="pt",
                           padding=True, truncation=True,
                           max_length=MAX_SOURCE_LENGTH).to(device)
        capture.clear()
        model.encoder(input_ids=enc_in.input_ids, attention_mask=enc_in.attention_mask)
        enc_mask = enc_in.attention_mask.bool()
        for layer in enc_layers:
            stash(f"encoder.{layer}", capture.store[f"encoder.{layer}"], enc_mask, prefix_len)
        for i in range(enc_in.input_ids.shape[0]):
            tok_ids["encoder"].append(enc_in.input_ids[i][enc_mask[i]].cpu())

        # --- decoder pass: encoder sees the toxic input, decoder forced on condition ---
        src = tokenizer([TASK_PREFIX + t for t in tox_batch], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=MAX_SOURCE_LENGTH).to(device)
        tgt = tokenizer(text_target=cond_batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_TARGET_LENGTH).to(device)
        decoder_input_ids = model._shift_right(tgt.input_ids)
        capture.clear()
        model(input_ids=src.input_ids, attention_mask=src.attention_mask,
              decoder_input_ids=decoder_input_ids,
              decoder_attention_mask=tgt.attention_mask)
        dec_mask = tgt.attention_mask.bool()
        for layer in dec_layers:
            # skip=1 drops the decoder start token from the content pooling
            stash(f"decoder.{layer}", capture.store[f"decoder.{layer}"], dec_mask, 1)
        for i in range(decoder_input_ids.shape[0]):
            tok_ids["decoder"].append(decoder_input_ids[i][dec_mask[i]].cpu())

    out: dict = {}
    for key, per_row in acc.items():
        side = key.split(".")[0]
        tokens, offsets = pack(per_row)
        out[key] = {
            "tokens": tokens, "offsets": offsets,
            "pooled_mean_all": torch.stack(pooled_all_acc[key]).to(POOLED_DTYPE),
            "pooled_mean_content": torch.stack(pooled_content_acc[key]).to(POOLED_DTYPE),
            "token_ids": torch.cat(tok_ids[side]) if tok_ids[side] else torch.empty(0),
            "n_positions_per_row": (offsets[1:] - offsets[:-1]),
            "max_abs_activation": max_abs[key],
        }
    return out


# ------------------------------------------------------------------------- driver

def run(seed: int, split: str, cfg: dict, device: str, smoke: bool,
        limit: int | None = None) -> dict:
    enc_layers, dec_layers = cfg["activations"]["encoder_layers"], cfg["activations"]["decoder_layers"]
    ckpt = MODEL_DIR / f"seed{seed}"
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = MT5ForConditionalGeneration.from_pretrained(str(ckpt)).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prefix_len = len(tokenizer(TASK_PREFIX, add_special_tokens=False).input_ids)
    rows = build_rows(split, seed, cfg)
    if limit:
        rows = rows.groupby("language").head(max(1, limit // 9)).reset_index(drop=True)

    capture = BlockCapture(model, enc_layers, dec_layers)
    conditions = [c for c in CONDITIONS if not (split == "train" and c == "generated")]

    collected = {}
    for cond in conditions:
        collected[cond] = collect_for_condition(
            model, tokenizer, rows, cond, capture, enc_layers, dec_layers, device, prefix_len)
    capture.remove()

    meta = {
        "seed": seed, "split": split, "n_rows": int(len(rows)),
        "conditions": conditions,
        "task_prefix": TASK_PREFIX, "prefix_token_len": prefix_len,
        "max_source_length": MAX_SOURCE_LENGTH, "max_target_length": MAX_TARGET_LENGTH,
        "tokenizer": str(ckpt.relative_to(REPO_ROOT)),
        "module_names": capture.module_names,
        "hook_point": "MT5Block forward output[0] (pre final layer norm)",
        "layer_indexing": "zero-indexed block index; layer 4 is the 5th of 12 blocks",
        "token_dtype": str(TOKEN_DTYPE), "pooled_dtype": str(POOLED_DTYPE),
        "pooling": {
            "pooled_mean_all": "mean over all non-pad positions",
            "pooled_mean_content": ("mean over non-pad positions excluding the task prefix "
                                    "(encoder) or the decoder start token (decoder)"),
        },
        "positions": "all non-pad positions retained; offsets index gives per-row spans",
        "decoder_teacher_forcing": ("encoder input is always the toxic input; decoder is "
                                    "forced onto the condition text via _shift_right"),
    }

    if smoke:
        del model
        torch.cuda.empty_cache()
        return {"meta": meta, "collected": collected, "rows": rows}

    written = []
    for cond, per_key in collected.items():
        for key, payload in per_key.items():
            side, layer = key.split(".")
            outdir = ACT_DIR / side / f"layer_{int(layer):02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"{split}_seed{seed}_{cond}.pt"
            torch.save({
                "meta": {**meta, "side": side, "layer": int(layer), "condition": cond,
                         "module_name": meta["module_names"][key],
                         "tokens_shape": list(payload["tokens"].shape),
                         "pooled_shape": list(payload["pooled_mean_all"].shape)},
                "pair_id": rows["pair_id"].tolist(),
                "language": rows["language"].tolist(),
                **payload,
            }, path)
            written.append(path)

    idx_dir = ACT_DIR / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(idx_dir / f"{split}_seed{seed}_rows.csv", index=False, encoding="utf-8")

    # Per-dimension normalisation statistics, fitted on TRAIN ONLY.
    #
    # mT5 carries a handful of massive outlier dimensions (peak |activation| ~2e5 against
    # a typical scale of order 1). Raw cosine similarity between decoder states is ~0.999
    # for every pair of conditions purely because those dimensions dominate the dot
    # product, which would make any correlation-based "shared circuit" claim an artefact
    # of the scale rather than evidence. Later stages should standardise with these
    # statistics; fitting them on train alone keeps dev and test out of the transform.
    if split == "train":
        norm_dir = ACT_DIR / "norm_stats"
        norm_dir.mkdir(parents=True, exist_ok=True)
        for cond, per_key in collected.items():
            for key, payload in per_key.items():
                side, layer = key.split(".")
                toks = payload["tokens"].to(torch.float32)
                torch.save({
                    "fitted_on": "train split only",
                    "seed": seed, "side": side, "layer": int(layer), "condition": cond,
                    "token_mean": toks.mean(0), "token_std": toks.std(0),
                    "pooled_mean": payload["pooled_mean_content"].mean(0),
                    "pooled_std": payload["pooled_mean_content"].std(0),
                    "max_abs_activation": payload["max_abs_activation"],
                    "n_tokens": int(toks.shape[0]),
                }, norm_dir / f"seed{seed}_{side}_layer{int(layer):02d}_{cond}.pt")

    del model
    torch.cuda.empty_cache()
    return {"meta": meta, "written": written, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small verification pass, writes nothing")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seeds = args.seeds or cfg["seeds_multi"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = integrity_targets()
    before = {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in targets}
    print(f"integrity baseline over {len(before)} files (splits, generations, checkpoints)")

    if args.smoke:
        print("\n=== SMOKE TEST: seed 42, dev, 18 rows, nothing written ===")
        res = run(42, "dev", cfg, device, smoke=True, limit=18)
        m, rows = res["meta"], res["rows"]
        print(f"rows={m['n_rows']} conditions={m['conditions']}")
        print(f"prefix='{m['task_prefix']}' -> {m['prefix_token_len']} tokens")
        print(f"hook point: {m['hook_point']}")
        print(f"layer indexing: {m['layer_indexing']}")
        for cond, per_key in res["collected"].items():
            print(f"  [{cond}]")
            for key, p in sorted(per_key.items()):
                print(f"    {m['module_names'][key]:20s} tokens={list(p['tokens'].shape)} "
                      f"{p['tokens'].dtype} pooled={list(p['pooled_mean_all'].shape)} "
                      f"{p['pooled_mean_all'].dtype} "
                      f"pos/row min={int(p['n_positions_per_row'].min())} "
                      f"max={int(p['n_positions_per_row'].max())} "
                      f"max|act|={p['max_abs_activation']:.0f}")
        print(f"  behaviour labels: {rows['behaviour_label'].value_counts().to_dict()}")
        finite_pooled = all(torch.isfinite(p["pooled_mean_all"]).all()
                            and torch.isfinite(p["pooled_mean_content"]).all()
                            for per in res["collected"].values() for p in per.values())
        finite_tokens = all(torch.isfinite(p["tokens"].to(torch.float32)).all()
                            for per in res["collected"].values() for p in per.values())
        peak = max(p["max_abs_activation"]
                   for per in res["collected"].values() for p in per.values())
        print(f"  all pooled activations finite: {finite_pooled}")
        print(f"  all stored token activations finite: {finite_tokens}")
        print(f"  peak |activation| = {peak:.0f} (float16 ceiling is 65504, "
              f"bfloat16 ~3.4e38)")
        after = {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in targets}
        changed = [k for k in before if before[k] != after[k]]
        print(f"  inputs unchanged: {'YES' if not changed else 'NO ' + str(changed)}")
        print("\nSMOKE TEST COMPLETE -- no artifacts written.")
        return 0

    ACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries, all_rows = [], []
    for seed in seeds:
        for split in args.splits:
            res = run(seed, split, cfg, device, smoke=False)
            rows = res["rows"]
            all_rows.append(rows)
            manifest_entries.append({
                "seed": seed, "split": split, "n_rows": int(len(rows)),
                "n_languages": int(rows["language"].nunique()),
                "rows_per_language": rows["language"].value_counts().sort_index().to_dict(),
                "conditions": res["meta"]["conditions"],
                "behaviour_labels": rows["behaviour_label"].value_counts().to_dict(),
                "files": [str(p.relative_to(REPO_ROOT)) for p in res["written"]],
                "meta": res["meta"],
            })
            print(f"  seed {seed} {split:5s}: {len(rows):5d} rows, "
                  f"{len(res['written'])} tensor files")

    after = {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in targets}
    changed = [k for k in before if before[k] != after[k]]

    act_files = sorted(ACT_DIR.rglob("*.pt"))
    checksums = {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in act_files}
    total_bytes = sum(p.stat().st_size for p in act_files)

    rows_all = pd.concat(all_rows, ignore_index=True)
    manifest = {
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": cfg["git_commit"],
        "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT)),
        "encoder_layers": cfg["activations"]["encoder_layers"],
        "decoder_layers": cfg["activations"]["decoder_layers"],
        "conditions": CONDITIONS,
        "behaviour_label_definitions": {
            "copy_input": "generation equals the input after NFC, whitespace and case normalisation",
            "generic_template": "generation string occurs >=2x within its (seed, split, language) cell",
            "ordinary_rewrite": "non-empty, no sentinel, not a copy, not a repeated template",
            "other_uncertain": "empty or sentinel-bearing output, or a non-copy identical to input",
            "no_generation": "train split, where Stage 4 produced no generations",
            "derivation": "structural only; never a function of STA/SIM/FL",
        },
        "totals": {
            "rows": int(len(rows_all)),
            "seeds": int(rows_all["seed"].nunique()),
            "languages": int(rows_all["language"].nunique()),
            "tensor_files": len(act_files),
            "total_bytes": total_bytes,
        },
        "rows_by_split": rows_all.groupby("split").size().to_dict(),
        "rows_by_seed": rows_all.groupby("seed").size().to_dict(),
        "rows_by_language": rows_all.groupby("language").size().to_dict(),
        "behaviour_labels_overall": rows_all["behaviour_label"].value_counts().to_dict(),
        "entries": manifest_entries,
        "activation_file_checksums": checksums,
        "input_files_unchanged": not changed,
        "changed_input_files": changed,
    }
    (ACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== totals ===")
    print(f"  rows            : {manifest['totals']['rows']}")
    print(f"  seeds/languages : {manifest['totals']['seeds']} / {manifest['totals']['languages']}")
    print(f"  tensor files    : {manifest['totals']['tensor_files']} "
          f"({total_bytes / 1024**3:.2f} GiB)")
    print(f"  rows by split   : {manifest['rows_by_split']}")
    print(f"  behaviour labels: {manifest['behaviour_labels_overall']}")
    print(f"  inputs unchanged: {'YES' if not changed else 'NO -> ' + str(changed)}")
    print(f"\nwrote {ACT_DIR.relative_to(REPO_ROOT)}/manifest.json")
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
