#!/usr/bin/env python3
"""
Linear probes on Aya-23-8B layer 20 activations (toxic vs detox).

Reports:
- Within-language AUC (all)
- Within-language AUC (high-resource)
- Cross-lingual AUC (HR → yo/xh)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "CohereLabs/aya-23-8B"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
TRAIN_ROOT = Path("results/training_aya_23_8b")
OUTPUT_ROOT = Path("results/mechanistic/aya_probes")

TARGET_LAYER = 20
MAX_EXAMPLES = 553  # full test set

SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)


def make_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def collect_activations(model, tokenizer, dataframe, layer_idx, max_examples=None):
    if max_examples is not None:
        dataframe = dataframe.iloc[:max_examples].reset_index(drop=True)

    toxic_acts = []
    detox_acts = []
    languages = []

    model.eval()
    with torch.inference_mode():
        for _, row in dataframe.iterrows():
            for text, target_list in [
                (str(row["toxic_input"]), toxic_acts),
                (str(row["detox_output"]), detox_acts),
            ]:
                prompt = make_prompt(tokenizer, text)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )

                hidden = (
                    outputs.hidden_states[layer_idx][0, -1, :]
                    .float()
                    .cpu()
                )
                target_list.append(hidden)

            languages.append(row["language"])

    return (
        np.stack(toxic_acts, axis=0),
        np.stack(detox_acts, axis=0),
        languages,
    )


def train_and_evaluate_probe(toxic_act, detox_act, languages, high_resource_langs):
    X = np.concatenate([toxic_act, detox_act], axis=0)
    y = np.concatenate(
        [np.zeros(len(toxic_act), dtype=int), np.ones(len(detox_act), dtype=int)],
        axis=0,
    )
    langs = np.array(languages + languages)

    # Within-language AUC (all)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    probe_all = LogisticRegression(max_iter=1000, random_state=42)
    probe_all.fit(X_train, y_train)
    y_pred_all = probe_all.predict_proba(X_test)[:, 1]
    auc_all = roc_auc_score(y_test, y_pred_all)

    # Within-language AUC (high-resource only)
    hr_mask = np.isin(langs, list(high_resource_langs))
    X_hr = X[hr_mask]
    y_hr = y[hr_mask]
    if y_hr.sum() > 0 and (1 - y_hr).sum() > 0:
        X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(
            X_hr, y_hr, test_size=0.3, random_state=42, stratify=y_hr
        )
        probe_hr = LogisticRegression(max_iter=1000, random_state=42)
        probe_hr.fit(X_train_hr, y_train_hr)
        y_pred_hr = probe_hr.predict_proba(X_test_hr)[:, 1]
        auc_hr = roc_auc_score(y_test_hr, y_pred_hr)
    else:
        auc_hr = np.nan

    # Cross-lingual AUC (HR → yo/xh)
    lr_mask = ~np.isin(langs, list(high_resource_langs))
    X_lr = X[lr_mask]
    y_lr = y[lr_mask]

    if X_lr.shape[0] > 0 and y_hr.sum() > 0 and (1 - y_hr).sum() > 0:
        probe_cross = LogisticRegression(max_iter=1000, random_state=42)
        probe_cross.fit(X_hr, y_hr)
        y_pred_cross = probe_cross.predict_proba(X_lr)[:, 1]
        auc_cross = roc_auc_score(y_lr, y_pred_cross)
    else:
        auc_cross = np.nan

    return {
        "auc_all": float(auc_all),
        "auc_hr": float(auc_hr),
        "auc_cross": float(auc_cross),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed

    adapter_path = TRAIN_ROOT / f"seed_{seed}" / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    test_path = DATA_ROOT / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, keep_default_na=False)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Test examples:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.config.use_cache = True

    high_resource_langs = {"ar", "de", "en", "es", "hi", "uk", "am"}

    results = {
        "model_id": MODEL_ID,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "target_layer": TARGET_LAYER,
        "max_examples": MAX_EXAMPLES,
        "high_resource_langs": sorted(high_resource_langs),
        "linear_probe": {},
    }

    print(f"Collecting layer {TARGET_LAYER} activations...")
    toxic_act, detox_act, languages = collect_activations(
        model,
        tokenizer,
        test_df,
        TARGET_LAYER,
        max_examples=MAX_EXAMPLES,
    )
    print(f"Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")

    print("Training and evaluating probe...")
    metrics = train_and_evaluate_probe(
        toxic_act,
        detox_act,
        languages,
        high_resource_langs,
    )
    print(f"  AUC (all): {metrics['auc_all']:.4f}")
    print(f"  AUC (HR):  {metrics['auc_hr']:.4f}")
    print(f"  AUC (cross HR→LR): {metrics['auc_cross']:.4f}")

    results["linear_probe"] = metrics

    (out_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Append to summary CSV
    summary_path = OUTPUT_ROOT / "probe_summary.csv"
    rows = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict(orient="records")

    summary_row = {
        "model": "aya_23_8b",
        "seed": seed,
        "auc_all": metrics["auc_all"],
        "auc_hr": metrics["auc_hr"],
        "auc_cross": metrics["auc_cross"],
    }
    rows.append(summary_row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'probe_results.json'}")
    print(f"  {summary_path}")

    del model
    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
