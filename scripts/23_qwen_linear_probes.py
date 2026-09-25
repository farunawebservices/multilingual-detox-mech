#!/usr/bin/env python3
"""
Stage 23: Linear probes on Qwen2.5-7B-Instruct QLoRA activations.

Goal (RQ1: shared circuits):
- Train logistic regression to distinguish toxic vs detox activations.
- Measure within-language AUC.
- Measure cross-lingual transfer AUC (HR → yo/xh).

Uses best checkpoint from each seed (epoch 2 / step 1292).
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


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
TRAIN_ROOT = Path("results/training_qwen2_5_7b")
OUTPUT_ROOT = Path("results/mechanistic/qwen_probes")

# Probe on these decoder layers (0-indexed).
PROBE_LAYERS = [10, 15, 20]

# Use up to this many examples per seed for probe training/evaluation.
MAX_EXAMPLES_PER_SEED = 553

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


def collect_activations(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    layer_idx: int,
    max_examples: int | None = None,
):
    """
    Collect last-token hidden states for toxic and detox inputs.
    Returns:
      toxic_act: np.ndarray [n_examples, d_model]
      detox_act: np.ndarray [n_examples, d_model]
      languages: list[str]
    """
    if max_examples is not None:
        # Balanced language sampling: preserve low-resource yo/xh examples
        # instead of taking the first rows of a language-sorted test file.
        low_resource = dataframe[dataframe["language"].isin(["yo", "xh"])]
        high_resource = dataframe[~dataframe["language"].isin(["yo", "xh"])]

        remaining = max(0, max_examples - len(low_resource))
        if remaining > 0 and len(high_resource) > remaining:
            per_language = max(1, remaining // high_resource["language"].nunique())
            sampled_hr = (
                high_resource.groupby("language", group_keys=False)
                .apply(
                    lambda group: group.sample(
                        n=min(len(group), per_language),
                        random_state=42,
                    )
                )
            )
            # Fill any small remainder deterministically.
            if len(sampled_hr) < remaining:
                remaining_pool = high_resource.drop(sampled_hr.index)
                extra = remaining_pool.sample(
                    n=min(remaining - len(sampled_hr), len(remaining_pool)),
                    random_state=42,
                )
                sampled_hr = pd.concat([sampled_hr, extra])
        else:
            sampled_hr = high_resource

        dataframe = (
            pd.concat([low_resource, sampled_hr])
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

    toxic_acts = []
    detox_acts = []
    languages = []

    model.eval()
    with torch.inference_mode():
        for _, row in dataframe.iterrows():
            toxic_text = str(row["toxic_input"])
            detox_text = str(row["detox_output"])
            lang = str(row["language"])

            for text, target_list in [
                (toxic_text, toxic_acts),
                (detox_text, detox_acts),
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

                # Last token, specified layer, cast to float32 before NumPy.
                hidden = (
                    outputs.hidden_states[layer_idx][0, -1, :]
                    .float()
                    .cpu()
                    .numpy()
                )
                target_list.append(hidden)

            languages.append(lang)

    return (
        np.stack(toxic_acts, axis=0),
        np.stack(detox_acts, axis=0),
        languages,
    )


def train_and_evaluate_probe(
    toxic_act: np.ndarray,
    detox_act: np.ndarray,
    languages: list[str],
    high_resource_langs: set[str],
):
    """
    Train logistic regression probe and evaluate:
    - Within-language AUC (all)
    - Within-language AUC (high-resource only)
    - Cross-lingual AUC (train on HR, test on low-resource)
    """
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

    # Cross-lingual: train on HR, test on low-resource (yo/xh)
    lr_mask = ~np.isin(langs, list(high_resource_langs))
    X_lr = X[lr_mask]
    y_lr = y[lr_mask]

    if X_lr.shape[0] > 0 and y_hr.sum() > 0 and (1 - y_hr).sum() > 0:
        # Train on HR, test on LR
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
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    test_path = DATA_ROOT / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, keep_default_na=False)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Test examples:", len(test_df))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen base model in 4-bit...")
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

    # Define high-resource languages for cross-lingual analysis.
    high_resource_langs = {"ar", "de", "en", "es", "hi", "uk", "am"}

    results = {
        "model_id": MODEL_ID,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "probe_layers": PROBE_LAYERS,
        "max_examples_per_seed": MAX_EXAMPLES_PER_SEED,
        "high_resource_langs": sorted(high_resource_langs),
        "linear_probes": {},
    }

    for layer_idx in PROBE_LAYERS:
        print(f"\nLayer {layer_idx}: collecting activations...")
        toxic_act, detox_act, languages = collect_activations(
            model,
            tokenizer,
            test_df,
            layer_idx,
            max_examples=MAX_EXAMPLES_PER_SEED,
        )
        print(f"  Toxic activations: {toxic_act.shape}")
        print(f"  Detox activations: {detox_act.shape}")

        print("  Training and evaluating probe...")
        metrics = train_and_evaluate_probe(
            toxic_act,
            detox_act,
            languages,
            high_resource_langs,
        )
        print(f"    AUC (all): {metrics['auc_all']:.4f}")
        print(f"    AUC (HR):  {metrics['auc_hr']:.4f}")
        print(f"    AUC (cross HR→LR): {metrics['auc_cross']:.4f}")

        results["linear_probes"][layer_idx] = metrics

    # Save results
    (out_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Create a compact summary CSV row
    summary_path = OUTPUT_ROOT / "probe_summary.csv"
    summary_exists = summary_path.exists()
    rows = []
    if summary_exists:
        rows = pd.read_csv(summary_path).to_dict(orient="records")

    summary_row = {
        "model": "qwen2_5_7b",
        "seed": seed,
    }
    for layer_idx in PROBE_LAYERS:
        m = results["linear_probes"][layer_idx]
        summary_row[f"layer{layer_idx}_auc_all"] = m["auc_all"]
        summary_row[f"layer{layer_idx}_auc_hr"] = m["auc_hr"]
        summary_row[f"layer{layer_idx}_auc_cross"] = m["auc_cross"]

    rows.append(summary_row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'probe_results.json'}")
    print(f"  {summary_path}")

    del model
    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
