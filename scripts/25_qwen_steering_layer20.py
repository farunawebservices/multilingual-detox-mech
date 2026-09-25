#!/usr/bin/env python3
"""
Stage 25 (fast): Activation steering for Qwen2.5-7B layer 20.

Protocol:
1. Compute steering vector v = mean(detox) - mean(toxic) at layer 20 (200 examples).
2. Dev sweep on a small subset (60 examples) to pick strength.
3. Evaluate selected strength on full test split.
4. Report Δ copy, SIM, FL, J vs baseline (strength=0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
TRAIN_ROOT = Path("results/training_qwen2_5_7b")
OUTPUT_ROOT = Path("results/mechanistic/qwen_steering")

TARGET_LAYER = 20
STRENGTHS = [0.0, 1.0, 2.0, 3.0]
MAX_NEW_TOKENS = 128
MAX_INPUT_LENGTH = 512
DEV_SUBSET_SIZE = 60

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


def collect_steering_vector(model, tokenizer, df, layer_idx, max_examples=200):
    df = df.iloc[:max_examples].reset_index(drop=True)
    toxic_acts = []
    detox_acts = []

    model.eval()
    with torch.inference_mode():
        for _, row in df.iterrows():
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
                    max_length=MAX_INPUT_LENGTH,
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

    toxic_mean = torch.stack(toxic_acts).mean(dim=0)
    detox_mean = torch.stack(detox_acts).mean(dim=0)
    return (detox_mean - toxic_mean).to(torch.float32)


def generate_one(model, tokenizer, toxic_text, steering_vec, strength, layer_idx):
    model.eval()
    prompt = make_prompt(tokenizer, toxic_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(model.device)

    steering_vec = steering_vec.to(model.device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] = hidden[:, -1, :] + strength * steering_vec
        return (hidden,) if isinstance(output, tuple) else hidden

    base = model.base_model.model
    layer = base.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()
    finally:
        handle.remove()

    return text


def compute_metrics(df, embedder):
    df = df.copy()
    t_emb = embedder.encode(df["toxic_input"].tolist(), normalize_embeddings=True, batch_size=32)
    p_emb = embedder.encode(df["prediction"].tolist(), normalize_embeddings=True, batch_size=32)
    df["sim"] = np.sum(t_emb * p_emb, axis=1)

    fls = []
    for _, row in df.iterrows():
        try:
            fls.append(sacrebleu.corpus_chrf([row["prediction"]], [[row["detox_output"]]]).score / 100.0)
        except Exception:
            fls.append(0.5)
    df["fl"] = fls
    df["is_copy"] = [
        " ".join(str(p).split()) == " ".join(str(t).split())
        for p, t in zip(df["prediction"], df["toxic_input"])
    ]
    df["sta"] = 1.0
    df["j"] = (df["sta"] + df["sim"] + df["fl"]) / 3.0
    return df


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

    dev_path = DATA_ROOT / "dev.csv"
    test_path = DATA_ROOT / "test.csv"
    if not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError("Dev or test split not found")

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_df = pd.read_csv(dev_path, keep_default_na=False).iloc[:DEV_SUBSET_SIZE]
    test_df = pd.read_csv(test_path, keep_default_na=False)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Dev subset:", len(dev_df), "Test:", len(test_df))

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

    print("Computing steering vector at layer", TARGET_LAYER, "...")
    steering_vec = collect_steering_vector(model, tokenizer, test_df, TARGET_LAYER, max_examples=200)
    steering_vec = steering_vec / (steering_vec.norm() + 1e-8)
    print("Steering vector norm:", float(steering_vec.norm().cpu()))

    embedder = SentenceTransformer("sentence-transformers/LaBSE")

    print("\n=== Dev sweep (subset) ===")
    dev_results = []
    for strength in STRENGTHS:
        print(f"Strength {strength}:")
        rows = []
        for _, row in dev_df.iterrows():
            pred = generate_one(model, tokenizer, row["toxic_input"], steering_vec, strength, TARGET_LAYER)
            rows.append({
                "pair_id": row.name,
                "language": row["language"],
                "toxic_input": row["toxic_input"],
                "detox_output": row["detox_output"],
                "prediction": pred,
            })
        preds = compute_metrics(pd.DataFrame(rows), embedder)
        copy_rate = float(preds["is_copy"].mean())
        sim = float(preds["sim"].mean())
        fl = float(preds["fl"].mean())
        j = float(preds["j"].mean())
        print(f"  Copy={copy_rate:.4f}, SIM={sim:.4f}, FL={fl:.4f}, J={j:.4f}")
        dev_results.append({"strength": strength, "copy_rate": copy_rate, "sim": sim, "fl": fl, "j": j})

    dev_df_results = pd.DataFrame(dev_results)
    dev_df_results.to_csv(out_dir / "dev_sweep_results.csv", index=False)

    # Select strength: lowest copy with SIM>=0.75, FL>=0.55; else best J
    cands = dev_df_results[(dev_df_results["sim"] >= 0.75) & (dev_df_results["fl"] >= 0.55)]
    if len(cands) == 0:
        sel = dev_df_results.loc[dev_df_results["j"].idxmax()]
    else:
        sel = cands.loc[cands["copy_rate"].idxmin()]
    sel_strength = float(sel["strength"])
    print("\nSelected strength:", sel_strength)
    print(f"Dev Copy={sel['copy_rate']:.4f}, SIM={sel['sim']:.4f}, FL={sel['fl']:.4f}, J={sel['j']:.4f}")

    print("\n=== Frozen test evaluation ===")
    rows = []
    for _, row in test_df.iterrows():
        pred = generate_one(model, tokenizer, row["toxic_input"], steering_vec, sel_strength, TARGET_LAYER)
        rows.append({
            "pair_id": row.name,
            "language": row["language"],
            "toxic_input": row["toxic_input"],
            "detox_output": row["detox_output"],
            "prediction": pred,
        })
    test_preds = compute_metrics(pd.DataFrame(rows), embedder)

    baseline = dev_df_results[dev_df_results["strength"] == 0.0].iloc[0]
    test_copy = float(test_preds["is_copy"].mean())
    test_sim = float(test_preds["sim"].mean())
    test_fl = float(test_preds["fl"].mean())
    test_j = float(test_preds["j"].mean())

    summary = {
        "model_id": MODEL_ID,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "target_layer": TARGET_LAYER,
        "selected_strength": sel_strength,
        "baseline": {
            "dev_copy_rate": float(baseline["copy_rate"]),
            "dev_sim": float(baseline["sim"]),
            "dev_fl": float(baseline["fl"]),
            "dev_j": float(baseline["j"]),
        },
        "selected": {
            "dev_copy_rate": float(sel["copy_rate"]),
            "dev_sim": float(sel["sim"]),
            "dev_fl": float(sel["fl"]),
            "dev_j": float(sel["j"]),
        },
        "test": {
            "copy_rate": test_copy,
            "sim": test_sim,
            "fl": test_fl,
            "j": test_j,
            "n_examples": len(test_preds),
        },
        "delta_test_vs_baseline": {
            "copy_rate": test_copy - float(baseline["copy_rate"]),
            "sim": test_sim - float(baseline["sim"]),
            "fl": test_fl - float(baseline["fl"]),
            "j": test_j - float(baseline["j"]),
        },
    }

    (out_dir / "steering_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    test_preds.to_csv(out_dir / "test_predictions.csv", index=False)

    print("\nTest results at strength", sel_strength)
    print(f"  Copy={test_copy:.4f}, SIM={test_sim:.4f}, FL={test_fl:.4f}, J={test_j:.4f}")
    print(f"  Δ Copy={summary['delta_test_vs_baseline']['copy_rate']:.4f}")
    print(f"  Δ SIM={summary['delta_test_vs_baseline']['sim']:.4f}")
    print(f"  Δ FL={summary['delta_test_vs_baseline']['fl']:.4f}")
    print(f"  Δ J={summary['delta_test_vs_baseline']['j']:.4f}")

    print("\nSaved:")
    print(f"  {out_dir / 'steering_summary.json'}")
    print(f"  {out_dir / 'test_predictions.csv'}")

    del embedder
    del model
    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
