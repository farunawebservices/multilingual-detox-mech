#!/usr/bin/env python3
"""
Rigorous activation steering for Aya-23-8B with detox-focused selection.

Protocol:
- Steering vector: train split only.
- Strength choice: stratified dev subset only, using detox-focused objective.
- Final metrics: frozen test at baseline (s=0) and selected (layer, strength).
- Test deltas always compare selected and baseline on the same frozen test set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------
# Data paths and constants
# ---------------------------

DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
ADAPTER_ROOT = Path("results/training_aya_23_8b")

DEV_PER_LANGUAGE = 8
TARGET_LAYER = 20  # default; will be overridden by sweep

# ---------------------------
# Helpers
# ---------------------------


def load_split(name: str) -> pd.DataFrame:
    path = DATA_ROOT / f"{name}.csv"
    return pd.read_csv(path, keep_default_na=False)


def stratified_dev_subset(dev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically sample up to DEV_PER_LANGUAGE examples per language,
    preserving all original columns, including `language`.
    """
    return (
        dev_df.groupby("language", group_keys=False)
        .sample(
            n=DEV_PER_LANGUAGE,
            random_state=42,
            replace=False,
        )
        .sort_index()
        .reset_index(drop=True)
    )


def get_decoder_layer(model, layer_idx: int):
    """
    Resolve a Cohere/Aya decoder block beneath the PEFT LoRA wrapper.

    PEFT model -> LoraModel -> CohereForCausalLM -> CohereModel -> layers.
    """
    return model.base_model.model.model.layers[layer_idx]


def make_prompt(tokenizer, toxic_text: str) -> str:
    prefix = (
        "Rewrite toxic text to be non-toxic while preserving its meaning "
        "and language. Output only the rewritten text.\n\nText: "
    )
    prompt = prefix + toxic_text
    if getattr(tokenizer, "bos_token", None):
        prompt = tokenizer.bos_token + prompt
    return prompt


# ---------------------------
# Toxicity classifier
# ---------------------------

toxicity_pipe = None


def init_toxicity_classifier():
    global toxicity_pipe
    if toxicity_pipe is not None:
        return
    from transformers import pipeline

    # Multilingual toxicity classifier (XLM-R based)
    model_name = "SkolkovoInstitute/roberta_toxicity_classifier"
    toxicity_pipe = pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=0,
    )


def score_toxicity(texts: list[str]) -> list[float]:
    """
    Return a toxicity score in [0, 1] for each text.
    Higher = more toxic.
    """
    global toxicity_pipe
    outs = toxicity_pipe(texts)
    scores = []
    for out in outs:
        # out is a list of label/score dicts
        # Assume one of the labels is "toxic"
        toxic_score = 0.0
        for item in out:
            if item["label"].lower() == "toxic":
                toxic_score = item["score"]
                break
        scores.append(toxic_score)
    return scores


# ---------------------------
# Steering vector
# ---------------------------


def collect_steering_vector(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    layer_idx: int = TARGET_LAYER,
):
    model.eval()
    toxic_states = []
    detox_states = []

    for i, row in train_df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"  Vector activations: {i + 1}/{len(train_df)}")

        for text, target_states in [
            (str(row["toxic_input"]), toxic_states),
            (str(row["detox_output"]), detox_states),
        ]:
            prompt = make_prompt(tokenizer, text)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[layer_idx][0, -1, :]
                target_states.append(hidden.detach().cpu())

    vector = torch.stack(detox_states).mean(0) - torch.stack(toxic_states).mean(0)
    return vector, len(toxic_states)


# ---------------------------
# Generation
# ---------------------------


def generate_one(
    model,
    tokenizer,
    toxic_text: str,
    steering_vector: torch.Tensor,
    strength: float,
) -> str:
    prompt = make_prompt(tokenizer, str(toxic_text))
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    vector = steering_vector.to(model.device, dtype=torch.bfloat16)

    def hook_fn(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] = hidden[:, -1, :] + strength * vector
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = get_decoder_layer(model, TARGET_LAYER).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    return tokenizer.decode(
        out_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    steering_vector: torch.Tensor,
    strength: float,
) -> pd.DataFrame:
    preds = []
    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"  dev@{strength}: {i + 1}/{len(df)}")
        prediction = generate_one(
            model,
            tokenizer,
            row["toxic_input"],
            steering_vector,
            strength,
        )
        preds.append(
            {
                "language": row["language"],
                "toxic_input": row["toxic_input"],
                "detox_output": row["detox_output"],
                "prediction": prediction,
            }
        )
    return pd.DataFrame(preds)


# ---------------------------
# Metrics
# ---------------------------


def compute_metrics(
    predictions_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> dict:
    df = predictions_df.copy()

    source_embeddings = source_df["toxic_input"].apply(
        lambda x: embedder.encode([str(x)], normalize_embeddings=True)[0]
    )
    output_embeddings = df["prediction"].apply(
        lambda x: embedder.encode([str(x)], normalize_embeddings=True)[0]
    )
    source_embeddings = np.stack(source_embeddings.tolist())
    output_embeddings = np.stack(output_embeddings.tolist())

    df["sim"] = np.sum(source_embeddings * output_embeddings, axis=1)

    fl_scores = []
    for pred, source in zip(df["prediction"], df["toxic_input"]):
        result = sacrebleu.corpus_bleu(
            [str(pred)],
            [[str(source)]],
            tokenize="13a",
            smooth_method="exp",
        )
        fl_scores.append(result.score / 100.0)
    df["fl"] = fl_scores

    df["is_copy"] = [
        pred.strip() == source.strip()
        for pred, source in zip(df["prediction"], df["toxic_input"])
    ]

    # Toxicity scoring
    toxic_scores = score_toxicity(df["prediction"].astype(str).tolist())
    df["toxic_score"] = toxic_scores
    df["sta"] = 1.0 - df["toxic_score"]  # higher = less toxic

    df["j"] = (df["sta"] + df["sim"] + df["fl"]) / 3.0

    return {
        "n_examples": len(df),
        "copy_rate": float(df["is_copy"].mean()),
        "sim": float(df["sim"].mean()),
        "fl": float(df["fl"].mean()),
        "sta": float(df["sta"].mean()),
        "toxic_score": float(df["toxic_score"].mean()),
        "j": float(df["j"].mean()),
    }


# ---------------------------
# Main
# ---------------------------

embedder = None


def main():
    global embedder

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)

    train_df = load_split("train")
    dev_df = load_split("dev")
    test_df = load_split("test")

    dev_subset = stratified_dev_subset(dev_df)

    print("Train / dev-subset / test:", len(train_df), len(dev_subset), len(test_df))

    adapter_path = ADAPTER_ROOT / f"seed_{seed}" / "adapter"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    embedder = SentenceTransformer("sentence-transformers/LaBSE")

    # Initialize toxicity classifier
    init_toxicity_classifier()

    # -----------------------
    # Dev sweep: layers + strengths
    # -----------------------

    LAYERS = [12, 16, 20, 24]
    STRENGTHS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    all_results = []

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        vector, n_vec = collect_steering_vector(model, tokenizer, train_df, layer_idx=layer)
        vector = vector / vector.norm()
        print(f"Vector norm: {vector.norm().item():.4f} from {n_vec} train examples")

        for strength in STRENGTHS:
            preds = predict_dataframe(model, tokenizer, dev_subset, vector, strength)
            metrics = compute_metrics(preds, dev_subset)

            # Detox-focused objective: prioritize low toxicity, preserve sim/fl
            w_sta, w_sim, w_fl = 0.6, 0.2, 0.2
            detox_obj = w_sta * metrics["sta"] + w_sim * metrics["sim"] + w_fl * metrics["fl"]

            all_results.append(
                {
                    "layer": layer,
                    "strength": strength,
                    "detox_obj": detox_obj,
                    "metrics": metrics,
                }
            )
            print(
                f"  s={strength:5.1f}: sta={metrics['sta']:.4f}, "
                f"sim={metrics['sim']:.4f}, fl={metrics['fl']:.4f}, "
                f"obj={detox_obj:.4f}"
            )

    best = max(all_results, key=lambda r: r["detox_obj"])
    selected_layer = best["layer"]
    selected_strength = best["strength"]
    print(f"\nSelected (layer, strength): ({selected_layer}, {selected_strength})")

    # -----------------------
    # Frozen test evaluation
    # -----------------------

    # Baseline at s=0, layer=selected_layer
    vector_base, _ = collect_steering_vector(
        model, tokenizer, train_df, layer_idx=selected_layer
    )
    vector_base = vector_base / vector_base.norm()

    print("\n=== Frozen test baseline ===")
    preds_base = predict_dataframe(model, tokenizer, test_df, vector_base, 0.0)
    baseline_metrics = compute_metrics(preds_base, test_df)
    print("Baseline:", json.dumps(baseline_metrics, indent=2))

    # Steered at selected strength
    print("\n=== Frozen test steered ===")
    preds_steered = predict_dataframe(
        model, tokenizer, test_df, vector_base, selected_strength
    )
    steered_metrics = compute_metrics(preds_steered, test_df)
    print("Steered:", json.dumps(steered_metrics, indent=2))

    # Deltas
    delta = {
        key: steered_metrics[key] - baseline_metrics[key]
        for key in ["copy_rate", "sim", "fl", "sta", "toxic_score", "j"]
    }

    # Test hash
    test_hash = hashlib.sha256(
        test_df.to_csv(index=False).encode("utf-8")
    ).hexdigest()

    summary = {
        "model_id": MODEL_ID,
        "seed": seed,
        "target_layer": int(selected_layer),
        "vector_source": "train.csv only",
        "vector_train_examples": n_vec,
        "dev_subset_size": len(dev_subset),
        "test_sha256": test_hash,
        "layers_searched": LAYERS,
        "strengths_searched": STRENGTHS,
        "selected_strength": float(selected_strength),
        "dev_best_detox_obj": float(best["detox_obj"]),
        "test_baseline": baseline_metrics,
        "test_steered": steered_metrics,
        "delta_steered_minus_baseline": delta,
    }

    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))

    out_dir = Path("results/mechanistic/aya_steering")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}_detox_sweep.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    MODEL_ID = "CohereLabs/aya-23-8B"
    main()
