#!/usr/bin/env python3
"""
Evaluate one Qwen2.5-7B-Instruct QLoRA seed using the project's
existing mT5/mT0/Llama metric definitions.

Canonical metrics:
  SIM = cosine(LaBSE(toxic_input), LaBSE(prediction))
  FL  = chrF(prediction, detox_output) / 100
  STA = 1.0, matching the existing project evaluator convention
  J   = (STA + SIM + FL) / 3
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = Path("results/training_v2_400plus/seed_2024/test.csv")
TRAIN_ROOT = Path("results/training_qwen2_5_7b")
EVAL_ROOT = Path("results/evaluation/qwen2_5_7b_qlora")
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def make_prompt(tokenizer, toxic_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": toxic_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def compute_metrics(dataframe: pd.DataFrame, embedder) -> pd.DataFrame:
    dataframe = dataframe.copy()

    toxic_embeddings = embedder.encode(
        dataframe["toxic_input"].tolist(),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    prediction_embeddings = embedder.encode(
        dataframe["prediction"].tolist(),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    dataframe["sim"] = np.sum(
        toxic_embeddings * prediction_embeddings,
        axis=1,
    )

    fluency_scores = []
    for _, row in dataframe.iterrows():
        try:
            score = sacrebleu.corpus_chrf(
                [str(row["prediction"])],
                [[str(row["detox_output"])]],
            ).score / 100.0
        except Exception:
            score = 0.5
        fluency_scores.append(score)

    dataframe["fl"] = fluency_scores
    dataframe["is_copy"] = [
        normalize_text(prediction) == normalize_text(toxic)
        for prediction, toxic in zip(
            dataframe["prediction"],
            dataframe["toxic_input"],
        )
    ]

    # Canonical project convention used in existing mT5/mT0/Llama tables.
    dataframe["sta"] = 1.0
    dataframe["j"] = (
        dataframe["sta"] + dataframe["sim"] + dataframe["fl"]
    ) / 3.0

    dataframe["is_empty"] = dataframe["prediction"].map(
        lambda value: not bool(str(value).strip())
    )
    dataframe["prediction_chars"] = dataframe["prediction"].map(
        lambda value: len(str(value))
    )

    return dataframe


def generate_predictions(model, tokenizer, dataframe) -> pd.DataFrame:
    rows = []
    model.eval()

    with torch.inference_mode():
        for row_index, row in dataframe.iterrows():
            prompt = make_prompt(tokenizer, str(row["toxic_input"]))

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_token_count = inputs["input_ids"].shape[1]
            completion_ids = generated_ids[0, prompt_token_count:]

            prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            rows.append(
                {
                    "pair_id": int(row_index),
                    "language": row["language"],
                    "toxic_input": row["toxic_input"],
                    "detox_output": row["detox_output"],
                    "prediction": prediction,
                }
            )

            if (row_index + 1) % 50 == 0:
                print(f"Generated {row_index + 1}/{len(dataframe)}")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    seed = args.seed

    adapter_path = TRAIN_ROOT / f"seed_{seed}" / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test split not found: {DATA_PATH}")

    out_dir = EVAL_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    if len(test_df) != 553:
        raise RuntimeError(
            f"Unexpected test count: {len(test_df)}; expected 553."
        )

    expected_hash = (
        "d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61"
    )
    observed_hash = sha256_file(DATA_PATH)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Test hash mismatch: {observed_hash} != {expected_hash}"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Test examples:", len(test_df))
    print("Test SHA256:", observed_hash)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    print("Generating full held-out test predictions...")
    predictions = generate_predictions(model, tokenizer, test_df)

    raw_path = out_dir / "predictions_raw.csv"
    predictions.to_csv(raw_path, index=False, encoding="utf-8")

    print("Loading canonical LaBSE embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")

    print("Computing canonical metrics...")
    metrics_df = compute_metrics(predictions, embedder)
    metrics_df["model"] = f"qwen2_5_7b_seed_{seed}"

    per_example_path = out_dir / "per_example_metrics.csv"
    metrics_df.to_csv(per_example_path, index=False, encoding="utf-8")

    aggregate = {
        "model": f"qwen2_5_7b_seed_{seed}",
        "seed": seed,
        "n_examples": int(len(metrics_df)),
        "sta": float(metrics_df["sta"].mean()),
        "sim": float(metrics_df["sim"].mean()),
        "fl": float(metrics_df["fl"].mean()),
        "j": float(metrics_df["j"].mean()),
        "copy_rate": float(metrics_df["is_copy"].mean()),
        "empty_rate": float(metrics_df["is_empty"].mean()),
        "unique_output_rate": float(
            metrics_df["prediction"].nunique() / len(metrics_df)
        ),
        "top_output_share": float(
            metrics_df["prediction"].value_counts(
                dropna=False,
                normalize=True,
            ).iloc[0]
        ),
        "mean_output_chars": float(
            metrics_df["prediction_chars"].mean()
        ),
    }

    aggregate_df = pd.DataFrame([aggregate])
    aggregate_path = out_dir / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    per_language = (
        metrics_df.groupby("language", as_index=False)
        .agg(
            n_examples=("pair_id", "count"),
            sta=("sta", "mean"),
            sim=("sim", "mean"),
            fl=("fl", "mean"),
            j=("j", "mean"),
            copy_rate=("is_copy", "mean"),
            empty_rate=("is_empty", "mean"),
            unique_output_rate=(
                "prediction",
                lambda values: values.nunique() / len(values),
            ),
            top_output_share=(
                "prediction",
                lambda values: values.value_counts(
                    dropna=False,
                    normalize=True,
                ).iloc[0],
            ),
            mean_output_chars=("prediction_chars", "mean"),
        )
        .sort_values("language")
    )

    per_language_path = out_dir / "per_language_metrics.csv"
    per_language.to_csv(per_language_path, index=False)

    manifest = {
        "model_id": MODEL_ID,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "test_path": str(DATA_PATH),
        "test_sha256": observed_hash,
        "metric_definition": {
            "sim": "LaBSE cosine(toxic_input, prediction)",
            "fl": "sacrebleu corpus_chrf(prediction, detox_output) / 100",
            "sta": "1.0 (canonical existing project convention)",
            "j": "(sta + sim + fl) / 3",
            "copy_rate": "normalized prediction equals normalized toxic_input",
        },
        "aggregate": aggregate,
    }

    manifest_path = out_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ QWEN SEED EVALUATION ================")
    for key, value in aggregate.items():
        print(f"{key}: {value}")

    print("\nPer-language metrics:")
    print(per_language.to_string(index=False))

    print("\nSaved:")
    print(raw_path)
    print(per_example_path)
    print(aggregate_path)
    print(per_language_path)
    print(manifest_path)

    del embedder
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
