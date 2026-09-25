#!/usr/bin/env python3
"""
Evaluate Aya-23-8B QLoRA using the project's canonical metrics.

SIM = LaBSE cosine(toxic_input, prediction)
FL  = SacreBLEU chrF(prediction, detox_output) / 100
STA = 1.0 (existing project convention)
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


MODEL_ID = "CohereLabs/aya-23-8B"
DATA_PATH = Path("results/training_v2_400plus/seed_2024/test.csv")
TRAIN_ROOT = Path("results/training_aya_23_8b")
OUTPUT_ROOT = Path("results/evaluation/aya_23_8b")
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)

SEEDS = [42, 1337]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def generate_predictions(model, tokenizer, dataframe):
    model.eval()
    rows = []

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

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_length = inputs["input_ids"].shape[1]
            completion_ids = output_ids[0, prompt_length:]
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


def compute_metrics(dataframe: pd.DataFrame, embedder):
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

    fl_scores = []
    for _, row in dataframe.iterrows():
        try:
            fl_scores.append(
                sacrebleu.corpus_chrf(
                    [str(row["prediction"])],
                    [[str(row["detox_output"])]],
                ).score / 100.0
            )
        except Exception:
            fl_scores.append(0.5)

    dataframe["fl"] = fl_scores
    dataframe["is_copy"] = [
        normalize_text(prediction) == normalize_text(toxic)
        for prediction, toxic in zip(
            dataframe["prediction"],
            dataframe["toxic_input"],
        )
    ]
    dataframe["sta"] = 1.0
    dataframe["j"] = (
        dataframe["sta"] + dataframe["sim"] + dataframe["fl"]
    ) / 3.0
    dataframe["is_empty"] = dataframe["prediction"].map(
        lambda value: not bool(str(value).strip())
    )

    return dataframe


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    test_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    if len(test_df) != 553:
        raise RuntimeError(f"Expected 553 test rows; found {len(test_df)}")

    test_hash = sha256_file(DATA_PATH)
    expected_hash = (
        "d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61"
    )
    if test_hash != expected_hash:
        raise RuntimeError(f"Test split hash mismatch: {test_hash}")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Test examples:", len(test_df))
    print("Test SHA256:", test_hash)

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

    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    aggregate_rows = []

    for seed in SEEDS:
        adapter_path = TRAIN_ROOT / f"seed_{seed}" / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)

        out_dir = OUTPUT_ROOT / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Evaluating Aya seed {seed} ===")
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

        predictions = generate_predictions(model, tokenizer, test_df)
        predictions.to_csv(
            out_dir / "predictions_raw.csv",
            index=False,
            encoding="utf-8",
        )

        metrics_df = compute_metrics(predictions, embedder)
        metrics_df["model"] = f"aya_23_8b_seed_{seed}"
        metrics_df.to_csv(
            out_dir / "per_example_metrics.csv",
            index=False,
            encoding="utf-8",
        )

        aggregate = {
            "model": f"aya_23_8b_seed_{seed}",
            "seed": seed,
            "n_examples": int(len(metrics_df)),
            "sta": float(metrics_df["sta"].mean()),
            "sim": float(metrics_df["sim"].mean()),
            "fl": float(metrics_df["fl"].mean()),
            "j": float(metrics_df["j"].mean()),
            "copy_rate": float(metrics_df["is_copy"].mean()),
            "empty_rate": float(metrics_df["is_empty"].mean()),
        }

        pd.DataFrame([aggregate]).to_csv(
            out_dir / "aggregate_metrics.csv",
            index=False,
        )

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
            )
            .sort_values("language")
        )
        per_language.to_csv(
            out_dir / "per_language_metrics.csv",
            index=False,
        )

        manifest = {
            "model_id": MODEL_ID,
            "seed": seed,
            "adapter_path": str(adapter_path),
            "test_path": str(DATA_PATH),
            "test_sha256": test_hash,
            "metric_definition": {
                "sim": "LaBSE cosine(toxic_input, prediction)",
                "fl": "sacrebleu corpus_chrf(prediction, detox_output) / 100",
                "sta": "1.0",
                "j": "(sta + sim + fl) / 3",
            },
            "aggregate": aggregate,
        }
        (out_dir / "evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        aggregate_rows.append(aggregate)

        print(json.dumps(aggregate, indent=2))

        del model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(
        OUTPUT_ROOT / "aggregate_2seed_metrics.csv",
        index=False,
    )

    summary = {}
    for metric in ["copy_rate", "sim", "fl", "j", "empty_rate"]:
        summary[metric] = {
            "mean": float(aggregate_df[metric].mean()),
            "std": float(aggregate_df[metric].std(ddof=0)),
        }
    summary["model_id"] = MODEL_ID
    summary["seeds"] = SEEDS
    summary["n_examples"] = 553

    (OUTPUT_ROOT / "aggregate_2seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Aya two-seed aggregate ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
