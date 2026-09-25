#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct QLoRA fine-tuning for multilingual detoxification.

Uses the frozen v2_400plus grouped splits:
  train.csv: training only
  dev.csv: checkpoint selection
  test.csv: held-out generation/evaluation

This script trains one seed per invocation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
OUTPUT_ROOT = Path("results/training_qwen2_5_7b")
MAX_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_prompt(tokenizer, toxic_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite toxic text to be non-toxic while preserving its "
                "meaning and the original language. Return only the rewritten text."
            ),
        },
        {"role": "user", "content": toxic_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_row(row: pd.Series, tokenizer):
    prompt = make_prompt(tokenizer, str(row["toxic_input"]))
    target = str(row["detox_output"]) + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    full = tokenizer(
        prompt + target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)

    # Output-only objective: never train on prompt tokens.
    mask_until = min(len(prompt_ids), len(labels))
    labels[:mask_until] = [-100] * mask_until

    if not any(label != -100 for label in labels):
        raise ValueError(
            "Example has no unmasked target tokens after truncation. "
            f"Input preview: {str(row['toxic_input'])[:120]}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DetoxDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.examples = [
            encode_row(self.dataframe.iloc[i], tokenizer)
            for i in range(len(self.dataframe))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in item.items()
        }


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full(
                            (pad_len,),
                            pad_id,
                            dtype=torch.long,
                        ),
                    ]
                )
            )
            batch["attention_mask"].append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
            batch["labels"].append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full(
                            (pad_len,),
                            -100,
                            dtype=torch.long,
                        ),
                    ]
                )
            )

        return {key: torch.stack(value) for key, value in batch.items()}


def load_splits():
    paths = {
        split: DATA_ROOT / f"{split}.csv"
        for split in ["train", "dev", "test"]
    }

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

    data = {
        split: pd.read_csv(path, keep_default_na=False)
        for split, path in paths.items()
    }

    required = {"language", "toxic_input", "detox_output"}
    for split, dataframe in data.items():
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"{split} missing columns: {sorted(missing)}")

    return paths, data


def verify_splits(paths, data):
    expected = {
        "train": {
            "n": 2583,
            "sha256": "db5b7a82f81677e80b085a7bfa08b78aca0e4df49b5799305c0992638bcb2b70",
        },
        "dev": {
            "n": 558,
            "sha256": "93ad0bede18130c6009af552cda7f69b13abc3c551f28173165e8fd13e247802",
        },
        "test": {
            "n": 553,
            "sha256": "d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61",
        },
    }

    for split in ["train", "dev", "test"]:
        observed_n = len(data[split])
        observed_hash = sha256_file(paths[split])

        if observed_n != expected[split]["n"]:
            raise RuntimeError(
                f"{split} count mismatch: {observed_n} != {expected[split]['n']}"
            )

        if observed_hash != expected[split]["sha256"]:
            raise RuntimeError(
                f"{split} hash mismatch:\n"
                f"observed={observed_hash}\n"
                f"expected={expected[split]['sha256']}"
            )

    print("Frozen split verification: PASS")
    for split in ["train", "dev", "test"]:
        print(
            f"  {split}: n={len(data[split])} "
            f"sha256={sha256_file(paths[split])}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(
        torch.cuda.get_device_properties(0).total_memory / 2**30,
        2,
    ))
    print("Seed:", seed)

    paths, data = load_splits()
    verify_splits(paths, data)

    print("Languages:", sorted(data["train"]["language"].unique().tolist()))
    print("Split sizes:", {key: len(value) for key, value in data.items()})

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Qwen in 4-bit NF4...")
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

    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = DetoxDataset(data["train"], tokenizer)
    dev_dataset = DetoxDataset(data["dev"], tokenizer)
    collator = CausalCollator(tokenizer)

    output_dir = OUTPUT_ROOT / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_id": MODEL_ID,
        "seed": seed,
        "data_root": str(DATA_ROOT),
        "train_examples": len(data["train"]),
        "dev_examples": len(data["dev"]),
        "test_examples": len(data["test"]),
        "train_sha256": sha256_file(paths["train"]),
        "dev_sha256": sha256_file(paths["dev"]),
        "test_sha256": sha256_file(paths["test"]),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    train_result = trainer.train()

    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    summary = {
        **manifest,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get(
            "train_samples_per_second"
        ),
        "log_history": trainer.state.log_history,
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Adapter saved:", output_dir / "adapter")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best dev loss:", trainer.state.best_metric)

    del trainer
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
