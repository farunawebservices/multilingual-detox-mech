#!/usr/bin/env python3
"""
Stage 3 v2: Full-corpus mT5-base fine-tuning.

Uses every available paired row. Splits are grouped by toxic_input to avoid
input leakage, and are performed per language using 70/15/15 of unique inputs.
Periodic Trainer checkpoint writes are disabled to avoid saving large optimizer
states; the final model is written once per seed.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


class Config:
    model_id = "google/mt5-base"
    corpus_path = "data/frozen_v2_9lang_irregular_yo_xh/multilingual_detox_9lang_irregular_yo_xh.csv"
    output_dir = "results/training_v2_400plus"
    max_input_length = 96
    max_target_length = 112
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    weight_decay = 0.01
    num_train_epochs = 20
    eval_steps = 100
    logging_steps = 10
    early_stopping_patience = 4
    bf16 = True
    task_prefix = "detoxify: "


def load_corpus(path):
    df = pd.read_csv(path, keep_default_na=False)
    required = {"language", "toxic_input", "detox_output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    if df[["language", "toxic_input", "detox_output"]].isna().any().any():
        raise ValueError("Unexpected null in language/toxic_input/detox_output.")
    print("Corpus language distribution (rows):")
    print(df.groupby("language").size().to_string())
    print("\nUnique toxic inputs by language:")
    print(df.groupby("language")["toxic_input"].nunique().to_string())
    return df


def make_splits(df, seed):
    """
    Use all rows. Assign each unique toxic_input to one split only.
    Rows sharing an input (multi-reference examples) stay together.
    """
    rng = np.random.default_rng(seed)
    train_parts, dev_parts, test_parts = [], [], []
    manifest = []

    for lang, group in df.groupby("language", sort=True):
        unique_inputs = group["toxic_input"].drop_duplicates().to_numpy()
        shuffled = rng.permutation(unique_inputs)
        n_unique = len(shuffled)

        n_train = int(n_unique * 0.70)
        n_dev = int(n_unique * 0.15)
        n_test = n_unique - n_train - n_dev

        if min(n_train, n_dev, n_test) < 1:
            raise ValueError(f"{lang}: insufficient unique inputs ({n_unique}).")

        train_inputs = set(shuffled[:n_train])
        dev_inputs = set(shuffled[n_train:n_train + n_dev])
        test_inputs = set(shuffled[n_train + n_dev:])

        if train_inputs & dev_inputs or train_inputs & test_inputs or dev_inputs & test_inputs:
            raise AssertionError(f"{lang}: input group overlap across splits.")

        train_g = group[group["toxic_input"].isin(train_inputs)].copy()
        dev_g = group[group["toxic_input"].isin(dev_inputs)].copy()
        test_g = group[group["toxic_input"].isin(test_inputs)].copy()

        if len(train_g) + len(dev_g) + len(test_g) != len(group):
            raise AssertionError(f"{lang}: not every row was assigned exactly once.")

        train_parts.append(train_g)
        dev_parts.append(dev_g)
        test_parts.append(test_g)

        manifest.append({
            "language": lang,
            "rows_total": len(group),
            "unique_inputs_total": n_unique,
            "train_unique_inputs": len(train_inputs),
            "dev_unique_inputs": len(dev_inputs),
            "test_unique_inputs": len(test_inputs),
            "train_rows": len(train_g),
            "dev_rows": len(dev_g),
            "test_rows": len(test_g),
        })

    splits = {
        "train": pd.concat(train_parts, ignore_index=True),
        "dev": pd.concat(dev_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }

    if sum(len(x) for x in splits.values()) != len(df):
        raise AssertionError("Global split assignment did not preserve all corpus rows.")

    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        left_keys = set(zip(splits[left]["language"], splits[left]["toxic_input"]))
        right_keys = set(zip(splits[right]["language"], splits[right]["toxic_input"]))
        overlap = left_keys & right_keys
        if overlap:
            raise AssertionError(f"Leakage: {len(overlap)} language/input groups overlap between {left} and {right}.")

    manifest_df = pd.DataFrame(manifest)
    return splits, manifest_df


class DetoxDataset(TorchDataset):
    def __init__(self, frame, tokenizer, config):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        source = self.config.task_prefix + row["toxic_input"]

        encoded_source = self.tokenizer(
            source,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=False,
        )
        encoded_target = self.tokenizer(
            row["detox_output"],
            max_length=self.config.max_target_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": encoded_source["input_ids"],
            "attention_mask": encoded_source["attention_mask"],
            "labels": encoded_target["input_ids"],
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_seed(config, seed):
    print(f"\n{'=' * 72}\nTRAINING SEED {seed}\n{'=' * 72}")
    set_seed(seed)

    corpus = load_corpus(config.corpus_path)
    splits, split_manifest = make_splits(corpus, seed)

    out_dir = Path(config.output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    print("\nSplit rows:")
    print({name: len(frame) for name, frame in splits.items()})
    print("\nSplit manifest:")
    print(split_manifest.to_string(index=False))

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)

    train_dataset = DetoxDataset(splits["train"], tokenizer, config)
    dev_dataset = DetoxDataset(splits["dev"], tokenizer, config)

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    steps_per_epoch = int(np.ceil(
        len(train_dataset) /
        (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    ))
    warmup_steps = max(1, int(steps_per_epoch * 0.10))
    print(f"\nSteps/epoch: {steps_per_epoch}; warmup steps: {warmup_steps}")

    args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="no",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        bf16=config.bf16,
        fp16=False,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )

    result = trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    eval_rows = [item for item in trainer.state.log_history if "eval_loss" in item]
    best_eval_loss = min((float(item["eval_loss"]) for item in eval_rows), default=None)
    best_eval_step = None
    if best_eval_loss is not None:
        best_row = min(eval_rows, key=lambda item: float(item["eval_loss"]))
        best_eval_step = int(best_row.get("step", -1))

    finite_parameters = all(torch.isfinite(p.detach()).all().item() for p in model.parameters())
    if not finite_parameters:
        raise RuntimeError(f"Seed {seed}: non-finite final parameter detected.")

    summary = {
        "seed": seed,
        "model_id": config.model_id,
        "split_rule": "grouped_by_toxic_input_70_15_15_all_rows",
        "train_samples": len(splits["train"]),
        "dev_samples": len(splits["dev"]),
        "test_samples": len(splits["test"]),
        "best_observed_eval_loss": best_eval_loss,
        "best_observed_eval_step": best_eval_step,
        "train_loss": float(result.training_loss),
        "global_step": int(result.global_step),
        "epochs_trained": float(trainer.state.epoch),
        "finite_final_parameters": bool(finite_parameters),
        "checkpoint_policy": "final_only_no_optimizer_checkpoint",
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSeed summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    config = Config()
    load_corpus(config.corpus_path)

    summaries = [train_seed(config, seed) for seed in (42, 1337, 2024)]
    aggregate = {
        "model_id": config.model_id,
        "split_rule": "grouped_by_toxic_input_70_15_15_all_rows",
        "seeds": [42, 1337, 2024],
        "per_seed_summaries": summaries,
        "mean_train_loss": float(np.mean([x["train_loss"] for x in summaries])),
        "mean_best_observed_eval_loss": float(np.mean([
            x["best_observed_eval_loss"] for x in summaries
            if x["best_observed_eval_loss"] is not None
        ])),
    }

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nALL SEEDS COMPLETE")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
