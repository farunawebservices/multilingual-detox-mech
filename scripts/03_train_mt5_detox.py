#!/usr/bin/env python3
"""
Stage 3: Fine-tune google/mt5-base on the frozen 9-language detox corpus.
Three seeds (42, 1337, 2024), bf16 precision, early stopping.
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
    output_dir = "results/training"
    max_input_length = 96
    max_target_length = 112
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    weight_decay = 0.01
    num_train_epochs = 20
    eval_steps = 100
    save_steps = 100
    save_total_limit = 3
    early_stopping_patience = 4
    bf16 = True
    task_prefix = "detoxify: "

def load_corpus(corpus_path):
    df = pd.read_csv(corpus_path, keep_default_na=False)
    lang_counts = df.groupby("language").size()
    print("Corpus language distribution:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")
    return df

def make_splits(df, seed):
    rng = np.random.default_rng(seed)
    train_list, dev_list, test_list = [], [], []
    
    for lang, g in df.groupby("language"):
        groups = g.groupby("toxic_input")
        group_ids = list(groups.groups.keys())
        shuffled = rng.permutation(group_ids)
        capped = shuffled[:175]
        
        train_groups = capped[:122]
        dev_groups = capped[122:122+27]
        test_groups = capped[122+27:122+27+26]
        
        train_list.append(g[g["toxic_input"].isin(train_groups)])
        dev_list.append(g[g["toxic_input"].isin(dev_groups)])
        test_list.append(g[g["toxic_input"].isin(test_groups)])
    
    return {
        "train": pd.concat(train_list, ignore_index=True),
        "dev": pd.concat(dev_list, ignore_index=True),
        "test": pd.concat(test_list, ignore_index=True),
    }

class DetoxDataset(TorchDataset):
    def __init__(self, df, tokenizer, config):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.config.task_prefix + row["toxic_input"]
        tgt = row["detox_output"]
        
        inp = self.tokenizer(src, max_length=self.config.max_input_length, truncation=True, padding=False)
        lbl = self.tokenizer(tgt, max_length=self.config.max_target_length, truncation=True, padding=False)
        
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"], "labels": lbl["input_ids"]}

def train_seed(config, seed):
    print(f"\n{'='*60}\nTRAINING SEED {seed}\n{'='*60}\n")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    df = load_corpus(config.corpus_path)
    splits = make_splits(df, seed)
    
    out_dir = Path(config.output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, split_df in splits.items():
        split_df.to_csv(out_dir / f"{name}.csv", index=False)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)
    
    train_ds = DetoxDataset(splits["train"], tokenizer, config)
    dev_ds = DetoxDataset(splits["dev"], tokenizer, config)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)
    
    num_train_steps = (len(train_ds) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)) + 1
    warmup_steps = int(num_train_steps * 0.1)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Steps per epoch: ~{num_train_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        bf16=config.bf16,
        seed=seed,
        save_total_limit=config.save_total_limit,
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    
    train_result = trainer.train()
    best_eval_loss = trainer.state.best_metric
    
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    
    summary = {
        "seed": seed,
        "model_id": config.model_id,
        "train_samples": len(splits["train"]),
        "dev_samples": len(splits["dev"]),
        "test_samples": len(splits["test"]),
        "best_eval_loss": float(best_eval_loss) if best_eval_loss is not None else None,
        "train_loss": float(train_result.training_loss),
        "global_step": train_result.global_step,
        "epochs_trained": float(trainer.state.epoch),
    }
    
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print(f"\n✓ Seed {seed} complete")
    print(f"  Best eval loss: {summary['best_eval_loss']:.4f}")
    print(f"  Epochs: {summary['epochs_trained']:.2f}\n")
    
    return summary

def main():
    config = Config()
    load_corpus(config.corpus_path)
    
    summaries = []
    for seed in [42, 1337, 2024]:
        summaries.append(train_seed(config, seed))
    
    agg = {
        "model_id": config.model_id,
        "seeds": [42, 1337, 2024],
        "per_seed_summaries": summaries,
        "avg_train_loss": float(np.mean([s["train_loss"] for s in summaries])),
        "avg_best_eval_loss": float(np.mean([s["best_eval_loss"] for s in summaries if s["best_eval_loss"] is not None])),
    }
    
    agg_path = Path(config.output_dir) / "aggregate_summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*60}\nALL SEEDS COMPLETE\n{'='*60}")
    print(f"Average training loss: {agg['avg_train_loss']:.4f}")
    print(f"Average best eval loss: {agg['avg_best_eval_loss']:.4f}\n")

if __name__ == "__main__":
    main()
