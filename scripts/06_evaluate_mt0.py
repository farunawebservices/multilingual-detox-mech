#!/usr/bin/env python3
"""Stage 6: Generate and evaluate mT0-base on 400+ pairs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import sacrebleu

class Config:
    model_base = "results/training_mt0"
    output_dir = "results/evaluation/mt0"
    seeds = [42, 1337, 2024]
    max_input_length = 96
    max_target_length = 112
    task_prefix = "detoxify: "
    device = "cuda"

class TestDataset(TorchDataset):
    def __init__(self, df, tokenizer, config):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.config.task_prefix + row["toxic_input"]
        inp = self.tokenizer(src, max_length=self.config.max_input_length, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inp["input_ids"].squeeze(0),
            "attention_mask": inp["attention_mask"].squeeze(0),
            "language": row["language"],
            "toxic_input": row["toxic_input"],
            "detox_output": row["detox_output"],
            "idx": idx,
        }

def load_test_splits(config):
    test_splits = {}
    for seed in config.seeds:
        path = Path(config.model_base) / f"seed_{seed}" / "test.csv"
        if path.exists():
            test_splits[seed] = pd.read_csv(path, keep_default_na=False)
    return test_splits

def generate_predictions(model, tokenizer, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128, num_beams=4, do_sample=False)
            out_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for i, out_text in enumerate(out_texts):
                all_preds.append({
                    "pair_id": f"{batch['language'][i]}_{batch['idx'][i].item()}",
                    "language": batch["language"][i],
                    "toxic_input": batch["toxic_input"][i],
                    "detox_output": batch["detox_output"][i],
                    "prediction": out_text,
                })
    return pd.DataFrame(all_preds)

def compute_sim(df, embedder):
    toxic_emb = embedder.encode(df["toxic_input"].tolist(), normalize_embeddings=True, batch_size=32)
    pred_emb = embedder.encode(df["prediction"].tolist(), normalize_embeddings=True, batch_size=32)
    df["sim"] = np.sum(toxic_emb * pred_emb, axis=1)
    return df

def compute_fluency(df):
    fl = []
    for _, row in df.iterrows():
        chr_f = sacrebleu.corpus_chrf([row["prediction"]], [[row["detox_output"]]])
        fl.append(chr_f.score / 100.0)
    df["fl"] = fl
    return df

def compute_copy_rate(df):
    df["is_copy"] = df["prediction"].str.strip() == df["toxic_input"].str.strip()
    return df

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_splits = load_test_splits(config)
    print(f"Loaded {len(test_splits)} test splits")
    for seed, df in test_splits.items():
        print(f"  Seed {seed}: {len(df)} examples")

    print("\nLoading LaBSE embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")

    all_preds = {}
    for seed in config.seeds:
        if seed not in test_splits:
            continue
        print(f"\n{'='*60}\nGenerating predictions for seed {seed}\n{'='*60}")
        model_path = Path(config.model_base) / f"seed_{seed}" / "final"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(config.device)
        model.eval()
        dataset = TestDataset(test_splits[seed], tokenizer, config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        preds = generate_predictions(model, tokenizer, dataloader, config.device)
        all_preds[seed] = preds
        preds.to_csv(out_dir / f"predictions_seed_{seed}.csv", index=False)
        print(f"Saved: {out_dir / f'predictions_seed_{seed}.csv'}")

    # Compute metrics
    print(f"\n{'='*60}\nComputing metrics\n{'='*60}")
    all_dfs = []
    for seed, pred_df in all_preds.items():
        print(f"  Seed {seed}...")
        pred_df = compute_sim(pred_df, embedder)
        pred_df = compute_fluency(pred_df)
        pred_df = compute_copy_rate(pred_df)
        pred_df["sta"] = 1.0
        pred_df["j"] = (pred_df["sta"] + pred_df["sim"] + pred_df["fl"]) / 3.0
        pred_df["model"] = f"mt0_seed_{seed}"
        all_dfs.append(pred_df)

    all_metrics = pd.concat(all_dfs, ignore_index=True)
    all_metrics.to_csv(out_dir / "per_example_metrics.csv", index=False)

    agg = all_metrics.groupby("model").agg({"sta": "mean", "sim": "mean", "fl": "mean", "j": "mean", "is_copy": "mean"}).round(4)
    agg.to_csv(out_dir / "aggregate_metrics.csv")

    agg_lang = all_metrics.groupby(["model", "language"]).agg({"sta": "mean", "sim": "mean", "fl": "mean", "j": "mean"}).round(4)
    agg_lang.to_csv(out_dir / "per_language_metrics.csv")

    print(f"\n✓ Evaluation complete: {out_dir}")
    print(f"\nAggregate metrics:\n{agg}")

if __name__ == "__main__":
    main()
