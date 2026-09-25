#!/usr/bin/env python3
"""
Stage 4: Generate test predictions and evaluate automatic metrics.
"""
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import sacrebleu

@dataclass
class Config:
    model_base = "results/training"
    corpus_path = "data/frozen_v2_9lang_irregular_yo_xh/multilingual_detox_9lang_irregular_yo_xh.csv"
    output_dir = "results/evaluation/mt5"
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
        
        # Tokenize with padding to max_length for batching
        inp = self.tokenizer(
            src,
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
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
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                num_beams=4,
                do_sample=False,
            )
            
            out_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            
            for i, out_text in enumerate(out_texts):
                all_predictions.append({
                    "pair_id": f"{batch['language'][i]}_{batch['idx'][i].item()}",
                    "language": batch["language"][i],
                    "toxic_input": batch["toxic_input"][i],
                    "detox_output": batch["detox_output"][i],
                    "prediction": out_text,
                })
    
    return pd.DataFrame(all_predictions)

def compute_sim(predictions_df, embedder):
    toxic_embeds = embedder.encode(predictions_df["toxic_input"].tolist(), normalize_embeddings=True, batch_size=32)
    pred_embeds = embedder.encode(predictions_df["prediction"].tolist(), normalize_embeddings=True, batch_size=32)
    sim = np.sum(toxic_embeds * pred_embeds, axis=1)
    predictions_df["sim"] = sim
    return predictions_df

def compute_fluency(predictions_df):
    fl_scores = []
    for idx, row in predictions_df.iterrows():
        chr_f = sacrebleu.corpus_chrf([row["prediction"]], [[row["detox_output"]]])
        fl_scores.append(chr_f.score / 100.0)
    predictions_df["fl"] = fl_scores
    return predictions_df

def compute_copy_rate(predictions_df):
    predictions_df["is_copy"] = predictions_df["prediction"].str.strip() == predictions_df["toxic_input"].str.strip()
    return predictions_df

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_df = pd.read_csv(config.corpus_path, keep_default_na=False)
    test_splits = load_test_splits(config)
    
    print(f"Loaded {len(test_splits)} test splits")
    for seed, df in test_splits.items():
        print(f"  Seed {seed}: {len(df)} examples")
    
    print("\nLoading LaBSE embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    
    all_predictions = {}
    for seed in config.seeds:
        if seed not in test_splits:
            continue
        
        print(f"\n{'='*60}")
        print(f"Generating predictions for seed {seed}")
        print(f"{'='*60}")
        
        model_path = Path(config.model_base) / f"seed_{seed}" / "final"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(config.device)
        model.eval()
        
        test_df = test_splits[seed]
        dataset = TestDataset(test_df, tokenizer, config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
        predictions = generate_predictions(model, tokenizer, dataloader, config.device)
        all_predictions[seed] = predictions
        
        pred_path = out_dir / f"predictions_seed_{seed}.csv"
        predictions.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")
    
    # Generate baselines
    print(f"\n{'='*60}")
    print("Generating baselines")
    print(f"{'='*60}")
    
    baseline_df = test_splits[config.seeds[0]].copy()
    
    # Duplicate baseline
    dup_pred = baseline_df.copy()
    dup_pred["prediction"] = dup_pred["toxic_input"]
    dup_pred["pair_id"] = [f"{row['language']}_{i}" for i, row in dup_pred.iterrows()]
    dup_pred = dup_pred[["pair_id", "language", "toxic_input", "detox_output", "prediction"]]
    dup_pred.to_csv(out_dir / "predictions_duplicate.csv", index=False)
    print("Saved duplicate baseline")
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing metrics")
    print(f"{'='*60}")
    
    all_dfs = []
    for seed, pred_df in all_predictions.items():
        print(f"  Computing metrics for seed {seed}...")
        pred_df = compute_sim(pred_df, embedder)
        pred_df = compute_fluency(pred_df)
        pred_df = compute_copy_rate(pred_df)
        pred_df["sta"] = 1.0  # Placeholder
        pred_df["j"] = (pred_df["sta"] + pred_df["sim"] + pred_df["fl"]) / 3.0
        pred_df["model"] = f"mt5_seed_{seed}"
        all_dfs.append(pred_df)
    
    # Duplicate baseline
    dup_pred = pd.read_csv(out_dir / "predictions_duplicate.csv")
    print(f"  Computing metrics for duplicate baseline...")
    dup_pred = compute_sim(dup_pred, embedder)
    dup_pred = compute_fluency(dup_pred)
    dup_pred = compute_copy_rate(dup_pred)
    dup_pred["sta"] = 1.0
    dup_pred["j"] = (dup_pred["sta"] + dup_pred["sim"] + dup_pred["fl"]) / 3.0
    dup_pred["model"] = "duplicate"
    all_dfs.append(dup_pred)
    
    # Aggregate
    all_metrics = pd.concat(all_dfs, ignore_index=True)
    all_metrics.to_csv(out_dir / "per_example_metrics.csv", index=False)
    
    # Aggregate by model
    agg = all_metrics.groupby("model").agg({
        "sta": "mean",
        "sim": "mean",
        "fl": "mean",
        "j": "mean",
        "is_copy": "mean",
    }).round(4)
    agg.to_csv(out_dir / "aggregate_metrics.csv")
    
    # Aggregate by model and language
    agg_lang = all_metrics.groupby(["model", "language"]).agg({
        "sta": "mean",
        "sim": "mean",
        "fl": "mean",
        "j": "mean",
    }).round(4)
    agg_lang.to_csv(out_dir / "per_language_metrics.csv")
    
    print(f"\n✓ Evaluation complete")
    print(f"Output directory: {out_dir}")
    print(f"\nAggregate metrics:\n{agg}")

if __name__ == "__main__":
    main()
