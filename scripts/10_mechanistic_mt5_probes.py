#!/usr/bin/env python3
"""
Stage 10: Mechanistic analysis of mT5 (400+ pairs).
Part A: Linear probes on encoder/decoder activations.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"  # Best seed
    corpus_path = "data/frozen_v2_9lang_irregular_yo_xh/multilingual_detox_9lang_irregular_yo_xh.csv"
    output_dir = "results/mechanistic/mt5_probes"
    layers_to_probe = [4, 6, 8, 10]  # Encoder and decoder layers
    batch_size = 16
    device = "cuda"

def load_data(config):
    """Load test split and create toxic vs detox classification dataset."""
    test_path = Path(config.model_path).parent / "test.csv"
    df = pd.read_csv(test_path, keep_default_na=False)
    return df

def extract_activations(model, tokenizer, df, config):
    """Extract hidden states for toxic inputs and detox outputs."""
    model.eval()
    model.to(config.device)
    
    encoder_activations = {layer: [] for layer in config.layers_to_probe}
    decoder_activations = {layer: [] for layer in config.layers_to_probe}
    labels = []  # 0 = toxic input, 1 = detox output
    languages = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            # Process toxic input
            toxic_text = config.task_prefix + row["toxic_input"]
            toxic_enc = tokenizer(toxic_text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
            toxic_enc_out = model.encoder(**toxic_enc, output_hidden_states=True)
            
            for layer in config.layers_to_probe:
                # Mean pool over sequence length
                h = toxic_enc_out.hidden_states[layer].mean(dim=1).squeeze(0).cpu().numpy()
                encoder_activations[layer].append(h)
            
            # Process detox output
            detox_text = row["detox_output"]
            detox_enc = tokenizer(detox_text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
            detox_enc_out = model.encoder(**detox_enc, output_hidden_states=True)
            
            for layer in config.layers_to_probe:
                h = detox_enc_out.hidden_states[layer].mean(dim=1).squeeze(0).cpu().numpy()
                decoder_activations[layer].append(h)
            
            labels.append(0)  # Toxic input
            labels.append(1)  # Detox output
            languages.append(row["language"])
            languages.append(row["language"])
    
    # Combine encoder (toxic) and decoder (detox) activations
    all_activations = {layer: np.array(encoder_activations[layer] + decoder_activations[layer]) 
                       for layer in config.layers_to_probe}
    
    return all_activations, np.array(labels), languages

def train_probes(activations, labels, languages, config):
    """Train linear probes to classify toxic vs detox at each layer."""
    results = {}
    
    # Split by language for cross-lingual analysis
    lang_to_idx = {lang: i for i, lang in enumerate(sorted(set(languages)))}
    
    for layer in config.layers_to_probe:
        X = activations[layer]
        y = labels
        
        # Train/test split (80/20)
        n = len(y)
        perm = np.random.permutation(n)
        train_idx = perm[:int(0.8*n)]
        test_idx = perm[int(0.8*n):]
        
        # Train probe
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X[train_idx], y[train_idx])
        
        # Evaluate
        y_pred = probe.predict(X[test_idx])
        y_prob = probe.predict_proba(X[test_idx])[:, 1]
        
        acc = accuracy_score(y[test_idx], y_pred)
        try:
            auc = roc_auc_score(y[test_idx], y_prob)
        except:
            auc = 0.5
        
        results[layer] = {
            "accuracy": acc,
            "auc": auc,
            "weights": probe.coef_,
            "intercept": probe.intercept_,
        }
        
        print(f"Layer {layer}: Acc={acc:.4f}, AUC={auc:.4f}")
    
    return results

def main():
    config = Config()
    config.task_prefix = "detoxify: "
    
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
    
    print("Loading data...")
    df = load_data(config)
    print(f"  {len(df)} examples")
    
    print("Extracting activations...")
    activations, labels, languages = extract_activations(model, tokenizer, df, config)
    
    print("Training probes...")
    results = train_probes(activations, labels, languages, config)
    
    # Save results
    probe_results = {
        "model_path": str(config.model_path),
        "n_examples": len(df),
        "layers_probed": config.layers_to_probe,
        "per_layer_results": {str(k): v for k, v in results.items()},
    }
    
    # Remove large arrays before saving
    for layer in probe_results["per_layer_results"]:
        probe_results["per_layer_results"][layer]["weights"] = probe_results["per_layer_results"][layer]["weights"].tolist()
        probe_results["per_layer_results"][layer]["intercept"] = probe_results["per_layer_results"][layer]["intercept"].tolist()
    
    (out_dir / "probe_results.json").write_text(json.dumps(probe_results, indent=2))
    print(f"\nSaved results to {out_dir / 'probe_results.json'}")

if __name__ == "__main__":
    main()
