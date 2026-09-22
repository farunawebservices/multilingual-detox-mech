#!/usr/bin/env python3
"""
Stage 10 v2: Linear probes on mT5 decoder activations.
Probe decoder layers where detoxification transformation occurs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    output_dir = "results/mechanistic/mt5_probes_v2"
    decoder_layers = [0, 2, 4, 6, 8, 10, 11]  # All decoder layers
    device = "cuda"

def extract_decoder_activations(model, tokenizer, df, config):
    """Extract decoder hidden states for toxic inputs and detox outputs."""
    model.eval()
    model.to(config.device)
    
    decoder_acts_toxic = {layer: [] for layer in config.decoder_layers}
    decoder_acts_detox = {layer: [] for layer in config.decoder_layers}
    languages = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            # Encode toxic input
            toxic_text = "detoxify: " + row["toxic_input"]
            toxic_enc = tokenizer(toxic_text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
            
            # Generate with decoder, capturing hidden states
            # We'll use teacher forcing with toxic input as target (to see what decoder sees for toxic)
            toxic_dec_out = model(
                **toxic_enc,
                labels=toxic_enc["input_ids"],  # Teacher forcing with same input
                output_hidden_states=True,
                return_dict=True,
            )
            
            for layer in config.decoder_layers:
                # Last token, last decoder layer
                h = toxic_dec_out.decoder_hidden_states[layer][0, -1, :].cpu().numpy()
                decoder_acts_toxic[layer].append(h)
            
            # Encode detox output
            detox_text = row["detox_output"]
            detox_enc = tokenizer(toxic_text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
            detox_tgt = tokenizer(detox_text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
            
            detox_dec_out = model(
                **detox_enc,
                labels=detox_tgt["input_ids"],
                output_hidden_states=True,
                return_dict=True,
            )
            
            for layer in config.decoder_layers:
                h = detox_dec_out.decoder_hidden_states[layer][0, -1, :].cpu().numpy()
                decoder_acts_detox[layer].append(h)
            
            languages.append(row["language"])
    
    # Combine: toxic=0, detox=1
    all_acts = {layer: np.array(decoder_acts_toxic[layer] + decoder_acts_detox[layer]) 
                for layer in config.decoder_layers}
    labels = np.array([0]*len(df) + [1]*len(df))
    
    return all_acts, labels, languages*2

def train_probes(activations, labels, config):
    """Train probes with proper standardization."""
    results = {}
    n = len(labels)
    perm = np.random.permutation(n)
    train_idx = perm[:int(0.8*n)]
    test_idx = perm[int(0.8*n):]
    
    for layer in config.decoder_layers:
        X = activations[layer]
        y = labels
        
        # Standardize using training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
        # Train probe
        probe = LogisticRegression(max_iter=5000, random_state=42)
        probe.fit(X_train_scaled, y_train:=y[train_idx])
        
        # Evaluate
        y_pred = probe.predict(X_test_scaled)
        y_prob = probe.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y[test_idx], y_pred)
        try:
            auc = roc_auc_score(y[test_idx], y_prob)
        except:
            auc = 0.5
        
        results[layer] = {"accuracy": acc, "auc": auc}
        print(f"Decoder Layer {layer}: Acc={acc:.4f}, AUC={auc:.4f}")
    
    return results

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
    
    print("Loading data...")
    test_path = Path(config.model_path).parent / "test.csv"
    df = pd.read_csv(test_path, keep_default_na=False)
    print(f"  {len(df)} examples")
    
    print("Extracting decoder activations...")
    activations, labels, languages = extract_decoder_activations(model, tokenizer, df, config)
    
    print("Training probes...")
    results = train_probes(activations, labels, config)
    
    # Save
    probe_results = {
        "model_path": str(config.model_path),
        "n_examples": len(df),
        "layers_probed": config.decoder_layers,
        "per_layer_results": results,
    }
    (out_dir / "probe_results.json").write_text(json.dumps(probe_results, indent=2))
    print(f"\nSaved to {out_dir / 'probe_results.json'}")

if __name__ == "__main__":
    main()
