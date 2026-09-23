#!/usr/bin/env python3
"""
Stage 14: Train Sparse Autoencoder (SAE) on mT5 encoder activations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/sae"
    encoder_layer = 6
    d_model = 768
    sae_hidden = 3072
    sparsity_coef = 1e-3
    batch_size = 32
    n_examples = 200
    device = "cuda"

def collect_activations(model, tokenizer, df, layer_idx, config, max_examples=200):
    model.eval()
    toxic_activations = []
    detox_activations = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= max_examples:
                break
            
            toxic_text = row["toxic_input"]
            detox_text = row["detox_output"]
            
            inputs_toxic = tokenizer(toxic_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_toxic, output_hidden_states=True)
            toxic_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
            
            inputs_detox = tokenizer(detox_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_detox, output_hidden_states=True)
            detox_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
            
            toxic_activations.append(toxic_act)
            detox_activations.append(detox_act)
    
    return torch.stack(toxic_activations), torch.stack(detox_activations)

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_hidden, sparsity_coef=1e-3):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)
        self.sparsity_coef = sparsity_coef
        
    def forward(self, x):
        hidden = self.encoder(x)
        sparse = F.relu(hidden)
        recon = self.decoder(sparse)
        l1_loss = sparse.abs().mean()
        return recon, l1_loss

def train_sae(toxic_act, detox_act, config):
    X = torch.cat([toxic_act, detox_act], dim=0)
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    sae = SparseAutoencoder(config.d_model, config.sae_hidden, config.sparsity_coef).to(config.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    
    print(f"Training SAE: d_model={config.d_model}, d_hidden={config.sae_hidden}")
    print(f"Dataset size: {len(X)} examples")
    
    for epoch in range(10):
        total_loss, total_l1, total_recon = 0, 0, 0
        for batch in loader:
            x = batch[0].to(config.device)
            recon, l1_loss = sae(x)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + config.sparsity_coef * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_recon += recon_loss.item()
        n_batches = len(loader)
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, recon={total_recon/n_batches:.4f}, l1={total_l1/n_batches:.4f}")
    
    return sae

def analyze_features(sae, toxic_act, detox_act, config):
    sae.eval()
    with torch.no_grad():
        toxic_hidden = F.relu(sae.encoder(toxic_act.to(config.device)))
        detox_hidden = F.relu(sae.encoder(detox_act.to(config.device)))
        toxic_mean = toxic_hidden.mean(dim=0).cpu()
        detox_mean = detox_hidden.mean(dim=0).cpu()
        diff = detox_mean - toxic_mean
        top_features = diff.argsort(descending=True)[:20]
        print("\nTop 20 features (detox > toxic):")
        for i, feat_idx in enumerate(top_features):
            print(f"  Feature {feat_idx.item()}: detox={detox_mean[feat_idx].item():.3f}, toxic={toxic_mean[feat_idx].item():.3f}, diff={diff[feat_idx].item():.3f}")
    return top_features

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    print("Loading test data...")
    df = pd.read_csv(config.test_path)
    print(f"  {len(df)} examples")
    
    print(f"\nCollecting activations from encoder layer {config.encoder_layer}...")
    toxic_act, detox_act = collect_activations(model, tokenizer, df, config.encoder_layer, config, max_examples=config.n_examples)
    print(f"  Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")
    
    print("\nTraining SAE...")
    sae = train_sae(toxic_act, detox_act, config)
    
    print("\nAnalyzing SAE features...")
    top_features = analyze_features(sae, toxic_act, detox_act, config)
    
    torch.save({
        'sae_state_dict': sae.state_dict(),
        'd_model': config.d_model,
        'd_hidden': config.sae_hidden,
        'sparsity_coef': config.sparsity_coef,
        'encoder_layer': config.encoder_layer,
        'top_features': top_features.tolist()
    }, out_dir / "sae_checkpoint.pt")
    
    print(f"\n✓ SAE saved to {out_dir}")

if __name__ == "__main__":
    main()
