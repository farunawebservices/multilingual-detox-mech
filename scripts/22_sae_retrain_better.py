#!/usr/bin/env python3
"""
Stage 22: Retrain SAE with better hyperparameters for feature selectivity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/sae_retrained"
    encoder_layer = 6
    d_model = 768
    sae_hidden = 3072  # 4x overcomplete
    sparsity_coef = 1e-2  # Higher sparsity (10x original)
    batch_size = 64
    n_examples = 200
    n_epochs = 20  # More epochs
    device = "cuda"

def collect_activations(model, tokenizer, df, layer_idx, config):
    model.eval()
    toxic_act, detox_act = [], []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= config.n_examples:
                break
            
            inputs_toxic = tokenizer(row["toxic_input"], return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_toxic, output_hidden_states=True)
            toxic_act.append(encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float())
            
            inputs_detox = tokenizer(row["detox_output"], return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_detox, output_hidden_states=True)
            detox_act.append(encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float())
    
    return torch.stack(toxic_act), torch.stack(detox_act)

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_hidden, sparsity_coef=1e-2):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)
        self.sparsity_coef = sparsity_coef
        # Initialize decoder to encoder transpose
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        
    def forward(self, x):
        hidden = self.encoder(x)
        sparse = F.relu(hidden)
        recon = self.decoder(sparse)
        l1_loss = sparse.abs().mean()
        return recon, l1_loss, sparse

def train_sae(toxic_act, detox_act, config):
    X = torch.cat([toxic_act, detox_act], dim=0)
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    
    loader = DataLoader(TensorDataset(X), batch_size=config.batch_size, shuffle=True)
    
    sae = SparseAutoencoder(config.d_model, config.sae_hidden, config.sparsity_coef).to(config.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    print(f"Training SAE: {config.d_model}→{config.sae_hidden}, sparsity={config.sparsity_coef}")
    print(f"Dataset: {len(X)} examples, {config.n_epochs} epochs")
    
    for epoch in range(config.n_epochs):
        total_loss, total_l1, total_recon = 0, 0, 0
        for batch in loader:
            x = batch[0].to(config.device)
            recon, l1_loss, sparse = sae(x)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + config.sparsity_coef * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tie decoder weights
            sae.decoder.weight.data = sae.encoder.weight.data.T.clone()
            
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_recon += recon_loss.item()
        
        n_batches = len(loader)
        avg_loss = total_loss / n_batches
        avg_l1 = total_l1 / n_batches
        avg_recon = total_recon / n_batches
        sparsity = (sparse.abs() > 1e-6).float().mean().item() * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, l1={avg_l1:.4f}, active={sparsity:.1f}%")
    
    return sae

def analyze_selectivity(sae, toxic_act, detox_act, config):
    """Analyze feature selectivity for toxic vs detox."""
    sae.eval()
    
    with torch.no_grad():
        toxic_hidden = F.relu(sae.encoder(toxic_act.to(config.device)))
        detox_hidden = F.relu(sae.encoder(detox_act.to(config.device)))
        
        toxic_mean = toxic_hidden.mean(dim=0).cpu()
        detox_mean = detox_hidden.mean(dim=0).cpu()
        diff = detox_mean - toxic_mean
        
        # Find most selective features
        top_detox = diff.argsort(descending=True)[:10]
        top_toxic = diff.argsort()[:10]
        
        print("\n" + "="*60)
        print("TOP DETOX-SELECTIVE FEATURES")
        print("="*60)
        for i, feat_idx in enumerate(top_detox):
            feat_idx = feat_idx.item()
            t_mean = toxic_mean[feat_idx].item()
            d_mean = detox_mean[feat_idx].item()
            d = d_mean - t_mean
            print(f"  {i+1}. Feature {feat_idx}: toxic={t_mean:.3f}, detox={d_mean:.3f}, diff=+{d:.3f}")
        
        print("\n" + "="*60)
        print("TOP TOXIC-SELECTIVE FEATURES")
        print("="*60)
        for i, feat_idx in enumerate(top_toxic):
            feat_idx = feat_idx.item()
            t_mean = toxic_mean[feat_idx].item()
            d_mean = detox_mean[feat_idx].item()
            d = t_mean - d_mean
            print(f"  {i+1}. Feature {feat_idx}: toxic={t_mean:.3f}, detox={d_mean:.3f}, diff=+{d:.3f}")
        
        # Count selective features
        n_detox_sel = (diff > 0.5).sum().item()
        n_toxic_sel = (diff < -0.5).sum().item()
        n_mixed = len(diff) - n_detox_sel - n_toxic_sel
        
        print(f"\n  DETOX-selective (diff > 0.5): {n_detox_sel}")
        print(f"  TOXIC-selective (diff < -0.5): {n_toxic_sel}")
        print(f"  MIXED: {n_mixed}")
        
        return top_detox, top_toxic

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    print("Collecting activations...")
    toxic_act, detox_act = collect_activations(model, tokenizer, df=pd.read_csv(config.test_path), layer_idx=config.encoder_layer, config=config)
    print(f"  Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")
    
    print("\nTraining SAE...")
    sae = train_sae(toxic_act, detox_act, config)
    
    print("\nAnalyzing selectivity...")
    top_detox, top_toxic = analyze_selectivity(sae, toxic_act, detox_act, config)
    
    # Save
    torch.save({
        'sae_state_dict': sae.state_dict(),
        'd_model': config.d_model,
        'd_hidden': config.sae_hidden,
        'sparsity_coef': config.sparsity_coef,
        'encoder_layer': config.encoder_layer,
        'top_detox_features': top_detox.tolist(),
        'top_toxic_features': top_toxic.tolist()
    }, out_dir / "sae_checkpoint.pt")
    
    print(f"\n✓ Saved to {out_dir}")

if __name__ == "__main__":
    main()
