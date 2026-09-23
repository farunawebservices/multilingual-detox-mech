#!/usr/bin/env python3
"""
Stage 17: SAE training and activation steering for mT0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_mt0/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/mt0_sae_steering"
    encoder_layer = 6
    d_model = 768
    sae_hidden = 3072
    sparsity_coef = 1e-3
    batch_size = 32
    n_examples = 100
    steering_strengths = [0.5, 1.0, 2.0, 3.0, 5.0]
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
    loader = DataLoader(TensorDataset(X), batch_size=config.batch_size, shuffle=True)
    
    sae = SparseAutoencoder(config.d_model, config.sae_hidden, config.sparsity_coef).to(config.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    
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

def compute_steering_vector(toxic_act, detox_act):
    steering_vec = detox_act.mean(dim=0) - toxic_act.mean(dim=0)
    return steering_vec / steering_vec.norm()

def generate_with_steering(model, tokenizer, toxic_text, steering_vec, strength, layer_idx, config):
    model.eval()
    
    def steering_hook(module, input, output):
        output_list = list(output)
        output_list[0][:, -1, :] += strength * steering_vec.to(output[0].device)
        return tuple(output_list)
    
    encoder_layer = model.encoder.block[layer_idx]
    hook = encoder_layer.register_forward_hook(steering_hook)
    
    inputs = tokenizer(toxic_text, return_tensors="pt").to(config.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    hook.remove()
    return pred

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading mT0 tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    print("Loading test data...")
    df = pd.read_csv(config.test_path)
    
    print(f"\nCollecting activations from encoder layer {config.encoder_layer}...")
    toxic_act, detox_act = collect_activations(model, tokenizer, df, config.encoder_layer, config)
    print(f"  Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")
    
    # === SAE TRAINING ===
    print("\n" + "="*60)
    print("SAE TRAINING")
    print("="*60)
    sae = train_sae(toxic_act, detox_act, config)
    
    # Save SAE
    torch.save({'sae_state_dict': sae.state_dict(), 'd_model': config.d_model, 'd_hidden': config.sae_hidden}, out_dir / "sae_checkpoint.pt")
    print(f"\n✓ SAE saved to {out_dir}")
    
    # === STEERING STRENGTH SWEEP ===
    print("\n" + "="*60)
    print("STEERING STRENGTH SWEEP")
    print("="*60)
    steering_vec = compute_steering_vector(toxic_act, detox_act)
    
    all_results = []
    for strength in config.steering_strengths:
        print(f"\nStrength {strength}:")
        copy_count = 0
        
        with torch.no_grad():
            for idx, row in df.iterrows():
                if idx >= 30:
                    break
                pred = generate_with_steering(model, tokenizer, row["toxic_input"], steering_vec, strength, config.encoder_layer, config)
                if pred.strip() == row["toxic_input"].strip():
                    copy_count += 1
        
        copy_rate = copy_count / 30
        print(f"  Copy rate: {copy_rate:.4f} ({copy_count}/30)")
        all_results.append({'strength': strength, 'copy_rate': copy_rate})
    
    # Save steering results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "steering_sweep_results.csv", index=False)
    torch.save({'steering_vector': steering_vec.cpu(), 'encoder_layer': config.encoder_layer}, out_dir / "steering_vector.pt")
    
    print(f"\n✓ mT0 mechanistic analysis complete!")

if __name__ == "__main__":
    main()
