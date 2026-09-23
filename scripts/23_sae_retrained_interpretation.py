#!/usr/bin/env python3
"""
Stage 23: Interpret retrained SAE features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    sae_path = "results/mechanistic/sae_retrained/sae_checkpoint.pt"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/sae_retrained_interpretation"
    encoder_layer = 6
    n_examples = 50
    device = "cuda"

def collect_examples(model, tokenizer, df, layer_idx, config):
    model.eval()
    examples = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= config.n_examples:
                break
            
            inputs_toxic = tokenizer(row["toxic_input"], return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_toxic, output_hidden_states=True)
            toxic_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
            
            inputs_detox = tokenizer(row["detox_output"], return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_detox, output_hidden_states=True)
            detox_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
            
            examples.append({
                'pair_id': idx,
                'language': row['language'],
                'toxic_input': row["toxic_input"],
                'detox_output': row["detox_output"],
                'toxic_act': toxic_act,
                'detox_act': detox_act
            })
    
    return examples

def interpret_features(encoder, examples, top_detox, top_toxic, config):
    encoder.eval()
    
    print("="*60)
    print("RETRAINED SAE FEATURE INTERPRETATION")
    print("="*60)
    
    all_toxic = torch.cat([ex['toxic_act'] for ex in examples], dim=0)
    all_detox = torch.cat([ex['detox_act'] for ex in examples], dim=0)
    all_act = torch.cat([all_toxic, all_detox], dim=0)
    all_act = (all_act - all_act.mean(dim=0)) / all_act.std(dim=0)
    
    with torch.no_grad():
        hidden = F.relu(encoder(all_act.to(config.device)))
        
        interpretations = []
        
        # Interpret top 5 detox-selective
        print("\n=== TOP 5 DETOX-SELECTIVE FEATURES ===")
        for feat_idx in top_detox[:5]:
            feat_idx = feat_idx.item()
            feat_acts = hidden[:, feat_idx].cpu()
            
            toxic_acts = feat_acts[:len(examples)]
            detox_acts = feat_acts[len(examples):]
            
            t_mean = toxic_acts.mean().item()
            d_mean = detox_acts.mean().item()
            
            # Find top activating examples
            top_idx = feat_acts.argsort(descending=True)[:3]
            
            print(f"\nFeature {feat_idx}: toxic={t_mean:.3f}, detox={d_mean:.3f}")
            print("  Top examples:")
            for i, ex_idx in enumerate(top_idx):
                actual_idx = ex_idx.item() if ex_idx < len(examples) else ex_idx.item() - len(examples)
                ex = examples[actual_idx]
                print(f"    {i+1}. [{ex['language']}] {ex['toxic_input'][:50]}...")
            
            interpretations.append({
                'feature': feat_idx,
                'type': 'DETOX-SELECTIVE',
                'toxic_mean': t_mean,
                'detox_mean': d_mean,
                'diff': d_mean - t_mean
            })
        
        # Interpret top 5 toxic-selective
        print("\n=== TOP 5 TOXIC-SELECTIVE FEATURES ===")
        for feat_idx in top_toxic[:5]:
            feat_idx = feat_idx.item()
            feat_acts = hidden[:, feat_idx].cpu()
            
            toxic_acts = feat_acts[:len(examples)]
            detox_acts = feat_acts[len(examples):]
            
            t_mean = toxic_acts.mean().item()
            d_mean = detox_acts.mean().item()
            
            top_idx = feat_acts.argsort(descending=True)[:3]
            
            print(f"\nFeature {feat_idx}: toxic={t_mean:.3f}, detox={d_mean:.3f}")
            print("  Top examples:")
            for i, ex_idx in enumerate(top_idx):
                actual_idx = ex_idx.item() if ex_idx < len(examples) else ex_idx.item() - len(examples)
                ex = examples[actual_idx]
                print(f"    {i+1}. [{ex['language']}] {ex['toxic_input'][:50]}...")
            
            interpretations.append({
                'feature': feat_idx,
                'type': 'TOXIC-SELECTIVE',
                'toxic_mean': t_mean,
                'detox_mean': d_mean,
                'diff': t_mean - d_mean
            })
    
    return interpretations

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and SAE...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    sae_ckpt = torch.load(config.sae_path, map_location=config.device)
    encoder = nn.Linear(sae_ckpt['d_model'], sae_ckpt['d_hidden']).to(config.device)
    encoder.load_state_dict({
        'weight': sae_ckpt['sae_state_dict']['encoder.weight'],
        'bias': sae_ckpt['sae_state_dict']['encoder.bias']
    })
    
    print("Loading examples...")
    df = pd.read_csv(config.test_path)
    examples = collect_examples(model, tokenizer, df, config.encoder_layer, config)
    
    top_detox = torch.tensor(sae_ckpt['top_detox_features'])
    top_toxic = torch.tensor(sae_ckpt['top_toxic_features'])
    
    interpretations = interpret_features(encoder, examples, top_detox, top_toxic, config)
    
    interp_df = pd.DataFrame(interpretations)
    interp_df.to_csv(out_dir / "retrained_feature_interpretations.csv", index=False)
    
    print(f"\n✓ Saved to {out_dir}")

if __name__ == "__main__":
    main()
