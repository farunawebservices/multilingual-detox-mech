#!/usr/bin/env python3
"""
Stage 21: Interpret SAE features for toxic→detox transformation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    sae_path = "results/mechanistic/sae/sae_checkpoint.pt"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/sae_interpretation"
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
            
            toxic_text = row["toxic_input"]
            detox_text = row["detox_output"]
            
            inputs_toxic = tokenizer(toxic_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_toxic, output_hidden_states=True)
            toxic_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
            
            inputs_detox = tokenizer(detox_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_detox, output_hidden_states=True)
            detox_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
            
            examples.append({
                'pair_id': idx,
                'language': row['language'],
                'toxic_input': toxic_text,
                'detox_output': detox_text,
                'toxic_act': toxic_act,
                'detox_act': detox_act
            })
    
    return examples

def interpret_features(encoder, examples, top_features, config):
    """Interpret SAE encoder features."""
    encoder.eval()
    
    print("="*60)
    print("SAE FEATURE INTERPRETATION")
    print("="*60)
    
    all_toxic = torch.cat([ex['toxic_act'] for ex in examples], dim=0)
    all_detox = torch.cat([ex['detox_act'] for ex in examples], dim=0)
    all_act = torch.cat([all_toxic, all_detox], dim=0)
    
    all_act = (all_act - all_act.mean(dim=0)) / all_act.std(dim=0)
    
    with torch.no_grad():
        hidden = F.relu(encoder(all_act.to(config.device)))
        
        print(f"\nAnalyzing top {len(top_features)} features...")
        
        interpretations = []
        for feat_idx in top_features[:10]:
            feat_idx = feat_idx.item()
            
            feat_acts = hidden[:, feat_idx].cpu()
            top_5_idx = feat_acts.argsort(descending=True)[:5]
            
            toxic_acts = feat_acts[:len(examples)]
            detox_acts = feat_acts[len(examples):]
            
            toxic_mean = toxic_acts.mean().item()
            detox_mean = detox_acts.mean().item()
            diff = detox_mean - toxic_mean
            
            print(f"\n{'='*60}")
            print(f"Feature {feat_idx}:")
            print(f"  Toxic mean: {toxic_mean:.3f}")
            print(f"  Detox mean: {detox_mean:.3f}")
            print(f"  Difference: {diff:+.3f}")
            
            print(f"\n  Top 5 examples:")
            for i, ex_idx in enumerate(top_5_idx):
                actual_idx = ex_idx.item() if ex_idx < len(examples) else ex_idx.item() - len(examples)
                ex = examples[actual_idx]
                act_val = feat_acts[ex_idx].item()
                print(f"    {i+1}. [{ex['language']}] Act={act_val:.3f}")
                print(f"       Toxic: {ex['toxic_input'][:60]}...")
            
            if diff > 0.5:
                interp = "DETOX-SELECTIVE"
            elif diff < -0.5:
                interp = "TOXIC-SELECTIVE"
            else:
                interp = "MIXED"
            
            print(f"\n  Interpretation: {interp}")
            
            interpretations.append({
                'feature_idx': feat_idx,
                'toxic_mean': toxic_mean,
                'detox_mean': detox_mean,
                'difference': diff,
                'interpretation': interp
            })
    
    return interpretations

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    print("Loading SAE...")
    sae_ckpt = torch.load(config.sae_path, map_location=config.device)
    
    # Load encoder directly (not as full SAE module)
    encoder = nn.Linear(sae_ckpt['d_model'], sae_ckpt['d_hidden']).to(config.device)
    encoder.load_state_dict({
        'weight': sae_ckpt['sae_state_dict']['encoder.weight'],
        'bias': sae_ckpt['sae_state_dict']['encoder.bias']
    })
    
    print("Loading test data...")
    df = pd.read_csv(config.test_path)
    
    print(f"\nCollecting examples from layer {config.encoder_layer}...")
    examples = collect_examples(model, tokenizer, df, config.encoder_layer, config)
    print(f"  {len(examples)} examples")
    
    top_features = torch.tensor(sae_ckpt['top_features'])
    print(f"\nTop features: {top_features[:10].tolist()}")
    
    interpretations = interpret_features(encoder, examples, top_features, config)
    
    interp_df = pd.DataFrame(interpretations)
    interp_df.to_csv(out_dir / "feature_interpretations.csv", index=False)
    
    print(f"\n✓ Saved to {out_dir}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    detox_sel = sum(1 for i in interpretations if "DETOX" in i['interpretation'])
    toxic_sel = sum(1 for i in interpretations if "TOXIC" in i['interpretation'])
    mixed = sum(1 for i in interpretations if "MIXED" in i['interpretation'])
    print(f"  DETOX-SELECTIVE: {detox_sel}")
    print(f"  TOXIC-SELECTIVE: {toxic_sel}")
    print(f"  MIXED: {mixed}")

if __name__ == "__main__":
    main()
