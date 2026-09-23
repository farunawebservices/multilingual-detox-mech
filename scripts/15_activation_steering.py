#!/usr/bin/env python3
"""
Stage 15: Activation steering for mT5.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/steering"
    encoder_layer = 6
    steering_strengths = [0.5, 1.0, 2.0, 3.0, 5.0]
    n_examples = 50
    device = "cuda"

def collect_activations(model, tokenizer, df, layer_idx, config, max_examples=100):
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

def compute_steering_vector(toxic_act, detox_act):
    toxic_mean = toxic_act.mean(dim=0)
    detox_mean = detox_act.mean(dim=0)
    steering_vec = detox_mean - toxic_mean
    steering_vec = steering_vec / steering_vec.norm()
    return steering_vec

def generate_with_steering(model, tokenizer, toxic_text, steering_vec, strength, layer_idx, config):
    model.eval()
    
    def steering_hook(module, input, output):
        output_list = list(output)
        output_list[0][:, -1, :] += strength * steering_vec.to(output[0].device)
        return tuple(output_list)
    
    # mT5 encoder uses 'block' not 'layers'
    encoder_layer = model.encoder.block[layer_idx]
    hook = encoder_layer.register_forward_hook(steering_hook)
    
    inputs = tokenizer(toxic_text, return_tensors="pt").to(config.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    hook.remove()
    return pred

def evaluate_steering(model, tokenizer, df, steering_vec, strength, layer_idx, config):
    predictions = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= config.n_examples:
                break
            
            pred = generate_with_steering(model, tokenizer, row["toxic_input"], steering_vec, strength, layer_idx, config)
            predictions.append({
                'pair_id': idx,
                'toxic_input': row["toxic_input"],
                'detox_output': row["detox_output"],
                'prediction': pred,
                'steering_strength': strength
            })
            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx+1}/{config.n_examples}...")
    
    return predictions

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
    
    print("\nComputing steering vector...")
    steering_vec = compute_steering_vector(toxic_act, detox_act)
    print(f"  Steering vector norm: {steering_vec.norm().item():.4f}")
    
    print("\n" + "="*60)
    print("STEERING STRENGTH SWEEP")
    print("="*60)
    
    all_results = []
    for strength in config.steering_strengths:
        print(f"\nStrength {strength}:")
        preds = evaluate_steering(model, tokenizer, df, steering_vec, strength, config.encoder_layer, config)
        all_results.extend(preds)
        
        copy_count = sum(1 for p in preds if p['prediction'].strip() == p['toxic_input'].strip())
        copy_rate = copy_count / len(preds)
        print(f"  Copy rate: {copy_rate:.4f} ({copy_count}/{len(preds)})")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "steering_sweep_results.csv", index=False)
    
    torch.save({
        'steering_vector': steering_vec,
        'encoder_layer': config.encoder_layer,
        'strengths_tested': config.steering_strengths
    }, out_dir / "steering_vector.pt")
    
    print(f"\n✓ Results saved to {out_dir}")

if __name__ == "__main__":
    main()
