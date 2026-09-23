#!/usr/bin/env python3
"""
Stage 16: Cross-lingual transfer baseline.
Compare within-language vs cross-language steering.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Config:
    model_path = "results/training_v2_400plus/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/cross_lingual_transfer"
    encoder_layer = 6
    steering_strength = 1.0
    n_examples = 30
    device = "cuda"

def collect_activations_by_language(model, tokenizer, df, layer_idx, config):
    """Collect activations grouped by language."""
    model.eval()
    lang_activations = {'toxic': {}, 'detox': {}}
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            lang = row['language']
            
            if lang not in lang_activations['toxic']:
                lang_activations['toxic'][lang] = []
                lang_activations['detox'][lang] = []
            
            if len(lang_activations['toxic'][lang]) >= config.n_examples:
                continue
            
            toxic_text = row["toxic_input"]
            detox_text = row["detox_output"]
            
            inputs_toxic = tokenizer(toxic_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_toxic, output_hidden_states=True)
            toxic_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
            
            inputs_detox = tokenizer(detox_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
            encoder_outputs = model.encoder(**inputs_detox, output_hidden_states=True)
            detox_act = encoder_outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
            
            lang_activations['toxic'][lang].append(toxic_act)
            lang_activations['detox'][lang].append(detox_act)
    
    # Stack
    for lang in lang_activations['toxic']:
        lang_activations['toxic'][lang] = torch.stack(lang_activations['toxic'][lang])
        lang_activations['detox'][lang] = torch.stack(lang_activations['detox'][lang])
    
    return lang_activations

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
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    
    print("Loading test data...")
    df = pd.read_csv(config.test_path)
    print(f"  {len(df)} examples")
    
    print(f"\nCollecting activations by language (layer {config.encoder_layer})...")
    lang_activations = collect_activations_by_language(model, tokenizer, df, config.encoder_layer, config)
    
    languages = list(lang_activations['toxic'].keys())
    print(f"  Languages: {languages}")
    
    # Compute steering vectors for each language
    print("\nComputing per-language steering vectors...")
    steering_vectors = {}
    for lang in languages:
        sv = compute_steering_vector(lang_activations['toxic'][lang], lang_activations['detox'][lang])
        steering_vectors[lang] = sv
        print(f"  {lang}: norm={sv.norm().item():.4f}")
    
    # Test within-language vs cross-language steering
    print("\n" + "="*60)
    print("CROSS-LINGUAL TRANSFER TEST")
    print("="*60)
    
    high_resource_langs = ['en', 'de', 'es', 'fr', 'ar', 'hi', 'uk']
    low_resource_langs = ['yo', 'xh']
    
    # Use English steering vector for cross-lingual test
    en_steering = steering_vectors.get('en', None)
    if en_steering is None:
        print("  English not available, using first high-resource language")
        en_steering = steering_vectors[high_resource_langs[0]]
    
    results = []
    
    for target_lang in low_resource_langs:
        print(f"\n=== Target: {target_lang} ===")
        
        # Get target language examples
        target_df = df[df['language'] == target_lang].head(config.n_examples)
        
        if len(target_df) == 0:
            print(f"  No examples for {target_lang}")
            continue
        
        # Within-language steering (if available)
        if target_lang in steering_vectors:
            within_copy = 0
            with torch.no_grad():
                for idx, row in target_df.iterrows():
                    pred = generate_with_steering(model, tokenizer, row["toxic_input"], steering_vectors[target_lang], config.steering_strength, config.encoder_layer, config)
                    if pred.strip() == row["toxic_input"].strip():
                        within_copy += 1
            within_copy_rate = within_copy / len(target_df)
            print(f"  Within-language steering: copy rate = {within_copy_rate:.4f}")
        else:
            within_copy_rate = None
            print(f"  Within-language steering: N/A (no steering vector)")
        
        # Cross-language steering (from English)
        cross_copy = 0
        with torch.no_grad():
            for idx, row in target_df.iterrows():
                pred = generate_with_steering(model, tokenizer, row["toxic_input"], en_steering, config.steering_strength, config.encoder_layer, config)
                if pred.strip() == row["toxic_input"].strip():
                    cross_copy += 1
        cross_copy_rate = cross_copy / len(target_df)
        print(f"  Cross-language steering (en→{target_lang}): copy rate = {cross_copy_rate:.4f}")
        
        results.append({
            'target_lang': target_lang,
            'within_copy_rate': within_copy_rate,
            'cross_copy_rate': cross_copy_rate
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "cross_lingual_results.csv", index=False)
    
    torch.save({
        'steering_vectors': {k: v.cpu() for k, v in steering_vectors.items()},
        'encoder_layer': config.encoder_layer,
        'results': results
    }, out_dir / "cross_lingual_steering.pt")
    
    print(f"\n✓ Results saved to {out_dir}")

if __name__ == "__main__":
    main()
