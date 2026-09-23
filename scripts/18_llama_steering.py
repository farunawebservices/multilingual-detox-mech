#!/usr/bin/env python3
"""
Stage 18: Activation steering for Llama-3-8B QLoRA.
"""
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Config:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "results/training_llama_lora/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/llama_steering"
    decoder_layer = 15  # Best probe layer
    steering_strengths = [0.5, 1.0, 2.0, 3.0, 5.0]
    n_examples = 30
    device = "cuda"

def collect_activations(model, tokenizer, df, layer_idx, config):
    model.eval()
    toxic_act, detox_act = [], []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= config.n_examples:
                break
            
            messages_toxic = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": row["toxic_input"]}]
            prompt_toxic = tokenizer.apply_chat_template(messages_toxic, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_toxic, return_tensors="pt").to(config.device)
            outputs = model(**inputs, output_hidden_states=True)
            toxic_act.append(outputs.hidden_states[layer_idx][0, -1, :].cpu().float())
            
            messages_detox = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": row["detox_output"]}]
            prompt_detox = tokenizer.apply_chat_template(messages_detox, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_detox, return_tensors="pt").to(config.device)
            outputs = model(**inputs, output_hidden_states=True)
            detox_act.append(outputs.hidden_states[layer_idx][0, -1, :].cpu().float())
    
    return torch.stack(toxic_act), torch.stack(detox_act)

def compute_steering_vector(toxic_act, detox_act):
    steering_vec = detox_act.mean(dim=0) - toxic_act.mean(dim=0)
    return steering_vec / steering_vec.norm()

def generate_with_steering(model, tokenizer, toxic_text, steering_vec, strength, layer_idx, config):
    model.eval()
    
    def steering_hook(module, input, output):
        output_list = list(output)
        output_list[0][:, -1, :] += strength * steering_vec.to(output[0].device)
        return tuple(output_list)
    
    # Access decoder layers through PeftModel
    base_model = model.base_model.model
    decoder_layer = base_model.model.layers[layer_idx]
    hook = decoder_layer.register_forward_hook(steering_hook)
    
    messages = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": toxic_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    hook.remove()
    return pred

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Llama-3-8B tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True)
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    
    print("Loading test data...")
    df = pd.read_csv(config.test_path)
    
    print(f"\nCollecting activations from decoder layer {config.decoder_layer}...")
    toxic_act, detox_act = collect_activations(model, tokenizer, df, config.decoder_layer, config)
    print(f"  Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")
    
    print("\n" + "="*60)
    print("STEERING STRENGTH SWEEP (Llama-3-8B)")
    print("="*60)
    steering_vec = compute_steering_vector(toxic_act, detox_act)
    
    all_results = []
    for strength in config.steering_strengths:
        print(f"\nStrength {strength}:")
        copy_count = 0
        
        with torch.no_grad():
            for idx, row in df.iterrows():
                if idx >= config.n_examples:
                    break
                pred = generate_with_steering(model, tokenizer, row["toxic_input"], steering_vec, strength, config.decoder_layer, config)
                if pred.strip() == row["toxic_input"].strip():
                    copy_count += 1
        
        copy_rate = copy_count / config.n_examples
        print(f"  Copy rate: {copy_rate:.4f} ({copy_count}/{config.n_examples})")
        all_results.append({'strength': strength, 'copy_rate': copy_rate})
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "steering_sweep_results.csv", index=False)
    torch.save({'steering_vector': steering_vec.cpu(), 'decoder_layer': config.decoder_layer}, out_dir / "steering_vector.pt")
    
    print(f"\n✓ Llama-3-8B mechanistic analysis complete!")

if __name__ == "__main__":
    main()
