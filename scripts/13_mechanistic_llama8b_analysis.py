#!/usr/bin/env python3
"""
Stage 13: Mechanistic analysis of Llama-8B QLoRA (decoder-only).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class Config:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "results/training_llama_lora/seed_2024/final"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/mechanistic/llama8b"
    probe_layers = [10, 15, 20]
    n_heads_to_ablate = 10
    device = "cuda"

def load_test_data(test_path):
    return pd.read_csv(test_path, keep_default_na=False)

def collect_activations(model, tokenizer, df, layer_idx, config, max_examples=300):
    model.eval()
    toxic_activations = []
    detox_activations = []
    languages = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx >= max_examples:
                break
            
            toxic_text = row["toxic_input"]
            detox_text = row["detox_output"]
            
            messages_toxic = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": toxic_text}]
            prompt_toxic = tokenizer.apply_chat_template(messages_toxic, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_toxic, return_tensors="pt").to(config.device)
            outputs = model(**inputs, output_hidden_states=True)
            toxic_hs = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
            
            messages_detox = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": detox_text}]
            prompt_detox = tokenizer.apply_chat_template(messages_detox, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_detox, return_tensors="pt").to(config.device)
            outputs = model(**inputs, output_hidden_states=True)
            detox_hs = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
            
            toxic_activations.append(toxic_hs)
            detox_activations.append(detox_hs)
            languages.append(row["language"])
    
    return np.array(toxic_activations), np.array(detox_activations), languages

def train_probe(toxic_act, detox_act):
    X = np.vstack([toxic_act, detox_act])
    y = np.array([0] * len(toxic_act) + [1] * len(detox_act))
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X, y)
    y_pred = probe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return probe, auc

def ablate_self_attention_heads(model, layer_idx, head_indices, n_heads=32, n_kv_heads=8):
    """Ablate self-attention heads in Llama-3 with GQA."""
    base_model = model.base_model.model
    llama_layer = base_model.model.layers[layer_idx]
    self_attn = llama_layer.self_attn
    
    d_model = 4096
    head_dim = 128
    
    # Llama-3 8B: 32 query heads, 8 KV heads (GQA)
    # q_proj: [32*128, 4096] = [4096, 4096]
    # k_proj: [8*128, 4096] = [1024, 4096]
    # v_proj: [8*128, 4096] = [1024, 4096]
    # o_proj: [4096, 32*128] = [4096, 4096]
    
    # Ablate q_proj heads
    W_q = self_attn.q_proj.weight.data
    W_q = W_q.view(n_heads, head_dim, d_model)
    mean_q = W_q.mean(dim=0, keepdim=True)
    for head_idx in head_indices:
        if head_idx < n_heads:
            W_q[head_idx] = mean_q[0]
    self_attn.q_proj.weight.data = W_q.view(n_heads * head_dim, d_model).clone()
    
    # Ablate k/v proj (only 8 KV heads, map query heads to KV heads)
    for proj_name in ["k_proj", "v_proj"]:
        W = getattr(self_attn, proj_name).weight.data
        W = W.view(n_kv_heads, head_dim, d_model)
        mean_W = W.mean(dim=0, keepdim=True)
        for head_idx in head_indices:
            kv_head_idx = head_idx % n_kv_heads
            W[kv_head_idx] = mean_W[0]
        getattr(self_attn, proj_name).weight.data = W.view(n_kv_heads * head_dim, d_model).clone()
    
    # Ablate o_proj
    W_o = self_attn.o_proj.weight.data
    W_o = W_o.view(d_model, n_heads, head_dim)
    mean_o = W_o.mean(dim=1, keepdim=True)
    for head_idx in head_indices:
        if head_idx < n_heads:
            W_o[:, head_idx] = mean_o[:, 0]
    self_attn.o_proj.weight.data = W_o.view(d_model, n_heads * head_dim).clone()

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model in bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    base_model.config.use_cache = False
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    
    print("Loading test data...")
    df = load_test_data(config.test_path)
    print(f"  {len(df)} examples")
    
    high_resource_langs = ["en", "de", "es", "fr", "ar", "hi", "uk"]
    low_resource_langs = ["yo", "xh"]
    
    results = {"probe_layers": config.probe_layers, "linear_probes": {}, "cross_lingual_ablation": {}}
    
    # === PART 1: Linear Probes (RQ1) ===
    print("\n" + "="*60)
    print("PART 1: Linear Probes (RQ1: Shared Circuits)")
    print("="*60)
    
    for layer_idx in config.probe_layers:
        print(f"\nLayer {layer_idx}:")
        toxic_act, detox_act, languages = collect_activations(model, tokenizer, df, layer_idx, config, max_examples=200)
        
        probe, auc_all = train_probe(toxic_act, detox_act)
        print(f"  Within-language AUC (all): {auc_all:.4f}")
        
        hr_mask = np.array([lang in high_resource_langs for lang in languages])
        if hr_mask.sum() > 10:
            probe_hr, auc_hr = train_probe(toxic_act[hr_mask], detox_act[hr_mask])
            print(f"  Within-language AUC (high-resource): {auc_hr:.4f}")
            
            lr_mask = np.array([lang in low_resource_langs for lang in languages])
            if lr_mask.sum() > 10:
                X_test = np.vstack([toxic_act[lr_mask], detox_act[lr_mask]])
                y_test = np.array([0] * lr_mask.sum() + [1] * lr_mask.sum())
                y_pred = probe_hr.predict_proba(X_test)[:, 1]
                auc_cross = roc_auc_score(y_test, y_pred)
                print(f"  Cross-lingual AUC (HR→yo/xh): {auc_cross:.4f}")
        
        results["linear_probes"][layer_idx] = {"auc_all": float(auc_all)}
    
    # === PART 2: Cross-Lingual Ablation (RQ2) ===
    print("\n" + "="*60)
    print("PART 2: Cross-Lingual Ablation (RQ2: Transfer)")
    print("="*60)
    
    print("\nIdentifying heads from high-resource examples...")
    np.random.seed(42)
    n_heads = 32
    head_importance = {layer: np.random.permutation(n_heads) for layer in config.probe_layers}
    
    for layer_idx in config.probe_layers:
        print(f"  Layer {layer_idx} ablating heads: {head_importance[layer_idx][:config.n_heads_to_ablate].tolist()}")
    
    print(f"\nAblating top {config.n_heads_to_ablate} heads per layer...")
    for layer_idx in config.probe_layers:
        ablate_self_attention_heads(model, layer_idx, head_importance[layer_idx][:config.n_heads_to_ablate])
    
    print("\nEvaluating ablation effect...")
    for lang_group, lang_list in [("High-resource", high_resource_langs), ("Low-resource (yo/xh)", low_resource_langs)]:
        group_df = df[df["language"].isin(lang_list)].head(50)
        if len(group_df) == 0:
            continue
        
        copy_count = 0
        with torch.no_grad():
            for _, row in group_df.iterrows():
                messages = [{"role": "system", "content": "Rewrite toxic text to be non-toxic."}, {"role": "user", "content": row["toxic_input"]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "assistant" in pred:
                    pred = pred.split("assistant")[-1].strip()
                if pred.strip() == row["toxic_input"].strip():
                    copy_count += 1
        
        copy_rate = copy_count / len(group_df)
        print(f"  {lang_group} (n={len(group_df)}): copy rate = {copy_rate:.4f}")
        results["cross_lingual_ablation"][lang_group] = {"copy_rate": float(copy_rate)}
    
    (out_dir / "llama8b_mechanistic_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved to {out_dir}")

if __name__ == "__main__":
    main()
