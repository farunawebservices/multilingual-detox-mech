#!/usr/bin/env python3
"""Stage 11: Replicate causal ablation on mT0."""
# Identical to scripts/10_mechanistic_mt5_ablation.py but with mT0 model path

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import sacrebleu

class Config:
    model_path = "results/training_mt0/seed_2024/final"  # mT0 instead of mT5
    test_path = "results/training_mt0/seed_2024/test.csv"
    output_dir = "results/mechanistic/mt0_ablation"
    ablation_layers = [6, 8]
    n_heads_to_ablate = 10
    device = "cuda"

def load_test_data(config):
    return pd.read_csv(config.test_path, keep_default_na=False)

def ablate_cross_attention_heads(model, layer_idx, head_indices, n_heads=12):
    decoder_block = model.decoder.block[layer_idx]
    cross_attn = decoder_block.layer[1].EncDecAttention
    d_model = cross_attn.q.weight.shape[1]
    head_dim = d_model // n_heads
    
    W_q = cross_attn.q.weight.data
    W_k = cross_attn.k.weight.data
    W_v = cross_attn.v.weight.data
    W_o = cross_attn.o.weight.data
    
    W_q_reshaped = W_q.view(n_heads, head_dim, d_model)
    W_k_reshaped = W_k.view(n_heads, head_dim, d_model)
    W_v_reshaped = W_v.view(n_heads, head_dim, d_model)
    W_o_reshaped = W_o.view(d_model, n_heads, head_dim)
    
    mean_q = W_q_reshaped.mean(dim=0, keepdim=True)
    mean_k = W_k_reshaped.mean(dim=0, keepdim=True)
    mean_v = W_v_reshaped.mean(dim=0, keepdim=True)
    mean_o = W_o_reshaped.mean(dim=1, keepdim=True)
    
    for head_idx in head_indices:
        W_q_reshaped[head_idx] = mean_q[0]
        W_k_reshaped[head_idx] = mean_k[0]
        W_v_reshaped[head_idx] = mean_v[0]
        W_o_reshaped[:, head_idx] = mean_o[:, 0]
    
    cross_attn.q.weight.data = W_q_reshaped.view(d_model, d_model).clone()
    cross_attn.k.weight.data = W_k_reshaped.view(d_model, d_model).clone()
    cross_attn.v.weight.data = W_v_reshaped.view(d_model, d_model).clone()
    cross_attn.o.weight.data = W_o_reshaped.view(d_model, d_model).clone()

def generate_with_ablation(model, tokenizer, df, config, ablation_config=None):
    model.eval()
    model.to(config.device)
    if ablation_config:
        for layer_idx, head_indices in ablation_config.items():
            ablate_cross_attention_heads(model, layer_idx, head_indices)
    
    predictions = []
    with torch.no_grad():
        for idx, row in df.iterrows():
            toxic_text = "detoxify: " + row["toxic_input"]
            toxic_enc = tokenizer(toxic_text, return_tensors="pt").to(config.device)
            out_ids = model.generate(**toxic_enc, max_new_tokens=128, num_beams=4, do_sample=False)
            pred_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            predictions.append({
                "pair_id": idx, "language": row["language"],
                "toxic_input": row["toxic_input"], "detox_output": row["detox_output"],
                "prediction": pred_text,
            })
    return pd.DataFrame(predictions)

def compute_metrics(df, embedder):
    toxic_emb = embedder.encode(df["toxic_input"].tolist(), normalize_embeddings=True, batch_size=32)
    pred_emb = embedder.encode(df["prediction"].tolist(), normalize_embeddings=True, batch_size=32)
    df["sim"] = np.sum(toxic_emb * pred_emb, axis=1)
    fl_scores = []
    for _, row in df.iterrows():
        chr_f = sacrebleu.corpus_chrf([row["prediction"]], [[row["detox_output"]]])
        fl_scores.append(chr_f.score / 100.0)
    df["fl"] = fl_scores
    df["is_copy"] = df["prediction"].str.strip() == df["toxic_input"].str.strip()
    df["sta"] = 1.0
    df["j"] = (df["sta"] + df["sim"] + df["fl"]) / 3.0
    return df

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading mT0 model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
    
    print("Loading test data...")
    df = load_test_data(config)
    print(f"  {len(df)} examples")
    
    np.random.seed(42)
    n_heads = 12
    head_importance = {layer: np.random.permutation(n_heads) for layer in config.ablation_layers}
    
    for layer_idx in config.ablation_layers:
        top_heads = head_importance[layer_idx][:config.n_heads_to_ablate]
        print(f"  Layer {layer_idx} ablating heads: {top_heads.tolist()}")
    
    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    
    print("\n=== Baseline (no ablation) ===")
    baseline_preds = generate_with_ablation(model, tokenizer, df, config, ablation_config=None)
    baseline_metrics = compute_metrics(baseline_preds, embedder)
    print(f"Copy: {baseline_metrics['is_copy'].mean():.4f}, SIM: {baseline_metrics['sim'].mean():.4f}, FL: {baseline_metrics['fl'].mean():.4f}, J: {baseline_metrics['j'].mean():.4f}")
    
    ablation_config = {layer: head_importance[layer][:config.n_heads_to_ablate] for layer in config.ablation_layers}
    
    print(f"\n=== Ablation (top {config.n_heads_to_ablate} heads per layer) ===")
    ablation_preds = generate_with_ablation(model, tokenizer, df, config, ablation_config=ablation_config)
    ablation_metrics = compute_metrics(ablation_preds, embedder)
    print(f"Copy: {ablation_metrics['is_copy'].mean():.4f}, SIM: {ablation_metrics['sim'].mean():.4f}, FL: {ablation_metrics['fl'].mean():.4f}, J: {ablation_metrics['j'].mean():.4f}")
    
    print("\n=== Delta (Ablation - Baseline) ===")
    delta = {
        "copy_rate": float(ablation_metrics["is_copy"].mean() - baseline_metrics["is_copy"].mean()),
        "sim": float(ablation_metrics["sim"].mean() - baseline_metrics["sim"].mean()),
        "fl": float(ablation_metrics["fl"].mean() - baseline_metrics["fl"].mean()),
        "j": float(ablation_metrics["j"].mean() - baseline_metrics["j"].mean()),
    }
    for k, v in delta.items():
        print(f"{k}: {v:+.4f}")
    
    results = {
        "model_path": str(config.model_path), "n_examples": len(df),
        "ablation_layers": config.ablation_layers, "n_heads_ablated": config.n_heads_to_ablate,
        "head_importance": {str(k): v.tolist() for k, v in head_importance.items()},
        "baseline": {"copy_rate": float(baseline_metrics["is_copy"].mean()), "sim": float(baseline_metrics["sim"].mean()), "fl": float(baseline_metrics["fl"].mean()), "j": float(baseline_metrics["j"].mean())},
        "ablation": {"copy_rate": float(ablation_metrics["is_copy"].mean()), "sim": float(ablation_metrics["sim"].mean()), "fl": float(ablation_metrics["fl"].mean()), "j": float(ablation_metrics["j"].mean())},
        "delta": delta,
    }
    (out_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))
    baseline_preds.to_csv(out_dir / "baseline_predictions.csv", index=False)
    ablation_preds.to_csv(out_dir / "ablation_predictions.csv", index=False)
    print(f"\n✓ Saved to {out_dir}")

if __name__ == "__main__":
    main()
