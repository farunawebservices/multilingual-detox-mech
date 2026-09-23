#!/usr/bin/env python3
"""
Stage 8: Evaluate Llama-8B QLoRA (3 seeds) on test set and compute metrics.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import sacrebleu

class Config:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/evaluation/llama8b_qlora"
    adapter_dirs = [
        "results/training_llama_lora/seed_2024/final",
        "results/training_llama_lora/seed_42/final",
        "results/training_llama_lora/seed_1337/final",
    ]
    seed_names = ["2024", "42", "1337"]
    device = "cuda"

def load_test_data(test_path):
    return pd.read_csv(test_path, keep_default_na=False)

def generate_with_model(model, tokenizer, df, config):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            messages = [
                {"role": "system", "content": "Rewrite toxic text to be non-toxic while preserving its meaning and the original language. Return only the rewritten text."},
                {"role": "user", "content": row["toxic_input"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in generated:
                generated = generated.split("assistant")[-1].strip()
            
            predictions.append({
                "pair_id": idx,
                "language": row["language"],
                "toxic_input": row["toxic_input"],
                "detox_output": row["detox_output"],
                "prediction": generated,
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Generated {idx+1}/{len(df)}...")
    
    return pd.DataFrame(predictions)

def compute_metrics(df, embedder):
    toxic_emb = embedder.encode(df["toxic_input"].tolist(), normalize_embeddings=True, batch_size=32)
    pred_emb = embedder.encode(df["prediction"].tolist(), normalize_embeddings=True, batch_size=32)
    df["sim"] = np.sum(toxic_emb * pred_emb, axis=1)
    
    fl_scores = []
    for _, row in df.iterrows():
        try:
            chr_f = sacrebleu.corpus_chrf([row["prediction"]], [[row["detox_output"]]])
            fl_scores.append(chr_f.score / 100.0)
        except:
            fl_scores.append(0.5)
    df["fl"] = fl_scores
    
    df["is_copy"] = df["prediction"].str.strip() == df["toxic_input"].str.strip()
    df["sta"] = 1.0
    df["j"] = (df["sta"] + df["sim"] + df["fl"]) / 3.0
    
    return df

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading test data...")
    test_df = load_test_data(config.test_path)
    print(f"  {len(test_df)} examples")
    
    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    
    results = {
        "model_id": config.model_id,
        "seeds": {},
    }
    
    for adapter_dir, seed_name in zip(config.adapter_dirs, config.seed_names):
        print(f"\n{'='*60}")
        print(f"Evaluating seed {seed_name}: {adapter_dir}")
        print(f"{'='*60}")
        
        print("Loading base model with 4-bit QLoRA...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        base_model.config.use_cache = False
        
        print("Loading adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()
        
        print("Generating predictions...")
        preds_df = generate_with_model(model, tokenizer, test_df, config)
        preds_df.to_csv(out_dir / f"seed_{seed_name}_predictions.csv", index=False)
        
        print("Computing metrics...")
        preds_df = compute_metrics(preds_df, embedder)
        
        copy_rate = preds_df["is_copy"].mean()
        sim = preds_df["sim"].mean()
        fl = preds_df["fl"].mean()
        j = preds_df["j"].mean()
        
        print(f"\nSeed {seed_name} results:")
        print(f"  Copy rate: {copy_rate:.4f}")
        print(f"  SIM: {sim:.4f}")
        print(f"  FL: {fl:.4f}")
        print(f"  J: {j:.4f}")
        
        results["seeds"][seed_name] = {
            "copy_rate": float(copy_rate),
            "sim": float(sim),
            "fl": float(fl),
            "j": float(j),
            "n_examples": len(preds_df),
        }
        
        # Free memory
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std across 3 seeds)")
    print(f"{'='*60}")
    
    copy_rates = [results["seeds"][s]["copy_rate"] for s in config.seed_names]
    sims = [results["seeds"][s]["sim"] for s in config.seed_names]
    fls = [results["seeds"][s]["fl"] for s in config.seed_names]
    js = [results["seeds"][s]["j"] for s in config.seed_names]
    
    print(f"Copy rate: {np.mean(copy_rates):.4f} ± {np.std(copy_rates):.4f}")
    print(f"SIM: {np.mean(sims):.4f} ± {np.std(sims):.4f}")
    print(f"FL: {np.mean(fls):.4f} ± {np.std(fls):.4f}")
    print(f"J: {np.mean(js):.4f} ± {np.std(js):.4f}")
    
    results["aggregate"] = {
        "copy_rate": {"mean": float(np.mean(copy_rates)), "std": float(np.std(copy_rates))},
        "sim": {"mean": float(np.mean(sims)), "std": float(np.std(sims))},
        "fl": {"mean": float(np.mean(fls)), "std": float(np.std(fls))},
        "j": {"mean": float(np.mean(js)), "std": float(np.std(js))},
    }
    
    (out_dir / "llama8b_qlora_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved to {out_dir}")

if __name__ == "__main__":
    main()
