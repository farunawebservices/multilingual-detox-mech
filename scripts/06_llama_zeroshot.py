#!/usr/bin/env python3
"""
Stage 6: Llama-3-8B zero-shot and few-shot detoxification baseline.
Tests whether Llama can detoxify without fine-tuning.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import sacrebleu

class Config:
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    test_path = "results/training_v2_400plus/seed_2024/test.csv"
    output_dir = "results/evaluation/llama_zeroshot"
    device = "cuda"
    n_examples_per_lang = 10
    max_new_tokens = 128

def load_test_data(test_path, n_examples_per_lang=10):
    df = pd.read_csv(test_path, keep_default_na=False)
    # Check columns
    print(f"  Columns: {df.columns.tolist()}")
    # Sample n_examples_per_lang per language
    if "language" in df.columns:
        sampled = df.groupby("language").apply(lambda x: x.sample(n=min(len(x), n_examples_per_lang), random_state=42)).reset_index(drop=True)
    else:
        # No language column, just sample randomly
        sampled = df.sample(n=min(len(df), n_examples_per_lang * 9), random_state=42).reset_index(drop=True)
    return sampled

def create_zero_shot_prompt(toxic_text):
    """Zero-shot prompt for Llama."""
    return f"""### Instruction:
Detoxify the following text while preserving meaning and language.

### Input:
{toxic_text}

### Response:
"""

def create_few_shot_prompt(toxic_text, examples):
    """Few-shot prompt with 2-3 examples."""
    prompt = "### Instruction:\nDetoxify the following text while preserving meaning and language.\n\n### Examples:\n"
    for ex in examples[:3]:
        prompt += f"Input: {ex['toxic']}\nOutput: {ex['detox']}\n\n"
    prompt += f"### Input:\n{toxic_text}\n\n### Response:\n"
    return prompt

def generate_with_llama(model, tokenizer, prompt, config):
    """Generate detoxified text using Llama."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part (after "### Response:")
    if "### Response:" in generated:
        generated = generated.split("### Response:")[-1].strip()
    
    return generated

def compute_metrics(df, embedder):
    """Compute SIM, FL, and copy rate."""
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
    
    print("Loading Llama model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading test data...")
    df = load_test_data(config.test_path, config.n_examples_per_lang)
    
    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    
    results = {
        "model_id": config.model_id,
        "n_examples": len(df),
        "zero_shot": {},
        "few_shot": {},
    }
    
    # === ZERO-SHOT ===
    print("\n" + "="*60)
    print("ZERO-SHOT GENERATION")
    print("="*60)
    
    predictions_zero = []
    for idx, row in df.iterrows():
        prompt = create_zero_shot_prompt(row["toxic_input"])
        pred = generate_with_llama(model, tokenizer, prompt, config)
        predictions_zero.append({
            "pair_id": idx,
            "toxic_input": row["toxic_input"],
            "detox_output": row["detox_output"],
            "prediction": pred,
        })
        if idx % 10 == 0:
            print(f"  Generated {idx+1}/{len(df)}...")
    
    df_zero = pd.DataFrame(predictions_zero)
    df_zero = compute_metrics(df_zero, embedder)
    
    print(f"\nZero-shot results:")
    print(f"  Copy rate: {df_zero['is_copy'].mean():.4f}")
    print(f"  SIM: {df_zero['sim'].mean():.4f}")
    print(f"  FL: {df_zero['fl'].mean():.4f}")
    print(f"  J: {df_zero['j'].mean():.4f}")
    
    results["zero_shot"] = {
        "copy_rate": float(df_zero["is_copy"].mean()),
        "sim": float(df_zero["sim"].mean()),
        "fl": float(df_zero["fl"].mean()),
        "j": float(df_zero["j"].mean()),
    }
    
    df_zero.to_csv(out_dir / "zeroshot_predictions.csv", index=False)
    
    # === FEW-SHOT ===
    print("\n" + "="*60)
    print("FEW-SHOT GENERATION")
    print("="*60)
    
    # Create few-shot examples from first 10 rows
    few_shot_examples = [
        {"toxic": row["toxic_input"], "detox": row["detox_output"]}
        for _, row in df.head(10).iterrows()
    ]
    
    predictions_few = []
    for idx, row in df.iterrows():
        prompt = create_few_shot_prompt(row["toxic_input"], few_shot_examples)
        pred = generate_with_llama(model, tokenizer, prompt, config)
        predictions_few.append({
            "pair_id": idx,
            "toxic_input": row["toxic_input"],
            "detox_output": row["detox_output"],
            "prediction": pred,
        })
        if idx % 10 == 0:
            print(f"  Generated {idx+1}/{len(df)}...")
    
    df_few = pd.DataFrame(predictions_few)
    df_few = compute_metrics(df_few, embedder)
    
    print(f"\nFew-shot results:")
    print(f"  Copy rate: {df_few['is_copy'].mean():.4f}")
    print(f"  SIM: {df_few['sim'].mean():.4f}")
    print(f"  FL: {df_few['fl'].mean():.4f}")
    print(f"  J: {df_few['j'].mean():.4f}")
    
    results["few_shot"] = {
        "copy_rate": float(df_few["is_copy"].mean()),
        "sim": float(df_few["sim"].mean()),
        "fl": float(df_few["fl"].mean()),
        "j": float(df_few["j"].mean()),
    }
    
    df_few.to_csv(out_dir / "fewshot_predictions.csv", index=False)
    
    # Save results
    (out_dir / "llama_zeroshot_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved to {out_dir}")
    
    # Qualitative examples
    print("\n" + "="*60)
    print("QUALITATIVE EXAMPLES (Zero-shot)")
    print("="*60)
    for i in range(min(3, len(df_zero))):
        row = df_zero.iloc[i]
        print(f"\nExample {i+1}:")
        print(f"  Toxic:    {row['toxic_input'][:80]}...")
        print(f"  Reference: {row['detox_output'][:80]}...")
        print(f"  Predicted: {row['prediction'][:80]}...")

if __name__ == "__main__":
    main()
