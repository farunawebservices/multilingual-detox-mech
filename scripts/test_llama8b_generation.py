#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "results/training_llama_lora/seed_2024/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model in 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
base_model.config.use_cache = False

# Load adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Load a few test examples
test_df = pd.read_csv("results/training_v2_400plus/seed_2024/test.csv", keep_default_na=False)

print("Testing generation on 3 examples per language...\n")

for lang in test_df["language"].unique()[:3]:  # Just first 3 languages for speed
    lang_df = test_df[test_df["language"] == lang].head(3)
    print(f"\n=== {lang.upper()} ===")
    
    for _, row in lang_df.iterrows():
        messages = [
            {"role": "system", "content": "Rewrite toxic text to be non-toxic while preserving its meaning and the original language. Return only the rewritten text."},
            {"role": "user", "content": row["toxic_input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
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
        
        print(f"\nToxic:    {row['toxic_input'][:80]}...")
        print(f"Ref:      {row['detox_output'][:80]}...")
        print(f"Generated: {generated[:80]}...")
        print(f"Copy? {generated.strip() == row['toxic_input'].strip()}")
