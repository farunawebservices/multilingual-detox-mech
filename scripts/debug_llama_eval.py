#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd

# Reuse the dataset logic from the training script
import sys, importlib.util
spec = importlib.util.spec_from_file_location("train_script", "scripts/07_llama_lora_finetune.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
DetoxDataset = train_mod.DetoxDataset
Config = train_mod.Config

config = Config()
config.output_dir = "results/training_llama_lora/seed_2024"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model in 4-bit...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
base_model.config.use_cache = False

print("Loading LoRA adapter...")
final_dir = Path(config.output_dir) / "final"
model = PeftModel.from_pretrained(base_model, str(final_dir))
model.eval()

print("Loading dev data...")
dev_df = pd.read_csv(config.dev_path, keep_default_na=False)
print(f"  Dev examples: {len(dev_df)}")

dev_dataset = DetoxDataset(dev_df, tokenizer, config)
loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=lambda xs: {
    "input_ids": torch.stack([x["input_ids"] for x in xs]),
    "attention_mask": torch.stack([x["attention_mask"] for x in xs]),
    "labels": torch.stack([x["labels"] for x in xs]),
})

total_loss = 0.0
n_batches = 0
n_valid = 0

print("Computing dev loss manually...")
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_valid += 1
        n_batches += 1
        if n_batches % 20 == 0:
            print(f"  Batch {n_batches}, valid batches so far: {n_valid}, running avg loss: {total_loss / max(1, n_valid)}")

if n_valid == 0:
    print("ERROR: all dev losses were NaN/Inf!")
else:
    avg_loss = total_loss / n_valid
    print(f"Dev batches: {n_batches}, valid batches: {n_valid}")
    print(f"Manual average dev loss: {avg_loss:.4f}")
