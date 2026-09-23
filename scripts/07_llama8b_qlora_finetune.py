#!/usr/bin/env python3
"""
Stage 7: Llama-3-8B-Instruct QLoRA fine-tuning for multilingual detoxification.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

class Config:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    train_path = "results/training_v2_400plus/seed_1337/train.csv"
    dev_path = "results/training_v2_400plus/seed_1337/dev.csv"
    output_dir = "results/training_llama_lora/seed_1337"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    batch_size = 1
    gradient_accumulation_steps = 16
    num_train_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 50
    max_length = 192
    seed = 1337

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DetoxDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.df)

    def _messages(self, toxic_text, detox_text=None):
        messages = [
            {
                "role": "system",
                "content": "Rewrite toxic text to be non-toxic while preserving its meaning and the original language. Return only the rewritten text.",
            },
            {"role": "user", "content": toxic_text},
        ]
        if detox_text is not None:
            messages.append({"role": "assistant", "content": detox_text})
        return messages

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = self.tokenizer.apply_chat_template(
            self._messages(row["toxic_input"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            self._messages(row["toxic_input"], row["detox_output"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.config.max_length)["input_ids"]
        encoding = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.config.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), self.config.max_length)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    config = Config()
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model with 4-bit NF4 QLoRA...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading frozen split files...")
    train_df = pd.read_csv(config.train_path, keep_default_na=False)
    dev_df = pd.read_csv(config.dev_path, keep_default_na=False)
    print(f"  Train: {len(train_df)} examples")
    print(f"  Dev: {len(dev_df)} examples")

    train_dataset = DetoxDataset(train_df, tokenizer, config)
    dev_dataset = DetoxDataset(dev_df, tokenizer, config)

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    print("Setting up training...")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Approx. optimizer steps: {steps_per_epoch * config.num_train_epochs}")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        logging_steps=25,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        optim="adamw_torch",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)

    print("Starting training...")
    train_result = trainer.train()

    print("Saving adapter and tokenizer...")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Manual dev loss
    print("Computing dev loss manually...")
    from torch.utils.data import DataLoader
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=quant_config, device_map="auto", low_cpu_mem_usage=True)
    base_model.config.use_cache = False
    model_eval = PeftModel.from_pretrained(base_model, str(final_dir))
    model_eval.eval()

    dev_dataset = DetoxDataset(dev_df, tokenizer, config)
    loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=lambda xs: {
        "input_ids": torch.stack([x["input_ids"] for x in xs]),
        "attention_mask": torch.stack([x["attention_mask"] for x in xs]),
        "labels": torch.stack([x["labels"] for x in xs]),
    })

    total_loss = 0.0
    n_valid = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_eval(**batch)
            loss = outputs.loss
            if torch.isfinite(loss):
                total_loss += loss.item()
                n_valid += 1

    eval_loss = total_loss / max(1, n_valid)
    print(f"Manual dev loss: {eval_loss:.4f} (from {n_valid} valid batches)")

    metadata = {
        "base_model_id": config.model_id,
        "split_seed": config.seed,
        "train_examples": len(train_df),
        "dev_examples": len(dev_df),
        "lora": {"r": config.lora_r, "alpha": config.lora_alpha, "dropout": config.lora_dropout, "target_modules": config.target_modules},
        "training": {"epochs": config.num_train_epochs, "batch_size": config.batch_size, "gradient_accumulation_steps": config.gradient_accumulation_steps, "learning_rate": config.learning_rate, "warmup_steps": config.warmup_steps, "max_length": config.max_length},
        "train_metrics": train_result.metrics,
        "eval_loss": eval_loss,
    }
    (final_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved to {final_dir}")

if __name__ == "__main__":
    main()
