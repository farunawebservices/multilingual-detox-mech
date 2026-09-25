from pathlib import Path

p = Path("scripts/07_llama_lora_finetune.py")
s = p.read_text()

# 1) Disable eval in TrainingArguments
old_eval = 'eval_strategy="epoch",'
new_eval = 'eval_strategy="no",'

if old_eval not in s:
    raise RuntimeError("Expected eval_strategy not found.")

s = s.replace(old_eval, new_eval)

# 2) Remove load_best_model_at_end and metric_for_best_model (not needed without eval)
s = s.replace('load_best_model_at_end=True,', 'load_best_model_at_end=False,')
s = s.replace('metric_for_best_model="eval_loss",', 'metric_for_best_model=None,')
s = s.replace('greater_is_better=False,', '# greater_is_better=False,  # no eval')

# 3) After training, compute dev loss manually and update metadata
old_save = '''    print("Saving best adapter and tokenizer...")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = trainer.evaluate()'''

new_save = '''    print("Saving adapter and tokenizer...")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Manual dev loss computation (Trainer eval is NaN with QLoRA)
    print("Computing dev loss manually...")
    import torch
    from torch.utils.data import DataLoader
    from transformers import BitsAndBytesConfig
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reload base + adapter
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

    if n_valid == 0:
        raise RuntimeError("All dev losses were NaN/Inf!")

    eval_loss = total_loss / n_valid
    print(f"Manual dev loss: {eval_loss:.4f} (from {n_valid} valid batches)")

    metrics = {"eval_loss": eval_loss}'''

if old_save not in s:
    raise RuntimeError("Expected save block not found.")

s = s.replace(old_save, new_save)

p.write_text(s)
print("Patched script to disable Trainer eval and compute dev loss manually.")
