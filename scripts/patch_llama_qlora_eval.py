from pathlib import Path

p = Path("scripts/07_llama_lora_finetune.py")
s = p.read_text()

# 1) Disable bf16/fp16 in TrainingArguments
old_args = '''        bf16=True,
        fp16=False,'''

new_args = '''        bf16=False,
        fp16=False,'''

if old_args not in s:
    raise RuntimeError("Expected bf16/fp16 block not found in TrainingArguments.")

s = s.replace(old_args, new_args)

# 2) Add a custom compute_loss to force FP32 in eval for QLoRA
# Insert after the Trainer instantiation, before train() call.
old_trainer = '''    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )'''

new_trainer = '''    class QLoRASafeTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Cast to FP32 for loss computation to avoid NaN in eval with QLoRA
            with torch.cuda.amp.autocast(dtype=torch.float32):
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    trainer = QLoRASafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )'''

if old_trainer not in s:
    raise RuntimeError("Expected Trainer instantiation not found.")

s = s.replace(old_trainer, new_trainer)

p.write_text(s)
print("Patched script for QLoRA-safe eval.")
