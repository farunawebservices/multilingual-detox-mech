"""
Smoke-test mT5 fine-tuning for the v2 grouped 9-language detox corpus.
Manual training loop to avoid transformers 5.x Trainer issues.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/mt5_data_v2_grouped_9lang"
OUT_DIR = Path(__file__).resolve().parent

SAMPLE_TRAIN = DATA_DIR / "mt5_detox_train_sample_20.jsonl"
SAMPLE_DEV   = DATA_DIR / "mt5_detox_dev.jsonl"
MODEL_NAME = "google/mt5-small"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 2
NUM_EPOCHS = 1
MAX_STEPS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10

def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

class DetoxDataset(Dataset):
    def __init__(self, records, tokenizer, max_input_len, max_target_len):
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        instruction = r["instruction"]
        inp = r["input"]
        out = r["output"]
        source_text = f"{instruction} Input: {inp}"

        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            out,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
        }

def main():
    print("=== SMOKE-TEST mT5 FINE-TUNING ===")
    print("Model:", MODEL_NAME)
    print("Train sample:", SAMPLE_TRAIN)
    print("Dev:", SAMPLE_DEV)

    train_records = load_jsonl(SAMPLE_TRAIN)
    dev_records   = load_jsonl(SAMPLE_DEV)

    print("Train records:", len(train_records))
    print("Dev records:",   len(dev_records))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    train_dataset = DetoxDataset(train_records, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    dev_dataset   = DetoxDataset(dev_records, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        return LEARNING_RATE

    print("\nStarting smoke training...")
    global_step = 0
    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = get_lr(global_step)

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = total_loss / 10
                print(f"Step {global_step}: loss={avg_loss:.4f}")
                total_loss = 0.0

    print(f"\nTraining completed. Global step: {global_step}")

    save_dir = OUT_DIR / "mt5_smoke_run"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics = {
        "train_samples": len(train_records),
        "dev_samples": len(dev_records),
        "model_name": MODEL_NAME,
        "max_steps": MAX_STEPS,
        "num_epochs": NUM_EPOCHS,
        "global_step": global_step,
        "avg_loss": float(total_loss / max(global_step % 10, 1)) if global_step > 0 else None,
    }
    metrics_path = OUT_DIR / "smoke_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Smoke test completed. Metrics:", metrics_path)
    print("Model saved to:", save_dir)

if __name__ == "__main__":
    main()
