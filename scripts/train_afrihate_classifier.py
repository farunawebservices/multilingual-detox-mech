import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)

hf_token = os.getenv("HF_TOKEN") or True

MODEL_NAME = "Davlan/afro-xlmr-base"

LABEL_MAPPING = {
    "normal": 0, "0": 0, 0: 0,
    "abuse": 1, "abusive": 1, "1": 1, 1: 1,
    "hate": 2, "2": 2, 2: 2
}

id2label = {0: "Normal", 1: "Abuse", 2: "Hate"}
label2id = {"Normal": 0, "Abuse": 1, "Hate": 2}

# 1. Load Datasets
def load_lang(lang_code):
    return load_dataset("afrihate/afrihate", lang_code, token=hf_token)

print("Loading Hausa & Yoruba datasets...")
ds_hau = load_lang("hau")
ds_yor = load_lang("yor")

train_ds = concatenate_datasets([ds_hau["train"], ds_yor["train"]])
val_ds = concatenate_datasets([ds_hau["validation"], ds_yor["validation"]])

# 2. Tokenization & Vocab Validation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
vocab_size = len(tokenizer)
unk_id = tokenizer.unk_token_id or 3

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["tweet"], 
        truncation=True, 
        max_length=128
    )
    
    # Ensure all token IDs are strictly within [0, vocab_size - 1]
    cleaned_input_ids = []
    for input_id_seq in model_inputs["input_ids"]:
        cleaned_seq = [
            tid if (0 <= tid < vocab_size) else unk_id 
            for tid in input_id_seq
        ]
        cleaned_input_ids.append(cleaned_seq)
    model_inputs["input_ids"] = cleaned_input_ids

    formatted_labels = []
    for lbl in examples["label"]:
        key = str(lbl).strip().lower() if isinstance(lbl, (str, int)) else lbl
        if key in LABEL_MAPPING:
            formatted_labels.append(LABEL_MAPPING[key])
        else:
            formatted_labels.append(int(lbl))
            
    model_inputs["labels"] = formatted_labels
    return model_inputs

print(f"Tokenizing datasets (Tokenizer Vocab Size: {vocab_size})...")
raw_cols = train_ds.column_names

train_encoded = train_ds.map(
    preprocess_function, 
    batched=True, 
    remove_columns=raw_cols
)
val_encoded = val_ds.map(
    preprocess_function, 
    batched=True, 
    remove_columns=raw_cols
)

train_encoded.set_format("torch")
val_encoded.set_format("torch")

# 3. Model & Trainer Configuration
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    token=hf_token
)

training_args = TrainingArguments(
    output_dir="./results_afrihate",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print("Starting training...")
trainer.train()

print("Saving final model checkpoint...")
trainer.save_model("./results_afrihate/final_model")
tokenizer.save_pretrained("./results_afrihate/final_model")
print("Training complete! Model successfully saved to ./results_afrihate/final_model")
