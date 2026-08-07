import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Paths to toxic-only corpora
ha_path = "data/processed/ha_toxic_only.csv"
yo_path = "data/processed/yo_toxic_only.csv"

for p in [ha_path, yo_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} does not exist. Run make_multilingual_hate_ha_yo.py first.")

df_ha = pd.read_csv(ha_path)
df_yo = pd.read_csv(yo_path)

print("Loaded Hausa toxic-only rows:", len(df_ha))
print("Loaded Yoruba toxic-only rows:", len(df_yo))

# Sample 400 rows per language for synthetic detox
N = 400

df_ha_sample = df_ha.sample(n=min(N, len(df_ha)), random_state=42).reset_index(drop=True)
df_yo_sample = df_yo.sample(n=min(N, len(df_yo)), random_state=42).reset_index(drop=True)

print("Hausa sample size:", len(df_ha_sample))
print("Yoruba sample size:", len(df_yo_sample))

# Load mt0-xl-detox-orpo model[web:65][web:68]
model_name = "s-nlp/mt0-xl-detox-orpo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def detoxify_texts(texts, lang_prompt):
    """
    texts: list of toxic inputs
    lang_prompt: language-specific prompt string, e.g. "Detoxify (Yoruba): "
    """
    detoxified = []
    for t in texts:
        prompt = f"{lang_prompt}{t}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                do_sample=False
            )
        det_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        detoxified.append(det_text)
    return detoxified

# Hausa synthetic detox
print("Generating synthetic Hausa detox outputs...")
ha_texts = df_ha_sample["toxic_input"].tolist()
ha_detox_outputs = detoxify_texts(ha_texts, lang_prompt="Detoxify (Hausa): ")

df_ha_syn = pd.DataFrame()
df_ha_syn["language"] = "ha"
df_ha_syn["toxic_input"] = ha_texts
df_ha_syn["detox_output"] = ha_detox_outputs
df_ha_syn["source"] = "mt0xl_detox_synthetic"
df_ha_syn["split"] = "ha_synthetic_400"

# Yoruba synthetic detox
print("Generating synthetic Yoruba detox outputs...")
yo_texts = df_yo_sample["toxic_input"].tolist()
yo_detox_outputs = detoxify_texts(yo_texts, lang_prompt="Detoxify (Yoruba): ")

df_yo_syn = pd.DataFrame()
df_yo_syn["language"] = "yo"
df_yo_syn["toxic_input"] = yo_texts
df_yo_syn["detox_output"] = yo_detox_outputs
df_yo_syn["source"] = "mt0xl_detox_synthetic"
df_yo_syn["split"] = "yo_synthetic_400"

os.makedirs("data/processed", exist_ok=True)
df_ha_syn.to_csv("data/processed/ha_detox_synthetic.csv", index=False)
df_yo_syn.to_csv("data/processed/yo_detox_synthetic.csv", index=False)

print(f"Saved {len(df_ha_syn)} synthetic Hausa pairs to data/processed/ha_detox_synthetic.csv")
print(f"Saved {len(df_yo_syn)} synthetic Yoruba pairs to data/processed/yo_detox_synthetic.csv")
