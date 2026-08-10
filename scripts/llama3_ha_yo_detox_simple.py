import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from textwrap import dedent

HA_TOXIC_PATH = "data/processed/ha_toxic_only.csv"
YO_TOXIC_PATH = "data/processed/yo_toxic_only.csv"

for p in [HA_TOXIC_PATH, YO_TOXIC_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found; create toxic-only files first.")

df_ha = pd.read_csv(HA_TOXIC_PATH)
df_yo = pd.read_csv(YO_TOXIC_PATH)

print("Hausa toxic-only:", len(df_ha))
print("Yoruba toxic-only:", len(df_yo))

N = 400
df_ha_sample = df_ha.sample(n=min(N, len(df_ha)), random_state=42).reset_index(drop=True)
df_yo_sample = df_yo.sample(n=min(N, len(df_yo)), random_state=42).reset_index(drop=True)

model_name = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

def detox_one(text, lang_name):
    instruction = dedent(f"""
    You are a text detoxification assistant for {lang_name}.
    Rewrite the following toxic sentence in {lang_name} in a neutral, polite way.
    Preserve the main meaning but remove insults, slurs, and abusive language.
    Very important:
    - Answer ONLY with the rewritten sentence.
    - Do NOT include explanations, preambles, or meta text.
    - Do NOT use English or any other language; use {lang_name} only.
    - Do NOT use emojis.
    Toxic sentence:
    {str(text).strip()}
    Neutral rewrite:
    """).strip()

    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1
        )

    generated_ids = gen[0][input_ids.shape[1]:]
    det = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return det.strip()

def detox_batch(texts, lang_name):
    outputs = []
    for i, t in enumerate(texts):
        det = detox_one(t, lang_name)
        outputs.append(det)
        print(f"{lang_name} {i+1}/{len(texts)}")
    return outputs

print("Generating Hausa pseudo-parallel detox (simple one-shot)...")
ha_texts = df_ha_sample["toxic_input"].tolist()
ha_detox = detox_batch(ha_texts, "Hausa")

print("Generating Yoruba pseudo-parallel detox (simple one-shot)...")
yo_texts = df_yo_sample["toxic_input"].tolist()
yo_detox = detox_batch(yo_texts, "Yoruba")

os.makedirs("data/processed", exist_ok=True)

df_ha_out = pd.DataFrame({
    "language": "ha",
    "toxic_input": ha_texts,
    "detox_output": ha_detox,
    "source": "llama3.1_8b_abliterated_oneshot",
    "split": "ha_pseudollama_400"
})

df_yo_out = pd.DataFrame({
    "language": "yo",
    "toxic_input": yo_texts,
    "detox_output": yo_detox,
    "source": "llama3.1_8b_abliterated_oneshot",
    "split": "yo_pseudollama_400"
})

df_ha_out.to_csv("data/processed/ha_detox_pseudo_llama.csv", index=False)
df_yo_out.to_csv("data/processed/yo_detox_pseudo_llama.csv", index=False)

print("Saved Hausa pseudo-parallel pairs to data/processed/ha_detox_pseudo_llama.csv")
print("Saved Yoruba pseudo-parallel pairs to data/processed/yo_detox_pseudo_llama.csv")
