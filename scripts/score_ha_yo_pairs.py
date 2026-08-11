import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

CLASSIFIER_PATH = "./results_afrihate/final_model"
LABSE_PATH = "sentence-transformers/LaBSE"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading classifier from", CLASSIFIER_PATH)
clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
clf_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(device).eval()
id2label = clf_model.config.id2label

clf_vocab_size = len(clf_tokenizer)

print("Loading LaBSE model from", LABSE_PATH)
labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_PATH)
labse_model = AutoModel.from_pretrained(LABSE_PATH).to(device).eval()

def sta_score(text):
    inputs = clf_tokenizer(str(text), truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=clf_vocab_size - 1)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = clf_model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
    
    normal_idx = None
    for k, v in id2label.items():
        if str(v).strip().lower() == "normal":
            normal_idx = int(k)
            break
            
    if normal_idx is None:
        normal_idx = 0
        
    return probs[normal_idx].item()

def labse_embed(texts):
    inputs = labse_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = labse_model(**inputs).pooler_output
    return F.normalize(out, dim=1)

def sim_score(text_a, text_b):
    emb = labse_embed([str(text_a), str(text_b)])
    return F.cosine_similarity(emb[0:1], emb[1:2]).item()

def score_file(in_path, out_path, lang_name):
    if not os.path.exists(in_path):
        print("Skipping " + lang_name + ": input path " + in_path + " does not exist.")
        return None

    df = pd.read_csv(in_path)
    print(f"{lang_name}: scoring {len(df)} rows...")

    sta_scores, sim_scores = [], []
    for i, row in df.iterrows():
        sta = sta_score(row["detox_output"])
        sim = sim_score(row["toxic_input"], row["detox_output"])
        sta_scores.append(sta)
        sim_scores.append(sim)
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"{lang_name} progress: {i+1}/{len(df)}")

    df["sta_score"] = sta_scores
    df["sim_score"] = sim_scores
    df.to_csv(out_path, index=False)
    print("Saved scored file to " + out_path)
    return df

df_ha = score_file(
    "data/processed/ha_detox_pseudo_llama_filtered.csv",
    "data/processed/ha_detox_pseudo_llama_scored.csv",
    "ha"
)
df_yo = score_file(
    "data/processed/yo_detox_pseudo_llama_filtered.csv",
    "data/processed/yo_detox_pseudo_llama_scored.csv",
    "yo"
)

if df_ha is not None:
    print("HA sta/sim describe:")
    print(df_ha[["sta_score","sim_score"]].describe())
if df_yo is not None:
    print("YO sta/sim describe:")
    print(df_yo[["sta_score","sim_score"]].describe())
