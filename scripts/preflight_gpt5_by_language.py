import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/workspace/multilingual-detox-mech")
DATA = ROOT / "data" / "external" / "afrihate"
OUT = ROOT / "outputs" / "phase2_gpt5_detox"

YO_PATH = DATA / "afrihate_yor_toxic_candidates_200.csv"
XH_PATH = DATA / "afrihate_xho_toxic_candidates_200.csv"
RESULTS = OUT / "gpt5_language_preflight_4.csv"

KEY = os.environ["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "Yorùbá": "openai/gpt-5",
    "isiXhosa": "openai/gpt-5-mini",
}

OUT.mkdir(parents=True, exist_ok=True)

def make_prompt(language, text):
    return f"""Rewrite this social-media post into a natural, non-toxic version in the same language.

Keep the same topic, target, basic proposition, and language mixture where appropriate.
Make the smallest possible change. Do not add context, apologize, praise, or explain.
Preserve non-toxic criticism. Remove or soften insults, profanity, hostile language, and threats.

If safe meaning-preserving rewriting is impossible, output exactly:
REVIEW_REQUIRED

Language: {language}
Post: {text}

Output only the rewritten post or REVIEW_REQUIRED.
"""

def call(model, prompt):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "gpt5-language-preflight",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output only the requested rewritten post or REVIEW_REQUIRED."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2500,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=240)
    body = response.json()

    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")
    finish_reason = choice.get("finish_reason", "")

    if isinstance(content, str) and content.strip():
        return content.strip(), finish_reason, response.status_code, ""

    return "", finish_reason, response.status_code, str(message)

yo = pd.read_csv(YO_PATH).head(2).copy()
xh = pd.read_csv(XH_PATH).head(2).copy()
test = pd.concat([yo, xh], ignore_index=True)

rows = []

for i, row in test.iterrows():
    language = row["language"]
    model = MODELS[language]
    source = row["toxic_input_processing"]

    print(f"[{i + 1}/4] {language} | {model} | {source[:80]}", flush=True)

    try:
        output, finish_reason, status, error = call(
            model,
            make_prompt(language, source),
        )
    except Exception as exc:
        output = ""
        finish_reason = ""
        status = ""
        error = repr(exc)

    rows.append({
        "row_id": i + 1,
        "source_id": row["id"],
        "language": language,
        "generator_model": model,
        "afrihate_label": row["label"],
        "toxic_input_processing": source,
        "generated_output": output,
        "finish_reason": finish_reason,
        "http_status": status,
        "error": error,
    })

    pd.DataFrame(rows).to_csv(RESULTS, index=False, encoding="utf-8")
    time.sleep(0.5)

print(f"\nWrote: {RESULTS}")