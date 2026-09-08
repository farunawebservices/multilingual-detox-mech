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
RESULTS = OUT / "gpt5_mini_detox_test_2.csv"

KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-5-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

OUT.mkdir(parents=True, exist_ok=True)

def prompt(language, text):
    return f"""You are creating one candidate rewrite for a text-detoxification dataset.

Language: {language}

SOURCE:
{text}

Task:
Rewrite the source into a natural, non-toxic version in the same language.
Make the smallest possible change. Preserve the topic, target, facts, tense,
modality, and speech act. Do not invent context. Do not add a generic apology.
Do not change the source into praise. Preserve any non-toxic criticism.

Remove or soften profanity, insults, hostile language, and threats.

Return only one line:
DETOX: <one rewritten sentence>

If safe meaning-preserving rewriting is impossible, return exactly:
DETOX: REVIEW_REQUIRED
"""

def call(prompt_text):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "gpt5-detox-test",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    body = response.json()

    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        return "", choice.get("finish_reason", ""), str(message), response.status_code

    match = re.search(r"^DETOX:\s*(.+)$", content.strip(), flags=re.MULTILINE)
    output = match.group(1).strip() if match else content.strip()

    return output, choice.get("finish_reason", ""), content, response.status_code

yo = pd.read_csv(YO_PATH).head(1).copy()
xh = pd.read_csv(XH_PATH).head(1).copy()
test = pd.concat([yo, xh], ignore_index=True)

rows = []

for i, row in test.iterrows():
    language = row["language"]
    source = row["toxic_input_processing"]

    print(f"[{i + 1}/10] {language}: {source[:80]}", flush=True)

    try:
        detox, finish_reason, raw, status = call(prompt(language, source))
        error = ""
    except Exception as exc:
        detox = ""
        finish_reason = ""
        raw = ""
        status = ""
        error = repr(exc)

    rows.append({
        "row_id": i + 1,
        "source_id": row["id"],
        "language": language,
        "afrihate_label": row["label"],
        "toxic_input_processing": source,
        "gpt5_detox_output": detox,
        "finish_reason": finish_reason,
        "http_status": status,
        "error": error,
        "raw_response": raw,
    })

    pd.DataFrame(rows).to_csv(RESULTS, index=False, encoding="utf-8")
    time.sleep(0.5)

print(f"\nWrote: {RESULTS}")