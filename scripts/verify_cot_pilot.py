import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = Path("/workspace/detox_cot_pilot")
OUT = ROOT / "outputs"
INPUT = OUT / "cot_pilot_generations.csv"
OUTPUT = OUT / "cot_pilot_verification.csv"
PARTIAL = OUT / "cot_pilot_verification_partial.csv"

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set.")
if not INPUT.exists():
    raise SystemExit(f"Missing input file: {INPUT}")

def prompt(language, source, candidate):
    return f"""You are a strict quality-control reviewer for a {language}
text-detoxification dataset.

Evaluate the source sentence and the proposed detoxified rewrite. This is an
automated screening step only. A native speaker must make the final decision.

Do not provide hidden reasoning. Return valid JSON only, with exactly these keys:

{{
  "source_status": "toxic_suitable|toxic_context_required|not_toxic",
  "detox_status": "accepted|needs_edit|rejected",
  "toxicity_removed": true,
  "meaning_preserved": 1,
  "naturalness": 1,
  "language_correct": true,
  "new_content_added": false,
  "generic_template": false,
  "copied_input": false,
  "reason": "brief explanation",
  "native_review_required": true
}}

Rules:
- "toxic_suitable" means the source is actually toxic in isolation and can be
  rewritten while preserving the main meaning.
- Use "toxic_context_required" if missing context or idiom interpretation makes
  reliable judgment impossible.
- Use "not_toxic" if the source is not actually toxic in isolation.
- "accepted" requires toxicity removal, meaning preservation, natural language,
  no invented content, and no generic template.
- A candidate that retains a threat, insult, identity attack, or equivalent
  toxicity is not accepted.
- A candidate that changes the proposition, target, facts, speech act, or
  pragmatic force substantially is not accepted.
- A generic safe response is not accepted.
- Score meaning_preserved and naturalness from 1 to 5.

Language: {language}

SOURCE:
{source}

CANDIDATE:
{candidate}
"""

def call_api(text):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "detox-cot-pilot-verifier",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                body = r.json()
                content = body["choices"][0]["message"]["content"]
                return json.loads(content), content, body
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last = repr(exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(last)

def main():
    df = pd.read_csv(INPUT)
    done = set()

    if PARTIAL.exists():
        prior = pd.read_csv(PARTIAL)
        done = set(prior["row_id"].astype(int))
        rows = prior.to_dict("records")
    else:
        rows = []

    for row_id, row in df.iterrows():
        if row_id in done:
            continue

        print(f"Verifying row {row_id+1}/{len(df)}: {row['source_id']} | {row['condition']}", flush=True)
        try:
            verdict, raw, api = call_api(
                prompt(row["language"], row["toxic_input"], row["generated_output"])
            )
            status = "ok"
            error = ""
        except Exception as exc:
            verdict = {}
            raw = ""
            api = {}
            status = "error"
            error = repr(exc)

        rows.append({
            "row_id": row_id,
            "source_id": row["source_id"],
            "language": row["language"],
            "condition": row["condition"],
            "toxic_input": row["toxic_input"],
            "generated_output": row["generated_output"],
            "verification_status": status,
            "error": error,
            "verifier_model": MODEL,
            "verifier_raw_response": raw,
            "source_status": verdict.get("source_status", ""),
            "detox_status": verdict.get("detox_status", ""),
            "toxicity_removed": verdict.get("toxicity_removed", ""),
            "meaning_preserved": verdict.get("meaning_preserved", ""),
            "naturalness": verdict.get("naturalness", ""),
            "language_correct": verdict.get("language_correct", ""),
            "new_content_added": verdict.get("new_content_added", ""),
            "generic_template": verdict.get("generic_template", ""),
            "copied_input": verdict.get("copied_input", ""),
            "reason": verdict.get("reason", ""),
            "native_review_required": verdict.get("native_review_required", ""),
            "response_id": api.get("id", ""),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })

        pd.DataFrame(rows).to_csv(PARTIAL, index=False, encoding="utf-8")
        time.sleep(0.4)

    pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\nWrote {OUTPUT}")

if __name__ == "__main__":
    main()
