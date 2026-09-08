import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = Path("/workspace/multilingual-detox-mech")
INPUT_DIR = ROOT / "data" / "external" / "afrihate"
OUT = ROOT / "outputs" / "phase1_source_screen"

INPUTS = [
    INPUT_DIR / "afrihate_yor_toxic_candidates_200.csv",
    INPUT_DIR / "afrihate_xho_toxic_candidates_200.csv",
]

RESULTS = OUT / "afrihate_gpt5_source_screen_400.csv"
PARTIAL = OUT / "afrihate_gpt5_source_screen_400_partial.csv"
MANIFEST = OUT / "afrihate_gpt5_source_screen_manifest.json"

MODEL = os.environ.get("SCREEN_MODEL", "openai/gpt-5")
KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set.")

for path in INPUTS:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")

OUT.mkdir(parents=True, exist_ok=True)

PROMPT = """You are screening a real social-media post for a multilingual
text-detoxification dataset. The source was previously annotated as Abuse or Hate
by a native-speaker dataset, but you must independently assess whether it is
appropriate for a meaning-preserving detoxification task.

Return only valid JSON:

{{
  "disposition": "ACCEPT_FOR_DETOX|REJECT_NOT_TARGET_LANGUAGE|REJECT_TOO_CODE_MIXED|REJECT_NOT_TOXIC_IN_ISOLATION|REJECT_NOT_DETOXIFIABLE|REJECT_TOO_SEVERE_OR_IDENTITY_TARGETED|REJECT_MALFORMED_OR_INCOMPLETE",
  "source_status": "toxic_suitable|toxic_context_required|not_toxic",
  "detoxifiable": true,
  "toxicity_level": "low|medium|high",
  "tone": "aggressive|dismissive|derogatory|threatening|vulgar|critical|other",
  "language_style": "informal|colloquial|neutral|other",
  "implied_sentiment": "negative|angry|disdainful|neutral|other",
  "toxic_span": "exact toxic word or phrase, or NONE",
  "toxic_function": "brief description",
  "recommended_strategy": "delete_replace|rephrase_derogatory|soften_criticism|review_required",
  "meaning_risk": "LOW|MEDIUM|HIGH",
  "reason": "brief explanation"
}}

Acceptance rules:
- ACCEPT_FOR_DETOX only when the post is primarily in the target language,
  clearly toxic in isolation, meaningful enough to interpret, and can be
  rewritten non-toxically while preserving its main proposition, target,
  facts, tense, modality, and speech act.
- Reject posts that are mostly another language, excessively code-mixed,
  malformed, incomplete, unclear, or not toxic in isolation.
- Reject deeply identity-targeted hate, severe threats, encouragement of
  violence, or material whose main harmful proposition cannot safely be
  preserved in a detoxified rewrite.
- A short insult can be accepted only if its intended target and proposition
  can be preserved.
- Usernames and URL placeholders are metadata, not toxic content.
- Do not generate a rewrite.

Language: {language}
AfriHate label: {label}

POST:
{text}
"""

def api_call(prompt):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "afrihate-gpt5-source-screen",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    last_error = ""

    for attempt in range(4):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                return json.loads(raw), raw, ""

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"

            if response.status_code not in {429, 500, 502, 503, 504}:
                break

        except Exception as exc:
            last_error = repr(exc)

        time.sleep(2 ** attempt)

    return {}, "", last_error

def get_value(obj, key, default=""):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def load_inputs():
    frames = []

    for path in INPUTS:
        df = pd.read_csv(path)
        df["screen_row_id"] = [
            f"{path.stem}_{i + 1}"
            for i in range(len(df))
        ]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def main():
    source_df = load_inputs()

    rows = []
    completed = set()

    if PARTIAL.exists():
        prior = pd.read_csv(PARTIAL)
        completed = set(prior["screen_row_id"].astype(str))
        rows = prior.to_dict("records")

    print(f"Total candidates: {len(source_df)}", flush=True)
    print(f"Already screened: {len(completed)}", flush=True)
    print(f"Model: {MODEL}", flush=True)

    for _, row in source_df.iterrows():
        screen_row_id = str(row["screen_row_id"])

        if screen_row_id in completed:
            continue

        language = str(row["language"])
        label = str(row["label"])
        text = str(row["toxic_input_processing"])

        print(
            f"Screening {screen_row_id} | {language} | {text[:90]}",
            flush=True,
        )

        prompt = PROMPT.format(
            language=language,
            label=label,
            text=text,
        )

        result, raw, error = api_call(prompt)

        rows.append({
            "screen_row_id": screen_row_id,
            "source_dataset": row.get("source_dataset", "AfriHate"),
            "source_config": row.get("source_config", ""),
            "source_split": row.get("source_split", ""),
            "source_id": row.get("id", ""),
            "language": language,
            "afrihate_label": label,
            "length": row.get("length", ""),
            "toxic_input_raw": row.get("toxic_input_raw", ""),
            "toxic_input_processing": text,
            "disposition": get_value(result, "disposition", "ERROR"),
            "source_status": get_value(result, "source_status", ""),
            "detoxifiable": get_value(result, "detoxifiable", ""),
            "toxicity_level": get_value(result, "toxicity_level", ""),
            "tone": get_value(result, "tone", ""),
            "language_style": get_value(result, "language_style", ""),
            "implied_sentiment": get_value(result, "implied_sentiment", ""),
            "toxic_span": get_value(result, "toxic_span", ""),
            "toxic_function": get_value(result, "toxic_function", ""),
            "recommended_strategy": get_value(result, "recommended_strategy", ""),
            "meaning_risk": get_value(result, "meaning_risk", ""),
            "screen_reason": get_value(result, "reason", ""),
            "screen_error": error,
            "screen_raw_response": raw,
            "screen_model": MODEL,
            "screened_utc": datetime.now(timezone.utc).isoformat(),
        })

        pd.DataFrame(rows).to_csv(PARTIAL, index=False, encoding="utf-8")
        time.sleep(0.35)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS, index=False, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "AfriHate",
        "screen_model": MODEL,
        "raw_candidate_count": int(len(source_df)),
        "method": "GPT-5 eligibility screen before detoxification",
        "files": {
            "results": str(RESULTS),
            "partial": str(PARTIAL),
            "inputs": [str(p) for p in INPUTS],
        },
    }

    with open(MANIFEST, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"\nWrote: {RESULTS}")
    print(f"Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()