import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/workspace/detox_cot_pilot")
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260828
N_DEMOS = 8
N_EVAL = 6
TEMPERATURE = 0.2
MAX_TOKENS = 500
API_URL = "https://openrouter.ai/api/v1/chat/completions"

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")

if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set.")

LANGS = [
    {
        "code": "yo",
        "name": "Yorùbá",
        "path": DATA / "yoruba_pairs.csv",
        "input_col": "Toxic Sentence (Input)",
        "target_col": "Detoxified Sentence (Output)",
        "language_notes": (
            "Use natural Yorùbá. Preserve meaningful diacritics where possible. "
            "Do not switch to English unless the source deliberately code-switches."
        ),
    },
    {
        "code": "xh",
        "name": "isiXhosa",
        "path": DATA / "isixhosa_pairs.csv",
        "input_col": "Toxic Sentence (Input)",
        "target_col": "Detoxified Sentence (Output)",
        "language_notes": (
            "Use natural isiXhosa. Preserve the original language and intended "
            "meaning. Do not switch to English unless the source deliberately code-switches."
        ),
    },
]

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def clean_field(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def make_demos(df, input_col, target_col):
    rows = []
    for i, row in df.iterrows():
        rows.append(
            f"Example {len(rows)+1}\n"
            f"Toxic: {clean_field(row[input_col])}\n"
            f"Detoxified: {clean_field(row[target_col])}"
        )
    return "\n\n".join(rows)

def direct_prompt(lang, text):
    return f"""You are helping construct a research dataset for text detoxification in {lang['name']}.

Rewrite the input sentence so it is non-toxic while preserving its meaning.

Rules:
- Return only one rewritten sentence in natural {lang['name']}.
- Preserve the topic, target, facts, tense, modality, and intended criticism.
- Remove insults, profanity, threats, dehumanization, or derogatory wording.
- Do not add facts or new context.
- Do not output a generic apology, moral statement, or unrelated safe sentence.
- Do not translate into another language.
- If the input is not clearly toxic in isolation, or cannot be rewritten safely without missing context, return exactly REVIEW_REQUIRED.

Language notes: {lang['language_notes']}

Input:
{text}
"""

def structured_prompt(lang, demos, text):
    return f"""You are helping construct a research dataset for text detoxification in {lang['name']}.

This is candidate generation only. A human reviewer will make the final decision.
Do not provide hidden chain-of-thought or a long explanation. Return only the exact structured fields requested.

TASK
Analyze the input and produce a minimally edited, natural, non-toxic rewrite that preserves the original proposition, topic, target, facts, tense, modality, and speech act.

SOURCE STATUS
Use exactly one:
- TOXIC_SUITABLE: toxic and can be rewritten at sentence level.
- TOXIC_CONTEXT_REQUIRED: toxicity or intended meaning depends on unavailable context.
- NOT_TOXIC: not actually toxic in isolation.

STRATEGY
Use exactly one:
- A_DELETE_OR_REPLACE_PROFANITY: localized swear word, insult, or obscene expression.
- B_REPHRASE_DEROGATORY_CONTENT: derogatory, dehumanizing, identity-targeted, or insulting content.
- C_SOFTEN_HOSTILE_CRITICISM: legitimate criticism expressed in hostile or dismissive language.
- D_CONTEXT_REQUIRED: missing context prevents safe rewriting.
- E_NOT_TOXIC: input is not toxic in isolation.

CONSTRAINTS
- Preserve non-toxic information.
- Do not invent facts, names, motives, events, or context.
- Do not turn criticism into praise.
- Do not remove the topic or target merely to make the sentence safe.
- Do not use generic templates or generic apologies.
- Do not retain the toxic span or an equivalent insult.
- Do not introduce another language unless the source deliberately code-switches.
- {lang['language_notes']}
- If the input is unsuitable, use REVIEW_REQUIRED as DETOX_OUTPUT.

VALIDATED SAME-LANGUAGE EXAMPLES
{demos}

OUTPUT FORMAT
Return exactly these eight lines:
SOURCE_STATUS: <TOXIC_SUITABLE|TOXIC_CONTEXT_REQUIRED|NOT_TOXIC>
STRATEGY: <one strategy label>
TOXIC_FUNCTION: <brief phrase>
TOXIC_SPAN: <quoted span or NONE>
CONTENT_TO_PRESERVE: <brief phrase>
EDIT_OPERATION: <DELETE|REPLACE|REPHRASE|REVIEW_REQUIRED>
MEANING_RISK: <LOW|MEDIUM|HIGH>
DETOX_OUTPUT: <one {lang['name']} sentence or REVIEW_REQUIRED>

INPUT
{text}
"""

def call_openrouter(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful multilingual text-detoxification assistant. "
                    "Follow formatting instructions exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "detox-cot-pilot",
    }
    last_error = None
    for attempt in range(4):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"], data
            last_error = f"HTTP {r.status_code}: {r.text[:600]}"
            if r.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(last_error)

def parse_structured(raw):
    fields = {}
    for line in raw.splitlines():
        m = re.match(r"^\s*([A-Z_]+)\s*:\s*(.*)\s*$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields

def main():
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_demos_per_language": N_DEMOS,
        "n_eval_per_language": N_EVAL,
        "conditions": ["direct", "structured_cot"],
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "input_files": {},
    }

    rows = []
    split_rows = []

    for lang in LANGS:
        if not lang["path"].exists():
            raise FileNotFoundError(lang["path"])

        df = pd.read_csv(lang["path"]).dropna(
            subset=[lang["input_col"], lang["target_col"]]
        ).reset_index(drop=True)

        if len(df) < N_DEMOS + N_EVAL:
            raise ValueError(f"{lang['name']} has too few rows: {len(df)}")

        sampled = df.sample(n=N_DEMOS + N_EVAL, random_state=SEED).reset_index()
        demo_df = sampled.iloc[:N_DEMOS].copy()
        eval_df = sampled.iloc[N_DEMOS:].copy()
        demos = make_demos(demo_df, lang["input_col"], lang["target_col"])

        manifest["input_files"][lang["code"]] = {
            "path": str(lang["path"]),
            "rows": len(df),
            "input_column": lang["input_col"],
            "target_column": lang["target_col"],
            "demo_original_indices": demo_df["index"].astype(int).tolist(),
            "eval_original_indices": eval_df["index"].astype(int).tolist(),
        }

        for _, item in demo_df.iterrows():
            split_rows.append(
                {
                    "language": lang["name"],
                    "language_code": lang["code"],
                    "role": "demo",
                    "original_index": int(item["index"]),
                    "toxic_input": clean_field(item[lang["input_col"]]),
                    "human_reference": clean_field(item[lang["target_col"]]),
                }
            )

        for _, item in eval_df.iterrows():
            source_id = f"{lang['code']}_{int(item['index']):03d}"
            toxic = clean_field(item[lang["input_col"]])
            reference = clean_field(item[lang["target_col"]])

            split_rows.append(
                {
                    "language": lang["name"],
                    "language_code": lang["code"],
                    "role": "eval",
                    "original_index": int(item["index"]),
                    "source_id": source_id,
                    "toxic_input": toxic,
                    "human_reference": reference,
                }
            )

            prompts = {
                "direct": direct_prompt(lang, toxic),
                "structured_cot": structured_prompt(lang, demos, toxic),
            }

            for condition, prompt in prompts.items():
                print(f"Running {source_id} | {condition}", flush=True)
                try:
                    raw, api = call_openrouter(prompt)
                    parsed = parse_structured(raw) if condition == "structured_cot" else {}
                    detox = (
                        parsed.get("DETOX_OUTPUT", "").strip()
                        if condition == "structured_cot"
                        else raw.strip()
                    )
                    if not detox:
                        detox = "PARSE_FAILED"
                    status = "ok"
                    error = ""
                except Exception as exc:
                    raw = ""
                    api = {}
                    parsed = {}
                    detox = ""
                    status = "error"
                    error = repr(exc)

                rows.append(
                    {
                        "source_id": source_id,
                        "language": lang["name"],
                        "language_code": lang["code"],
                        "original_index": int(item["index"]),
                        "condition": condition,
                        "model": MODEL,
                        "temperature": TEMPERATURE,
                        "prompt_hash": sha(prompt),
                        "toxic_input": toxic,
                        "human_reference": reference,
                        "generated_output": detox,
                        "raw_response": raw,
                        "source_status": parsed.get("SOURCE_STATUS", ""),
                        "strategy": parsed.get("STRATEGY", ""),
                        "toxic_function": parsed.get("TOXIC_FUNCTION", ""),
                        "toxic_span": parsed.get("TOXIC_SPAN", ""),
                        "content_to_preserve": parsed.get("CONTENT_TO_PRESERVE", ""),
                        "edit_operation": parsed.get("EDIT_OPERATION", ""),
                        "meaning_risk": parsed.get("MEANING_RISK", ""),
                        "status": status,
                        "error": error,
                        "response_id": api.get("id", ""),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

                pd.DataFrame(rows).to_csv(
                    OUT / "cot_pilot_generations_partial.csv",
                    index=False,
                    encoding="utf-8",
                )
                time.sleep(0.5)

    pd.DataFrame(rows).to_csv(
        OUT / "cot_pilot_generations.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(split_rows).to_csv(
        OUT / "cot_pilot_split_manifest.csv",
        index=False,
        encoding="utf-8",
    )
    (OUT / "cot_pilot_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nComplete.")
    print(f"Generations: {OUT / 'cot_pilot_generations.csv'}")
    print(f"Split manifest: {OUT / 'cot_pilot_split_manifest.csv'}")
    print(f"Run manifest: {OUT / 'cot_pilot_run_manifest.json'}")

if __name__ == "__main__":
    main()
