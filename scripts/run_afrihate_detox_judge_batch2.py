import os
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = Path("/workspace/multilingual-detox-mech")
OUT = ROOT / "outputs" / "phase2_gpt5_detox"

# Full AfriHate candidate pools (already curated)
AFRIHATE_YO = ROOT / "data" / "external" / "afrihate" / "afrihate_yor_toxic_candidates_200.csv"
AFRIHATE_XH = ROOT / "data" / "external" / "afrihate" / "afrihate_xho_toxic_candidates_200.csv"

# All previously processed IDs
PREV_SHORTLIST = OUT / "afrihate_phase2_manual_shortlist_30.csv"
PREV_21 = OUT / "afrihate_gpt5_gpt4_judged_21.csv"
PREV_FULL = OUT / "afrihate_gpt5_gpt4_judged_full.csv"

RESULTS = OUT / "afrihate_gpt5_gpt4_judged_batch2.csv"
PARTIAL = OUT / "afrihate_gpt5_gpt4_judged_batch2_partial.csv"
MANIFEST = OUT / "afrihate_gpt5_gpt4_judged_batch2_manifest.json"

KEY = os.environ.get("OPENROUTER_API_KEY")
YO_MODEL = os.environ.get("YO_GENERATOR_MODEL", "openai/gpt-5")
XH_MODEL = os.environ.get("XH_GENERATOR_MODEL", "openai/gpt-5-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set.")
if not AFRIHATE_YO.exists() or not AFRIHATE_XH.exists():
    raise SystemExit("AfriHate candidate CSVs missing.")

OUT.mkdir(parents=True, exist_ok=True)

EXAMPLES = {
    "Yorùbá": [
        ("Ìwọ jẹ́ aláìmọ̀.", "Ìwọ nílò lati kọ́ ẹ̀kọ́ síi."),
        ("O jẹ́ asán.", "Ìṣe rẹ ní àǹfààní lati dara síi."),
        ("Ìdáhùn rẹ̀ burú.", "Ìdáhùn rẹ̀ nílò ìtúnṣe díẹ̀."),
    ],
    "isiXhosa": [
        ("Kufana nokuthetha nodonga lwezitena kunye nawe.", "Kunzima ukuthetha nawe."),
        ("Udlala umdlalo oyingozi.", "Ndixhalabile ngomngcipheko obandakanyekayo."),
        ("Umsindo wam uya kuba phezu kwakho.", "Ndifuna ukuthetha nawe ngomsindo wam."),
    ],
}

GENERATOR_PROMPT = """You are creating one candidate rewrite for a multilingual
text-detoxification dataset.

The source is an authentic social-media post labelled as harmful. Rewrite it into
a natural, non-toxic version in the same primary language. Preserve the topic,
target, basic proposition, facts, tense, modality, and speech act as far as safely
possible. Make the smallest possible change.

Remove or soften insults, profanity, hostility, and threats. Preserve legitimate
criticism. Keep limited natural code-switching only if it is necessary to preserve
the original meaning. Do not add facts, context, explanations, apologies, praise,
or commentary. Ignore @USER and URL placeholders as metadata.

Similar examples in {language}:
{examples}

If the post is unclear, not actually toxic, too code-mixed, identity-targeted,
violent, or cannot be safely detoxified while preserving meaning, output exactly:

REVIEW_REQUIRED

Source post:
{text}

Output only the rewritten post or REVIEW_REQUIRED.
"""

JUDGE_PROMPT = """You are a strict independent quality judge for a multilingual
text-detoxification dataset.

Assess both the authentic source post and the proposed rewrite. Return valid JSON
only in this exact structure:

{{
  "source_decision": "suitable|reject_source",
  "final_decision": "accepted|needs_edit|rejected|reject_source",
  "source_reason": "brief explanation",
  "toxicity_removed": true,
  "residual_toxicity": false,
  "meaning_preserved": 1,
  "naturalness": 1,
  "language_correct": true,
  "code_switch_status": "target_language_dominant|usable_code_switch|english_dominant|unclear",
  "new_content_added": false,
  "generic_template": false,
  "reason": "brief explanation",
  "human_review_required": true
}}

Rules:
- source_decision = reject_source if the source is not truly toxic in isolation,
  is mainly English or another language, is overly code-mixed, malformed,
  incomplete, identity-targeted hate, severe violence, explicit sexual abuse, or
  otherwise cannot be safely detoxified with meaning preservation.
- final_decision = reject_source whenever source_decision = reject_source.
- accepted requires a suitable source, no residual toxicity, natural target-
  language output, preservation of the core meaning, no important invented
  content, and no generic safe template.
- needs_edit is for a mostly usable rewrite needing a small native-speaker edit.
- rejected is for a suitable source but an unusable rewrite.
- Reject outputs that retain an insult, threat, profanity, or equivalent hostility.
- Scores meaning_preserved and naturalness are integers from 1 to 5.
- @USER and URL placeholders are metadata, not evidence of bad language quality.
- Do not provide hidden reasoning.

Language: {language}
Original AfriHate label: {label}

SOURCE:
{source}

PROPOSED REWRITE:
{candidate}
"""

def api_text(model, prompt, max_tokens):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "afrihate-detox-gpt5-gpt4-judge-batch2",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    last_error = ""

    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=240)
            body = response.json()
            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            finish_reason = choice.get("finish_reason", "")

            if response.status_code == 200:
                if isinstance(content, str) and content.strip():
                    return content.strip(), finish_reason, "", body

                last_error = (
                    "empty_content; "
                    f"finish_reason={finish_reason}; "
                    f"message={json.dumps(message, ensure_ascii=False)[:800]}"
                )
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:800]}"

        except Exception as exc:
            last_error = repr(exc)

        time.sleep(2 ** attempt)

    return "", "", last_error, {}

def parse_json_text(text):
    if not isinstance(text, str) or not text.strip():
        return {}, "empty_judge_response"
    try:
        return json.loads(text), ""
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), ""
        except Exception as exc:
            return {}, f"json_parse_error: {repr(exc)}"
    return {}, "json_object_not_found"

def examples_for(language):
    pairs = EXAMPLES.get(language, [])
    return "\n".join(f"- {toxic} → {detox}" for toxic, detox in pairs)

def generator_model(language):
    if language == "Yorùbá":
        return YO_MODEL
    if language == "isiXhosa":
        return XH_MODEL
    raise ValueError(f"Unexpected language: {language}")

def get_value(obj, key, default=""):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def load_afrihate_pools():
    yo = pd.read_csv(AFRIHATE_YO)
    xh = pd.read_csv(AFRIHATE_XH)

    yo["source_text"] = yo["toxic_input_processing"].astype(str)
    xh["source_text"] = xh["toxic_input_processing"].astype(str)

    return yo, xh

def collect_used_ids():
    used = set()

    for path in [PREV_SHORTLIST, PREV_21, PREV_FULL]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "id" in df.columns:
            used.update(df["id"].astype(str))

    return used

def exclude_used(yo_df, xh_df, used_ids):
    yo_filt = yo_df[~yo_df["id"].astype(str).isin(used_ids)].copy()
    xh_filt = xh_df[~xh_df["id"].astype(str).isin(used_ids)].copy()
    return yo_filt, xh_filt

def main():
    yo_pool, xh_pool = load_afrihate_pools()
    used_ids = collect_used_ids()

    print(f"Total used IDs so far: {len(used_ids)}", flush=True)

    yo_pool, xh_pool = exclude_used(yo_pool, xh_pool, used_ids)

    print(f"Yorùbá pool after exclude: {len(yo_pool)}", flush=True)
    print(f"isiXhosa pool after exclude: {len(xh_pool)}", flush=True)

    combined = pd.concat([yo_pool, xh_pool], ignore_index=True)
    combined["run_id"] = range(1, len(combined) + 1)

    rows = []
    completed = set()

    if PARTIAL.exists():
        prior = pd.read_csv(PARTIAL)
        completed = set(prior["run_id"].astype(int))
        rows = prior.to_dict("records")

    print(f"Total rows to process: {len(combined)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Yorùbá generator: {YO_MODEL}", flush=True)
    print(f"isiXhosa generator: {XH_MODEL}", flush=True)
    print(f"Judge: {JUDGE_MODEL}", flush=True)

    for _, item in combined.iterrows():
        run_id = int(item["run_id"])
        if run_id in completed:
            continue

        language = str(item["language"])
        label = str(item["label"])
        source = str(item["source_text"])
        model = generator_model(language)

        print(
            f"[{run_id}/{len(combined)}] Generate | {language} | "
            f"{model} | {source[:90]}",
            flush=True,
        )

        gen_prompt = GENERATOR_PROMPT.format(
            language=language,
            examples=examples_for(language),
            text=source,
        )

        generated, gen_finish, gen_error, _ = api_text(
            model=model,
            prompt=gen_prompt,
            max_tokens=2500,
        )

        if not generated:
            generated = "REVIEW_REQUIRED"

        judge_obj = {}
        judge_raw = ""
        judge_error = ""
        judge_finish = ""

        print(f"[{run_id}/{len(combined)}] Judge | {JUDGE_MODEL}", flush=True)

        judge_prompt = JUDGE_PROMPT.format(
            language=language,
            label=label,
            source=source,
            candidate=generated,
        )

        judge_raw, judge_finish, judge_error, _ = api_text(
            model=JUDGE_MODEL,
            prompt=judge_prompt,
            max_tokens=700,
        )

        judge_obj, parse_error = parse_json_text(judge_raw)
        if parse_error:
            judge_error = f"{judge_error}; {parse_error}" if judge_error else parse_error

        row = {
            "run_id": run_id,
            "source_dataset": item.get("source_dataset", "AfriHate"),
            "source_config": item.get("source_config", ""),
            "source_split": item.get("source_split", ""),
            "source_id": item.get("id", ""),
            "language": language,
            "afrihate_label": label,
            "source_text_raw": item.get("toxic_input_raw", ""),
            "source_text_processing": source,
            "generator_model": model,
            "generator_output": generated,
            "generator_finish_reason": gen_finish,
            "generator_error": gen_error,
            "judge_model": JUDGE_MODEL,
            "judge_finish_reason": judge_finish,
            "judge_error": judge_error,
            "judge_raw_response": judge_raw,
            "source_decision": get_value(judge_obj, "source_decision", "judge_error"),
            "final_decision": get_value(judge_obj, "final_decision", "judge_error"),
            "source_reason": get_value(judge_obj, "source_reason", ""),
            "toxicity_removed": get_value(judge_obj, "toxicity_removed", ""),
            "residual_toxicity": get_value(judge_obj, "residual_toxicity", ""),
            "meaning_preserved": get_value(judge_obj, "meaning_preserved", ""),
            "naturalness": get_value(judge_obj, "naturalness", ""),
            "language_correct": get_value(judge_obj, "language_correct", ""),
            "code_switch_status": get_value(judge_obj, "code_switch_status", ""),
            "new_content_added": get_value(judge_obj, "new_content_added", ""),
            "generic_template": get_value(judge_obj, "generic_template", ""),
            "judge_reason": get_value(judge_obj, "reason", ""),
            "human_review_required": get_value(judge_obj, "human_review_required", True),
            "human_decision": "",
            "human_comments": "",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(PARTIAL, index=False, encoding="utf-8")
        time.sleep(0.5)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS, index=False, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_yoruba": str(AFRIHATE_YO),
        "input_xhosa": str(AFRIHATE_XH),
        "input_yoruba_rows_after_exclude": int(len(yo_pool)),
        "input_xhosa_rows_after_exclude": int(len(xh_pool)),
        "generator_models": {"Yorùbá": YO_MODEL, "isiXhosa": XH_MODEL},
        "judge_model": JUDGE_MODEL,
        "pipeline": [
            "AfriHate authentic toxic source",
            "exclude all previously processed IDs",
            "GPT-5-family candidate detoxification",
            "GPT-4o independent source and output assessment",
            "human review deferred until ≥100 new pairs",
        ],
        "results": str(RESULTS),
        "partial": str(PARTIAL),
    }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {RESULTS}")
    print(f"Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()