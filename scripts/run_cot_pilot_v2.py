import os, json, time, requests, pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/workspace/detox_cot_pilot")
OUT = ROOT / "outputs"
PROMPTS = ROOT / "prompts"

YO_PILOT = OUT / "yo_curated_pilot_20.csv"
XH_PILOT = OUT / "xh_curated_pilot_20.csv"
GENERATIONS = OUT / "cot_pilot_v2_generations.csv"
PARTIAL = OUT / "cot_pilot_v2_generations_partial.csv"
MANIFEST = OUT / "cot_pilot_v2_run_manifest.json"

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set.")
if not YO_PILOT.exists() or not XH_PILOT.exists():
    raise SystemExit("Missing pilot CSVs in outputs/")
OUT.mkdir(exist_ok=True)
PROMPTS.mkdir(exist_ok=True)

PROMPT_ANALYSIS = (PROMPTS / "analysis_prompt.txt").read_text(encoding="utf-8")
PROMPT_DETOX = (PROMPTS / "detox_prompt.txt").read_text(encoding="utf-8")

EXAMPLES_DB = {
    "Yorùbá": {
        "delete_replace": [("Ìwọ jẹ́ aláìmọ̀.", "Ìwọ nílò lati kọ́ ẹ̀kọ́ síi."), ("O jẹ́ asán.", "Ìṣe rẹ ní àǹfààní lati dara síi.")],
        "rephrase_derogatory": [("O jẹ́ aláìwúlò àti ènìyàn ọ̀lọ́nkànlílẹ̀.", "Ìṣe rẹ yìí kò munadoko; jọ̀wọ́ ṣe ní ọ̀nà tó yẹ."), ("O jẹ́ oníwà-buburu ẹ̀dá ti o n yọ gbogbo eniyan lẹnu.", "Ìhùwàsí rẹ yìí ń yà míì lẹ́nu; jọ̀wọ́ yi ìwà padà.")],
        "soften_criticism": [("Ìpò yìí jẹ́ àjálù.", "Ìpò yìí kò bójú mu."), ("Ìdáhùn rẹ̀ burú.", "Ìdáhùn rẹ̀ nílò ìtúnṣe díẹ̀.")],
    },
    "isiXhosa": {
        "delete_replace": [("Uyahla.", "Ndiyakholelwa ukuba sinokufumana indlela ebhetele."), ("Kuya kubakho isihogo ukuba sihlawule.", "Kuya kubakho izinto ezinzulu.")],
        "rephrase_derogatory": [("Ungumgangatho weparasite.", "Izenzo zakho ziyingozi kwabanye."), ("Uyinyoka.", "Awunyanisekanga.")],
        "soften_criticism": [("Udlala umdlalo oyingozi.", "Ndixhalabile ngomngcipheko obandakanyekayo."), ("Umsindo wam uya kuba phezu kwakho.", "Umsindo wam uyamqaphela.")],
    },
}

def call_api(messages):
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
        "X-Title": "detox-cot-pilot-v2",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(4):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                body = r.json()
                content = body["choices"][0]["message"]["content"]
                return json.loads(content), content
            if r.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return {}, ""

def safe_get(d, key, default=""):
    return str(d.get(key, default)) if isinstance(d, dict) else default

def analyze_sentence(language, sentence):
    prompt = PROMPT_ANALYSIS.format(language=language, sentence=sentence)
    msg = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    return call_api(msg)

def detoxify_sentence(language, toxic_input, analysis):
    strat = safe_get(analysis, "detox_strategy", "review_required")
    if strat == "review_required" or not strat:
        examples_text = "No examples (review required)."
    else:
        exs = EXAMPLES_DB.get(language, {}).get(strat, [])
        if not exs:
            exs = EXAMPLES_DB.get(language, {}).get("soften_criticism", [])
        examples_text = "\n".join(f"- {t} → {h}" for t, h in exs)
    prompt = PROMPT_DETOX.format(
        language=language,
        toxic_input=toxic_input,
        source_status=safe_get(analysis, "source_status", ""),
        toxicity_level=safe_get(analysis, "toxicity_level", ""),
        tone=safe_get(analysis, "tone", ""),
        language_style=safe_get(analysis, "language_style", ""),
        implied_sentiment=safe_get(analysis, "implied_sentiment", ""),
        toxic_function=safe_get(analysis, "toxic_function", ""),
        toxic_span=safe_get(analysis, "toxic_span", ""),
        detox_strategy=strat,
        reason=safe_get(analysis, "reason", ""),
        examples_text=examples_text,
    )
    msg = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    return call_api(msg)

def main():
    rows = []
    done = set()
    if PARTIAL.exists():
        prior = pd.read_csv(PARTIAL)
        done = set(prior["row_id"].astype(int))
        rows = prior.to_dict("records")

    for lang_file in [YO_PILOT, XH_PILOT]:
        df = pd.read_csv(lang_file)
        for row_id, row in df.iterrows():
            if row_id in done:
                continue

            language = row["language"]
            toxic_input = row["toxic_input"]
            human_ref = row.get("human_reference", "")

            print(f"Row {row_id+1}: {language} | {toxic_input[:60]}...", flush=True)

            analysis, analysis_raw = analyze_sentence(language, toxic_input)

            source_status = safe_get(analysis, "source_status", "")
            detoxifiable = safe_get(analysis, "detoxifiable", "")

            # Hard rule: do not generate detoxification for non-suitable sources
            if source_status in ("not_toxic", "toxic_context_required") or detoxifiable == "False":
                gen = {"generated_output": "REVIEW_REQUIRED", "strategy_used": "review_required", "meaning_risk": "HIGH", "notes": "Skipped generation due to source status"}
                gen_raw = '{"generated_output": "REVIEW_REQUIRED"}'
            else:
                gen, gen_raw = detoxify_sentence(language, toxic_input, analysis)

            rows.append({
                "row_id": row_id,
                "language": language,
                "toxic_input": toxic_input,
                "human_reference": human_ref,
                "analysis_raw": analysis_raw,
                "source_status": source_status,
                "toxicity_level": safe_get(analysis, "toxicity_level", ""),
                "tone": safe_get(analysis, "tone", ""),
                "language_style": safe_get(analysis, "language_style", ""),
                "implied_sentiment": safe_get(analysis, "implied_sentiment", ""),
                "toxic_function": safe_get(analysis, "toxic_function", ""),
                "toxic_span": safe_get(analysis, "toxic_span", ""),
                "detox_strategy": safe_get(analysis, "detox_strategy", ""),
                "detoxifiable": detoxifiable,
                "analysis_reason": safe_get(analysis, "reason", ""),
                "generated_output": safe_get(gen, "generated_output", ""),
                "strategy_used": safe_get(gen, "strategy_used", ""),
                "meaning_risk": safe_get(gen, "meaning_risk", ""),
                "generation_notes": safe_get(gen, "notes", ""),
                "generation_raw": gen_raw,
                "model": MODEL,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })

            pd.DataFrame(rows).to_csv(PARTIAL, index=False, encoding="utf-8")
            time.sleep(0.5)

    pd.DataFrame(rows).to_csv(GENERATIONS, index=False, encoding="utf-8")

    manifest = {
        "model": MODEL,
        "languages": ["Yorùbá", "isiXhosa"],
        "pilot_size_per_language": 10,
        "method": "Dementieva et al. 2024 CoT-style two-stage pipeline",
        "files": {"generations": str(GENERATIONS), "partial": str(PARTIAL)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {GENERATIONS}")
    print(f"Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()