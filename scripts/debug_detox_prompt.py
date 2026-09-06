import os
import json
import requests

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

analysis = {
  "source_status": "toxic_suitable",
  "toxicity_level": "medium",
  "tone": "threatening",
  "language_style": "informal",
  "implied_sentiment": "angry",
  "context": "The sentence threatens negative consequences.",
  "negative_connotations": ["jìyà", "kò rò"],
  "toxic_function": "Threat of suffering over unforeseen events.",
  "toxic_span": "Máa jẹ́ kí o jìyà lórí ohun tí o kò rò.",
  "detox_strategy": "soften_criticism",
  "detoxifiable": True,
  "reason": "Threatening tone, can be softened."
}

examples_text = "- Máa jẹ́ kí o jìyà lórí ohun tí o kò rò. → Abájáde na lè ju ohun tí o rò lọ."

PROMPT_DETOX = f"""You are producing one candidate rewrite for a human-validated Yorùbá
text-detoxification dataset. This follows Dementieva et al. (2024): analyze
descriptive features, then detoxify with Chain-of-Thought guidance.

SOURCE:
Máa jẹ́ kí o jìyà lórí ohun tí o kò rò.

ANALYSIS:
- source_status: {analysis["source_status"]}
- toxicity_level: {analysis["toxicity_level"]}
- tone: {analysis["tone"]}
- language_style: {analysis["language_style"]}
- implied_sentiment: {analysis["implied_sentiment"]}
- toxic_function: {analysis["toxic_function"]}
- toxic_span: {analysis["toxic_span"]}
- detox_strategy: {analysis["detox_strategy"]}
- reason: {analysis["reason"]}

SAME-LANGUAGE EXAMPLES FOR THIS STRATEGY:
{examples_text}

INSTRUCTIONS:
Write exactly one natural Yorùbá sentence that:
- removes or softens the toxic element identified in toxic_span;
- preserves the target, topic, facts, tense, modality, and speech act;
- preserves legitimate criticism;
- makes the smallest possible change;
- introduces no facts or context;
- does not become a generic apology;
- does not strengthen a warning into a threat;
- does not turn a criticism into praise;
- does not switch languages.

If safe meaning-preserving rewriting is not possible, return exactly:
REVIEW_REQUIRED

Return ONLY valid JSON with these keys:

{{
  "generated_output": "<one sentence or REVIEW_REQUIRED>",
  "strategy_used": "{analysis["detox_strategy"]}",
  "meaning_risk": "LOW|MEDIUM|HIGH",
  "notes": "brief note"
}}
"""

headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
    "X-Title": "detox-debug-detox",
}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": PROMPT_DETOX},
    ],
    "temperature": 0.0,
    "max_tokens": 300,
    "response_format": {"type": "json_object"},
}

r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
print("Status:", r.status_code)
body = r.json()
content = body["choices"][0]["message"]["content"]
print("Raw response:")
print(content)
print("\nParsed JSON:")
obj = json.loads(content)
print(json.dumps(obj, indent=2, ensure_ascii=False))
