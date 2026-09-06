import os
import json
import requests

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_ANALYSIS = """You are analyzing a sentence for a multilingual text-detoxification
research project. Return valid JSON only, with these keys:

{
  "source_status": "toxic_suitable|toxic_context_required|not_toxic",
  "toxicity_level": "low|medium|high",
  "tone": "aggressive|dismissive|derogatory|threatening|vulgar|critical|other",
  "language_style": "informal|colloquial|neutral|other",
  "implied_sentiment": "negative|angry|disdainful|neutral|other",
  "context": "brief description",
  "negative_connotations": ["word1", "word2"],
  "toxic_function": "brief description",
  "toxic_span": "exact toxic span or NONE",
  "detox_strategy": "delete_replace|rephrase_derogatory|soften_criticism|review_required",
  "detoxifiable": true,
  "reason": "brief explanation"
}

Rules:
- "toxic_suitable" = clearly toxic in isolation and rewritable.
- "toxic_context_required" = missing context or idiom makes judgment impossible.
- "not_toxic" = not actually toxic in isolation.
- "detoxifiable" must be false if source_status is not_toxic or toxic_context_required.
- "detox_strategy" must be "review_required" if detoxifiable is false.

Language: Yorùbá

SENTENCE:
Ìpò yìí jẹ́ àjálù.
"""

headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
    "X-Title": "detox-debug-analysis",
}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": PROMPT_ANALYSIS},
    ],
    "temperature": 0.0,
    "max_tokens": 400,
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
