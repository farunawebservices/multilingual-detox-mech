import os
import json
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-5"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

prompt = """Return valid JSON only:

{
  "status": "ok",
  "language": "Yorùbá",
  "rewrite": "example"
}

Task: Return the JSON object exactly. Do not add commentary.
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.0,
    "max_tokens": 100,
    "response_format": {"type": "json_object"},
}

headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/farunawebservices/multilingual-detox-mech",
    "X-Title": "gpt5-response-debug",
}

response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

print("HTTP:", response.status_code)
body = response.json()

print("\nTop-level keys:")
print(list(body.keys()))

print("\nFull first choice:")
print(json.dumps(body.get("choices", [None])[0], indent=2, ensure_ascii=False))
