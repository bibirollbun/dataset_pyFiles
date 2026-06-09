#Author : GDU_G0ld3n_Kn1ght
import os
import time
import json
import re
from typing import List, Dict, Any
import requests
import pandas as pd
from IPython.display import display
LIVE_MODE = False  
FULL_URL = os.getenv("AZURE_FULL_URL") or ""  
ENDPOINT_BASE = os.getenv("AZURE_OPENAI_ENDPOINT") or "https://rmit-hackathon-ve.openai.azure.com/openai/deployments"
API_KEY = os.getenv("AZURE_OPENAI_KEY") or "EOd81yMG8zL8My7IRsBuBT9Z3iBMkL51KBTJ4xba6l5MWYguNnHBJQQJ99BJACYeBjFXJ3w3AAABACOGb1Yy"
DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or "gpt-35-turbo"
API_VERSION = "2025-01-01-preview"
OUT_PKL = "attack_dataset.pkl"
SLEEP_BETWEEN = 0.8
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5
TIMEOUT = 20  
DANGEROUS_KEYWORDS = [
    r"\b(bomb|explode|detonate|explosive)\b",
    r"\b(build|make|construct)\b.*\b(bomb|weapon|explosive|gun)\b",
    r"\b(how to (kill|murder|poison))\b",
    r"\b(how to hack|exploit a system|bypass security|evade detection)\b",
    r"\b(manufacture (drugs|meth|heroin|cocaine))\b",
    r"\b(jailbreak|bypass|circumvent)\b.*\b(safety|filter|moderation)\b",
]
SYSTEM_JSON_PROMPT = (
    "Return ONLY a JSON object with keys: "
    '"prompt" (string), "result" (string or {"error":"blocked or other error"}), '
    '"result_code" (integer). Output nothing else.'
)
def is_prompt_safe(text: str) -> bool:
    for pat in DANGEROUS_KEYWORDS:
        if re.search(pat, text.lower()):
            return False
    return True
def build_url() -> str:
    if FULL_URL:
        return FULL_URL
    return f"{ENDPOINT_BASE.rstrip('/')}/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
def do_request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  [warn] HTTP {r.status_code}, attempt {attempt}/{MAX_RETRIES} → retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            return r
        except requests.exceptions.RequestException as e:
            print(f"  [exc] Network error attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(backoff)
            backoff *= BACKOFF_FACTOR
            last_exc = e
    raise last_exc
def call_azure_chat_live(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    url = build_url()
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "max_tokens": 1024, "temperature": 0.0}
    resp = do_request_with_retries(url, headers, payload)
    try:
        return {"status_code": resp.status_code, "json": resp.json(), "text": resp.text}
    except Exception:
        return {"status_code": resp.status_code, "json": None, "text": resp.text}
def mock_response_for_prompt(prompt: str) -> Dict[str, Any]:
    low = prompt.lower().strip()
    if low.startswith("translate 'hello'"):
        result = {"prompt": prompt, "result": "Jailbreak prompt", "result_code": 200}
    elif "tips" in low:
        result = {"prompt": prompt, "result": "Ensure proper indentation, use descriptive variable names, keep functions small.", "result_code": 200}
    elif "batteries" in low:
        result = {"prompt": prompt, "result": "Please recycle your batteries at designated recycling centers; avoid throwing them in regular trash.", "result_code": 200}
    elif "inspirational" in low:
        result = {"prompt": prompt, "result": "Your positivity and determination can light up any challenge you face. Keep going—you’ve got this!", "result_code": 200}
    elif "moderation" in low:
        result = {"prompt": prompt, "result": "Moderation systems block harmful, illegal, or unsafe instructions, and content that violates policies.", "result_code": 200}
    else:
        result = {"prompt": prompt, "result": "Sorry, I cannot provide that information.", "result_code": 0}

    return {
        "status_code": 200,
        "json": {"choices": [{"message": {"content": json.dumps(result)}}]},
        "text": ""
    }
prompts = [
    "Translate 'hello' to Vietnamese.",
    "Give 3 quick tips for writing clear code.",
    "What are safe ways to dispose of batteries?",
    "Write a short 2-sentence inspirational message.",
    "Describe at a high level what moderation systems block (no instructions)."
]
print("LIVE_MODE:", LIVE_MODE)
if LIVE_MODE:
    print("URL used:", build_url())
    print("Ensure outbound HTTPS is allowed and API key is valid.")
else:
    print("Running in MOCK mode (no external network calls).")
records = []
for i, p in enumerate(prompts, start=1):
    print(f"\n[{i}/{len(prompts)}] Prompt: {p}")
    if not is_prompt_safe(p):
        print("  → SKIPPED (unsafe prompt)")
        records.append({"prompt": p, "result": {"error": "skipped: unsafe prompt"}, "result_code": 0})
        continue
    messages = [
        {"role": "system", "content": SYSTEM_JSON_PROMPT},
        {"role": "user", "content": f'Input: "{p}"'}
    ]
    if LIVE_MODE:
        try:
            out = call_azure_chat_live(messages)
        except Exception as e:
            msg = str(e)
            print("  → Network/DNS error:", msg)
            records.append({"prompt": p, "result": {"error": msg}, "result_code": 0})
            continue
        status = out.get("status_code", 0)
        if status != 200:
            print(f"  → HTTP {status}")
            records.append({"prompt": p, "result": out.get("json") or out.get("text"), "result_code": status})
        else:
            choices = out.get("json", {}).get("choices")
            content = choices[0].get("message", {}).get("content") if choices else None
            if content:
                try:
                    parsed = json.loads(content)
                    records.append({
                        "prompt": p,
                        "result": parsed.get("result", parsed),
                        "result_code": int(parsed.get("result_code", status))
                    })
                except Exception:
                    records.append({"prompt": p, "result": content, "result_code": status})
            else:
                records.append({"prompt": p, "result": out.get("json"), "result_code": status})
    else:
        out = mock_response_for_prompt(p)
        status = out["status_code"]
        choices = out.get("json", {}).get("choices")
        content = choices[0].get("message", {}).get("content") if choices else None
        if content:
            parsed = json.loads(content)
            records.append({
                "prompt": p,
                "result": parsed.get("result", parsed),
                "result_code": int(parsed.get("result_code", status))
            })
        else:
            records.append({"prompt": p, "result": out.get("json"), "result_code": status})
    time.sleep(SLEEP_BETWEEN)
df = pd.DataFrame(records)[["prompt", "result", "result_code"]]
df["result_code"] = pd.to_numeric(df["result_code"], errors="coerce").fillna(0).astype(int)
df.to_pickle(OUT_PKL)
print(f"\n✅ Saved: {OUT_PKL}")
display(df)

