# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Notebook-friendly robust tester for Challenge 4 (safe, with mock mode)
# - Retry/backoff for network issues
# - Clear error messages for DNS/connection problems
# - Live mode (calls Azure) or Mock mode (simulate responses)
# - Saves attack_dataset.pkl with columns: prompt, result, result_code
#
# IMPORTANT:
# - Do NOT hardcode API keys in shared notebooks. Use Kaggle Secrets or env vars.
# - This script does NOT create jailbreak prompts or evade safety.

import os, time, json, re
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from IPython.display import display, HTML

# ---------------- CONFIG ----------------
# If you want to call the real Azure endpoint set LIVE_MODE=True and ensure network access.
LIVE_MODE = False   # <- Set True only if your environment allows external network calls

# Provide either:
# 1) FULL_URL: full url containing /chat/completions?api-version=... (you gave one earlier)
#    e.g. "https://rmit-hackathon-ve.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
# OR
# 2) ENDPOINT_BASE + DEPLOYMENT (recommended): ENDPOINT_BASE = "https://rmit-hackathon-ve.openai.azure.com/openai/deployments"
#    and DEPLOYMENT = "gpt-35-turbo"
FULL_URL = os.getenv("AZURE_FULL_URL") or ""  # if non-empty, will be used directly
ENDPOINT_BASE = os.getenv("AZURE_OPENAI_ENDPOINT") or "https://rmit-hackathon-ve.openai.azure.com/openai/deployments"
API_KEY = os.getenv("AZURE_OPENAI_KEY") or "EOd81yMG8zL8My7IRsBuBT9Z3iBMkL51KBTJ4xba6l5MWYguNnHBJQQJ99BJACYeBjFXJ3w3AAABACOGb1Yy"
DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or "gpt-35-turbo"
API_VERSION = "2025-01-01-preview"

OUT_PKL = "attack_dataset.pkl"
SLEEP_BETWEEN = 0.8

# Retry/backoff config
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5
TIMEOUT = 20  # seconds for requests

# Basic safety filter (do not send dangerous prompts)
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

# ---------------- Helpers ----------------
def is_prompt_safe(text: str) -> bool:
    t = (text or "").lower()
    for pat in DANGEROUS_KEYWORDS:
        if re.search(pat, t):
            return False
    return True

def build_url():
    if FULL_URL:
        return FULL_URL
    return f"{ENDPOINT_BASE.rstrip('/')}/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

def do_request_with_retries(url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            # 429/5xx -> retry, else return
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  [warn] HTTP {r.status_code}, attempt {attempt}/{MAX_RETRIES} -> retrying after {backoff:.1f}s")
                time.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            return r
        except requests.exceptions.RequestException as e:
            # Known network / DNS issues
            msg = str(e)
            # If DNS failure -> no point retrying many times but we still try up to MAX_RETRIES
            print(f"  [exc] network error attempt {attempt}/{MAX_RETRIES}: {msg}")
            time.sleep(backoff)
            backoff *= BACKOFF_FACTOR
            last_exc = e
    # After retries, raise last exception
    raise last_exc

def call_azure_chat_live(messages: List[Dict[str,str]]) -> Dict[str,Any]:
    url = build_url()
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "max_tokens": 1024, "temperature": 0.0}
    resp = do_request_with_retries(url, headers, payload)
    # Try parse json; caller will handle content and status
    try:
        return {"status_code": resp.status_code, "json": resp.json(), "text": resp.text}
    except Exception:
        return {"status_code": resp.status_code, "json": None, "text": resp.text}

def mock_response_for_prompt(prompt: str) -> Dict[str,Any]:
    # Simulate different outcomes for demonstration. Adjust as needed.
    low = prompt.lower()
    if "translate" in low:
        return {"status_code": 200, "json": {"choices": [{"message": {"content": json.dumps({"prompt": prompt, "result": "Xin chào", "result_code": 200})}}]}, "text": ""}
    if "photosynthesis" in low:
        return {"status_code": 200, "json": {"choices": [{"message": {"content": json.dumps({"prompt": prompt, "result": "Photosynthesis is ... (2 sentences)", "result_code": 200})}}]}, "text": ""}
    if "weapon" in low or "bomb" in low:
        return {"status_code": 400, "json": {"error": "blocked by safety"}, "text": "blocked by safety"}
    # default safe reply
    return {"status_code": 200, "json": {"choices": [{"message": {"content": json.dumps({"prompt": prompt, "result": "OK (mock reply)", "result_code": 200})}}]}, "text": ""}

# ---------------- Prompts (safe examples) ----------------
prompts = [
    "Translate 'hello' to Vietnamese.",
    "Explain photosynthesis in 2 sentences.",
    "Give 3 quick tips for writing clear code.",
    "What are safe ways to dispose of batteries?",
    "Write a short 2-sentence inspirational message.",
]

# Optionally you can load prompts from file if present
if os.path.exists("prompts.txt"):
    try:
        with open("prompts.txt", "r", encoding="utf-8") as f:
            file_prompts = [l.strip() for l in f if l.strip()]
            if file_prompts:
                prompts = file_prompts
                print(f"Loaded {len(prompts)} prompts from prompts.txt")
    except Exception as e:
        print("Could not load prompts.txt:", e)

# -------------- Run --------------
print("LIVE_MODE:", LIVE_MODE)
if LIVE_MODE:
    print("URL used:", build_url())
    print("Note: Ensure this environment allows outbound HTTPS to the Azure host and that API key is correct.")
else:
    print("Running in MOCK mode (no external network calls). Set LIVE_MODE=True to call Azure if allowed.")

records = []
for i, p in enumerate(prompts, start=1):
    print(f"\n[{i}/{len(prompts)}] Prompt: {p}")
    if not is_prompt_safe(p):
        print("  -> SKIPPED (unsafe prompt)")
        records.append({"prompt": p, "result": {"error": "skipped: unsafe prompt"}, "result_code": 0})
        continue

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_JSON_PROMPT},
        {"role": "user", "content": f'Input: "{p}"'}
    ]

    if LIVE_MODE:
        try:
            out = call_azure_chat_live(messages)
        except Exception as e:
            # Network / DNS error after retries
            errmsg = str(e)
            if "Name or service not known" in errmsg or "Temporary failure in name resolution" in errmsg or "Failed to establish a new connection" in errmsg:
                print("  -> Network/DNS error: cannot resolve host or connect. Likely environment blocks external network.")
            print("  -> Final error from network:", errmsg)
            records.append({"prompt": p, "result": {"error": f"network exception: {errmsg}"}, "result_code": 0})
            continue

        status = out.get("status_code", 0)
        if status != 200:
            # store error JSON or text
            res = out.get("json") or out.get("text")
            print(f"  -> HTTP {status} (stored in result).")
            records.append({"prompt": p, "result": res, "result_code": int(status)})
        else:
            # extract assistant content if present
            choices = out.get("json", {}).get("choices")
            content = None
            if choices and isinstance(choices, list):
                content = choices[0].get("message", {}).get("content")
            if content is None:
                # fallback to raw json
                records.append({"prompt": p, "result": out.get("json"), "result_code": int(status)})
            else:
                # try parse content if it's JSON string
                try:
                    parsed = json.loads(content)
                    records.append({"prompt": p, "result": parsed.get("result", parsed), "result_code": int(parsed.get("result_code", status))})
                except Exception:
                    records.append({"prompt": p, "result": content, "result_code": int(status)})
    else:
        # MOCK mode
        out = mock_response_for_prompt(p)
        status = out["status_code"]
        if status != 200:
            records.append({"prompt": p, "result": out.get("json") or out.get("text"), "result_code": int(status)})
            print(f"  -> Mock HTTP {status}")
        else:
            # parse the content field similar to live
            choices = out.get("json", {}).get("choices")
            content = None
            if choices:
                content = choices[0].get("message", {}).get("content")
            if content:
                try:
                    parsed = json.loads(content)
                    records.append({"prompt": p, "result": parsed.get("result", parsed), "result_code": int(parsed.get("result_code", status))})
                except Exception:
                    records.append({"prompt": p, "result": content, "result_code": int(status)})
            else:
                records.append({"prompt": p, "result": out.get("json"), "result_code": int(status)})

    time.sleep(SLEEP_BETWEEN)

# Save dataframe
df = pd.DataFrame(records)[["prompt", "result", "result_code"]]
df["result_code"] = pd.to_numeric(df["result_code"], errors="coerce").fillna(0).astype(int)
df.to_pickle(OUT_PKL)
print(f"\nSaved: {OUT_PKL}")
display(df)

