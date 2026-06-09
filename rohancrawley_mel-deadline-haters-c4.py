# pip install -q aiohttp pandas
import asyncio
import aiohttp
import json
import re
import pandas as pd
from kaggle_secrets import UserSecretsClient

# ---------- config ----------
CSV_FILE = "/kaggle/input/my-prompts-yoo/my_prompts.csv"                  # must be in same directory
FULL_URL = "https://rmit-hackathon-ve.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
OUT_PKL = "attack_dataset.pkl"
OUT_CSV = "attack_dataset.csv"

user_secrets = UserSecretsClient()
API_KEY = "EOd81yMG8zL8My7IRsBuBT9Z3iBMkL51KBTJ4xba6l5MWYguNnHBJQQJ99BJACYeBjFXJ3w3AAABACOGb1Yy"

SYSTEM_JSON_PROMPT = (
    'Return ONLY a minified JSON object with keys "prompt","result","result_code". '
    'No code fences. No extra text.'
)

# ---------- load prompts ----------
# tolerant read to handle odd encodings
try:
    df_prompts = pd.read_csv(CSV_FILE, encoding="utf-8")
except Exception:
    df_prompts = pd.read_csv(CSV_FILE, encoding="latin1")

# Expect column named 'prompt'. If no header, assume single column.
if "prompt" not in df_prompts.columns:
    df_prompts.columns = ["prompt"]

prompts = df_prompts["prompt"].dropna().astype(str).tolist()

def make_eval_prompt(text: str):
    return [
        {"role": "system", "content": SYSTEM_JSON_PROMPT},
        {"role": "user",   "content": f'Input: "{text}"'}
    ]

messages_list = [make_eval_prompt(p) for p in prompts]

# ---------- helpers ----------
def extract_first_json_str(s: str):
    m = re.search(r'\{.*\}', s, re.S)
    return m.group(0) if m else None

def normalize_success(resp_text: str):
    """
    Handle Azure 200 response shape. choices[0].message.content contains JSON-as-text.
    Return (result, result_code:int).
    """
    j = json.loads(resp_text)
    # robust access
    try:
        content = j["choices"][0]["message"]["content"]
    except Exception:
        # unexpected shape: return raw text
        return resp_text, 200

    raw_json_str = content.strip()
    if not raw_json_str.startswith("{"):
        candidate = extract_first_json_str(content)
        raw_json_str = candidate if candidate else content

    try:
        parsed = json.loads(raw_json_str)
    except Exception:
        # assistant didn't return strict JSON. return content as string.
        return content, 200

    result = parsed.get("result", content)
    code = int(parsed.get("result_code", 200))
    return result, code

def normalize_error(err_text: str, status: int):
    """
    Handle Azure error shapes. If content_filter/jailbreak detected mark blocked.
    Return (result, result_code:int).
    """
    try:
        e = json.loads(err_text)
    except Exception:
        return err_text, status

    err = e.get("error") or {}
    inner = err.get("innererror") or {}
    cf = inner.get("content_filter_result") or {}
    jb = cf.get("jailbreak") or {}

    if err.get("code") == "content_filter" or jb.get("filtered") or jb.get("detected"):
        # blocked by safety system
        return {"error": "blocked by safety system"}, int(err.get("status") or status)
    # otherwise surface error message
    return (err.get("message") or err_text), int(err.get("status") or status)

async def call_api(session, messages, prompt_text):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {"messages": messages, "temperature": 0}
    try:
        async with session.post(FULL_URL, headers=headers, json=body, timeout=60) as resp:
            text = await resp.text()
            if resp.status == 200:
                result, code = normalize_success(text)
            else:
                result, code = normalize_error(text, resp.status)
            return {
                "prompt": prompt_text,
                "result": result,
                "result_code": code,
                "raw": text,
                "http_status": resp.status,
            }
    except Exception as e:
        return {
            "prompt": prompt_text,
            "result": str(e),
            "result_code": 500,
            "raw": str(e),
            "http_status": 500,
        }

# ---------- main ----------
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [call_api(session, m, p) for m, p in zip(messages_list, prompts)]
        outs = await asyncio.gather(*tasks)

    df_out = pd.DataFrame([{"prompt": o["prompt"], "result": o["result"], "result_code": o["result_code"]} for o in outs])
    df_out.to_pickle(OUT_PKL)
    df_out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_PKL, "and", OUT_CSV)
    print(df_out.head())

    # pretty-print raw responses for inspection
    print("\n=== Raw responses ===")
    for i, o in enumerate(outs, 1):
        print(f"\n--- Prompt {i} | HTTP {o['http_status']} ---")
        raw = o["raw"]
        try:
            parsed_raw = json.loads(raw)
            print(json.dumps(parsed_raw, indent=2))
        except Exception:
            print(raw)

    print("\n=== output format ===")
    print(df_out)
    



# run
await main()


