"""Challenge 4 - Azure Submission"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import requests

print("=" * 80)
print("HANOI EMPTYQUEUE - CHALLENGE 4 SUBMISSION")
print("=" * 80)

# --- Azure OpenAI configuration (hard-coded) --------------------------------
ENDPOINT = "https://rmit-hackathon-ve.openai.azure.com"
API_KEY = "EOd81yMG8zL8My7IRsBuBT9Z3iBMkL51KBTJ4xba6l5MWYguNnHBJQQJ99BJACYeBjFXJ3w3AAABACOGb1Yy"
DEPLOYMENT_NAME = "gpt-35-turbo"
API_VERSION = "2025-01-01-preview"

PROMPTS_PATH = Path("/kaggle/input/rmit-2025-emptyqueue-data-c4/prompts.txt")
OUT_PKL = Path("attack_dataset.pkl")



prompts: List[str] = [line.strip() for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}.")



def call_azure_openai(prompt_text: str) -> Dict:
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 800,
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    try:
        data = response.json()
    except Exception:
        data = {"error": {"message": response.text or "Unknown error"}}
    return {
        "status_code": response.status_code,
        "data": data,
    }



results: List[Dict] = []
print("[2/3] Sending prompts to Azure...")

for idx, prompt_text in enumerate(prompts, start=1):
    print(f"Prompt {idx}/{len(prompts)} | length={len(prompt_text)} chars")
    response = call_azure_openai(prompt_text)
    status_code = response["status_code"]
    payload = response["data"]

    if status_code == 200:
        print(f"  -> BYPASSED (code {status_code})")
        results.append({
            "prompt": prompt_text,
            "result": payload,
            "result_code": status_code,
        })
    else:
        print(f"  -> BLOCKED (code {status_code})")
        message = payload.get("error", {}).get("message", "Blocked by content filter")
        results.append({
            "prompt": prompt_text,
            "result": {"error": message},
            "result_code": status_code,
        })
    time.sleep(1)  # gentle rate limiting



with open(OUT_PKL, "wb") as f:
    pickle.dump(results, f)

print("[3/3] Saved submission file.")
print(f"[OK] Wrote {OUT_PKL}")



blocked = sum(1 for r in results if r["result_code"] != 200)
bypassed = len(results) - blocked

print("=" * 80)
print("SUBMISSION SUMMARY")
print("=" * 80)
print(f"Total prompts: {len(results)}")
print(f"Blocked: {blocked}")
print(f"Bypassed: {bypassed}")
print("=" * 80)
for r in results:
    status = "BYPASSED" if r["result_code"] == 200 else "BLOCKED"
    print(f"[{status}] {r['prompt'][:60]}{'...' if len(r['prompt']) > 60 else ''}")


results




