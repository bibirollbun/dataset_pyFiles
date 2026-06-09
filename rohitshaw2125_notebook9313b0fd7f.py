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


# Install required libraries (optional)
!pip install --quiet requests pandas

import os
import json
import requests
import pandas as pd
from time import sleep
from IPython.display import display, HTML



# --- FIX: Load OpenAI API key from Kaggle Secrets correctly ---
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value = user_secrets.get_secret("OPENAI_API_KEY")

if not secret_value:
    raise RuntimeError("OPENAI_API_KEY missing in Settings → Secrets")

# Make the key available globally
os.environ["OPENAI_API_KEY"] = secret_value

print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))



sample_csv = """subject,body,category
Cannot login,Hi I cannot login since yesterday.,login
Charge issue,I see an unauthorized charge.,billing
Order not arrived,My order #12345 not delivered.,shipping
Refund issue,I returned item but no refund.,returns
App crash,App crashes when uploading photo.,technical
"""

with open("sample_tickets.csv", "w", encoding="utf-8") as f:
    f.write(sample_csv)

print("Created sample_tickets.csv")



CSV_PATH = "sample_tickets.csv"
df = pd.read_csv(CSV_PATH)
df.index = range(1, len(df) + 1)
display(df)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL_CLASSIFY = "gpt-4o-mini"
OPENAI_MODEL_REPLY = "gpt-4o-mini"

def call_openai_chat(messages, model=OPENAI_MODEL_CLASSIFY, temperature=0.0, max_tokens=600):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

CLASSIFY_SYSTEM = {
    "role": "system",
    "content": (
        "You are a concise customer support triage assistant. "
        "Return JSON with: intent, category, priority, escalate, confidence, rationale."
    )
}

REPLY_SYSTEM = {
    "role": "system",
    "content": (
        "You are a helpful customer support agent. "
        "Write a short professional reply (50-150 words)."
    )
}



import re

def extract_json_from_text(s: str):
    try:
        return json.loads(s)
    except:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except:
                pass
    raise ValueError("Invalid JSON output")

def classify_ticket_text(text: str):
    user_msg = {"role": "user", "content": f"TICKET:\n\n{text}\n\nReturn JSON only."}
    resp = call_openai_chat([CLASSIFY_SYSTEM, user_msg])
    raw = resp["choices"][0]["message"]["content"]
    parsed = extract_json_from_text(raw)
    return parsed, raw

def generate_reply(ticket_text: str, intent: str, priority: str, escalate: bool):
    user_msg = {
        "role":"user",
        "content":(
            f"Ticket:\n{ticket_text}\n\nContext: intent={intent}, priority={priority}, escalate={escalate}\n"
            "Write a professional reply."
        )
    }
    resp = call_openai_chat([REPLY_SYSTEM, user_msg], model=OPENAI_MODEL_REPLY, temperature=0.1)
    reply_text = resp["choices"][0]["message"]["content"].strip()
    return reply_text, resp



FORCED_ESCALATE = ["credit card", "bank", "suicide", "dead", "legal", "lawsuit", "exploit"]

def check_forced_escalation(text):
    t = text.lower()
    return any(kw in t for kw in FORCED_ESCALATE)



results = []

for idx, row in df.iterrows():
    ticket_text = f"Subject: {row['subject']}\nBody: {row['body']}"
    forced = check_forced_escalation(ticket_text)

    try:
        parsed, raw_class = classify_ticket_text(ticket_text)
    except Exception as e:
        parsed = {"intent":"unknown","category":"general","priority":"medium","escalate":False,"confidence":0}
        raw_class = str(e)

    intent = parsed.get("intent", "unknown")
    category = parsed.get("category", "general")
    priority = parsed.get("priority", "medium")
    escalate_flag = parsed.get("escalate", False) or forced
    confidence = parsed.get("confidence", 0.0)

    try:
        reply_text, raw_reply = generate_reply(ticket_text, intent, priority, escalate_flag)
    except Exception as e:
        reply_text = "Error generating reply."
        raw_reply = str(e)

    results.append({
        "id": idx,
        "subject": row["subject"],
        "body": row["body"],
        "intent": intent,
        "category": category,
        "priority": priority,
        "escalate": escalate_flag,
        "confidence": confidence,
        "suggested_reply": reply_text,
        "raw_class_output": raw_class,
        "raw_reply_output": raw_reply,
    })

    sleep(0.5)   # avoid rate-limits

results_df = pd.DataFrame(results)
display(results_df[["id","subject","category","priority","escalate","confidence"]])



results_df.to_csv("triage_results.csv", index=False)
print("Saved triage_results.csv — download from the Files panel.")


