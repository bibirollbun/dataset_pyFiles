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


!pip install --quiet google-genai
print("Gemini client installed.")


import os
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("API key loaded.")



import json, re, time, uuid
from typing import List, Dict, Any

REFERENCE_PDF = "/mnt/data/Introduction to Agents.pdf"
MEMORY_FILE = "/kaggle/working/user_memory.json"
OUTPUT_FILE = "/kaggle/working/dream_agent_output.json"

# Gemini client
from google import genai
genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("Gemini client ready.")



def call_gemini(prompt, model="gemini-2.5-flash", temperature=0.25):
    try:
        resp = genai_client.generate_text(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=1000
        )
        return resp.text
    except Exception as e:
        return str(e)



def extract_json(text: str):
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except:
        cleaned = m.group(0).replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except:
            return {}



def interpret_dream(dream_text, memories=None):
    mem_block = "\n".join(memories or [])
    
    prompt = f"""
You are a compassionate dream interpreter.
Use emotional context and symbolism.

Grounding reference: {REFERENCE_PDF}

Dream:
\"\"\"{dream_text}\"\"\"

User memories:
{mem_block}

Return ONLY JSON:
{{
 "themes": [...],
 "emotions": [...],
 "subconscious_signals": [...],
 "interpretation": "..."
}}
"""
    raw = call_gemini(prompt)
    parsed = extract_json(raw)
    parsed["raw"] = raw
    return parsed



def generate_plan(themes, emotions, signals):
    prompt = f"""
Create a structured plan using themes, emotions, and signals.

Input:
themes = {themes}
emotions = {emotions}
signals = {signals}

Return ONLY JSON:
{{
 "short_term_plan": [...],   # 7 days
 "long_term_plan": [...],    # 4 weeks
 "advice": "..."
}}
"""
    raw = call_gemini(prompt, temperature=0.2)
    parsed = extract_json(raw)
    parsed["raw"] = raw
    return parsed



def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE,"r"))
    return {}

def save_memory(data):
    json.dump(data, open(MEMORY_FILE,"w"), indent=2)

def append_memory(user_id, text):
    data = load_memory()
    data.setdefault(user_id, [])
    data[user_id].append({"ts": int(time.time()), "text": text})
    data[user_id] = data[user_id][-200:]
    save_memory(data)

def recent_memories(user_id, limit=5):
    data = load_memory()
    return [m["text"] for m in data.get(user_id, [])][-limit:]



def dream_agent(user_id, dream_text):
    
    # 1) Fetch recent memories
    mems = recent_memories(user_id, limit=5)
    
    # 2) Interpret dream (Gemini tool)
    interp = interpret_dream(dream_text, mems)
    
    # 3) Save memory snippet
    themes = interp.get("themes", [])
    if themes:
        mem_snip = "dream_themes:" + ",".join(themes[:2])
    else:
        mem_snip = "dream_raw:" + dream_text[:40]
    
    append_memory(user_id, mem_snip)
    
    # 4) Generate 7-day & 30-day plan
    plan = generate_plan(
        interp.get("themes", []),
        interp.get("emotions", []),
        interp.get("subconscious_signals", [])
    )
    
    # 5) Return final agent output
    return {
        "dream": dream_text,
        "interpretation": interp,
        "plan": plan,
        "saved_memory": mem_snip
    }



dream = "I was lost in a huge maze and couldn't find the exit."
output = dream_agent("varun", dream)
output



json.dump(output, open(OUTPUT_FILE,"w"), indent=2)
print("Saved:", OUTPUT_FILE)





