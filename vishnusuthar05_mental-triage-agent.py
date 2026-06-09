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

#installs (run in Kaggle; pip may be restricted but usually allowed)
!pip install -q flask flask-restful openai



# Cell 1: imports & quick config
import re, time, uuid, json, os, threading
from datetime import datetime
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
# OpenAI usage (enable by setting OPENAI_API_KEY env var)
USE_OPENAI = bool(os.getenv('OPENAI_API_KEY', ''))
if USE_OPENAI:
    import openai



# Cell 2: utilities: structured logger, memory, sample resources
LOGFILE = "logs.jsonl"
MEMFILE = "memory.json"

def log_event(event: dict):
    event['ts'] = datetime.utcnow().isoformat()
    with open(LOGFILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

def load_memory():
    try:
        with open(MEMFILE) as f:
            return json.load(f)
    except:
        return {}
def save_memory(mem):
    with open(MEMFILE,"w") as f:
        json.dump(mem, f, indent=2)

memory = load_memory()

# Sample minimal resources DB (replace/extend for your region)
RESOURCES = [
    {"name":"National Suicide Hotline (USA)", "region":"US", "phone":"+1-800-273-8255", "type":"hotline", "url":"https://988lifeline.org"},
    {"name":"Samaritans (UK & ROI)", "region":"UK", "phone":"+44 8457 90 90 90", "type":"hotline", "url":"https://www.samaritans.org"},
    {"name":"Local Low-cost Clinic Example", "region":"IN", "phone":"+91-22-12345678","type":"clinic", "url":"https://exampleclinic.org"}
]



# Cell 3: Intake Agent (rule-based safety checks)
HIGH_RISK_PATTERNS = [r"\bkill myself\b", r"\bi want to die\b", r"\bsuicid", r"\bhurt myself\b", r"\bend my life\b"]
def intake_agent(session_id, text):
    t = text.lower()
    for p in HIGH_RISK_PATTERNS:
        if re.search(p, t):
            out = {
                "session_id": session_id,
                "agent":"intake",
                "risk":"CRISIS",
                "message":"Immediate risk detected. Provide emergency instructions and hotline."
            }
            log_event({"session":session_id, "agent":"intake", "detected":p, "risk":"CRISIS"})
            return out
    # otherwise collect short metadata
    out = {"session_id": session_id, "agent":"intake", "risk":"UNKNOWN", "message":"Intake ok", "text": text}
    log_event({"session":session_id, "agent":"intake", "risk":"UNKNOWN"})
    return out



# Cell 4: Triage Agent (LLM if available, otherwise rule-based)
def triage_rules(text):
    t = text.lower()
    # quick heuristics
    if any(w in t for w in ["panic","panic attack","can't breathe","chest pain","hurt"]):
        return {"level":"URGENT","reason":"physical/acute symptoms or severe panic"}
    if any(w in t for w in ["sad","hopeless","depressed","lost interest"]):
        return {"level":"NONURGENT","reason":"depressive symptoms - recommend resources"}
    return {"level":"NONURGENT","reason":"general support"}

def triage_agent(session_id, text):
    start = time.time()
    result = None
    if USE_OPENAI:
        prompt = (
            "You are a safe, conservative triage assistant. "
            "Given user text, output JSON with keys: level (CRISIS/URGENT/NONURGENT), reason, and advice. "
            "If user expresses immediate self-harm, return CRISIS."
            f"\n\nUser text: '''{text}'''"
        )
        # minimal call; safe system message recommended in production
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if False else "gpt-4o-mini", # change if needed
                messages=[{"role":"user","content":prompt}],
                max_tokens=200, temperature=0.0
            )
            # crude parse attempt
            content = resp['choices'][0]['message']['content']
            # try to find JSON in content
            import ast
            try:
                parsed = ast.literal_eval(content.strip())
                result = parsed
            except:
                # fallback to rules
                result = triage_rules(text)
        except Exception as e:
            result = triage_rules(text)
    else:
        result = triage_rules(text)
    duration = int((time.time()-start)*1000)
    log_event({"session":session_id, "agent":"triage", "result":result, "duration_ms":duration})
    return {"session_id":session_id, "agent":"triage", **result}





# Cell 5: Resource Finder Agent (static DB search)
def resource_finder(session_id, level, region_hint=None):
    # if crisis -> return hotlines matching region if possible, else global hotline
    region = (region_hint or "").upper()
    matches = []
    for r in RESOURCES:
        if level == "CRISIS" and r['type']=="hotline":
            if region and r['region'].upper()==region:
                matches.append(r)
            elif not region:
                matches.append(r)
        elif level!="CRISIS":
            # add all resource types
            if (not region) or (r['region'].upper()==region):
                matches.append(r)
    if not matches:
        # fallback: global hotline first entry
        matches = [RESOURCES[0]]
    log_event({"session":session_id, "agent":"resource_finder", "match_count":len(matches), "requested_level":level})
    return {"session_id":session_id, "agent":"resource_finder", "resources":matches}



# Cell 6: Coordinator & simple Flask API for demo + in-notebook runner
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

def ensure_memory(session_id):
    if session_id not in memory:
        memory[session_id] = {"history":[], "created": datetime.utcnow().isoformat()}
        save_memory(memory)

@app.route('/api/triage', methods=['POST'])
def api_triage():
    payload = request.json or {}
    text = payload.get("text","")
    session_id = payload.get("session_id") or str(uuid.uuid4())
    ensure_memory(session_id)
    # Intake
    intake = intake_agent(session_id, text)
    memory[session_id]["history"].append({"agent":"intake","out":intake})
    save_memory(memory)
    if intake.get("risk") == "CRISIS":
        # immediate instructions + resources
        res = resource_finder(session_id, "CRISIS", payload.get("region"))
        out = {"session_id":session_id, "final":"CRISIS", "message":"EMERGENCY: Please call the hotline immediately. If danger, call emergency services.", "resources":res['resources']}
        log_event({"session":session_id, "flow":"CRISIS_TERMINATE"})
        return jsonify(out)
    # Otherwise triage
    tri = triage_agent(session_id, text)
    memory[session_id]["history"].append({"agent":"triage","out":tri})
    save_memory(memory)
    # resources
    res = resource_finder(session_id, tri.get("level","NONURGENT"), payload.get("region"))
    memory[session_id]["history"].append({"agent":"resource_finder","out":res})
    save_memory(memory)
    final = {"session_id":session_id, "final_level":tri.get("level"), "reason":tri.get("reason"), "resources":res['resources']}
    log_event({"session":session_id, "flow":"COMPLETED", "final_level":tri.get("level")})
    return jsonify(final)

# Run the app in background thread to keep notebook responsive on Kaggle:
def run_app():
    app.run(host='0.0.0.0', port=8080, debug=False)

# Starting the server (non-blocking)
thread = threading.Thread(target=run_app, daemon=True)
thread.start()
print("Server started on port 8080 - use web preview or curl to call /api/triage")



# Cell 7: quick demo calls (run after server started)
import requests, time
BASE = "http://127.0.0.1:8080"
examples = [
    {"text":"I feel so hopeless and worthless. I have thought about ending my life.", "region":"US"},
    {"text":"I've been very sad for a few weeks and can't concentrate.", "region":"IN"},
    {"text":"I had a panic attack and can't stop shaking.", "region":"UK"},
]
for ex in examples:
    r = requests.post(BASE+"/api/triage", json=ex)
    print("INPUT:", ex["text"][:80])
    print("OUTPUT:", r.json())
    print("-"*60)
    time.sleep(0.8)



# Cell 8: Evaluation harness - simple test cases and scoring
tests = [
    ("I want to die", "CRISIS"),
    ("I am feeling extremely hopeless", "NONURGENT"),
    ("My chest hurts and I'm hyperventilating", "URGENT"),
    ("I have been crying all day", "NONURGENT"),
    ("I will kill myself tonight", "CRISIS"),
    ("I feel low motivation at work", "NONURGENT"),
    ("I lost interest in hobbies", "NONURGENT"),
    ("I feel like I might hurt myself", "CRISIS"),
    ("I had a panic attack today", "URGENT"),
    ("I need a low-cost clinic", "NONURGENT")
]
correct = 0
for text, expected in tests:
    r = requests.post(BASE+"/api/triage", json={"text":text})
    out = r.json()
    level = out.get("final_level") or out.get("final") or out.get("final_level")
    # normalize
    if level == "CRISIS" and expected=="CRISIS":
        correct +=1
    elif expected=="URGENT" and level=="URGENT":
        correct +=1
    elif expected=="NONURGENT" and level in ("NONURGENT", "NON-URGENT", None):
        correct +=1
print(f"Evaluation: {correct}/{len(tests)} correct (note: simple rule-based baseline).")



# Cell 9: show logs (last 20) and memory summary
def tail_logs(n=20):
    lines=[]
    try:
        with open(LOGFILE) as f:
            lines = f.readlines()[-n:]
    except:
        lines=[]
    return [json.loads(l) for l in lines]

print("Recent logs:")
for l in tail_logs(20):
    print(l)
print("\nMemory snapshot (first 3 sessions):")
for k in list(memory.keys())[:3]:
    print(k, memory[k])



# Cell 10: pack key files into an attachments folder to upload on Kaggle
!mkdir -p submission_files
!cp logs.jsonl memory.json submission_files/ 2>/dev/null || true
print("Saved logs/memory to submission_files/")


