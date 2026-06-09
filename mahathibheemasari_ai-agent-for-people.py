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


# Cell 1: Setup & imports
import uuid
import json
import re
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# TF-IDF retriever
from sklearn.feature_extraction.text import TfidfVectorizer


knowledge_docs = [
    {"id":"doc_fire_1","title":"Home Fire Safety - Evacuation", "text":
     "If you detect fire or heavy smoke, get everyone out immediately. Use stairs not elevators. Close doors behind you to slow spread. Call the fire department. If clothes are on fire, stop, drop, and roll."},
    {"id":"doc_fire_2","title":"Fire Extinguisher Basics","text":
     "Use an extinguisher only if the fire is small and you know how to operate it. Aim at the base, sweep side-to-side. Prioritize evacuating people first."},
    {"id":"doc_gas_1","title":"Gas Leak Safety", "text":
     "If you smell gas (LPG) or chemical fumes, do not switch electrical appliances on or off. Open windows and doors if safe. Evacuate the area and call gas emergency services. Avoid using phones near the leak."},
    {"id":"doc_medical_1","title":"Medical Emergency - Basic Triage","text":
     "For unconscious or unresponsive persons, call ambulance immediately. If there is heavy bleeding apply direct pressure. If trained, start CPR on non-breathing persons. Keep the patient warm and do not move them unnecessarily."},
    {"id":"doc_crime_1","title":"Crime Response - Safety First","text":
     "If you observe a burglary or intruder, stay in a secure location, lock doors, call the police, do not confront intruders. Note descriptions and direction of escape from a safe location."},
    {"id":"doc_electrical_1","title":"Electrical Fire & Short Circuit", "text":
     "If you smell burning or see sparks, switch off the main power if safe. Avoid touching exposed wires. Evacuate if fire spreads. Do not use water on live electrical fire."},
    {"id":"doc_watchman_1","title":"Watchman Protocol", "text":
     "Community watchman contact procedure: on confirmed emergency, notify watchman with location and issue type. Include caller name, flat, and brief instructions. For simulated systems, require confirmation before sending messages."}
]



# Cell 3: Build TF-IDF retriever (fast, deterministic)
docs_text = [d["text"] for d in knowledge_docs]
vectorizer = TfidfVectorizer().fit(docs_text)
doc_vecs = vectorizer.transform(docs_text)

def retrieve(query: str, top_k:int=3) -> List[Dict[str,Any]]:
    """Return top_k knowledge docs (with similarity > 0)."""
    qv = vectorizer.transform([query])
    sims = (doc_vecs * qv.T).toarray().ravel()
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        if sims[i] > 0:
            results.append({**knowledge_docs[i], "score": float(sims[i])})
    return results


# Cell 4: Classifier (keyword + simple scoring) and severity heuristic
EMERGENCY_KEYWORDS = {
    "fire": ["smoke","fire","flames","caught fire","burning"],
    "medical": ["unconscious","bleeding","not breathing","heart attack","faint","severe bleeding"],
    "gas": ["gas smell","gas leak","lpg","chemical smell","fumes","odor"],
    "crime": ["burglary","break in","intruder","robbery","theft","suspicious"],
    "electrical": ["sparks","short circuit","burning smell", "wires","electric shock"],
    "flood": ["water flooding","leak","flood","burst pipe"]
}

def classify_emergency(text: str) -> Tuple[str, str]:
    t = text.lower()
    scores = {}
    for k, kws in EMERGENCY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[k] = scores.get(k,0)+1
    if not scores:
        return ("unknown", "low")
    # pick best
    kind = max(scores, key=lambda x: scores[x])
    # severity heuristic: presence of certain tokens increases severity
    severity_tokens = ["unconscious","not breathing","severe","heavy","intense","collapsed"]
    severity = "high" if any(tok in t for tok in severity_tokens) or scores[kind] >= 2 else "medium"
    return (kind, severity)


# Cell 5: Sensitivity detection (medical/legal/psychiatric -> sensitive)
SENSITIVE_KEYWORDS = {"unconscious","suicide","self harm","stroke","heart attack","severe bleeding","not breathing"}

def is_sensitive(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in SENSITIVE_KEYWORDS)



# Cell 6: Guidance generator - combines templates + retrieved docs
def guidance_from_kb(kind: str, retrieved: List[Dict[str,Any]]) -> List[str]:
    """Return prioritized action list using retrieved docs + templates."""
    templates = {
        "fire": [
            "Evacuate everyone immediately—use stairs, not elevators.",
            "Call fire department (local fire number).",
            "If safe and trained, use a fire extinguisher on small fires."
        ],
        "medical": [
            "Call ambulance immediately.",
            "If heavy bleeding, apply firm pressure. If trained, begin CPR if not breathing.",
            "Keep the person warm and monitor breathing."
        ],
        "gas": [
            "Do not switch electrical appliances on or off.",
            "Ventilate if safe, then evacuate the area.",
            "Call gas emergency services."
        ],
        "crime":[
            "Find a safe place and lock doors. Do not confront intruders.",
            "Call the police immediately with your location.",
            "Observe from a safe spot and note descriptions if possible."
        ],
        "electrical":[
            "Switch off the electrical main if safe to do so.",
            "Avoid touching exposed wires and evacuate if fire or sparks persist.",
            "Call a qualified electrician or building maintenance."
        ],
        "flood":[
            "Move to higher ground inside the building.",
            "Turn off electricity if safe and avoid contact with water.",
            "Call building maintenance and emergency services if required."
        ],
        "unknown":[
            "Please provide more detail (e.g., 'There is smoke in my kitchen' or 'Someone is unconscious')."
        ]
    }
    base = templates.get(kind, templates["unknown"])
    # augment with the most relevant sentences from retrieved docs
    augment = []
    for d in retrieved:
        # take top sentence-like slices (simple split) to avoid long paragraphs
        sents = re.split(r'\.\s+', d["text"])
        for s in sents[:2]:
            if len(s.strip())>20 and s.strip() not in augment and s.strip() not in base:
                augment.append(s.strip())
    # prioritize base templates and add augmentations (but keep list short)
    result = base + augment
    return result[:6]


# Cell 7: Simple refiner to check for missing safety rules & produce final text
def refine_draft(draft_lines: List[str], kind: str, severity: str) -> str:
    """
    Basic reflection: ensure core 'do not re-enter' for fire, ensure 'call ambulance' for high medical severity.
    Returns a single formatted string.
    """
    lines = draft_lines.copy()
    text_lower = " ".join(lines).lower()
    # Safety rules
    if kind == "fire" and "do not re-enter" not in text_lower and "do not go back inside" not in text_lower:
        lines.insert(1, "Do not re-enter the building after you have evacuated.")
    if kind == "medical" and severity == "high" and not any("call ambulance" in l.lower() or "call 108" in l.lower() for l in lines):
        lines.insert(0, "Call the ambulance immediately (if in India dial 108).")
    # final formatting
    header = f"Detected incident: {kind.upper()} (severity: {severity})"
    timestamp = f"Timestamp: {datetime.utcnow().isoformat()} UTC"
    final = header + "\n" + timestamp + "\n\n" + "\n".join(f"{i+1}. {l}" for i,l in enumerate(lines))
    return final


# Cell 8: Notifier (simulated) and brief composer
watchmen_db = {
    "Sunrise_Apartments": {"name":"Ramesh","phone":"+911234567890"},
    "Lakeview_Heights": {"name":"Kumar","phone":"+919876543210"}
}

def notify_watchman_sim(community_key: str, brief: Dict[str,Any]) -> Dict[str,Any]:
    """Simulated notify - returns delivery confirmation object."""
    wk = watchmen_db.get(community_key)
    if not wk:
        return {"success": False, "message": "Watchman contact not found (simulated)."}
    # simulated SMS content
    content = f"ALERT: {brief['type'].upper()} at {brief.get('location','Unknown')}. Actions: {brief['recommended_actions'][:3]}"
    # in real system we would send SMS; here we simulate success
    return {"success": True, "message": f"Simulated SMS to {wk['name']} ({wk['phone']}): {content}"}

def compose_brief(incident: Dict[str,Any], retrieved_docs: List[Dict[str,Any]]) -> Dict[str,Any]:
    brief = {
        "incident_id": f"INC-{uuid.uuid4().hex[:8]}",
        "type": incident["type"],
        "severity": incident.get("severity","unknown"),
        "timestamp": datetime.utcnow().isoformat(),
        "location": incident.get("location"),
        "reporter": incident.get("reporter"),
        "recommended_actions": incident.get("actions"),
        "evidence_doc_ids": [d["id"] for d in retrieved_docs],
        "notes": incident.get("notes","")
    }
    return brief


# Cell 9: SQLite setup for logging incidents and agent actions
conn = sqlite3.connect(":memory:")  # in-memory for demo; change to file for persistence
c = conn.cursor()

# Create tables
c.execute("""
CREATE TABLE incidents (
    id TEXT PRIMARY KEY,
    type TEXT,
    severity TEXT,
    timestamp TEXT,
    location TEXT,
    reporter TEXT,
    notified_watchman INTEGER,
    brief_json TEXT
)
""")

c.execute("""
CREATE TABLE agent_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT,
    agent_name TEXT,
    action TEXT,
    note TEXT,
    timestamp TEXT
)
""")

conn.commit()

def log_incident(brief: Dict[str,Any], notified: bool):
    c.execute("INSERT INTO incidents (id,type,severity,timestamp,location,reporter,notified_watchman,brief_json) VALUES (?,?,?,?,?,?,?,?)",
              (brief["incident_id"], brief["type"], brief["severity"], brief["timestamp"], brief.get("location"), brief.get("reporter"), int(notified), json.dumps(brief)))
    conn.commit()

def log_action(incident_id: str, agent_name: str, action: str, note: str=""):
    c.execute("INSERT INTO agent_logs (incident_id,agent_name,action,note,timestamp) VALUES (?,?,?,?,?)",
              (incident_id, agent_name, action, note, datetime.utcnow().isoformat()))
    conn.commit()

# helper to show incidents as DataFrame
def show_incidents_df():
    df = pd.read_sql_query("SELECT id,type,severity,timestamp,location,reporter,notified_watchman FROM incidents ORDER BY timestamp DESC", conn)
    return df



# Cell 10: Master pipeline function
def pipeline(user_input: str, community_key: str, reporter: str=None, location: str=None, require_notify_confirm: bool=True) -> Dict[str,Any]:
    """
    Runs the full Phase-1 pipeline:
    1) sensitivity check
    2) retrieval (RAG)
    3) classification + severity
    4) guidance draft
    5) refinement
    6) admin confirmation for sensitive
    7) simulated notify, brief composition, logging
    Returns response dict with final_text, brief, notify_result
    """
    # Normalize input
    is_voice = False
    if user_input.strip().lower().startswith("(voice)"):
        is_voice = True
        user_input = user_input.strip()[len("(voice)"):].strip()

    # 1) Sensitivity check
    sensitive = is_sensitive(user_input)

    # 2) Retrieve context
    retrieved = retrieve(user_input, top_k=3)
    log_action("-", "RetrieverAgent", "retrieve", note=f"retrieved {len(retrieved)} docs")

    # 3) Classify
    kind, severity = classify_emergency(user_input)
    log_action("-", "ClassifierAgent", "classify", note=f"class={kind}, severity={severity}")

    # 4) Guidance draft from KB
    actions = guidance_from_kb(kind, retrieved)
    log_action("-", "GuidanceAgent", "draft", note=f"drafted {len(actions)} actions")

    # 5) Refine
    final_text = refine_draft(actions, kind, severity)
    log_action("-", "RefinerAgent", "refine", note="refinement complete")

    # 6) Compose brief
    incident = {"type": kind, "severity": severity, "location": location, "reporter": reporter, "actions": actions, "notes": user_input}
    brief = compose_brief(incident, retrieved)
    log_action(brief["incident_id"], "RecorderAgent", "compose_brief", note="brief composed")

    # 7) Admin check for sensitive or if notify required
    notify_result = {"success": False, "message": "No notification sent."}
    if require_notify_confirm:
        # For notebook demo, we auto-confirm non-sensitive; require manual confirm for sensitive
        if sensitive:
            # don't auto-notify sensitive incidents automatically
            notify_decision = False
            log_action(brief["incident_id"], "AdminAgent", "require_confirm", note="sensitive incident - confirmation required")
        else:
            notify_decision = True
    else:
        notify_decision = True  # force notify

    # 8) Notify (simulated) if confirmed
    if notify_decision:
        notify_result = notify_watchman_sim(community_key, brief)
        log_action(brief["incident_id"], "NotifierAgent", "notify", note=notify_result.get("message",""))
        notified_flag = notify_result.get("success", False)
    else:
        notified_flag = False

    # 9) Persist brief & incident log
    log_incident(brief, notified_flag)

    # 10) Return structured response (speak flag if voice)
    response = {"final_text": final_text, "brief": brief, "notify_result": notify_result, "sensitive": sensitive, "speak": is_voice}
    return response



# Cell 11: Demo runs - replace community_key/location/reporter as needed

demos = [
    {"input":"(voice) There is heavy smoke in my kitchen and my neighbor is shouting","community":"Sunrise_Apartments","reporter":"Priya","location":"Tower A, Flat 302"},
    {"input":"My father collapsed and is not breathing","community":"Sunrise_Apartments","reporter":"Ravi","location":"Tower B, Flat 101"},
    {"input":"I can smell gas in the corridor, it's pungent","community":"Lakeview_Heights","reporter":"Anita","location":"Block C, Flat 10"},
    {"input":"Someone is trying to break into our house, there is a noise","community":"Sunrise_Apartments","reporter":"Karan","location":"Tower A, Flat 110"},
    {"input":"There are sparks and burning smell from the switchboard","community":"Lakeview_Heights","reporter":"Meera","location":"Block A, Flat 5"}
]

for idx, d in enumerate(demos,1):
    print(f"\n--- Demo #{idx} input: {d['input']}\n")
    out = pipeline(d['input'], community_key=d['community'], reporter=d['reporter'], location=d['location'])
    print("FINAL RESPONSE (refined):\n")
    print(out["final_text"])
    print("\nNOTIFY RESULT:\n", out["notify_result"]["message"])
    print("\nBRIEF (first lines):\n", json.dumps({k:out["brief"][k] for k in ["incident_id","type","severity","location"]}, indent=2))



