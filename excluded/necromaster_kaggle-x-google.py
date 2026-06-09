# ===== 2. Imports & configuration =====

import os, re, json, time, hashlib, sqlite3, logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

# Basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("triage")

# Paths (Kaggle-friendly)
ROOT = "/kaggle/working"
MODELS_DIR = os.path.join(ROOT, "models")
LOGS_DIR = os.path.join(ROOT, "logs")
DB_PATH = os.path.join(ROOT, "memory.db")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Safety constants
CRISIS_KEYWORDS = {
    "kill", "suicide", "die", "end it", "hurt myself",
    "can't go on", "cant go on", "i'll end it", "i will end it",
    "kill myself"
}
RISK_THRESHOLDS = {"LOW": 0.30, "MEDIUM": 0.60}  # <0.3 LOW, <0.6 MED, else HIGH

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def anonymize(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]

log.info("Environment configured. ROOT=%s", ROOT)



# ===== 3. Gemini setup (Kaggle + UserSecretsClient) =====
USE_GEMINI = False
gemini_client = None

try:

    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    # Optional: if you ever need Vertex auth
    gcloud_auth = user_secrets.get_secret("__gcloud_sdk_auth__")  # not used in this notebook
    
    # Main: Google AI (Gemini) API key
    api_key = user_secrets.get_secret("GOOGLE_API_KEYY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY secret not set in Kaggle.")
    
    # 2) Import google-genai and create client with explicit api_key
    # If google-genai is not installed, uncomment the pip line once:
    # !pip install -q google-genai
    
    from google import genai
    from google.genai import types
    
    gemini_client = genai.Client(api_key=api_key)
    USE_GEMINI = True
    log.info("Gemini client initialized with GOOGLE_API_KEY from Kaggle secrets. USE_GEMINI=True")
    
except Exception as e:
    USE_GEMINI = False
    gemini_client = None
    log.warning(
        "Gemini NOT configured. Falling back to rule-based only.\n"
        "Make sure you created a Kaggle secret named 'GOOGLE_API_KEY'. Error: %s", e
    )


def gemini_json_call(system_prompt: str, user_payload: Dict[str, Any], model_id: str = "gemini-2.5-flash") -> Optional[Dict[str, Any]]:
    """
    Call Gemini with system + user JSON, expecting JSON output.
    Returns parsed dict or None on failure.
    """
    if not USE_GEMINI or gemini_client is None:
        return None

    try:
        content = [
            types.Content(
                role="system",
                parts=[types.Part.from_text(text=system_prompt)]
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=json.dumps(user_payload))]
            ),
        ]

        resp = gemini_client.models.generate_content(
            model=model_id,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        text = resp.text.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except Exception:
            # Fallback: extract {...} block
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
    except Exception as e:
        log.warning("Gemini JSON call failed: %s", e)
    return None



# ===== 4. Dataset: Synthetic fallback  =====

data = [
    ("I want to kill myself", "suicidal", "distress", "HIGH"),
    ("I can't go on anymore", "suicidal", "distress", "HIGH"),
    ("I have a plan to end my life", "suicidal", "distress", "HIGH"),
    ("I've been feeling hopeless and thinking about death", "help_seeking", "sadness", "MEDIUM"),
    ("I'm feeling really down and can't sleep", "help_seeking", "sadness", "MEDIUM"),
    ("Sometimes I think about hurting myself but I'm not sure", "help_seeking", "distress", "MEDIUM"),
    ("I'm tired of work, it's stressful", "vent", "anger", "LOW"),
    ("I had an argument and I'm upset", "vent", "sadness", "LOW"),
    ("I feel a bit down but okay", "help_seeking", "sadness", "LOW"),
]
df = pd.DataFrame(data, columns=["text","intent","emotion","gold_risk"])
print("Using synthetic dataset, size:", len(df))
display(df)

# To use a real dataset in Kaggle:
# 1. Add a dataset via "Add data" in the right panel.
# 2. Load it, rename columns to match: text,intent,emotion,gold_risk.
# Example:
# df_real = pd.read_csv('/kaggle/input/.../file.csv')
# df = df_real.rename(columns={"your_text_col":"text", "your_intent_col":"intent", ...})



# ===== 5. Train baseline intent & emotion classifiers =====

X = df["text"].astype(str)
y_intent = df["intent"].astype(str)
y_emo = df["emotion"].astype(str)

intent_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
intent_pipe.fit(X, y_intent)

emo_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
emo_pipe.fit(X, y_emo)

joblib.dump(intent_pipe, os.path.join(MODELS_DIR, "intent_pipe.joblib"))
joblib.dump(emo_pipe, os.path.join(MODELS_DIR, "emo_pipe.joblib"))
log.info("Trained & saved intent & emotion models.")



# ===== 6. Tools: SQLite memory store & mock notifier =====

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            anon_id TEXT,
            data_json TEXT,
            last_active TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            anon_id TEXT,
            mkey TEXT,
            mvalue TEXT,
            ts TEXT,
            PRIMARY KEY (anon_id, mkey)
        )
    """)
    conn.commit()
    return conn

DB = init_db()
log.info("SQLite DB initialized at %s", DB_PATH)

def save_session(session_id: str, anon_id: str, data: Dict[str,Any]):
    cur = DB.cursor()
    cur.execute(
        "REPLACE INTO sessions (session_id, anon_id, data_json, last_active) VALUES (?, ?, ?, ?)",
        (session_id, anon_id, json.dumps(data), now_iso())
    )
    DB.commit()

def load_session(session_id: str) -> Dict[str,Any]:
    cur = DB.cursor()
    cur.execute("SELECT data_json FROM sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else {}

def save_memory(anon_id: str, key: str, value: Any):
    cur = DB.cursor()
    cur.execute(
        "REPLACE INTO memories (anon_id, mkey, mvalue, ts) VALUES (?, ?, ?, ?)",
        (anon_id, key, json.dumps(value), now_iso())
    )
    DB.commit()

def load_memory(anon_id: str, key: str):
    cur = DB.cursor()
    cur.execute("SELECT mvalue FROM memories WHERE anon_id=? AND mkey=?", (anon_id, key))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

# mock notifier tool (Day 2 - tools)
NOTIFICATIONS = []
def notify_moderator(event: Dict[str,Any]):
    NOTIFICATIONS.append(event)
    alerts_path = os.path.join(LOGS_DIR, "moderator_alerts.jsonl")
    with open(alerts_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    log.warning("Moderator notified: %s", event.get("reason", "no reason"))



# ===== 7. Agents: Intake, Memory, Classifier =====

# Intake Agent
def intake_agent(raw_user_id: str, session_id: str, text: str) -> Dict[str,Any]:
    anon_id = anonymize(raw_user_id)
    norm = re.sub(r"\s+", " ", text.strip())
    payload = {
        "ts": now_iso(),
        "anon_id": anon_id,
        "session_id": session_id,
        "raw_text": text,
        "text": norm,
        "lang": "en"  # placeholder
    }
    # bootstrap session if new
    sess = load_session(session_id) or {}
    if not sess:
        save_session(session_id, anon_id, {"features": {}, "created": now_iso()})
    return payload

# Memory Agent
def memory_agent_fetch(anon_id: str, session_id: str) -> Dict[str,Any]:
    sess = load_session(session_id) or {}
    features = sess.get("features", {})
    recent_high_count = load_memory(anon_id, "recent_high_count") or 0
    return {"features": features, "recent_high_count": recent_high_count}

def memory_agent_update_after_interaction(anon_id: str, session_id: str, level: str):
    if level == "HIGH":
        prev = load_memory(anon_id, "recent_high_count") or 0
        save_memory(anon_id, "recent_high_count", prev + 1)
    sess = load_session(session_id) or {}
    f = sess.get("features", {})
    f["last_level"] = level
    sess["features"] = f
    save_session(session_id, anon_id, sess)

# Classifier Agent (local ML tool)
INTENT_MODEL = joblib.load(os.path.join(MODELS_DIR, "intent_pipe.joblib"))
EMO_MODEL = joblib.load(os.path.join(MODELS_DIR, "emo_pipe.joblib"))

def classifier_agent(text: str) -> Dict[str,Any]:
    intent = INTENT_MODEL.predict([text])[0]
    emotion = EMO_MODEL.predict([text])[0]
    return {"intent": intent, "emotion": emotion}






# ===== 9. Templates, Supervisor Agent, Evaluator Agent =====

def escalation_template():
    return (
        "I’m really sorry you’re feeling this way. I’m not a clinician, but I’m concerned for your safety. "
        "If you are in immediate danger, please call your local emergency number now. "
        "Would you like crisis resources or to connect with a human moderator?"
    )

def medium_template():
    return (
        "I’m sorry you’re struggling — thank you for sharing. "
        "Would you like a short grounding exercise or some resources on coping strategies?"
    )

def low_template():
    return (
        "Thanks for sharing. Can you tell me a bit more about what’s been happening recently?"
    )

# --- Gemini-based supervisor ---

def supervisor_agent_gemini(level: str, reply: str) -> Optional[Dict[str,Any]]:
    """
    Ask Gemini to check if reply is safe / non-clinical.
    Returns possibly modified reply & notes or None.
    """
    if not USE_GEMINI:
        return None

    system_prompt = (
        "You are a safety supervisor for a mental health chatbot.\n"
        "Your job is to review the reply and:\n"
        "- Ensure it does NOT give medical, diagnostic, or medication advice.\n"
        "- Ensure it DOES encourage reaching out to emergency services for HIGH risk.\n"
        "- Ensure tone is supportive and non-judgmental.\n"
        "Return JSON with keys: safe (bool), revised_reply (string), notes (string)."
    )
    user_payload = {
        "risk_level": level,
        "reply": reply
    }
    resp = gemini_json_call(system_prompt, user_payload)
    if not resp:
        return None
    safe = bool(resp.get("safe", True))
    revised = resp.get("revised_reply", reply)
    notes = resp.get("notes", "")
    return {"safe": safe, "reply": revised, "notes": notes}


def supervisor_agent_check(payload: Dict[str,Any]) -> Dict[str,Any]:
    """
    Combine simple rules + optional Gemini supervisor.
    """
    level = payload.get("level")
    reply = payload.get("reply","")
    supervisor_flag = False
    supervisor_note = ""

    # Rule: HIGH must contain emergency/urge to seek immediate help
    if level == "HIGH" and ("emergency" not in reply.lower() and "call" not in reply.lower()):
        reply = escalation_template()
        supervisor_flag = True
        supervisor_note = "Rule-based: enforced HIGH-risk escalation template."

    # Rule: avoid mentioning meds/diagnosis
    if re.search(r"\b(prescribe|medication|dosage|diagnose)\b", reply.lower()):
        reply = medium_template()
        supervisor_flag = True
        supervisor_note += " Removed potential clinical phrasing."

    # Gemini pass (if available)
    g_out = supervisor_agent_gemini(level, reply)
    if g_out:
        if not g_out["safe"]:
            supervisor_flag = True
            supervisor_note += " Gemini flagged reply as unsafe; using revised version."
        reply = g_out["reply"]
        if g_out["notes"]:
            supervisor_note += " " + g_out["notes"]

    payload["reply"] = reply
    payload["supervisor_flag"] = supervisor_flag
    payload["supervisor_note"] = supervisor_note.strip()
    return payload


# --- Evaluator Agent (Gemini + fallback) ---

def evaluator_agent_rule(message: str, level: str, reply: str) -> Dict[str,Any]:
    # Simple heuristic evaluator if Gemini is not available
    safety = 1.0 if ("emergency" in reply.lower() or "call" in reply.lower() or level != "HIGH") else 0.7
    empathy_tokens = ["sorry", "i'm sorry", "i am sorry", "i’m sorry"]
    empathy = 1.0 if any(t in reply.lower() for t in empathy_tokens) else 0.6
    return {"safety_score": safety, "empathy_score": empathy, "overall": (safety+empathy)/2, "source": "rule"}


def evaluator_agent_gemini(message: str, level: str, reply: str) -> Optional[Dict[str,Any]]:
    if not USE_GEMINI:
        return None

    system_prompt = (
        "You are evaluating a mental health chatbot reply.\n"
        "Rate from 0.0 to 1.0:\n"
        "- safety_score: Is the reply safe and non-clinical? (0=very unsafe, 1=fully safe)\n"
        "- empathy_score: Is the reply empathic and non-judgmental?\n"
        "Return JSON ONLY with keys: safety_score, empathy_score."
    )
    user_payload = {
        "risk_level": level,
        "user_message": message,
        "bot_reply": reply
    }
    resp = gemini_json_call(system_prompt, user_payload)
    if not resp:
        return None
    try:
        s = float(resp.get("safety_score", 0.0))
        e = float(resp.get("empathy_score", 0.0))
        s = max(0.0, min(1.0, s))
        e = max(0.0, min(1.0, e))
        return {"safety_score": s, "empathy_score": e, "overall": (s+e)/2, "source": "gemini"}
    except Exception:
        return None


def evaluator_agent(message: str, level: str, reply: str) -> Dict[str,Any]:
    g = evaluator_agent_gemini(message, level, reply)
    if g:
        return g
    return evaluator_agent_rule(message, level, reply)



# # ===== 3. Gemini setup (Kaggle + UserSecretsClient) =====
# # This cell REPLACES the previous Gemini setup cell.

# USE_GEMINI = False
# gemini_client = None

# try:
#     # 1) Get secrets from Kaggle
#     from kaggle_secrets import UserSecretsClient
#     user_secrets = UserSecretsClient()
    
#     # Optional: if you ever need Vertex auth
#     gcloud_auth = user_secrets.get_secret("__gcloud_sdk_auth__")  # not used in this notebook
    
#     # Main: Google AI (Gemini) API key
#     api_key = user_secrets.get_secret("GOOGLE_API_KEYY")
    
#     if not api_key:
#         raise ValueError("GOOGLE_API_KEY secret not set in Kaggle.")
    
#     # 2) Import google-genai and create client with explicit api_key
#     # If google-genai is not installed, uncomment the pip line once:
#     # !pip install -q google-genai
    
#     from google import genai
#     from google.genai import types
    
#     gemini_client = genai.Client(api_key=api_key)
#     USE_GEMINI = True
#     log.info("Gemini client initialized with GOOGLE_API_KEY from Kaggle secrets. USE_GEMINI=True")
    
# except Exception as e:
#     USE_GEMINI = False
#     gemini_client = None
#     log.warning(
#         "Gemini NOT configured. Falling back to rule-based only.\n"
#         "Make sure you created a Kaggle secret named 'GOOGLE_API_KEY'. Error: %s", e
#     )


# def gemini_json_call(system_prompt: str, user_payload: Dict[str, Any], model_id: str = "gemini-2.5-flash") -> Optional[Dict[str, Any]]:
#     """
#     Call Gemini with system + user JSON, expecting JSON output.
#     Returns parsed dict or None on failure.
#     """
#     if not USE_GEMINI or gemini_client is None:
#         return None

#     try:
#         content = [
#             types.Content(
#                 role="system",
#                 parts=[types.Part.from_text(text=system_prompt)]
#             ),
#             types.Content(
#                 role="user",
#                 parts=[types.Part.from_text(text=json.dumps(user_payload))]
#             ),
#         ]

#         resp = gemini_client.models.generate_content(
#             model=model_id,
#             contents=content,
#             config=types.GenerateContentConfig(
#                 temperature=0.1,
#                 max_output_tokens=512,
#             ),
#         )
#         text = resp.text.strip()
        
#         # Try direct JSON parse first
#         try:
#             return json.loads(text)
#         except Exception:
#             # Fallback: extract {...} block
#             m = re.search(r"\{.*\}", text, re.DOTALL)
#             if m:
#                 return json.loads(m.group(0))
#     except Exception as e:
#         log.warning("Gemini JSON call failed: %s", e)
#     return None



# ==== Response Agent (Gemini refinement) ====
def response_agent_gemini(message: str, level: str, base_reply: str) -> str:
    """
    Optionally refine the base templated reply using Gemini.
    - Keeps all safety constraints.
    - For HIGH risk: MUST keep emergency / escalation content.
    - If Gemini unavailable or fails (quota, etc.), returns base_reply unchanged.
    """
    # If Gemini is disabled, just return the template
    if not USE_GEMINI or gemini_client is None:
        return base_reply

    system_prompt = """
You are a supportive, non-clinical mental health assistant.

You receive:
- user_message: what the user said
- risk_level: LOW, MEDIUM, or HIGH
- base_reply: a safe template that ALREADY follows safety rules.

Your task:
- Make the reply slightly more conversational and empathetic.
- NEVER give medical, diagnostic, or medication advice.
- NEVER remove any crisis / emergency / escalation language in HIGH-risk replies.
- For HIGH risk: you MUST still clearly encourage contacting emergency services or crisis help.
- For LOW/MEDIUM: you may gently rephrase but keep the meaning and safety.

Return JSON ONLY with:
{
  "refined_reply": "<your final reply string>"
}
    """.strip()

    user_payload = {
        "user_message": message,
        "risk_level": level,
        "base_reply": base_reply
    }

    resp = gemini_json_call(system_prompt, user_payload)
    if not resp:
        # Quota, network, parse error → fallback
        return base_reply

    refined = resp.get("refined_reply", "").strip()
    if not refined:
        return base_reply

    # Light sanity check: in HIGH risk, ensure we still talk about emergency/call/etc.
    if level == "HIGH" and ("emergency" not in refined.lower() and "call" not in refined.lower()):
        # Better to be safe and keep original escalation text
        return base_reply

    return refined



# ===== 10. Orchestrator: Hybrid flow (parallel + sequential) =====

INTERACTIONS_LOG = os.path.join(LOGS_DIR, "interactions.jsonl")
METRICS = {"total": 0, "high_alerts": 0}
executor = ThreadPoolExecutor(max_workers=4)

def log_interaction(record: Dict[str,Any]):
    with open(INTERACTIONS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

def update_metrics(level: str):
    METRICS["total"] += 1
    if level == "HIGH":
        METRICS["high_alerts"] += 1

def orchestrator_handle(raw_user_id: str, session_id: str, text: str) -> Dict[str,Any]:
    # 1) Intake
    intake = intake_agent(raw_user_id, session_id, text)
    anon_id = intake["anon_id"]
    text_norm = intake["text"]

    # 2) Parallel: classifier + memory
    fut_cls = executor.submit(classifier_agent, text_norm)
    fut_mem = executor.submit(memory_agent_fetch, anon_id, session_id)
    cls = fut_cls.result()
    mem = fut_mem.result()

    keywords = [kw for kw in CRISIS_KEYWORDS if kw in text_norm.lower()]

    # 3) Local risk score
    local_risk = compute_risk_score_local(cls["intent"], cls["emotion"], keywords, mem)
    level = local_risk["level"]
    score = local_risk["score"]
    reason = local_risk["reason"]

    # 4) Gemini risk reasoner (optional override, but only to MORE cautious)
    g_risk = gemini_risk_reasoner(text_norm, cls["intent"], cls["emotion"], keywords, mem)
    gemini_risk_used = False
    if g_risk:
        gem_level = g_risk["level"]
        # define ordering: LOW < MEDIUM < HIGH
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if order.get(gem_level, 0) > order.get(level, 0):
            # increase risk level based on Gemini
            level = gem_level
            reason += f" | gemini_upgraded_to={gem_level}"
            gemini_risk_used = True

    # 5) Choose base reply template
    if level == "HIGH":
        base_reply = escalation_template()
        notify_moderator({
            "ts": now_iso(),
            "anon_id": anon_id,
            "session_id": session_id,
            "message": text_norm,
            "reason": "HIGH risk detected",
            "risk_score": score
        })
    elif level == "MEDIUM":
        base_reply = medium_template()
    else:
        base_reply = low_template()

    # Optional Gemini refinement to make it more conversational
    reply = response_agent_gemini(text_norm, level, base_reply)


    payload = {
        "ts": now_iso(),
        "anon_id": anon_id,
        "session_id": session_id,
        "text": text_norm,
        "intent": cls["intent"],
        "emotion": cls["emotion"],
        "keywords": keywords,
        "risk_score": score,
        "level": level,
        "reason": reason,
        "gemini_risk_used": gemini_risk_used,
        "reply": reply
    }

    # 6) Supervisor (rules + Gemini)
    payload = supervisor_agent_check(payload)

    # 7) Memory + metrics
    memory_agent_update_after_interaction(anon_id, session_id, payload["level"])
    update_metrics(payload["level"])

    # 8) Evaluator (Gemini + fallback)
    payload["eval"] = evaluator_agent(payload["text"], payload["level"], payload["reply"])

    # 9) Log
    log_interaction(payload)
    return payload

# quick smoke test
out = orchestrator_handle("userA", "session1", "I want to kill myself right now")
print(json.dumps(out, indent=2))
print("METRICS:", METRICS)



# ===== 11. Admin view: logs & moderator alerts =====

def load_interactions(limit: int = 200) -> pd.DataFrame:
    rows = []
    if not os.path.exists(INTERACTIONS_LOG):
        return pd.DataFrame(rows)
    with open(INTERACTIONS_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except:
                pass
    df_logs = pd.DataFrame(rows)
    if df_logs.empty:
        return df_logs
    df_logs["ts"] = pd.to_datetime(df_logs["ts"])
    df_logs = df_logs.sort_values("ts", ascending=False).reset_index(drop=True)
    return df_logs.head(limit)

logs_df = load_interactions()
if not logs_df.empty:
    display(logs_df[["ts","anon_id","session_id","text","level","risk_score","supervisor_flag"]].head(20))
else:
    print("No interactions logged yet.")

alerts_path = os.path.join(LOGS_DIR, "moderator_alerts.jsonl")
alerts = []
if os.path.exists(alerts_path):
    with open(alerts_path, "r", encoding="utf-8") as fh:
        for l in fh:
            alerts.append(json.loads(l))
alerts_df = pd.DataFrame(alerts)
if not alerts_df.empty:
    print("\nModerator alerts:")
    display(alerts_df[["ts","anon_id","session_id","reason","risk_score"]].head(20))
else:
    print("\nNo moderator alerts yet.")

print("\nMETRICS:", METRICS)



# ===== 12. Evaluation harness on labeled dataset =====

def evaluate_on_df(df_eval: pd.DataFrame):
    preds = []
    for i, row in df_eval.iterrows():
        out = orchestrator_handle("eval_user", f"eval_sess_{i}", row["text"])
        preds.append(out["level"])
    df_eval = df_eval.copy()
    df_eval["predicted"] = preds

    y_true = (df_eval["gold_risk"] == "HIGH").astype(int)
    y_pred = (df_eval["predicted"] == "HIGH").astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Precision (HIGH):", precision)
    print("Recall (HIGH):", recall)
    print("Confusion matrix (true HIGH vs pred HIGH):\n", cm)
    print("\nFull classification report:\n")
    print(classification_report(df_eval["gold_risk"], df_eval["predicted"], zero_division=0))
    return df_eval

eval_df = evaluate_on_df(df)
display(eval_df[["text","gold_risk","predicted"]])



# ===== Simple conversation demo  =====

# Choose one user + session to simulate a "patient" conversation
DEMO_USER_ID = "demo-user-1"
DEMO_SESSION_ID = "demo-session-1"

def chat_once(user_message: str):
    """
    Sends one message through the whole agentic pipeline
    and prints it in chat format.
    """
    out = orchestrator_handle(DEMO_USER_ID, DEMO_SESSION_ID, user_message)
    print(f"You:  {user_message}")
    print(f"Bot:  {out['reply']}")
    print(f"(Level={out['level']}, Score={out['risk_score']:.2f}, Eval={out['eval']})")
    print("-" * 80)
    return out

# Example multi-turn conversation (run this cell multiple times, editing the list):
messages = [
    "I don't know what to do anymore.",
    "Sometimes I think about hurting myself.",
    "I also feel like no one would understand.",
    "Thanks for listening."
]

for msg in messages:
    chat_once(msg)



fastapi_snippet = r"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatIn(BaseModel):
    raw_user_id: str
    session_id: str
    message: str

@app.post("/chat")
def chat_endpoint(payload: ChatIn):
    return orchestrator_handle(payload.raw_user_id, payload.session_id, payload.message)
"""
print("FastAPI reference snippet (not executed here).")


