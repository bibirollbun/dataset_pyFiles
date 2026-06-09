from IPython.display import Image, display

# show the image you uploaded
display(Image("/kaggle/input/image1/architecture.png"))



# === Install the new Google Gen AI SDK (run once in the Kaggle notebook kernel) ===
!pip install -q google-genai

# === Imports ===
import sys, os, time, json, warnings
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# Import the new SDK
from google import genai

# Kaggle secrets helper (optional)
try:
    from kaggle_secrets import UserSecretsClient
except Exception:
    UserSecretsClient = None

from IPython.display import display, HTML, clear_output

print("âœ“ Base libraries loaded; google-genai imported.")



GOOGLE_API_KEY = None
if UserSecretsClient:
    try:
        user_secrets = UserSecretsClient()
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
        print("âœ“ GOOGLE_API_KEY loaded from Kaggle Secrets (not printed).")
    except Exception as e:
        print("âš  Could not load secrets:", e)


if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("âœ“ genai.Client created with provided API key.")
else:
   
    client = genai.Client()
    print("âš  No API key configured. Model calls will fail until you set GOOGLE_API_KEY in Kaggle Secrets.")
    

CONFIG = {
    "team": "GraminHealth",
    "model": "gemini-2.5-flash",   
    "temperature": 0.2,
    "max_tokens": 1500,
    "version": "1.0.0"
}

print("Agent config ready.")
for k, v in CONFIG.items():
    print(f"  â€¢ {k}: {v}")


# Symptom rules engine (custom tool)
SYMPTOM_RULES = [
    {"red_flags": ["chest pain", "shortness of breath", "loss of consciousness", "uncontrolled bleeding"], "triage":"EMERGENCY"},
    {"red_flags": ["high fever >3 days", "persistent vomiting", "severe abdominal pain"], "triage":"SEE_DOCTOR_SOON"},
]

def symptom_rule_checker(symptoms: List[str], notes: str = "") -> Dict[str, Any]:
    """
    Simple deterministic rule engine that flags red flags.
    Returns: {'triage_level': 'SELF_CARE'|'SEE_DOCTOR_SOON'|'EMERGENCY', 'matches': [...]}
    """
    text = " ".join(symptoms).lower() + " " + notes.lower()
    matches = []
    for rule in SYMPTOM_RULES:
        for flag in rule['red_flags']:
            if flag in text:
                matches.append(flag)
    if any(flag in text for rule in SYMPTOM_RULES for flag in rule['red_flags'] if rule['triage']=="EMERGENCY"):
        return {"triage_level":"EMERGENCY","matches":matches}
    if any(flag in text for rule in SYMPTOM_RULES for flag in rule['red_flags'] if rule['triage']=="SEE_DOCTOR_SOON"):
        return {"triage_level":"SEE_DOCTOR_SOON","matches":matches}
    return {"triage_level":"SELF_CARE","matches":matches}

print("âœ“ Symptom rule checker ready")



# Small medical info DB (safe, curated phrases)
MED_INFO_DB = {
    "fever": {
        "explain": "Fever is your body fighting an infection. Rest, hydrate, and take paracetamol if needed. Seek medical care if fever persists >3 days, or if there is difficulty breathing.",
        "self_care": ["Rest", "Drink fluids", "Paracetamol as per local guidelines"],
    },
    "cough": {
        "explain": "Coughing can be from cold, viral infection, or other causes. If cough is severe, producing blood, or accompanied by shortness of breath, see a doctor.",
        "self_care": ["Warm fluids", "Saltwater gargle for sore throat", "Monitor breathing"],
    }
}

def fetch_medical_info(keyword: str) -> Dict[str, Any]:
    key = keyword.lower().strip()
    return MED_INFO_DB.get(key, {"explain":"We don't have a specific entry. Seek local guidelines or consult a provider.","self_care":[]})

print("âœ“ Medical info tool ready")



# Simple translator mapping (localization tool)
TRANSLATIONS = {
    "hi": {
        "Rest": "à¤†à¤°à¤¾à¤® à¤•à¤°à¥‡à¤‚",
        "Drink fluids": "à¤¤à¤°à¤² à¤ªà¤¦à¤¾à¤°à¥�à¤¥ à¤ªà¤¿à¤�à¤‚",
        "Paracetamol as per local guidelines": "à¤¸à¥�à¤¥à¤¾à¤¨à¥€à¤¯ à¤¦à¤¿à¤¶à¤¾à¤¨à¤¿à¤°à¥�à¤¦à¥‡à¤¶à¥‹à¤‚ à¤•à¥‡ à¤…à¤¨à¥�à¤¸à¤¾à¤° à¤ªà¥ˆà¤°à¤¾à¤¸à¤¿à¤Ÿà¤¾à¤®à¥‰à¤² à¤²à¥‡à¤‚",
        "See a doctor": "à¤¡à¥‰à¤•à¥�à¤Ÿà¤° à¤•à¥‡ à¤ªà¤¾à¤¸ à¤œà¤¾à¤�à¤‚"
    }
}

def translate_to_local(text: str, lang: str = "hi") -> str:
    # Very small deterministic mapping for demo; fallback returns original text
    if lang not in TRANSLATIONS:
        return text
    for en, local in TRANSLATIONS[lang].items():
        text = text.replace(en, local)
    return text

print("âœ“ Translator tool ready")



# Reminder scheduler (long-running tool stub)
REMINDERS_DB = []

def schedule_reminder(user_id: str, message: str, when_ts: float) -> Dict[str, Any]:
    entry = {"id": len(REMINDERS_DB)+1, "user_id":user_id, "message":message, "when":when_ts, "sent":False}
    REMINDERS_DB.append(entry)
    return entry

def due_reminders(now_ts: float):
    return [r for r in REMINDERS_DB if not r['sent'] and r['when'] <= now_ts]

def mark_sent(reminder_id:int):
    for r in REMINDERS_DB:
        if r['id']==reminder_id:
            r['sent']=True
            return r
    return None

print("âœ“ Reminder scheduler ready (in-memory demo)")



@dataclass
class LongTermMemory:
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def save_profile(self, user_id: str, profile: Dict[str,Any]):
        existing = self.profiles.get(user_id, {})
        existing.update(profile)
        # compaction example: only keep key summary
        if len(existing.get('history',[])) > 10:
            existing['history'] = existing['history'][-10:]
        self.profiles[user_id] = existing

    def get_profile(self, user_id: str) -> Dict[str,Any]:
        return self.profiles.get(user_id, {})

lt_memory = LongTermMemory()
print("âœ“ Long-term memory ready")



@dataclass
class SessionMemory:
    sessions: Dict[str, List[Dict[str,str]]] = field(default_factory=dict)
    max_len: int = 30

    def add_message(self, session_id: str, role: str, text: str):
        self.sessions.setdefault(session_id, []).append({"role":role,"content":text,"timestamp":datetime.now().isoformat()})
        if len(self.sessions[session_id]) > self.max_len:
            self.sessions[session_id] = self.sessions[session_id][-self.max_len:]

    def get_context(self, session_id: str, last_n: int = 6):
        return self.sessions.get(session_id, [])[-last_n:]

session_memory = SessionMemory()
print("âœ“ Session memory ready")



@dataclass
class AgentLogger:
    logs: List[Dict[str,Any]] = field(default_factory=list)
    def info(self, event: str, details: Dict[str,Any] = None):
        self.logs.append({"level":"INFO","event":event,"details":details or {}, "ts":datetime.now().isoformat()})
    def warn(self, event: str, details: Dict[str,Any] = None):
        self.logs.append({"level":"WARN","event":event,"details":details or {}, "ts":datetime.now().isoformat()})
    def error(self, event: str, details: Dict[str,Any] = None):
        self.logs.append({"level":"ERROR","event":event,"details":details or {}, "ts":datetime.now().isoformat()})
    def recent(self, n=10): return self.logs[-n:]

logger = AgentLogger()
logger.info("logger_initialized")
print("âœ“ Logger ready")



metrics = {"queries":0,"triage_emergency":0,"triage_see_doctor":0,"triage_self_care":0,"reminders_scheduled":0}



class GraminHealthAgent:
    def __init__(self, config, logger, session_memory, longterm):
        self.config = config
        self.logger = logger
        self.session = session_memory
        self.longterm = longterm
        self.model = client if GOOGLE_API_KEY else None
        self.logger.info("agent_initialized", {"team": config.get("team")})

    # ---------------------- INTAKE ----------------------
    def intake(self, session_id: str, user_id: str, message: str):
        """Store user message + extract symptoms."""
        self.session.add_message(session_id, "user", message)
        symptoms = [s.strip() for s in message.replace(".", ",").split(",") if s.strip()]
        if not symptoms:
            symptoms = [message]
        self.logger.info("intake_parsed", {"symptoms": symptoms})
        return symptoms

    # ---------------------- TRIAGE ----------------------
    def triage(self, symptoms: list, notes: str = ""):
        result = symptom_rule_checker(symptoms, notes)
        self.logger.info("triage_result", result)
        return result

    # ---------------------- EDUCATION ----------------------
    def educate(self, primary_symptom: str, triage_level: str):
        info = fetch_medical_info(primary_symptom)
        explanation = info["explain"]
        suggestions = info.get("self_care", [])

        # If emergency â†’ override education
        if triage_level == "EMERGENCY":
            explanation = "This appears serious. Please seek emergency medical care immediately."
            suggestions = ["See a doctor"]

        self.logger.info("education_generated", {
            "symptom": primary_symptom,
            "triage": triage_level
        })
        return {"explanation": explanation, "suggestions": suggestions}

    # ---------------------- LOCALIZATION ----------------------
    def localize(self, text: str, lang="hi"):
        localized = translate_to_local(text, lang)
        self.logger.info("localized_text", {"lang": lang})
        return localized

    # ---------------------- REMINDERS ----------------------
    def schedule_followup(self, user_id: str, text: str, delay_hours: int = 24):
        when = time.time() + delay_hours * 3600
        reminder = schedule_reminder(user_id, text, when)
        metrics["reminders_scheduled"] += 1
        self.logger.info("reminder_scheduled", {"reminder": reminder})
        return reminder

    # ---------------------- MAIN API: handle_user_message ----------------------
    def handle_user_message(self, session_id: str, user_id: str, message: str, lang="hi"):
        symptoms = self.intake(session_id, user_id, message)
        primary = symptoms[0] if symptoms else "general"

        triage_res = self.triage(symptoms, message)
        edu = self.educate(primary, triage_res["triage_level"])
        localized_suggestions = [self.localize(s, lang) for s in edu["suggestions"]]

        reminder = None
        if triage_res["triage_level"] == "SELF_CARE" and localized_suggestions:
            reminder = self.schedule_followup(
                user_id,
                f"Reminder: {localized_suggestions[0]}",
                delay_hours=24
            )

        self.longterm.save_profile(
            user_id,
            {
                "last_visit": datetime.now().isoformat(),
                "last_triage": triage_res["triage_level"],
                "history": [message]
            }
        )

        response = {
            "triage": triage_res["triage_level"],
            "explanation": self.localize(edu["explanation"], lang),
            "suggestions": localized_suggestions,
            "reminder": reminder,
        }

        self.session.add_message(session_id, "agent", json.dumps(response))
        return response

    # ---------------------- NEW: UNIFIED .run() METHOD ----------------------
    def run(self, session_id: str, user_id: str, message: str, lang: str = "hi"):
        """
        Clean, user-friendly API for Kaggle Demos
        """
        return self.handle_user_message(session_id, user_id, message, lang)
agent = GraminHealthAgent(CONFIG, logger, session_memory, lt_memory)
logger.info("agent_initialized", {"team":CONFIG['team']})
print("âœ“ GraminHealthAgent initialized")


# Demo 1: Mild fever case
print("Demo 1: I have fever and headache for 2 days")
resp1 = agent.handle_user_message("sess_1","user_1","fever, headache", lang="hi")
print(json.dumps(resp1, indent=2, ensure_ascii=False))

# Demo 2: Red-flag example
print("\nDemo 2: Chest pain and shortness of breath")
resp2 = agent.handle_user_message("sess_2","user_2","chest pain, shortness of breath", lang="hi")
print(json.dumps(resp2, indent=2, ensure_ascii=False))



def show_dashboard():
    print("=== METRICS ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    print("\n=== RECENT LOGS ===")
    for l in logger.recent(10):
        print(f"[{l['ts']}] {l['level']} - {l['event']} - {l['details']}")

show_dashboard()



TEST_CASES = [
    {"case_id":1, "desc":"Mild fever 1 day","message":"fever, mild", "expected":"SELF_CARE"},
    {"case_id":2, "desc":"High fever 4 days", "message":"high fever, vomiting", "expected":"SEE_DOCTOR_SOON"},
    {"case_id":3, "desc":"Chest pain and sweat","message":"chest pain, sweating, faint", "expected":"EMERGENCY"}
]

def run_evaluation():
    results=[]
    for c in TEST_CASES:
        resp = agent.handle_user_message(f"eval_{c['case_id']}", f"eval_user_{c['case_id']}", c['message'], lang="hi")
        ok = resp['triage']==c['expected']
        results.append({"case":c['case_id'],"expected":c['expected'],"pred":resp['triage'],"ok":ok})
    return results

eval_res = run_evaluation()
print(json.dumps(eval_res, indent=2))



def export_conversation_history(filename="conversation_history.txt"):
    try:
        with open(filename,"w") as f:
            for sid, msgs in session_memory.sessions.items():
                f.write(f"SESSION {sid}\n")
                for m in msgs:
                    f.write(f"[{m['timestamp']}] {m['role'].upper()}: {m['content']}\n")
                f.write("\n")
        print("âœ“ Exported to", filename)
    except Exception as e:
        print("â�Œ Export failed:", e)

def search_conversation(keyword):
    matches=[]
    for sid, msgs in session_memory.sessions.items():
        for idx, m in enumerate(msgs):
            if keyword.lower() in m['content'].lower():
                matches.append({"session":sid,"idx":idx,"role":m['role'],"content":m['content']})
    return matches

def reset_agent():
    session_memory.sessions.clear()
    lt_memory.profiles.clear()
    REMINDERS_DB.clear()
    metrics.update({"queries":0,"triage_emergency":0,"triage_see_doctor":0,"triage_self_care":0,"reminders_scheduled":0})
    logger.logs.clear()
    print("âœ“ Agent state reset")

def batch_query(queries):
    out={}
    for q in queries:
        out[q]=agent.handle_user_message("batch", "batch_user", q, lang="hi")
    return out

def collect_feedback(entry):
    if not hasattr(agent,'feedback'): agent.feedback=[]
    agent.feedback.append({"ts":datetime.now().isoformat(),"entry":entry})
    return agent.feedback[-1]

def validate_response(question, response_text):
    checks = {"min_length":len(response_text)>30, "contains_action":"recommend" in response_text.lower() or "see" in response_text.lower()}
    return {"score":sum(checks.values()), "checks":checks}

print("âœ“ Utilities ready")


