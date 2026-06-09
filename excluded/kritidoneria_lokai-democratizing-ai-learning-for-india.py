!pip install --quiet google-genai pandas openpyxl
print('Install complete (if needed).')



import os
import json
import sqlite3
import datetime
import re
from typing import Any, Dict, Optional
from collections import Counter

try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

print('GENAI_AVAILABLE =', GENAI_AVAILABLE)




class LLMClient:
    """
    Simple Gemini-first client wrapper. Requires GOOGLE_API_KEY in environment for live calls.
    """
    def __init__(self, model: str = "gemini-2.5", api_key_env: str = "GOOGLE_API_KEY"):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.client = None
        if GENAI_AVAILABLE:
            Client = getattr(genai, "Client", None)
            try:
                if Client is not None:
                    self.client = Client(api_key=self.api_key) if self.api_key else Client()
            except Exception:
                try:
                    self.client = Client()
                except Exception:
                    self.client = None

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
        if not self.client:
            return "[llm_unavailable] Gemini client not configured. Set GOOGLE_API_KEY to enable."
        try:
            resp = self.client.models.generate_content(model=self.model, contents=[prompt])
            if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                return resp.candidates[0].content.strip()
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            return str(resp)
        except Exception as e:
            try:
                resp = self.client.generate_text(model=self.model, prompt=prompt)
                if isinstance(resp, dict) and "candidates" in resp:
                    return resp["candidates"][0]["content"].strip()
                return str(resp)
            except Exception as e2:
                return f"[llm_error] {e} | fallback: {e2}"




DB_PATH = "lokai_memory.sqlite"

class MemoryStore:
    """
    Lightweight SQLite-backed memory for sessions and user profiles.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_seen TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TEXT,
                data TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (user_id, key)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                level TEXT,
                component TEXT,
                message TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                name TEXT PRIMARY KEY,
                value REAL
            )
        """)
        conn.commit()
        conn.close()

    def upsert_user(self, user_id: str):
        now = datetime.datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (user_id, created_at, last_seen) VALUES (?, ?, ?)", (user_id, now, now))
        c.execute("UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id))
        conn.commit()
        conn.close()

    def save_session(self, session_id: str, user_id: str, data: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO sessions (session_id, user_id, created_at, data) VALUES (?, ?, ?, ?)", (session_id, user_id, datetime.datetime.utcnow().isoformat(), json.dumps(data, ensure_ascii=False)))
        conn.commit()
        conn.close()

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    def set_profile(self, user_id: str, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO user_profile (user_id, key, value) VALUES (?, ?, ?)", (user_id, key, value))
        conn.commit()
        conn.close()

    def get_profile(self, user_id: str) -> Dict[str, str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT key, value FROM user_profile WHERE user_id = ?", (user_id,))
        rows = c.fetchall()
        conn.close()
        return {k: v for k, v in rows}




ROADMAP_PROMPT_TEMPLATE = (
    "You are a curriculum planning assistant. Given the user info, produce a JSON roadmap with: "
    "goal (string), timeline_weeks (int), milestones (list of {week_range, title, objectives:list[str], hours:int, resources:list[{title,url,type,description}]}) . "
    "Be concise and return only JSON.\n\nUser info:\n{user_info}\n\nConstraints:\n- Keep timeline realistic given hours/week.\n- Include at least one small project per milestone when possible."
)




def generate_roadmap(llm: LLMClient, user_info: Dict[str, Any], temperature: float = 0.0, max_tokens: int = 900) -> Dict[str, Any]:
    prompt = ROADMAP_PROMPT_TEMPLATE.format(user_info=json.dumps(user_info, ensure_ascii=False))
    raw = llm.call(prompt, temperature=temperature, max_tokens=max_tokens)
    try:
        start = raw.find("{")
        json_text = raw[start:]
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        return {"goal": user_info.get("goal", ""), "notes": raw}




import pandas as pd

VERB_KEYWORDS = ['build','implement','practice','complete','learn','apply','create','design','develop','explore','read','solve']

def contains_actionable_objective(obj_text: str) -> bool:
    txt = (obj_text or "").lower()
    return any(kw in txt for kw in VERB_KEYWORDS)

def evaluate_roadmap_basic(roadmap: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
    total_hours = sum(int(m.get("hours", 0) or 0) for m in roadmap.get("milestones", []))
    availability = user_info.get("hours_per_week", 5)
    estimated_weeks = roadmap.get("timeline_weeks") or max(1, int(total_hours / max(1, availability)))
    feasible = estimated_weeks <= user_info.get("max_weeks", estimated_weeks)
    per_milestone = []
    for idx, m in enumerate(roadmap.get("milestones", []), start=1):
        resources = m.get("resources", []) or []
        has_project = any((r.get("type","").lower() == "project" or "project" in r.get("title","").lower()) for r in resources)
        objectives = m.get("objectives", []) or []
        actionable_count = sum(1 for o in objectives if contains_actionable_objective(o))
        per_milestone.append({
            "milestone_index": idx,
            "title": m.get("title",""),
            "n_resources": len(resources),
            "has_project_resource": has_project,
            "total_objectives": len(objectives),
            "actionable_objectives": actionable_count
        })
    summary = {"total_hours": total_hours, "estimated_weeks": estimated_weeks, "feasible": feasible, "n_milestones": len(roadmap.get("milestones", []))}
    return {"summary": summary, "per_milestone": pd.DataFrame(per_milestone)}




EXTENDED_TRANSLATIONS = {
    'hi': {'Foundations':'à¤¬à¥�à¤¨à¤¿à¤¯à¤¾à¤¦à¥€', 'Applied':'à¤µà¥�à¤¯à¤¾à¤µà¤¹à¤¾à¤°à¤¿à¤•', 'Project':'à¤ªà¤°à¤¿à¤¯à¥‹à¤œà¤¨à¤¾', 'Intro':'à¤ªà¤°à¤¿à¤šà¤¯', 'Basics':'à¤¬à¥�à¤¨à¤¿à¤¯à¤¾à¤¦à¥€'},
    'bn': {'Foundations':'à¦­à¦¿à¦¤à§�à¦¤à¦¿', 'Applied':'à¦ªà§�à¦°à¦¯à¦¼à§‹à¦—', 'Project':'à¦ªà§�à¦°à¦•à¦²à§�à¦ª', 'Intro':'à¦ªà¦°à¦¿à¦šà¦¯à¦¼', 'Basics':'à¦­à¦¿à¦¤à§�à¦¤à¦¿'}
}

def translate_text_via_gemini(text: str, target_language: str, model: str = "gemini-2.5") -> Optional[str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        Client = getattr(genai, 'Client', None)
        client = Client(api_key=api_key) if Client is not None else None
        if client and hasattr(client, 'models'):
            prompt = f"Translate the following text to the language code '{target_language}'. Return ONLY the translated text, do not add commentary.\n\nText:\n{text}"
            resp = client.models.generate_content(model=model, contents=[prompt])
            if hasattr(resp, 'candidates') and len(resp.candidates)>0:
                return resp.candidates[0].content.strip()
            if hasattr(resp, 'text') and resp.text:
                return resp.text.strip()
            return str(resp)
    except Exception:
        return None

def localize_roadmap_with_gemini(roadmap: Dict[str, Any], language: str, use_gemini: bool = True) -> Dict[str, Any]:
    from copy import deepcopy
    new_rm = deepcopy(roadmap)
    if not language or language == 'en':
        return new_rm
    mapping = EXTENDED_TRANSLATIONS.get(language, {})
    for m in new_rm.get('milestones', []):
        for eng, trans in mapping.items():
            if eng in m.get('title',''):
                m['title'] = m['title'].replace(eng, trans)
        objectives = m.get('objectives', [])
        if objectives:
            joined = ' ; '.join(objectives)
            translated = None
            if use_gemini:
                translated = translate_text_via_gemini(joined, language)
            if translated:
                parts = [p.strip() for p in translated.split(';') if p.strip()]
                if len(parts) == len(objectives):
                    m['objectives'] = parts
                else:
                    m['objectives'] = [translated]
            else:
                new_objs = []
                for obj in objectives:
                    s = obj
                    for eng, trans in mapping.items():
                        if eng in s:
                            s = s.replace(eng, trans)
                    new_objs.append(s)
                m['objectives'] = new_objs
        for r in m.get('resources', []):
            title = r.get('title','')
            translated_title = None
            if use_gemini and title:
                translated_title = translate_text_via_gemini(title, language)
            if translated_title:
                r['title'] = translated_title
            else:
                for eng, trans in mapping.items():
                    if eng in title:
                        r['title'] = title.replace(eng, trans)
            desc = r.get('description') or r.get('summary') or ''
            if desc:
                translated_desc = None
                if use_gemini:
                    translated_desc = translate_text_via_gemini(desc, language)
                if translated_desc:
                    r['description_translated'] = translated_desc
                else:
                    s = desc
                    for eng, trans in mapping.items():
                        if eng in s:
                            s = s.replace(eng, trans)
                    r['description_translated'] = s
    return new_rm




ASL_GLOSS = {
    'Foundations': 'FOUNDATION',
    'Applied': 'APPLY',
    'Project': 'PROJECT',
    'Intro': 'INTRO',
    'Basics': 'BASIC'
}

def add_asl_gloss(roadmap: Dict[str, Any]) -> Dict[str, Any]:
    from copy import deepcopy
    import re
    new_rm = deepcopy(roadmap)
    for m in new_rm.get('milestones', []):
        title = m.get('title','')
        gloss_tokens = []
        for token in re.findall(r"[A-Za-z]+", title):
            gloss_tokens.append(ASL_GLOSS.get(token, token.upper()))
        m['asl_gloss'] = ' '.join(gloss_tokens)
    if 'asl_resources' not in new_rm:
        new_rm['asl_resources'] = [
            {'title': 'ASL University (Lifeprint)', 'url': 'https://www.lifeprint.com/', 'type': 'resource'},
            {'title': 'Gallaudet University ASL resources', 'url': 'https://www.gallaudet.edu/','type': 'resource'},
            {'title': 'Signing Savvy (dictionary)', 'url': 'https://www.signingsavvy.com/','type': 'resource'}
        ]
    return new_rm




import pandas as pd

def export_roadmap_to_excel(roadmap: Dict[str, Any], filename: str = "roadmap_output.xlsx", max_resources_per_milestone: int = 3):
    rows = []
    for idx, m in enumerate(roadmap.get("milestones", []), start=1):
        row = {
            "milestone_index": idx,
            "week_range": m.get("week_range",""),
            "title": m.get("title",""),
            "objectives": "; ".join(m.get("objectives", [])),
            "hours": m.get("hours","")
        }
        resources = m.get("resources", []) or []
        for i in range(max_resources_per_milestone):
            j = i + 1
            if i < len(resources):
                r = resources[i]
                row[f"resource_{j}_title"] = r.get("title","")
                row[f"resource_{j}_url"] = r.get("url","")
                row[f"resource_{j}_type"] = r.get("type","")
            else:
                row[f"resource_{j}_title"] = ""
                row[f"resource_{j}_url"] = ""
                row[f"resource_{j}_type"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    return filename




import re, os
from collections import Counter

def compact_sessions_summary(memory: MemoryStore, user_id: str, use_gemini: bool = False, max_chars: int = 3000) -> str:
    db_path = getattr(memory, "db_path", None) or "lokai_memory.sqlite"
    combined_texts = []
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT data FROM sessions WHERE user_id = ?", (user_id,))
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        print("Failed to read sessions from DB:", e)
        rows = []
    for r in rows:
        try:
            d = json.loads(r[0])
            if isinstance(d, dict) and 'roadmap' in d and isinstance(d['roadmap'], dict):
                parts = []
                for m in d['roadmap'].get('milestones', []):
                    title = m.get('title','')
                    objectives = m.get('objectives', []) or []
                    parts.append(f"{title}: " + "; ".join(objectives))
                if parts:
                    combined_texts.append(" | ".join(parts))
        except Exception:
            continue
    combined = "\n".join(combined_texts).strip()
    if not combined:
        summary = "Compact profile: (no historical sessions found)"
    else:
        summary = None
        if use_gemini and globals().get("GENAI_AVAILABLE") and os.environ.get("GOOGLE_API_KEY"):
            try:
                llm = LLMClient()
                prompt = "Summarize the following learner history into a short profile (1-2 sentences) including strengths, gaps, and learning preferences. Return ONLY the summary.\n\n" + combined[:max_chars]
                resp = llm.call(prompt)
                if resp and isinstance(resp, str) and resp.strip():
                    summary = resp.strip()
            except Exception as e:
                print("Gemini summarization failed:", e)
                summary = None
        if not summary:
            words = re.findall(r"\w+", combined.lower())
            if words:
                cnt = Counter(words)
                common = ", ".join([w for w, _ in cnt.most_common(12)])
            else:
                common = ""
            sample_lines = []
            for part in combined_texts[:6]:
                snippet = part if len(part) < 200 else part[:197] + "..."
                sample_lines.append(snippet)
            sample = " | ".join(sample_lines)
            summary = f"Compact profile: keywords [{common}] â€” examples: {sample}"
    try:
        if hasattr(memory, "set_profile"):
            memory.set_profile(user_id, "compact_summary", summary)
        else:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO user_profile (user_id, key, value) VALUES (?, ?, ?)", (user_id, "compact_summary", summary))
            conn.commit()
            conn.close()
    except Exception as e:
        print("Failed to persist compact summary:", e)
    return summary




def init_observability(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, component TEXT, message TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS metrics (name TEXT PRIMARY KEY, value REAL)")
    conn.commit()
    conn.close()

def log_event(level: str, component: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO logs (ts, level, component, message) VALUES (?, ?, ?, ?)", (datetime.datetime.utcnow().isoformat(), level, component, message))
    conn.commit()
    conn.close()

def inc_metric(name: str, amount: float = 1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM metrics WHERE name = ?", (name,))
    row = c.fetchone()
    current = row[0] if row else 0
    new = current + amount
    c.execute("INSERT OR REPLACE INTO metrics (name, value) VALUES (?, ?)", (name, new))
    conn.commit()
    conn.close()

init_observability()




import uuid, time, os, json
from pprint import pprint

def _choose_planner_roadmap(user_info: dict, demo: bool):
    if demo:
        # Use planner_agent if defined, else raise to force non-demo usage
        try:
            return planner_agent(user_info, demo=True)
        except Exception:
            return {
                "goal": user_info.get("goal", ""),
                "timeline_weeks": user_info.get("max_weeks", 12),
                "milestones": []
            }
    llm = LLMClient()
    if not llm.client:
        raise RuntimeError("LLM client not configured. Set GOOGLE_API_KEY or run with demo=True.")
    roadmap = generate_roadmap(llm, user_info)
    return roadmap

def run_pipeline(user_info: Dict[str, Any], demo: bool = False, use_gemini: bool = True, max_resources: int = 3):
    """
    Run the cohesive pipeline for a single user_info dict.
    Specify custom inputs in user_info and call this function.
    """
    mem = MemoryStore()
    mem.upsert_user(user_info.get("user_id", "user_" + str(int(time.time()))))
    session_id = user_info.get("user_id", "session_" + str(uuid.uuid4()))
    try:
        roadmap = _choose_planner_roadmap(user_info, demo=demo)
    except RuntimeError as e:
        print("Planner (LLM) unavailable:", e)
        print("Falling back to demo planner.")
        roadmap = _choose_planner_roadmap(user_info, demo=True)
    eval_report = evaluate_roadmap_basic(roadmap, user_info)
    iterations = 0
    while iterations < 3 and (not eval_report["summary"]["feasible"] or any(r["actionable_objectives"]==0 for _,r in eval_report["per_milestone"].iterrows())):
        iterations += 1
        try:
            _ = refiner_agent(demo=demo)
            for m in roadmap.get("milestones", []):
                if isinstance(m.get("hours",0), (int,float)):
                    m["hours"] = int(max(1, m.get("hours",0) * 0.9))
        except Exception:
            for m in roadmap.get("milestones", []):
                if isinstance(m.get("hours",0), (int,float)):
                    m["hours"] = int(max(1, m.get("hours",0) * 0.9))
        eval_report = evaluate_roadmap_basic(roadmap, user_info)
    try:
        if use_gemini and GENAI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
            localized = localize_roadmap_with_gemini(roadmap, user_info.get("language","en"), use_gemini=True)
        else:
            localized = localize_roadmap_extended(roadmap, user_info.get("language","en"))
    except Exception as e:
        print("Localization failed:", e)
        localized = roadmap
    try:
        asl_ready = add_asl_gloss(localized)
    except Exception:
        asl_ready = localized
    try:
        fname = export_roadmap_to_excel(asl_ready, filename=f"roadmap_{user_info.get('user_id','out')}.xlsx", max_resources_per_milestone=max_resources)
    except Exception as e:
        print("Export failed:", e)
        fname = None
    try:
        mem.save_session(session_id, user_info.get("user_id", session_id), {"user_info": user_info, "roadmap": asl_ready, "eval": eval_report["summary"]})
    except Exception as e:
        print("Failed to save session:", e)
    try:
        inc_metric("pipeline_runs", 1)
        log_event("INFO", "pipeline", f"Completed pipeline for {user_info.get('user_id')} iterations={iterations}")
    except Exception:
        pass
    return {"roadmap": asl_ready, "evaluation": eval_report, "excel": fname, "iterations": iterations}




# Replace the fields below with your custom learning goal and preferences, then run this cell.
user_info = {
    "user_id": "Kriti",
    "goal": "Master agentic AI for healthcare",
    "background": "intermediate python, basic statistics",
    "hours_per_week": 10,
    "max_weeks": 20,
    "language": "hi"
}

# Run the pipeline; set demo=False to require a configured Gemini key (GOOGLE_API_KEY)
result = run_pipeline(user_info, demo=True, use_gemini=False, max_resources=3)
from pprint import pprint
pprint({"excel": result["excel"], "iterations": result["iterations"], "roadmap_goal": result["roadmap"].get("goal")})








