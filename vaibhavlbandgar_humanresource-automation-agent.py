import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Google ADK / GenAI (as in main.ipynb)
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Standard utilities
import os
import json
import re
from pathlib import Path
from collections import defaultdict
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Data & plotting
import pandas as pd
import matplotlib.pyplot as plt


MODEL_NAME = "gemini-2.5-pro"

RETRY_CONFIG = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 502, 503, 504],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "memory_store.json"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

A2A_HOST = "127.0.0.1"
A2A_PORT = 8000

MAX_WORKERS = 4


def log(msg):
    ts = datetime.datetime.now().isoformat()
    print(f"[{ts}] {msg}")

def compact_context(messages, max_tokens=1500):
    # simple greedy compaction (message lengths by characters)
    joined = []
    total = 0
    for m in reversed(messages):
        l = len(m)
        if total + l > max_tokens:
            break
        joined.append(m)
        total += l
    return "\n\n".join(reversed(joined))

def safe_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_read_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


people = [
    {"id": 1, "name": "Priya Sharma", "email": "priya+demo@example.com",
     "experience_years": 4, "skills": "python, machine learning, pandas, sql",
     "resume_text": "Priya worked on ML models, ETL pipelines and productionized models using Python and pandas. 4 years experience."},
    {"id": 2, "name": "Rahul Verma", "email": "rahul+demo@example.com",
     "experience_years": 6, "skills": "java, spring, microservices, sql",
     "resume_text": "Rahul is an experienced backend engineer focused on microservices and distributed systems. 6 years experience."},
    {"id": 3, "name": "Ananya Desai", "email": "ananya+demo@example.com",
     "experience_years": 2, "skills": "react, javascript, ui, css",
     "resume_text": "Ananya has built modern UI with React and strong product sensibilities. 2 years experience."},
]

df_people = pd.DataFrame(people)
csv_path = DATA_DIR / "people_demo.csv"
df_people.to_csv(csv_path, index=False)
log(f"Saved demo people dataset to {csv_path}")

JOB_JD = {
    "id": "jd_001",
    "title": "Machine Learning Engineer",
    "min_experience": 3,
    "must_have": ["python", "machine learning", "pandas"],
    "nice_to_have": ["sql", "docker"]
}

ONBOARDING_TEMPLATE = "Welcome {name} to {role}. Day {day}: {tasks}"
OFFBOARDING_TEMPLATE = "Offboard {name}: {tasks}"
PERFORMANCE_TEMPLATE = "Performance summary for {name}: {summary}"


def parse_resume_text(resume_text):
    txt = (resume_text or "").lower()
    known_skills = ["python","machine learning","pandas","sql","java","spring","microservices","react","javascript","docker","etl","ui","css"]
    found = [s for s in known_skills if s in txt]
    years_match = re.search(r"(\d+)\s*(?:years|yrs)", txt)
    years = int(years_match.group(1)) if years_match else None
    summary = re.sub(r"\s+", " ", txt).strip()
    return {"skills": found, "experience_years": years, "summary": summary}


def code_execution_tool(code_str, timeout=5):
    """
    Simple subprocess based runner. Not secure for untrusted code.
    """
    try:
        proc = subprocess.Popen(["python3", "-c", code_str],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -2}

def mock_search_tool(query, top_k=3):
    results = []
    for p in people:
        score = sum(1 for w in query.lower().split() if w in p["resume_text"].lower())
        results.append({"id": p["id"], "name": p["name"], "score": score, "snippet": p["resume_text"][:200]})
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results

# Wrap simple tool descriptors if your main.ipynb expects dict-like tools
code_tool = {"name": "code_exec", "callable": code_execution_tool}
search_tool = {"name": "search", "callable": mock_search_tool}


model = Gemini(model_name=MODEL_NAME)
tools_list = [search_tool, code_tool]

hr_agent = LlmAgent(
    model=model,
    tools=tools_list,
    retry_options=RETRY_CONFIG
)

log("Global HR agent (hr_agent) created with Gemini and retry config.")



def call_agent_and_parse(prompt):
    """
    Call the global hr_agent with prompt and return parsed JSON or {'raw': text} on fallback.
    """
    resp = hr_agent.run(prompt)  # matches your notebook pattern
    text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {"raw": text}
        return {"raw": text}


def build_screening_prompt(candidate, jd):
    parsed = parse_resume_text(candidate.get("resume_text", ""))
    candidate_summary = (
        f"Name: {candidate.get('name')}. Experience: {candidate.get('experience_years') or parsed.get('experience_years')}. "
        f"Skills: {', '.join(parsed.get('skills', []))}. Resume: {parsed.get('summary')}"
    )
    prompt = f"""
You are an HR screening assistant.
Job Title: {jd['title']}
JD must-have skills: {', '.join(jd['must_have'])}
JD nice-to-have: {', '.join(jd.get('nice_to_have', []))}

Candidate summary: {candidate_summary}

Task:
1) Provide a numeric suitability score 0-100.
2) List matched skills from the JD.
3) Short justification (1-2 sentences).
4) Label: 'Strong Fit', 'Potential Fit', or 'Not a Fit'.

Return JSON with keys: score (number), matched_skills (list), justification (string), label (string).
"""
    return prompt

def screen_candidate(candidate, jd=JOB_JD):
    prompt = build_screening_prompt(candidate, jd)
    return call_agent_and_parse(prompt)


def batch_screen_candidates(df_candidates, jd=JOB_JD):
    results = []
    for _, row in df_candidates.iterrows():
        candidate = row.to_dict()
        try:
            out = screen_candidate(candidate, jd)
            results.append({"candidate_id": candidate["id"], "name": candidate["name"], "result": out})
            log(f"Screened {candidate['name']} -> label: {out.get('label')}, score: {out.get('score')}")
        except Exception as e:
            log(f"Error screening {candidate['name']}: {e}")
            results.append({"candidate_id": candidate['id'], "name": candidate['name'], "result": {"error": str(e)}})
    return results


def parallel_batch_screen_candidates(df_candidates, jd=JOB_JD, max_workers=MAX_WORKERS):
    candidates = df_candidates.to_dict(orient="records")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(screen_candidate, c, jd): c for c in candidates}

        for fut in as_completed(futures):
            c = futures[fut]
            try:
                out = fut.result()
                results.append({
                    "candidate_id": c["id"],
                    "name": c["name"],
                    "result": out
                })
                log(f"Parallel screened {c['name']} -> label: {out.get('label')}, score: {out.get('score')}")
            except Exception as e:
                log(f"Parallel screening error for {c['name']}: {e}")
                results.append({
                    "candidate_id": c["id"],
                    "name": c["name"],
                    "result": {"error": str(e)}
                })

    return results


def iterative_refinement(initial_candidates, jd=JOB_JD, max_iters=5):
    state = initial_candidates
    for i in range(max_iters):
        log(f"Refinement iteration {i+1}")
        # run screening on current candidates
        df_state = pd.DataFrame(state)
        screened = batch_screen_candidates(df_state, jd)
        # sort by score (if available) and keep top N
        def score_of(r):
            s = r.get("result", {}).get("score")
            try:
                return float(s) if s is not None else 0.0
            except:
                return 0.0
        screened_sorted = sorted(screened, key=score_of, reverse=True)
        top_ids = [r["candidate_id"] for r in screened_sorted[:2]]
        new_state = [next((p for p in state if p["id"] == cid), None) for cid in top_ids]
        new_state = [s for s in new_state if s]
        # stop condition: if IDs unchanged
        old_ids = [c["id"] for c in state]
        new_ids = [c["id"] for c in new_state]
        if old_ids == new_ids:
            log("Refinement stable â€” stopping.")
            return screened_sorted
        state = new_state
    return screened_sorted


def build_onboarding_prompt(candidate, role_title):
    parsed = parse_resume_text(candidate.get("resume_text", ""))
    candidate_summary = f"Name: {candidate.get('name')}. Skills: {', '.join(parsed.get('skills', []))}. Experience: {candidate.get('experience_years') or parsed.get('experience_years')}"
    prompt = f"""
You are an HR onboarding assistant for role: {role_title}.
Candidate info: {candidate_summary}

Produce:
1) A 7-day onboarding checklist with daily tasks.
2) Required accesses and documents.
3) A short welcome message (1-2 sentences).

Return JSON: {{ "checklist": [{{"day":1,"tasks":[...]}}], "accesses": [...], "welcome": "..." }}
"""
    return prompt

def generate_onboarding(candidate, role_title):
    prompt = build_onboarding_prompt(candidate, role_title)
    return call_agent_and_parse(prompt)


def build_offboarding_prompt(candidate, role_title):
    parsed = parse_resume_text(candidate.get("resume_text", ""))
    candidate_summary = f"Name: {candidate.get('name')}. Skills: {', '.join(parsed.get('skills', []))}."
    prompt = f"""
You are an HR offboarding assistant for role: {role_title}.
Candidate info: {candidate_summary}

Produce:
1) Exit checklist with tasks (revoke accesses, asset return, etc).
2) Knowledge transfer notes summary.
3) Suggested timeline (days).

Return JSON: {{ "checklist": [...], "knowledge_transfer": "...", "timeline_days": N }}
"""
    return prompt

def generate_offboarding(candidate, role_title):
    prompt = build_offboarding_prompt(candidate, role_title)
    return call_agent_and_parse(prompt)


def build_performance_prompt(name, feedback_list):
    prompt = f"""
You are an HR performance review assistant.
Employee: {name}
Feedback: {feedback_list}

Produce:
1) Concise performance summary (3-4 sentences).
2) Strengths (3 bullets).
3) Areas for improvement (3 bullets).
4) Suggested goals for next period.

Return JSON: {{ "summary": "...", "strengths": [...], "improvements": [...], "goals": [...] }}
"""
    return prompt

def generate_performance(name, feedback_list):
    prompt = build_performance_prompt(name, feedback_list)
    return call_agent_and_parse(prompt)


def mock_calendar_find_slots(preferred_days=3, slots_per_day=3):
    now = datetime.datetime.now()
    slots = []
    for d in range(1, preferred_days+1):
        day = now + datetime.timedelta(days=d)
        for s in range(slots_per_day):
            slot_time = (day.replace(hour=10 + s*2, minute=0, second=0, microsecond=0)).isoformat()
            slots.append(slot_time)
    return slots

def send_mock_email(to, subject, body):
    log(f"[MOCK EMAIL] To: {to} | Subject: {subject}\n{body[:200]}...")
    return {"status": "sent", "to": to, "subject": subject}

def propose_slots_for_candidate(candidate_email):
    slots = mock_calendar_find_slots()
    return {"candidate": candidate_email, "proposed_slots": slots[:5], "status": "proposed"}


MCP = {}

def register_component(name, fn):
    MCP[name] = fn
    log(f"Registered MCP component: {name}")

def call_component(name, *args, **kwargs):
    if name not in MCP:
        raise ValueError(f"Component {name} not registered")
    return MCP[name](*args, **kwargs)

# register components (functions)
register_component("screen", lambda df, jd=JOB_JD: batch_screen_candidates(df, jd))
register_component("parallel_screen", lambda df, jd=JOB_JD: parallel_batch_screen_candidates(df, jd))
register_component("onboard", lambda candidate, role: generate_onboarding(candidate, role))
register_component("offboard", lambda candidate, role: generate_offboarding(candidate, role))
register_component("schedule", lambda email: propose_slots_for_candidate(email))
register_component("performance", lambda name, feedback: generate_performance(name, feedback))


# in-memory
memory_store = defaultdict(list)

def memory_add(key, message):
    memory_store[str(key)].append({"ts": datetime.datetime.now().isoformat(), "text": message})

def memory_get_context(key, max_tokens=1500):
    msgs = [m["text"] for m in memory_store.get(str(key), [])]
    return compact_context(msgs, max_tokens=max_tokens)

# file-backed persistent memory
persistent_memory = safe_read_json(MEMORY_FILE) or {}

def persistent_memory_add(key, message):
    persistent_memory.setdefault(str(key), []).append({"ts": datetime.datetime.now().isoformat(), "text": message})
    safe_write_json(MEMORY_FILE, persistent_memory)

def persistent_memory_get_context(key, max_tokens=1500):
    msgs = [m["text"] for m in persistent_memory.get(str(key), [])]
    return compact_context(msgs, max_tokens=max_tokens)


def evaluate_results(results, jd=JOB_JD):
    total = len(results)
    strong = sum(1 for r in results if (r.get("result", {}).get("score") or 0) >= 70)
    coverages = []
    for r in results:
        matched = r.get("result", {}).get("matched_skills", []) or []
        cov = 0
        if jd.get("must_have"):
            cov = len(set(matched) & set(jd["must_have"])) / max(1, len(jd["must_have"]))
        coverages.append(cov)
    avg_cov = sum(coverages) / max(1, len(coverages))
    metrics = {"total": total, "strong_count": strong, "strong_pct": (strong/total)*100 if total else 0, "avg_must_have_coverage": avg_cov}
    fname = METRICS_DIR / f"metrics_{int(time.time())}.json"
    safe_write_json(fname, metrics)
    return metrics

def plot_metrics(metrics):
    labels = ["strong_pct", "avg_must_have_coverage"]
    values = [metrics["strong_pct"], metrics["avg_must_have_coverage"]]
    plt.figure(figsize=(6,3))
    plt.bar(labels, values)
    plt.title("Screening Metrics")
    plt.ylabel("Value")
    plt.show()


def hr_service_handler(payload):
    try:
        action = payload.get("action")
        if action == "screen":
            candidates = payload.get("candidates")
            df = pd.DataFrame(candidates) if candidates else df_people
            return {"status": "ok", "results": batch_screen_candidates(df, JOB_JD)}
        elif action == "parallel_screen":
            candidates = payload.get("candidates")
            df = pd.DataFrame(candidates) if candidates else df_people
            return {"status": "ok", "results": parallel_batch_screen_candidates(df, JOB_JD)}
        elif action == "onboard":
            candidate = payload.get("candidate")
            role = payload.get("role", "Employee")
            return {"status": "ok", "result": generate_onboarding(candidate, role)}
        elif action == "offboard":
            candidate = payload.get("candidate")
            role = payload.get("role", "Employee")
            return {"status": "ok", "result": generate_offboarding(candidate, role)}
        elif action == "schedule":
            email = payload.get("email")
            return {"status": "ok", "proposal": propose_slots_for_candidate(email)}
        elif action == "performance":
            name = payload.get("name")
            feedback = payload.get("feedback", [])
            return {"status": "ok", "result": generate_performance(name, feedback)}
        else:
            return {"status": "error", "message": f"unknown action {action}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

hr_agent_a2a = to_a2a(hr_service_handler)
log("Prepared HR service as A2A agent (hr_agent_a2a).")


# import uvicorn
# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.post("/a2a/hr")
# def hr_endpoint(payload: dict):
#     return hr_service_handler(payload)
#
# if __name__ == "__main__":
#     uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)


# Example 1: Sequential screening (will call your LLM)
# results_seq = batch_screen_candidates(df_people, JOB_JD)
# print(results_seq)

# Example 2: Parallel screening
# results_par = parallel_batch_screen_candidates(df_people, JOB_JD)
# print(results_par)

# Example 3: Iterative refinement
# refined = iterative_refinement(people, JOB_JD, max_iters=3)
# print(refined)

# Example 4: Onboarding
# print(generate_onboarding(people[0], JOB_JD["title"]))

# Example 5: Offboarding
# print(generate_offboarding(people[1], JOB_JD["title"]))

# Example 6: Performance
# print(generate_performance(people[0]["name"], ["Delivered X", "Needs better docs"]))




