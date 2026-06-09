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
        print(os.path.join(dirname, "HR Multi-Agent Interview Board"))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -------------------------
# Imports & Global Helpers
# -------------------------
import os, re, json, sqlite3, time, random, uuid, statistics
from typing import Any, Dict, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

# Observability stores
METRICS = defaultdict(int)
TRACE_LOG = []

# Logging (simpler than full logging config in notebook)
def trace(event: str, details: Dict[str,Any]=None):
    TRACE_LOG.append({"ts": time.time(), "event": event, "details": details or {}})
    METRICS[f"trace.{event}"] += 1

def metric_inc(name: str, v:int=1):
    METRICS[name] += v

# Deterministic random for demo reproducibility
random.seed(42)



# -------------------------
# Offline LLM Simulator
# -------------------------
def compact_context(context_tokens: List[str], max_tokens: int = 200) -> List[str]:
    """
    Example context compaction: keep the most recent fraction plus longest tokens.
    This is a deterministic, explainable heuristic for demo.
    """
    if len(context_tokens) <= max_tokens:
        return context_tokens
    recent_cut = int(max_tokens * 0.4)
    recent = context_tokens[-recent_cut:]
    rest = context_tokens[:-recent_cut]
    rest_sorted = sorted(rest, key=lambda s: len(s), reverse=True)
    keep = rest_sorted[:max_tokens - recent_cut] + recent
    return keep[:max_tokens]

def call_llm(prompt: str, system: str=None, max_tokens: int=256, context: List[str]=None) -> str:
    """
    Offline deterministic LLM simulation. Replace with a provider wrapper later if allowed.
    - Uses hashing to pick responses so runs are reproducible.
    """
    # tiny latency
    time.sleep(0.05)
    p = (prompt or "").lower()
    # technical question generation
    if "generate a technical question" in p or "generate a question targeting" in p:
        m = re.search(r'targeting\s+([\w\-]+)', p)
        skill = m.group(1) if m else "python"
        bank = {
            "python": [
                "Explain the difference between a list and a tuple and when you'd use each.",
                "What are Python decorators and give a simple use case."
            ],
            "sql": [
                "Explain the difference between WHERE and HAVING.",
                "What is an index and how does it help?"
            ],
            "aws": [
                "When use Lambda vs EC2?",
                "Explain IAM roles vs policies."
            ]
        }
        arr = bank.get(skill, bank["python"])
        return arr[abs(hash(prompt)) % len(arr)]
    # simulate candidate answer
    if "simulate a candidate answer" in p:
        samples = [
            "I would decompose the problem, write tests and use modular design for maintainability.",
            "I implemented similar systems using microservices, retries and observability."
        ]
        return samples[abs(hash(prompt)) % len(samples)]
    # evaluation / scoring
    if "evaluate the answer" in p or ("provide score" in p and "answer" in p):
        score = 5 + (abs(hash(prompt)) % 6)  # gives 5..10
        return f"Score: {score} — evaluator note"
    # HR evaluation
    if "evaluate communication" in p or "culture fit" in p:
        score = 5 + (abs(hash(prompt)) % 5)  # 5..9
        return f"Score: {score} — hr note"
    # fallback
    return "SIMULATED_RESPONSE"



# -------------------------
# Resume Parser (custom tool)
# -------------------------
SKILL_KEYWORDS = ["python","java","sql","aws","docker","kubernetes","react","node","ml","tensorflow","git"]

def parse_resume_text(text: str) -> Dict[str,Any]:
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    years = re.search(r"(\d+)\s+years", text.lower())
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else "Unknown"
    skills = [kw for kw in SKILL_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text.lower())]
    years_val = int(years.group(1)) if years else None
    summary = (lines[1] if len(lines) > 1 else "")[:200]
    parsed = {"name": name, "email": email.group(0) if email else None, "skills": skills, "years_experience": years_val, "summary": summary, "raw": text}
    trace("parser.run", {"candidate": parsed["email"]})
    metric_inc("parser.invocations")
    return parsed

# scoring helper
def compute_weighted_score(tech_score: float, hr_score: float, weights=(0.7,0.3)):
    return round(weights[0]*tech_score + weights[1]*hr_score, 2)



# -------------------------
# Session service (InMemory) & Memory Bank (SQLite)
# -------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}  # session_id -> dict

    def create_session(self, candidate_id: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"candidate_id": candidate_id, "history": [], "created_at": time.time()}
        trace("session.create", {"session": sid, "candidate": candidate_id})
        metric_inc("session.create")
        return sid

    def append_event(self, session_id: str, event: Dict[str,Any]):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({"ts": time.time(), **event})
            trace("session.append", {"session": session_id, "event": event.get("agent")})
        else:
            raise KeyError("Session not found")

    def get_session(self, session_id: str) -> Dict[str,Any]:
        return self.sessions.get(session_id, {})

    def pause_session(self, session_id: str):
        trace("session.pause", {"session": session_id})

    def resume_session(self, session_id: str) -> Dict[str,Any]:
        trace("session.resume", {"session": session_id})
        return self.get_session(session_id)

# init session store
session_store = InMemorySessionService()

# SQLite Memory Bank
MEM_DB = "memory_bank.sqlite"
def init_memory_db(db=MEM_DB):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidate_memory (
            candidate_id TEXT PRIMARY KEY,
            meta_json TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

def save_candidate_memory(candidate_id: str, meta: Dict[str,Any]):
    conn = sqlite3.connect(MEM_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO candidate_memory(candidate_id, meta_json)
        VALUES(?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET meta_json=excluded.meta_json, last_updated=CURRENT_TIMESTAMP
    """, (candidate_id, json.dumps(meta)))
    conn.commit(); conn.close()
    trace("memory.save", {"candidate": candidate_id})

def load_candidate_memory(candidate_id: str) -> Dict[str,Any]:
    conn = sqlite3.connect(MEM_DB)
    c = conn.cursor()
    c.execute("SELECT meta_json FROM candidate_memory WHERE candidate_id=?", (candidate_id,))
    row = c.fetchone(); conn.close()
    return json.loads(row[0]) if row else {}

init_memory_db()



# -------------------------
# Agent-to-Agent in-memory messaging (A2A)
# -------------------------
A2A_INBOX = defaultdict(deque)  # receiver -> deque of messages

def a2a_send(sender: str, receiver: str, payload: Dict[str,Any]):
    msg = {"from": sender, "to": receiver, "payload": payload, "ts": time.time()}
    A2A_INBOX[receiver].append(msg)
    trace("a2a.send", {"from": sender, "to": receiver})
    metric_inc("a2a.sent")

def a2a_receive(receiver: str) -> List[Dict[str,Any]]:
    msgs = []
    while A2A_INBOX[receiver]:
        msgs.append(A2A_INBOX[receiver].popleft())
    if msgs:
        trace("a2a.receive", {"receiver": receiver, "count": len(msgs)})
        metric_inc("a2a.received", len(msgs))
    return msgs



# -------------------------
# Agent base & simple helper dataclass
# -------------------------
@dataclass
class AgentResponse:
    agent_name: str
    output: Any
    metadata: Dict[str,Any] = field(default_factory=dict)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, *args, **kwargs) -> AgentResponse:
        raise NotImplementedError

# ResumeParserAgent
class ResumeParserAgent(BaseAgent):
    def __init__(self):
        super().__init__("ResumeParser")

    def run(self, resume_text: str) -> AgentResponse:
        parsed = parse_resume_text(resume_text)
        return AgentResponse(self.name, parsed)



# -------------------------
# TechInterviewerAgent (Loop)
# -------------------------
class TechInterviewerAgent(BaseAgent):
    def __init__(self, max_rounds: int = 3):
        super().__init__("TechInterviewer")
        self.max_rounds = max_rounds

    def run(self, profile: Dict[str,Any], provided_answers: List[str]=None, session_id: str=None) -> AgentResponse:
        metric_inc(f"{self.name}.invocations")
        trace("agent.start", {"agent": self.name})
        provided_answers = provided_answers or []
        skills = profile.get("skills") or ["python"]
        history = []
        context_tokens = []
        for i in range(self.max_rounds):
            skill = skills[i] if i < len(skills) else skills[0]
            q_prompt = f"Generate a technical question targeting {skill}"
            q = call_llm(q_prompt, context=context_tokens)
            # candidate answer: provided or simulated
            if i < len(provided_answers):
                ans = provided_answers[i]
            else:
                ans = call_llm(f"Simulate a candidate answer for: {q}", context=context_tokens)
            eval_resp = call_llm(f"Evaluate the answer: '{ans}' for question: '{q}'. Provide score.", context=context_tokens)
            m = re.search(r'(\d+)', eval_resp)
            score = float(m.group(1)) if m else 6.0
            qa = {"q": q, "a": ans, "score": score}
            history.append(qa)
            # append event to session store for pause/resume
            if session_id:
                session_store.append_event(session_id, {"agent": self.name, "round": i, "qa": qa})
            # context engineering: append and compact
            context_tokens.append(q); context_tokens.append(ans)
            context_tokens = compact_context(context_tokens, max_tokens=100)
            trace("tech.round", {"session": session_id, "round": i, "score": score})
            metric_inc(f"{self.name}.round")
        avg = statistics.mean([x["score"] for x in history]) if history else 0.0
        trace("agent.end", {"agent": self.name})
        return AgentResponse(self.name, {"q_and_a": history, "avg_score": round(avg,2)})



# -------------------------
# HRInterviewerAgent (Parallel-friendly)
# -------------------------
class HRInterviewerAgent(BaseAgent):
    def __init__(self, rounds: int = 3):
        super().__init__("HRInterviewer")
        self.rounds = rounds

    def run(self, profile: Dict[str,Any], provided_answers: List[str]=None, session_id: str=None) -> AgentResponse:
        metric_inc(f"{self.name}.invocations")
        trace("agent.start", {"agent": self.name})
        questions = [
            "Tell me about a time you handled conflict in a team.",
            "Describe a situation where you took ownership of a problem.",
            "How do you prioritize when facing multiple deadlines?"
        ]
        provided_answers = provided_answers or []
        history = []
        for i in range(self.rounds):
            q = questions[i % len(questions)]
            if i < len(provided_answers):
                ans = provided_answers[i]
            else:
                ans = call_llm(f"Simulate candidate behavioral answer for: {q}")
            eval_resp = call_llm(f"Evaluate communication: '{ans}' for Q: '{q}'. Provide score.")
            m = re.search(r'(\d+)', eval_resp)
            score = float(m.group(1)) if m else 6.0
            qa = {"q": q, "a": ans, "score": score}
            history.append(qa)
            if session_id:
                session_store.append_event(session_id, {"agent": self.name, "round": i, "qa": qa})
            trace("hr.round", {"session": session_id, "round": i, "score": score})
            metric_inc(f"{self.name}.round")
        avg = statistics.mean([x["score"] for x in history]) if history else 0.0
        trace("agent.end", {"agent": self.name})
        return AgentResponse(self.name, {"q_and_a": history, "avg_score": round(avg,2)})




class ScoringAgent(BaseAgent):
    def __init__(self, weight_tech=0.7, weight_hr=0.3):
        super().__init__("ScoringAgent")
        self.weight_tech = weight_tech
        self.weight_hr = weight_hr

    def run(self, candidate_id: str, tech_out: Dict, hr_out: Dict, resume_meta: Dict, session_id: str=None) -> AgentResponse:
        metric_inc(f"{self.name}.invocations")
        tech_score = tech_out.get("avg_score", 0.0)
        hr_score = hr_out.get("avg_score", 0.0)
        final = compute_weighted_score(tech_score, hr_score, weights=(self.weight_tech, self.weight_hr))
        summary = {
            "candidate_id": candidate_id,
            "tech_score": tech_score,
            "hr_score": hr_score,
            "final_score": final,
            "recommendation": ("Hire" if final>=7 else ("Hire with training" if final>=5 else "Reject")),
            "resume_meta": resume_meta
        }
        # persist
        existing = load_candidate_memory(candidate_id)
        existing['last_interview'] = summary
        save_candidate_memory(candidate_id, existing)
        # A2A: send to evaluator
        a2a_send(self.name, "EvaluatorAgent", {"summary": summary})
        trace("scoring.complete", {"candidate": candidate_id})
        return AgentResponse(self.name, summary)

class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvaluatorAgent")

    def run(self) -> AgentResponse:
        metric_inc(f"{self.name}.invocations")
        msgs = a2a_receive(self.name)
        findings = []
        for m in msgs:
            summ = m['payload']['summary']
            # simple fairness check: large mismatch between tech/hr
            mismatch = abs(summ['tech_score'] - summ['hr_score'])
            flag = "ok" if mismatch <= 3 else "score_mismatch"
            findings.append({"candidate": summ['candidate_id'], "mismatch": mismatch, "flag": flag})
            trace("evaluator.check", {"candidate": summ['candidate_id'], "flag": flag})
        return AgentResponse(self.name, findings)




def run_full_interview(resume_text: str, candidate_override_id: str=None, tech_answers=None, hr_answers=None, enable_pause=False):
    # 1. parse synchronously
    parser = ResumeParserAgent()
    parsed = parser.run(resume_text).output
    candidate_id = candidate_override_id or parsed.get("email") or parsed.get("name") or str(uuid.uuid4())
    # create session
    sid = session_store.create_session(candidate_id)
    # instantiate agents
    tech = TechInterviewerAgent(max_rounds=3)
    hr = HRInterviewerAgent(rounds=3)
    scorer = ScoringAgent()
    evaluator = EvaluatorAgent()
    # optional: demo pause semantics (if enable_pause True, TechInterviewer may pause mid-way)
    global SIMULATED_PAUSE
    SIMULATED_PAUSE = enable_pause
    # run tech + hr in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(tech.run, parsed, tech_answers, sid): 'tech',
            ex.submit(hr.run, parsed, hr_answers, sid): 'hr'
        }
        results = {}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    # scoring
    score_resp = scorer.run(candidate_id, results['tech'].output, results['hr'].output, parsed, sid)
    # evaluator (reads A2A messages)
    eval_resp = evaluator.run()
    return {"session_id": sid, "parsed": parsed, "tech": results['tech'], "hr": results['hr'], "scoring": score_resp, "evaluator": eval_resp}



# -------------------------
# Demo candidate (single run)
# -------------------------
demo_resume = """
Naina Pawar
naina.p@example.com
Software engineer with 4 years experience working with Python, Docker, AWS, React, and SQL.
Built backend services, microservices, and automated deployment pipelines.
"""

out = run_full_interview(demo_resume)
print("=== FINAL SCORECARD ===")
print(json.dumps(out['scoring'].output, indent=2))
print("\n=== TECH Q&A SAMPLE ===")
for qa in out['tech'].output['q_and_a']:
    print(f"Q: {qa['q']}\nA: {qa['a']}\nScore: {qa['score']}\n")
print("\n=== HR Q&A SAMPLE ===")
for qa in out['hr'].output['q_and_a']:
    print(f"Q: {qa['q']}\nA: {qa['a']}\nScore: {qa['score']}\n")
print("\n=== EVALUATOR FINDINGS ===")
print(out['evaluator'].output)




resumes = {
    "rahul": """
    Rahul Sharma
    rahul.sharma@example.com
    Full-stack developer with 3 years experience in Python, JavaScript, React, and AWS.
    """,
    "anita": """
    Anita Verma
    anita.verma@example.com
    Data analyst with 5 years experience in SQL, Python, Tableau.
    """,
    "john": """
    John Mathews
    john.mathews@example.com
    DevOps engineer with 4 years experience in Docker, Kubernetes, Jenkins, and AWS.
    """
}

batch_results = {}
for key, txt in resumes.items():
    print(f"\n--- Running for {key} ---")
    r = run_full_interview(txt)
    batch_results[key] = r
    print("Candidate:", r['parsed'].get('email') or key, "-> Recommendation:", r['scoring'].output['recommendation'], "Score:", r['scoring'].output['final_score'])



# -------------------------
# Export batch summary and ranking
# -------------------------
summary = []
for k,v in batch_results.items():
    summary.append({"id": v['parsed'].get('email') or k, "final_score": v['scoring'].output['final_score'], "recommendation": v['scoring'].output['recommendation']})
# rank
ranking = sorted(summary, key=lambda x: x['final_score'], reverse=True)
print("=== RANKING ===")
for r in ranking:
    print(r)
# save JSON
with open("batch_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved batch_summary.json")




# Print a simple observability report

print("=== METRICS SAMPLE ===")
for k,v in list(METRICS.items())[:50]:
    print(k, ":", v)
print("\n=== LAST 10 TRACES ===")
for t in TRACE_LOG[-10:]:
    print(t)








