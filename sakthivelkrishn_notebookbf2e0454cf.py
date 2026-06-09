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


# Notebook-style demo for "SmartStudy Mentor" capstone (Kaggle-ready, runnable here).
# This is a complete, runnable demonstration that includes:
# - Sequential multi-agent system (Explainer -> QuestionGenerator)
# - Tools (mock Google Search + custom formatter)
# - Sessions & InMemorySessionService + MemoryBank (file-backed)
# - Pause/Resume (long-running op) with resume token
# - Chunked work for safe pause/resume
# - Simple observability (logging)
# - Tiny evaluation with sample prompts
#
# NOTE:
# - This demo simulates LLM/tool calls because this environment has no network or ADK/Gemini access.
# - To adapt to real ADK/Gemini, replace the `mock_llm_call` and `mock_google_search` with real API/ADK calls.
#
# Run as a notebook: each cell is represented by sequential code blocks below.
# The code is intentionally clear, commented, and modular for easy replacement with real services.
import os
import json
import uuid
import time
import random
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Config & Paths
# ---------------------------
DATA_DIR = "/mnt/data/smartstudy_mentor_demo"
os.makedirs(DATA_DIR, exist_ok=True)
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "agent_run.log")

# ---------------------------
# Simple logging helper
# ---------------------------
def log(msg: str):
    timestamp = datetime.now().isoformat(timespec="seconds")
    entry = f"{timestamp} | {msg}"
    print(entry)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

# ---------------------------
# Mock LLM and Tool Implementations (replaceable)
# ---------------------------
def mock_google_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Simulated Google Search results. Replace this with a real Search tool or API.
    Returns a list of dicts with 'title' and 'snippet' keys.
    """
    # deterministic-ish pseudo-results based on query seed
    seed = sum(ord(c) for c in query) % 1000
    random.seed(seed)
    results = []
    for i in range(top_k):
        title = f"Article {i+1} on {query}"
        snippet = f"A concise point about {query}: {random.choice(['definition','application','example','history'])} and its core idea."
        results.append({"title": title, "snippet": snippet, "url": f"https://example.com/{i+1}/{query.replace(' ','_')}"})
    return results

def mock_llm_call(prompt: str, max_tokens: int = 256) -> str:
    """
    Simulate an LLM response by reformatting the prompt. In real use, call Gemini/ADK here.
    Keep outputs human-readable and deterministic for testing.
    """
    # Basic heuristics: if asked to explain, produce a simple explanation; if asked to create questions, produce Q/A lists.
    p_lower = prompt.lower()
    if "explain" in p_lower or "explanation" in p_lower or "what is" in p_lower:
        # produce a 3-point explanation with examples
        topic = prompt.split("about")[-1].strip() if "about" in prompt else "the topic"
        return textwrap.dedent(f"""\
            Explanation of {topic}:\n
            1. Core idea: {topic} is fundamentally about understanding its principles.\n
            2. Example: A simple real-world example helps illustrate {topic}.\n
            3. Why it matters: {topic} is useful because it helps solve related problems.\n""")
    if "generate questions" in p_lower or "practice" in p_lower or "mcq" in p_lower:
        # create a few practice questions based on the prompt content
        base = "Based on the explanation above"
        questions = []
        for i in range(1, 6):
            questions.append({
                "q": f"What is key point {i} for {base}?",
                "a": f"Key point {i} is ... (answer placeholder)"
            })
        # return JSON-like formatted string (we'll parse it)
        return json.dumps(questions)
    # default fallback - echo brief summary
    return "Summary: " + (prompt[:200] + ("..." if len(prompt) > 200 else ""))

# ---------------------------
# Session & Memory Services
# ---------------------------
class InMemorySessionService:
    """
    Holds ephemeral session data during a notebook run.
    For persistence across restarts or long-term storage, use MemoryBank.
    """
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, topic: str, difficulty: str = "medium") -> str:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "topic": topic,
            "difficulty": difficulty,
            "progress_index": 0,
            "paused": False,
            "resume_token": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": []  # stores per-chunk outputs for auditing
        }
        self.sessions[session_id] = session
        log(f"Created session {session_id} for user {user_id} topic '{topic}' difficulty {difficulty}")
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def update(self, session_id: str, updates: Dict[str, Any]):
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session not found")
        s.update(updates)
        s["updated_at"] = datetime.now().isoformat()
        log(f"Updated session {session_id}: {list(updates.keys())}")

    def set_paused(self, session_id: str, paused: bool, resume_token: Optional[str] = None):
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session not found")
        s["paused"] = paused
        s["resume_token"] = resume_token
        s["updated_at"] = datetime.now().isoformat()
        log(f"Session {session_id} paused={paused} resume_token={resume_token}")

class MemoryBank:
    """
    Simple file-backed memory bank for snapshots. Stores JSON files keyed by token.
    """
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

    def save_snapshot(self, token: str, snapshot: Dict[str, Any]):
        path = os.path.join(self.dir, f"{token}.json")
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        log(f"Saved snapshot to {path}")

    def load_snapshot(self, token: str) -> Dict[str, Any]:
        path = os.path.join(self.dir, f"{token}.json")
        if not os.path.exists(path):
            raise FileNotFoundError("Snapshot not found for token: " + token)
        with open(path, "r") as f:
            snapshot = json.load(f)
        log(f"Loaded snapshot from {path}")
        return snapshot

# ---------------------------
# Tokens & Utilities
# ---------------------------
def make_resume_token(session_id: str) -> str:
    # For demo, token is simple uuid; in production consider signed tokens (JWT)
    token = str(uuid.uuid4())
    return token

# ---------------------------
# Agent Implementations (Explainer & QuestionGen)
# ---------------------------
class AgentBase:
    def __init__(self, name: str):
        self.name = name

    def call(self, input_data: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class ConceptExplainerAgent(AgentBase):
    """
    Uses mock_google_search + mock_llm_call to produce an explanation and small facts list.
    """
    def __init__(self, name="concept_explainer"):
        super().__init__(name)

    def call(self, input_data: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data.get("topic")
        log(f"[{self.name}] Searching for topic: {topic}")
        results = mock_google_search(topic, top_k=3)
        # build a prompt using top snippets
        prompt = f"Explain {topic} in simple language using these snippets: " + " ".join(r["snippet"] for r in results)
        explanation = mock_llm_call(prompt)
        output = {
            "explanation": explanation,
            "search_results": results
        }
        # write minimal history entry
        session_entry = {"agent": self.name, "output": {"explanation_snippet": explanation[:200]}, "timestamp": datetime.now().isoformat()}
        session["history"].append(session_entry)
        log(f"[{self.name}] Completed explanation for topic: {topic}")
        return output

class QuestionGeneratorAgent(AgentBase):
    """
    Takes the explanation and generates a list of practice questions.
    Supports chunked generation (so we can pause/resume between chunks).
    """
    def __init__(self, name="question_gen", chunk_size: int = 2):
        super().__init__(name)
        self.chunk_size = chunk_size

    def call_chunk(self, explanation: str, start_index: int = 0) -> Tuple[List[Dict[str,str]], int]:
        """
        Generate a chunk of questions. Returns (questions, new_progress_index)
        """
        prompt = f"Generate practice questions from the following explanation. Output JSON list of question-answer dicts.\n\n{explanation}\n\nMake questions clear and label difficulty."
        llm_out = mock_llm_call("generate questions: " + prompt)
        # llm_out is JSON string in our mock; parse it
        try:
            all_questions = json.loads(llm_out)
        except Exception:
            # fallback: create a few simple questions
            all_questions = [{"q": f"Demo question {i}", "a": "demo answer"} for i in range(1, 11)]

        # slice chunk
        end_index = min(start_index + self.chunk_size, len(all_questions))
        chunk = all_questions[start_index:end_index]
        new_index = end_index
        log(f"[{self.name}] Generated questions chunk {start_index}..{end_index} (size {len(chunk)})")
        return chunk, new_index

# ---------------------------
# Coordinator (Orchestrator) with Pause/Resume & Chunking
# ---------------------------
class Coordinator:
    def __init__(self, explainer: ConceptExplainerAgent, qgen: QuestionGeneratorAgent,
                 session_service: InMemorySessionService, memory_bank: MemoryBank):
        self.explainer = explainer
        self.qgen = qgen
        self.session_service = session_service
        self.memory_bank = memory_bank

    def start_session(self, user_id: str, topic: str, difficulty: str = "medium") -> str:
        session_id = self.session_service.create_session(user_id, topic, difficulty)
        return session_id

    def run_full_workflow(self, session_id: str, pause_after_chunks: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the sequential workflow: explain -> generate questions (in chunks).
        If pause_after_chunks is provided (int), pause after generating that many chunks.
        """
        session = self.session_service.get(session_id)
        if not session:
            raise KeyError("Session not found")

        # 1) If no explanation yet, run explainer
        if "explanation" not in session:
            input_data = {"topic": session["topic"], "difficulty": session["difficulty"]}
            expl_out = self.explainer.call(input_data, session)
            session["explanation"] = expl_out["explanation"]
            session["search_results"] = expl_out["search_results"]
            self.session_service.update(session_id, {"explanation": session["explanation"], "search_results": session["search_results"]})

        # 2) Run question generator in chunked mode
        progress = session.get("progress_index", 0)
        chunks_done = 0
        all_questions = session.get("questions", [])  # accumulate
        while True:
            if session.get("paused"):
                log(f"[Coordinator] Detected paused session {session_id}; aborting run loop")
                break

            chunk, new_progress = self.qgen.call_chunk(session["explanation"], start_index=progress)
            # append chunk to session questions
            for q in chunk:
                all_questions.append(q)
            progress = new_progress
            session["progress_index"] = progress
            session["questions"] = all_questions
            # persist snapshot automatically after each chunk
            snapshot = {"session": session}
            token = session.get("resume_token")
            if token:
                # overwrite existing snapshot for this token
                self.memory_bank.save_snapshot(token, snapshot)
            # also save a temporary snapshot keyed by session_id for quick reload
            self.memory_bank.save_snapshot(session_id, snapshot)
            self.session_service.update(session_id, {"progress_index": progress, "questions": all_questions})

            chunks_done += 1
            # if we've generated all questions, break
            # Here we detect completion by comparing with an expected larger number in mock; else break when no new chunk
            if new_progress == 0 or new_progress >= 5:  # our mock yields 5 questions from mock_llm_call
                log(f"[Coordinator] Completed question generation for session {session_id}")
                break

            # Check pause-after-chunks
            if pause_after_chunks and chunks_done >= pause_after_chunks:
                # set paused and create resume token/snapshot
                resume_token = make_resume_token(session_id)
                session["paused"] = True
                session["resume_token"] = resume_token
                self.memory_bank.save_snapshot(resume_token, {"session": session})
                self.session_service.set_paused(session_id, True, resume_token)
                log(f"[Coordinator] Paused session {session_id} after {chunks_done} chunks; resume_token={resume_token}")
                break

            # small sleep to simulate long-running work
            time.sleep(0.1)  # reduce in demo

        return {"session": session}

    def pause_session(self, session_id: str) -> str:
        session = self.session_service.get(session_id)
        if not session:
            raise KeyError("session not found")
        resume_token = make_resume_token(session_id)
        session["paused"] = True
        session["resume_token"] = resume_token
        self.memory_bank.save_snapshot(resume_token, {"session": session})
        self.session_service.set_paused(session_id, True, resume_token)
        log(f"[Coordinator] Paused session {session_id}; resume_token={resume_token}")
        return resume_token

    def resume_session(self, resume_token: str) -> Dict[str, Any]:
        snapshot = self.memory_bank.load_snapshot(resume_token)
        session = snapshot["session"]
        session_id = session["session_id"]
        # restore into session service
        self.session_service.sessions[session_id] = session
        # unset paused flag and resume_token (we'll clear resume_token on success)
        session["paused"] = False
        session["resume_token"] = None
        self.session_service.update(session_id, {"paused": False, "resume_token": None, "questions": session.get("questions", []), "progress_index": session.get("progress_index", 0)})
        log(f"[Coordinator] Resumed session {session_id} from token {resume_token}")
        # continue generating from where left off
        return self.run_full_workflow(session_id, pause_after_chunks=None)

# ---------------------------
# Wiring everything together
# ---------------------------
session_service = InMemorySessionService()
memory_bank = MemoryBank(SNAPSHOT_DIR)
explainer = ConceptExplainerAgent()
qgen = QuestionGeneratorAgent(chunk_size=2)
coordinator = Coordinator(explainer, qgen, session_service, memory_bank)

# ---------------------------
# Demo: Start a session and run, pause, and resume
# ---------------------------
user_id = "user_sakthivel"
topic = "Photosynthesis"
difficulty = "easy"

# 1) Start session
session_id = coordinator.start_session(user_id, topic, difficulty)
log(f"Demo session_id: {session_id}")

# 2) Run workflow but request pause after 1 chunk of question generation to create a resume token
result = coordinator.run_full_workflow(session_id, pause_after_chunks=1)
log("First run (paused) result summary keys: " + ", ".join(result["session"].keys()))

# show partial outputs
session_snapshot = session_service.get(session_id)
print("\n=== Partial Session Snapshot ===")
print(json.dumps(session_snapshot, indent=2)[:1000] + ("\n... [truncated]" if len(json.dumps(session_snapshot, indent=2)) > 1000 else ""))

# 3) Extract resume token and resume
resume_token = session_snapshot.get("resume_token")
print(f"\nResume token provided: {resume_token}\n")

if resume_token:
    resumed_result = coordinator.resume_session(resume_token)
    log("Resumed run complete. Session keys: " + ", ".join(resumed_result["session"].keys()))
    print("\n=== Final Session Snapshot ===")
    print(json.dumps(resumed_result["session"], indent=2)[:1500] + ("\n... [truncated]" if len(json.dumps(resumed_result["session"], indent=2)) > 1500 else ""))

# ---------------------------
# Small evaluation: run agent on 3 sample prompts and save outputs for submission
# ---------------------------
evaluation_prompts = [
    ("Photosynthesis", "easy"),
    ("Newton's First Law", "medium"),
    ("Binary Search Algorithm", "hard")
]

evaluation_results = []
for tpc, diff in evaluation_prompts:
    sid = coordinator.start_session(user_id, tpc, diff)
    # run without pausing to get full output
    res = coordinator.run_full_workflow(sid, pause_after_chunks=None)
    evaluation_results.append({
        "session_id": sid,
        "topic": tpc,
        "difficulty": diff,
        "explanation": res["session"].get("explanation", "")[:500],
        "num_questions": len(res["session"].get("questions", []))
    })

# save evaluation to file
eval_path = os.path.join(DATA_DIR, "evaluation_results.json")
with open(eval_path, "w") as f:
    json.dump(evaluation_results, f, indent=2)
log(f"Saved evaluation results to {eval_path}")

# print short evaluation summary
print("\n=== Evaluation Summary ===")
for er in evaluation_results:
    print(f"- {er['topic']} ({er['difficulty']}): explanation_len={len(er['explanation'])} chars, questions={er['num_questions']}")

# show log file path and snapshots list
print("\n=== Files produced ===")
print(f"- Log file: {LOG_FILE}")
print(f"- Snapshot dir: {SNAPSHOT_DIR} (files: {os.listdir(SNAPSHOT_DIR)})")
print(f"- Evaluation file: {eval_path}")

# End of demo cell. You can adapt mock_llm_call and mock_google_search to real services.



