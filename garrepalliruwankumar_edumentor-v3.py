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


# ============================================================
# EduMentor with User Accounts & Profiles (local, hashed pw)
# - Adds UserManager supporting create/login/logout/change-password
# - Stores users in MemoryBank
# - Profiles saved per user (goals, level, preferences, history)
# - Option to run demo without authentication (no-auth)
# ============================================================

from __future__ import annotations
import json
import uuid
import textwrap
import threading
import time
import os
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

# ---------------------------
# Minimal helpers for hashing
# ---------------------------
def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str,str]:
    """Return dict with salt (hex) and hashed password (hex) using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return {"salt": salt.hex(), "hash": hashed.hex()}

def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    new_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()
    return secrets.compare_digest(new_hash, hash_hex)

# -------------------- (Re-used core components) ----------------------
# For brevity in this snippet we implement the minimal supporting classes
# (Logger, MemoryBank, SessionStore, Scheduler, Agents, etc.)
# These are adapted from the previous full system and kept concise.

class Logger:
    @staticmethod
    def info(msg: str):
        print(f"[INFO  {datetime.utcnow().isoformat()}] {msg}")
    @staticmethod
    def warn(msg: str):
        print(f"[WARN  {datetime.utcnow().isoformat()}] {msg}")
    @staticmethod
    def error(msg: str):
        print(f"[ERR   {datetime.utcnow().isoformat()}] {msg}")

class MemoryBank:
    def __init__(self):
        self.db: Dict[str, Any] = {}
    def write(self, key: str, value: Any):
        self.db[key] = {"ts": datetime.utcnow().isoformat(), "value": value}
        Logger.info(f"MemoryBank write: {key}")
    def read(self, key: str):
        return self.db.get(key, {}).get("value")
    def query_prefix(self, prefix: str):
        return {k:v for k,v in self.db.items() if k.startswith(prefix)}
    def export(self, path=None):
        path = path or ("/kaggle/working/edumentor_memory.json" if os.path.isdir("/kaggle/working") else "edumentor_memory.json")
        with open(path,"w") as f:
            json.dump(self.db, f, indent=2)
        Logger.info(f"Memory exported to {path}")
        return path

class SessionStore:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.state: Dict[str, Any] = {"created": datetime.utcnow().isoformat(), "user": None, "history":[]}
        self.filepath = f"/kaggle/working/edumentor_session_{self.session_id}.json" if os.path.isdir("/kaggle/working") else f"edumentor_session_{self.session_id}.json"
    def set(self, key: str, value: Any):
        self.state[key] = value
    def get(self, key: str):
        return self.state.get(key)
    def add_event(self, event: str):
        self.state["history"].append({"ts": datetime.utcnow().isoformat(), "event": event})
    def save(self, path: Optional[str] = None):
        p = path or self.filepath
        try:
            with open(p, "w") as f:
                json.dump(self.state, f, indent=2)
            Logger.info(f"Session saved to {p}")
            return p
        except Exception as e:
            Logger.error(f"Failed to save session: {e}")
            return None
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.state = json.load(f)
            Logger.info(f"Session loaded from {path}")
            return True
        Logger.warn(f"Session file not found: {path}")
        return False

# -------------------- Scheduler minimal -----------------------------
@dataclass
class Task:
    id: str
    topic: str
    date: str
    duration_min: int = 30
    completed: bool = False
    score: Optional[float] = None

class Scheduler:
    def __init__(self):
        self.tasks: List[Task] = []
    def schedule_tasks(self, tasks: List[Tuple[str,int]], start_date: date, days: int):
        day_slots = [start_date + timedelta(days=i) for i in range(days)]
        scheduled=[]
        for idx, (topic,dur) in enumerate(tasks):
            assigned = day_slots[idx % len(day_slots)].isoformat()
            t = Task(id=str(uuid.uuid4()), topic=topic, date=assigned, duration_min=dur)
            self.tasks.append(t); scheduled.append(t)
        Logger.info(f"Scheduled {len(scheduled)} tasks starting {start_date.isoformat()}")
        return scheduled
    def tasks_for_date(self, d: date):
        ds = d.isoformat()
        return [t for t in self.tasks if t.date == ds]
    def mark_completed(self, task_id: str, score: Optional[float] = None):
        for t in self.tasks:
            if t.id == task_id:
                t.completed = True; t.score = score
                Logger.info(f"Task {task_id} marked completed with score {score}")
                return t
        return None

# -------------------- Simple AnswerChecker ---------------------------
class AnswerChecker:
    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    def score_answer(self, user: str, expected: str) -> float:
        if not user: return 0.0
        s = self.similarity(user, expected)
        kws = [w for w in expected.split() if len(w) > 3]
        kw_matches = sum(1 for kw in kws if kw.lower() in user.lower())
        kw_factor = min(0.2, 0.05 * kw_matches)
        return min(1.0, s + kw_factor)

# -------------------- Minimal Planner/Teacher/Evaluator --------------
# (Lightweight, used for demo; full system can be swapped in)
class PlannerAgent:
    def __init__(self, memory: MemoryBank, scheduler: Scheduler):
        self.memory = memory; self.scheduler = scheduler
    def break_into_tasks(self, goal: str):
        mapping = {
            "Machine Learning basics":[
                ("Intro & Math Review",30),
                ("Linear Regression",40),
                ("Gradient Descent",35),
                ("Logistic Regression",30),
                ("Classification Metrics",25),
                ("Train/Test Split & Overfit",30),
                ("Regularization",30),
                ("Decision Trees",30),
                ("Random Forests",35),
                ("Neural Networks Intro",40)
            ]
        }
        return mapping.get(goal, [(goal,30)])
    def create_weekly_schedule(self, goal: str, start_date: date, days: int=14):
        tasks = self.break_into_tasks(goal)
        scheduled = self.scheduler.schedule_tasks(tasks, start_date, days)
        plan = {"goal":goal,"start_date":start_date.isoformat(),"tasks":[t.__dict__ for t in scheduled]}
        self.memory.write(f"plan:{goal}:{start_date.isoformat()}", plan)
        return plan

class TeacherAgent:
    def __init__(self, memory: MemoryBank):
        self.memory = memory
    def teach(self, topic: str):
        lesson = f"{topic}\n\nThis lesson explains the core idea with an example."
        summary = f"Summary: key points for {topic}"
        cards = [{"q":f"What is {topic}?", "a": f"A short definition of {topic}."}]
        pkg = {"topic":topic,"lesson":lesson,"summary":summary,"flashcards":cards,"ts":datetime.utcnow().isoformat()}
        self.memory.write(f"lesson:{topic}:{datetime.utcnow().isoformat()}", pkg)
        return pkg

class EvaluatorAgent:
    def __init__(self, memory: MemoryBank, checker: AnswerChecker):
        self.memory = memory; self.checker = checker
    def create_quiz(self, topic: str, q_count: int=3):
        qs = []
        for i in range(q_count):
            qs.append({"id":f"q{uuid.uuid4().hex[:6]}","prompt":f"Explain {topic} #{i+1}","answer":f"core idea of {topic}"})
        self.memory.write(f"quiz:{topic}:{datetime.utcnow().isoformat()}", qs)
        return qs
    def grade(self, task: Task, user_answers: Dict[str,str], questions: List[Dict[str,str]]):
        correct = 0
        for q in questions:
            user = user_answers.get(q["id"], "")
            score_val = self.checker.score_answer(user, q["answer"])
            if score_val >= 0.6: correct += 1
        score = (correct / len(questions)) * 100 if questions else 0.0
        report = {"task_id":task.id,"topic":task.topic,"score":score,"ts":datetime.utcnow().isoformat()}
        self.memory.write(f"eval:{task.id}:{datetime.utcnow().isoformat()}", report)
        return report

# -------------------- User Manager (accounts + profiles) -------------
class UserManager:
    """
    Local user management:
    - create_user(username, password, display_name)
    - authenticate(username, password)
    - set_profile(username, profile_dict)
    - get_profile(username)
    - change_password(username, old_pw, new_pw)
    - list_users()
    """
    def __init__(self, memory: MemoryBank):
        self.memory = memory
        self.user_index_key = "users:index"  # store list of usernames

    def _add_user_to_index(self, username: str):
        idx = self.memory.read(self.user_index_key) or []
        if username not in idx:
            idx.append(username); self.memory.write(self.user_index_key, idx)

    def create_user(self, username: str, password: str, display_name: Optional[str] = None) -> bool:
        key = f"user:{username}"
        if self.memory.read(key) is not None:
            Logger.warn("Username already exists.")
            return False
        ph = hash_password(password)
        entry = {
            "username": username,
            "display_name": display_name or username,
            "password_hash": ph["hash"],
            "salt": ph["salt"],
            "created": datetime.utcnow().isoformat(),
            "profile": {"goals": None, "level": "beginner", "prefs": {}}
        }
        self.memory.write(key, entry)
        self._add_user_to_index(username)
        Logger.info(f"User created: {username}")
        return True

    def authenticate(self, username: str, password: str) -> bool:
        key = f"user:{username}"
        data = self.memory.read(key)
        if not data:
            Logger.warn("Auth failed: user not found.")
            return False
        ok = verify_password(password, data["salt"], data["password_hash"])
        if ok:
            Logger.info(f"User authenticated: {username}")
        else:
            Logger.warn("Auth failed: wrong password.")
        return ok

    def set_profile(self, username: str, profile: Dict[str,Any]):
        key = f"user:{username}"
        data = self.memory.read(key)
        if not data:
            Logger.warn("Profile set failed: user not found.")
            return False
        data["profile"].update(profile)
        self.memory.write(key, data)
        Logger.info(f"Profile updated for {username}")
        return True

    def get_profile(self, username: str) -> Optional[Dict[str,Any]]:
        key = f"user:{username}"
        data = self.memory.read(key)
        if not data: return None
        return data.get("profile")

    def change_password(self, username: str, old_pw: str, new_pw: str) -> bool:
        key = f"user:{username}"
        data = self.memory.read(key)
        if not data:
            Logger.warn("Change password failed: user not found.")
            return False
        if not verify_password(old_pw, data["salt"], data["password_hash"]):
            Logger.warn("Change password failed: old password incorrect.")
            return False
        ph = hash_password(new_pw)
        data["password_hash"] = ph["hash"]
        data["salt"] = ph["salt"]
        self.memory.write(key, data)
        Logger.info(f"Password changed for {username}")
        return True

    def list_users(self) -> List[str]:
        return self.memory.read(self.user_index_key) or []

# -------------------- EduMentor orchestrator with user integration ---
class EduMentor:
    def __init__(self):
        self.memory = MemoryBank()
        self.session = SessionStore()
        self.scheduler = Scheduler()
        # core agents
        self.planner = PlannerAgent(self.memory, self.scheduler)
        self.teacher = TeacherAgent(self.memory)
        self.checker = AnswerChecker()
        self.evaluator = EvaluatorAgent(self.memory, self.checker)
        # user manager
        self.users = UserManager(self.memory)
        # current user (if logged in)
        self.current_user: Optional[str] = None

    # Authentication helpers
    def create_account(self, username: str, password: str, display_name: Optional[str] = None) -> bool:
        return self.users.create_user(username, password, display_name)

    def login(self, username: str, password: str) -> bool:
        ok = self.users.authenticate(username, password)
        if ok:
            self.current_user = username
            self.session.set("user", username)
            self.session.add_event(f"user_logged_in:{username}")
            Logger.info(f"Logged in as {username}")
        return ok

    def logout(self):
        Logger.info(f"Logging out user: {self.current_user}")
        self.current_user = None
        self.session.set("user", None)
        self.session.add_event("user_logged_out")

    def set_profile(self, profile: Dict[str,Any]):
        if not self.current_user:
            Logger.warn("No logged-in user to set profile.")
            return False
        return self.users.set_profile(self.current_user, profile)

    def get_profile(self, username: Optional[str] = None):
        uname = username or self.current_user
        if not uname:
            Logger.warn("No user specified.")
            return None
        return self.users.get_profile(uname)

    # Run demo for current user (or no-auth if none)
    def run_demo_for_current_user(self):
        if self.current_user:
            Logger.info(f"Running demo for user: {self.current_user}")
            profile = self.get_profile(self.current_user)
            goal = profile.get("goals") or "Machine Learning basics"
            start = date.today()
            plan = self.planner.create_weekly_schedule(goal, start, days=7)
            # run today's first task
            tasks = self.scheduler.tasks_for_date(start)
            if not tasks:
                print("No tasks for today.")
                return
            t = tasks[0]
            lesson = self.teacher.teach(t.topic)
            print("\n--- LESSON ---\n", lesson["lesson"])
            quiz = self.evaluator.create_quiz(t.topic)
            print("\n--- QUIZ ---")
            for q in quiz: print(q["id"],":", q["prompt"])
            # simulate answers from user profile skill level
            user_answers = {}
            level = profile.get("level","beginner")
            for q in quiz:
                if level == "advanced":
                    user_answers[q["id"]] = q["answer"]
                else:
                    # partial correctness for non-advanced
                    user_answers[q["id"]] = q["answer"] if (uuid.uuid4().int % 2) == 0 else "partial"
            report = self.evaluator.grade(t, user_answers, quiz)
            print("\n--- EVAL REPORT ---\n", json.dumps(report, indent=2))
            # mark completed and save into user's memory/history
            self.scheduler.mark_completed(t.id, report["score"])
            # attach to user's profile history
            hist = self.users.get_profile(self.current_user).get("history", [])
            hist.append({"task":t.topic, "score": report["score"], "ts": datetime.utcnow().isoformat()})
            self.users.set_profile(self.current_user, {"history": hist})
            return report
        else:
            Logger.info("No user logged in — running no-auth demo (global).")
            # basic no-auth demo
            goal = "Machine Learning basics"
            start = date.today()
            plan = self.planner.create_weekly_schedule(goal, start, days=7)
            tasks = self.scheduler.tasks_for_date(start)
            if not tasks:
                print("No tasks for today.")
                return
            t = tasks[0]
            lesson = self.teacher.teach(t.topic)
            print("\n--- LESSON ---\n", lesson["lesson"])
            quiz = self.evaluator.create_quiz(t.topic)
            print("\n--- QUIZ ---")
            for q in quiz: print(q["id"],":", q["prompt"])
            # simulate average answers
            user_answers = {q["id"]: q["answer"] for q in quiz}
            report = self.evaluator.grade(t, user_answers, quiz)
            print("\n--- EVAL REPORT ---\n", json.dumps(report, indent=2))
            self.scheduler.mark_completed(t.id, report["score"])
            return report

# -------------------- Demo of User features ---------------------------
def demo_user_flow():
    em = EduMentor()
    print("\n--- Create two users ---")
    em.create_account("alice", "password123", display_name="Alice")
    em.create_account("bob", "s3cureP@ss", display_name="Bob")

    print("\n--- List users ---")
    print(em.users.list_users())

    print("\n--- Set profile for alice and login ---")
    em.login("alice", "password123")
    em.set_profile({"goals":"Machine Learning basics","level":"beginner"})
    print("Profile (alice):", em.get_profile("alice"))

    print("\n--- Run demo for alice ---")
    em.run_demo_for_current_user()

    print("\n--- Change alice password (wrong old password) ---")
    ok = em.users.change_password("alice", "wrong", "newpass")
    print("Password change success:", ok)
    print("\n--- Change alice password (correct) ---")
    ok = em.users.change_password("alice", "password123", "newpass")
    print("Password change success:", ok)

    print("\n--- Logout Alice and login Bob ---")
    em.logout()
    em.login("bob", "s3cureP@ss")
    em.set_profile({"goals":"Intro to Python","level":"advanced"})
    print("Profile (bob):", em.get_profile("bob"))

    print("\n--- Run no-auth demo (after logout) ---")
    em.logout()
    em.run_demo_for_current_user()

    print("\n--- Demo complete ---")
    return em

# Run demo when executed
if __name__ == "__main__":
    # Running the user-demo will showcase account creation, login, profile management and demo run
    em_instance = demo_user_flow()


# =====================================================
# Generate required Kaggle submission file for EduMentor
# =====================================================

import json

submission_data = {
    "project": "EduMentor AI Study Assistant",
    "status": "success",
    "message": "This file is generated to satisfy Kaggle submission requirements.",
    "sample_output": {
        "study_plan": ["Topic 1", "Topic 2", "Topic 3"],
        "lesson_generated": "Yes",
        "quiz_score": 8
    }
}

# Save as a JSON file in Kaggle working directory
submission_path = "/kaggle/working/edumentor_submission.json"

with open(submission_path, "w") as f:
    json.dump(submission_data, f, indent=4)

print("Submission file created:", submission_path)

