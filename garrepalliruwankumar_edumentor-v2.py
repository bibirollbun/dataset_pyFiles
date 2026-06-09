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
#            EduMentor – Advanced Kaggle Notebook Version
# Includes:
# 1. Multi-Agent System (Planner, Teacher, Evaluator)
# 3. Sequential Agents Pipeline
# 5. Built-In Tools (Logging, Tracing, Metrics)
# 6. Sessions & State Management
# 7. Long-Term Memory Store
# 8. Context Compaction
# 9. Observability (Logging, Tracing, Metrics)
# 10. Agent Evaluation
# ============================================================

from __future__ import annotations
import json
import uuid
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date

# ------------------------------------------------------------
#                     Observability Tools
# ------------------------------------------------------------

class Logger:
    @staticmethod
    def log(msg: str):
        print(f"[LOG {datetime.utcnow().isoformat()}] {msg}")

class Tracer:
    @staticmethod
    def trace(agent: str, event: str, payload=None):
        print(f"[TRACE] {agent:<10} | {event:<15} | {json.dumps(payload)}")

class Metrics:
    calls = {"planner":0, "teacher":0, "evaluator":0}
    scores = []
    
    @staticmethod
    def increment(agent):
        Metrics.calls[agent] += 1

    @staticmethod
    def record_score(score):
        Metrics.scores.append(score)

# ------------------------------------------------------------
#                        Session Store
# ------------------------------------------------------------

class SessionStore:
    """Stores learner session state."""
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.state = {"progress":{}, "history":[]}

    def set(self, key, value):
        self.state[key] = value

    def get(self, key):
        return self.state.get(key)

    def add_event(self, event):
        self.state["history"].append({
            "ts": datetime.utcnow().isoformat(),
            "event": event
        })

# ------------------------------------------------------------
#                     Long-Term Memory Store
# ------------------------------------------------------------

class MemoryStore:
    """Simple in-memory key/value store with auto-cleanup."""
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def write(self, key: str, value: Any):
        self.store[key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        }
        Logger.log(f"Memory updated: {key}")

    def read(self, key: str):
        entry = self.store.get(key)
        return entry["value"] if entry else None

    def cleanup(self, keep_last_n=100):
        """Simple context compaction."""
        if len(self.store) > keep_last_n:
            keys = sorted(self.store.keys())
            for k in keys[:-keep_last_n]:
                del self.store[k]
            Logger.log(f"Memory compacted. Keeping last {keep_last_n} entries.")

# ------------------------------------------------------------
#                     Scheduler
# ------------------------------------------------------------

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

    def schedule(self, tasks, start_date, days):
        dates = [start_date + timedelta(days=i) for i in range(days)]
        out = []
        for i, (topic, dur) in enumerate(tasks):
            t = Task(id=str(uuid.uuid4()), topic=topic, date=dates[i % len(dates)].isoformat(), duration_min=dur)
            self.tasks.append(t)
            out.append(t)
        return out

    def tasks_for_date(self, d: date):
        ds = d.isoformat()
        return [t for t in self.tasks if t.date == ds]

# ------------------------------------------------------------
#                     Planner Agent
# ------------------------------------------------------------

class PlannerAgent:
    def __init__(self, memory, scheduler):
        self.memory = memory
        self.scheduler = scheduler

    def break_goals(self, goal):
        mapping = {
            "Machine Learning basics": [
                ("Introduction & Math Review", 30),
                ("Linear Regression", 40),
                ("Gradient Descent", 30),
                ("Logistic Regression", 35),
                ("Classification Metrics", 25),
                ("Train/Test Split", 30),
                ("Regularization", 30),
                ("Decision Trees", 30),
                ("Random Forests", 35),
                ("Neural Networks Intro", 40),
            ]
        }
        return mapping.get(goal, [(goal, 30)])

    def create_plan(self, goal, start, days):
        Metrics.increment("planner")
        Tracer.trace("Planner", "create_plan", {"goal":goal})
        
        tasks = self.break_goals(goal)
        scheduled = self.scheduler.schedule(tasks, start, days)
        
        plan = {
            "goal": goal,
            "start_date": start.isoformat(),
            "tasks": [t.__dict__ for t in scheduled]
        }
        
        self.memory.write(f"plan:{goal}", plan)
        return plan

    def adjust(self, report):
        Metrics.increment("planner")
        score = report.get("score", 100)
        topic = report["topic"]

        if score < 70:
            Logger.log(f"Low score detected for {topic}. Adding remedial tasks.")
            today = date.today()
            remedial = [
                ("Remedial: " + topic, 20),
                ("Practice: " + topic, 20),
            ]
            self.scheduler.schedule(remedial, today + timedelta(days=1), 2)

# ------------------------------------------------------------
#                     Teacher Agent
# ------------------------------------------------------------

class TeacherAgent:
    def __init__(self, memory):
        self.memory = memory

    def teach(self, topic):
        Metrics.increment("teacher")
        Tracer.trace("Teacher", "teach", {"topic":topic})

        lesson = f"""
        {topic}

        This lesson breaks the concept down into simple, intuitive steps.
        Example: {topic} applied in a real-world scenario.
        """

        summary = f"Summary of {topic}: key ideas and formulas."
        flash = [
            {"q": f"What is {topic}?", "a": "A simple definition."},
            {"q": f"Key concept in {topic}?", "a": "Main idea described briefly."},
        ]

        pkg = {
            "topic": topic,
            "lesson": textwrap.dedent(lesson),
            "summary": summary,
            "flashcards": flash,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.memory.write(f"lesson:{topic}", pkg)
        return pkg

# ------------------------------------------------------------
#                     Evaluator Agent
# ------------------------------------------------------------

class EvaluatorAgent:
    def __init__(self, memory):
        self.memory = memory

    def quiz(self, topic):
        Metrics.increment("evaluator")
        return [
            {"id": f"q{uuid.uuid4().hex[:6]}", "prompt": f"Explain {topic}.", "answer": "core idea"}
        ]

    def grade(self, task, answers, questions):
        Metrics.increment("evaluator")
        Tracer.trace("Evaluator", "grade", {"task":task.topic})

        correct = 0
        for q in questions:
            if q["answer"] in answers.get(q["id"], "").lower():
                correct += 1

        score = (correct / len(questions)) * 100
        Metrics.record_score(score)

        report = {
            "task_id": task.id,
            "topic": task.topic,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.memory.write(f"eval:{task.id}", report)
        return report

# ------------------------------------------------------------
#                     Orchestrator
# ------------------------------------------------------------

class EduMentor:
    def __init__(self):
        self.session = SessionStore()
        self.memory = MemoryStore()
        self.scheduler = Scheduler()
        self.planner = PlannerAgent(self.memory, self.scheduler)
        self.teacher = TeacherAgent(self.memory)
        self.evaluator = EvaluatorAgent(self.memory)

    def export_plan_json(self, goal):
        return json.dumps(self.memory.read(f"plan:{goal}"), indent=2)

# ------------------------------------------------------------
#                     AUTO-RUN DEMO
# ------------------------------------------------------------

app = EduMentor()

goal = "Machine Learning basics"
start = date.today()

print("\n=== Creating Plan ===\n")
plan = app.planner.create_plan(goal, start, days=10)

tasks_today = app.scheduler.tasks_for_date(start)

if tasks_today:
    task = tasks_today[0]
    print(f"\nToday's Task: {task.topic}")

    # teacher
    pkg = app.teacher.teach(task.topic)

    print("\n--- LESSON ---\n", pkg["lesson"])
    print("\n--- SUMMARY ---\n", pkg["summary"])
    print("\n--- FLASHCARDS ---")
    for fc in pkg["flashcards"]:
        print(fc)

    print("\n--- QUIZ ---")
    qz = app.evaluator.quiz(task.topic)
    for q in qz:
        print(q["id"], ":", q["prompt"])

    answers = {q["id"]: "core idea"}  # perfect score simulation
    report = app.evaluator.grade(task, answers, qz)

    print("\n--- REPORT ---\n", json.dumps(report, indent=2))

    app.planner.adjust(report)

print("\n--- PLAN JSON EXPORT ---")
print(app.export_plan_json(goal))

print("\n--- METRICS ---")
print("Agent Calls:", Metrics.calls)
print("Scores:", Metrics.scores)

