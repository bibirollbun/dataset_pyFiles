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
#  DUALMIND - FULL PROJECT (Single Cell Auto-Installer)
# ============================================================

import os, json, textwrap

ROOT = "DualMind"
AGENTS = os.path.join(ROOT, "agents")
TOOLS = os.path.join(ROOT, "tools")

os.makedirs(AGENTS, exist_ok=True)
os.makedirs(TOOLS, exist_ok=True)

# -----------------------------
# helper to write files
def w(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))

# ============================================================
# 1) main.py
# ============================================================

w(f"{ROOT}/main.py", """
import json
from agents.pro_agent import ProAgent
from agents.con_agent import ConAgent
from agents.research_agent import ResearchAgent
from agents.evaluator_agent import EvaluatorAgent
from tools.memory_manager import MemoryManager

class Orchestrator:
    def __init__(self):
        self.memory = MemoryManager()
        self.pro = ProAgent(self.memory)
        self.con = ConAgent(self.memory)
        self.research = ResearchAgent(self.memory)
        self.evaluator = EvaluatorAgent(self.memory)

    def run(self, user_input: str, user_id: str = "user_1"):
        session = {"question": user_input, "user_id": user_id}

        pro_out = self.pro.generate(user_input)
        con_out = self.con.generate(user_input)

        pro_verified = self.research.verify_and_expand(pro_out)
        con_verified = self.research.verify_and_expand(con_out)

        verdict = self.evaluator.evaluate(pro_verified, con_verified, session)

        self.memory.add_decision_log(user_id, user_input, pro_verified, con_verified, verdict)

        return {"pro": pro_verified, "con": con_verified, "verdict": verdict}


if __name__ == "__main__":
    orch = Orchestrator()

    questions = [
        "Should I move to a new city?",
        "Should I buy a new laptop?"
    ]

    for q in questions:
        print("\\nQUESTION:", q)
        out = orch.run(q)

        print("\\n--- PRO ---")
        for a in out["pro"]:
            print("-", a["text"])
            for e in a["evidence"]:
                print("   *", e["title"])

        print("\\n--- CON ---")
        for a in out["con"]:
            print("-", a["text"])
            for e in a["evidence"]:
                print("   *", e["title"])

        print("\\n--- VERDICT ---")
        print(json.dumps(out["verdict"], indent=2))
""")

# ============================================================
# 2) agents/pro_agent.py
# ============================================================

w(f"{AGENTS}/pro_agent.py", """
from tools.web_search import web_search

class ProAgent:
    def __init__(self, memory):
        self.memory = memory

    def generate(self, question: str):
        points = self.generate_arguments(question, stance="pro")
        for p in points:
            p["evidence"] = web_search(p["text"])
        return points

    def generate_arguments(self, question: str, stance="pro"):
        q = question.lower()

        if "move" in q:
            return [
                {"text": "Better job opportunities in the new city."},
                {"text": "Potential for higher salary and career growth."},
                {"text": "Improved lifestyle and amenities."},
            ]

        if "laptop" in q or "computer" in q:
            return [
                {"text": "Better performance for work and productivity."},
                {"text": "Longer battery life and reliability."},
                {"text": "Access to latest tools and features."},
            ]

        return [
            {"text": f"{stance.upper()} argument: positive potential related to '{question}'."},
            {"text": "Opportunity for personal or professional growth."},
            {"text": "Potential long-term benefits."},
        ]
""")

# ============================================================
# 3) agents/con_agent.py
# ============================================================

w(f"{AGENTS}/con_agent.py", """
from tools.web_search import web_search

class ConAgent:
    def __init__(self, memory):
        self.memory = memory

    def generate(self, question: str):
        points = self.generate_arguments(question, stance="con")
        for p in points:
            p["evidence"] = web_search(p["text"])
        return points

    def generate_arguments(self, question: str, stance="con"):
        q = question.lower()

        if "move" in q:
            return [
                {"text": "Increased cost of living could reduce net savings."},
                {"text": "Separation from friends and family."},
                {"text": "Adjustment stress in a new environment."},
            ]

        if "laptop" in q:
            return [
                {"text": "High upfront cost compared to repairing old device."},
                {"text": "Learning curve and initial setup time."},
                {"text": "Old laptop may still work adequately."},
            ]

        return [
            {"text": f"{stance.upper()} argument: negative risks about '{question}'."},
            {"text": "Potential unwanted expenses."},
            {"text": "Possible uncertainties or hidden issues."},
        ]
""")

# ============================================================
# 4) agents/research_agent.py
# ============================================================

w(f"{AGENTS}/research_agent.py", """
from tools.web_search import web_search

class ResearchAgent:
    def __init__(self, memory):
        self.memory = memory

    def verify_and_expand(self, arguments):
        out = []
        for arg in arguments:
            text = arg["text"]
            facts = web_search(text)
            evidence = facts[:2]
            out.append({
                "text": text,
                "evidence": evidence,
                "reliability": self.estimate_reliability(evidence)
            })
        return out

    def estimate_reliability(self, evidence):
        if not evidence:
            return 0.2
        return min(0.2 + 0.4 * len(evidence), 1.0)
""")

# ============================================================
# 5) agents/evaluator_agent.py
# ============================================================

w(f"{AGENTS}/evaluator_agent.py", """
from tools.scoring import score_arguments

class EvaluatorAgent:
    def __init__(self, memory):
        self.memory = memory

    def evaluate(self, pros, cons, session):
        pro_scores = [score_arguments(a) for a in pros]
        con_scores = [score_arguments(a) for a in cons]

        pro_total = sum(s["score"] for s in pro_scores)
        con_total = sum(s["score"] for s in con_scores)

        total = pro_total + con_total if pro_total + con_total > 0 else 1

        pro_pct = round((pro_total / total) * 100, 1)
        con_pct = round((con_total / total) * 100, 1)

        if pro_pct > con_pct:
            verdict = "Proceed"
        elif con_pct > pro_pct:
            verdict = "Hold/Review"
        else:
            verdict = "Neutral"

        return {
            "pro_score": round(pro_total, 2),
            "con_score": round(con_total, 2),
            "pro_pct": pro_pct,
            "con_pct": con_pct,
            "recommendation": verdict,
            "confidence": round(abs(pro_pct - con_pct), 1)
        }
""")

# ============================================================
# 6) tools/web_search.py
# ============================================================

w(f"{TOOLS}/web_search.py", """
def web_search(query: str):
    if len(query.strip()) < 5:
        return []
    return [
        {"title": f"Fact about '{query}' - source A", "url": "https://example.com/a"},
        {"title": f"Fact about '{query}' - source B", "url": "https://example.com/b"},
    ]
""")

# ============================================================
# 7) tools/memory_manager.py
# ============================================================

w(f"{TOOLS}/memory_manager.py", """
class MemoryManager:
    def __init__(self):
        self.user_profiles = {}
        self.decision_logs = []

    def get_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    def update_profile(self, user_id, data):
        self.user_profiles[user_id] = {**self.user_profiles.get(user_id, {}), **data}

    def add_decision_log(self, user_id, question, pros, cons, verdict):
        self.decision_logs.append({
            "user_id": user_id,
            "question": question,
            "pros": pros,
            "cons": cons,
            "verdict": verdict,
        })
""")

# ============================================================
# 8) tools/scoring.py
# ============================================================

w(f"{TOOLS}/scoring.py", """
def score_arguments(argument):
    score = 0
    evid = argument.get("evidence", [])
    score += min(len(evid) * 1.0, 3.0)
    rel = argument.get("reliability", 0)
    score += rel * 3.0
    score += min(len(argument["text"].split()) * 0.1, 2.0)
    return {"score": round(score,2), "argument": argument}
""")

# ============================================================
print("✔️ DualMind project files created successfully!")
print("Run this next:")
print("!python DualMind/main.py")


!python DualMind/main.py

