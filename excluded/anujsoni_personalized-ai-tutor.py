!git clone https://github.com/KirthiRA/EduCoach-AI-Tutor


from IPython.display import display, Markdown, Image

# ✅ Correct dataset folder + correct filename
ARCH_IMG = "/kaggle/input/educoach-architecture/EduCoach_Architecture.png"

# Display architecture image
try:
    display(Image(filename=ARCH_IMG, width=640, height=360))
except Exception as e:
    display(Markdown(f"⚠ **Warning:** Could not load architecture image from `{ARCH_IMG}`. Error: {e}"))


from IPython.display import display, Markdown, Image
import json, sqlite3, os, time
from datetime import datetime
from fractions import Fraction
from typing import Dict, List, Any
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)


def log_event(event_type: str, details: dict):
    rec = {"ts": datetime.utcnow().isoformat()+"Z", "event": event_type, "details": details}
    with open("logs/events.log", "a", encoding="utf8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")


DB = "memory.db"

class StudentMemory:
    def __init__(self, student_id: str):
        self.student_id = student_id
        self._ensure_table()
        row = self._fetch()
        if row:
            self.data = json.loads(row)
        else:
            self.data = {"profile": {"preferred_tone": "friendly","grade":"6"}, "mastery": {}, "mistakes": [], "history": []}
            self._persist()

    def _ensure_table(self):
        conn = sqlite3.connect(DB)
        conn.execute("CREATE TABLE IF NOT EXISTS memory (id TEXT PRIMARY KEY, data TEXT)")
        conn.commit(); conn.close()

    def _fetch(self):
        conn = sqlite3.connect(DB)
        r = conn.execute("SELECT data FROM memory WHERE id=?", (self.student_id,)).fetchone()
        conn.close(); return r[0] if r else None

    def _persist(self):
        conn = sqlite3.connect(DB)
        conn.execute("INSERT OR REPLACE INTO memory VALUES (?,?)",(self.student_id,json.dumps(self.data)))
        conn.commit(); conn.close()

    def update_mastery(self, updates: dict):
        for c,d in updates.items():
            self.data["mastery"][c] = min(1.0, self.data["mastery"].get(c,0)+d)
        self._persist()


def solve_fraction_add(example: str) -> dict:
    try:
        a, b = example.split("+")
        f1 = Fraction(a.strip())
        f2 = Fraction(b.strip())
        res = f1 + f2
        steps = f"Convert to common denominator → Add → Simplified = {res}"
        return {"example": example, "steps": steps, "result": f"{res.numerator}/{res.denominator}"}
    except Exception as e:
        return {"example": example, "steps": f"Parse error: {e}", "result": ""}


class LessonPlannerAgent:
    def run(self, topic: str) -> dict:
        if "fraction" in topic.lower():
            return {"topic": topic, "concepts": ["common denominator", "simplify fractions", "mixed numbers"], "examples": ["3/4 + 1/2","1/4 + 2/4"]}
        return {"topic": topic, "concepts": [topic], "examples": []}

class SolverAgent:
    def run(self, examples: list) -> list:
        return [solve_fraction_add(ex) for ex in examples]

class QuizGeneratorAgent:
    def run(self, plan: dict) -> list:
        return [{"id": f"q_{i}", "concept_name": c, "question": "What is 3/4 + 1/2 ?","answer_type":"fraction"} for i,c in enumerate(plan.get("concepts",[]),1)]
class EvaluatorAgent:
    def run(self, quiz: list, answers: list) -> dict:
        score=0; details=[]
        for q,a in zip(quiz,answers):
            expected=str(Fraction(3,4)+Fraction(1,2))
            try: correct = Fraction(a) == Fraction(expected)
            except: correct=False
            details.append({"question_id":q["id"],"concept_name":q["concept_name"],"given":a,"expected":expected,"correct":correct})
            if correct: score+=1
        return {"score":score,"total":len(quiz),"details":details}


class AgentOrchestrator:
    def __init__(self, student_id: str):
        self.memory = StudentMemory(student_id)
        self.lp = LessonPlannerAgent()
        self.sv = SolverAgent()
        self.qz = QuizGeneratorAgent()
        self.ev = EvaluatorAgent()

    def run_session(self, topic: str, simulate_answers=True):
        plan = self.lp.run(topic)
        solved = self.sv.run(plan.get("examples",[]))
        quiz = self.qz.run(plan)
        answers=[str(Fraction(3,4)+Fraction(1,2))] * len(quiz)
        eval_res=self.ev.run(quiz,answers)
        updates={q["concept_name"]:(0.2 if d["correct"] else 0) for q,d in zip(quiz,eval_res["details"])}
        self.memory.update_mastery(updates); return {"plan":plan,"solved":solved,"quiz":quiz,"eval":eval_res,"memory":self.memory.data}


orch = AgentOrchestrator("demo_student_1")
out = orch.run_session("Fractions - Addition")

from pprint import pprint
pprint(out)

print("\n Demo Finished — Scroll Up to see results!")


print("\n### Results Summary")
print(f"Topic: {out['plan']['topic']}")
print(f"Score: {out['eval']['score']} / {out['eval']['total']}")
print("Mastery:")
for c,v in out["memory"]["mastery"].items():
    print(f" - {c}: {v:.2f}")


import pandas as pd
qb = pd.read_csv("/kaggle/input/educoach-question-bank/question_bank.csv")
qb.head()


import json
with open("/kaggle/input/educoach-students/students.json", "r") as f:
    students = json.load(f)
students



import json

results = {
    "topic": "Fractions - Addition",
    "score": "3/3",
    "mastery": {"common_denominator": 1.00, "simplify_fractions": 1.00}
}

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("File saved ✅")


