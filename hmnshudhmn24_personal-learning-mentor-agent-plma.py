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


import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from jinja2 import Template

# ---------------------------
# Utilities
# ---------------------------

OUTPUT_DIR = "outputs"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# Mock LLM
# ---------------------------

class MockLLM:
    """A small deterministic mock that returns templated text.
    This allows the notebook to run without any external API.
    Use the prompt name + context to control output.
    """
    @staticmethod
    def call(kind: str, **kwargs) -> str:
        # Basic templating for kinds we need
        if kind == "plan":
            goal = kwargs.get("goal")
            topics = MockLLM._topics_for_goal(goal)
            bullets = "\n".join([f"- {t}" for t in topics])
            return f"Learning plan for '{goal}':\n{bullets}"
        if kind == "lesson":
            topic = kwargs.get("topic")
            length = kwargs.get("length", "short")
            explanation = MockLLM._explain_topic(topic)
            return explanation
        if kind == "quiz":
            topic = kwargs.get("topic")
            n = kwargs.get("n", 5)
            return json.dumps(MockLLM._generate_quiz(topic, n), ensure_ascii=False)
        if kind == "evaluate":
            # kwargs: answers (list), correct (list)
            answers = kwargs.get("answers", [])
            correct = kwargs.get("correct", [])
            score = sum(1 for a, c in zip(answers, correct) if a == c)
            return json.dumps({"score": score, "total": len(correct)})
        return ""

    @staticmethod
    def _topics_for_goal(goal: str) -> List[str]:
        # rudimentary mapping
        goal_l = goal.lower()
        if "python" in goal_l:
            return ["Variables & Types", "If/Else", "Loops", "Functions", "Lists & Dicts"]
        if "web" in goal_l or "html" in goal_l:
            return ["HTML Basics", "CSS Basics", "Responsive Layouts", "Intro to JavaScript"]
        # fallback
        return ["Foundations", "Core Concepts", "Practice & Projects"]

    @staticmethod
    def _explain_topic(topic: str) -> str:
        # small library of explanations
        explanations = {
            "Variables & Types": (
                "Variables store values. In Python, you write: x = 10\n"
                "Common types: int, float, str, bool. Use type(x) to check."
            ),
            "If/Else": (
                "If statements let you branch code: if condition:\n    do_something()\n" 
                "Use elif and else for multiple branches."
            ),
            "Loops": (
                "For loops iterate over sequences: for i in range(5):\n    print(i)\n"
                "While loops run while a condition is true."
            ),
            "Functions": (
                "Functions package logic: def add(a, b):\n    return a + b\n"
                "They make code reusable and testable."
            ),
            "Lists & Dicts": (
                "Lists hold ordered items: lst = [1,2,3]\nDicts map keys to values: d = {'a':1}"
            ),
        }
        return explanations.get(topic, f"Short lesson about {topic}. Concepts, examples, and a quick exercise.")

    @staticmethod
    def _generate_quiz(topic: str, n: int) -> List[Dict[str, Any]]:
        base_q = []
        if topic == "Variables & Types":
            base_q = [
                {"q": "What type is 3.14 in Python?", "choices": ["int", "float", "str", "bool"], "a": "float"},
                {"q": "Which of these is a string?", "choices": ["True", "3", "\"hello\"", "None"], "a": "\"hello\""},
                {"q": "How do you check the type of x?", "choices": ["typeof(x)", "type(x)", "check(x)", "x.type()"], "a": "type(x)"},
            ]
        elif topic == "If/Else":
            base_q = [
                {"q": "Which keyword checks multiple conditions in order?", "choices": ["switch", "elif", "else if", "case"], "a": "elif"},
                {"q": "What does 'else' do?", "choices": ["Only runs when loop ends", "Runs if previous if/elif were false", "Stops the program", "Declares a variable"], "a": "Runs if previous if/elif were false"},
            ]
        elif topic == "Loops":
            base_q = [
                {"q": "How to loop 5 times?", "choices": ["for i in 5:", "for i in range(5):", "loop(5)", "repeat 5 times"], "a": "for i in range(5):"},
                {"q": "Which loop may never terminate if condition never false?", "choices": ["for", "while", "do-while", "foreach"], "a": "while"},
            ]
        else:
            base_q = [
                {"q": f"Basic question about {topic}", "choices": ["A", "B", "C", "D"], "a": "A"}
            ]
        # sample up to n
        out = []
        for i in range(min(n, len(base_q))):
            out.append(base_q[i])
        # if need more, duplicate with small perturbation
        while len(out) < n:
            q = random.choice(base_q).copy()
            q['q'] = q['q'] + " (variant)"
            out.append(q)
        return out

# ---------------------------
# Agents
# ---------------------------

@dataclass
class Session:
    user_id: str = "default_user"
    goal: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)

class PlannerAgent:
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def create_plan(self, goal: str) -> List[str]:
        resp = self.llm.call("plan", goal=goal)
        # parse naive
        lines = [l.strip().lstrip("- ") for l in resp.splitlines() if l.strip() and l.strip().startswith("-")]
        if not lines:
            # fallback to simple split
            lines = self.llm._topics_for_goal(goal)
        return lines

class TutorAgent:
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def teach(self, topic: str) -> str:
        lesson = self.llm.call("lesson", topic=topic)
        # small enrichment
        template = Template("""
### Lesson: {{topic}}\n\n{{lesson}}\n\n**Quick exercise:** Try one small example related to the topic.\n""")
        return template.render(topic=topic, lesson=lesson)

class QuizAgent:
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def create_quiz(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        raw = self.llm.call("quiz", topic=topic, n=n)
        try:
            quiz = json.loads(raw)
        except Exception:
            quiz = MockLLM._generate_quiz(topic, n)
        # ensure structure
        for q in quiz:
            if 'choices' not in q:
                q['choices'] = ["A","B","C","D"]
        return quiz

class EvaluationAgent:
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def evaluate(self, quiz: List[Dict[str, Any]], answers: List[str]) -> Dict[str, Any]:
        correct = [q.get('a') for q in quiz]
        resp = self.llm.call("evaluate", answers=answers, correct=correct)
        try:
            result = json.loads(resp)
        except Exception:
            score = sum(1 for a, c in zip(answers, correct) if a == c)
            result = {"score": score, "total": len(correct)}
        # attach details
        details = []
        for q, a, c in zip(quiz, answers, correct):
            details.append({"q": q['q'], "your": a, "correct": c, "ok": a == c})
        result['details'] = details
        return result

class ProgressTracker:
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.file = progress_file
        self.data = load_json(self.file, default={}) or {}

    def update(self, user_id: str, topic: str, score: int, total: int):
        user = self.data.setdefault(user_id, {})
        topic_record = user.setdefault(topic, {"attempts": 0, "best": 0})
        topic_record['attempts'] += 1
        pct = int(score / total * 100) if total > 0 else 0
        if pct > topic_record['best']:
            topic_record['best'] = pct
        self._save()

    def _save(self):
        save_json(self.file, self.data)

    def get_progress(self, user_id: str):
        return self.data.get(user_id, {})

# ---------------------------
# Orchestrator
# ---------------------------

class Orchestrator:
    def __init__(self, session: Session):
        self.llm = MockLLM()
        self.planner = PlannerAgent(self.llm)
        self.tutor = TutorAgent(self.llm)
        self.quiz = QuizAgent(self.llm)
        self.eval = EvaluationAgent(self.llm)
        self.tracker = ProgressTracker()
        self.session = session

    def run_learning_session(self, goal: str, topics_to_run: int = 1, questions_per_topic: int = 5, interactive: bool = False):
        self.session.goal = goal
        plan = self.planner.create_plan(goal)
        save_json(os.path.join(OUTPUT_DIR, "plan.json"), {"goal": goal, "plan": plan})
        print(f"Plan created: {plan}")

        results_summary = []
        for idx, topic in enumerate(plan[:topics_to_run]):
            print(f"\n--- Topic {idx+1}/{topics_to_run}: {topic} ---")
            lesson = self.tutor.teach(topic)
            lesson_file = os.path.join(OUTPUT_DIR, f"lesson_{idx+1}_{topic.replace(' ','_')}.md")
            with open(lesson_file, "w", encoding="utf-8") as f:
                f.write(lesson)
            print("Lesson written to", lesson_file)

            qset = self.quiz.create_quiz(topic, n=questions_per_topic)
            quiz_file = os.path.join(OUTPUT_DIR, f"quiz_{idx+1}_{topic.replace(' ','_')}.json")
            save_json(quiz_file, qset)
            print("Quiz created with", len(qset), "questions. Saved to", quiz_file)

            # Get answers (interactive or simulated)
            answers = []
            for q in qset:
                # For interactive mode, ask the user
                if interactive:
                    print("Q:", q['q'])
                    for i, choice in enumerate(q['choices']):
                        print(f"  {i+1}. {choice}")
                    sel = input("Enter choice text exactly (or index): ").strip()
                    # allow index
                    if sel.isdigit():
                        sel_i = int(sel)-1
                        if 0 <= sel_i < len(q['choices']):
                            sel = q['choices'][sel_i]
                    answers.append(sel)
                else:
                    # Simulate a learner: pick correct with 60% chance else random wrong
                    if random.random() < 0.6:
                        answers.append(q['a'])
                    else:
                        wrongs = [c for c in q['choices'] if c != q['a']]
                        answers.append(random.choice(wrongs) if wrongs else q['choices'][0])

            eval_result = self.eval.evaluate(qset, answers)
            print(f"Scored {eval_result['score']} / {eval_result['total']}")

            # update progress
            self.tracker.update(self.session.user_id, topic, eval_result['score'], eval_result['total'])
            user_progress = self.tracker.get_progress(self.session.user_id)

            # save detailed result
            result_obj = {
                "topic": topic,
                "score": eval_result['score'],
                "total": eval_result['total'],
                "details": eval_result['details']
            }
            results_summary.append(result_obj)
            save_json(os.path.join(OUTPUT_DIR, f"result_{idx+1}_{topic.replace(' ','_')}.json"), result_obj)
            # also save a pandas-friendly CSV
            df = pd.DataFrame(eval_result['details'])
            df.to_csv(os.path.join(OUTPUT_DIR, f"result_{idx+1}_{topic.replace(' ','_')}.csv"), index=False)

        # final summary
        save_json(os.path.join(OUTPUT_DIR, "session_summary.json"), {"goal": goal, "results": results_summary})
        print("\nSession complete. Outputs saved to the outputs/ folder.")
        return {"plan": plan, "results": results_summary, "progress": self.tracker.get_progress(self.session.user_id)}

# ---------------------------
# Demo run
# ---------------------------

if __name__ == '__main__':
    # Demo: runs a 1-topic session for 'Python basics' non-interactively (deterministic-ish)
    sess = Session(user_id="hmnshudhmn24")
    orch = Orchestrator(sess)
    summary = orch.run_learning_session("Learn Python basics", topics_to_run=1, questions_per_topic=5, interactive=False)
    print(json.dumps(summary, indent=2))

# End of notebook/script


