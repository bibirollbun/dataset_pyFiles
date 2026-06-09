# ============================================================
#                 EduMentor – Kaggle Notebook Version
#        Full working code with auto-run demo on execution
# ============================================================

from __future__ import annotations
import json
import uuid
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date

# ------------------------------------------------------------
#                     Memory Store
# ------------------------------------------------------------

class MemoryStore:
    """Simple in-memory key/value store."""
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def write(self, key: str, value: Any):
        # auto-serialize date objects to isoformat
        def convert(v):
            if isinstance(v, (datetime, date)):
                return v.isoformat()
            if isinstance(v, dict):
                return {k: convert(v2) for k, v2 in v.items()}
            if isinstance(v, list):
                return [convert(i) for i in v]
            return v

        self.store[key] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "value": convert(value),
        }

    def read(self, key: str):
        entry = self.store.get(key)
        return entry["value"] if entry else None

    def query_keys(self, prefix: str):
        return {k: v for k, v in self.store.items() if k.startswith(prefix)}


# ------------------------------------------------------------
#                     Scheduler
# ------------------------------------------------------------

@dataclass
class Task:
    id: str
    topic: str
    date: str       # store as string for JSON safety
    duration_min: int = 30
    notes: Optional[str] = None
    completed: bool = False
    score: Optional[float] = None


class Scheduler:
    def __init__(self):
        self.tasks: List[Task] = []

    def schedule_tasks(self, tasks: List[Tuple[str, int]], start_date: date, days: int):
        day_list = [start_date + timedelta(days=i) for i in range(days)]
        scheduled: List[Task] = []

        for idx, (topic, duration) in enumerate(tasks):
            assigned_date = day_list[idx % len(day_list)]
            t = Task(
                id=str(uuid.uuid4()),
                topic=topic,
                date=assigned_date.isoformat(),  # store string
                duration_min=duration
            )
            self.tasks.append(t)
            scheduled.append(t)

        return scheduled

    def tasks_for_date(self, query_date: date):
        q = query_date.isoformat()
        return [t for t in self.tasks if t.date == q]

    def mark_completed(self, task_id: str, score: Optional[float] = None):
        for t in self.tasks:
            if t.id == task_id:
                t.completed = True
                t.score = score
                return t
        return None

    def reschedule_task(self, task_id: str, new_date: date):
        for t in self.tasks:
            if t.id == task_id:
                t.date = new_date.isoformat()
                t.completed = False
                t.score = None
                return t
        return None


# ------------------------------------------------------------
#                     Planner Agent
# ------------------------------------------------------------

class PlannerAgent:
    def __init__(self, memory: MemoryStore, scheduler: Scheduler):
        self.memory = memory
        self.scheduler = scheduler

    def break_into_tasks(self, goal: str):
        mapping = {
            "Machine Learning basics": [
                ("Introduction & Math Review", 30),
                ("Linear Regression", 40),
                ("Gradient Descent", 30),
                ("Logistic Regression", 35),
                ("Classification Metrics", 25),
                ("Train/Test Split & Overfit", 30),
                ("Regularization", 30),
                ("Decision Trees", 30),
                ("Ensembles / Random Forests", 35),
                ("Intro to Neural Networks", 40),
            ]
        }
        return mapping.get(goal, [(goal, 30)])

    def create_weekly_schedule(self, goal: str, start_date: date, days: int = 14):
        tasks = self.break_into_tasks(goal)
        scheduled = self.scheduler.schedule_tasks(tasks, start_date, days)

        plan = {
            "goal": goal,
            "start_date": start_date.isoformat(),
            "days": days,
            "tasks": [t.__dict__ for t in scheduled],
        }

        self.memory.write(
            f"plan:{goal}:{start_date.isoformat()}",
            plan
        )

        return plan

    def update_plan_from_evaluator(self, evaluator_report):
        score = evaluator_report.get("score", 100)
        topic = evaluator_report.get("topic")
        task_id = evaluator_report.get("task_id")

        if score < 70:
            for t in self.scheduler.tasks:
                if t.id == task_id:
                    base_date = datetime.fromisoformat(t.date).date()

                    rem1 = Task(
                        id=str(uuid.uuid4()),
                        topic=f"Remedial: {topic}",
                        date=(base_date + timedelta(days=1)).isoformat(),
                        duration_min=25
                    )
                    rem2 = Task(
                        id=str(uuid.uuid4()),
                        topic=f"Remedial: {topic} - Practice",
                        date=(base_date + timedelta(days=2)).isoformat(),
                        duration_min=25
                    )

                    self.scheduler.tasks.extend([rem1, rem2])

                    self.memory.write(
                        f"remedial:{task_id}",
                        {
                            "topic": topic,
                            "reason": evaluator_report.get("feedback"),
                            "created": datetime.utcnow().isoformat()
                        }
                    )
                    print(f"[Planner] Low score on '{topic}' – remedial sessions added.")
                    return True

        print(f"[Planner] Score OK for {topic}. No adjustments.")
        return False


# ------------------------------------------------------------
#                     Teacher Agent
# ------------------------------------------------------------

class TeacherAgent:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def teach(self, topic: str, user_profile=None):
        lesson = self._generate_explanation(topic)
        summary = self._generate_summary(topic)
        flashcards = self._generate_flashcards(topic)
        practice_prompt = self._generate_practice(topic)

        package = {
            "topic": topic,
            "lesson": lesson,
            "summary": summary,
            "flashcards": flashcards,
            "practice_prompt": practice_prompt,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.memory.write(
            f"lesson:{topic}:{datetime.utcnow().date().isoformat()}",
            package
        )

        return package

    def _generate_explanation(self, topic):
        text = f"""
        {topic}

        This lesson explains the concept with an intuitive example.

        Example: {self._topic_example(topic)}
        """
        return textwrap.dedent(text)

    def _generate_summary(self, topic):
        return f"Summary of {topic}: key ideas and usage."

    def _generate_flashcards(self, topic):
        return [
            {"q": f"What is {topic}?", "a": f"A short definition of {topic}."},
            {"q": f"One key idea in {topic}?", "a": "A core idea or formula."},
        ]

    def _generate_practice(self, topic):
        return f"Solve a small exercise related to {topic}."

    def _topic_example(self, topic):
        examples = {
            "Linear Regression": "Predicting house prices.",
            "Gradient Descent": "Iteratively minimizing a loss function.",
            "Introduction & Math Review": "Vectors, matrices, and calculus basics.",
        }
        return examples.get(topic, "A simple real-world example.")


# ------------------------------------------------------------
#                     Evaluator Agent
# ------------------------------------------------------------

class EvaluatorAgent:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def create_quiz(self, topic, q_count=5):
        questions = []
        for i in range(q_count):
            qtype = "mcq" if i % 2 == 0 else "short"
            qid = f"q-{uuid.uuid4().hex[:6]}"

            questions.append({
                "id": qid,
                "type": qtype,
                "topic": topic,
                "prompt": f"Question {i+1} about {topic}.",
                "choices": ["A", "B", "C", "D"] if qtype == "mcq" else None,
                "answer": f"core idea of {topic} #{i+1}"
            })

        return questions

    def grade(self, task: Task, user_answers, questions):
        correct = 0
        feedback = []

        for q in questions:
            qid = q["id"]
            expected = q["answer"]
            user = user_answers.get(qid, "")

            if expected.lower().split()[0] in user.lower():
                correct += 1
                feedback.append({"id": qid, "ok": True})
            else:
                feedback.append({"id": qid, "ok": False})

        score = (correct / len(questions)) * 100

        report = {
            "task_id": task.id,
            "topic": task.topic,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.memory.write(f"result:{task.id}", report)
        return report


# ------------------------------------------------------------
#                     EduMentor Orchestrator
# ------------------------------------------------------------

class EduMentor:
    def __init__(self):
        self.memory = MemoryStore()
        self.scheduler = Scheduler()
        self.planner = PlannerAgent(self.memory, self.scheduler)
        self.teacher = TeacherAgent(self.memory)
        self.evaluator = EvaluatorAgent(self.memory)

    def export_plan_json(self, goal: str, start_date: date):
        key = f"plan:{goal}:{start_date.isoformat()}"
        plan = self.memory.read(key)
        if plan:
            return json.dumps(plan, indent=2)
        return None


# ------------------------------------------------------------
#                     AUTO-RUN DEMO
# ------------------------------------------------------------

app = EduMentor()

goal = "Machine Learning basics"
start = date.today()

print(f"=== Creating Plan for '{goal}' ===\n")
plan = app.planner.create_weekly_schedule(goal, start)

today_tasks = app.scheduler.tasks_for_date(start)

if today_tasks:
    task = today_tasks[0]
    print(f"Today's Task: {task.topic}\n")

    # teacher
    pkg = app.teacher.teach(task.topic)
    print("---- Lesson ----\n", pkg["lesson"])
    print("---- Summary ----\n", pkg["summary"])
    print("---- Flashcards ----")
    for fc in pkg["flashcards"]:
        print(fc)

    # evaluator
    print("\n---- Quiz ----")
    quiz = app.evaluator.create_quiz(task.topic)
    for q in quiz:
        print(q["id"], ":", q["prompt"])

    # simulate answers
    answers = {q["id"]: q["answer"] for q in quiz}

    report = app.evaluator.grade(task, answers, quiz)
    print("\n---- Evaluation Report ----")
    print(json.dumps(report, indent=2))

    # finalize
    app.scheduler.mark_completed(task.id, report["score"])
    app.planner.update_plan_from_evaluator(report)

# export plan
print("\n\n===== PLAN JSON EXPORT =====")
print(app.export_plan_json(goal, start))

