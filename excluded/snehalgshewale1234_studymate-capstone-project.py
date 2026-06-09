# StudyMate Capstone Project 


import json
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any
import pprint

# -----------------------------
# tools
# -----------------------------
SAMPLE_RESOURCES: List[Dict[str, Any]] = [
    {"id": "res1", "title": "Intro to Algorithms (Article)", "type": "article", "topic": "algorithms", "difficulty": "easy", "url": "https://example.com/algorithms-intro"},
    {"id": "res2", "title": "Sorting Algorithms (Video)", "type": "video", "topic": "algorithms", "difficulty": "medium", "url": "https://example.com/sorting-video"},
    {"id": "res3", "title": "Dynamic Programming (Article)", "type": "article", "topic": "dynamic-programming", "difficulty": "hard", "url": "https://example.com/dp"},
    {"id": "res4", "title": "Binary Trees (Article)", "type": "article", "topic": "data-structures", "difficulty": "medium", "url": "https://example.com/trees"},
    {"id": "res5", "title": "Practice Problems: Arrays", "type": "exercise", "topic": "arrays", "difficulty": "easy", "url": "https://example.com/arrays-problems"},
]


def find_resources(topic: str, difficulty: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Return resources matching a topic and optionally difficulty."""
    if not topic:
        return SAMPLE_RESOURCES[:limit]
    results = [r for r in SAMPLE_RESOURCES if r.get("topic") == topic]
    if difficulty:
        results = [r for r in results if r.get("difficulty") == difficulty]
    return results[:limit] if results else SAMPLE_RESOURCES[:limit]


def generate_quiz(topic: str, n_questions: int = 5) -> List[Dict[str, Any]]:
    """Produce a list of simple MCQ questions for the topic.
    Placeholder implementation for offline demos.
    """
    pool = [
        {"q": "What is the time complexity of binary search?", "choices": ["O(log n)", "O(n)", "O(1)", "O(n log n)"], "answer": "O(log n)"},
        {"q": "Which data structure is best for FIFO operations?", "choices": ["Stack", "Queue", "Heap", "Tree"], "answer": "Queue"},
        {"q": "What does DP stand for in algorithms context?", "choices": ["Dynamic Programming", "Data Processing", "Direct Pass", "Double Pointer"], "answer": "Dynamic Programming"},
        {"q": "Which algorithm sorts by partitioning?", "choices": ["Merge Sort", "Quick Sort", "Insertion Sort", "Bubble Sort"], "answer": "Quick Sort"},
        {"q": "Which data structure is commonly used for LIFO?", "choices": ["Queue", "Stack", "Graph", "Heap"], "answer": "Stack"},
    ]
    out = []
    for i in range(max(1, n_questions)):
        out.append(random.choice(pool))
    return out


def next_review_date(current_date: str, quality: int, factor: float = 2.5) -> str:
    """Compute next review date (ISO). quality in 0-5 range."""
    try:
        dt = datetime.date.fromisoformat(current_date)
    except Exception:
        dt = datetime.date.today()
    if quality < 3:
        next_dt = dt + datetime.timedelta(days=1)
    else:
        days = int(max(1, round(factor ** (quality - 2))))
        next_dt = dt + datetime.timedelta(days=days)
    return next_dt.isoformat()

# -----------------------------
# persistence
# -----------------------------
DB_PATH = Path("studymate_db.json")


def load_db() -> Dict[str, Any]:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"profiles": {}, "plans": {}, "progress": {}}
    return {"profiles": {}, "plans": {}, "progress": {}}


def save_db(db: Dict[str, Any]):
    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def get_profile(user_id: str) -> Dict[str, Any]:
    db = load_db()
    return db.get("profiles", {}).get(user_id, {})


def save_profile(user_id: str, profile: Dict[str, Any]):
    db = load_db()
    db.setdefault("profiles", {})[user_id] = profile
    save_db(db)


def save_plan(user_id: str, plan: Dict[str, Any]):
    db = load_db()
    db.setdefault("plans", {})[user_id] = plan
    save_db(db)


def get_plan(user_id: str) -> Dict[str, Any]:
    db = load_db()
    return db.get("plans", {}).get(user_id, {})


def save_progress(user_id: str, progress: Dict[str, Any]):
    db = load_db()
    db.setdefault("progress", {})[user_id] = progress
    save_db(db)


def get_progress(user_id: str) -> Dict[str, Any]:
    db = load_db()
    return db.get("progress", {}).get(user_id, {})

# -----------------------------
# agents
# -----------------------------
class ResourceAgent:
    """Curates resources using the offline DB."""
    def get_resources_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        res = find_resources(topic)
        return res if res else SAMPLE_RESOURCES


class PlannerAgent:
    """Generates a day-by-day plan using resources and profile."""
    def __init__(self, resource_agent: ResourceAgent):
        self.resource_agent = resource_agent

    def run(self, profile: Dict[str, Any], topic: str, weeks: int = 4) -> Dict[str, Any]:
        days = max(1, weeks) * 7
        plan: Dict[str, Any] = {}
        resources = self.resource_agent.get_resources_for_topic(topic)
        if not resources:
            resources = SAMPLE_RESOURCES
        for d in range(1, days + 1):
            day_key = f"Day {d}"
            res = resources[(d - 1) % len(resources)]
            plan[day_key] = {"topic": res.get("topic", topic), "resource": res, "planned_minutes": profile.get("daily_minutes", 60)}
        meta = {"topic": topic, "weeks": weeks, "created_at": datetime.date.today().isoformat()}
        return {"plan": plan, "meta": meta}


class QuizAgent:
    def make_quiz(self, topic: str, n_questions: int = 5) -> List[Dict[str, Any]]:
        return generate_quiz(topic, n_questions)

    def grade_quiz(self, quiz: List[Dict[str, Any]], answers: List[str]) -> Dict[str, Any]:
        correct = 0
        for q, a in zip(quiz, answers):
            if a and a.strip().lower() == q.get("answer", "").strip().lower():
                correct += 1
        score = int(100 * correct / len(quiz)) if quiz else 0
        return {"correct": correct, "total": len(quiz), "score": score}


class ProgressAgent:
    def update_progress(self, user_progress: Dict[str, Any], topic: str, quiz_result: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.date.today().isoformat()
        hist = user_progress.get(topic, {}).get("history", [])
        hist.append({"date": now, "score": quiz_result.get("score", 0)})
        score = quiz_result.get("score", 0)
        if score >= 90:
            quality = 5
        elif score >= 75:
            quality = 4
        elif score >= 50:
            quality = 2
        else:
            quality = 1
        next_rev = next_review_date(now, quality)
        user_progress.setdefault(topic, {})["history"] = hist
        user_progress[topic]["next_review"] = next_rev
        return user_progress

# -----------------------------
# demo & tests
# -----------------------------

def demo(user_id: str = "kaggle_demo_user") -> None:
    """Run a demo sequence and print outputs. Suitable for Kaggle Notebook cell."""
    print("=== Agent StudyMate Demo ===")
    # profile
    profile = {"name": "Kaggle Student", "daily_minutes": 45}
    save_profile(user_id, profile)
    print("Saved profile:")
    pprint.pprint(get_profile(user_id))

    # generate plan
    topic = "algorithms"
    resource_agent = ResourceAgent()
    planner = PlannerAgent(resource_agent)
    result = planner.run(profile, topic, weeks=2)
    plan = result["plan"]
    meta = result["meta"]
    print("Plan meta:")
    pprint.pprint(meta)

    print("Plan sample (first 7 days):")
    for d in list(plan.keys())[:7]:
        info = plan[d]
        print(f"{d}: {info['resource']['title']} ({info['resource']['type']}) - {info['planned_minutes']} min")

    # quiz
    quiz_agent = QuizAgent()
    quiz = quiz_agent.make_quiz(topic, n_questions=4)
    print("Generated Quiz:")
    for i, q in enumerate(quiz, 1):
        print(f"Q{i}: {q['q']}")
        for j, c in enumerate(q['choices'], 1):
            print(f"   {j}. {c}")

    # simulate answers (first choice)
    answers = [q['choices'][0] for q in quiz]
    result_score = quiz_agent.grade_quiz(quiz, answers)
    print("Simulated quiz result:", result_score)

    # update progress
    
    progress_agent = ProgressAgent()
    user_prog = get_progress(user_id) or {}
    updated_prog = progress_agent.update_progress(user_prog, topic, result_score)
    save_progress(user_id, updated_prog)
    print("Saved progress for user:")
    pprint.pprint(get_progress(user_id))

    # save plan
    save_plan(user_id, {"plan": plan, "meta": meta})
    print("Saved plan. Inspect 'studymate_db.json' in notebook files to view DB.")


def basic_tests() -> None:
    """Run a small battery of tests. Raises AssertionError on failure."""
    print("Running basic tests...")
    # tools
    quiz = generate_quiz("algorithms", n_questions=3)
    assert len(quiz) == 3, "generate_quiz returned wrong number of questions"
    print("generate_quiz OK")

    date_now = datetime.date.today().isoformat()
    nr = next_review_date(date_now, quality=5)
    assert isinstance(nr, str)
    print("next_review_date OK ->", nr)

    res = find_resources("algorithms")
    assert isinstance(res, list) and len(res) > 0
    print("find_resources OK")

    # persistence
    USER = "test_user"
    save_profile(USER, {"name": "Tester", "daily_minutes": 30})
    p = get_profile(USER)
    assert p["name"] == "Tester"
    print("persistence profile OK")

    # agents
    ra = ResourceAgent()
    planner = PlannerAgent(ra)
    plan_res = planner.run(p, "algorithms", weeks=1)
    assert "plan" in plan_res and "meta" in plan_res
    print("PlannerAgent OK")

    qa = QuizAgent()
    quiz2 = qa.make_quiz("algorithms", n_questions=2)
    # grade with correct answers
    answers = [q['answer'] for q in quiz2]
    graded = qa.grade_quiz(quiz2, answers)
    assert graded['score'] == 100
    print("QuizAgent OK")

    pg = ProgressAgent()
    prog = {}
    prog2 = pg.update_progress(prog, "algorithms", {"score": 85})
    assert "algorithms" in prog2 and "next_review" in prog2["algorithms"]
    print("ProgressAgent OK")

    print("All basic tests passed.")


# If executed as script in notebook cell, expose demo() and basic_tests() to user
if __name__ == "__main__":
    demo()





