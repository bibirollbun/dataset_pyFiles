# =============================================================
# Deepali's IntelliLearn Agent System (Freestyle Capstone Track)
# =============================================================

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import time
import concurrent.futures

# -------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------
@dataclass
class LearningObjective:
    topic: str
    difficulty: float
    prerequisites: List[str]

@dataclass
class Resource:
    title: str
    url: str
    summary: str
    relevance: float

@dataclass
class Flashcard:
    question: str
    answer: str
    ease: float = 2.5


# -------------------------------------------------------------
# In-memory session storage
# -------------------------------------------------------------
class DeepaliSessionStore:
    """Session store for personalized learning."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_name: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "user": user_name,
            "created_at": time.time(),
            "state": {}
        }
        return sid

    def get(self, sid: str):
        return self.sessions.get(sid, {})

    def set_state(self, sid: str, key: str, val: Any):
        self.sessions[sid]["state"][key] = val


# -------------------------------------------------------------
# Mock LLM Agent (rewritten for Deepali)
# -------------------------------------------------------------
class DeepaliLLMAgent:
    def summarize(self, text: str, length: int = 50) -> str:
        return text[:length] + "..."

    def explain(self, topic: str, depth: str = "intro"):
        return f"[{depth.upper()} Explanation] {topic} → A clear, easy-to-grasp breakdown with examples."

    def generate_quiz(self, topic: str, n=3):
        return [
            {"q": f"What is concept {i+1} in {topic}?", "a": f"A short answer about concept {i+1}."}
            for i in range(n)
        ]

    def informative(self, topic: str, qtype: str):
        mapping = {
            "prerequisites": f"Basics needed before learning {topic}: Logical thinking, curiosity, and fundamentals.",
            "key_points": f"Important ideas inside {topic}: definitions, workflow, algorithms, use-cases.",
            "practice": f"Practice tasks for {topic}: mini projects, exercises, and hands-on tasks."
        }
        return mapping.get(qtype, "General study recommendation.")


# -------------------------------------------------------------
# Search Agent (rewritten – different sources & structure)
# -------------------------------------------------------------
class DeepaliSearchAgent:
    def run(self, topic: str) -> List[Resource]:
        time.sleep(0.3)

        return [
            Resource(
                title=f"{topic} — Official Documentation",
                url="https://developer.mozilla.org/",
                summary=f"Helpful reference docs for {topic}.",
                relevance=0.92
            ),
            Resource(
                title=f"{topic} Beginner Guide",
                url="https://www.geeksforgeeks.org/",
                summary=f"Beginner-friendly explanation and examples for {topic}.",
                relevance=0.80
            ),
            Resource(
                title=f"{topic} Crash Course Video",
                url="https://www.youtube.com/watch?v=2HaSBYZS___",
                summary=f"Short video introduction to understand the fundamentals of {topic}.",
                relevance=0.88
            )
        ]


# -------------------------------------------------------------
# Flashcard Agent
# -------------------------------------------------------------
class DeepaliFlashcardAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, topic: str, n=5):
        cards = []
        for i in range(n):
            q = f"What is idea {i+1} in {topic}?"
            a = self.llm.explain(topic, depth="short")
            cards.append(Flashcard(question=q, answer=a))
        return cards


# -------------------------------------------------------------
# Master Coordinator (Deepali's IntelliLearn Engine)
# -------------------------------------------------------------
class IntelliLearnCoordinator:
    def __init__(self):
        self.sessions = DeepaliSessionStore()
        self.llm = DeepaliLLMAgent()
        self.search = DeepaliSearchAgent()
        self.flashcards = DeepaliFlashcardAgent(self.llm)

    def create_session(self):
        return self.sessions.create_session("Deepali")

    def build_plan(self, sid: str, topics: List[str], hours=5):
        self.sessions.set_state(sid, "topics", topics)
        self.sessions.set_state(sid, "hours", hours)

        results = {"resources": {}, "flashcards": {}, "quizzes": {}, "info": {}}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
            futures_search = {exe.submit(self.search.run, t): t for t in topics}
            futures_flash = {exe.submit(self.flashcards.generate, t, 4): t for t in topics}

            for fut in concurrent.futures.as_completed(futures_search):
                t = futures_search[fut]
                results["resources"][t] = fut.result()

            for fut in concurrent.futures.as_completed(futures_flash):
                t = futures_flash[fut]
                results["flashcards"][t] = fut.result()

        for t in topics:
            results["quizzes"][t] = self.llm.generate_quiz(t)
            results["info"][t] = {
                "prerequisites": self.llm.informative(t, "prerequisites"),
                "key_points": self.llm.informative(t, "key_points"),
                "practice": self.llm.informative(t, "practice")
            }

        self.sessions.set_state(sid, "final_plan", results)
        return results


# -------------------------------------------------------------
# Demo Run
# -------------------------------------------------------------
coord = IntelliLearnCoordinator()
sid = coord.create_session()
print("Session ID:", sid)

topics = [
    "AI Agents Fundamentals",
    "Tool Calling & Integrations",
    "Cognitive Planning",
    "Autonomous Budget Optimization",
    "Deployment Pipelines"
]

final_output = coord.build_plan(sid, topics, 8)

print("\n--- Deepali's Study Plan Summary ---")
for t in topics:
    print(f"\nTopic: {t}")
    print(" Resources:")
    for r in final_output["resources"][t]:
        print(f"   - {r.title} | {r.url} | {r.summary}")

    print(f" Flashcards: {len(final_output['flashcards'][t])}")
    print(f" Quiz Questions: {len(final_output['quizzes'][t])}")
    print(" Informative Insights:")
    for k, v in final_output["info"][t].items():
        print(f"   • {k}: {v}")

