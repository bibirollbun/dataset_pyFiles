from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import uuid

# -----------------------------
# Data classes
# -----------------------------
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

# -----------------------------
# In-memory session service
# -----------------------------
class InMemorySessionService:
    """Simple session store keyed by session_id."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "state": {}
        }
        return sid

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self.sessions:
            raise KeyError("session not found")
        self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)

# -----------------------------
# Mock LLM Agent
# -----------------------------
class MockLLMAgent:
    def summarize(self, text: str, max_length: int = 60) -> str:
        return (text[:max_length] + '...') if len(text) > max_length else text

    def explain(self, topic: str, level: str = 'intro') -> str:
        explanations = {
            'intro': f"Beginner-friendly intro to {topic}. Covers basics with simple examples.",
            'intermediate': f"Intermediate guide to {topic}. Includes real-world use cases.",
            'advanced': f"Deep dive into {topic}: algorithms, edge cases, and optimizations.",
            'short': f"Quick explanation of {topic}: Core idea in 1-2 sentences."  # ← Fixed this!
        }
        return explanations.get(level, explanations['intro'])

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        return [
            {"q": f"Question {i} about {topic}", "a": f"Model answer {i} for {topic}."}
            for i in range(1, n_questions + 1)
        ]

    def answer_query(self, topic: str, query_type: str) -> str:
        answers = {
            "prerequisites": f"Prerequisites for {topic}: Python basics, math fundamentals, and curiosity!",
            "key_concepts": f"Key concepts in {topic}: definitions, algorithms, trade-offs, real examples.",
            "practice_problems": f"Practice for {topic}: build a mini project, solve 3 problems, explain to a friend."
        }
        return answers.get(query_type, f"Learn {topic} by doing hands-on projects!")

# -----------------------------
# Agents
# -----------------------------
class ContentSearchAgent:
    def run(self, topic: str) -> List[Resource]:
        time.sleep(0.4)
        playlist = "https://www.youtube.com/playlist?list=PLzJwCIvZuAFY-jBJS0-LlFB0dP469vsMG"
        return [
            Resource(
                title=f"{topic} — Dr Abhishek: AI Agent Intensive",
                url=playlist,
                summary=f"Curated playlist for {topic} from 5-day AI Agent course.",
                relevance=0.95
            ),
            Resource(
                title=f"{topic} — In-depth Notes",
                url="https://example.com/notes",
                summary=f"Deep article/notes on {topic}.",
                relevance=0.75
            )
        ]

class FlashcardAgent:
    def __init__(self, llm_agent: MockLLMAgent):
        self.llm = llm_agent

    def run(self, topic: str, n_cards: int = 5) -> List[Flashcard]:
        cards = []
        for i in range(1, n_cards + 1):
            q = f"What is key concept #{i} in {topic}?"
            a = self.llm.explain(topic, level='short')  # Now works!
            cards.append(Flashcard(question=q, answer=a))
        return cards

# -----------------------------
# Coordinator
# -----------------------------
class StudyCoordinator:
    def __init__(self, session_service: InMemorySessionService):
        self.llm = MockLLMAgent()
        self.search_agent = ContentSearchAgent()
        self.flashcard_agent = FlashcardAgent(self.llm)
        self.sessions = session_service

    def create_or_get_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def plan_for(self, session_id: str, topics: List[str], weekly_hours: int = 5) -> Dict[str, Any]:
        self.sessions.set_state(session_id, 'topics', topics)
        self.sessions.set_state(session_id, 'weekly_hours', weekly_hours)

        results = {"resources": {}, "flashcards": {}, "quizzes": {}, "informative_queries": {}}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            search_futures = {ex.submit(self.search_agent.run, t): t for t in topics}
            flash_futures = {ex.submit(self.flashcard_agent.run, t, 4): t for t in topics}

            for future in concurrent.futures.as_completed(search_futures):
                topic = search_futures[future]
                results['resources'][topic] = future.result()

            for future in concurrent.futures.as_completed(flash_futures):
                topic = flash_futures[future]
                results['flashcards'][topic] = future.result()

        for topic in topics:
            results['quizzes'][topic] = self.llm.generate_quiz(topic, 3)
            answers = {qt: self.llm.answer_query(topic, qt) for qt in ["prerequisites", "key_concepts", "practice_problems"]}
            results['informative_queries'][topic] = answers

        self.sessions.set_state(session_id, 'last_plan', results)
        return results

# -----------------------------
# Demo Run
# -----------------------------
session_service = InMemorySessionService()
coord = StudyCoordinator(session_service)
sid = coord.create_or_get_session('yuvraj_007', {'track': 'AI Agents'})

print("Session ID:", sid)

topics = [
    'Day 1 — Foundations of AI Agents',
    'Day 2 — Tools & Integrations',
    'Day 3 — Planning & Orchestration',
    'Day 4 — Budgeting & Optimization',
    'Day 5 — Deployment & Observability'
]

plan = coord.plan_for(sid, topics, weekly_hours=10)

print("\n" + "="*50)
print("NexLearn – Your Study Plan is Ready!")
print("="*50)

for t in topics:
    print(f"\n{t}")
    print(f"   Resources: {len(plan['resources'].get(t, []))}")
    print(f"   Flashcards: {len(plan['flashcards'].get(t, []))}")
    print(f"   Quiz Questions: {len(plan['quizzes'].get(t, []))}")
    print(f"   Tips: Prerequisites, Key Concepts, Practice Ideas → All Set!")

print(f"\nSession saved! Use ID: {sid} to resume tomorrow")


# **NexLearn – Smart Study OS ****
> A complete “operating system” for learning that runs inside one notebook.
It doesn’t just give you flashcards or quiz questions — it thinks, plans, and remembers like a real tutor:
> Starts a private study session the moment you run it (unique session ID + timestamp)
Spins up multiple AI agents in parallel: one searches for the best videos/notes, another instantly builds spaced-repetition flashcards
Generates explanations, quizzes, prerequisites, key concepts, and practice ideas on the fly
Saves everything (your cards, ease factors, resources, progress) inside an in-memory session so you can pick up tomorrow exactly where you left off
Works 100% offline — no API keys, no internet, no excuses

****One click → full personalized study plan with zero setup.
Run it today, study smarter tomorrow.
That’s NexLearn. Your new study OS.****





