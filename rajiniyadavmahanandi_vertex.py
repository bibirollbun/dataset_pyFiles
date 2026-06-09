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
    """Mocked LLM agent."""
    def summarize(self, text: str, max_length: int = 60) -> str:
        return (text[:max_length] + '...') if len(text) > max_length else text

    def explain(self, topic: str, level: str = 'intro') -> str:
        return f"Explanation ({level}) for {topic}: concise explanation."

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        return [
            {"q": f"Sample question {i} about {topic}",
             "a": f"Short answer for question {i} on {topic}."}
            for i in range(1, n_questions + 1)
        ]

    def answer_query(self, topic: str, query_type: str) -> str:
        if query_type == "prerequisites":
            return (f"Prerequisites for {topic}: basic Python, linear algebra, probability.")
        if query_type == "key_concepts":
            return (f"Key concepts in {topic}: definitions, algorithms, trade-offs, examples.")
        if query_type == "practice_problems":
            return (f"Practice for {topic}: build example, solve 3 problems, reproduce tutorial result.")
        return f"General guidance for {topic}."

# -----------------------------
# Search & Flashcard Agents
# -----------------------------
class ContentSearchAgent:
    def run(self, topic: str) -> List[Resource]:
        time.sleep(0.4)
        playlist = "https://www.youtube.com/playlist?list=PLzJwCIvZuAFY-jBJS0-LlFB0dP469vsMG"
        return [
            Resource(
                title=f"{topic} — Dr Abhishek Playlist",
                url=playlist,
                summary=f"A curated playlist covering {topic}.",
                relevance=0.95
            ),
            Resource(
                title=f"{topic} — Deep Dive Article",
                url="https://example.com/deep",
                summary=f"In-depth article for {topic}.",
                relevance=0.75
            )
        ]

class FlashcardAgent:
    def __init__(self, llm_agent: MockLLMAgent):
        self.llm = llm_agent

    def run(self, topic: str, n_cards: int = 5) -> List[Flashcard]:
        cards = []
        for i in range(n_cards):
            q = f"What is key idea {i+1} in {topic}?"
            a = self.llm.explain(topic, level='short')
            cards.append(Flashcard(question=q, answer=a))
        return cards

# -----------------------------
# Study Coordinator
# -----------------------------
class StudyCoordinator:
    def __init__(self, session_service: InMemorySessionService):
        self.llm = MockLLMAgent()
        self.search_agent = ContentSearchAgent()
        self.flashcard_agent = FlashcardAgent(self.llm)
        self.sessions = session_service

    def create_or_get_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def plan_for(self, session_id: str, topics: List[str], weekly_hours: int = 5) -> Dict[str, Any]:
        self.sessions.set_state(session_id, 'topics', topics)
        self.sessions.set_state(session_id, 'weekly_hours', weekly_hours)

        results = {"resources": {}, "flashcards": {}, "quizzes": {}, "informative_queries": {}}

        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            future_search = {ex.submit(self.search_agent.run, t): t for t in topics}
            future_flash = {ex.submit(self.flashcard_agent.run, t, 4): t for t in topics}

            for fut in concurrent.futures.as_completed(future_search):
                topic = future_search[fut]
                try:
                    results['resources'][topic] = fut.result()
                except:
                    results['resources'][topic] = []

            for fut in concurrent.futures.as_completed(future_flash):
                topic = future_flash[fut]
                try:
                    results['flashcards'][topic] = fut.result()
                except:
                    results['flashcards'][topic] = []

        # Sequential LLM tasks
        for topic in topics:
            results['quizzes'][topic] = self.llm.generate_quiz(topic, 3)

        query_types = ["prerequisites", "key_concepts", "practice_problems"]
        for topic in topics:
            answers = {}
            for qt in query_types:
                answers[qt] = self.llm.answer_query(topic, qt)
            results['informative_queries'][topic] = answers

        self.sessions.set_state(session_id, 'last_plan', results)
        return results

# -----------------------------
# Demo run
# -----------------------------
session_service = InMemorySessionService()
coord = StudyCoordinator(session_service)

sid = coord.create_or_get_session(user_id='user_123', metadata={'preferred_format': 'videos+problems'})
print("Created session id:", sid)

topics = [
    'Day 1 — Foundations of AI Agents',
    'Day 2 — Tools & Integrations',
    'Day 3 — Planning & Orchestration',
    'Day 4 — Budgeting & Optimization',
    'Day 5 — Deployment & Observability'
]

plan_result = coord.plan_for(sid, topics, weekly_hours=10)

print("\n--- Plan Summary ---")
for t in topics:
    print(f"\nTopic: {t}")
    print("  Resources:")
    for r in plan_result["resources"][t]:
        print(f"    - {r.title} | {r.url} | {r.summary[:70]}...")
    print(f"  Flashcards: {len(plan_result['flashcards'][t])}")
    print(f"  Quiz Questions: {len(plan_result['quizzes'][t])}")
    print("  Informative queries:")
    for k, v in plan_result["informative_queries"][t].items():
        print(f"    - {k}: {v}")

print("\nStored session keys:", list(session_service.get(sid)["state"].keys()))


