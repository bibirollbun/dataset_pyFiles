from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import uuid

# ----------------------------------------
# Data classes
# ----------------------------------------

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

# ----------------------------------------
# In-memory session service
# ----------------------------------------

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

# ----------------------------------------
# Mock LLM Agent (feature: Agent powered by an LLM)
# ----------------------------------------

class MockLLMAgent:
    """A mocked LLM agent that 'generates' summaries, explanations, quiz questions and answers queries.
    In a real system this would call an LLM API with prompt engineering and safety checks..."""

    def summarize(self, text: str, max_length: int = 100) -> str:
        # Light-weight fake summarization
        return text[:max_length] + "..." if len(text) > max_length else text

    def explain(self, topic: str, level: str = "intro") -> str:
        return f"Explanation ({level}) for {topic}: This is a concise, learner-friendly explanation covering the main ideas."

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        q = []
        for i in range(1, n_questions + 1):
            q.append({
                "q": f"Sample question {i} about {topic}?",
                "a": f"Short model answer for question {i} on {topic}."
            })
        return q

    def answer_query(self, topic: str, query_type: str) -> str:
        """Return short, informative answers for a set of predefined query types."""
        if query_type == "prerequisites":
            return f"Prerequisites for {topic}: basic Python, linear algebra concepts, probability fundamentals, and comfort with math notation."
        if query_type == "common_failures":
            return f"Common Failures: data and real-world examples."
        if query_type == "key_concepts":
            return f"Key concepts for {topic}: core definitions and real-world examples."
        if query_type == "practice_problems":
            return f"Practice item for {topic}: implement a small exercise, solve 3 conceptual problems, and reproduce one result from a short tutorial."
        return f"Short guidance for {topic} regarding {query_type}: try learning by building a small project."

# ----------------------------------------
# Other agents (search & flashcard generator)
# ----------------------------------------

class ContentSearchAgent:
    def find(self, topic: str) -> List[Resource]:
        """Simulate YouTube playlist and blog-article searches for Day-specific entries."""
        return [Resource(
            title=f"Resource for {topic}",
            url=f"https://example.com/{topic.replace(' ', '_')}",
            summary=f"Summary about {topic}",
            relevance=0.9
        )]

class FlashcardAgent:
    def generate(self, topic: str, n_cards: int = 4) -> List[Flashcard]:
        cards = []
        for i in range(1, n_cards + 1):
            cards.append(Flashcard(
                question=f"Q{i}: What is a key concept of {topic}?",
                answer=f"A{i}: Key concept explanation for {topic}."
            ))
        return cards

# ----------------------------------------
# Study Coordinator (main orchestrator)
# ----------------------------------------

class StudyCoordinator:
    def __init__(self, session_service):
        self.sessions = session_service
        self.llm = MockLLMAgent()
        self.search = ContentSearchAgent()
        self.flashcard = FlashcardAgent()
    
    def create_or_get_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)
    
    def plan_for(self, session_id: str, topics: List[str], weekly_hours: int = 10):
        results = {
            "resources": {},
            "flashcards": {},
            "quizzes": {},
            "informative_queries": {}
        }
        
        # Use parallel agents to search content and generate flashcards
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_resources = {executor.submit(self.search.find, t): t for t in topics}
            future_cards = {executor.submit(self.flashcard.generate, t): t for t in topics}
            
            for fut in concurrent.futures.as_completed(future_resources):
                topic = future_resources[fut]
                try:
                    results["resources"][topic] = fut.result()
                except Exception:
                    results["resources"][topic] = []
            
            for fut in concurrent.futures.as_completed(future_cards):
                topic = future_cards[fut]
                try:
                    results["flashcards"][topic] = fut.result()
                except Exception:
                    results["flashcards"][topic] = []
        
        # Use the LLM agent sequentially to create a short quiz per topic
        for topic in topics:
            quiz = self.llm.generate_quiz(topic, n_questions=3)
            results["quizzes"][topic] = quiz
        
        # Add informative queries (prerequisites, key concepts, practice problems)
        query_types = ["prerequisites", "key_concepts", "practice_problems"]
        for topic in topics:
            answers = {}
            for qt in query_types:
                ans = self.llm.answer_query(topic, qt)
                answers[qt] = ans
            results["informative_queries"][topic] = answers
        
        # Save summary into session for later retrieval
        self.sessions.set_state(session_id, "last_plan", results)
        return results

# ----------------------------------------
# Demo run (using 5-day AI Agent Intensive topics)
# ----------------------------------------

session_service = InMemorySessionService()
coord = StudyCoordinator(session_service)

sid = coord.create_or_get_session(user_id="user_123", metadata={"preferred_format": "video+problems"})
print("Created session id:", sid)

# Use 5 topics representing the 5-day AI Agent Intensive by DR Abhishek
topics = [
    "Day 1 - Foundations of AI Agents",
    "Day 2 - Tools & Integration",
    "Day 3 - Planning & Orchestration",
    "Day 4 - Budgeting & Optimization",
    "Day 5 - Deployment & Observability"
]

plan_result = coord.plan_for(sid, topics, weekly_hours=10)

print("\n--- Plan Summary (topics) ---")
for t in topics:
    res = plan_result["resources"].get(t, [])
    print(f"\nTopic: {t}")
    for r in res:
        print(f"  - {r.title} | {r.url} | {r.summary[:50]}...")
    
    cards = plan_result["flashcards"].get(t, [])
    print(f"  Flashcards: {len(cards)}")
    
    quiz = plan_result["quizzes"].get(t, [])
    print(f"  Quiz questions: {len(quiz)}")


