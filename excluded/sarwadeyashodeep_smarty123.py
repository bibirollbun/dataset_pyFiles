from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import uuid


# -----------------------
# Data Classes
# -----------------------

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


# -----------------------
# In-Memory Session Store
# -----------------------

class InMemorySessionService:
    """Simple session store keyed by session id."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "state": {},
        }
        return sid

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self.sessions:
            raise KeyError("Session not found.")
        self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)


# -----------------------
# Mock LLM Agent
# -----------------------

class MockLLMAgent:
    """A lightweight fake LLM that generates summaries, explanations, quizzes, and answers queries."""

    def summarize(self, text: str, max_length: int = 60) -> str:
        """Return a simple truncated summary."""
        return text[:max_length] + "..." if len(text) > max_length else text

    def explain(self, topic: str, level: str = "intro") -> str:
        """Return a simple topic explanation."""
        return (
            f"Explanation ({level}) for {topic}: "
            f"This is a concise learner-friendly explanation covering the main ideas."
        )

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        """Return a list of fake quiz questions."""
        qs = []
        for i in range(1, n_questions + 1):
            qs.append({
                "q": f"Sample question {i} about {topic}.",
                "a": f"Short model answer for question {i} on {topic}."
            })
        return qs

    def answer_query(self, topic: str, query_type: str) -> str:
        """Return short, informative answers for predefined query types."""
        query_type = query_type.lower().strip()

        if query_type == "prerequisites":
            return (
                f"Prerequisites for {topic}: basic Python, linear algebra (vectors/matrices), "
                f"probability fundamentals, and comfort with reading technical blogs."
            )

        if query_type == "key concepts":
            return (
                f"Key concepts for {topic}: core definitions, main algorithms, complexity trade-offs, "
                f"common failure modes, and real-world examples."
            )

        if query_type == "practice problems":
            return (
                f"Practice ideas for {topic}: implement a small example from scratch, "
                f"solve 3 conceptual problems, and reproduce one result from a short tutorial."
            )

        # default fallback
        return (
            f"Short guidance for {topic} regarding '{query_type}': "
            f"Try learning by building a small project or example."
        )


