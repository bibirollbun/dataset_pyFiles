from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import uuid

# ----------------------------------------
# Data classes
# ----------------------------------------
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
    """
    A mocked LLM agent that 'generates' explanations, quiz questions and answers queries.
    In a real system this would call an LLM API with prompt engineering and safety checks.
    """
    def explain(self, topic: str, level: str = 'intro') -> str:
        return f"Explanation ({level}) for {topic}: This is a concise, learner-friendly explanation covering the main ideas."

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        qs = []
        for i in range(1, n_questions + 1):
            qs.append({
                "q": f"Sample question {i} about {topic}",
                "a": f"Short model answer for question {i} on {topic}."
            })
        return qs

    def answer_query(self, topic: str, query_type: str) -> str:
        """Return short, informative answers for a set of predefined query types."""
        if query_type == "prerequisites":
            return (f"Prerequisites for {topic}: basic Python, linear algebra (vectors/matrices), "
                    "probability fundamentals, and comfort with reading technical blogs.")
        if query_type == "key_concepts":
            return (f"Key concepts for {topic}: core definitions, main algorithms, complexity trade-offs, "
                    "common failure modes, and real-world examples.")
        if query_type == "practice_problems":
            return (f"Practice ideas for {topic}: implement a small example from scratch, solve 3 conceptual problems, "
                    "and reproduce one result from a short tutorial.")
        # default fallback
        return f"Short guidance for {topic} regarding '{query_type}': try learning by building a small project."

# ----------------------------------------
# Other agents (search & flashcard generator)
# ----------------------------------------
class ContentSearchAgent:
    def run(self, topic: str) -> List[Resource]:
        # Simulate latency
        time.sleep(0.4)
        playlist = "https://youtube.com/playlist?list=PLhQjrBD2T383q7Vn8QnTsVgSvyLpsqL_R&si=Sa2T7rJfRJbV6KwT"
        return [
            Resource(
                title=f"{topic} - CS50X",
                url=playlist,
                summary=f"A curated playlist covering {topic} as part of the CS50X",
                relevance=0.95
            )
        ]

class FlashcardAgent:
    def __init__(self, llm_agent: MockLLMAgent):
        self.llm = llm_agent

    def run(self, topic: str, n_cards: int = 5) -> List[Flashcard]:
        # Use the mocked LLM to create question/answer pairs
        cards = []
        for i in range(n_cards):
            q = f"What is the key idea {i+1} in {topic}?"
            a = self.llm.explain(topic, level='short')
            cards.append(Flashcard(question=q, answer=a))
        return cards

# ----------------------------------------
# Coordinator demonstrating parallel agents + sessions + informative queries
# ----------------------------------------
class StudyCoordinator:
    def __init__(self, session_service: InMemorySessionService):
        self.llm = MockLLMAgent()
        self.search_agent = ContentSearchAgent()
        self.flashcard_agent = FlashcardAgent(self.llm)
        self.sessions = session_service

    def create_or_get_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def plan_for(self, session_id: str, topics: List[str], weekly_hours: int = 5) -> Dict[str, Any]:
        # Save requested topics to session state
        self.sessions.set_state(session_id, 'topics', topics)
        self.sessions.set_state(session_id, 'weekly_hours', weekly_hours)

        # Run content search and flashcard generation in parallel for each topic
        results = {"resources": {}, "flashcards": {}, "quizzes": {}, "informative_queries": {}}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            # schedule futures
            future_to_topic_search = {ex.submit(self.search_agent.run, t): t for t in topics}
            future_to_topic_flash = {ex.submit(self.flashcard_agent.run, t, 4): t for t in topics}
            
            # collect search results
            for fut in concurrent.futures.as_completed(list(future_to_topic_search.keys())):
                topic = future_to_topic_search[fut]
                try:
                    resources = fut.result()
                except Exception as e:
                    resources = []
                results['resources'][topic] = resources

            # collect flashcards
            for fut in concurrent.futures.as_completed(list(future_to_topic_flash.keys())):
                topic = future_to_topic_flash[fut]
                try:
                    cards = fut.result()
                except Exception as e:
                    cards = []
                results['flashcards'][topic] = cards

        # Generate quizzes (sequential for simplicity, could also be parallel)
        for topic in topics:
            quiz = self.llm.generate_quiz(topic, n_questions=3)
            results['quizzes'][topic] = quiz

        # Add informative queries and answer them via LLM agent
        query_types = ["prerequisites", "key_concepts", "practice_problems"]
        for topic in topics:
            answers = {}
            for qt in query_types:
                ans = self.llm.answer_query(topic, qt)
                answers[qt] = ans
            results['informative_queries'][topic] = answers

        # Save summary into session for later retrieval
        self.sessions.set_state(session_id, 'last_plan', results)
        return results

# ----------------------------------------
# Demo run (CS50X)
# ----------------------------------------
if __name__ == "__main__":
    session_service = InMemorySessionService()
    coord = StudyCoordinator(session_service)

    sid = coord.create_or_get_session(user_id='user_123', metadata={'preferred_format': 'videos+problems'})
    print('Created session id:', sid)

    topics = [
        'Day 1 - Lecture 0 - Scratch',
        'Day 2 - Lecture 1 - C',
        'Day 3 - Lecture 2 - Arrays',
        'Day 4 - Lecture 3 - Algorithms',
        'Day 5 - Lecture 4 - Memory'
    ]
    plan_result = coord.plan_for(sid, topics, weekly_hours=10)

    print('\n--- Plan Summary (topics) ---')
    for t in topics:
        res = plan_result['resources'].get(t, [])
        print(f"\nTopic: {t}")
        print('  Resources:')
        for r in res:
            print(f"    - {r.title} | {r.url} | {r.summary[:80]}...")
        cards = plan_result['flashcards'].get(t, [])
        print(f"  Flashcards: {len(cards)} created")
        quiz = plan_result['quizzes'].get(t, [])
        print(f"  Quiz: {len(quiz)} questions")
        # Print informative queries
        info = plan_result['informative_queries'].get(t, {})
        print('  Informative Queries:')
        for qtype, answer in info.items():
            print(f"    - {qtype}: {answer}")

    print(f"\nSession stored state keys: {list(session_service.get(sid).get('state', {}).keys())}")

