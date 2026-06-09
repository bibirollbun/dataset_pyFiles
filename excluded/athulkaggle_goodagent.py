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


from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import uuid

#------------------------------
# Data classes
#------------------------------
@dataclass
class ImpactGoal:
    area: str
    urgency: float
    dependencies: List[str]

@dataclass
class GoodResource:
    title: str
    url: str
    summary: str
    relevance: float

@dataclass
class AwarenessCard:
    question: str
    answer: str
    ease: float = 2.5

#---------------------------------
# In-memory session service
#---------------------------------
class InMemorySessionService:
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
        self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)

#--------------------------------------------
# Mock LLM Agent — Agents for Good
#--------------------------------------------
class MockLLMAgent:
    def summarize(self, text: str, max_length: int = 60) -> str:
        return (text[:max_length] + "...") if len(text) > max_length else text

    def explain(self, topic: str, level: str = "intro") -> str:
        return f"Simple explanation ({level}) for {topic}: Focuses on impact, social good and community benefits."

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, Any]]:
        qs = []
        for i in range(1, n_questions + 1):
            qs.append({
                "q": f"Awareness question {i} on {topic}",
                "a": f"Helpful guidance for question {i} about {topic}."
            })
        return qs

    def answer_query(self, topic: str, query_type: str) -> str:
        if query_type == "prerequisites":
            return f"Basic understanding of {topic}, local community issues, and willingness to help."
        if query_type == "key_concepts":
            return f"Key concepts: root causes, stakeholders, sustainable solutions, long-term community impact."
        if query_type == "action_steps":
            return f"Action steps: volunteer, promote awareness, partner with NGOs, start small community initiatives."
        return f"General guidance for {topic}: stay consistent, collaborate, measure impact."

#--------------------------------------------------
# Search and Awareness Card Agents
#--------------------------------------------------
class ImpactSearchAgent:
    def run(self, area: str) -> List[GoodResource]:
        time.sleep(0.4)
        return [
            GoodResource(
                title=f"{area} – WHO Global Insights",
                url="https://www.who.int",
                summary=f"Trusted global resources addressing {area} challenges.",
                relevance=0.95
            ),
            GoodResource(
                title=f"{area} – Community Action Guide",
                url="https://example.com/action",
                summary=f"Detailed community-level strategies for improving {area}.",
                relevance=0.78
            )
        ]

class AwarenessCardAgent:
    def __init__(self, llm: MockLLMAgent):
        self.llm = llm

    def run(self, area: str, n_cards: int = 5) -> List[AwarenessCard]:
        cards = []
        for i in range(n_cards):
            q = f"What is an important fact about {area}?"
            a = self.llm.explain(area, level="short")
            cards.append(AwarenessCard(question=q, answer=a))
        return cards

#-----------------------------------------------
# Coordinator — parallel agents + sessions
#-----------------------------------------------
class GoodCoordinator:
    def __init__(self, session_service: InMemorySessionService):
        self.llm = MockLLMAgent()
        self.search_agent = ImpactSearchAgent()
        self.card_agent = AwarenessCardAgent(self.llm)
        self.sessions = session_service

    def create_or_get_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def plan_for(self, session_id: str, areas: List[str], weekly_hours: int = 5) -> Dict[str, Any]:
        self.sessions.set_state(session_id, "areas", areas)
        self.sessions.set_state(session_id, "weekly_hours", weekly_hours)

        results = {"resources": {}, "cards": {}, "quizzes": {}, "informative": {}}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            future_search = {ex.submit(self.search_agent.run, a): a for a in areas}
            future_cards = {ex.submit(self.card_agent.run, a, 4): a for a in areas}

            for fut in concurrent.futures.as_completed(future_search):
                area = future_search[fut]
                try:
                    results["resources"][area] = fut.result()
                except:
                    results["resources"][area] = []

            for fut in concurrent.futures.as_completed(future_cards):
                area = future_cards[fut]
                try:
                    results["cards"][area] = fut.result()
                except:
                    results["cards"][area] = []

        for area in areas:
            results["quizzes"][area] = self.llm.generate_quiz(area, n_questions=3)

        query_types = ["prerequisites", "key_concepts", "action_steps"]
        for area in areas:
            ans_map = {}
            for qt in query_types:
                ans_map[qt] = self.llm.answer_query(area, qt)
            results["informative"][area] = ans_map

        self.sessions.set_state(session_id, "last_plan", results)
        return results

#-------------------------------------------------
# Demo run — Healthcare, Education, Sustainability
#-------------------------------------------------
session_service = InMemorySessionService()
coord = GoodCoordinator(session_service)

sid = coord.create_or_get_session("user_123", metadata={"format": "awareness+action"})
print("Created session id:", sid)

areas = [
    "Healthcare Access",
    "Quality Education",
    "Clean Water & Sanitation",
    "Climate Action",
    "Sustainable Cities"
]

plan = coord.plan_for(sid, areas, weekly_hours=8)

print("\n--- Community Impact Plan (Areas) ---")
for a in areas:
    print(f"\nArea: {a}")
    print(" Resources:")
    for r in plan["resources"].get(a, []):
        print(f"   - {r.title} | {r.url} | {r.summary[:80]}...")
    print(f"  Cards: {len(plan['cards'].get(a, []))}")
    print(f"  Quiz: {len(plan['quizzes'].get(a, []))}")
    print("  Informative:")
    for qtype, ans in plan["informative"].get(a, {}).items():
        print(f"   - {qtype}: {ans}")

print("\nStored state keys:", list(session_service.get(sid).get("state", {}).keys()))





