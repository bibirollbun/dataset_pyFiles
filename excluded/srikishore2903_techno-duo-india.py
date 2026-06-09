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
# Smart Study Agent - Multi-Agent AI Learning Assistant
# Created for Google AI Agents Intensive Course Capstone Project

# Import required libraries
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


@dataclass
class LearningResource:
    """Store educational resources with metadata"""
    title: str
    url: str
    summary: str
    resource_type: str  # video, article, tutorial



@dataclass
class LearningResource:
    """Store educational resources with metadata"""
    title: str
    url: str
    summary: str
    resource_type: str  # video, article, tutorial



@dataclass
class FlashCard:
    """Flashcard for spaced repetition learning"""
    question: str
    answer: str
    ease_factor: float = 2.5  # For spaced repetition logic



@dataclass
class StudySession:
    """Track a userâ€™s study session"""
    session_id: str
    user_id: str
    timestamp: datetime
    topic: str
    resources: List[LearningResource] = field(default_factory=list)
    flashcards: List[FlashCard] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)



class MemorySessionService:
    """Manages user sessions and in-notebook memory"""
    
    def __init__(self):
        self.sessions: Dict[str, StudySession] = {}
        
    def create_session(self, user_id: str, topic: str) -> str:
        session_id = str(uuid.uuid4())
        session = StudySession(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            topic=topic
        )
        self.sessions[session_id] = session
        print(f"âœ… Session created: {session_id} for topic: {topic}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[StudySession]:
        return self.sessions.get(session_id)
    
    def add_resource(self, session_id: str, resource: LearningResource):
        if session_id in self.sessions:
            self.sessions[session_id].resources.append(resource)
    
    def add_flashcard(self, session_id: str, flashcard: FlashCard):
        if session_id in self.sessions:
            self.sessions[session_id].flashcards.append(flashcard)



class LLMAgent:
    """Agent for explanation, summarization, and query analysis"""
    
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model_name = model_name
        
    def explain_topic(self, topic: str) -> str:
        explanations = {
            "Foundation of AI Agents": "AI Agents are autonomous systems that perceive their environment, make decisions, and act to achieve goals.",
            "Tools and Integration": "Agents can use external tools and APIs to perform actions beyond text generation.",
            "Planning and Orchestration": "Agents can split tasks into subtasks, run them sequentially or in parallel, and coordinate the workflow.",
            "Budgeting and Optimization": "Agents should manage token usage, latency, and cost while staying reliable.",
            "Deployment and Observability": "Production agents need logging, monitoring, and evaluation to stay safe and performant."
        }
        return explanations.get(topic, f"Overview of {topic}: key concepts and practical use cases.")
    
    def analyze_query(self, query: str) -> Dict:
        return {
            "intent": "study_planning",
            "topics": ["AI Agents"],
            "complexity": "intermediate"
        }
    
    def summarize_content(self, content: str) -> str:
        return f"Summary: {content[:200]}..."



class ContentSearchAgent:
    """Agent for finding learning resources (mocked)"""
    
    def __init__(self):
        self.resource_database = {
            "Foundation of AI Agents": [
                {"title": "Intro to AI Agents", "url": "https://youtube.com/watch?v=example1", "type": "video"},
                {"title": "Agent Architectures", "url": "https://blog.example.com/agents", "type": "article"}
            ],
            "Tools and Integration": [
                {"title": "Using Tools with Agents", "url": "https://youtube.com/watch?v=example2", "type": "video"},
                {"title": "MCP and APIs", "url": "https://docs.example.com/mcp", "type": "article"}
            ],
            "Planning and Orchestration": [
                {"title": "Multi-Agent Orchestration", "url": "https://youtube.com/watch?v=example3", "type": "video"},
                {"title": "Sequential vs Parallel Agents", "url": "https://docs.example.com/orchestration", "type": "article"}
            ],
            "Budgeting and Optimization": [
                {"title": "Optimizing Agent Costs", "url": "https://youtube.com/watch?v=example4", "type": "video"},
                {"title": "Token Budgeting Guide", "url": "https://guide.example.com/budget", "type": "article"}
            ],
            "Deployment and Observability": [
                {"title": "Deploying AI Agents", "url": "https://youtube.com/watch?v=example5", "type": "video"},
                {"title": "Monitoring Agents in Prod", "url": "https://docs.example.com/observability", "type": "article"}
            ]
        }
    
    def search_resources(self, topic: str) -> List[LearningResource]:
        resources = []
        for res in self.resource_database.get(topic, []):
            resources.append(
                LearningResource(
                    title=res["title"],
                    url=res["url"],
                    summary=f"Key points about {topic}",
                    resource_type=res["type"]
                )
            )
        return resources



class FlashCardAgent:
    """Agent that generates flashcards (static templates)"""
    
    def generate_flashcards(self, topic: str, count: int = 4) -> List[FlashCard]:
        templates = {
            "Foundation of AI Agents": [
                ("What is an AI agent?", "An autonomous system that perceives, reasons, and acts toward goals."),
                ("How is an agent different from a plain LLM?", "Agents can use tools, remember context, and take actions."),
                ("Give one key property of agents.", "Autonomy in decision-making."),
                ("Where are AI agents useful?", "In assistants, automation, customer support, and many workflows.")
            ],
            "Tools and Integration": [
                ("Why do agents need tools?", "To perform real actions like web search, database queries, and APIs."),
                ("What is an API in this context?", "A defined interface that lets the agent call external services."),
                ("What is MCP?", "A protocol for connecting agents with tools in a structured way."),
                ("Give one example of a tool.", "A function that fetches stock prices for the agent.")
            ],
            "Planning and Orchestration": [
                ("What is a sequential agent?", "An agent that runs steps one after another, passing results forward."),
                ("What is a parallel agent?", "An agent that runs multiple steps at the same time to save time."),
                ("When to use loops in agents?", "When you need to repeat a step until a condition is met."),
                ("What is multi-agent orchestration?", "Coordinating multiple specialized agents on one big task.")
            ]
        }
        
        cards_data = templates.get(topic, [])[:count]
        flashcards = [
            FlashCard(question=q, answer=a) for q, a in cards_data
        ]
        return flashcards



class ParallelAgentOrchestrator:
    """Runs content search + flashcard generation together"""
    
    def __init__(self, search_agent: ContentSearchAgent, flashcard_agent: FlashCardAgent):
        self.search_agent = search_agent
        self.flashcard_agent = flashcard_agent
    
    def process_topic(self, topic: str) -> Dict:
        print(f"ğŸ”„ Running parallel agents for topic: {topic}")
        
        resources = self.search_agent.search_resources(topic)
        flashcards = self.flashcard_agent.generate_flashcards(topic)
        
        return {
            "topic": topic,
            "resources": resources,
            "flashcards": flashcards
        }



class SmartStudyAgent:
    """Main orchestrator for the Smart Study system"""
    
    def __init__(self):
        self.memory_service = MemorySessionService()
        self.llm_agent = LLMAgent()
        self.content_agent = ContentSearchAgent()
        self.flashcard_agent = FlashCardAgent()
        self.parallel_orchestrator = ParallelAgentOrchestrator(
            self.content_agent,
            self.flashcard_agent
        )
    
    def start_learning_session(self, user_id: str, topics: List[str]):
        print("=" * 60)
        print("ğŸ�“ SMART STUDY AGENT - 5-Day Learning Plan")
        print("=" * 60)
        
        for day, topic in enumerate(topics, start=1):
            print(f"\n{'='*60}")
            print(f"ğŸ“š DAY {day}: {topic}")
            print(f"{'='*60}\n")
            
            # 1. Create session
            session_id = self.memory_service.create_session(user_id, topic)
            
            # 2. LLM explanation
            explanation = self.llm_agent.explain_topic(topic)
            print("ğŸ’¡ Concept Overview:\n")
            print(explanation)
            print()
            
            # 3. Parallel agents: resources + flashcards
            result = self.parallel_orchestrator.process_topic(topic)
            
            # 4. Show resources
            print("ğŸ“– Recommended Resources:")
            for i, res in enumerate(result["resources"], start=1):
                print(f"  {i}. [{res.resource_type.upper()}] {res.title}")
                print(f"     ğŸ”— {res.url}")
                self.memory_service.add_resource(session_id, res)
            
            # 5. Show flashcards
            print("\nğŸƒ� Flashcards:")
            for i, card in enumerate(result["flashcards"], start=1):
                print(f"  Q{i}: {card.question}")
                print(f"  A{i}: {card.answer}\n")
                self.memory_service.add_flashcard(session_id, card)
            
            print(f"âœ… Session stored with ID: {session_id}")



# Initialize the Smart Study Agent
study_agent = SmartStudyAgent()

# 5-day AI Agents curriculum (you can edit names)
curriculum = [
    "Foundation of AI Agents",
    "Tools and Integration",
    "Planning and Orchestration",
    "Budgeting and Optimization",
    "Deployment and Observability"
]

# Start the learning plan
study_agent.start_learning_session(
    user_id="student_001",
    topics=curriculum
)


