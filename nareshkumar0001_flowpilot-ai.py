# ==========================
# CELL 8 — FlowPilot Agent Skeleton
# ==========================

# Import required libraries
from typing import List, Dict
from dataclasses import dataclass
import uuid
import logging

# --------------------------
# Setup Logging for Observability
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# Flashcard Data Class
# --------------------------
@dataclass
class Flashcard:
    question: str
    answer: str
    source_url: str
    topic: str

# --------------------------
# Memory and Session Management
# --------------------------
class MemoryBank:
    def __init__(self):
        self.long_term_memory = {}
    
    def store_flashcard(self, flashcard: Flashcard):
        self.long_term_memory[flashcard.question] = flashcard

    def get_flashcard(self, question: str):
        return self.long_term_memory.get(question, None)

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.short_term_memory = []

    def add_to_session(self, item):
        self.short_term_memory.append(item)

# --------------------------
# Agent Skeletons
# --------------------------
class SearchAgent:
    def fetch_resources(self, query: str) -> List[Dict]:
        # Example: fetch top URLs and titles using Google Search / OpenAPI
        logging.info(f"Searching for resources related to '{query}'")
        # Placeholder for demo
        return [{"title": "Example Article", "url": "https://example.com"}]

class LLMExplanationAgent:
    def explain_content(self, content: str) -> str:
        logging.info("Generating explanation using LLM")
        # Placeholder for demo
        return f"Explanation for: {content}"

class FlashcardAgent:
    def generate_flashcards(self, content: str, topic: str) -> List[Flashcard]:
        logging.info("Generating flashcards")
        # Placeholder example
        flashcard = Flashcard(
            question=f"What is {topic}?",
            answer=f"{content}",
            source_url="https://example.com",
            topic=topic
        )
        return [flashcard]

# --------------------------
# Example of multi-agent workflow
# --------------------------
def flowpilot_demo(query: str, topic: str):
    session = Session()
    memory = MemoryBank()

    search_agent = SearchAgent()
    llm_agent = LLMExplanationAgent()
    flashcard_agent = FlashcardAgent()

    resources = search_agent.fetch_resources(query)
    explanation = llm_agent.explain_content(resources[0]['title'])
    flashcards = flashcard_agent.generate_flashcards(explanation, topic)

    session.add_to_session({"query": query, "resources": resources, "flashcards": flashcards})
    for card in flashcards:
        memory.store_flashcard(card)

    return {"session_id": session.session_id, "resources": resources, "flashcards": flashcards}

# --------------------------
# Run Demo
# --------------------------
demo_result = flowpilot_demo("Artificial Intelligence Basics", "AI Foundation")
demo_result


# ==========================
# CELL 9 — Multi-Agent Demo with Flashcards & Gemini Sub-Agent
# ==========================

# -----------------------------
# Mock Session Service
# -----------------------------
class SessionService:
    def __init__(self):
        self.memory = {}
    
    def create_session(self, user_id):
        session_id = f"session_{user_id}_{len(self.memory)+1}"
        self.memory[session_id] = {}
        return session_id
    
    def update_memory(self, session_id, key, value):
        if session_id in self.memory:
            self.memory[session_id][key] = value
    
    def get_memory(self, session_id):
        return self.memory.get(session_id, {})

# Initialize session service
session_service = SessionService()
session_id = session_service.create_session("tech_warrior")

# -----------------------------
# Define Agents
# -----------------------------
class LLM_Agent:
    def summarize(self, content):
        return f"Summary: {content[:50]}..."  # Mock summary

class SearchAgent:
    def search(self, query):
        return [{"title": f"Result {i+1} for {query}", 
                 "url": f"http://example.com/{i+1}", 
                 "summary": f"Summary {i+1}"} for i in range(3)]

class FlashCardAgent:
    def generate_flashcards(self, content):
        return [{"question": f"What is {word}?", "answer": word, "topic": "AI Basics"} 
                for word in content.split()[:5]]

# -----------------------------
# Mock Gemini Sub-Agent
# -----------------------------
class GeminiSubAgent:
    def enhance_flashcards(self, flashcards):
        # Adds extra hints or examples
        for card in flashcards:
            card['hint'] = f"Hint: Think about {card['answer']} in AI context"
        return flashcards

# -----------------------------
# Run Multi-Agent Demo
# -----------------------------
user_query = "Explain AI Agents and their workflow"

# Initialize agents
llm_agent = LLM_Agent()
search_agent = SearchAgent()
flashcard_agent = FlashCardAgent()
gemini_agent = GeminiSubAgent()  # Gemini sub-agent

# Run agents in parallel (simulated)
search_results = search_agent.search(user_query)
summary = llm_agent.summarize(user_query)
flashcards = flashcard_agent.generate_flashcards(user_query)

# Enhance flashcards using Gemini sub-agent
flashcards = gemini_agent.enhance_flashcards(flashcards)

# Store results in session memory
session_service.update_memory(session_id, "search_results", search_results)
session_service.update_memory(session_id, "summary", summary)
session_service.update_memory(session_id, "flashcards", flashcards)

# -----------------------------
# Display Results
# -----------------------------
print("=== Search Results ===")
for r in search_results:
    print(f"{r['title']} - {r['url']}")

print("\n=== LLM Summary ===")
print(summary)

print("\n=== Flashcards Generated (with Gemini hints) ===")
for idx, f in enumerate(flashcards, start=1):
    print(f"{idx}. Q: {f['question']} | A: {f['answer']} | Topic: {f['topic']} | Hint: {f['hint']}")

# -----------------------------
# Display Session Memory
# -----------------------------
print("\n=== Session Memory Snapshot ===")
print(session_service.get_memory(session_id))


# ===== Example Input Query =====
user_query = "Explain context engineering in AI agents and give a quiz question."

# ===== LLM Agent: Summarization & Explanation =====
def llm_agent(query):
    # Example output (replace with actual LLM integration)
    explanation = "Context engineering is the process of dynamically managing information within an agent's context to provide stateful and personalized experiences."
    return explanation

# ===== Flashcard Agent: Quiz Generation =====
def flashcard_agent(topic):
    # Example flashcard
    question = "What is context engineering in AI agents?"
    answer = "Managing information dynamically within an agent's context for stateful and personalized interactions."
    return {"question": question, "answer": answer}

# ===== Content Search Agent (Simulated) =====
def content_search_agent(topic):
    resources = [
        {"title": "Context Engineering Whitepaper", "summary": "Details best practices for sessions & memory in AI agents.", "url": "https://example.com/whitepaper"}
    ]
    return resources

# ===== Running Agents =====
explanation = llm_agent(user_query)
flashcard = flashcard_agent("Context Engineering")
resources = content_search_agent("Context Engineering")

# ===== Display Output =====
print("=== Explanation ===")
print(explanation, "\n")

print("=== Flashcard ===")
print(f"Q: {flashcard['question']}")
print(f"A: {flashcard['answer']}\n")

print("=== Resources ===")
for res in resources:
    print(f"Title: {res['title']}")
    print(f"Summary: {res['summary']}")
    print(f"URL: {res['url']}\n")


# CELL 15: Observability & Logging Setup
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_agent_action(agent_name, action, status="Success"):
    logging.info(f"{agent_name} performed action: {action} | Status: {status}")

# Example usage
log_agent_action("LLM Agent", "Explained context engineering")
log_agent_action("Flashcard Agent", "Generated 10 flashcards")
log_agent_action("Content Search Agent", "Fetched 5 relevant resources")


# CELL 16: Session & Memory Example
import uuid

# Memory store to persist session data
memory_bank = {}

def create_session(user_name):
    session_id = str(uuid.uuid4())
    memory_bank[session_id] = {
        "user": user_name,
        "queries": [],
        "flashcards": [],
        "resources": []
    }
    return session_id

def log_query(session_id, query, flashcards=[], resources=[]):
    if session_id in memory_bank:
        memory_bank[session_id]["queries"].append(query)
        memory_bank[session_id]["flashcards"].extend(flashcards)
        memory_bank[session_id]["resources"].extend(resources)

# Example usage
user_session = create_session("Naresh Kumar")
log_query(user_session, "Explain context engineering",
          flashcards=["What is context engineering?", "Why is it important?"],
          resources=["https://link-to-article.com"])

# Display memory for this session
memory_bank[user_session]


# -----------------------------
# MULTI-AGENT SYSTEM DEMO
# -----------------------------

# Import libraries
import random
from adk.session import InMemorySessionService  # Import session service

# -----------------------------
# Initialize Session Service
# -----------------------------
session_service = InMemorySessionService()
session_id = session_service.create_session(user_id="user_001")  # Create a unique session for user

# -----------------------------
# Define Agents
# -----------------------------
# Simple LLM Agent (mock)
class LLM_Agent:
    def summarize(self, content):
        return f"Summary: {content[:50]}..."  # Mock summary

# Content Search Agent (mock)
class SearchAgent:
    def search(self, query):
        # Simulated search results
        return [{"title": f"Result {i+1} for {query}", 
                 "url": f"http://example.com/{i+1}", 
                 "summary": f"Summary {i+1}"} for i in range(3)]

# Flash Card Agent
class FlashCardAgent:
    def generate_flashcards(self, content):
        # Generate simple Q&A flashcards
        return [{"question": f"What is {word}?", "answer": word} for word in content.split()[:5]]

# -----------------------------
# Example User Query
# -----------------------------
user_query = "Explain AI Agents and their workflow"

# -----------------------------
# Initialize Agents
# -----------------------------
llm_agent = LLM_Agent()
search_agent = SearchAgent()
flashcard_agent = FlashCardAgent()

# -----------------------------
# Run Agents in Parallel (Simulated)
# -----------------------------
search_results = search_agent.search(user_query)
summary = llm_agent.summarize(user_query)
flashcards = flashcard_agent.generate_flashcards(user_query)

# -----------------------------
# Store Results in Session Memory
# -----------------------------
session_service.set(session_id, "search_results", search_results)
session_service.set(session_id, "summary", summary)
session_service.set(session_id, "flashcards", flashcards)

# -----------------------------
# Display Output
# -----------------------------
print("=== Search Results ===")
for r in search_results:
    print(f"{r['title']} - {r['url']}")

print("\n=== LLM Summary ===")
print(summary)

print("\n=== Flashcards Generated ===")
for f in flashcards:
    print(f"Q: {f['question']} | A: {f['answer']}")


# -----------------------------
# OBSERVABILITY, EVALUATION & CONTEXT ENGINEERING
# -----------------------------

# Mock Observability
logs = []
def log_action(action):
    logs.append(f"{datetime.now()} - {action}")

# Log agent actions
log_action("LLM summary generated")
log_action("Search results retrieved")
log_action("Flashcards generated")

# Context Engineering (compact memory)
def compact_memory(session_id):
    mem = session_service.get_session(session_id)["memory"]
    # Only keep necessary keys for efficiency
    compacted = {k: mem[k] for k in ["summary", "flashcards"]}
    return compacted

compacted_memory = compact_memory(session_id)
print("Compacted Memory:", compacted_memory)

# Evaluation Metrics (mock)
def evaluate_agent():
    summary_score = random.randint(8, 10)  # out of 10
    flashcard_score = random.randint(8, 10)
    return {"summary_score": summary_score, "flashcard_score": flashcard_score}

evaluation = evaluate_agent()
print("Agent Evaluation Scores:", evaluation)


# -----------------------------
# DEPLOYMENT READY MOCK
# -----------------------------
deployment_info = {
    "agent_name": "TaskMaster AI",
    "version": "1.0",
    "session_id": session_id,
    "features": ["LLM", "Search", "Flashcards", "Session & Memory", "Observability", "Evaluation"]
}

print("Deployment Ready Info:")
for k, v in deployment_info.items():
    print(f"{k}: {v}")

# Export notebook as a Kaggle submission or for Vertex AI deployment
# (Actual deployment requires Vertex AI Agent Engine and API key setup)


# -----------------------------
# GEMINI AGENT DEMO 
# -----------------------------
# User query to show effective use of Gemini Agent
user_query_2 = "How can I improve my study productivity?"
gemini_response = gemini_agent.answer(user_query_2)

# Display the response
print("=== Gemini Agent Bonus Demo ===")
print(f"Query: {user_query_2}")
print(f"Answer: {gemini_response}")

