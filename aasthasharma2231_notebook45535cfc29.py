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


"""
Concierge Multi-Agent System with Travel Concierge Agent

"""

# ===============================
# IMPORTS
# ===============================
import time
import logging
import random
from typing import Dict, Any, List, Callable
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===============================
# SESSION & MEMORY
# ===============================

class InMemorySessionService:
    def __init__(self):
        self.sessions = {}  # Initialize sessions dictionary

    def get(self, session_id):
        return self.sessions.get(session_id, {})

    def update(self, session_id, key, value):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id][key] = value


class MemoryBank:
    def __init__(self):
        self.memories = []

    def save(self, memory: str):
        logging.info(f"[MEMORY] Saved: {memory}")
        self.memories.append(memory)

    def recall(self, query: str):
        return [m for m in self.memories if query.lower() in m.lower()]


session_service = InMemorySessionService()
memory_bank = MemoryBank()

# ===============================
# TOOLS
# ===============================

def google_search_tool(query: str) -> str:
    logging.info(f"[Google Search] Query: {query}")
    return f"Google Search Result for '{query}'"

def code_execution_tool(code: str):
    logging.info("[Code Execution] Running code...")
    try:
        exec_locals = {}
        exec(code, {}, exec_locals)
        return exec_locals
    except Exception as e:
        return str(e)

def custom_weather_tool(city: str):
    return f"Weather in {city}: {random.choice(['Sunny', 'Rainy', 'Cloudy'])}"

def openapi_tool(endpoint: str):
    return f"(Mocked) OpenAPI response from {endpoint}"

def mcp_tool(action: str):
    return f"MCP executed action: {action}"

# ===============================
# BASE AGENT CLASS
# ===============================

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def llm_response(self, prompt: str) -> str:
        """Mock LLM Response"""
        logging.info(f"[LLM:{self.name}] Prompt: {prompt}")
        return f"{self.name} says: {prompt}"

    def run(self, *args, **kwargs):
        raise NotImplementedError

# ===============================
# AGENTS
# ===============================

class GreetingAgent(BaseAgent):
    def run(self, user_msg: str):
        memory_bank.save(f"User said: {user_msg}")
        return self.llm_response("Hello! How can I help you today?")

class RestaurantRecommendationAgent(BaseAgent):
    def run(self, city: str):
        result = google_search_tool(f"Best restaurants in {city}")
        return self.llm_response(result)

class TravelConciergeAgent(BaseAgent):
    """
    Travel Planning Agent – Competition-ready
    """

    def run(self, destination: str, days: int, budget: str = "medium", preferences: List[str] = None):
        if preferences is None:
            preferences = ["sightseeing", "food", "shopping"]

        weather = custom_weather_tool(destination)
        hotels = google_search_tool(f"best hotels in {destination} for {budget} budget")
        flights = google_search_tool(f"cheap flights to {destination}")

        # Generate day-by-day itinerary
        itinerary = []
        for day in range(1, days + 1):
            activities = random.sample(preferences, min(len(preferences), 2))
            itinerary.append({
                "day": day,
                "activities": activities,
                "suggested_places": [f"{destination} {activity} spot" for activity in activities]
            })

        # Summarize via LLM mock
        summary_prompt = (
            f"Create a friendly travel guide for {destination} for {days} days.\n"
            f"Include: weather: {weather}, hotels: {hotels}, flights: {flights}, "
            f"daily itinerary with activities and local attractions, budget: {budget}\n"
        )
        itinerary_summary = self.llm_response(summary_prompt)

        return {
            "destination": destination,
            "days": days,
            "budget": budget,
            "weather": weather,
            "hotels": hotels,
            "flights": flights,
            "itinerary": itinerary,
            "summary": itinerary_summary
        }

class ParallelAgent(BaseAgent):
    """
    Runs multiple agents simultaneously
    """
    def run(self, agents: List[Callable]):
        logging.info("[ParallelAgent] Running tasks in parallel...")
        return [fn() for fn in agents]

class SequentialAgent(BaseAgent):
    """
    Runs tasks one after another
    """
    def run(self, agents: List[Callable]):
        logging.info("[SequentialAgent] Running tasks sequentially...")
        output = []
        for fn in agents:
            output.append(fn())
        return output

class LoopAgent(BaseAgent):
    """
    Retry logic for failed operations
    """
    def run(self, action: Callable, retries: int = 3):
        for i in range(retries):
            logging.info(f"[LoopAgent] Attempt {i+1}")
            try:
                return action()
            except:
                time.sleep(1)
        return "Action failed after retries"

# ===============================
# ORCHESTRATOR
# ===============================

class ConciergeOrchestrator:
    def __init__(self):
        self.agents = {
            "greet": GreetingAgent("Greeter"),
            "restaurants": RestaurantRecommendationAgent("FoodBot"),
            "travel": TravelConciergeAgent("TravelBot"),
        }

    def handle(self, user_msg: str, session_id="default"):

        # Save message to session
        session_service.update(session_id, "last_message", user_msg)

        # Travel request
        if "travel" in user_msg.lower():
            return self.agents["travel"].run(
                destination="Goa",
                days=4,
                budget="high",
                preferences=["beach", "waterfalls", "spa", "seafood"]
            )
        
        # Restaurant request
        if "food" in user_msg.lower() or "restaurant" in user_msg.lower():
            return self.agents["restaurants"].run("Delhi")

        # Default greeting
        return self.agents["greet"].run(user_msg)

# ===============================
# DEMO RUN
# ===============================

if __name__ == "__main__":
    system = ConciergeOrchestrator()

    print("\n=== USER: I want travel help ===")
    travel_plan = system.handle("I want travel help")
    print(json.dumps(travel_plan, indent=2))

    print("\n=== USER: Suggest restaurants ===")
    print(system.handle("Suggest restaurants"))

    print("\n=== USER: Hello ===")
    print(system.handle("Hello"))


