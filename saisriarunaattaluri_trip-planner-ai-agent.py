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


import os
import json
import time
from typing import List, Dict, Any, Callable
import google.generativeai as genai
from dataclasses import dataclass, field

# --- CONFIGURATION ---
# Replace with your actual Gemini API Key
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- 1. TOOLS & UTILITIES ---

def google_search_tool(query: str) -> str:
    """
    Simulates a Google Search. 
    In a production environment, use Google Custom Search JSON API or SerpApi.
    """
    print(f"\n[Tool Execution] ğŸ”� Searching Google for: {query}")
    
    # MOCK RESPONSES for demonstration (so the code runs without an extra SERP API key)
    # You would replace this logic with actual API calls to Google Search.
    mock_db = {
        "tourist places": "Top places: 1. The Ancient Fort, 2. Crystal Lake, 3. Old Market, 4. Sunset Peak, 5. The Royal Palace, 6. City Museum.",
        "cost": "Flight: $300, Hotel: $100/night, Food: $50/day. Entry fees: $50 total. Local Transport: $40.",
        "hotel": "Best Hotels: 1. Grand Plaza ($120/night), 2. Backpackers Hostel ($40/night), 3. Seaside Resort ($200/night).",
        "food": "Famous food: Spicy Noodles, Dumplings, Local Curry. Best restaurants: Dragon Inn ($30/meal), The Food Court ($15/meal)."
    }
    
    # Return a generic mock if exact keyword not found, otherwise specific mock
    for key, value in mock_db.items():
        if key in query.lower():
            return value
    return f"Search results for {query}: Found generic travel info."

# --- 2. MEMORY SYSTEMS ---

class InMemorySessionService:
    """Short-term memory for the current active planning session."""
    def __init__(self):
        self.session_state: Dict[str, Any] = {}
        self.chat_history: List[Dict] = []

    def update_state(self, key: str, value: Any):
        self.session_state[key] = value

    def get_state(self, key: str):
        return self.session_state.get(key)

    def add_log(self, sender: str, message: str):
        self.chat_history.append({"sender": sender, "message": message})

class LongTermMemory:
    """Persistent storage (Memory Bank) saving to a JSON file."""
    def __init__(self, filepath="agent_memory_bank.json"):
        self.filepath = filepath
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {}

    def save_plan(self, destination: str, plan_details: Dict):
        self.memory[destination] = plan_details
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=4)
        print(f"ğŸ’¾ Plan for {destination} saved to Long Term Memory.")

# --- 3. AGENT FRAMEWORK (The "ADK" Simulation) ---

@dataclass
class Agent:
    name: str
    model_name: str
    instruction: str
    tools: List[Callable]
    
    def __post_init__(self):
        # Initialize the Gemini model with tools
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=self.tools,
            system_instruction=self.instruction
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    def send_message(self, message: str) -> str:
        """Sends a message to the agent and gets a response."""
        print(f"\nğŸ¤– [{self.name}] is thinking...")
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            return f"Error: {e}"

# --- 4. SUB-AGENTS DEFINITIONS ---

def create_explorer_agent() -> Agent:
    return Agent(
        name="Tourist_Explorer",
        model_name="gemini-2.0-flash", # Or gemini-1.5-flash
        instruction="""
        You are an expert Tourist Attraction Finder. 
        1. When given a destination, use google_search_tool to find a minimum of 6 famous tourist places.
        2. List them with brief descriptions.
        3. ALWAYS end your response by asking the user explicitly: "Do you want to proceed with logistics and cost estimation?"
        """,
        tools=[google_search_tool]
    )

def create_cost_agent() -> Agent:
    return Agent(
        name="Cost_Estimator",
        model_name="gemini-2.0-flash",
        instruction="""
        You are a Travel Accountant.
        1. You receive an origin, a destination, a list of tourist places, and hotel/food options.
        2. Use google_search_tool to estimate costs.
        3. You MUST display the costs in this exact format:
           - Total Travel Cost (Flight/Train + Local transport): $X
           - Total Cost of Stay (for a standard 3-day trip): $Y
           - Total Cost of Food (for 3 days): $Z
           - Entry Fees & Extras: $A
           -----------------------------------
           - **FINAL TOTAL TOUR COST**: $SUM
        4. Add all these values to show the final cost of the tour clearly.
        """,
        tools=[google_search_tool]
    )

def create_logistics_agent() -> Agent:
    return Agent(
        name="Logistics_Manager",
        model_name="gemini-2.0-flash",
        instruction="""
        You are a Lifestyle & Hospitality Expert.
        1. Search for top 3 hotels (Budget, Mid-range, Luxury) in the destination with their prices.
        2. Search for famous local food and best restaurants with approximate meal prices.
        3. Provide this list to help the user choose and for the accountant to estimate costs.
        """,
        tools=[google_search_tool]
    )

# --- 5. ORCHESTRATOR & MAIN EXECUTION ---

class TravelOrchestrator:
    def __init__(self):
        self.session = InMemorySessionService()
        self.ltm = LongTermMemory()
        
        # Initialize Agents
        self.explorer = create_explorer_agent()
        self.cost_estimator = create_cost_agent()
        self.logistics = create_logistics_agent()

    def run_plan(self):
        print("ğŸŒ� --- AI TRAVEL PLANNER SYSTEM STARTED --- ğŸŒ�")
        
        # 1. Input Phase
        origin = input("Enter Origin City: ")
        destination = input("Enter Destination City: ")
        
        self.session.update_state("origin", origin)
        self.session.update_state("destination", destination)

        # --- PHASE 1: EXPLORER AGENT ---
        print("\n--- Phase 1: Exploration ---")
        prompt_1 = f"I want to visit {destination}. Find me at least 6 famous tourist places."
        response_1 = self.explorer.send_message(prompt_1)
        print(f"\n{response_1}")
        
        # User Feedback Loop (Interactive)
        user_input = input("\nUser Input (Yes/No to proceed): ")
        
        if "yes" not in user_input.lower():
            print("â�Œ Planning aborted by user.")
            return

        self.session.update_state("attractions", response_1)

        # --- PHASE 2: LOGISTICS AGENT (Swapped to run before Cost) ---
        print("\n--- Phase 2: Stay & Food ---")
        prompt_2 = f"Find hotels and food for {destination}. Include prices."
        response_2 = self.logistics.send_message(prompt_2)
        print(f"\n{response_2}")

        self.session.update_state("logistics", response_2)

        # --- PHASE 3: COST AGENT (Runs last to sum everything) ---
        print("\n--- Phase 3: Cost Estimation ---")
        # A2A Protocol: Passing context from Agent 1 and Agent 2 to Agent 3
        prompt_3 = (
            f"Calculate the final tour cost from {origin} to {destination}. "
            f"Tourist Places: {self.session.get_state('attractions')}. "
            f"Hotels and Food info: {self.session.get_state('logistics')}. "
            "Assume a 3-day trip. Calculate Total Cost of Food, Total Cost of Stay, Travel Cost, and the Final Total."
        )
        response_3 = self.cost_estimator.send_message(prompt_3)
        print(f"\n{response_3}")
        
        self.session.update_state("budget", response_3)

        # --- FINALIZATION ---
        print("\n--- ğŸ’¾ Saving to Long Term Memory ---")
        full_plan = {
            "origin": origin,
            "attractions": self.session.get_state("attractions"),
            "budget": self.session.get_state("budget"),
            "logistics": self.session.get_state("logistics")
        }
        self.ltm.save_plan(destination, full_plan)
        print("âœ… Trip planning complete!")


