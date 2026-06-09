import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import os
import json
import uuid  # Used for generating a unique session ID
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from typing import Dict, Any

# --- Configuration ---
MODEL_NAME = 'gemini-2.5-flash-lite'
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- Tool/Function Definitions ---

def search_flights(destination: str, dates: str) -> str:
    """
    Finds available flights to a specified destination for the given dates. 
    (Mock function for demonstration.)
    """
    if "tokyo" in destination.lower():
        return json.dumps({
            "options": [
                {"id": "F101", "airline": "Japan Air", "price": 1200, "details": "Direct, 14 hours"},
                {"id": "F102", "airline": "Delta", "price": 950, "details": "1 stop in SEA, 18 hours"},
            ]
        })
    return json.dumps({"options": "No direct flights found."})

def search_hotels(destination: str, dates: str, budget: float) -> str:
    """
    Finds and recommends accommodation based on constraints.
    (Mock function for demonstration.)
    """
    if "tokyo" in destination.lower() and budget >= 150:
        return json.dumps({
            "top_3_options": [
                {"id": "H201", "name": "Shibuya Central Inn", "price_per_night": 220, "proximity": "Near subway"},
                {"id": "H202", "name": "Asakusa Temple View", "price_per_night": 180, "proximity": "Near temples"},
            ]
        })
    return json.dumps({"top_3_options": []})

# Convert Python functions to ADK FunctionTools
flight_tool = FunctionTool(func=search_flights)
hotel_tool = FunctionTool(func=search_hotels)

# --- ADK Agent Callback ---
def handle_tool_error(context: Dict[str, Any], error: Exception) -> str:
    """A callback to write a descriptive error message to the output key."""
    # This prevents KeyError in downstream agents by always writing a string to state
    return f"TOOL ERROR: The required tool failed to run. Summary is unavailable. Please provide a placeholder for the itinerary."

print("âœ… Block 1: Imports, Configuration, and Tool Definitions complete.")


# --- ADK Specialist Agent Blueprints (Revised to force output) ---

# 1. Transportation Agent
transportation_agent_blueprint = {
    "name": "TransportationAgent",
    "model": Gemini(model=MODEL_NAME, retry_options=RETRY_CONFIG),
    # FIX: Strict instruction to output ONLY the summary
    "instruction": """You are the expert Transportation Agent. You MUST use the `search_flights` tool.
        Find and summarize the top 2 flight options (price, duration, layovers) for the trip to **{destination}** on **{dates}**. Your entire response MUST be **ONLY** the concise summary report. 
        Do not include any conversational opening or closing phrases.""",
    "tools": [flight_tool],
    "output_key": "transport_report", 
    "on_tool_error_callback": handle_tool_error,
}

# 2. Housing Agent
housing_agent_blueprint = {
    "name": "HousingAgent",
    "model": Gemini(model=MODEL_NAME, retry_options=RETRY_CONFIG),
    # FIX: Strict instruction to output ONLY the summary
    "instruction": """You are the expert Housing Agent. You MUST use the `search_hotels` tool.
        Find and summarize the top 2 hotel options (price, location) for the trip to **{destination}** for a budget of around **${budget}** per night on **{dates}**. Your entire response MUST be 
        **ONLY** the concise summary report. Do not include any conversational opening or closing phrases.""",
    "tools": [hotel_tool],
    "output_key": "housing_report",
    "on_tool_error_callback": handle_tool_error,
}

# 3. Synthesis Agent
synthesis_agent_blueprint = {
    "name": "SynthesisAgent",
    "model": Gemini(model=MODEL_NAME, retry_options=RETRY_CONFIG),
    "instruction": """You are the Central Orchestrator. Synthesize the following reports 
        into a single, cohesive, professional, 5-day itinerary for a trip to **{destination}**
        focused on food and temples. 
        
        * **Transportation Report:** {transport_report}
        * **Housing Report:** {housing_report}
        
        Structure your final output with a title, a brief summary of confirmed bookings, and a detailed 5-day plan.""",
    "output_key": "final_itinerary",
}
print("âœ… Block 2: Agent Blueprints defined with strict instructions.")


# --- Execution Block ---

# 1. Define Initial State and Run Parameters
initial_state = {
    "destination": "Tokyo",
    "dates": "July 15-20",
    "budget": 200.00,
    "user_query": "Plan a 5-day trip to Tokyo in July for two people with a focus on food and temples"
}
user_id = "user-123"
session_id = str(uuid.uuid4())
FINAL_OUTPUT_KEY = "final_itinerary" 

# 2. RE-INITIALIZE AGENTS
transportation_agent = Agent(**transportation_agent_blueprint)
housing_agent = Agent(**housing_agent_blueprint)
synthesis_agent = Agent(**synthesis_agent_blueprint)

# 3. Define the Sequential Orchestrator
atp_orchestrator = SequentialAgent(
    name="ATP_Pipeline",
    sub_agents=[
        transportation_agent,
        housing_agent,
        synthesis_agent,
    ],
)
print("âœ… Agents and Orchestrator successfully re-initialized.")


# 4. Initialize Runner and Create Session
runner = InMemoryRunner(
    agent=atp_orchestrator, 
    app_name="TravelPlannerApp"
)

# Create Session with Initial State
print("Creating session with initial state...")
session = await runner.session_service.create_session(
    app_name=runner.app_name, 
    user_id=user_id, 
    session_id=session_id,
    state=initial_state
)


# 5. Run the Agent
print(f"\n--- ğŸš€ Starting ADK Workflow ---")
event_list = await runner.run_debug( 
    initial_state['user_query'],
    user_id=user_id,
    session_id=session_id
)


# 6. RETRIEVE OUTPUT DIRECTLY FROM SESSION STATE
updated_session = await runner.session_service.get_session(
    app_name=runner.app_name,
    user_id=user_id,
    session_id=session_id
)

final_response_text = updated_session.state.get(FINAL_OUTPUT_KEY)

# Print the result, omitting the print block if no text is found
if final_response_text:
    print("\n" + "="*70)
    print("ğŸ�† FINAL AGENTIC TRAVEL PLANNER (ATP) OUTPUT (via ADK)")
    print("="*70)
    print(final_response_text)
    print("="*70)

