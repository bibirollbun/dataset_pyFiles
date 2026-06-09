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
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… Setup and authentication complete.")



import uuid
import json
import warnings
from typing import List, Dict

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

warnings.filterwarnings("ignore")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

print("âœ… ADK imports ready.")



# Very simple mock tools for an urban help use case.
# In a real project, these could call real APIs (maps, transport, etc.).

CITY_SERVICES = {
    "hospital": ["City General Hospital - 24/7 emergency", "Metro Care Clinic - 9 AM to 9 PM"],
    "police": ["Central Police Station - emergency 100", "Neighborhood Police Outpost"],
    "fire": ["City Fire Station - emergency 101"],
}

TRANSPORT_OPTIONS = {
    "airport": "Metro Line 2 to Airport Station, or CityCab (~45 minutes in normal traffic).",
    "railway station": "Bus 12A / 12B or Metro Line 1 to Central Station.",
    "college": "Shared autos near main market; buses 5, 8, 12 from bus stand.",
}

def find_city_service(service_type: str) -> str:
    """
    Simple tool: given a service type (hospital, police, fire, etc.)
    returns nearby options from a mock database.
    """
    key = service_type.lower().strip()
    options = CITY_SERVICES.get(key)
    if not options:
        available = ", ".join(sorted(CITY_SERVICES.keys()))
        return (
            f"Sorry, I don't have data for '{service_type}'. "
            f"Available service types: {available}."
        )
    return f"Nearby {service_type} options:\n- " + "\n- ".join(options)

def get_transport_help(destination: str) -> str:
    """
    Simple tool: given a destination type, returns basic transport suggestions.
    """
    key = destination.lower().strip()
    info = TRANSPORT_OPTIONS.get(key)
    if not info:
        available = ", ".join(sorted(TRANSPORT_OPTIONS.keys()))
        return (
            f"Sorry, I don't have specific routes to '{destination}'. "
            f"Known destinations: {available}."
        )
    return f"Transport help for {destination}:\n{info}"

print("âœ… Tools defined: find_city_service, get_transport_help")



# FIXED Planner agent: better at recognizing hospital/police as "service"
planner_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="urban_planner_agent",
    description="Understands the user's urban help query and decides what information is needed.",
    instruction="""
You are an urban help planner for a city concierge system.

Classify user queries into these categories ONLY:
- "service" for hospitals, police, fire, clinics, emergency services
- "transport" for airport, railway station, college, bus routes, travel
- "general" for safety, areas, night travel, tips

ALWAYS respond with VALID JSON only (no extra text):

{
  "category": "service" | "transport" | "general",
  "targets": ["hospital", "police", "airport", "college"],
  "notes": "brief notes"
}

EXAMPLES:
User: "find hospital" â†’ {"category": "service", "targets": ["hospital"], "notes": "emergency medical"}
User: "go to airport" â†’ {"category": "transport", "targets": ["airport"], "notes": "public transport"}
User: "safe at night" â†’ {"category": "general", "targets": [], "notes": "safety advice"}
""",
)

print("âœ… FIXED Planner agent created - better hospital detection!")



# Main concierge agent â€“ talks to the user, calls tools, uses the planner.

urban_concierge_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="urban_concierge_agent",
    description="City concierge that helps with services, transport, and basic city guidance.",
    instruction="""
You are the main Urban Help Concierge Agent for city residents and visitors.

Behavior:
- Talk to the user in a friendly, concise way.
- When you need structured understanding of the user's goal,
  call the 'urban_planner_agent' sub-agent.
- When you need real data about services or transport, use the tools:
  - find_city_service(service_type: str)
  - get_transport_help(destination: str)

Flow:
1. Use the planner sub-agent to classify the request into category and targets.
2. Based on the planner output:
   - If category == "service": call find_city_service for each target, summarize results.
   - If category == "transport": call get_transport_help for each target, summarize.
   - If category == "general": answer directly using your own knowledge in a short way.
3. Always explain answers in natural language, but stay concise.

Be honest about limitations: the tools are using a simple mock database.
""",
    sub_agents=[planner_agent],
    tools=[find_city_service, get_transport_help],
)

print("âœ… Urban concierge agent created with:")
print("   - 1 sub-agent (planner)")
print("   - 2 tools (services + transport)")



session_service = InMemorySessionService()
APP_NAME = "urban_help_concierge_app"

print("âœ… InMemorySessionService initialized.")

async def run_concierge_chat(user_query: str):
    """
    Runs a single interaction with the Urban Concierge Agent using sessions.
    """
    user_id = "demo_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # Create session first (this matches ADK best practices).
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=urban_concierge_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(parts=[types.Part(text=user_query)])

    print(f"\nðŸ‘¤ User: {user_query}\n")
    print("ðŸ¤– Urban Concierge:\n" + "-" * 50)

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)
                else:
                    print("[No text response received]")


    print("-" * 50)

print("âœ… Chat runner ready.")



# Example 1: asking for emergency services
await run_concierge_chat(
    "My friend is injured near the city center. Can you find a nearby hospital?"
)

# Example 2: asking for transport help
await run_concierge_chat(
    "How do I go from my hostel to the airport using public transport?"
)

# Example 3: general city question
await run_concierge_chat(
    "Is it safe to travel alone at night in the city, and which areas are crowded?"
)


