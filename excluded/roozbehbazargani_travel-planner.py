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


from typing import Any, Dict

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.tools.google_search_tool import google_search
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools import AgentTool

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


place_search = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="place_search_agent",
    description="An agent that suggests the places to visit in a city",
    instruction="""
    You are a simple places to visit search agent.
    Your only tool is google_search.
    
    Given:
    - a city
    - optional date range
    - optional number of days
    - optional preferences for the activity (museum, historica sites, nightlife, festivals, food, etc)
    
    Do:
    If not given the number of days or date range, find when is the best time to travel to the city and how many days are needed to attend to those places.
    Use google_search to find top attractions of the city while trying to add prefenrences more but try to add highly recommended activities, even if they are not in the preferences.
    Make sure the places you are recommending are open in the selected day range and the activities should be enough for number of days the user is staying.

    Return the number of days, date range, and list of places with their category
    """,
    tools=[google_search]
)


route_planner = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="route_planner_agent",
    description="An agent that plans the route to reduce backtracking and checks opening hours.",
    instruction="""
    You are a route planning agent.
    
    You receive:
    - a list of places
    - number of days (optional)
    - arrival/departure dates (optional)
    - user preferences (optional)
    
    Your job is to Create a day-by-day plan that fits in number of days given (feel free to drop some of the places if there's not enough time):
    - minimize backtracking
    - group by geographic clusters
    - keep each day balanced (3â€“5 hours morning, 3â€“5 hours afternoon, optional for night specially if an activity is better at night)
    Keep mind of the followings:
    1. If days not provided:
       Estimate how many days are needed to do all the suggested activities (if dates are provided suggest the number of days in the requested date)
    2. Make sure the places are open when you are planning and if they need tickets, the tickets are available for that time and date
    
    Return opening JSON like:
    {
      "days_needed": 3,
      "itinerary": [
         {
           "day": 1,
           "places": [
             {"name": "Sagrada Familia", "start": "09:00", "duration": "2h", "Opening hours": "..."},
             {"name": "Recinte Modernista", "start": "12:00", "duration": "1.5h", "Opening hours": "..."}
           ]
         }
      ],
      "reasoning": "..."
    }
    """,
    tools=[google_search]
)


evaluator = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="evaluator_agent",
    description="An agent that evaluates the recommended travel plan",
    instruction="""
    You are a travel evaluator. You get a visit plan based on number of days and opening hours and eveluate if it is a valid plan and the locations are open, next to each other, and the plan almost satisfies the best route planning without unnecessary backtracking.
    If you approve the plan, ONLY return APPROVED, else, return your concern and issue that you have found so the plan get updated.
    """,
)


import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class JsonMemory:
    """
    Minimal memory store with structure:

    {
        "user_123": {
            "notes": [
                {
                    "time": "2025-11-27T12:34:56.123456",
                    "text": "The user likes visiting tourist sites."
                }
            ]
        }
    }
    """

    def __init__(self, path: str = "memory.json"):
        self.path = Path(path)
        self._data = self._load()

    # ----------------- Internal -----------------
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save(self):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _ensure_user(self, user_id: str):
        if user_id not in self._data:
            self._data[user_id] = {"notes": []}
        return self._data[user_id]

    # ----------------- Public API -----------------

    def add_note(self, user_id: str, text: str):
        """Append a note for this user."""
        user_mem = self._ensure_user(user_id)
        user_mem["notes"].append({
            "time": datetime.utcnow().isoformat(),
            "text": text
        })
        self._save()

    def get_notes(self, user_id: str) -> List[Dict[str, str]]:
        """Return raw notes list for prompting/reading."""
        return self._data.get(user_id, {}).get("notes", [])


# 1) Your JsonMemory instance
memory_store = JsonMemory("memory.json")

session_service = InMemorySessionService()

# 2) Tool factory that CLOSES OVER user_id
def make_memory_tools_for_user(user_id: str):
    def save_memory(text: str):
        """Save a memory note for the current user."""
        memory_store.add_note(user_id, text)
        return "ok"

    def retrieve_memory():
        """Retrieve memory notes for the current user."""
        return memory_store.get_notes(user_id)

    save_memory_tool = FunctionTool(
        func=save_memory
    )

    retrieve_memory_tool = FunctionTool(
        func=retrieve_memory
    )

    return save_memory_tool, retrieve_memory_tool


def make_runner_for_user(user_id: str) -> Runner:
    save_memory_tool, retrieve_memory_tool = make_memory_tools_for_user(user_id)

    orchestrator = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite"),
        name="Orchestrator",
        instruction="""
        You are the orchestrator agent for planning a trip with days and timelines.

        You have a few tools:
        - retrieve_memory()
        - save_memory(text)
        - place_search agent that provides a list of places to visit in a city after searching through Google
        - route_planner agent to plan the trip and provide opening hours
        - evaluator agent that approves the plan of route_planner

        Use retrieve_memory() before answering the user to see what you know about them (e.g., name, long-term preferences, travel plans).
        After answering the user, use save_memory(text) to store any important, stable facts
        about the user (e.g., name, long-term preferences, travel plans).
        Do not store trivial or fleeting details.

        You should try your best to handle opening hours and ticket availability and ask the user to check if it's really needed.
        You're output should contain opening hours and be in the following structure containing the place and hour of visit and for how long. It's important to have the plan in hours so user only follows that.

        day 1:
        Queen Elizabeth Park: 9:30-11:30
        VanDusen Botanical Garden: 12:30-14:00
        Chinatown: 16:00-17:30
        Dr. Sun Yat-Sen Classical Chinese Garden: 17:45-18-45
        day 2:
        ...
        
        Feel free to add extra notes for a better comminucation to the user.
        The notes should be consistent with user's prompt, for example, do not say the plan is approved by the evaluator agent. User does not need to know the process underneath for planning.
        

        I would take the following steps:
        Step 1: send the preference of the user, destination, and number of days to the place_search agent to suggest places
        Step 2: use route planner to suggest you the plan based on the suggested places of the place_search agent and also get the opening hours
        Step 3: Check if evaluator agent approves the plan based on opening hours, user preferences, destination, and number of days. If it does, output the plan to the user.
        If not, communicate the evaluator feedback to the place_search agent or route_planner agent to consider them in order to get approval.
        If evaluator does not accept the plan after 3 feedback, just use the latest suggested plan to make an output for the user.

        Example of a good response format for a 2-day trip to Paris:
        **Day 1: Icons and River Views**

        *   **Eiffel Tower:** 9:30 AM - 11:30 AM (Book tickets in advance for a specific time slot)
        *   **Champ de Mars Park:** 11:30 AM - 1:00 PM
        *   **Champs-Ã‰lysÃ©es:** 2:00 PM - 3:30 PM
        *   **Arc de Triomphe:** 3:30 PM - 5:00 PM
        *   **Seine River Cruise:** 6:00 PM - 7:00 PM
        
        **Day 2: Art, Gardens, and Notre Dame Area**
        
        *   **Louvre Museum:** 9:00 AM - 12:00 PM (Book timed-entry tickets in advance, closed Tuesdays)
        *   **Jardin des Tuileries:** 12:00 PM - 1:00 PM
        *   **Notre Dame Cathedral (Exterior) & Ã�le de la CitÃ©:** 2:00 PM - 3:30 PM
        *   **Luxembourg Gardens:** 3:30 PM - 5:00 PM
        
        **Important Notes:**
        
        *   **Bookings:** It is highly recommended to book tickets for the Eiffel Tower and the Louvre Museum online in advance to save time and guarantee entry.
        *   **Transportation:** Paris has an excellent public transportation system (Metro). Consider purchasing a pass for easy travel between locations.
        *   **Walking:** Be prepared for a good amount of walking, so comfortable shoes are a must.
        *   **Flexibility:** This is a suggested itinerary. Feel free to adjust timings based on your family's energy levels and interests.
        *   **Opening Hours:** Please double-check the official websites for the most up-to-date opening hours, especially for Notre Dame Cathedral, which is currently undergoing reconstruction.
        
        Enjoy your family trip to Paris!
        """,
        tools=[save_memory_tool,
               retrieve_memory_tool,
               AgentTool(agent=place_search),
              AgentTool(agent=route_planner),
              AgentTool(agent=evaluator)],
    )

    runner = Runner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
    )
    return runner



APP_NAME = "default"  # Application
USER_ID = "default"  # User

MODEL_NAME = "gemini-2.5-flash-lite"

async def run_session(
    user_queries: list[str] | str, user_id: str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    runner = make_runner_for_user(user_id)

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("âœ… Orchestrator agent created.")


await run_session(
    [
    "My name is Roozbeh. I want to go to Paris with Family for 2 days, I want to visit some of the famous places of the city during the stay. Please provide a travel plan",
    ],
    user_id = "user-1",
    session_id = "session-1-0",
)


await run_session(
    [
    "Do you remember my name? Could you plan a trip with family to Rome?",
    ],
    user_id = "user-1",
    session_id = "session-1-1",
)


await run_session(
    [
    "Let's plan for 2 days in June, we like to visit historical sites in Rome in a fast-paced as we only have 2 days.",
    ],
    user_id = "user-1",
    session_id = "session-1-1",
)


await run_session(
    [
    "Yes, please.",
    ],
    user_id = "user-1",
    session_id = "session-1-1",
)


memory_store._load()




