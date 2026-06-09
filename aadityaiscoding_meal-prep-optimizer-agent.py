pip install -q google-adk[a2a]


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


import pandas as pd

df_meals = pd.read_csv("/kaggle/input/meals-data/meals.csv")
df_meals.head()



user_profile = {
  "diet_type": "low_carb",
  "calorie_target": 2000,
  "preferred_cuisines": ["Thai","Japanese"],
  "max_prep_time": 30,
  "budget_per_serving": 10.0
}



# Example of a daily candidate meal-plan structure
candidate_plan = {
    "day": "Monday",
    "budget_usd_per_serving": 10.0,
    "meals": [
        {"slot": "breakfast", "meal_id": 3},
        {"slot": "lunch",     "meal_id": 47},
        {"slot": "dinner",    "meal_id": 82},
        {"slot": "snack1",    "meal_id": 129},
        {"slot": "snack2",    "meal_id": 154}
    ]
}



def sample_daily_plan(df, user_profile):
    # filter by diet
    df_filt = df[df["diet_type"] == user_profile["diet_type"]].copy()

    # very naive: just pick 3 random meals
    chosen = df_filt.sample(3, replace=False).reset_index(drop=True)

    slots = ["breakfast", "lunch", "dinner"]
    meals = []
    for slot, (_, row) in zip(slots, chosen.iterrows()):
        meals.append({
            "slot": slot,
            "meal_id": int(row["meal_id"])
        })

    return {
        "day": "Monday",
        "meals": meals
    }

candidate_plan = sample_daily_plan(df_meals, user_profile)
candidate_plan



from typing import Optional

def get_meal_options(
    diet_type: str,
    max_calories: Optional[int] = None,
    max_cost_per_serving_usd: Optional[float] = None,
    meal_type: Optional[str] = None,
    min_protein_g: Optional[int] = None,
    max_carb_g: Optional[int] = None,
    max_prep_time_min: Optional[int] = None,
    top_k: int = 5
) -> dict:
    """
    Flexible meal lookup based on user inputs.

    Args:
      diet_type (str): e.g. "keto", "high_protein", "bulking", "balanced"
      max_calories (Optional[int]): upper bound calories (or None)
      max_cost_per_serving_usd (Optional[float]): upper bound cost (or None)
      meal_type (Optional[str]): breakfast, lunch, dinner, snack or None
      min_protein_g (Optional[int]): minimum protein grams required
      max_carb_g (Optional[int]): maximum carbs grams allowed
      max_prep_time_min (Optional[int]): max prep time in minutes
      top_k (int): number of suggestions to return

    Returns:
      dict with keys: status, count, meals (list), error_message (if status=="error")
    """
    if top_k < 1:
        return {"status": "error", "count": 0, "meals": [], "error_message": "top_k must be >= 1"}

    df_filtered = df_meals.copy()

    # Core filter: diet_type
    df_filtered = df_filtered[df_filtered["diet_type"] == diet_type]

    # Apply optional constraints if given
    if max_calories is not None:
        df_filtered = df_filtered[df_filtered["calories"] <= max_calories]
    if max_cost_per_serving_usd is not None:
        df_filtered = df_filtered[df_filtered["cost_per_serving_usd"] <= max_cost_per_serving_usd]
    if min_protein_g is not None:
        df_filtered = df_filtered[df_filtered["protein_g"] >= min_protein_g]
    if meal_type:
        df_filtered = df_filtered[df_filtered["meal_type"] == meal_type]
    if max_carb_g is not None:
        df_filtered = df_filtered[df_filtered["carbs_g"] <= max_carb_g]
    if max_prep_time_min is not None:
        df_filtered = df_filtered[df_filtered["prep_time_min"] <= max_prep_time_min]

    if df_filtered.empty:
        return {
            "status": "error",
            "count": 0,
            "meals": [],
            "error_message": "No meals found matching the given constraints."
        }

    # Sort for â€œbetterâ€� matches: high protein, low carbs as a heuristic
    df_sorted = df_filtered.sort_values(
        by=["protein_g", "carbs_g"],
        ascending=[False, True]
    ).head(top_k)

    meals_list = []
    for _, row in df_sorted.iterrows():
        meals_list.append({
            "meal_id": int(row["meal_id"]),
            "name": str(row["name"]),
            "meal_type": str(row["meal_type"]),
            "diet_type": str(row["diet_type"]),
            "cuisine": str(row.get("cuisine", "")),
            "calories": int(row["calories"]),
            "protein_g": int(row["protein_g"]),
            "carbs_g": int(row["carbs_g"]),
            "fat_g": int(row["fat_g"]),
            "cost_per_serving_usd": float(row["cost_per_serving_usd"]),
            "prep_time_min": int(row["prep_time_min"])
        })

    return {
        "status": "success",
        "count": len(meals_list),
        "meals": meals_list
    }



meal_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_catalog_agent",
    description="Agent that suggests meals from a structured catalog based on diet, calories, and budget.",
    instruction="""
    You are a meal planning assistant.
    When the user asks for meal ideas, always call the get_meal_options tool
    with sensible parameters based on their goals (diet_type, calories, cost, and meal_type).
    You respond with clear, concise suggestions.
    If nothing matches, suggest relaxing constraints.
    """,
    tools=[get_meal_options],
)

print("âœ… Meal Catalog Agent created successfully!")



from google.adk.a2a.utils.agent_to_a2a import to_a2a

meal_prep_a2a_app = to_a2a(meal_catalog_agent, port=8001)

print("âœ… A2A app created for meal_catalog_agent on port 8001")



import os

meal_catalog_agent_code = '''
import os
import pandas as pd

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Retry config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Load the meal catalog CSV
df_meals = pd.read_csv("/kaggle/input/meals-data/meals.csv")

from typing import Optional

def get_meal_options(
    diet_type: str,
    max_calories: Optional[int] = None,
    max_cost_per_serving_usd: Optional[float] = None,
    meal_type: Optional[str] = None,
    min_protein_g: Optional[int] = None,
    max_carb_g: Optional[int] = None,
    max_prep_time_min: Optional[int] = None,
    top_k: int = 5
) -> dict:
    """
    Flexible meal lookup based on user inputs.

    Args:
      diet_type (str): e.g. "keto", "high_protein", "bulking", "balanced"
      max_calories (Optional[int]): upper bound calories (or None)
      max_cost_per_serving_usd (Optional[float]): upper bound cost (or None)
      meal_type (Optional[str]): breakfast, lunch, dinner, snack or None
      min_protein_g (Optional[int]): minimum protein grams required
      max_carb_g (Optional[int]): maximum carbs grams allowed
      max_prep_time_min (Optional[int]): max prep time in minutes
      top_k (int): number of suggestions to return

    Returns:
      dict with keys: status, count, meals (list), error_message (if status=="error")
    """
    if top_k < 1:
        return {"status": "error", "count": 0, "meals": [], "error_message": "top_k must be >= 1"}

    df_filtered = df_meals.copy()

    # Core filter: diet_type
    df_filtered = df_filtered[df_filtered["diet_type"] == diet_type]

    # Apply optional constraints if given
    if max_calories is not None:
        df_filtered = df_filtered[df_filtered["calories"] <= max_calories]
    if max_cost_per_serving_usd is not None:
        df_filtered = df_filtered[df_filtered["cost_per_serving_usd"] <= max_cost_per_serving_usd]
    if min_protein_g is not None:
        df_filtered = df_filtered[df_filtered["protein_g"] >= min_protein_g]
    if meal_type:
        df_filtered = df_filtered[df_filtered["meal_type"] == meal_type]
    if max_carb_g is not None:
        df_filtered = df_filtered[df_filtered["carbs_g"] <= max_carb_g]
    if max_prep_time_min is not None:
        df_filtered = df_filtered[df_filtered["prep_time_min"] <= max_prep_time_min]

    if df_filtered.empty:
        return {
            "status": "error",
            "count": 0,
            "meals": [],
            "error_message": "No meals found matching the given constraints."
        }

    # Sort for â€œbetterâ€� matches: high protein, low carbs as a heuristic
    df_sorted = df_filtered.sort_values(
        by=["protein_g", "carbs_g"],
        ascending=[False, True]
    ).head(top_k)

    meals_list = []
    for _, row in df_sorted.iterrows():
        meals_list.append({
            "meal_id": int(row["meal_id"]),
            "name": str(row["name"]),
            "meal_type": str(row["meal_type"]),
            "diet_type": str(row["diet_type"]),
            "cuisine": str(row.get("cuisine", "")),
            "calories": int(row["calories"]),
            "protein_g": int(row["protein_g"]),
            "carbs_g": int(row["carbs_g"]),
            "fat_g": int(row["fat_g"]),
            "cost_per_serving_usd": float(row["cost_per_serving_usd"]),
            "prep_time_min": int(row["prep_time_min"])
        })

    return {
        "status": "success",
        "count": len(meals_list),
        "meals": meals_list
    }


meal_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_catalog_agent",
    description="External meal catalog agent that suggests meals based on diet, calories, and budget.",
    instruction="""
You are a meal planning assistant for an external meal catalog.
When the user asks for meal ideas, use the get_meal_options tool
with sensible parameters based on their goals (diet_type, calories, cost, and meal_type).
Respond with clear, concise suggestions.
If nothing matches, suggest relaxing constraints.
""",
    tools=[get_meal_options],
)

# Expose as A2A app
app = to_a2a(meal_catalog_agent, port=8001)

'''

# Write the meal catalog agent to a temporary file
with open("/tmp/meal_catalog_server.py", "w") as f:
    f.write(meal_catalog_agent_code)

print("ğŸ“� Meal Catalog agent code saved to /tmp/meal_catalog_server.py")



server_process = subprocess.Popen(
    [
        "uvicorn",
        "meal_catalog_server:app",  # module:app
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd="/tmp",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},
)

print("ğŸš€ Starting Meal Catalog Agent server...")
print("   Waiting for server to be ready...")

max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Meal Catalog Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

globals()["meal_catalog_server_process"] = server_process


# Fetch the agent card from the running server
try:
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Meal Prep Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Meal Prep Agent server is running on port 8001.")


# Create a RemoteA2aAgent that connects to our Meal Prep Agent
# This acts as a client-side proxy so other agents can call its skills
remote_meal_prep_agent = RemoteA2aAgent(
    name="meal_prep_agent",
    description="Remote Meal Prep Agent that handles meal retrieval, filtering, and meal plan creation.",
    # Point to the agent-card.json (A2A metadata)
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Meal Prep Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   Other agents can now use the Meal Prep Agent like a local sub-agent!")



nutrition_coach_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_coach_agent",
    description="A high-level nutrition assistant that generates personalized meal plans using the Meal Prep Agent.",
    instruction="""
    You are a friendly, smart Nutrition Coach.

    Your job:
    1. Understand the user's dietary goals (calories, macros, diet type, budget, prep time).
    2. ALWAYS call the meal_prep_agent sub-agent to fetch meals, filter meals, 
       or generate daily/weekly meal plans.
    3. Do NOT invent meals on your own â€” always ask the meal_prep_agent for real data.
    4. After receiving meal options or a plan from the sub-agent, summarize it clearly.
    5. Keep your tone helpful and encouraging.

    Rules:
    - Never hallucinate meals or macros.
    - Always defer to meal_prep_agent for actual meal information.
    - You are the coordinator, not the meal generator.
    """,
    sub_agents=[remote_meal_prep_agent],  # Add the remote Meal Prep Agent as a sub-agent!
)

print("âœ… Nutrition Coach Agent created!")
print("   Model: gemini-2.5-flash-lite")
print("   Sub-agents: 1 (remote Meal Prep Agent)")
print("   Ready to generate personalized meal plans!")


import uuid
import asyncio
# Assuming 'InMemorySessionService', 'Runner', 'nutrition_coach_agent', 
# and 'types' are defined and imported elsewhere.

async def run_one_turn(runner, user_id, session_id, user_query):
    """
    Helper function to send one message and print the agent's response.
    (Slightly modified to not print the user query, as input() does.)
    """
    print(f"\nğŸ¥— Nutrition Coach:")
    print("-" * 60)
    
    test_content = types.Content(parts=[types.Part(text=user_query)])
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id, # Uses the SAME session_id every time
        new_message=test_content,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)
    
    print("-" * 60)


async def test_interactive_flow():
    """
    Tests a live, interactive, multi-turn conversational flow.
    
    This function:
    1. Creates ONE session and ONE runner.
    2. Enters a loop that waits for user input.
    3. Sends the user's message using the SAME session.
    """
    
    # --- 1. Setup Session (Done ONCE) ---
    session_service = InMemorySessionService()
    app_name = "meal_prep_app"
    user_id = "demo_user"
    session_id = f"conv_session_{uuid.uuid4().hex[:8]}" # One session ID

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # --- 2. Setup Runner (Done ONCE) ---
    runner = Runner(
        agent=nutrition_coach_agent, # Assumes this agent has the conversational prompt
        app_name=app_name,
        session_service=session_service,
    )
    
    print("ğŸ§ª Starting INTERACTIVE flow...")
    print(f"Session ID: {session_id}")
    print("Type 'quit' or 'exit' to end the chat.\n")

    # --- 3. Run the conversation turn-by-turn (Live) ---
    while True:
        # Get live input from you
        user_query = input("ğŸ‘¤ You: ")
        
        # Check for exit command
        if user_query.lower() in ["quit", "exit"]:
            print("\nğŸ‘‹ Ending chat. Goodbye!")
            break
            
        # Send the query and get a response
        await run_one_turn(runner, user_id, session_id, user_query)


# uncomment to run the chat_based_workflow
# await test_interactive_flow()

