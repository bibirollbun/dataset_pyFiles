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

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("âœ… Helper functions defined.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Initialize Memory
memory_service = (
    InMemoryMemoryService()
)

# Define constants used
APP_NAME = "MealPlannerApp"
USER_ID = "demo_user"


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


print("âœ… Callback created.")


# Applies to all agent and model calls
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""

    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    # Callback 1: Runs before an agent is called. You can add any custom logic here.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Count agent runs."""
        self.agent_count += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")

    # Callback 2: Runs before a model is called. You can add any custom logic here.
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")


# Outline Agent: Creates the initial meal plan outline.
meal_outline_agent = Agent(
    name="MealOutlineAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Create a list of unique meals for each day based on the number of meals per day and number of days provided by the user. Take into account:
        - Allergies and dietary restrictions
        - Preferred cuisines
        - Taste preferences
        
    Use load_memory tool to avoild propose previous generated meals.
    
    Output JSON Schema Example:
    [
        {{ "day": "Day 1", "meal": "Breakfast", "meal_name": "Spicy Sweet Potato Hash" }},
        {{ "day": "Day 1", "meal": "Lunch", "meal_name": "Asian Chili Chicken Salad (No Peanuts)" }}
    ]
    """,
    tools=[load_memory],
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
    output_key="meal_plan_outline",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… meal_outline_agent created.")


# Meal Info Agent: Focuses on ingredients list, total calories, carbs, protein and fat info.
meal_info_agent = Agent(
    name="MealInfoAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Using the meal plan from {meal_plan_outline}, generate for each meal:
        - Ingredients list
        - Total calories
        - Carbs (grams)
        - Protein (grams)
        - Fat (grams)

    Output Format Example (JSON list):
    [
      {{
        "meal_name": "Tuna Salad Bowl",
        "ingredients": [
            "3 cans (5oz each) tuna in water, drained",
            "1/2 cup dairy-free mayonnaise",
            "Tomato slices"
        ],
        "total_calories": 300,
        "carbs": 45,
        "protein": 25,
        "fat": 18
      }}
    ]
    """,
    output_key="meal_info",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… meal_info_agent created.")


# Cooking Steps Agent: Focuses on cooking steps and cooking time.
cooking_steps_agent = Agent(
    name="CookingStepsAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Based on {meal_plan_outline}, generate for each meal:
        - Numbered cooking steps
        - Estimated cooking time in minutes

    Output JSON Example:
    [
      {{
        "meal_name": "Tuna Salad Bowl",
        "cooking_steps": [
          "Drain the tuna and set aside.",
          "Whisk together mayonnaise, lemon juice, garlic, salt, and pepper.",
          "Mix together and serve."
        ],
        "cooking_time": "25 min"
      }}
    ]
    """,
    output_key="cooking_steps_info",  # The result will be stored with this key.
)

print("âœ… cooking_steps_agent created.")


# Meal Image Agent: Focuses on finding image for the meal.
meal_image_agent = Agent(
    name="MealImageAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Based on {meal_plan_outline}, find a relevant image for each meal.
    For each meal:
    1. Extract meal_name.
    2. Search on Google using "meal_name".
    3. Pick a safe-for-work image URL from the search results.
    4. Return a JSON list with:
       - day
       - meal
       - meal_name
       - image_url
    
    Output Format:
    [
      {{
        "day": "Day 1",
        "meal": "Breakfast",
        "meal_name": "Spicy Sweet Potato Hash",
        "image_url": "https://..."
      }}
    ]
    """,
    tools=[google_search],
    output_key="meal_images",  # The result will be stored with this key.
)

print("âœ… meal_image_agent created.")


# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Combine the outputs from all previous agents into a single detailed meal plan.

    You are given:
        - {meal_plan_outline}
        - {meal_info}
        - {cooking_steps_info}
        - {meal_images}
    
    Match meals by day and meal_name, and merge all information into one JSON list.
    
    Output Example:
    [
      {{
        "day": "Day 1",
        "meal": "Breakfast",
        "meal_name": "Spicy Sweet Potato Hash",
        "image_url": "...",
        "ingredients": [...],
        "total_calories": 300,
        "carbs": 40,
        "protein": 20,
        "fat": 10,
        "cooking_steps": [...],
        "cooking_time": "25 min"
      }}
    ]
    """,
    output_key="detailed_meal_plan",  # This will be the final output of the entire system.
)

print("âœ… aggregator_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_agent_team = ParallelAgent(
    name="ParallelAgentTeam",
    sub_agents=[meal_info_agent, cooking_steps_agent, meal_image_agent],
)

# This SequentialAgent defines the high-level workflow: run the meal_outline_agent first, then parallel team, finally run the aggregator.
root_agent = SequentialAgent(
    name="MealPlanerSystem",
    sub_agents=[meal_outline_agent, parallel_agent_team, aggregator_agent],
)

print("âœ… Parallel and Sequential Agents created.")


# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name="MealPlannerApp",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
    plugins=[CountInvocationPlugin()], #add plugins for logging
)

print("âœ… Agent and Runner created with memory support!")


await run_session(
    runner,
    "Generate 3 day meal plans including breakfast, lunch, dinner for each day. I have below preference and requirements: Allergies: Milk, Wheat, Peanut; Dietary resitrictions: Fitness, Slim; Preferred cuisines: Asian, American, Mexican; Taste preference: Spicy, Sweet.",
    "conversation-01", # Session ID
)

