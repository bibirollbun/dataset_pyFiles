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


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


import logging

logging.basicConfig(
    filename="logger.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info("ğŸš€ Meal Planner system started")


import uuid
from google.genai import types

import os
from datetime import date

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


session_memory = InMemorySessionService()
session = session_memory.create_session_sync(
    app_name="MealPlannerApp",
    user_id="user123"
)

session_memory.current_session = session
session_id = session.id
today = date.today().isoformat()


meal_planning = Agent(
    name="MealAssistant",
    model=Gemini(model="gemini-2.5-flash-lite"),
    description=f"""
You are a Meal Planning AI Agent.

Your work happens in two steps:

----------------------------------------------------
PART 1 â€” Understand the User
----------------------------------------------------
From the user's latest message, extract their meal preferences.  
Identify the following (use defaults when not stated):

- Diet type  
- Allergies or dietary restrictions  
- Disliked ingredients  
- Preferred number of meals per day (default: 3)  
- Maximum cooking time (default: 20 minutes)

Store this data internally in session memory as user_profile.  
Do NOT display or print the profile in your response.

----------------------------------------------------
PART 2 â€” Generate a 7-Day Healthy Meal Plan
----------------------------------------------------
Using the internal user_profile:

- Create a 7-day structured meal plan  
- Include the correct number of meals per day  
- Avoid allergens and disliked foods  
- Use healthy cooking methods (steam, sautÃ©, boil, bake, air-fry)  
- Keep meals simple and within the user's cook_time  
- Add variety and avoid repetition by using today's date: {today}

Save the final plan to session memory as 'meal_plan'.

Your response to the user must include **only** the natural language meal plan.
Do NOT print JSON, memory keys, or internal data structures.
"""
)

print("âœ… Meal Planning Agent Ready")




recipe_agent = Agent(
    name="RecipeAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    description="Generates healthy recipes for each meal.",
    instruction="""
    Read 'meal_plan' from memory.

    For EACH meal:
      - Generate a healthy recipe
      - Avoid allergens & disliked items
      - Use healthy cooking methods only
      - Provide ingredients + step-by-step instructions + cooking time

    Save results in memory as 'recipes'.
    """,
    tools=[google_search],
    output_key="recipes",
)

print("âœ… Recipe Agent Ready")




grocery_agent = Agent(
    name="GroceryAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    description="Creates categorized grocery list",
    instruction="""
    Read 'recipes' from memory.

    Extract ALL ingredients.
    Group into categories:
      - Vegetables
      - Fruits
      - Grains & Cereals
      - Protein Sources
      - Spices & Condiments
      - Pantry Items
      - Others

    Save as 'grocery_list'.
    """,
    output_key="grocery_list",
)

print("âœ… Grocery Agent Ready")




root_agent = SequentialAgent(
    name="MealPlanPipeline",
    sub_agents=[meal_planning, recipe_agent, grocery_agent],
)

print("âœ… Sequential Pipeline Ready")



runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
    "I'm vegetarian, allergic to oats, dislike mushrooms. "
    "I want 2 meals per day under 15 minutes. "
    "Create my weekly meal plan with recipes and grocery list."
)

logging.info("Pipeline completed. Response delivered to user.")

print("ğŸ”� Log saved to logger.log")


!cat logger.log

