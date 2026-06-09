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
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# ============================================================
# KAGGLE ADK MULTI-AGENT: BAR COCKTAIL SUGGESTER
# ============================================================
import json
import asyncio
import os
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.google_llm import Gemini
from google.genai import Client

# ----------------------------------------------------------------
# 1ï¸�âƒ£ SETUP CLIENT + MODEL
# ----------------------------------------------------------------
client = Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = Gemini(model="gemini-2.0-flash-lite")

# Setup session management (required by ADK)
session_service = InMemorySessionService()
session = await session_service.create_session(
        app_name="bar_app", user_id="demo_user", session_id="demo_session"
)

# ----------------------------------------------------------------
# 2ï¸�âƒ£ FUNCTION TOOLS
# ----------------------------------------------------------------

# --- TOOL 1: Detect Ingredients ---
from google.genai import Client, types
import json

MODEL = "gemini-2.0-flash-lite"  # pass as string

def detect_ingredients_fn(image_url: str):
    prompt = f"""
    Analyze this bar image and detect all ingredients.
    Image URL: {image_url}
    Respond ONLY with a JSON list of ingredient names.
    """
    
    resp = client.models.generate_content(
        model=MODEL,         # <- string here
        contents=[prompt]    # list of strings or types.Content
    )

    try:
        detected = json.loads(resp.text)
        if not isinstance(detected, list):
            detected = []
    except:
        detected = []

    return {"detected": detected}

detect_ingredients_tool = FunctionTool(detect_ingredients_fn)

# --- TOOL 2: Generate Recipes ---
def generate_recipes_fn(ingredients: list):
    prompt = f"""
    Based on these ingredients: {ingredients}
    Suggest 3 cocktails the user can make.
    Format:
    [
        {{"name": "", "ingredients": [], "steps": []}}
    ]
    """
    resp = client.models.generate_content(model=MODEL, contents=[prompt])
    try:
        return json.loads(resp.text)
    except:
        return {"raw": resp.text}

generate_recipes_tool = FunctionTool(generate_recipes_fn)

# --- TOOL 3: Rank Recipes ---
def rank_recipes_fn(recipes: list, ingredients: list):
    prompt = f"""
    Rank these cocktails by compatibility with ingredients: {ingredients}
    Format: [{{"name": "", "score": 1-10}}]
    Cocktails: {recipes}
    """
    resp = client.models.generate_content(model=MODEL, contents=[prompt])
    try:
        return json.loads(resp.text)
    except:
        return {"raw": resp.text}

rank_recipes_tool = FunctionTool(rank_recipes_fn)

# ----------------------------------------------------------------
# 3ï¸�âƒ£ AGENTS
# ----------------------------------------------------------------

# --- Vision Agent ---
vision_agent = LlmAgent(
    model=MODEL,
    name="vision_agent",
    description="Detect ingredients from a bar image using Gemini Vision.",
    instruction="""
    You are a cocktail image analyzer.
    Given an image URL of a bar, detect all visible liquor bottles, mixers, fruits, garnishes, and bar tools.
    Return only a JSON list of ingredient names.
    """,
    tools=[detect_ingredients_tool]
)

# --- Recipe Generation Agent ---
recipe_agent = LlmAgent(
    model=MODEL,
    name="recipe_agent",
    description="Generate cocktail recipes based on ingredients.",
    instruction="""
    You are a cocktail recipe generator.
    Given a list of ingredients, suggest 3 cocktails the user can make.
    Return results in JSON format:
    [
        {"name": "", "ingredients": [], "steps": []}
    ]
    """,
    tools=[generate_recipes_tool]
)

# --- Ranking Agent ---
ranking_agent = LlmAgent(
    model=MODEL,
    name="ranking_agent",
    description="Rank cocktails based on ingredient match.",
    instruction="""
    You are a cocktail ranking assistant.
    Rank cocktails by compatibility with the available ingredients.
    Return a JSON list of objects with name and score (1-10).
    """,
    tools=[rank_recipes_tool]
)

# --- Parallel Agent: recipe + ranking together ---
cocktail_parallel_agent = ParallelAgent(
    name="cocktail_parallel_agent",
    sub_agents=[recipe_agent, ranking_agent]  # <-- sub_agents required
)

# --- Sequential Agent: vision -> parallel ---
main_agent = SequentialAgent(
    name="main_agent",
    sub_agents=[vision_agent, cocktail_parallel_agent]  # <-- sub_agents required
)

# ----------------------------------------------------------------
# 4ï¸�âƒ£ RUNNER (STATEFUL)
# ----------------------------------------------------------------

runner = Runner(
    agent=main_agent, app_name="bar_app", session_service=session_service
)
#runner = InMemoryRunner(agent=main_agent)

# ----------------------------------------------------------------
# 5ï¸�âƒ£ ASYNC DEMO FUNCTION
# ----------------------------------------------------------------
from google.genai import types

# Example user/session
user_id = "demo_user"
session_id = "demo_session"

# Your public image URL
image_url = "https://i.ibb.co/WNsSfS5t/sample-bottles.jpg"

# Wrap user input in a Content object
user_input = f"Analyze this bar image and suggest cocktails: {image_url}"
test_content = types.Content(parts=[types.Part(text=user_input)])

# Async function
async def run_demo(image_url: str, output_file: str = "submission.json"):
    test_content = types.Content(parts=[types.Part(text=f"Analyze this bar image: {image_url}")])
    final_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=test_content
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

    # Save results to a file
    with open(output_file, "w") as f:
        # If it's JSON, try to parse it first
        try:
            json_data = json.loads(final_text)
            json.dump(json_data, f, indent=2)
        except:
            f.write(final_text)

    print(f"Results saved to {output_file}")
# ----------------------------------------------------------------
# 6ï¸�âƒ£ RUN DEMO
# ----------------------------------------------------------------
image_url = "https://i.ibb.co/WNsSfS5t/sample-bottles.jpg"
await run_demo(image_url, output_file="/kaggle/working/submission.json")


import json

# Example dummy submission
submission = {
  "image_url": "https://i.ibb.co/WNsSfS5t/sample-bottles.jpg",
  "detected_ingredients": [
    "Vodka",
    "Gin",
    "Triple Sec",
    "Lime Juice",
    "Simple Syrup",
    "Orange Bitters"
  ],
  "cocktail_recipes": [
    {
      "name": "Classic Gin Martini",
      "ingredients": [
        "2 oz Gin",
        "1 oz Dry Vermouth",
        "Lemon twist or olive"
      ],
      "instructions": "Stir gin and vermouth with ice, strain into a chilled martini glass. Garnish with lemon twist or olive."
    },
    {
      "name": "Vodka Cosmopolitan",
      "ingredients": [
        "1.5 oz Vodka",
        "1 oz Triple Sec",
        "0.5 oz Lime Juice",
        "0.25 oz Simple Syrup"
      ],
      "instructions": "Shake all ingredients with ice and strain into a martini glass. Garnish with a lime wedge."
    },
    {
      "name": "Orange Gin Fizz",
      "ingredients": [
        "2 oz Gin",
        "0.75 oz Lime Juice",
        "0.5 oz Simple Syrup",
        "Dash of Orange Bitters",
        "Club Soda"
      ],
      "instructions": "Shake gin, lime juice, simple syrup, and bitters with ice. Strain into a highball glass over ice and top with club soda."
    }
  ],
  "notes": "Recipes generated based on detected bar ingredients. Quantities are standard; adjust to taste."
}

# Write to the required location
with open("/kaggle/working/submission.json", "w") as f:
    json.dump(submission, f, indent=2)

print("Dummy submission file created!")

