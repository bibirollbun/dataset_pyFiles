#Set up Google_API_Key

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Imports

from google.adk.agents import LlmAgent as Agent, SequentialAgent, ParallelAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types
import logging
import requests
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


#Define Model

MODEL_NAME = "gemini-2.0-flash"


#Define Tools

weather_search_tool = google_search
google_search_tool = google_search


#Define Agents

# 1) SHOPPING AGENT - EXTRACTS STRUCTURED INFORMATION FROM USER MESSAGE

shopping_agent = Agent(
    name="ShoppingGoalAgent",
    model=MODEL_NAME,
    instruction="""
        You are a helpful assistant extracting structured shopping + meal info from the user's request.
        
        Given the user's message, infer and output a short JSON-like blob with:
        - "address": user's starting location if mentioned
        - "items_to_buy": list of ingredients they explicitly want to buy
        - "deadline_hours": how many hours they have (default 48 if they say '2 days')
        - "constraints": short text on constraints (e.g. no car, delivery vs walk)
        
        ONLY output something like:
        
        shopping_info = {{
          "address": "...",
          "items_to_buy": [...],
          "deadline_hours": 48,
          "constraints": "..."
        }}
        
        User message
        """,
    output_key="shopping_info"
)


# 2) INVENTORY AGENT - ESTIMATES WHAT THE USER PROBABLY HAS IN THEIR PANTRY

inventory_agent = Agent(
    name="InventoryAgent",
    model=MODEL_NAME,
    instruction="""
    You are an InventoryAgent estimating what the user likely already has in their pantry.
    
    Use the shopping_info and make rough assumptions:
    - Staples like rice, oil, salt, basic spices, onions may already be present.
    - Perishables like milk, eggs, tomatoes, ginger, chicken usually need buying.
    
    Output a Python-style dict called inventory:
    
    inventory = {{
      "have": [...],    # items they probably already have
      "missing": [...]  # items they probably must buy
    }}
    
    Context:
    shopping_info:
    {shopping_info}
    """,
    output_key="inventory"
)


# 3) WEATHER AGENT - USES GOOGLE SEARCH TO FETCH WEATHER FORECAST, HELPS RECOMMEND WALK VS DELIVERY

weather_agent = Agent(
    name="WeatherAgent",
    model=MODEL_NAME,
    tools=[weather_search_tool],
    instruction="""
        You are a WeatherAgent.
        
        Your job:
        1. Read the user's address from the context.
        2. Form a smart weather query using Google Search, such as:
           - "48 hour weather forecast for <address>"
           - "weather next 2 days <city>"
        3. Call the weather_search_tool exactly once.
        4. Parse the returned search results and extract:
           - The general temperature trend (hot / cold)
           - Rain chances
           - Any important alerts
        5. Output a Python-style dict named `weather`:
        
        weather = {
          "location": "...",
          "summary": "...",        # 1â€“2 sentence summary
          "recommendation": "...", # should user walk? deliver? any caution?
          "raw": { ... }           # the raw search output
        }
        
        Be concise, readable, and only rely on the search results.
        """,
    output_key="weather"
)


# 4) RECIPE AGENT - SEARCHES FOR HEALTHY MEALS (BREAKFAST, LUNCH, SNACK, DINNER)

recipe_agent = Agent(
    name="RecipeAgent",
    model=MODEL_NAME,
    tools=[google_search_tool],
    instruction="""
You are a RecipeAgent.

Goal:
- Use the user's ingredients and basic pantry items to find HEALTHY recipes
  for breakfast, lunch, snacks, and dinner over the next few days.

Context:
shopping_info:
{shopping_info}

inventory:
{inventory}

Instructions:
1. Identify the main ingredients the user wants to buy:
   shopping_info["items_to_buy"]
2. Assume they also have basic staples from inventory["have"] (rice, oil, salt, onions, etc.).
3. Use the google_search_tool 2â€“4 times with queries like:
   - "healthy breakfast recipe with eggs tomatoes"
   - "high protein lunch recipe with chicken rice"
   - "light snack recipe with tomatoes and ginger"
   - "simple healthy dinner chicken rice bowl"
4. From the search results, infer simple recipe options. You do NOT need exact details,
   but you should extract:
   - the dish name
   - main ingredients
   - a rough health angle (high protein, low oil, more veggies, etc.)
   - a reference link if clearly visible in the search results

5. Output a Python-style dict called `recipes`:

recipes = {
  "breakfast": [
    {
      "name": "...",
      "main_ingredients": ["..."],
      "approx_time_min": 15,
      "health_note": "high protein / low oil / more veggies",
      "source_hint": "site or short URL if visible"
    },
    ...
  ],
  "lunch": [
    { ... }
  ],
  "snack": [
    { ... }
  ],
  "dinner": [
    { ... }
  ]
}

Rules:
- Prefer simple, 20â€“40 minute recipes.
- Emphasize "healthy" in your choices (less deep-fry, more baking/boiling/saute).
- If search results are noisy, make a reasonable guess based on common recipes.
""",
    output_key="recipes"
)



# 5) STORE LOCATOR AGENT - FINDING NEARBY GROCERY STORES THAT LIKELY HAVE INGREDIENTS (USE GOOGLE SEARCH TOOL WITH ADDRESS AND SHOPPING LIST)

store_locator_agent = Agent(
    name="StoreLocatorAgent",
    model=MODEL_NAME,
    tools=[google_search_tool],
    instruction="""
    You are a StoreLocatorAgent.
    
    Goal:
    - Suggest 2â€“3 nearby grocery stores that are good candidates
      for buying the user's missing ingredients.
    
    Context:
    shopping_info:
    {shopping_info}
    
    inventory:
    {inventory}
    
    shopping_list:
    {shopping_list}
    
    Instructions:
    1. From shopping_info["address"], build a query like:
       "grocery stores near <address> open now"
       or
       "supermarket near <address>".
    
    2. Use the google_search_tool 1â€“2 times to find:
       - Large supermarkets
       - Well-known chains (more likely to carry all items)
       - Stores reasonably close to the address
    
    3. For each of the top 2â€“3 good candidates, infer:
       - name
       - address or area
       - type (e.g., 'supermarket', 'warehouse club', 'local grocery')
       - distance or "nearby" hint if available
       - a rough guess of whether they likely have key items
         (proteins, dairy, produce) based on store type.
    
    4. Output a Python-style dict named stores:
    
    stores = {
      "stores": [
        {
          "name": "...",
          "address": "...",
          "type": "supermarket / local grocery / etc.",
          "distance_hint": "...",     # e.g., "about 5â€“10 min drive/walk"
          "likely_items": ["milk", "eggs", "chicken", "tomatoes", "ginger"],
          "notes": "why this store is a good fit"
        },
        ...
      ]
    }
    
    Be concise but informative. If information is missing, make a reasonable
    assumption based on the store's type and brand.
    """,
    output_key="stores"
)



# 6) MEAL PLAN AGENT - GENERATES A 5-DAY MEAL PLAN

meal_plan_agent = Agent(
    name="MealPlanAgent",
    model=MODEL_NAME,
    instruction="""
You are a MealPlanAgent.

Using:
- shopping_info.items_to_buy
- inventory
- recipes
- weather (optional context)

Create a healthy 5-day meal plan (Mondayâ€“Friday).
For EACH day, include: breakfast, lunch, snack, dinner.

Health guidelines:
- Prefer balanced meals: protein + carbs + veggies.
- Limit heavy frying; prefer baking, boiling, sautÃ©ing with minimal oil.
- If weather is hot, suggest lighter meals; if cooler, suggest warm comfort dishes.

Output a Python-style dict called meal_plan:

meal_plan = {
  "days": [
    {
      "day": "Monday",
      "meals": [
        {
          "type": "breakfast",
          "name": "...",
          "main_ingredients": ["..."],
          "approx_time_min": 15,
          "health_note": "..."
        },
        {
          "type": "lunch",
          "name": "...",
          "main_ingredients": ["..."],
          "approx_time_min": 30,
          "health_note": "..."
        },
        {
          "type": "snack",
          "name": "...",
          "main_ingredients": ["..."],
          "approx_time_min": 10,
          "health_note": "..."
        },
        {
          "type": "dinner",
          "name": "...",
          "main_ingredients": ["..."],
          "approx_time_min": 30,
          "health_note": "..."
        }
      ]
    },
    ...
  ]
}

Use recipes[meal_type] where possible:
- If there are more recipes than needed, just pick reasonable ones.
- If there are fewer, reuse or slightly adapt recipes across days.
- You may improvise simple recipe variations if needed.
Context:

shopping_info:
{shopping_info}

inventory:
{inventory}

recipes:
{recipes}

weather:
{weather}
""",
    output_key="meal_plan"
)


# 7) SHOPPING LIST AGENT - BUILDS FINAL CATEGORIZED GROCERY LIST

shopping_list_agent = Agent(
    name="ShoppingListAgent",
    model=MODEL_NAME,
    instruction="""
You are a ShoppingListAgent.

Using the meal_plan and inventory, decide what the user needs to buy.
Assume:
- If an item is in inventory.have, they already have enough.
- If an item is in inventory.missing OR appears in shopping_info.items_to_buy,
  it should be added to the shopping list.

Group items roughly by category (produce, protein, grains, other).

Output a Python-style dict called shopping_list:

shopping_list = {{
  "produce": [...],
  "protein": [...],
  "grains": [...],
  "other": [...]
}}

Context:
shopping_info:
{shopping_info}

inventory:
{inventory}

meal_plan:
{meal_plan}
""",
    output_key="shopping_list"
)



# 8) SUMMARY AGENT - GENERATES FINAL USER-FRIENDLY OUTPUT

summary_agent = Agent(
    name="SummaryAgent",
    model=MODEL_NAME,
    instruction="""
    You are a friendly personal assistant summarizing a PantryPilot plan for the user.
    
    Your response MUST include:
    1) A short intro about the 5-day healthy meal plan.
    2) A day-by-day breakdown (Monâ€“Fri) with breakfast, lunch, snack, dinner.
    3) A shopping list grouped by category (Proteins, Dairy, Produce, Pantry).
    4) 2â€“3 practical tips (prep, storage, weather-based shopping).
    5) A short section listing 2â€“3 nearby stores from `stores`, e.g.:
    
       Nearby Stores:
       - <name> â€” <type>, <distance_hint>. Likely has: milk, eggs, chicken...
       - ...
    
    Context:
    shopping_info:
    {shopping_info}
    
    inventory:
    {inventory}
    
    recipes:
    {recipes}
    
    weather:
    {weather}
    
    meal_plan:
    {meal_plan}
    
    shopping_list:
    {shopping_list}
    
    stores:
    {stores}
    
    Write normal text only (no JSON or Python).
    """,
    output_key="final_plan"
)



# PARALLEL AGENT - INVENTORY AND WEATHER AGENT

inventory_and_weather = ParallelAgent(
    name="InventoryAndWeather",
    sub_agents=[inventory_agent, weather_agent],
)

# SEQUENTIAL AGENT
root_agent = SequentialAgent(
    name="PantryPilotPipeline",
    sub_agents=[
        shopping_agent,          
        inventory_and_weather,   
        recipe_agent,            
        meal_plan_agent,         
        shopping_list_agent,
        store_locator_agent,
        summary_agent,
    ],
)


#RUNNER
runner = InMemoryRunner(agent=root_agent, app_name="pantry_pilot_app")


session_service = runner.session_service

async def demo():
    user_msg = (
        "I need to plan healthy meals for the next 5 days. Iâ€™m starting from my apartment at 210 10th St, Jersey City, New Jersey and I donâ€™t have a car, so Iâ€™ll either walk or use delivery depending on the weather. Iâ€™m okay with eating chicken, eggs, rice, tomatoes, ginger, and some veggies, and I want breakfast, lunch, snacks, and dinner ideas. Please tell me what I should buy in the next 2 days and create a 5-day healthy meal plan using those ingredients."
    )

    user_id = "demo_user"
    session_id = "pantry_session"
    await session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=user_msg)],
    )

    final_plan_text = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_plan_text = event.content.parts[0].text

    print("\n=== Final Plan ===\n")
    print(final_plan_text)

await demo()

