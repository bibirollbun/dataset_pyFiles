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


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


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


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


###############################################################
# SMART MEAL PLANNER - SINGLE FILE VERSION (KAGGLE FRIENDLY)
###############################################################

import json
import datetime

# ---------------------------------------------
# LOGGING (OBSERVABILITY)
# ---------------------------------------------
def log(message):
    print(f"[{datetime.datetime.now()}] {message}")

# ---------------------------------------------
# MEMORY BANK (LONG TERM MEMORY)
# ---------------------------------------------
class MemoryBank:
    def __init__(self):
        self.memory = {}

    def store(self, key, value):
        self.memory[key] = value

    def get(self, key):
        return self.memory.get(key, None)

# ---------------------------------------------
# SESSION STATE (PAUSE/RESUME)
# ---------------------------------------------
class SessionState:
    def __init__(self):
        self.state = {}

    def save(self, key, value):
        self.state[key] = value

    def load(self, key):
        return self.state.get(key)

# ---------------------------------------------
# CUSTOM TOOLS
# ---------------------------------------------
class PantryTool:
    def __init__(self):
        self.items = {}

    def add(self, item):
        self.items[item] = "ok"

    def list_items(self):
        return list(self.items.keys())

    def get_expiring_items(self):
        return []  # placeholder

class PriceLookupTool:
    def lookup(self, item):
        return {"price": "$2.99", "store": "Demo Store"}

# ---------------------------------------------
# AGENTS
# ---------------------------------------------

# 1) MEAL PLANNER AGENT
class MealPlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = MemoryBank()

    def generate_weekly_plan(self, user_prefs):
        log("MealPlannerAgent: Generating weekly meal plan")

        past = self.memory.get("past_meals")

        prompt = f"""
        Create a 7-day vegetarian meal plan.
        Consider user preferences: {user_prefs}
        Avoid past meals: {past}
        """

        output = self.llm(prompt)
        self.memory.store("past_meals", output)

        return output

# 2) GROCERY OPTIMIZER AGENT
class GroceryOptimizerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.pantry = PantryTool()
        self.price_tool = PriceLookupTool()

    def optimize_grocery_list(self, meal_plan):
        log("GroceryOptimizerAgent: Optimizing grocery list")

        # Extract ingredients using LLM
        ingredients = self.llm(f"Extract ingredients only from this plan:\n{meal_plan}")
        ingredients = ingredients.split(",")  # crude but ok for demo
        ingredients = [i.strip() for i in ingredients]

        # Remove pantry items
        pantry_items = self.pantry.list_items()
        needed = [x for x in ingredients if x not in pantry_items]

        # Lookup prices
        shopping_list = {}
        for item in needed:
            shopping_list[item] = self.price_tool.lookup(item)

        return shopping_list

# 3) COOKING ASSISTANT AGENT (PAUSE/RESUME)
class CookingAssistantAgent:
    def __init__(self, llm):
        self.llm = llm
        self.session = SessionState()

    def start_recipe(self, recipe):
        log("CookingAssistant: Starting recipe")
        steps = self.llm(f"Break down {recipe} into numbered cooking steps.")

        steps = steps.split("\n")
        steps = [s for s in steps if s.strip()]

        self.session.save("steps", steps)
        self.session.save("current", 0)

        return "Cooking started. Say next_step() to continue."

    def next_step(self):
        steps = self.session.load("steps")
        idx = self.session.load("current")

        if idx >= len(steps):
            return "Recipe completed!"

        step = steps[idx]
        self.session.save("current", idx + 1)
        return step

# ---------------------------------------------
# WORKFLOW
# ---------------------------------------------
def run_weekly_workflow(planner, grocery, prefs):
    log("Running weekly workflow...")

    plan = planner.generate_weekly_plan(prefs)
    groceries = grocery.optimize_grocery_list(plan)

    return {
        "meal_plan": plan,
        "grocery_list": groceries
    }

# ---------------------------------------------
# MOCK LLM
# ---------------------------------------------
def mock_llm(prompt):
    # This simulates LLM behavior for Kaggle demo
    if "Extract ingredients" in prompt:
        return "rice, tomato, onion, garlic"
    if "numbered cooking steps" in prompt:
        return "1. Prep ingredients\n2. Cook items\n3. Serve"
    return "Mock meal plan: stir fry, salad, tofu bowl, pasta, soup, curry, tacos"

# ---------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------
planner = MealPlannerAgent(mock_llm)
grocery = GroceryOptimizerAgent(mock_llm)
cook = CookingAssistantAgent(mock_llm)

# Weekly planning
result = run_weekly_workflow(planner, grocery, "vegetarian, high protein")
print("\nFINAL WEEKLY PLAN OUTPUT:\n", result)

# Cooking
print("\n--- Cooking Example ---")
print(cook.start_recipe("Vegetable Stir Fry"))
print(cook.next_step())
print(cook.next_step())
print(cook.next_step())
print(cook.next_step())


