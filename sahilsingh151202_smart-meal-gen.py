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





!pip install google-generativeai
!pip install google-ai-python
!pip install adk



!pip install google-adk

import os
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")

# Load secret
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



print("Key loaded:", bool(os.environ.get("GOOGLE_API_KEY")))






import asyncio

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# User Preference Agent

user_pref_agent = Agent(
    name="UserPreferenceAgent",
    model=Gemini(),
    description ="""
You are a dietary preference collection agent.

Your job:
1. Read the user's message.
2. Extract dietary preferences.
3. ALWAYS assume the user is vegetarian unless they explicitly say otherwise.
4. Output clean, structured JSON only.

JSON format:
{
 "diet": "vegetarian",
 "cuisines": [...],
 "allergies": [...],
 "avoid": [...],
 "calorie_target": 2000,
 "budget_per_week_in_inr": 1500
}
"""
)

print("âœ… User Preference Agent created!")



# Create Runner
runner = InMemoryRunner(agent=user_pref_agent)



# Run Agent (ASYNC)
async def test_agent():
    response = await runner.run_debug(
        "I prefer Indian food, avoid mushrooms, no allergies. Budget 2000 rupees."
    )
    print("\nAgent Output:\n", response)

await test_agent()


# Meal Planner Agent

meal_planner_agent = Agent(
    name="MealPlannerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are a Meal Planning AI Agent.

Input:
- A JSON user profile containing diet, preferred cuisines, allergies, avoid-list, calorie target, and budget.

Task:
- Generate a 7-day vegetarian meal plan.
- Each day must include breakfast, lunch, and dinner.
- Prioritize Indian vegetarian dishes.
- Avoid ingredients listed in "avoid".
- Output structured JSON only.

Format:
{
 "day_1": {
   "breakfast": "...",
   "lunch": "...",
   "dinner": "..."
 },
 ...
 "day_7": {
   "breakfast": "...",
   "lunch": "...",
   "dinner": "..."
 }
}
""",
    tools=[],
    output_key="meal_plan",
)

print("âœ… Meal Planner Agent created!")



# Test Meal Planner Agent
runner2 = InMemoryRunner(agent=meal_planner_agent)

async def test_meal_planner():
    response = await runner2.run_debug(
        """
        {
           "diet": "vegetarian",
           "cuisines": ["Indian"],
           "allergies": [],
           "avoid": ["mushrooms"],
           "calorie_target": 2000,
           "budget_per_week_in_inr": 2000
        }
        """
    )
    print("\nMeal Plan Output:\n", response)

await test_meal_planner()



def extract_ingredients(meal_plan_text: str) -> dict:
    """
    Accepts a string.
    If it contains JSON, parse it.
    Otherwise treat the whole thing as meal text.

    Returns:
        {"meal_text": "..."}
    """
    import json

    # Try to parse JSON
    try:
        data = json.loads(meal_plan_text)

        # If JSON has days â†’ extract dishes
        if isinstance(data, dict):
            combined_text = ""
            for day, meals in data.items():
                if isinstance(meals, dict):
                    for meal_type, dish in meals.items():
                        combined_text += dish + "\n"
            return {"meal_text": combined_text}

    except Exception:
        # Not JSON â†’ treat raw text as meal_text
        pass

    # Fallback: treat as plain text
    return {"meal_text": meal_plan_text}



# Improved Grocery List Extraction Agent

grocery_agent_improved = Agent(
    name="GroceryListAgentImproved",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are an expert grocery ingredient extractor.

MANDATORY RULES (break any of these â†’ WRONG OUTPUT):

1. You MUST call extract_ingredients() exactly once.
2. After receiving meal_text, DO NOT call the tool again.
3. Only extract RAW INGREDIENTS.
4. REMOVE all dish names completely:
   Examples: "poha", "upma", "rajma chawal", "baingan bharta", 
   "masala dosa", "biryani", "mutter paneer", "dal tadka", 
   "kadhi chawal", "dal makhani", etc.
5. REMOVE anything that is:
   - more than 1 word (except when in parentheses, e.g. "green peas")
   - a cooking method
   - a preparation (pulao, chutney, raita, masala, curry)
   - bread types (roti, naan, paratha, phulka)
   - generic items ("mixed vegetables", "salad")
   - meals ("breakfast", etc.)

6. Ingredients MUST be:
   - single-word
   - raw food items (vegetables, grains, spices, dairy)
   - lowercase
   - singular (carrots â†’ carrot)
   - deduplicated

VALID EXAMPLE OUTPUT:
{
 "grocery_list": [
   "pea",
   "carrot",
   "onion",
   "potato",
   "spinach",
   "paneer",
   "rice",
   "cucumber",
   "tomato",
   "chili",
   "ginger",
   "garlic"
 ]
}

"""

,
    tools=[extract_ingredients],
    output_key="grocery_list",
)

print("ðŸ”¥ Improved Grocery Agent created!")



meal_plan_example = {
  "day_1": {
    "breakfast": "Poha with vegetables (peas, carrots, onions)",
    "lunch": "Dal Tadka with Jeera Rice and a side of Aloo Gobi",
    "dinner": "Palak Paneer with Roti and a small bowl of salad"
  },
  "day_2": {
    "breakfast": "Upma with mixed vegetables",
    "lunch": "Rajma Chawal with a dollop of yogurt",
    "dinner": "Baingan Bharta with Phulka and a cucumber raita"
  },
  "day_3": {
    "breakfast": "Masala Dosa with Sambar and Coconut Chutney",
    "lunch": "Chole Bhature (opt for roti for a lighter option)",
    "dinner": "Vegetable Pulao with a side of Raita"
  },
  "day_4": {
    "breakfast": "Idli with Sambar and Tomato Chutney",
    "lunch": "Paneer Butter Masala with Naan",
    "dinner": "Lentil Soup (Dal Shorba) with whole wheat bread"
  },
  "day_5": {
    "breakfast": "Aloo Paratha with pickle and yogurt",
    "lunch": "Mix Vegetable Curry with Roti",
    "dinner": "Gatte ki Sabzi with Phulka"
  },
  "day_6": {
    "breakfast": "Besan Cheela with mint chutney",
    "lunch": "Kadhi Chawal with a side of Bhindi Masala",
    "dinner": "Dal Makhani with Jeera Rice and a simple green salad"
  },
  "day_7": {
    "breakfast": "Sabudana Khichdi",
    "lunch": "Vegetable Biryani with Raita",
    "dinner": "Mutter Paneer with Roti and a side of Daal Fry"
  }
}

print("Meal plan stored in variable!")



import json

meal_plan_json = json.dumps(meal_plan_example)
print("Converted meal plan to JSON string!")



runner_grocery2 = InMemoryRunner(agent=grocery_agent_improved)

async def test_grocery_agent2():
    response = await runner_grocery2.run_debug(meal_plan_json)
    print("\nImproved Grocery List Output:\n", response)

await test_grocery_agent2()





