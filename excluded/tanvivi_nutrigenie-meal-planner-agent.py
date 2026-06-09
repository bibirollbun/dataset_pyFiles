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


# Cell 2 â€” Imports and logging setup
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Observability: simple logger config
logger = logging.getLogger("nutrigenie")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

logger.info("NutriGenie logger initialized")



# Cell 3 â€” Simple config + constants
DEFAULT_CALORIES = 2000
DEFAULT_PROTEIN = 70  # grams per day
SUPPORTED_PREFERENCES = ["indian", "mixed", "high_protein", "vegetarian"]


# Cell 4 â€” Tools (atomic, easily replaceable by ADK Tools)
# Each tool returns predictable data structures.

def nutrition_calculator(calories: int, protein_target: int) -> Dict[str, Any]:
    """
    Simple macro calculator:
      - Protein = protein_target (g)
      - Carbs = 40% calories / 4
      - Fats = 25% calories / 9
    """
    carbs_g = (calories * 0.40) / 4
    fats_g = (calories * 0.25) / 9
    macros = {
        "calories": calories,
        "protein_g": int(round(protein_target)),
        "carbs_g": int(round(carbs_g)),
        "fats_g": int(round(fats_g))
    }
    logger.debug(f"nutrition_calculator -> {macros}")
    return macros

# Simple in-memory recipe database (expandable)
RECIPE_DB = {
    "indian": [
        {"name": "Paneer Tikka Bowl", "protein_g": 30, "calories": 450, "ingredients": ["paneer", "yogurt", "spices", "veg"]},
        {"name": "Masoor Dal + Rice", "protein_g": 18, "calories": 420, "ingredients": ["masoor dal", "rice", "spices"]}
    ],
    "high_protein": [
        {"name": "Grilled Chicken + Veg", "protein_g": 40, "calories": 500, "ingredients": ["chicken", "veg", "spices"]},
        {"name": "Tofu Stir-Fry", "protein_g": 28, "calories": 420, "ingredients": ["tofu","veg","soy sauce"]}
    ],
    "vegetarian": [
        {"name": "Chickpea Salad", "protein_g": 22, "calories": 350, "ingredients": ["chickpeas","veg","olive oil"]},
        {"name": "Paneer Bhurji + Roti", "protein_g": 25, "calories": 480, "ingredients": ["paneer","eggs? (optional)","spices","roti"]}  # note: adjust if eggs not allowed
    ],
    "mixed": [
        {"name": "Salmon + Quinoa", "protein_g": 35, "calories": 520, "ingredients": ["salmon","quinoa","veg"]},
        {"name": "Veggie Pulao + Raita", "protein_g": 12, "calories": 400, "ingredients": ["rice","veg","curd"]}
    ]
}

def recipe_search(preference: str, min_protein: int = 0) -> List[Dict[str, Any]]:
    """
    Return recipes matching preference and approximate protein needs.
    """
    logger.debug(f"recipe_search pref={preference} min_protein={min_protein}")
    db = RECIPE_DB.get(preference, RECIPE_DB["mixed"])
    # Simple filter: prefer recipes with protein >= min_protein/number_of_meals heuristic
    return [r for r in db if r.get("protein_g", 0) >= min_protein]
    
def generate_grocery_list(meal_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Derive a simple grocery list as counts of ingredients.
    """
    items = {}
    for meal in meal_list:
        for ing in meal.get("ingredients", []):
            items[ing] = items.get(ing, 0) + 1
    logger.debug(f"grocery_list generated: {items}")
    return items



# Cell 5 â€” Memory and Sessions
class UserMemory:
    """
    Lightweight persistent memory simulation (in-memory).
    You can extend to file-based or replace with ADK MemoryBank.
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._created_at = datetime.utcnow()

    def save(self, key: str, value: Any):
        logger.info(f"Memory SAVE: {key} = {value}")
        self._store[key] = value

    def get(self, key: str, default=None):
        v = self._store.get(key, default)
        logger.info(f"Memory GET: {key} -> {v}")
        return v

    def dump(self):
        return dict(self._store)

# Session service (short-lived context)
class Session:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.context: Dict[str, Any] = {}
        logger.info(f"Session started for {user_id}")

    def set(self, k, v):
        self.context[k] = v

    def get(self, k, default=None):
        return self.context.get(k, default)



# Cell 6 â€” Agents (Orchestrator + Sub-agents)
class NutritionAgent:
    def __init__(self, tool_func=nutrition_calculator):
        self.tool = tool_func

    def run(self, calories: int, protein_target: int) -> Dict[str, Any]:
        logger.info("NutritionAgent: calculating macros")
        return self.tool(calories, protein_target)

class RecipeAgent:
    def __init__(self, tool_func=recipe_search):
        self.tool = tool_func

    def run(self, preference: str, meals_per_day: int, daily_protein_target: int) -> List[Dict[str, Any]]:
        logger.info("RecipeAgent: searching recipes")
        # Try to allocate per meal protein = daily_protein_target / meals_per_day
        per_meal_protein = max(1, int(daily_protein_target / max(1, meals_per_day)))
        candidates = self.tool(preference, per_meal_protein)
        # If not enough candidates, fallback to broader DB
        if not candidates:
            candidates = self.tool("mixed", per_meal_protein)
        # Simple selection: pick up to meals_per_day items cycling if necessary
        selected = []
        idx = 0
        while len(selected) < meals_per_day:
            selected.append(candidates[idx % len(candidates)])
            idx += 1
        logger.debug(f"RecipeAgent selected: {[m['name'] for m in selected]}")
        return selected

class GroceryAgent:
    def __init__(self, tool_func=generate_grocery_list):
        self.tool = tool_func

    def run(self, meals: List[Dict[str, Any]]) -> Dict[str, int]:
        logger.info("GroceryAgent: generating grocery list")
        return self.tool(meals)



# Cell 7 â€” Orchestrator (ties everything together)
class Orchestrator:
    def __init__(self, user_id: str = "user_default"):
        self.memory = UserMemory()
        self.session = Session(user_id)
        self.nutrition_agent = NutritionAgent()
        self.recipe_agent = RecipeAgent()
        self.grocery_agent = GroceryAgent()

    def plan_week(self, calories: Optional[int] = None, protein: Optional[int] = None,
                  preference: str = "mixed", meals_per_day: int = 3, days: int = 7) -> Dict[str, Any]:
        # defaulting and saving preferences
        calories = calories or self.memory.get("calories", DEFAULT_CALORIES)
        protein = protein or self.memory.get("protein", DEFAULT_PROTEIN)
        preference = preference if preference in SUPPORTED_PREFERENCES else "mixed"

        logger.info(f"Orchestrator: planning {days} days | {meals_per_day} meals/day | pref={preference}")
        self.memory.save("calories", calories)
        self.memory.save("protein", protein)
        self.memory.save("preference", preference)

        macros = self.nutrition_agent.run(calories, protein)

        week_plan = []
        for d in range(days):
            day_meals = self.recipe_agent.run(preference, meals_per_day, protein)
            week_plan.append({
                "day": d + 1,
                "meals": day_meals
            })

        # Flatten meals for grocery generation
        all_meals = []
        for day in week_plan:
            all_meals.extend(day["meals"])
        grocery = self.grocery_agent.run(all_meals)

        result = {
            "macros": macros,
            "week_plan": week_plan,
            "grocery_list": grocery,
            "user_memory_snapshot": self.memory.dump()
        }
        logger.info("Orchestrator: plan complete")
        return result

    # minimal API for updating preferences mid-session
    def update_user_pref(self, key, value):
        self.memory.save(key, value)



# Cell 8 â€” Demo: Run Planner (default scenario)
orch = Orchestrator(user_id="vivi")
plan = orch.plan_week(calories=1800, protein=75, preference="indian", meals_per_day=3, days=7)

# Print short summary
print("=== Daily Macros ===")
print(plan["macros"])
print("\n=== Day 1 Meals ===")
for m in plan["week_plan"][0]["meals"]:
    print("-", m["name"], f"(protein: {m['protein_g']}g, cal: {m['calories']})")
print("\n=== Grocery List Sample ===")
gkeys = list(plan["grocery_list"].items())[:10]
print(dict(gkeys))



# Cell 9 â€” Example: Adjust plan based on user feedback (session & memory demo)
# Save a user-specific restriction and re-plan
orch.update_user_pref("avoid_ingredients", ["eggs"])  # user doesn't want eggs
# You would change recipe_search to filter by avoid_ingredients; demo storing preference:
orch.memory.save("avoid_ingredients", ["eggs"])

# Re-plan using same settings but now memory contains avoid list (recipe filter not implemented here)
plan2 = orch.plan_week(calories=1600, protein=70, preference="vegetarian", meals_per_day=3, days=3)
print("Planned 3-day vegetarian (1600 kcal):")
for day in plan2["week_plan"]:
    print(f"Day {day['day']}: {[m['name'] for m in day['meals']]}")



# Cell 10 â€” Evaluation: Basic tests to validate behavior
def test_macros():
    m = nutrition_calculator(2000, 80)
    assert m["calories"] == 2000
    assert m["protein_g"] == 80
    assert isinstance(m["carbs_g"], int)
    assert isinstance(m["fats_g"], int)
    print("test_macros passed")

def test_recipe_selection():
    ra = RecipeAgent()
    meals = ra.run("high_protein", meals_per_day=2, daily_protein_target=80)
    assert len(meals) == 2
    print("test_recipe_selection passed")

def test_grocery_generation():
    sample = [{"name":"A","ingredients":["x","y"]}, {"name":"B","ingredients":["y","z"]}]
    gl = generate_grocery_list(sample)
    assert gl["y"] == 2
    print("test_grocery_generation passed")

# Run tests
test_macros()
test_recipe_selection()
test_grocery_generation()
print("All small tests passed.")








