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


# Cell 1 - Imports & configuration
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import random
import csv
from collections import defaultdict, Counter

# Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")  # 'openai' or 'gemini' or 'mock'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)  # set in Kaggle secret or env
MEMORY_FILE = "memory.json"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("meal_agent")



# Cell 2 - Simple MemoryBank
class MemoryBank:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.data = {"users": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                logger.info("Memory loaded.")
            except Exception as e:
                logger.warning("Failed to load memory: %s", e)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        logger.info("Memory saved.")

    def get_user(self, user_id):
        return self.data["users"].get(user_id, {})

    def set_user(self, user_id, info):
        self.data["users"][user_id] = info
        self.save()

memory = MemoryBank()



# Cell 3 - LLM interface (mock + optional real)
def mock_llm(prompt: str) -> str:
    # Simple deterministic-ish generator for offline usage
    choices = [
        "A quick pasta with tomato sauce, garlic, basil. Ingredients: pasta, tomatoes, garlic, basil, olive oil. Steps: Boil pasta; make sauce; mix.",
        "Grilled chicken salad with lettuce, cherry tomatoes, cucumber, olive oil. Ingredients: chicken breast, lettuce, cucumber, tomatoes.",
        "Stir-fried tofu with vegetables. Ingredients: tofu, bell pepper, onion, soy sauce, rice."
    ]
    return random.choice(choices) + " (mock recipe generated)"

# If you want to use OpenAI, enable below.
def real_llm_openai(prompt: str) -> str:
    try:
        import openai
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set.")
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change if needed
            messages=[{"role":"user","content":prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("OpenAI call failed: %s", e)
        return mock_llm(prompt)

def call_llm(prompt: str) -> str:
    logger.info("LLM called. Provider=%s", LLM_PROVIDER)
    if LLM_PROVIDER == "openai":
        return real_llm_openai(prompt)
    else:
        return mock_llm(prompt)



# Replace mock_llm with a prompt-aware version (so it obeys "vegetarian" in prompt)
import random
def mock_llm(prompt: str) -> str:
    prompt_l = prompt.lower()
    if "vegetarian" in prompt_l or "no meat" in prompt_l or "no fish" in prompt_l:
        veg_choices = [
            "A quick vegetarian pasta with tomato sauce. Ingredients: pasta, tomatoes, garlic, basil, olive oil. Steps: Boil pasta; make sauce; mix.",
            "Stir-fried tofu with vegetables. Ingredients: tofu, bell pepper, onion, soy sauce, rice.",
            "Grilled paneer salad with lettuce, cherry tomatoes, cucumber, olive oil. Ingredients: paneer, lettuce, cucumber, tomatoes."
        ]
        return random.choice(veg_choices) + " (mock recipe generated)"
    else:
        nonveg_choices = [
            "Grilled chicken salad with lettuce, cherry tomatoes, cucumber, olive oil. Ingredients: chicken breast, lettuce, cucumber, tomatoes.",
            "Beef stir-fry. Ingredients: beef, broccoli, soy sauce, rice."
        ]
        return random.choice(nonveg_choices) + " (mock recipe generated)"

# rebind call_llm to use the new mock (call_llm uses `LLM_PROVIDER`, so no need to change)
print("mock_llm replaced. Next calls will respect 'vegetarian' in prompt if present.")



# Cell 4 - Agent implementations
class PlannerAgent:
    def __init__(self): pass

    def plan_week(self, user_prefs: Dict[str,Any]) -> List[Dict]:
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        meals_per_day = user_prefs.get("meals_per_day", 2)
        plan = []
        for d in days:
            for m in range(meals_per_day):
                slot = {
                    "day": d,
                    "meal": "Lunch" if m==0 else "Dinner",
                    "prompt_hint": f"{user_prefs['diet']} {user_prefs.get('allergies','')} quick budget-friendly"
                }
                plan.append(slot)
        logger.info("Planner created %d meal slots", len(plan))
        return plan

class RecipeAgent:
    def __init__(self): pass

    def generate_recipe(self, slot: Dict, user_prefs: Dict) -> Dict:
        prompt = f"Generate a recipe for: {slot['meal']} on {slot['day']}. Constraints: {user_prefs}"
        recipe_text = call_llm(prompt)
        # naive parse: split ingredients from steps by heuristic
        ing_line = "Ingredients:"
        if "Ingredients:" in recipe_text:
            ingredients = recipe_text.split("Ingredients:")[1].split("Steps:")[0].strip().split(",")
        else:
            # fallback
            ingredients = ["ingredient1", "ingredient2"]
        steps = ["Step 1: ..."]
        recipe = {"title": f"{slot['meal']} on {slot['day']}", "ingredients": [i.strip() for i in ingredients], "steps": steps, "raw": recipe_text}
        logger.info("RecipeAgent produced: %s", recipe["title"])
        return recipe

class ShopperAgent:
    def __init__(self): pass

    def aggregate_shopping_list(self, recipes: List[Dict]) -> List[Dict]:
        counts = Counter()
        for r in recipes:
            for ing in r.get("ingredients", []):
                key = ing.lower()
                counts[key] += 1
        # transform into list
        result = [{"item": k, "est_qty": v, "section": "general"} for k, v in counts.items()]
        logger.info("Shopper aggregated %d items", len(result))
        return result



# Cell 5 - Controller (orchestrator)
class ControllerAgent:
    def __init__(self, planner, recipe_agent, shopper, memory: MemoryBank):
        self.planner = planner
        self.recipe = recipe_agent
        self.shopper = shopper
        self.memory = memory

    def run(self, user_id: str, user_prefs: Dict) -> Dict:
        logger.info("Controller started for user %s", user_id)
        # load prior prefs or save
        user = self.memory.get_user(user_id)
        user.update(user_prefs)
        self.memory.set_user(user_id, user)

        plan_slots = self.planner.plan_week(user_prefs)
        recipes = []
        for slot in plan_slots:
            recipe = self.recipe.generate_recipe(slot, user_prefs)
            recipes.append(recipe)
        shopping_list = self.shopper.aggregate_shopping_list(recipes)

        output = {"plan": plan_slots, "recipes": recipes, "shopping_list": shopping_list, "timestamp": str(datetime.utcnow())}
        # persist to memory
        user.setdefault("past_plans", []).append(output)
        self.memory.set_user(user_id, user)
        logger.info("Controller finished run. Meals: %d", len(recipes))
        return output

planner = PlannerAgent()
recipe_agent = RecipeAgent()
shopper = ShopperAgent()
controller = ControllerAgent(planner, recipe_agent, shopper, memory)



# Cell 6 - Evaluation functions
def variety_score(plan_output):
    # number of unique ingredients
    s = set()
    for r in plan_output["recipes"]:
        for i in r.get("ingredients", []):
            s.add(i.strip().lower())
    return min(100, len(s))

def budget_score(user_prefs):
    # simple heuristic: lower budget -> lower score
    b = user_prefs.get("budget_per_person_per_day", 5)
    if b >= 10:
        return 100
    else:
        return int((b/10)*100)

def prep_time_score(user_prefs):
    t = user_prefs.get("max_prep_minutes", 30)
    return 100 if t >= 30 else int((t/30)*100)

def capstone_score(plan_output, user_prefs):
    v = variety_score(plan_output)
    b = budget_score(user_prefs)
    p = prep_time_score(user_prefs)
    # weighted average
    score = int(0.5*v + 0.3*b + 0.2*p)
    return {"variety": v, "budget": b, "prep_time": p, "capstone": score}



# Cell 7 - Export helper (shopping_list.csv)
def export_shopping_csv(shopping_list, filename="shopping_list.csv"):
    with open(filename, "w", newline='') as csvfile:
        fieldnames = ["item","est_qty","section"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in shopping_list:
            writer.writerow(row)
    logger.info("Exported shopping list to %s", filename)
    return filename



# Cell 8 - Example run (user input cell)
# Provide user preferences here. If using real LLM, set LLM_PROVIDER env var and API key accordingly.
user_id = "adarsh_demo"
user_prefs = {
    "diet": "vegetarian",               # vegetarian / vegan / omnivore / pescatarian
    "allergies": "nuts",                # free text
    "meals_per_day": 2,                 # 2 (lunch,dinner)
    "budget_per_person_per_day": 8,     # USD approximate
    "max_prep_minutes": 30
}

output = controller.run(user_id, user_prefs)
print("Generated", len(output["recipes"]), "recipes.")



# Cell 9 - Show a sample recipe & export shopping list
print("Sample recipe raw text:\n", output["recipes"][0]["raw"][:600])
export_file = export_shopping_csv(output["shopping_list"])
print("Downloaded:", export_file)
scores = capstone_score(output, user_prefs)
print("Evaluation scores:", scores)



# Cell 10 - Save final artifacts for submission
with open("capstone_output_summary.json", "w") as f:
    json.dump({"output": output, "scores": scores}, f, indent=2)
logger.info("Saved capstone_output_summary.json")



# Add this cell AFTER you've run the earlier cells (controller, planner, recipe_agent, shopper, memory exist)
import re
import random

def parse_request(text):
    # Very simple parsing for "<N> days" and "<X> calories"
    days_match = re.search(r'(\d+)\s*days?', text, re.IGNORECASE)
    cal_match = re.search(r'(\d+)\s*calories?', text, re.IGNORECASE)
    if days_match:
        days = int(days_match.group(1))
    else:
        days = 7  # default
    if cal_match:
        max_cal = int(cal_match.group(1))
    else:
        max_cal = 800  # default per-meal cap
    return days, max_cal

def supervisor_agent(natural_text, user_id="adarsh_demo", base_user_prefs=None):
    """
    Simple supervisor wrapper:
    - parse natural_text for <N> days and <X> calories
    - build user_prefs (merge with existing memory or base_user_prefs)
    - call controller.run and trim to requested days
    - add a simple estimated calorie value to each recipe (mock estimator)
    """
    # parse
    n_days, max_cal = parse_request(natural_text)
    # get saved prefs if present
    saved = memory.get_user(user_id) or {}
    prefs = saved.copy()
    if base_user_prefs:
        prefs.update(base_user_prefs)
    # ensure meals_per_day exists
    meals_per_day = prefs.get("meals_per_day", 2)
    # for a short plan, ensure planner will produce enough slots — planner produces 7-day plan by default,
    # so we'll slice the results to n_days*meals_per_day
    prefs["meals_per_day"] = meals_per_day
    # run the controller (this persists memory too)
    output = controller.run(user_id, prefs)
    # trim to requested days
    total_slots = n_days * meals_per_day
    output["plan"] = output["plan"][:total_slots]
    output["recipes"] = output["recipes"][:total_slots]
    output["shopping_list"] = output["shopping_list"]  # keep full aggregation (or you can re-aggregate from recipes)
    # attach mock calorie estimates that respect the max_cal constraint
    for r in output["recipes"]:
        # simple mock estimator: random int between 200 and max_cal (so it stays under)
        est = random.randint(200, max(250, max_cal))
        # ensure not exceeding max_cal (just in case)
        r["est_calories"] = min(est, max_cal)
    output["meta"] = {"requested_days": n_days, "max_cal_per_meal": max_cal}
    return output

# Example usage (run this after you run the cell above):
# response = supervisor_agent("Plan my meals for 3 days under 500 calories each.")
# print("Meals generated:", len(response['recipes']))
# for r in response['recipes']:
#     print(r['title'], "-", r.get('est_calories', '?'), "kcal")



# --- Diet-enforcing supervisor wrapper (paste after supervisor_agent or instead of it) ---
import re

# small helper to detect non-veg tokens
NON_VEG_TOKENS = {"chicken", "beef", "pork", "fish", "shrimp", "salmon", "tuna", "bacon", "lamb", "anchovy"}

def recipe_violates_diet(recipe: dict, diet: str) -> bool:
    """
    Returns True if recipe contains ingredients that violate diet.
    diet can be 'vegetarian' or 'vegan' or others.
    """
    if diet not in ("vegetarian", "vegan"):
        return False
    # create a single joined lower string of ingredients + raw text
    text = " ".join(recipe.get("ingredients", []) + [recipe.get("raw","")]).lower()
    for token in NON_VEG_TOKENS:
        if re.search(r'\b' + re.escape(token) + r'\b', text):
            return True
    return False

def supervisor_agent_enforce_diet(natural_text, user_id="adarsh_demo", base_user_prefs=None, max_retries=3):
    # parse same as previous parse_request
    n_days, max_cal = parse_request(natural_text)
    saved = memory.get_user(user_id) or {}
    prefs = saved.copy()
    if base_user_prefs:
        prefs.update(base_user_prefs)
    meals_per_day = prefs.get("meals_per_day", 2)
    prefs["meals_per_day"] = meals_per_day

    output = controller.run(user_id, prefs)  # initial run (may contain violations)

    # Check each recipe and regenerate if it violates diet
    for idx, recipe in enumerate(output["recipes"]):
        retry = 0
        while recipe_violates_diet(recipe, prefs.get("diet", "")) and retry < max_retries:
            # regenerate with stronger prompt hint: ask explicitly for vegetarian version
            slot = output["plan"][idx]
            stronger_prompt = f"Generate a strictly {prefs.get('diet','vegetarian')} recipe for {slot['meal']} on {slot['day']}. No meat/fish; avoid allergens: {prefs.get('allergies','none')}."
            # call LLM directly and re-parse (mimic RecipeAgent.generate_recipe)
            new_raw = call_llm(stronger_prompt)
            # parse ingredients (same heuristic)
            if "Ingredients:" in new_raw:
                parts = new_raw.split("Ingredients:")
                if len(parts) > 1:
                    rest = parts[1]
                    if "Steps:" in rest:
                        ingredients = rest.split("Steps:")[0].strip().split(",")
                    else:
                        ingredients = rest.strip().split(",")
                else:
                    ingredients = ["ingredient1"]
            else:
                ingredients = ["ingredient1"]
            new_recipe = {
                "title": recipe["title"],
                "ingredients": [i.strip() for i in ingredients if i.strip()],
                "steps": ["Step 1: Follow the recipe steps in raw output."],
                "raw": new_raw
            }
            # replace
            output["recipes"][idx] = new_recipe
            recipe = new_recipe
            retry += 1

    # Trim to requested days
    total_slots = n_days * meals_per_day
    output["plan"] = output["plan"][:total_slots]
    output["recipes"] = output["recipes"][:total_slots]
    # attach est_calories similar to previous supervisor_agent
    for r in output["recipes"]:
        est = random.randint(200, max(250, max_cal))
        r["est_calories"] = min(est, max_cal)
    output["meta"] = {"requested_days": n_days, "max_cal_per_meal": max_cal}
    # persist final version to memory again
    user = memory.get_user(user_id) or {}
    user.setdefault("past_plans", []).append(output)
    memory.set_user(user_id, user)
    return output



response = supervisor_agent("Plan my meals for 3 days under 500 calories each.")
response 


# Cell A — Normalize & re-aggregate shopping list (quick one-off)
import re
from collections import defaultdict
def normalize_token(tok):
    tok = str(tok)
    tok = re.sub(r'\(.*?\)', '', tok)          # remove parenthesis content
    tok = tok.replace("(mock recipe generated)", "")
    tok = tok.strip().lower()
    tok = re.sub(r'[^a-z0-9\s-]', '', tok)     # drop punctuation like periods, commas
    tok = re.sub(r'\s+', ' ', tok).strip()
    return tok

def normalize_and_aggregate_from_response(resp):
    counts = defaultdict(int)
    for r in resp.get("recipes", []):
        seen = set()
        for ing in r.get("ingredients", []):
            n = normalize_token(ing)
            if not n:
                continue
            if n in seen:  # avoid counting same ingredient twice in same recipe
                continue
            seen.add(n)
            counts[n] += 1
    normalized = [{"item": item, "est_qty": qty, "section": "general"} for item, qty in sorted(counts.items(), key=lambda x:-x[1])]
    return normalized

# run normalization on current `response` object
shopping_normalized = normalize_and_aggregate_from_response(response)
response["shopping_list"] = shopping_normalized

# export CSV + JSON
import csv, json
csv_path = "shopping_list_normalized.csv"
with open(csv_path, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["item","est_qty","section"])
    writer.writeheader()
    for row in shopping_normalized:
        writer.writerow(row)

with open("supervisor_output_normalized.json", "w") as f:
    json.dump(response, f, indent=2)

print("Normalized shopping list saved to:", csv_path)
print("Top items:")
for r in shopping_normalized[:20]:
    print(f"- {r['item']} (used in {r['est_qty']} meals)")


